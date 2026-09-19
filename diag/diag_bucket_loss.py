# scripts/diag_bucket_loss.py

import os, re, math, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

N_BUCKET = 6
BS, SEQ  = 16, 2048
N_BATCH  = 60         
DEVICE   = "cuda"

TAGS = {  
    "ddp":      "d20_ddp",             
    "runA":     "d20_ddp_n9500",
    "pro_0.2":  "d20_diloco_pro_0.2",
    "orth_0.8": "d20_diloco_orth_0.9",
    "diloco":   "d20_diloco",
}


def load_model(tag, step=None, device=DEVICE):
    ckpt_dir = os.path.join("/home/lyq/.cache/nanochat/base_checkpoints/", tag)
    files = sorted(f for f in os.listdir(ckpt_dir) if re.match(r"model_\d+\.pt$", f))
    assert files, f"no model_*.pt in {ckpt_dir}"
    fname = files[-1] if step is None else f"model_{step:06d}.pt"
    blob = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu")
    sd = blob.get("model", blob)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # 防 torch.compile 前缀

    # 直接从权重形状推配置，不依赖 meta 格式
    vocab, dim = sd["transformer.wte.weight"].shape
    n_layer = 1 + max(int(m.group(1)) for k in sd
                      if (m := re.match(r"transformer\.h\.(\d+)\.", k)))
    n_head = max(1, (dim + 127) // 128)        # nanochat 的推导规则
    cfg = dict(sequence_len=SEQ, vocab_size=vocab, n_layer=n_layer,
               n_head=n_head, n_kv_head=n_head, n_embd=dim)

    with torch.device("meta"):
        m = GPT(GPTConfig(**cfg))
    m.to_empty(device=device)
    m.load_state_dict(sd, strict=True)
    m.transformer.wte.to(dtype=torch.bfloat16)                 # 别丢 bf16
    head_dim = dim // n_head
    m.cos, m.sin = m._precompute_rotary_embeddings(m.rotary_seq_len, head_dim)
    m.eval()
    print(f"[{tag}] {fname}  n_layer={n_layer} dim={dim} vocab={vocab}")
    return m

# ---------------------------------------------------------------- 分桶
MASS_EDGES = [1/6, 2/6, 3/6, 4/6, 5/6, 0.875, 0.9167, 0.9583, 1.0]
N_BUCKET = len(MASS_EDGES)          # = 9
def make_buckets(counts, mass_edges=MASS_EDGES):
    order = torch.argsort(counts, descending=True)
    cum = torch.cumsum(counts[order], 0) / counts.sum()
    nb = len(mass_edges)
    bucket_of = torch.full_like(counts, nb - 1, dtype=torch.long)
    prev, sizes = 0, []
    for b, m in enumerate(mass_edges):
        e = int((cum <= m).sum().item())
        e = max(e, prev + 1)
        if b == nb - 1:
            e = len(order)
        bucket_of[order[prev:e]] = b
        sizes.append(e - prev)
        prev = e
    return bucket_of, sizes

# ---------------------------------------------------------------- 分桶 bpb
@torch.inference_mode()
def bucket_bpb(model, batches, bucket_of, token_bytes, nb=N_BUCKET):
    V = token_bytes.numel()
    nats  = torch.zeros(nb, dtype=torch.float64, device=DEVICE)
    bytes_= torch.zeros(nb, dtype=torch.float64, device=DEVICE)
    cnt   = torch.zeros(nb, dtype=torch.float64, device=DEVICE)
    sq    = torch.zeros(nb, dtype=torch.float64, device=DEVICE)
    tnats = torch.zeros(V,  dtype=torch.float64, device=DEVICE)   # 新增
    tcnt  = torch.zeros(V,  dtype=torch.float64, device=DEVICE)   # 新增
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    for x, y in batches:
        with ctx:
            loss = model(x, y, loss_reduction='none').float()
        yf = y.reshape(-1); lf = loss.double(); b = bucket_of[yf]
        nats.index_add_(0, b, lf)
        bytes_.index_add_(0, b, token_bytes[yf].double())
        cnt.index_add_(0, b, torch.ones_like(lf))
        sq.index_add_(0, b, lf ** 2)
        tnats.index_add_(0, yf, lf)                               # 新增
        tcnt.index_add_(0, yf, torch.ones_like(lf))               # 新增

    bpb  = (nats / bytes_ / math.log(2)).cpu()
    mean = nats / cnt
    var  = (sq / cnt - mean ** 2).clamp(min=0)
    se   = ((var * cnt).sqrt() / bytes_ / math.log(2)).cpu()
    return bpb, se, cnt.cpu(), tnats.cpu(), tcnt.cpu()             # 多返回两个
    
    
def build_groups(tokenizer):
    def ids(strs):
        out = set()
        for s in strs:
            t = tokenizer(s)              # 单串编码，返回 id 列表
            if len(t) == 1:
                out.add(t[0])             # 只收单 token，避免混入子词
        return sorted(out)

    letters = [f" {c}" for c in "ABCD"] + list("ABCD")
    digits  = [str(d) for d in range(10)] + [f" {d}" for d in range(10)]
    code    = ["(", ")", "{", "}", "[", "]", "=", ";", ":", "==", "->",
               " def", "def", " return", "return", " import", " self", "_",
               " if", " for", " None", " True", " False"]
    math_op = ["+", "-", "*", "/", "%", " =", " +", " -", "$"]
    return {"answer_letters": ids(letters), "digits": ids(digits),
            "code_syms": ids(code), "math_ops": ids(math_op)}

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=DEVICE)
    counts = torch.load(os.path.join(get_base_dir(), "token_freq.pt"))
    bucket_of, sizes = make_buckets(counts)
    bucket_of = bucket_of.to(DEVICE)
    print(f"bucket vocab sizes (high low freq): {sizes}")

    # 所有模型复用同一批 val batch
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, BS, SEQ, "val")
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(N_BATCH)]
    print(f"val batches: {N_BATCH} x {BS} x {SEQ} = {N_BATCH*BS*SEQ:,} tokens")
    
    
    groups = build_groups(tokenizer)
    for g, v in groups.items():
        bk = bucket_of.cpu()[torch.tensor(v)]
        print(f"{g:16s} n_ids={len(v):3d}  buckets={sorted(set(bk.tolist()))}")

    res, tres = {}, {}
    for name, tag in TAGS.items():
        model = load_model(tag)
        bpb, se, cnt, tnats, tcnt = bucket_bpb(model, batches, bucket_of, token_bytes)
        res[name] = bpb
        tres[name] = (tnats, tcnt)
        overall = (bpb * cnt).sum() / cnt.sum()
        print(f"\n{name:10s} overall = {overall:.4f}, approximal")
        for b in range(N_BUCKET):
            print(f"  b{b}: bpb={bpb[b]:.4f} +-{se[b]:.4f}  n={int(cnt[b]):,}")
        del model; torch.cuda.empty_cache()

    print("\n=== relative degradation vs DDP, % (b0=highest fre 2 b5=longest tail) ===")
    ddp = res["ddp"]
    for name, bpb in res.items():
        if name == "ddp": continue
        rel = (bpb - ddp) / ddp * 100
        print(f"{name:10s} " + "  ".join(f"b{i}:{v:+.2f}" for i, v in enumerate(rel)))
        
    # token 组：per-token nats（不除字节，组内 token 固定所以可比）
    print("\n=== per-token nats by group, and % vs DDP ===")
    tb_ref = None
    for name in TAGS:
        tn, tc = tres[name]
        row = {}
        for g, v in groups.items():
            idx = torch.tensor([i for i in v if tc[i] >= 50])   # 样本太少的 id 丢掉
            row[g] = (tn[idx].sum() / tc[idx].sum()).item() if len(idx) else float('nan')
        if tb_ref is None:
            tb_ref = row
            print(f"{name:10s} " + "  ".join(f"{g}:{v:.4f}" for g, v in row.items()))
        else:
            print(f"{name:10s} " + "  ".join(
                f"{g}:{v:.4f}({(v-tb_ref[g])/tb_ref[g]*100:+.1f}%)" for g, v in row.items()))