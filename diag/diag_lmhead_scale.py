# scripts/diag_lmhead_scale.py
#1D 扫描：把 lm_head 乘 s，看 bpb / 饱和率 / CORE 怎么变。
#不是恒等变换 —— 缩小 s 削弱 softcap 压缩，有效 logit 间距反而变大。"""
import os, re, math, torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

CKPT_ROOT = "/home/lyq/.cache/nanochat/base_checkpoints/"
DEVICE, BS, SEQ, N_BATCH = "cuda", 16, 2048, 30
SOFTCAP, STRIDE = 15.0, 211
SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
TARGET = "d20_diloco"
REFS   = {"ddp": "d20_ddp", "runA": "d20_ddp_n9500"}

def load_model(tag, device=DEVICE):
    d = os.path.join(CKPT_ROOT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    blob = torch.load(os.path.join(d, f), map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in blob.get("model", blob).items()}
    vocab, dim = sd["transformer.wte.weight"].shape
    n_layer = 1 + max(int(m.group(1)) for k in sd
                      if (m := re.match(r"transformer\.h\.(\d+)\.", k)))
    n_head = max(1, (dim + 127) // 128)
    with torch.device("meta"):
        m = GPT(GPTConfig(sequence_len=2048, vocab_size=vocab, n_layer=n_layer,
                          n_head=n_head, n_kv_head=n_head, n_embd=dim))
    m.to_empty(device=device)
    m.load_state_dict(sd, strict=True)
    m.transformer.wte.to(dtype=torch.bfloat16)
    m.cos, m.sin = m._precompute_rotary_embeddings(m.rotary_seq_len, dim // n_head)
    m.eval()
    print(f"[{tag}] {f}")
    return m

@torch.inference_mode()
def measure(m, batches, token_bytes):
#一次遍历同时拿 bpb 和 pre-softcap logit 统计"""
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    nats = bytes_ = 0.0
    zs = []
    for x, y in batches:
        with ctx:
            h = m.transformer.wte(x)
            h = F.rms_norm(h, (h.size(-1),))
            cs = (m.cos[:, :x.size(1)], m.sin[:, :x.size(1)])
            for blk in m.transformer.h:
                h = blk(h, cs, None)
            h = F.rms_norm(h, (h.size(-1),))
            z = m.lm_head(h)
        zf = z.float()
        zs.append(zf.flatten()[::STRIDE].cpu())
        logits = SOFTCAP * torch.tanh(zf / SOFTCAP)
        yf = y.reshape(-1)
        nats += F.cross_entropy(logits.view(-1, logits.size(-1)), yf,
                                ignore_index=-1, reduction='sum').double().item()
        bytes_ += token_bytes[yf].double().sum().item()
    z = torch.cat(zs)
    a = z.abs().sort().values
    gs = (1 - torch.tanh(z / SOFTCAP) ** 2).sort().values
    return dict(bpb=nats / bytes_ / math.log(2),
                z_rms=z.pow(2).mean().sqrt().item(),
                sat=(a > SOFTCAP).float().mean().item() * 100,
                grad=(1 - torch.tanh(z / SOFTCAP) ** 2).mean().item(),
                grad_p1=gs[int(0.01 * (gs.numel() - 1))].item())

if __name__ == "__main__":
    tok = get_tokenizer()
    token_bytes = get_token_bytes(device=DEVICE)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, SEQ, "val")
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(N_BATCH)]
    print(f"val: {N_BATCH*BS*SEQ:,} tokens\n")

    hdr = f"{'model/scale':14s} {'bpb':>8s} {'z_rms':>7s} {'sat%':>7s} {'grad':>7s} {'grad_p1':>8s}"
    print(hdr); print("-" * len(hdr))

    for name, tag in REFS.items():
        m = load_model(tag)
        r = measure(m, batches, token_bytes)
        print(f"{name:14s} {r['bpb']:8.4f} {r['z_rms']:7.2f} {r['sat']:6.2f}% "
              f"{r['grad']:7.4f} {r['grad_p1']:8.4f}")
        del m; torch.cuda.empty_cache()
    print()

    m = load_model(TARGET)
    W0 = m.lm_head.weight.detach().clone()
    for s in SCALES:
        with torch.no_grad():
            m.lm_head.weight.copy_(W0 * s)
        r = measure(m, batches, token_bytes)
        print(f"{'diloco s='+f'{s:.2f}':14s} {r['bpb']:8.4f} {r['z_rms']:7.2f} "
              f"{r['sat']:6.2f}% {r['grad']:7.4f} {r['grad_p1']:8.4f}")
    with torch.no_grad():
        m.lm_head.weight.copy_(W0)