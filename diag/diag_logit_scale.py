# scripts/diag_logit_scale.py
#查 softcap 饱和：DiLoCo 的 lm_head 大 7x，pre-softcap logit 是否进入
#tanh 饱和区导致输出层梯度被掐断。"""
import os, re, torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

CKPT_ROOT = "/home/lyq/.cache/nanochat/base_checkpoints/"
DEVICE    = "cuda"
BS, SEQ   = 8, 1024
N_BATCH   = 4
SOFTCAP   = 15.0
STRIDE    = 211          # logit 抽样步长，省内存

TAGS = {
    "ddp":      "d20_ddp",
    "runA":     "d20_ddp_n9500",
    "diloco":   "d20_diloco",
    "orth_0.9": "d20_diloco_orth_0.9",
    "pro_0.2":  "d20_diloco_pro_0.2",
}

def load_model(tag, device=DEVICE):
    d = os.path.join(CKPT_ROOT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    blob = torch.load(os.path.join(d, f), map_location="cpu")
    sd = blob.get("model", blob)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    vocab, dim = sd["transformer.wte.weight"].shape
    n_layer = 1 + max(int(m.group(1)) for k in sd
                      if (m := re.match(r"transformer\.h\.(\d+)\.", k)))
    n_head = max(1, (dim + 127) // 128)
    cfg = dict(sequence_len=2048, vocab_size=vocab, n_layer=n_layer,
               n_head=n_head, n_kv_head=n_head, n_embd=dim)
    with torch.device("meta"):
        m = GPT(GPTConfig(**cfg))
    m.to_empty(device=device)
    m.load_state_dict(sd, strict=True)
    m.transformer.wte.to(dtype=torch.bfloat16)
    m.cos, m.sin = m._precompute_rotary_embeddings(m.rotary_seq_len, dim // n_head)
    m.eval()
    print(f"[{tag}] {f}")
    return m

@torch.inference_mode()
def logit_stats(m, batches):
    #复刻 forward 的 trunk，但在 softcap 之前截住 logits"""
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    zs, hrms = [], []
    for x, y in batches:
        with ctx:
            h = m.transformer.wte(x)
            h = F.rms_norm(h, (h.size(-1),))
            cs = (m.cos[:, :x.size(1)], m.sin[:, :x.size(1)])
            for blk in m.transformer.h:
                h = blk(h, cs, None)
            h = F.rms_norm(h, (h.size(-1),))
            z = m.lm_head(h)
        hrms.append(h.float().pow(2).mean().sqrt().item())
        zs.append(z.float().flatten()[::STRIDE].cpu())
    return torch.cat(zs), sum(hrms) / len(hrms)

if __name__ == "__main__":
    tok = get_tokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, SEQ, "val")
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(N_BATCH)]
    print(f"probe: {N_BATCH} x {BS} x {SEQ} = {N_BATCH*BS*SEQ:,} positions\n")

    hdr = (f"{'model':10s} {'z_rms':>8s} {'z_p99':>8s} {'z_max':>8s} "
           f"{'sat%':>7s} {'grad':>7s} {'grad_p1':>8s} {'h_rms':>7s}")
    print(hdr); print("-" * len(hdr))

    for name, tag in TAGS.items():
        m = load_model(tag)
        z, h_rms = logit_stats(m, batches)
        g = 1 - torch.tanh(z / SOFTCAP) ** 2          # softcap 的梯度因子
        print(f"{name:10s} {z.pow(2).mean().sqrt():8.3f} "
              f"{z.abs().quantile(0.99):8.3f} {z.abs().max():8.2f} "
              f"{(z.abs()>SOFTCAP).float().mean()*100:6.2f}% "
              f"{g.mean():7.4f} {g.quantile(0.01):8.4f} {h_rms:7.3f}")
        del m, z, g; torch.cuda.empty_cache()

    print("\ngrad = softcap ") #梯度因子均值 越小说明饱和越严重 --1.0 = 完全不饱和
    print("grad_p1 = ") #最饱和的 百分之1  位置上的梯度因子