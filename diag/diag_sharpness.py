# scripts/diag_sharpness.py
#Hessian trace (Hutchinson) + lambda_max (power iteration).
#检验 DiLoCo 是否走向 flatter minima
import os, re, math, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit




DEVICE   = "cuda"
BS, SEQ  = 2, 512      # HVP 要保留计算图，比 eval 小得多；OOM 就降到 1x512
N_BATCH  = 16           # 每次 HVP 在这 4 个 batch 上累加（固定，跨模型一致）
N_HUTCH  = 20          # Hutchinson 采样数
N_POWER  = 60        # power iteration 步数
SEED     = 1234

TAGS = {
    "ddp":      "d20_ddp",
    "runA":     "d20_ddp_n9500",
    "diloco":   "d20_diloco",
    "orth_0.9": "d20_diloco_orth_0.9",
    "pro_0.2":  "d20_diloco_pro_0.2",
}


import nanochat.gpt as gptmod
_orig_forward = gptmod.GPT.forward
def _forward_nocheck(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
    x = self.transformer.wte(idx)
    x = gptmod.norm(x)
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)
    x = gptmod.norm(x)
    logits = self.lm_head(x)
    logits = 15 * torch.tanh(logits / 15)
    if targets is None:
        return logits
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, reduction=loss_reduction)
gptmod.GPT.forward = _forward_nocheck



# ---------------------------------------------------------------- SDPA math 后端
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def math_sdpa(): return sdpa_kernel(SDPBackend.MATH)
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def math_sdpa():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        yield
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

# ---------------------------------------------------------------- 加载（全 fp32）
def load_model_fp32(tag, device=DEVICE):
    ckpt_dir = os.path.join("/home/lyq/.cache/nanochat/base_checkpoints/", tag)
    files = sorted(f for f in os.listdir(ckpt_dir) if re.match(r"model_\d+\.pt$", f))
    fname = files[-1]
    blob = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu")
    sd = blob.get("model", blob)
    sd = {k.replace("_orig_mod.", ""): v.float() for k, v in sd.items()}   # 全转 fp32

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
    m.float()                                   # wte 默认 bf16，必须覆盖
    head_dim = dim // n_head
    cos, sin = m._precompute_rotary_embeddings(m.rotary_seq_len, head_dim)
    m.cos, m.sin = cos.float(), sin.float()     # forward 里的 bf16 assert 需要放宽，见下
    m.eval()
    print(f"[{tag}] {fname}")
    return m
    
    
def loss_on(model, batches):
    #在固定 batch 上的平均 loss，保留计算图
    tot = 0
    for x, y in batches:
        tot = tot + model(x, y)
    return tot / len(batches)

def hvp(model, params, batches, v):
    loss = loss_on(model, batches)
    g = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((gi * vi).sum() for gi, vi in zip(g, v))
    Hv = torch.autograd.grad(dot, params, retain_graph=False)
    return [h.detach() for h in Hv], loss.item()

def rademacher_like(params, gen):
    return [(torch.randint(0, 2, p.shape, generator=gen, device=p.device,
                           dtype=torch.float32) * 2 - 1) for p in params]

def hutchinson_trace(model, params, batches, n=N_HUTCH):
    gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)  # 跨模型同一组向量
    ests = []
    for i in range(n):
        v = rademacher_like(params, gen)
        Hv, _ = hvp(model, params, batches, v)
        ests.append(sum((h * vi).sum().item() for h, vi in zip(Hv, v)))
    t = torch.tensor(ests)
    return t.mean().item(), (t.std() / math.sqrt(n)).item()

SHIFT = 800.0   # 需 > |λ_min|；pro 的量级到 540，取 800 保险

def power_iteration(model, params, batches, n=N_POWER, shift=SHIFT):
    gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED + 1)
    v = [torch.randn(p.shape, generator=gen, device=p.device) for p in params]
    nrm = math.sqrt(sum((x*x).sum().item() for x in v)); v = [x/nrm for x in v]
    lam, prev = 0.0, None
    for i in range(n):
        Hv, _ = hvp(model, params, batches, v)
        Hv = [h + shift * vi for h, vi in zip(Hv, v)]        # 位移
        lam = sum((h*vi).sum().item() for h, vi in zip(Hv, v)) - shift
        nrm = math.sqrt(sum((h*h).sum().item() for h in Hv))
        if nrm < 1e-12: break
        v = [h/nrm for h in Hv]
        if prev is not None and abs(lam-prev) < 1e-3*max(abs(lam),1e-8):
            print(f"    converged at iter {i}: {lam:.4f}"); break
        prev = lam
    else:
        print(f"    WARNING not converged, last={lam:.4f}")
    return lam
    
    
if __name__ == "__main__":
    tokenizer = get_tokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, BS, SEQ, "val")
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(N_BATCH)]
    print(f"HVP batches: {N_BATCH} x {BS} x {SEQ}")

    print(f"\n{'model':10s} {'group':6s} {'loss':>7s} {'trace':>12s} {'lam_max':>10s} "
          f"{'|theta|':>9s} {'lam*|th|^2':>12s} {'tr/d':>10s}")
    with math_sdpa():
        for name, tag in TAGS.items():
            model = load_model_fp32(tag)
            for gname, sel in [("body", lambda n: n.startswith("transformer.h.")),
                               ("full", lambda n: True)]:
                params = [p for n_, p in model.named_parameters() if sel(n_)]
                for p in model.parameters(): p.requires_grad_(False)
                for p in params: p.requires_grad_(True)

                d = sum(p.numel() for p in params)
                theta = math.sqrt(sum((p.detach() ** 2).sum().item() for p in params))
                tr, tr_se = hutchinson_trace(model, params, batches)
                lam = power_iteration(model, params, batches)
                with torch.no_grad():
                    L = sum(model(x, y).item() for x, y in batches) / len(batches)
                print(f"{name:10s} {gname:6s} {L:7.4f} {tr:12.2f} {lam:10.4f} "
                      f"{theta:9.2f} {lam*theta**2:12.1f} {tr/d:10.3e}   (se_tr={tr_se:.2f})")
            del model; torch.cuda.empty_cache() 