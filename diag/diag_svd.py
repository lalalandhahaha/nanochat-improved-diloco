# scripts/diag_svd.py
import os, re, torch
CKPT = "/data1/users/liuyingqi2/.cache/nanochat/base_checkpoints/"
TAGS = {
    "ddp":      "d12-ddp-4gpu-0909",
    "diloco":   "d12-diloco-4gpu-0909",
    "regular_1e-4": "d12-diloco-regular-1e-4-4gpu-0909",
}

def sd(tag):
    d = os.path.join(CKPT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    b = torch.load(os.path.join(d, f), map_location="cpu")
    return {k.replace("_orig_mod.",""): v for k,v in b.get("model",b).items()}

def spec(W):
    W = W.float().cuda()
    if W.shape[0] > 20000:                    # lm_head: �� Gram ����
        s = torch.linalg.eigvalsh(W.T @ W).clamp(min=0).sqrt().flip(0)
    else:
        s = torch.linalg.svdvals(W)
    fro = s.norm()
    p = (s / fro) ** 2
    return dict(smax=s[0].item(), fro=fro.item(),
                conc=(s[0]/fro).item(),                    # �׼��ж�
                srank=(fro**2/s[0]**2).item(),             # stable rank
                erank=torch.exp(-(p*p.clamp(min=1e-12).log()).sum()).item())

if __name__ == "__main__":
    KEYS = ([f"transformer.h.{i}.{m}" for i in [0,3,6,9,11] 
             for m in ["attn.c_q.weight","attn.c_v.weight","mlp.c_fc.weight"]]
            + ["lm_head.weight"])
    out = {}
    for name, tag in TAGS.items():
        m = sd(tag); out[name] = {k: spec(m[k]) for k in KEYS}
        print(f"[{name}] done"); del m; torch.cuda.empty_cache()

    for k in KEYS:
        print(f"\n{k}")
        r = out["ddp"][k]
        for n in TAGS:
            v = out[n][k]
            print(f"  {n:9s} smax={v['smax']:8.3f}({v['smax']/r['smax']:5.2f}x) "
                  f"conc={v['conc']:.4f}  srank={v['srank']:7.1f}  erank={v['erank']:7.1f}")