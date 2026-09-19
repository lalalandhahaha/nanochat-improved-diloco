# scripts/diag_spectral.py
"""幂迭代测最大奇异值（含 top-k），覆盖所有层 + wte/lm_head。
比全 SVD 快得多，可以直接搬进训练循环做逐步追踪。"""
import os, re, torch

CKPT_ROOT = "/data1/users/liuyingqi2/.cache/nanochat/base_checkpoints/"
DEVICE  = "cuda"
TOPK    = 4
N_ITER  = 300
TOL     = 1e-7
SEED    = 0

TAGS = {
    "ddp":      "d12-ddp-4gpu-0909",
    "diloco":   "d12-diloco-4gpu-0909",
    "regular_1e-4": "d12-diloco-regular-1e-4-4gpu-0909",
}
MODULES = ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

# --------------------------------------------- 幂迭代
def top_sv(W, k=TOPK, n_iter=N_ITER, tol=TOL, seed=SEED):
    """块正交迭代（k=1 时退化为普通幂迭代）。返回 (奇异值降序, 迭代数, 是否收敛)"""
    W = W.float().to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    k = min(k, min(W.shape))
    V = torch.randn(W.shape[1], k, device=DEVICE, generator=g)
    V, _ = torch.linalg.qr(V)
    prev, conv = None, False
    for i in range(n_iter):
        U, _ = torch.linalg.qr(W @ V)
        V, R = torch.linalg.qr(W.T @ U)
        s = R.diagonal().abs()
        if prev is not None and (s - prev).abs().max() < tol * s.max().clamp(min=1e-12):
            conv = True; break
        prev = s
    return s.sort(descending=True).values.cpu(), i + 1, conv

def stats(W, k=TOPK):
    s, it, conv = top_sv(W, k)
    fro = W.float().to(DEVICE).norm().item()
    return dict(s=s, fro=fro, conc=s[0].item() / fro,
                srank=(fro / s[0].item()) ** 2, iters=it, conv=conv)

# ------------------------------------------- 加载
def load_sd(tag):
    d = os.path.join(CKPT_ROOT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    b = torch.load(os.path.join(d, f), map_location="cpu")
    return {k.replace("_orig_mod.", ""): v for k, v in b.get("model", b).items()}, f

# --------------------------------------------- sanity check
def verify():
    g = torch.Generator(device=DEVICE).manual_seed(7)
    A = torch.randn(1280, 1280, device=DEVICE, generator=g)
    ref = torch.linalg.svdvals(A)[:TOPK].cpu()
    got, it, conv = top_sv(A)
    err = ((got - ref).abs() / ref).max().item()
    print(f"[verify] power-iter vs svdvals: max rel err={err:.2e} "
          f"(iters={it}, conv={conv})  {'OK' if err < 1e-4 else 'FAIL'}\n")

# --------------------------------------- main
if __name__ == "__main__":
    verify()
    out = {}

    for name, tag in TAGS.items():
        sd, f = load_sd(tag)
        print(f"[{name}] {f}", flush=True)
        rec = {}

        # 输出层 / 嵌入层：多报 top-k 和去 DC 后的 s1
        for key in ["lm_head.weight", "transformer.wte.weight"]:
            W = sd[key]
            r = stats(W)
            Wc = W.float().to(DEVICE)
            Wc = Wc - Wc.mean(0, keepdim=True)          # 去掉跨词表共享方向(DC)
            r["s1_centered"] = top_sv(Wc, k=1)[0][0].item()
            rec[key] = r

        # body：逐层逐模块，只要 s1
        for i in range(12):
            for mod in MODULES:
                key = f"transformer.h.{i}.{mod}.weight"
                if key in sd:
                    rec[key] = stats(sd[key], k=1)

        out[name] = rec
        del sd; torch.cuda.empty_cache()

    # ------------------------------------------------ 输出层
    for key in ["lm_head.weight", "transformer.wte.weight"]:
        print(f"\n=== {key} ===")
        print(f"{'model':10s} {'s1':>10s} {'s2':>9s} {'s3':>9s} {'s4':>9s} "
              f"{'s2/s1':>7s} {'fro':>10s} {'conc':>7s} {'srank':>8s} "
              f"{'s1_noDC':>10s} {'DC frac':>8s}  it")
        ref = out["ddp"][key]["s"][0].item()
        for n in TAGS:
            r = out[n][key]; s = r["s"]
            dc = 1 - (r["s1_centered"] / s[0].item()) ** 2   # s1 里 DC 分量的能量占比
            print(f"{n:10s} {s[0]:10.2f} {s[1]:9.2f} {s[2]:9.2f} {s[3]:9.2f} "
                  f"{s[1]/s[0]:7.3f} {r['fro']:10.2f} {r['conc']:7.4f} "
                  f"{r['srank']:8.2f} {r['s1_centered']:10.2f} {dc:7.1%}  ")
                  #f"{r['iters']}{' if r['conv'] else '!'}  ({s[0]/ref:.2f}x vs ddp)") 

    # ------------------------------------ body 绝对值
    print(f"\n=== body s1 (absolute) ===")
    for n in TAGS:
        print(f"\n{n}")
        print("  layer " + " " .join(f"{m.split('.')[-1]:>10s}" for m in MODULES))
        for i in range(12):
            vals = [out[n].get(f"transformer.h.{i}.{m}.weight", {}).get("s", None)
                    for m in MODULES]
            print(f"  L{i:<4d} " + "".join(
                f"{v[0].item():10.2f}" if v is not None else f"{'-':>10s}" for v in vals))

    # ------------------------------------------------ body 比值
    print(f"\n=== body s1 ratio vs DDP ===")
    for n in TAGS:
        if n == "ddp": continue
        print(f"\n{n}")
        print("  layer " + " " .join(f"{m.split('.')[-1]:>9s}" for m in MODULES))
        for i in range(12):
            row = []
            for m in MODULES:
                k = f"transformer.h.{i}.{m}.weight"
                a, b = out["ddp"].get(k), out[n].get(k)
                row.append(f"{b['s'][0].item()/a['s'][0].item():8.2f}x"
                           if a and b else f"{'-':>9s}")
            print(f"  L{i:<4d} " + "".join(row))

    nonconv = [(n, k) for n in out for k, r in out[n].items() if not r["conv"]]
    if nonconv:
        print(f"\nWARNING 未收敛 {len(nonconv)} 项，前几个: {nonconv[:5]}")