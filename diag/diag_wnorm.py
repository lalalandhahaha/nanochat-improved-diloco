# scripts/diag_wnorm.py
#按模块/逐层报权重范数，检查 DiLoCo 的 5.8x 范数差异是全局还是局部。"""
import os, re, torch

CKPT_ROOT = "/home/lyq/.cache/nanochat/base_checkpoints/"
TAGS = {
    "ddp":      "d20_ddp",
    "runA":     "d20_ddp_n9500",
    "diloco":   "d20_diloco",
    "orth_0.9": "d20_diloco_orth_0.9",
    "pro_0.2":  "d20_diloco_pro_0.2",
}

def load_sd(tag):
    d = os.path.join(CKPT_ROOT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    blob = torch.load(os.path.join(d, f), map_location="cpu")
    sd = blob.get("model", blob)
    return {k.replace("_orig_mod.", ""): v.float() for k, v in sd.items()}, f

def stats(sd, keys):
    n2 = sum((sd[k] ** 2).sum().item() for k in keys)
    d  = sum(sd[k].numel() for k in keys)
    return n2 ** 0.5, (n2 / d) ** 0.5          # |W|, rms

if __name__ == "__main__":
    rows = {}
    for name, tag in TAGS.items():
        sd, f = load_sd(tag)
        print(f"\n=== {name}  ({f}) ===")
        groups = {
            "wte":     ["transformer.wte.weight"],
            "lm_head": ["lm_head.weight"],
            "attn":    [k for k in sd if ".attn." in k],
            "mlp":     [k for k in sd if ".mlp."  in k],
            "body":    [k for k in sd if k.startswith("transformer.h.")],
        }
        rows[name] = {}
        for g, keys in groups.items():
            w, rms = stats(sd, keys)
            rows[name][g] = rms
            d = sum(sd[k].numel() for k in keys)
            print(f"  {g:8s} |W|={w:11.2f}  rms={rms:.6f}  d={d:,}")

        print("  per-layer rms:", end="")
        for i in range(0, 20, 2):
            ks = [k for k in sd if k.startswith(f"transformer.h.{i}.")]
            print(f" L{i}:{stats(sd, ks)[1]:.4f}", end="")
        print()

        for sub in ["c_q", "c_k", "c_v", "attn.c_proj", "c_fc", "mlp.c_proj"]:
            ks = [k for k in sd if sub in k]
            if ks:
                print(f"    {sub:12s} rms={stats(sd, ks)[1]:.6f}")
        del sd

    print("\n=== rms ratio vs DDP ===")
    ref = rows["ddp"]
    for name, r in rows.items():
        print(f"{name:10s} " + "  ".join(f"{g}:{r[g]/ref[g]:5.2f}x" for g in ref))