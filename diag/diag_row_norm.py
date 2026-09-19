# scripts/diag_row_norm.py
import os, re, torch
from nanochat.common import get_base_dir
CKPT = "/home/lyq/.cache/nanochat/base_checkpoints/"

def sd(tag):
    d = os.path.join(CKPT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    b = torch.load(os.path.join(d, f), map_location="cpu")
    return {k.replace("_orig_mod.", ""): v.float() for k, v in b.get("model", b).items()}

counts = torch.load(os.path.join(get_base_dir(), "token_freq.pt"))
order = torch.argsort(counts, descending=True)
cum = torch.cumsum(counts[order], 0) / counts.sum()
edges = [int((cum <= (b+1)/6).sum()) for b in range(6)]

a, c = sd("d20_ddp"), sd("d20_diloco")
import torch.nn.functional as F
r = sd("d20_ddp_n9500")          # Run A
o = sd("d20_diloco_orth_0.9")
p = sd("d20_diloco_pro_0.2")

RAND = 1 / (1280 ** 0.5)
for key in ["transformer.wte.weight", "lm_head.weight"]:
    print(f"\n{key}   (random baseline cos = {RAND:.4f})")
    for label, m in [("runA", r), ("diloco", c), ("orth", o), ("pro", p)]:
        prev, out = 0, []
        for b, e in enumerate(edges):
            idx = order[prev:e]; prev = e
            A, C = a[key][idx], m[key][idx]
            cos = F.cosine_similarity(A, C, dim=1).mean().item()
            rat = (C.norm(dim=1).mean() / A.norm(dim=1).mean()).item()
            out.append(f"b{b}:cos={cos:.3f}({cos/RAND:4.1f}x) r={rat:.2f}")
        print(f"  {label:7s} " + "  ".join(out))

for key in ["transformer.wte.weight", "lm_head.weight"]:
    print(f"\n{key} - direction vs scale")
    prev = 0
    for b, e in enumerate(edges):
        idx = order[prev:e]; prev = e
        A, C = a[key][idx], c[key][idx]
        cos = torch.nn.functional.cosine_similarity(A, C, dim=1).mean()
        print(f"  b{b}: cos={cos:.4f}  ratio={C.norm(dim=1).mean()/A.norm(dim=1).mean():.2f}x")


for key in ["transformer.wte.weight", "lm_head.weight"]:
    print(f"\n{key}  (predict 7.5x if coherent, -1.9x if 1-worker-only)")
    prev = 0
    for b, e in enumerate(edges):
        idx = order[prev:e]; prev = e
        ra = a[key][idx].norm(dim=1).mean()
        rc = c[key][idx].norm(dim=1).mean()
        print(f"  b{b} (n={len(idx):6d}): ddp={ra:.4f} diloco={rc:.4f} ratio={rc/ra:5.2f}x")