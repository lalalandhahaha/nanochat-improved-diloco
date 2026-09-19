import os, re, torch
from nanochat.common import get_base_dir
CKPT = "/home/lyq/.cache/nanochat/base_checkpoints/"

counts = torch.load(os.path.join(get_base_dir(), "token_freq.pt"))
logf = (counts + 1).log().cuda()

for tag in ["d20_ddp", "d20_ddp_n9500", "d20_diloco"]:
    d = os.path.join(CKPT, tag)
    f = sorted(x for x in os.listdir(d) if re.match(r"model_\d+\.pt$", x))[-1]
    b = torch.load(os.path.join(d, f), map_location="cpu")
    W = {k.replace("_orig_mod.",""): v for k,v in b.get("model",b).items()}["lm_head.weight"].float().cuda()
    # Gram 路线，省内存
    e, V = torch.linalg.eigh(W.T @ W)
    s1 = e[-1].clamp(min=0).sqrt(); v1 = V[:, -1]
    u1 = (W @ v1) / s1                                    # (vocab,)
    if u1.mean() < 0: u1 = -u1                            # 符号任意，归一
    r = torch.corrcoef(torch.stack([u1, logf]))[0,1]
    print(f"{tag:16s} s1={s1:9.2f}  corr(u1, log_freq)={r:+.4f}  "
          f"u1_mean/std={u1.mean()/u1.std():+.3f}")