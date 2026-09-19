import json
import wandb

run = wandb.init(
    project="nanochat-improved-diloco",
    name="replay-metrics",
    mode="offline",
    dir="./wandb_replay",
)

with open("/data1/users/liuyingqi2/nanochat-improved-diloco/wandb_local/20260911_dummy/metrics.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        wandb.log(data)

run.finish()