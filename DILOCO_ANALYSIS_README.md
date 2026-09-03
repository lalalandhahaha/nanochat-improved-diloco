# DiLoCo 权重分析和训练改进

本文档说明新增的 DiLoCo 节点差异分析功能和训练改进。

## 1. 权重矩阵分析功能

### 1.1 功能概述

训练过程中会自动分析以下指标：

**单个权重矩阵指标：**
- `s_max`: 最大奇异值
- `s_min`: 最小奇异值  
- `condition_number`: 条件数（s_max / s_min）
- `stable_rank`: 稳定秩（||W||_F² / ||W||_2²）
- `effective_rank`: 有效秩（基于奇异值熵）
- `frobenius_norm`: Frobenius 范数
- `spectral_norm`: 谱范数

**DiLoCo 节点间差异指标（仅 DiLoCo 模式）：**
- `cosine_sim_mean/std/min`: 节点间权重余弦相似度
- `weight_diff_mean/std/max`: 节点间权重差异范数

### 1.2 分析的层

- 自动等间隔采样5层（首层、1/4、中间、3/4、尾层）
- 适应任意模型规模（10层、20层、30层等）
- 每层分析：`c_q, c_k, c_v, c_proj`（attention）和 `c_fc, c_proj`（MLP）

### 1.3 输出位置

**控制台输出：**
```
====================================================================================================
Weight Matrix Analysis at Step 500
====================================================================================================
lm_head.weight                           | shape: (50257, 1280) | s_max:   1.5730 | ...
layer_0.attn.c_q.weight                  | shape: (1280, 1280) | s_max:   0.0015 | ...
...
====================================================================================================
```

**W&B 日志：**
- 普通权重分析：`weights/{层名}/{指标名}`
- DiLoCo 差异分析：`diloco_diff/{层名}/{指标名}`

**本地日志（dummy 模式）：**
- 目录：`wandb_local/{YYYYMMDD}_{run_name}/`
- 文件：`metrics.jsonl`（逐行）、`summary.json`（总结）

## 2. DiLoCo 节点差异分析

### 2.1 分析时机

**严格在聚合前观测：**
- 在 `DiLoCoWrapper._outer_step()` 的 `all_reduce` **之前**触发回调
- 确保观测到的是各节点独立训练后、同步前的权重差异
- 只在 eval step 且恰好是 outer sync 时触发

### 2.2 关键指标

**最重要的漂移指标：**
1. `cosine_sim_mean`: 越接近1越好（权重相似）
2. `weight_diff_mean`: 越小越好（权重差异小）

**结构健康度指标：**
3. `condition_number`: 过大说明矩阵病态
4. `stable_rank` 和 `effective_rank`: 监控秩的变化

### 2.3 输出示例

```
====================================================================================================
DiLoCo Node Weight Differences at Step 500 (Before Outer Sync)
====================================================================================================
lm_head.weight
  Max Singular Value  : mean=1.9632 std=0.012345 min=1.9500 max=1.9800
  Condition Number    : mean=12.08 std=0.50
  Stable Rank         : mean=7.07 std=0.12
  Effective Rank      : mean=41.67 std=0.15
  Frobenius Norm      : mean=5.22 std=0.01
  Cosine Similarity   : mean=0.999800 std=0.000050 min=0.999700
  Weight Diff (vs rank0): mean=0.012345 std=0.001234 max=0.015678
====================================================================================================
```

## 3. 本地日志 (Dummy Mode)

### 3.1 启用方式

```bash
# 默认就是 dummy mode（不需要 wandb API key）
python -m scripts.base_train --run dummy

# 使用真实 wandb
python -m scripts.base_train --run my_experiment_name
```

### 3.2 日志文件结构

```
wandb_local/
├── 20260903_dummy/                    # 日期 + run name
│   ├── metrics.jsonl                  # 每行一个 JSON 对象
│   └── summary.json                   # 最终总结
├── 20260903_my_experiment/
│   ├── metrics.jsonl
│   └── summary.json
└── ...
```

### 3.3 清理日志

```bash
# 手动删除旧日期的文件夹
rm -rf wandb_local/20260901_*
rm -rf wandb_local/20260902_*
```

## 4. Checkpoint Resume

### 4.1 保存的状态

训练脚本会保存：
- ✅ 模型参数
- ✅ Optimizer 状态（包括 DiLoCo 的 inner 和 outer optimizer）
- ✅ Dataloader 状态
- ✅ 循环状态（step、min_val_bpb、smooth_train_loss、total_training_time）
- ⚠️ **未保存**：随机数状态（不能保证 bit-exact resume）

### 4.2 使用方式

```bash
# 从 step 1000 恢复训练
python -m scripts.base_train --resume-from-step 1000

# 配合 --save-every 定期保存
python -m scripts.base_train --save-every 500

# 完整示例：从 step 500 恢复，每 500 步保存一次
torchrun --nproc_per_node=8 -m scripts.base_train \
  --resume-from-step 500 \
  --save-every 500 \
  --run my_diloco_experiment
```

### 4.3 DiLoCo Resume 特别说明

DiLoCo 会恢复：
- ✅ Inner optimizer 状态（AdamW + Muon）
- ✅ Outer optimizer 状态（SGD with momentum）
- ✅ `params_offloaded`（上次同步的权重快照）
- ✅ `inner_step_count`（当前 inner step 计数）

## 5. 独立评估脚本 (base_eval.py)

### 5.1 W&B 支持

```bash
# Dummy mode（本地保存）
python -m scripts.base_eval --model-tag d24 --run dummy

# W&B mode（上传到 wandb）
python -m scripts.base_eval --model-tag d24 --run eval_d24_step1000

# 多 GPU 评估
torchrun --nproc_per_node=8 -m scripts.base_eval \
  --model-tag d24 \
  --run eval_d24 \
  --eval core,bpb,sample
```

### 5.2 上传的指标

- `eval/train_bpb` 和 `eval/val_bpb`
- `eval/core_metric`（总体 CORE 分数）
- `eval/core/{task_name}/accuracy` 和 `eval/core/{task_name}/centered`
- `eval/conditioned_samples` 和 `eval/unconditioned_samples`（文本）

## 6. 模块化代码结构

```
nanochat/
├── weight_analysis.py        # 权重矩阵分析（单个节点）
│   ├── power_iteration()
│   ├── inverse_power_iteration()
│   ├── compute_weight_metrics()
│   └── analyze_model_weights()
│
├── diloco_analysis.py        # DiLoCo 节点差异分析
│   ├── compute_cosine_similarity()
│   ├── compute_weight_difference_norm()
│   ├── gather_weights_from_all_nodes()
│   └── analyze_node_weight_differences()
│
├── diloco.py                 # DiLoCo wrapper
│   └── DiLoCoWrapper
│       ├── pre_sync_callback  # 新增：聚合前回调
│       ├── state_dict()
│       └── load_state_dict()
│
└── common.py
    └── DummyWandb              # 新增：本地日志支持
        ├── __init__(save_local=True)
        ├── log()
        └── finish()
```

## 7. 故障排查

### 7.1 DiLoCo 分析未触发

**检查：**
1. 是否启用了 DiLoCo：`--use-diloco 1`
2. 是否启用了 eval：`--eval-every 500`
3. 步数是否恰好对齐：eval step 必须是 diloco_H 的倍数

### 7.2 条件数爆炸（> 10^6）

**可能原因：**
- DiLoCo 的局部优化导致权重矩阵病态
- Projection 层（c_proj）最敏感

**建议：**
1. 降低 `--diloco-H`（更频繁同步）
2. 增加 `--weight-decay`
3. 降低 `--diloco-outer-lr`

### 7.3 节点间权重差异过大

**监控指标：**
- `cosine_sim_mean < 0.99`：警告
- `weight_diff_mean > 0.1`：严重

**建议：**
1. 降低 `--diloco-H` 从 100 到 50
2. 检查数据是否均匀分布到各节点
3. 验证学习率是否过大

## 8. 最佳实践

### 8.1 训练监控

**重点关注：**
1. `diloco_diff/lm_head.weight/cosine_sim_mean`（最敏感层）
2. `diloco_diff/*/cond_mean`（所有层的条件数）
3. `weights/*/stable_rank`（稳定秩变化趋势）

### 8.2 实验设置

**推荐配置：**
```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
  --use-diloco 1 \
  --diloco-H 100 \
  --diloco-outer-lr 0.75 \
  --diloco-outer-momentum 0.9 \
  --eval-every 500 \
  --save-every 500 \
  --run diloco_experiment_$(date +%Y%m%d)
```

**调试配置（快速验证）：**
```bash
python -m scripts.base_train \
  --depth 4 \
  --max-seq-len 512 \
  --device-batch-size 1 \
  --total-batch-size 512 \
  --num-iterations 20 \
  --use-diloco 1 \
  --diloco-H 5 \
  --eval-every 5 \
  --run debug
```

## 9. 参考文献

- **DiLoCo**: [Distributed Low-Communication Training](https://arxiv.org/abs/2311.08105)
- **条件数与训练稳定性**: 条件数过大会导致梯度消失/爆炸
- **稳定秩**: 衡量矩阵的"有效维度"，值越高说明权重更新越分散
