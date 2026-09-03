#!/bin/bash

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=20 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --run=d20_ddp_4gpu \
    --model-tag=d20-ddp-4gpu-0903
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- --device-batch-size=16 --run=d20_ddp_4gpu --model-tag=d20-ddp-4gpu-0903

torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=20 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d20-diloco-4gpu-0903 \
    --run=d20_diloco_4gpu

torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- --device-batch-size=16 --run=d20_diloco_4gpu --model-tag=d20-diloco-4gpu-0903

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=d20_ddp_4gpu_sft   --model-tag=d20-ddp-4gpu-sft-0903
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft --model-tag=d20-ddp-4gpu-sft-0903
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=d20_diloco_4gpu_sft   --model-tag=d20-diloco-4gpu-sft-0903
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft --model-tag=d20-diloco-4gpu-sft-0903
# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"
