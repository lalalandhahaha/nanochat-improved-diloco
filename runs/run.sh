#!/bin/bash

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
CUDA_VISIBLE_DEVICES=4,5,6,0 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --run=d12_ddp_4gpu \
    --model-tag=d12-ddp-4gpu-0909
# evaluate the model: CORE metric, BPB on train/val, and draw samples
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    --run=d12_ddp_4gpu_sft   \
    --model-tag=d12-ddp-4gpu-sft-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    --run=d12_ddp_4gpu_sft   \
    --model-tag=d12-ddp-4gpu-sft-0909 




CUDA_VISIBLE_DEVICES=4,5,6,0 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d12-diloco-4gpu-0909 \
    --run=d12_diloco_4gpu
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    --run=d12_diloco_4gpu_sft   \
    --model-tag=d12-diloco-4gpu-sft-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    --run=d12_diloco_4gpu_sft   \
    --model-tag=d12-diloco-4gpu-sft-0909 




CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d12-diloco-regular-1e-4-4gpu-0909 \
    --run=d12_diloco_regular_1e-4_4gpu \
    --lambda-reg=1e-4
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    --run=d12_diloco_regular_1e-4_4gpu_sft   \
    --model-tag=d12-diloco-regular-1e-4-4gpu-sft-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    --run=d12_diloco_regular_1e-4_4gpu_sft   \
    --model-tag=d12-diloco-regular-1e-4-4gpu-sft-0909 


# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    --device-batch-size=8 \
    --run=d12_ddp_4gpu \
    --model-tag=d12-ddp-4gpu-0909
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \ 
    --i sft \
    --model-tag=d12-ddp-4gpu-sft-0909 \

    


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
     --device-batch-size=16 \
     --run=d12_diloco_4gpu \
     --model-tag=d12-diloco-4gpu-0909 \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \
    --i sft \
    --model-tag=d12-diloco-4gpu-sft-0909




CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
     --device-batch-size=16 \
     --run=d12_diloco_regular_1e-4_4gpu \
     --model-tag=d12-diloco-regular-1e-4-4gpu-0909
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \
    --i sft \
    --model-tag=d12-diloco-regular-1e-4-4gpu-sft-0909


# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"
