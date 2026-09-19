#!/bin/bash

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --run=d24_ddp_4gpu \
    --model-tag=d24-ddp-4gpu-0909
# evaluate the model: CORE metric, BPB on train/val, and draw samples
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_ddp_4gpu_sft   \
    --model-tag=d24-ddp-4gpu-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_ddp_4gpu_rl   \
    --model-tag=d24-ddp-4gpu-0909 

# CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
#      -- --device-batch-size=4 \
#      --model-tag=d24-ddp-4gpu-0909 
# CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \
#     -- --i sft \
#     --model-tag=d24-ddp-4gpu-0909





CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-4gpu-0909 \
    --run=d24_diloco_4gpu



CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-regular-1e-4-4gpu-0909 \
    --run=d24_diloco_regular_1e-4_4gpu \
    --lambda-reg=1e-4



# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"



CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft   \
    --model-tag=d24-diloco-4gpu-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl   \
    --model-tag=d24-diloco-4gpu-0909 
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_regular_1e-4_4gpu_sft   \
    --model-tag=d24-diloco-regular-1e-4-4gpu-0909 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_regular_1e-4_4gpu_rl   \
    --model-tag=d24-diloco-regular-1e-4-4gpu-0909 




CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    -- --device-batch-size=4 \
     --model-tag=d24-diloco-regular-1e-4-4gpu-0909
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \
    -- --i sft \
    --model-tag=d24-diloco-regular-1e-4-4gpu-0909

CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
     -- --device-batch-size=4 \
     --model-tag=d24-diloco-4gpu-0909 
CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval \
    -- --i sft \
    --model-tag=d24-diloco-4gpu-0909