#!/bin/bash

## h=[50,100,200,500],size=d24,node = 4
##

###ddp没有inner step，diloco有inner step
############################################################ diloco ##############################################
##inner step =50
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=50 \
    --model-tag=d24-diloco-4gpu-0919-h50 \
    --run=d24_diloco_4gpu_h50 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft_h50   \
    --model-tag=d24-diloco-4gpu-0919-h50 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl_h50   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h50


##inner step =100 这组事baseline 不用跑
# CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
#     --depth=24 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=1 \
#     --diloco-H=100 \
#     --model-tag=d24-diloco-2gpu-0919-h100 \
#     --run=d24_diloco_2gpu_h100
# CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
#     -- --run=d24_diloco_2gpu_sft_h100   \
#     --model-tag=d24-diloco-2gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl \
#     -- --run=d24_diloco_2gpu_rl_h100   \
#     --model-tag=d24-diloco-2gpu-0919-h100 


##inner step =200
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=200 \
    --model-tag=d24-diloco-4gpu-0919-h200 \
    --run=d24_diloco_4gpu_h200
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft_h200   \
    --model-tag=d24-diloco-4gpu-0919-h200 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl_h200   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h200 


##inner step =500
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=500 \
    --model-tag=d24-diloco-4gpu-0919-h500 \
    --run=d24_diloco_4gpu_h500
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft_h500   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h500 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl_h500   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h500 

############################################################ ours ##############################################
