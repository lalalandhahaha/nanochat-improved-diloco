#!/bin/bash

## h=100,size=[d12,d24,d36,d48],node = 4
##

############################################################ ddp ##############################################
##size =d12 4090上跑完了
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
#     --depth=12 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=0 \
#     --model-tag=d12-ddp-4gpu-0919-h100 \
#     --run=d12_ddp_4gpu_h100 
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
#     -- --run=d12_ddp_4gpu_sft_h100   \
#     --model-tag=d12-ddp-4gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
#     -- --run=d12_ddp_4gpu_rl   \
#     --model-tag=d12-ddp-4gpu-0919-h100


##size =d24 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=2 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --model-tag=d24-ddp-4gpu-0919-h100 \
    --run=d24_ddp_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_sft \
    -- --run=d24_ddp_4gpu_sft_h100   \
    --model-tag=d24-ddp-4gpu-0919-h100 \
    --device-batch-size=2 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_rl \
    -- --run=d24_ddp_4gpu_rl   \
    --device-batch-size=2 \
    --model-tag=d24-ddp-4gpu-0919-h100 


##size =d36
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
    --depth=36 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --model-tag=d36-ddp-4gpu-0919-h100 \
    --run=d36_ddp_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d36_ddp_4gpu_sft_h100   \
    --model-tag=d36-ddp-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d36_ddp_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d36-ddp-4gpu-0919-h100 


##size =d48
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=48 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=0 \
    --model-tag=d48-ddp-4gpu-0919-h100 \
    --run=d48_ddp_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_sft \
    -- --run=d48_ddp_4gpu_sft_h100   \
    --model-tag=d48-ddp-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_rl \
    -- --run=d48_ddp_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d48-ddp-4gpu-0919-h100 




############################################################ diloco ##############################################
##size =d12 4090上跑完了
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
#     --depth=12 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=1 \
#     --diloco-H=100 \
#     --model-tag=d12-diloco-4gpu-0919-h100 \
#     --run=d12_diloco_4gpu_h100 
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
#     -- --run=d12_diloco_4gpu_sft_h100   \
#     --model-tag=d12-diloco-4gpu-0919-h100 \
#     --device-batch-size=4 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
#     -- --run=d12_diloco_4gpu_rl   \
#     --device-batch-size=4 \
#     --model-tag=d12-diloco-4gpu-0919-h100


##size =d24
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-4gpu-0919-h100 \
    --run=d24_diloco_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft_h100   \
    --model-tag=d24-diloco-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h100 


##size =d36
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
    --depth=36 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d36-diloco-4gpu-0919-h100 \
    --run=d36_diloco_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d36_diloco_4gpu_sft_h100   \
    --model-tag=d36-diloco-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d36_diloco_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d36-diloco-4gpu-0919-h100 


##size =d48
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=48 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d48-diloco-4gpu-0919-h100 \
    --run=d48_diloco_4gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_sft \
    -- --run=d48_diloco_4gpu_sft_h100   \
    --model-tag=d48-diloco-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4-m scripts.chat_rl \
    -- --run=d48_diloco_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d48-diloco-4gpu-0919-h100 

############################################################ ours ##############################################
