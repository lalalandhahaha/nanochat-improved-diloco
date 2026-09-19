#!/bin/bash

## h=100,size=d24,node = [1,2,4,8]
##node的ddp还需要再考虑，因为没有累计的问题，累计token是一致的。

# ############################################################ ddp ##############################################
# ##node =1
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
#     --depth=24 \
#     --max-seq-len=2048 \
#     --device-batch-size=2 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=0 \
#     --model-tag=d24-ddp-1gpu-0919-h100 \
#     --run=d24_ddp_1gpu_h100 
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft \
#     -- --run=d24_ddp_1gpu_sft_h100   \
#     --model-tag=d24-ddp-1gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl \
#     -- --run=d24_ddp_1gpu_rl   \
#     --model-tag=d24-ddp-1gpu-0919-h100


# ##node =2
# CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
#     --depth=24 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=0 \
#     --model-tag=d24-ddp-2gpu-0919-h100 \
#     --run=d24_ddp_2gpu_h100
# CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
#     -- --run=d24_ddp_2gpu_sft_h100   \
#     --model-tag=d24-ddp-2gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl \
#     -- --run=d24_ddp_2gpu_rl   \
#     --model-tag=d24-ddp-2gpu-0919-h100 


# ##node =4
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
#     --depth=24 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=0 \
#     --model-tag=d24-ddp-4gpu-0919-h100 \
#     --run=d24_ddp_4gpu_h100
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
#     -- --run=d24_ddp_4gpu_sft_h100   \
#     --model-tag=d24-ddp-4gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
#     -- --run=d24_ddp_4gpu_rl   \
#     --model-tag=d24-ddp-4gpu-0919-h100 


# ##node =8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8  -m scripts.base_train -- \
#     --depth=24 \
#     --max-seq-len=2048 \
#     --device-batch-size=4 \
#     --total-batch-size=524288 \
#     --num-iterations=21400 \
#     --target-param-data-ratio=-1 \
#     --use-diloco=0 \
#     --model-tag=d24-ddp-8gpu-0919-h100 \
#     --run=d24_ddp_8gpu_h100
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
#     -- --run=d24_ddp_8gpu_sft_h100   \
#     --model-tag=d24-ddp-8gpu-0919-h100 \
#     --load-optimizer=0
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
#     -- --run=d24_ddp_8gpu_rl   \
#     --model-tag=d24-ddp-8gpu-0919-h100 




############################################################ diloco ##############################################
##node =1 6000pro 0919
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-1gpu-0919-h100 \
    --run=d24_diloco_1gpu_h100 
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft \
    -- --run=d24_diloco_1gpu_sft_h100   \
    --model-tag=d24-diloco-1gpu-0919-h100 \
    --device-batch-size=16 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl \
    -- --run=d24_diloco_1gpu_rl   \
    --device-batch-size=16 \
    --model-tag=d24-diloco-1gpu-0919-h100


##node =2 6000pro 0919
CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-2gpu-0919-h100 \
    --run=d24_diloco_2gpu_h100
CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
    -- --run=d24_diloco_2gpu_sft_h100   \
    --model-tag=d24-diloco-2gpu-0919-h100 \
    --device-batch-size=16 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl \
    -- --run=d24_diloco_2gpu_rl   \
    --device-batch-size=16 \
    --model-tag=d24-diloco-2gpu-0919-h100 


##node =4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  -m scripts.base_train -- \
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
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d24_diloco_4gpu_sft_h100   \
    --model-tag=d24-diloco-4gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl \
    -- --run=d24_diloco_4gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-4gpu-0919-h100 


##node =8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8  -m scripts.base_train -- \
    --depth=24 \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --num-iterations=21400 \
    --target-param-data-ratio=-1 \
    --use-diloco=1 \
    --diloco-H=100 \
    --model-tag=d24-diloco-8gpu-0919-h100 \
    --run=d24_diloco_8gpu_h100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    -- --run=d24_diloco_8gpu_sft_h100   \
    --model-tag=d24-diloco-8gpu-0919-h100 \
    --device-batch-size=4 \
    --load-optimizer=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
    -- --run=d24_diloco_8gpu_rl   \
    --device-batch-size=4 \
    --model-tag=d24-diloco-8gpu-0919-h100 

############################################################ ours ##############################################
