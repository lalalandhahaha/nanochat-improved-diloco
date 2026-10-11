#!/bin/bash

# d12 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --device-batch-size=8  --diloco-outer-lr=0.3  --diloco-outer-momentum=0.9\  
    --use-diloco=1 \
    --run=d12_diloco_4gpu_outerlr_03_outermo_09 \
    --model-tag=d12_diloco_4gpu_outerlr_03_outermo_09
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_outerlr_03_outermo_09 \
    --device-batch-size=8 \
    --model-tag=d12_diloco_4gpu_outerlr_03_outermo_09 \


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --device-batch-size=8  --diloco-outer-lr=0.7  --diloco-outer-momentum=0.5\  
    --use-diloco=1 \
    --run=d12_diloco_4gpu_outerlr_07_outermo_05 \
    --model-tag=d12_diloco_4gpu_outerlr_07_outermo_05d
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_outerlr_07_outermo_05 \
    --device-batch-size=8 \
    --model-tag=d12_diloco_4gpu_outerlr_07_outermo_05 \


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --device-batch-size=8  --diloco-outer-lr=0.3  --diloco-outer-momedntum=0.5\  
    --use-diloco=1 \
    --run=d12_diloco_4gpu_outerlr_03_outermo_05 \
    --model-tag=d12_diloco_4gpu_outerlr_03_outermo_05
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_outerlr_03_outermo_05 \
    --device-batch-size=8 \
    --model-tag=d12_diloco_4gpu_outerlr_03_outermo_05 \




# d12 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
CUDA_VISIBLE_DEVICES=0,1,4,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \  
    --use-diloco=1 \
    --lambda-reg=1e-3 \
    --run=d12_diloco_4gpu_reghead_1e-3 \
    --model-tag=d12_diloco_4gpu_reghead_1e-3_1003
# evaluate the model: CORE metric, BPB on train/val, and draw samples
CUDA_VISIBLE_DEVICES=0,1,4,3 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_reghead_1e-3_sft   \
    --device-batch-size=8 \
    --model-tag=d12_diloco_4gpu_reghead_1e-3_1003 \
    --load-optimizer=0


CUDA_VISIBLE_DEVICES=0,1,4,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \  
    --use-diloco=1 \
    --lambda-reg=1e-2 \
    --run=d12_diloco_4gpu_reghead_1e-2 \
    --model-tag=d12_diloco_4gpu_reghead_1e-2_1003
# evaluate the model: CORE metric, BPB on train/val, and draw samples
CUDA_VISIBLE_DEVICES=0,1,4,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_reghead_1e-2_sft   \
    --device-batch-size=8 \
    --model-tag=d12_diloco_4gpu_reghead_1e-2_1003 \
    --load-optimizer=0


CUDA_VISIBLE_DEVICES=0,1,4,6 torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    -- --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --num-iterations=21400 \  
    --use-diloco=1 \
    --lambda-reg=1e-1 \
    --run=d12_diloco_4gpu_reghead_1e-1 \
    --model-tag=d12_diloco_4gpu_reghead_1e-1_1003
# evaluate the model: CORE metric, BPB on train/val, and draw samples
CUDA_VISIBLE_DEVICES=0,1,4,6 torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    -- --run=d12_diloco_4gpu_reghead_1e-1_sft   \
    --model-tag=d12_diloco_4gpu_reghead_1e-1_1003 \
    --load-optimizer=0



