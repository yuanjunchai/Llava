#!/bin/bash
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_BLOCKING_WAIT=600
# export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_DEBUG=INFO

### specify GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0" 

export PYTHONPATH=$PYTHONPATH:/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
export PATH=$PATH:/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

### LLM model
# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION="vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${LLM_VERSION}"
### Vision model
# VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
# VISION_MODEL_VERSION="CLIP-ViT-L-14-laion2B-s32B-b82K"
VISION_MODEL_VERSION="CoCa-ViT-L-14-laion2B-s13B-b90k"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${VISION_MODEL_VERSION}"
IMAGE_FOLDER="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Pretrain/images"


############### Pretrain ################

NUM_GPUS=4
NNODES=1
RANK=0
ADDR="localhost"
PORT=29500
PROMPT_VERSION=plain

BASE_RUN_NAME="hyak_llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# deepspeed --master_port 30000 \
# ACCELERATE_CPU_AFFINITY=1 torchrun llava/train/train_mem.py \

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/projectors/${BASE_RUN_NAME}_${timestamp} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn