# export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
# export CUDA_VISIBLE_DEVICES="0, 1"
export CUDA_VISIBLE_DEVICES="0, 1"
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# # export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=10
# export NCCL_PROTO=simple

export PYTHONPATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT:$PYTHONPATH"
# export PYTHONPATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

LLM_VERSION="vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${LLM_VERSION}"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
VISION_MODEL_VERSION="CLIP-ViT-L-14-laion2B-s32B-b82K"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${VISION_MODEL_VERSION}"
IMAGE_FOLDER="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Pretrain/images"


NUM_GPUS=2  # 根据实际使用的GPU数量进行设置
NNODES=1  # 单机训练时设为1
RANK=0  # 单机训练时设为0
ADDR="localhost"  # 地址可以是localhost或具体的节点地址
PORT=42666

############### Pretrain ################

BASE_RUN_NAME="llavanext-ddp-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION=v1
RUN_NAME="try_LoRA-ft-llava-next-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-stage2-${timestamp}" 
PREV_STAGE_CHECKPOINT="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/llavanext-CLIP-ViT-L-14-laion2B-s32B-b82K-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-27_03-03-23" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# ACCELERATE_CPU_AFFINITY=1 torchrun \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} \

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} llava/train/train_mem.py \
deepspeed --num_gpus=${NUM_GPUS} --master_port=${PORT} llava/train/train_mem.py \
     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_PATH} \
    --version ${PROMPT_VERSION} \
    --pretrain_mm_mlp_adapter /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/llavanext-CLIP-ViT-L-14-laion2B-s32B-b82K-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-27_03-03-23/mm_projector.bin \
    --data_path /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune \
    --vision_tower ${VISION_MODEL_PATH} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/stage2/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --verbose_logging True \
    # --frames_upbound 32
    # --image_aspect_ratio anyres_max_9 \
    # --image_grid_pinpoints  "(1x1),...,(6x6)" \
    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    # --mm_vision_tower_lr=2e-6 \
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
