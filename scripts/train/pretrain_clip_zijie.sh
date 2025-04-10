# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

# export CUDA_VISIBLE_DEVICES="7" 
export CUDA_VISIBLE_DEVICES="0,1" 

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION="model/vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/root/work/nlp/dizj/research/LLaVA-main/playground/${LLM_VERSION}"
# VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
# VISION_MODEL_VERSION="model/CLIP-ViT-L-14-laion2B-s32B-b82K"
# VISION_MODEL_VERSION="model/CoCa-ViT-L-14-laion2B-s13B-b90k"
VISION_MODEL_VERSION="CLIP-ViT-L-14-laion2B-s32B-b82K"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PATH="/root/work/nlp/dizj/research/LLaVA-main/playground/model/${VISION_MODEL_VERSION}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
#deepspeed --master_port 30000 llava/train/train_mem.py \

ACCELERATE_CPU_AFFINITY=1 torchrun --master_addr="127.0.0.1" --master_port=30000 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /root/work/nlp/dizj/research/LLaVA-main/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /root/work/nlp/dizj/research/LLaVA-main/playground/data/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_PATH} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /root/work/nlp/dizj/research/LLaVA-NeXT/checkpoints/projectors/${BASE_RUN_NAME}_${timestamp} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn