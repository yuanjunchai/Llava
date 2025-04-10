#!/bin/bash

#SBATCH --job-name=twoGPULoRACoCa
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2    # 每个节点启动4个进程（与GPU数匹配）
#SBATCH --cpus-per-task=4   # 每个进程使用1个CPU核心
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yjchai@uw.edu
#SBATCH --chdir=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
#SBATCH --output=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --error=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --time=120:00:00
#SBATCH --export=ALL             # 传递所有环境变量

# --- 加载环境 ---
source /mmfs1/home/yjchai/.bashrc
module load cuda/11.8.0

# --- 设置GPU可见性 ---
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0, 1

# --- show up the GPU information ---
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status

conda activate llava

# --- 设置Python路径 ---
export PYTHONPATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT:$PYTHONPATH"

# --- 激活Conda环境 ---

# --- 环境检查 ---
echo "检查CUDA版本："
nvcc --version

echo "检查Conda环境："
conda info --envs
echo "当前环境：$CONDA_DEFAULT_ENV"

# --- 依赖检查 ---
conda list clip  # 检查clip包是否安装
which python
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用性: {torch.cuda.is_available()}')"

export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# ==== 分布式训练配置 ====
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_JOBID % 50000) ))  # 扩大端口范围避免冲突
export WORLD_SIZE=$SLURM_NTASKS

# Timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

LLM_VERSION="vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${LLM_VERSION}"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
VISION_MODEL_VERSION="CoCa-ViT-L-14-laion2b-s13b-b90k"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${VISION_MODEL_VERSION}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-ddp-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION=v1
RUN_NAME="twoGPU_Speedup_LoRA-ft-llava-next-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-stage2-${timestamp}" 
PREV_STAGE_CHECKPOINT="/root/work/nlp/dizj/research/LLaVA-NeXT/checkpoints/projectors/llavanext-CoCa-ViT-L-14-laion2B-s13B-b90k-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-25_22-57-51" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# ==== 拓扑感知启动命令 ====
#srun --cpu-bind=cores --gpu-bind=closest \
#bash /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/scripts/train/finetune_coca.sh
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} llava/train/train_mem.py \
     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_PATH} \
    --version $PROMPT_VERSION \
    --pretrain_mm_mlp_adapter /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/llavanext-CoCa-ViT-L-14-laion2B-s13B-b90k-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-25_22-57-51/mm_projector.bin \
    --data_path /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune \
    --vision_tower ${VISION_MODEL_VERSION} \
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
    --per_device_eval_batch_size 4 \
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