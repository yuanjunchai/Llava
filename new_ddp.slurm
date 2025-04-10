#!/bin/bash

#SBATCH --job-name=ddpcoca
#SBATCH --partition=gpu-a100
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2     # 每个节点启动4个进程（与GPU数匹配）
#SBATCH --cpus-per-task=2   # 每个进程使用1个CPU核心
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yjchai@uw.edu
#SBATCH --chdir=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
#SBATCH --output=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --error=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --export=ALL             # 传递所有环境变量

# --- 加载环境 ---
source /mmfs1/home/yjchai/.bashrc
module load cuda/11.8.0

# --- 设置GPU可见性 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

# --- show up the GPU information ---
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status

# --- 时间戳 ---
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

# --- 模型路径 ---
LLM_VERSION="vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${LLM_VERSION}"

VISION_MODEL_VERSION="CoCa-ViT-L-14-laion2b-s13b-b90k"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

RUN_NAME="ddp-stage2-llava-next-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${timestamp}" 
echo "MID_RUN_NAME: ${RUN_NAME}"

# --- 设置Python路径 ---
export PYTHONPATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT:$PYTHONPATH"

# --- 激活Conda环境 ---
conda activate llavacp

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


# # 基于PCIe PIX拓扑的优化
# export NCCL_ALGO=RING                 # 针对PCIe拓扑使用环形算法
# export NCCL_P2P_LEVEL=NVL             # 强制使用NVLink级别通信（即使处于inactive状态）
# export NCCL_P2P_DISABLE=0             # 启用点对点通信
# export NCCL_SHM_USE_CUDA_MEMCPY=1     # 使用CUDA内存拷贝优化
# export NCCL_IGNORE_CPU_AFFINITY=1     # 禁用自动CPU亲和性设置
# export NCCL_DEBUG=INFO

# # NUMA绑定优化
# # 显式禁用自动绑定以避免冲突
# export SLURM_CPU_BIND=none

# # 更新后的NUMA绑定策略
# export GOMP_CPU_AFFINITY="0-31:4"
export OMP_NUM_THREADS=4


# ==== 分布式训练配置 ====
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_JOBID % 50000) ))  # 扩大端口范围避免冲突
export WORLD_SIZE=$SLURM_NTASKS

# --- NCCL调优 ---
# export NCCL_ALGO=Ring               # 强制使用Ring算法
# export NCCL_DEBUG=INFO              # 启用NCCL调试信息
# export NCCL_P2P_DISABLE=1          # 启用P2P通信
# export NCCL_IB_DISABLE=1          # 启用InfiniBand（如果集群支持）
# export NCCL_SOCKET_IFNAME=eth0  # [!NEW] 确认物理网卡名称
# # export NCCL_NSOCKS_PERTHREAD=4  # [!NEW] 增加Socket数
# # export NCCL_SOCKET_NTHREADS=2   # [!NEW] 多线程通信
# export NCCL_BUFFSIZE=16777216   # [!NEW] 增大缓冲区

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 启用分布式调试信息
# # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # 限制OpenMP线程数
# export CUDA_LAUNCH_BLOCKING=1   # [!NEW] 同步调试

# --- 优化NCCL设置 ---
# 增加NCCL超时时间
# export NCCL_TIMEOUT=3600000  # 设置为1小时（毫秒）
# # 启用NCCL调试信息
# export NCCL_DEBUG=INFO
# # 设置NCCL通信方式，避免NVLink不可用时的问题
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=1
# # 设置socket通信超时
# export NCCL_SOCKET_NTHREADS=4
# export NCCL_NSOCKS_PERTHREAD=4
# # 为超大模型优化内存访问
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1          # 禁用InfiniBand（使用以太网时需要）
# export NCCL_NVLINK_ENABLE=1 
# # export NCCL_SOCKET_IFNAME=eth0    # 使用Mellanox适配器
# # export NCCL_TIMEOUT=1800          # 将超时时间延长至30分钟
# export NCCL_P2P_DISABLE=1
# # export NCCL_ALGO=Ring             # 强制使用Ring算法
# export NCCL_NSOCKS_PERTHREAD=4    # 增加网络socket数量
# export NCCL_PRIORITY=HIGH # Set high priority


# ==== 拓扑感知启动命令 ====
# srun --cpu-bind=cores --gpu-bind=closest \
deepspeed --num_gpus ${SLURM_NTASKS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_PATH} \
    --version v1 \
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
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
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