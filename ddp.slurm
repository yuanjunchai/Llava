#!/bin/bash

#SBATCH --job-name=ddpcoca
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 每个节点启动4个进程（与GPU数匹配）
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yjchai@uw.edu
#SBATCH --qos=high
#SBATCH --chdir=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
#SBATCH --output=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --error=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --export=ALL             # 传递所有环境变量


source /mmfs1/home/yjchai/.bashrc

module load cuda/11.8.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 基于PCIe PIX拓扑的优化
# export NCCL_ALGO=RING                 # 针对PCIe拓扑使用环形算法
# export NCCL_P2P_LEVEL=NVL             # 强制使用NVLink级别通信（即使处于inactive状态）
# export NCCL_P2P_DISABLE=0             # 启用点对点通信
# export NCCL_DEBUG=INFO
# export NCCL_ALGO=TREE
# export NCCL_P2P_LEVEL=PIX
# export NCCL_IB_DISABLE=1
# export NCCL_BUFFSIZE=16777216
# export CUDA_LAUNCH_BLOCKING=1  # 调试阶段启用同步

# NUMA绑定优化
# export OMP_NUM_THREADS=1
# export GOMP_CPU_AFFINITY="0-7"

# export PYTHONPATH=$PYTHONPATH:/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
# export PATH=$PATH:/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT


# --- show up the GPU information ---
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status
# nvidia-smi -q | less

echo "===== SLURM环境变量 ====="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "PYTHONPATH: $PYTHONPATH"
conda list clip

# export PYTHONPATH="$PYTHONPATH:/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT"

# export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$(( RANDOM % 1000 + 20000 ))  # 自动选择随机端口
export MASTER_PORT=$(( 20000 + (SLURM_JOBID % 1000) + (RANDOM % 5000) ))  # Wider port range
echo "使用主端口：$MASTER_PORT"
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export LOCAL_RANK=$SLURM_LOCALID
export GLOBAL_RANK=$SLURM_PROCID

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}


# 在启动训练前添加端口检查
if nc -zv $MASTER_ADDR $MASTER_PORT 2>/dev/null; then
    echo "错误：端口 $MASTER_PORT 已被占用！"
    exit 1
else
    echo "端口 $MASTER_PORT 可用"
fi
# ==== 新增网络检查 ====
echo "===== 网络状态检查 ====="
ping -c 3 $MASTER_ADDR
nc -zv $MASTER_ADDR $MASTER_PORT || echo "端口可用"


# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export NCCL_TIMEOUT=60
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_ALGO=RING 

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_ALGO=Tree
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=3600

# export NCCL_SOCKET_IFNAME=eth0
# # 使用系统共享内存而不是GPU直接通信
# export NCCL_SHM_DISABLE=0
# # 禁用树形算法，使用简单的环形通信，可能更稳定
# export NCCL_ALGO=RING
# # 设置更大的缓冲区大小
# export NCCL_BUFFSIZE=16777216

# NCCL Settings (try these in various combinations)
export NCCL_DEBUG=INFO   # Get more verbose NCCL logging
# # unset NCCL_IB_DISABLE
# export NCCL_IB_DISABLE=1              # 必须禁用InfiniBand
# export NCCL_NET_GDR_READ=1
# export NCCL_NVLINK_ENABLE=1
# export NCCL_ALGO=TREE
# export NCCL_MIN_NCHANNELS=4           # 匹配物理通道数
# export NCCL_MAX_NCHANNELS=8
# # export OMP_NUM_THREADS=4
# # export NCCL_IB_DISABLE=1  #Try disabling Infiniband
# export NCCL_TIMEOUT=3600 # Increase timeout to 2 hours
# export NCCL_BUFFSIZE=16777216         # 增大缓冲区应对PCIe延迟
# export NCCL_SOCKET_IFNAME=eth0 # or eth0 (find correct interface with `ip addr`)
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo # Use loopback interface


LLM_VERSION="vicuna-13b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
LLM_PATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/${LLM_VERSION}"

VISION_MODEL_VERSION="CoCa-ViT-L-14-laion2b-s13b-b90k"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

RUN_NAME="ddp-stage2-llava-next-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${timestamp}" 
echo "MID_RUN_NAME: ${RUN_NAME}"

# 优先使用开发版代码
export PYTHONPATH="/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT:$PYTHONPATH"

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

# echo "Testing NCCL communication..."
# python -c "
# import torch
# assert torch.cuda.device_count() >= ${NUM_GPUS}, f'Expected at least ${NUM_GPUS} GPUs, got {torch.cuda.device_count()}'
# a = torch.ones(1).cuda()
# torch.distributed.init_process_group('nccl', init_method='env://')
# print('NCCL initialized successfully')
# " || echo "NCCL test failed, but continuing..."

# 添加调试输出
# export PYTHONVERBOSE=1  # 显示详细导入信息

# bash /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/scripts/train/finetune_coca.sh
#deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \

# torchrun  --standalone --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
# deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \

# --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
# --mm_vision_tower_lr=2e-6 \

echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo SLURM_LOCALID: $SLURM_LOCALID
# --- 启动训练 ---
# torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} --master_port=${MASTER_PORT} \

# srun --mpi=pmi2 --export=ALL bash -c "\
#     export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; \
# srun --mpi=pmi2 --export=ALL bash -c "\
# export CUDA_VISIBLE_DEVICES=0,1,2,3


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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
# "
