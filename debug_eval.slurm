#!/bin/bash
# filepath: /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/scripts/eval/debug_vqav2.slurm

#SBATCH --job-name=debug_vqav2
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 使用单个GPU
#SBATCH --cpus-per-task=4   # 每个进程使用12个CPU核心
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yjchai@uw.edu
#SBATCH --chdir=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT
#SBATCH --output=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --error=/mmfs1/gscratch/cse/yjchai/slurm/slurm-%j.out
#SBATCH --time=2:00:00
#SBATCH --export=ALL

# --- 加载环境 ---
source /mmfs1/home/yjchai/.bashrc
conda activate llava
module load cuda/11.8.0

# Set CUDA_VISIBLE_DEVICES to use only the first GPU (GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Model checkpoint
CKPT="llava-v1.5-13b"

# Dataset split
SPLIT="llava_vqav2_mscoco_test-dev2015"

# Remove the loop and directly execute the evaluation script
python -m llava.eval.model_vqa_loader \
    --model-path /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/stage2/Speedup_LoRA-ft-llava-next-CoCa-ViT-L-14-laion2b-s13b-b90k-vicuna-13b-v1.5-stage2-2025-03-27_09-27-56/checkpoint-17000 \
    --model-base /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/llavanext-CoCa-ViT-L-14-laion2B-s13B-b90k-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-25_22-57-51 \
    --question-file /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/$SPLIT.jsonl \
    --image-folder /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/test2015 \
    --answers-file /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_answer.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

# The merging and conversion steps are still necessary, but adapted for the single file.
output_file=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_merge.jsonl

# Rename the output file to the merge file
# mv /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_answer.jsonl "$output_file"

# python scripts/archived/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT