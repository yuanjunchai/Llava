#!/bin/bash

# Set CUDA_VISIBLE_DEVICES to use only the first GPU (GPU 0)
CUDA_VISIBLE_DEVICES=0

# Model checkpoint
CKPT="llava-v1.5-13b"

# Dataset split
SPLIT="llava_vqav2_mscoco_test-dev2015"

# Remove the loop and directly execute the evaluation script
python -m llava.eval.model_vqa_loader \
    --model-path /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/stage2/Speedup_LoRA-ft-llava-next-CoCa-ViT-L-14-laion2b-s13b-b90k-vicuna-13b-v1.5-stage2-2025-03-27_09-27-56/checkpoint-17000 \
    --model-base /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/model_openclip/vicuna-13b-v1.5 \
    --question-file /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/$SPLIT.jsonl \
    --image-folder /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/test2015 \
    --answers-file /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_answer.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

# # The merging and conversion steps are still necessary, but adapted for the single file.
# output_file=/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_merge.jsonl

# # Rename the output file to the merge file
# mv /mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-eval/vqav2/answers/$SPLIT/$CKPT/single_gpu_answer.jsonl "$output_file"

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT