import json
import os

with open('/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune/llava_v1_5_mix665k.json') as f:
    samples = json.load(f)

missing_cnt = 0  # 记录图像路径不存在的个数
for sample in samples:
    if 'image' not in sample:
        continue

    img_path = os.path.join('/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/playground/llava_data/LLaVA-Finetune', sample['image'])
    if not os.path.exists(img_path):
        # # 处理 OCR-VQA 图像后缀和 llava_v1_5_mix665k.json 中的后缀不一致问题
        # img_path_wo_ext = os.path.splitext(img_path)[0]
        # for ext in ['.png', '.gif']:
        #     print(img_path_wo_ext)
        #     real_path = img_path_wo_ext + ext
        #     if os.path.exists(real_path):
        #         # 重命名
        #         os.replace(real_path, img_path)
        #         break
        
        print(img_path)
        missing_cnt += 1

print(missing_cnt)