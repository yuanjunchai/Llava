import open_clip
import torch

model_name = "coca_ViT-L-14"
pretrained = "laion2b_s13b_b90k"
pretrained_model = "/mmfs1/gscratch/cse/yjchai/huggingface/hub/models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k/snapshots/74207cb7fde8eafc9864451ebd332fa8e75b150f/open_clip_pytorch_model.bin"
# vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, precision="fp32", device="cpu")
vision_tower, image_processor = open_clip.create_model_from_pretrained(model_name=model_name, pretrained=pretrained_model, precision="fp32", device="cpu")
print(vision_tower)
print(image_processor)