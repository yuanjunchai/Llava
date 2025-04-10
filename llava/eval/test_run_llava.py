from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


model_path = "/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/stage2/Speedup_LoRA-ft-llava-next-CoCa-ViT-L-14-laion2b-s13b-b90k-vicuna-13b-v1.5-stage2-2025-03-27_09-27-56/checkpoint-17000"
query = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"
model_base = "/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/checkpoints/llavanext-CoCa-ViT-L-14-laion2B-s13B-b90k-model_vicuna-13b-v1.5-mlp2x_gelu-pretrain_blip558k_plain_2025-02-25_22-57-51"
# "lmsys/vicuna-13b-v1.5"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path),
    "query": query,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)