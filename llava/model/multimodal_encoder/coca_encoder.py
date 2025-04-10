import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
from llava.utils import rank0_print
import os
import sys

try:
    import open_clip
    import torchvision
    # from open_clip.transformer import _expand_token ###ImportError: cannot import name '_expand_token' from 'open_clip.transformer'
except ImportError:
    rank0_print("OpenCLIP not installed")
    # open_clip = None

# Define hidden sizes for COCA models
HIDDEN_SIZE_DICT = {
    "coca_ViT-L-14": 768,  # Update with actual hidden size
    "coca_ViT-B-32": 512,  # Update with actual hidden size
}


class COCAVisionTower(nn.Module):
    def __init__(self, vision_tower_ori_name, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower_ori_name.split(":")[-1]
        self.model_name = "coca_ViT-L-14"
        self.pretrained = "laion2b_s13b_b90k"
        rank0_print(f"Loading OpenCLIP model: {self.model_name}, {self.pretrained}")
        # self.model_name = vision_tower.replace("open_clip_hub:", "")
        # self.pretrained = args.vision_tower_pretrained
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Model init Loading vision tower: {self.vision_tower_name}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

    def load_model(self, device_map="auto"):
        rank0_print(f"Loading OpenCLIP model: {self.model_name}")
        rank0_print(f"Pretrained: {self.pretrained}")
        import open_clip
        rank0_print("Load open_clip package successfully.")
        rank0_print("关键环境变量:")
        rank0_print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
        rank0_print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
        rank0_print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        rank0_print(sys.path)
        rank0_print("open_clip 版本:", open_clip.version.__version__)

        # Load COCA model using open_clip
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_tower, _, image_processor = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="laion2b_s13b_b90k",
            precision="fp32",
            device=device
        )
        # print("Vision tower state dict keys:")
        # for key, value in vision_tower.state_dict().items():
        #     print(key, value.shape)


        # from open_clip import get_model_config

        # try:
        #     vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cpu")
        #     vision_tower = vision_tower.to("cuda")
        # except Exception as e:
        #     rank0_print(f"标准加载失败，错误: {e}")
        
        # try:
        #     # 尝试手动构建模型结构，但不加载权重
        #     model_cfg = open_clip.get_model_config('coca_ViT-L-14')
            
        #     # 检查模型配置
        #     rank0_print(f"Model config: {model_cfg}")
            
        #     # 创建一个没有预训练权重的模型
        #     model, _, transforms = open_clip.create_model_and_transforms(
        #         model_name='coca_ViT-L-14',
        #         pretrained=None,  # 不加载预训练权重
        #         precision="fp32",
        #         device="cpu"
        #     )
            
        #     # 然后手动加载权重
        #     checkpoint_path = '/mmfs1/gscratch/cse/yjchai/huggingface/hub/models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k/snapshots/74207cb7fde8eafc9864451ebd332fa8e75b150f/open_clip_pytorch_model.bin'
        #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
        #     # 打印checkpoint的键，了解其结构
        #     keys = list(checkpoint.keys())
        #     rank0_print(f"Checkpoint keys (first 10): {keys[:10]}")
            
        #     # 检查模型和checkpoint的状态字典是否匹配
        #     model_state = model.state_dict()
        #     model_keys = set(model_state.keys())
        #     checkpoint_keys = set(checkpoint.keys())
            
        #     # 查看差异
        #     missing_keys = model_keys - checkpoint_keys
        #     unexpected_keys = checkpoint_keys - model_keys
            
        #     rank0_print(f"Missing keys count: {len(missing_keys)}")
        #     rank0_print(f"Unexpected keys count: {len(unexpected_keys)}")
            
        #     if len(missing_keys) > 0:
        #         rank0_print(f"Sample missing keys: {list(missing_keys)[:5]}")
        #     if len(unexpected_keys) > 0:
        #         rank0_print(f"Sample unexpected keys: {list(unexpected_keys)[:5]}")
            
        #     # 尝试不严格加载权重
        #     incompatible = model.load_state_dict(checkpoint, strict=False)
        #     rank0_print(f"Incompatible keys: {incompatible}")
            
        #     # 移动到CUDA
        #     model = model.to("cuda")
        # except Exception as e2:
        #     rank0_print(f"手动加载也失败了，错误: {e2}")
        # rank0_print("open_clip 版本:", open_clip.__version__)
        #### This is the original code ####
        # vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cuda")
        
        
        # vision_tower, _, image_processor = open_clip.create_model_from_pretrained(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cpu")


        # model_cfg = get_model_config('coca_ViT-L-14')
        # vision_tower, _, image_processor = open_clip.create_model_from_pretrained(
        #     model_name='coca_ViT-L-14',
        #     pretrained='/mmfs1/gscratch/cse/yjchai/huggingface/hub/models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k/snapshots/74207cb7fde8eafc9864451ebd332fa8e75b150f/open_clip_pytorch_model.bin',
        #     precision="fp32",
        #     device="cpu",
        #     # text_cfg=model_cfg['text_cfg'],
        #     # vision_cfg=model_cfg['vision_cfg'],
        # )
        self.image_processor = image_processor
        rank0_print(f"Loaded vision tower: {vision_tower}")

        resize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Resize)][0]
        normalize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
        self.resize_transform_size = resize_transform.size  # 224 or 384
        self.patch_size = vision_tower.visual.conv1.kernel_size[0] if hasattr(vision_tower.visual, 'conv1') else 14 # 14 or 16

        # self.image_processor = CLIPImageProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14",
        #     crop_size=resize_transform.size,
        #     size={"shortest_edge": resize_transform.size},
        #     image_mean=list(normalize_transform.mean),
        #     image_std=list(normalize_transform.std),
        # )
        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower = vision_tower.visual
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs[self.select_layer] #
        if self.select_feature == "patch":
            # image_features = image_forward_outs[:, 1:]
            ### orginal code ###
            image_features = image_forward_outs
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "conv_flatten":
            image_features = image_features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward_visual(self, x, output_hidden_states=False):
        if hasattr(self.vision_tower, "trunk") and hasattr(self.vision_tower.trunk, "_intermediate_layers"):
            return self.vision_tower.trunk._intermediate_layers(x, abs(self.select_layer))
        else:

            def forward_coca_visual(self, x: torch.Tensor):
                features = []
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

                # class embeddings and positional embeddings
                x = torch.cat(
                    [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x],
                    dim=1,
                )
                # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)

                x = self.patch_dropout(x)
                x = self.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                for r in self.transformer.resblocks:
                    x = r(x, attn_mask=None)
                    features.append(x)
                return features

            return forward_coca_visual(self.vision_tower, x)

    def forward(self, images):
        # breakpoint()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.forward_visual(image.to(self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # image_forward_outs = self.forward_visual(images.to(self.dtype), output_hidden_states=True)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs[1]).to(images.dtype) #[1] 
            # (Pdb) image_forward_outs[0].shape
            # torch.Size([1, 768])
            # (Pdb) image_forward_outs[1].shape
            # torch.Size([1, 255, 768])
            rank0_print(f"image_features.shape: {image_features.shape}")
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.dtype
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.dtype
        raise NotImplementedError

    @property
    def device(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.device
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.device
        raise NotImplementedError

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        if self.model_name in HIDDEN_SIZE_DICT:
            return HIDDEN_SIZE_DICT[self.model_name]
        else:
            return int(768)

    @property
    def num_patches(self):
        image_size = self.resize_transform_size if isinstance(self.resize_transform_size, int) else self.resize_transform_size[0]
        _num_patches = (image_size // self.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.resize_transform_size

    @property
    def num_patches_per_side(self):
        return self.resize_transform_size // self.patch_size


if __name__ == "__main__":
    # Test the COCAVisionTower class
    args = type('', (), {})()  # Create a simple object to hold attributes
    args.mm_vision_select_layer = "layer4"
    args.mm_vision_select_feature = "patch"
    args.mm_vision_tower_pretrained = "laion2b_s13b_b90k"
    
    tower = COCAVisionTower("open_clip_hub:coca_ViT-L-14", args)
    print(tower)  # Print the model structure to verify it loaded correctly
    # Test with a dummy image
    from PIL import Image
    import numpy
    # width, height = 256, 256
    # image = Image.new('RGB', (width, height), (255, 255, 255))
    imarray = numpy.random.rand(100, 100, 3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    dummy_image = tower.image_processor(im).unsqueeze(0)
    breakpoint()
    # dummy_image = torch.randn(1, 3, tower.image_size, tower.image_size).to(tower.device)
    features = tower.forward(dummy_image)
    print("Feature shape:", features[0].shape)  # Print the shape of the extracted features
    # Ensure the feature shape matches expectations
    # assert features[0].shape == (1, tower.hidden_size, tower.num_patches), "Feature shape mismatch"
    # rank0_print("COCAVisionTower test passed!")  # Indicate the test passed