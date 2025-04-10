import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from .vit_utils import apply_masks
from einops import rearrange, repeat
import json

class ViT3D(nn.Module):
    def __init__(
        self, 
        img_size=224,
        patch_size=16,
        model_name="vit-base",
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        norm_layer=nn.LayerNorm,
        pretrained=False,
        pool=None,
    ):
        super(ViT3D, self).__init__()
        vit_model_name=f"google/{model_name}-patch{patch_size}-{img_size}-in21k"
        config = ViTConfig.from_pretrained(vit_model_name, attn_implementation="eager")
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.patch_embed_3d = nn.Conv3d(
            in_channels=in_chans, 
            out_channels=config.hidden_size,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        
        if pretrained:
            # Load the 2D ViT model configuration and weights
            self.vit2d = ViTModel.from_pretrained(vit_model_name, attn_implementation="eager")
            # Load the weights from the 2D convolution
            self._load_2d_weights_into_3d_conv()
            print("initializing from pretrained model")
        else:
            self.vit2d = ViTModel(config)
        
        self._initialize_3d_positional_embeddings(num_frames=num_frames//tubelet_size)
        
        # Retain the rest of the ViT layers and settings
        self.encoder = self.vit2d.encoder
        
        if norm_layer is not None: self.norm = norm_layer(self.embed_dim)
        else: self.norm = nn.Identity()
        
        if pool is not None: self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pool = pool
        
    def load_image_model(self, state_dict):
        msg = self.vit2d.load_state_dict(state_dict, strict=False)
        self._load_2d_weights_into_3d_conv()
        self._initialize_3d_positional_embeddings(num_frames=self.num_frames//self.tubelet_size)
        self.encoder = self.vit2d.encoder
        return msg
        

    def _load_2d_weights_into_3d_conv(self):
        # Extract the 2D convolution weights
        weight_2d = self.vit2d.embeddings.patch_embeddings.projection.weight  # Shape: (out_channels, in_channels, h, w)
        
        # Expand along the temporal dimension and average the weights
        tubelet_size = self.patch_embed_3d.kernel_size[0]
        print("weight_2d shape:", weight_2d.shape)
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / tubelet_size  # Shape: (out_channels, in_channels, d, h, w)
        
        # Set the 3D convolution weights and bias
        self.patch_embed_3d.weight = nn.Parameter(weight_3d)
        self.patch_embed_3d.bias = self.vit2d.embeddings.patch_embeddings.projection.bias

    def _initialize_3d_positional_embeddings(self, num_frames):
        # Repeat 2D positional embeddings along the temporal dimension
        pos_embed_2d = self.vit2d.embeddings.position_embeddings[:, 1:]  # Exclude [CLS] token
        num_patches = pos_embed_2d.shape[1]
        hidden_dim = pos_embed_2d.shape[2]
        
        # Reshape to (1, height, width, d) and repeat along the temporal dimension
        spatial_size = int(num_patches ** 0.5)  # Assuming square patch grid
        pos_embed_2d = pos_embed_2d.view(1, spatial_size, spatial_size, hidden_dim)
        pos_embed_3d = pos_embed_2d.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # Shape: (1, t, h, w, d)
        
        # Flatten to match 3D input token shape
        pos_embed_3d = pos_embed_3d.view(1, num_frames * num_patches, hidden_dim)

        # Concatenate CLS token position embedding and assign
        self.positional_embeddings_3d = nn.Parameter(
            torch.cat([self.vit2d.embeddings.position_embeddings[:, :1], pos_embed_3d], dim=1)
        )

    def forward(self, x, masks=None, output_attention=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        
        # Expecting x to have shape (batch_size, channels, depth, height, width)
        x = self.patch_embed_3d(x)  # Apply 3D patch embedding
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose to match ViT input shape
        
        x = x + self.positional_embeddings_3d[:, : x.size(1), :]

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
        
        
        if self.pool is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through the rest of the ViT model
        outputs = self.encoder(x,  output_attentions=output_attention)
        x = outputs[0]
        attention_scores = outputs[1] if output_attention else None
        
        x = self.norm(x)
        
        if output_attention: 
            return x, attention_scores
        else: 
            return x
        
    
if __name__ == '__main__':
    # from calflops import calculate_flops
    # # Example usage
    
    # data = {}
    # flops = 0
    # for model_name in [ 'vit-base', 'vit-large', 'vit-huge', ]:
    #     flop_list = []
    #     for frame_number in [16,32,64]:
    #         if model_name == 'vit-huge' and frame_number == 64: flops = flops * 2
    #         else:
    #             model = ViT3D(model_name=model_name, patch_size=16 if model_name!='vit-huge' else 14, num_frames=frame_number).cuda()
                
    #             # Calculate FLOPs, MACs, and Params
    #             input_shape = (1, 3, frame_number, 224, 224)  # Shape of the input tensor
    #             flops, macs, params = calculate_flops(
    #                 model=model,
    #                 input_shape=input_shape,
    #                 output_as_string=False,
    #                 output_precision=4
    #             )

    #             # Print results
    #             print("ViT3D FLOPs: %s   MACs: %s   Params: %s \n" % (flops, macs, params))
    #         flop_list.append(flops / 1e9)
    #     data[model_name] = flop_list
    
    # with open('/weka/chenhaoz/home/videotok/parse_flops/model_stats_vit3d.json', 'w') as f:
    #     json.dump(data, f, indent=4)
        
        

    
    data = {}
    vrams = 0
    for model_name in [ 'vit-base']:
        vram_list = []
        for frame_number in [16,32,64]:
            if model_name == 'vit-huge' and frame_number == 64:
                vrams = vrams * 4
            else:
                model = ViT3D(model_name=model_name, patch_size=16 if model_name!='vit-huge' else 14, num_frames=frame_number).cuda()
                
                # Calculate FLOPs, MACs, and Params
                input_shape = (1, 3, frame_number, 224, 224)  # Shape of the input tensor
                input = torch.zeros(input_shape).cuda()
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                model(input)
                # Get peak memory usage
                peak_allocated = torch.cuda.max_memory_allocated()

                # Print results
                print(f"Peak memory allocated: {peak_allocated / (1024 ** 2):.2f} MB")
            
            vram_list.append(peak_allocated / (1024 ** 2))
        data[model_name] = vram_list
    
    with open('/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT/model_vram_vit3d.json', 'w') as f:
        json.dump(data, f, indent=4)