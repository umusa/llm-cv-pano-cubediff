import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

from .attention import inflate_attention_layer
from .normalization import replace_group_norms
from .positional_encoding import CubemapPositionalEncoding


class CubeDiffModel(nn.Module):
    """
    CubeDiff model architecture with internal CLIP text encoder and cubemap UNet.
    """
    def __init__(self, pretrained_model_name="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        # Load full SD pipeline to extract both text encoder and U-Net
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32
        )
        # ICE: CLIP text encoder (frozen)
        self.text_encoder = pipe.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # Base U-Net (we will inflate and modify it)
        self.base_unet = pipe.unet
        # Remove unused branches
        del pipe

        # Replace attention with inflated versions
        for name, module in list(self.base_unet.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                parent = self.base_unet
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], inflate_attention_layer(module))

        # Sync group norms
        replace_group_norms(self.base_unet, in_place=True)

        # Positional encoding for cubemap faces
        self.positional_encoding = CubemapPositionalEncoding(
            embedding_dim=4,
            max_resolution=64
        )

        # Modify conv_in to accept additional UV channels
        in_ch = self.base_unet.conv_in.in_channels
        new_conv = nn.Conv2d(
            in_ch + 2,
            self.base_unet.conv_in.out_channels,
            kernel_size=self.base_unet.conv_in.kernel_size,
            stride=self.base_unet.conv_in.stride,
            padding=self.base_unet.conv_in.padding
        )
        with torch.no_grad():
            new_conv.weight[:, :in_ch].copy_(self.base_unet.conv_in.weight)
            new_conv.bias.copy_(self.base_unet.conv_in.bias)
        self.base_unet.conv_in = new_conv

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the U-Net"""
        self.base_unet.enable_gradient_checkpointing()

    
    def forward(
        self,
        latents,                   # [B, F, C, H, W]
        timesteps,                 # [B] or scalar
        encoder_hidden_states=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Forward pass for CubeDiff:
        - If `encoder_hidden_states` is None, uses `input_ids`+`attention_mask` to compute them.
        - Adds UV positional encoding,
            flattens faces so U-Net sees [B*F, C+2, H, W],
            repeats timesteps per face,
            then reshapes back to [B, F, C, H, W].
        """
        # 1) if you only got raw tokens, build embeddings now
        if encoder_hidden_states is None:
            if input_ids is None or attention_mask is None:
                raise ValueError("Must provide either encoder_hidden_states or both input_ids+attention_mask")
            with torch.no_grad():
                clip_out = self.text_encoder(input_ids, attention_mask=attention_mask)
                encoder_hidden_states = clip_out.last_hidden_state  # [B, L, D]

        B, F, C, H, W = latents.shape

        # 2) positional encode and flatten latents to [B*F, C+2, H, W]
        # lat = self.positional_encoding(latents)
        lat = self.positional_encoding(latents, resolution=H)
        lat = lat.view(B * F, C + 2, H, W)

        # 3) repeat timesteps per face → [B*F]
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.unsqueeze(1).repeat(1, F).view(-1)

        # 4) tile the CLIP embeddings to match that same B*F batch
        #    from [B, L, D] → [B, 1, L, D] → [B, F, L, D] → [B*F, L, D]
        hs = encoder_hidden_states
        B2, L, D = hs.shape
        if B2 != B:
            raise ValueError(f"batch mismatch: embeddings batch {B2} vs latents batch {B}")
        hs = hs.unsqueeze(1).expand(B, F, L, D).reshape(B * F, L, D)

        # 5) run your modified UNet with the correctly‐sized embeddings
        out = self.base_unet(lat, timesteps, encoder_hidden_states=hs).sample  # [B*F, C, H, W]

        # 6) un‐flatten the face dimension
        return out.view(B, F, *out.shape[1:])

