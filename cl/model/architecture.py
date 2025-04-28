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
        latents,                   # [B, num_faces, C, H, W]
        timesteps,                 # [B] or scalar
        encoder_hidden_states=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Forward with either precomputed embeddings or raw tokens.
        """
        # If embeddings not provided, compute from tokens
        if encoder_hidden_states is None:
            if input_ids is None:
                raise ValueError("Provide either encoder_hidden_states or input_ids + attention_mask")
            with torch.no_grad():
                out = self.text_encoder(input_ids, attention_mask=attention_mask)
                encoder_hidden_states = out.last_hidden_state
        # Apply positional encoding to latents
        B, F, C, H, W = latents.shape
        lat = self.positional_encoding(latents)
        # Reshape for U-Net: merge face dimension
        lat = lat.view(B * F, C + 2, H, W)
        # Repeat timesteps per face
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.unsqueeze(1).repeat(1, F).view(-1)
        # U-Net forward
        pred = self.base_unet(
            lat,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        # Restore face batch dimension
        return pred.view(B, F, *pred.shape[1:])
