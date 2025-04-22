import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from .attention import inflate_attention_layer
from .normalization import replace_group_norms
from .positional_encoding import CubemapPositionalEncoding

class CubeDiffModel(nn.Module):
    """
    CubeDiff model architecture.
    """
    def __init__(self, pretrained_model_name="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        
        # Load base UNet from pretrained model
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
        )
        
        # Modify architecture for cubemap processing
        self.inflate_attention_layers()
        self.synchronize_normalization_layers()
        
        # Add positional encoding
        self.positional_encoding = CubemapPositionalEncoding(
            embedding_dim=4,
            max_resolution=64  # Latent space resolution
        )
        
        # Increase input channels to account for positional encoding
        in_channels = self.base_unet.conv_in.in_channels
        self.modified_conv_in = nn.Conv2d(
            in_channels + 2,  # +2 for UV coordinates
            self.base_unet.conv_in.out_channels,
            kernel_size=self.base_unet.conv_in.kernel_size,
            stride=self.base_unet.conv_in.stride,
            padding=self.base_unet.conv_in.padding,
        )
        
        # Initialize with pretrained weights
        with torch.no_grad():
            self.modified_conv_in.weight[:, :in_channels, :, :].copy_(
                self.base_unet.conv_in.weight
            )
            self.modified_conv_in.bias.copy_(self.base_unet.conv_in.bias)
        
        # Permanently replace the conv_in layer with our modified version
        self.base_unet.conv_in = self.modified_conv_in
    
    def inflate_attention_layers(self):
        """
        Replace all attention layers with inflated versions.
        """
        # Process all attention layers in the model
        for name, module in self.base_unet.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                parent_module = self.base_unet
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                # Replace attention layer
                child_name = name.split('.')[-1]
                setattr(parent_module, child_name, inflate_attention_layer(module))
    
    def synchronize_normalization_layers(self):
        """
        Replace all GroupNorm layers with synchronized versions.
        """
        replace_group_norms(self.base_unet, in_place=True)
    
    def forward(self, latents, timesteps, encoder_hidden_states):
        """
        Forward pass for CubeDiff model.
        
        Args:
            latents: Latent vectors of shape [batch, num_faces, channels, height, width]
            timesteps: Diffusion timesteps
            encoder_hidden_states: Text encoder hidden states
            
        Returns:
            Denoised latent vectors
        """
        batch_size, num_faces, C, H, W = latents.shape
        
        # Add positional encoding
        latents_with_pos = self.positional_encoding(latents)
        
        # Process each face separately but with cross-face attention
        outputs = []
        
        for face_idx in range(num_faces):
            face_latent = latents_with_pos[:, face_idx]
            
            # Forward through UNet with the permanently modified conv_in
            noise_pred = self.base_unet(
                face_latent,
                timesteps,
                encoder_hidden_states,
            ).sample
            
            outputs.append(noise_pred)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch, num_faces, C, H, W]
        
        return stacked_outputs