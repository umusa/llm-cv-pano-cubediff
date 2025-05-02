import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from cl.model.positional_encoding import CubemapPositionalEncoding
from cl.model.attention        import inflate_attention_layer
from cl.model.normalization    import replace_group_norms

class CubeDiffModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_faces: int = 6,
        uv_dim: int   = 9
    ):
        super().__init__()

        # — 1) Load only the U-Net
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float32
        )

        # — 2) Inflate its attention to cross‐face
        for name, module in list(self.base_unet.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                parent = self.base_unet
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], inflate_attention_layer(module))

        # — 3) Sync GroupNorm
        replace_group_norms(self.base_unet, in_place=True)

        # — 4) Positional encoding (adds UV channels)
        self.positional_encoding = CubemapPositionalEncoding(
            num_faces=num_faces,
            embedding_dim=uv_dim
        )

        # — 5) Patch conv_in to accept the UV channels
        in_ch   = self.base_unet.conv_in.in_channels
        out_ch  = self.base_unet.conv_in.out_channels
        kernel, stride, pad = (
            self.base_unet.conv_in.kernel_size,
            self.base_unet.conv_in.stride,
            self.base_unet.conv_in.padding,
        )
        new_in = nn.Conv2d(in_ch + uv_dim, out_ch, kernel, stride, pad)
        with torch.no_grad():
            new_in.weight[:, :in_ch].copy_(self.base_unet.conv_in.weight)
            new_in.bias.copy_(self.base_unet.conv_in.bias)
        self.base_unet.conv_in = new_in

        # — 6) Gradient checkpointing on the U-Net
        self.base_unet.enable_gradient_checkpointing()

        # — 7) Circular padding everywhere
        for m in self.base_unet.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = "circular"

    def forward(
        self,
        latents: torch.Tensor,               # [B,6,C,H,W]
        timesteps: torch.Tensor,             # [B] or [B*6]
        encoder_hidden_states: torch.Tensor,  # [B,L,D] or [B*6,L,D]
        **kwargs   # ignored                  # Add this to catch extra parameters
    ) -> torch.Tensor:
        # print(f"architecture.py - cubeDiffModel - forward() - kwargs is {kwargs}\n")
        B, F, C, H, W = latents.shape
        E = self.positional_encoding.embedding_dim

        # 1) Add UV enc & flatten → [B*6, C+E, H, W]
        lat = self.positional_encoding(latents)
        lat = lat.view(B * F, C + E, H, W)

        # 2) Tile timesteps if needed → [B*6]
        if timesteps.dim()==1 and timesteps.numel()==B:
            timesteps = timesteps.unsqueeze(1).repeat(1, F).view(-1)

        # 3) Tile text embeddings if needed → [B*6, L, D]
        if encoder_hidden_states.dim()==3 and encoder_hidden_states.size(0)==B:
            B2, L, D = encoder_hidden_states.shape
            encoder_hidden_states = (
                encoder_hidden_states
                  .unsqueeze(1)     # [B,1,L,D]
                  .expand(B, F, L, D)
                  .reshape(B * F, L, D)
            )
        elif encoder_hidden_states.size(0) != B * F:
            raise ValueError(f"Wrong embedding batch: got {encoder_hidden_states.size(0)}, expected {B*F}")

        # 4) Call the U-Net with exactly (sample, timestep, encoder_hidden_states)
        # print(f"architecture.py - cubeDiffModel - forward() - before unet_out = self.base_unet, lat is {lat}, encoder_hidden_states is {encoder_hidden_states}, timesteps is {timesteps} \n")
        unet_out = self.base_unet(
            sample=lat,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states
        )
        # print(f"architecture.py - cubeDiffModel - forward() - after unet_out = self.base_unet\n")
        out = unet_out.sample  # [B*6, C, H, W]

        # 5) reshape back → [B,6,C,H,W]
        return out.view(B, F, *out.shape[1:])

