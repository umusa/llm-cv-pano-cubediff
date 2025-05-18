import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn

# if torch.xpu doesn’t exist, insert a stub so is_available() returns False
if not hasattr(torch, "xpu"):
    class _XPU:
        @staticmethod
        def is_available():
            return False
    torch.xpu = _XPU

from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention  # <— true SD attention class  
from cl.model.positional_encoding import CubemapPositionalEncoding
from cl.model.attention        import inflate_attention_layer
from cl.model.normalization    import replace_group_norms
from transformers import BitsAndBytesConfig


class CubeDiffModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_faces: int = 6,
        uv_dim: int   = 10,
        skip_weight_copy: bool = False # When U-Net is 4-bit, skip copying pretrained weights into the inflated layers
                                        # (so avoid all dequantize/view_as hacks).
    ):
        super().__init__()

        # — 1) Load only the U-Net in BF16
        self.skip_weight_copy = skip_weight_copy
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        # — 2) Move the U-Net onto CUDA and record device/dtype
        self.model_dtype = torch.bfloat16
        self.base_unet = self.base_unet.to("cuda", dtype=self.model_dtype)
        self.device    = next(self.base_unet.parameters()).device

        # — 3) Inflate Stable Diffusion’s (only the UNet’s self‐attention) layers → cross‐face Attention  
        for name, module in list(self.base_unet.named_modules()):
            # Only inflate Stable Diffusion’s ention _self_-attention, skip text cross-attention
            if isinstance(module, Attention) and module.cross_attention_dim is None:
                parent = self.base_unet
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(
                    parent,
                    parts[-1],
                    inflate_attention_layer(
                        original_attn=module,
                        skip_copy=skip_weight_copy # keep memory footprint low and training fast for 4 bit config at UNet2DConditionModel
                    )
                )

        # — 4) Replace all GroupNorms with Sync GroupNorm
        replace_group_norms(self.base_unet, in_place=True)
        assert not any(isinstance(m, nn.GroupNorm) for m in self.base_unet.modules()), \
            "GroupNorm still present!"
        
        # Ensure all other layers remain frozen; This guarantees nothing outside the intended layers is drifting.
        for name, p in self.base_unet.named_parameters():
            if "inflated_attn" not in name and "sync_group_norm" not in name:
                p.requires_grad_(False)

        # — 5) Positional encoding (UV channels)
        self.positional_encoding = CubemapPositionalEncoding(
            num_faces=num_faces,
            embedding_dim=uv_dim
        ).to(self.device)  # ensure it lives on GPU as well

        # — 6) Patch conv_in to accept [latent(4) + mask(1) + uv_dim] channels
        old_conv = self.base_unet.conv_in
        in_ch    = old_conv.in_channels             # typically 4
        mask_ch  = 1
        out_ch   = old_conv.out_channels            # typically 320
        k, s, p  = old_conv.kernel_size, old_conv.stride, old_conv.padding

        new_conv = nn.Conv2d(
            in_channels=in_ch + mask_ch + uv_dim, # in_ch is latent_channels, 4 + 1 + 10 = 15
            out_channels=out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=(old_conv.bias is not None),
        )
        
        # -----------------------------------------------------------
        # copy the old latent weights & biases, zero‐init new channels
        # with torch.no_grad():
        #     if not skip_weight_copy:
        #         new_conv.weight[:, :in_ch].copy_(old_conv.weight)
        #     new_conv.weight[:, in_ch:].zero_()
        #     if old_conv.bias is not None:
        #         new_conv.bias.copy_(old_conv.bias)
        # -----------------------------------------------------------
        
        # copy the old latent weights & biases, zero‐init new channels
        # if no copy, the panorama will be noisy colors even trained for 12k+ data samples
        with torch.no_grad():
            # 1) copy original latent→feature weights
            new_conv.weight[:, :in_ch].copy_(self.base_unet.conv_in.weight)
            # 2) zero-init UV & mask weights so model can learn from scratch
            new_conv.weight[:, in_ch:].zero_()
            new_conv.bias.copy_(self.base_unet.conv_in.bias)
        
        # move the new conv to the same device/dtype
        new_conv = new_conv.to(device=self.device, dtype=self.model_dtype)
        self.base_unet.conv_in = new_conv

        # — 7) Enable gradient checkpointing on the U-Net
        self.base_unet.enable_gradient_checkpointing()

        # — 8) Switch all convs to circular padding for seamless cubemaps
        for m in self.base_unet.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = "circular"

        # — 9) Face-ID & spherical embeddings
        self.face_emb = nn.Embedding(num_faces, uv_dim).to(self.device, self.model_dtype)
        self.sph_emb  = nn.Sequential(
            nn.Linear(2, uv_dim),
            nn.SiLU(),
            nn.Linear(uv_dim, uv_dim),
        ).to(self.device, self.model_dtype)

    def check_dtypes(self):
        """Debug helper to check tensor dtypes in the model"""
        print("\nChecking dtypes in CubeDiffModel:")
        
        # Check base UNet dtypes
        unet_dtype = next(self.base_unet.parameters()).dtype
        print(f"Base UNet dtype: {unet_dtype}")
        
        # Check attention modules
        for i, module in enumerate(self.base_unet.modules()):
            if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
                print(f"Attention module {i}:")
                print(f"  Query dtype: {module.to_q.weight.dtype}")
                print(f"  Key dtype: {module.to_k.weight.dtype}")
                print(f"  Value dtype: {module.to_v.weight.dtype}")
                
                # Only check a few modules to avoid too much output
                if i >= 2:
                    print("...")
                    break
                
    def forward(
        self,
        latents: torch.Tensor,               # [B,6,C,H,W]
        timesteps: torch.Tensor,             # [B] or [B*6]
        encoder_hidden_states: torch.Tensor,  # [B,L,D] or [B*6,L,D]
        **kwargs   # ignored                  # Add this to catch extra parameters
    ) -> torch.Tensor:
        """
        latents: [B, F, 4, H, W]
        timesteps: [B*F] or [B] → will be broadcast
        encoder_hidden_states: [B*F, C_text] or [B, C_text]
        """

        # ─── 0) Move to correct device & dtype ───
        latents = latents.to(device=self.device, dtype=self.model_dtype)
        timesteps = timesteps.to(device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(
            device=self.device, dtype=self.model_dtype
        )

        # unpack shapes
        B, F, C, H, W = latents.shape
        E = self.positional_encoding.embedding_dim  # the uv_dim
        print(f"architecture.py - cubediffmodel - forward - latents.shape is {latents.shape}, E = {E}, B = {B}, F = {F}, C = {C}, H = {H}, W = {W}")
        
        # 1) Add UV enc & flatten → [B*6, C+E, H, W]
        # tiling of latents and encoder_hidden_states into a [B*F,…]
        lat = self.positional_encoding(latents)
        # get the UV embedding channels
        uv  = lat[:, C:, :, :]              # [B*F, E, H, W]

        # …then make sure timesteps and text embeddings are length B*F…
        if timesteps.ndim == 1 and timesteps.shape[0] == B:
            # e.g. [2] → [2*6=12]
            # repeat each timestep for the F faces in that batch
            timesteps = timesteps.repeat_interleave(F)       # → [B*F]
        
        # ——— tile text embeddings to B*F ———
        # if encoder_hidden_states came in as [B, C_text]:
        # handle both [B, C_text]  and  [B, seq_len, C_text]
        if encoder_hidden_states.ndim == 2 and encoder_hidden_states.shape[0] == B:
            # single‐vector conditioning → repeat per face
            encoder_hidden_states = (
                encoder_hidden_states
                .unsqueeze(1)               # [B,1,C_text]
                .expand(-1, F, -1)          # [B,F,C_text]
                .reshape(B*F, -1)           # [B*F,C_text]
            )
        elif encoder_hidden_states.dim() == 3 and encoder_hidden_states.shape[0] == B:
            # per‐token conditioning → repeat per face
            seq_len, C_text = encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]
            encoder_hidden_states = (
                encoder_hidden_states
                .unsqueeze(1)                           # [B,1,seq_len,C_text]
                .expand(-1, F, -1, -1)                 # [B,F,seq_len,C_text]
                .reshape(B*F, seq_len, C_text)         # [B*F,seq_len,C_text]
            )        
        elif encoder_hidden_states.size(0) != B * F:
            raise ValueError(f"architecture.py - architecture.py - Wrong embedding batch: got {encoder_hidden_states.size(0)}, expected {B*F}")
        
        # ─── 1) Build the 1-channel mask ───
        lat = latents.view(B * F, C, H, W)
        # add a 1‐channel image-mask 
        mask = torch.ones((B * F, 1, H, W), device=self.device, dtype=self.model_dtype)
        print(f"architecture.py - cubediffmodel - forward - add a 1‐channel image-mask - C = {C}, lat.shape = {lat.shape}, mask.shape = {mask.shape}")

        # ─── 2) Face-ID positional embedding ───
        # make a repeated 0,1,2..F-1 vector of length B*F
        face_ids = (
            torch.arange(F, device=self.device)
                .unsqueeze(0)
                .repeat(B, 1)
                .view(-1)
        )  # [B*F]
        fe = self.face_emb(face_ids)               # [B*F, E]
        fe = fe.view(B * F, E, 1, 1).expand(-1, -1, H, W)  # [B*F, E, H, W]        

        # ─── 3) Spherical (u,v) embedding ───
        # build a single [H,W,2] UV grid on device
        theta = (torch.arange(W, device=self.device) / W) * 2 * math.pi - math.pi # make theta [-pi , +pi]
        phi   = (torch.arange(H, device=self.device) / H) * math.pi # make phi [0, pi]
        ph, th = torch.meshgrid(phi, theta, indexing="ij")  # both [H,W], Spherical meshgrid
        coords = torch.stack([th, ph], dim=-1).view(-1, 2)   # [H*W,2]

        # cast them to BF16 so your MLP weights match
        coords = coords.to(device=self.device, dtype=self.model_dtype)

        # run the small MLP in BF16
        sph = self.sph_emb(coords)          # [H*W, E]

        # reshape → [E, H, W]
        sph = sph.view(H, W, E).permute(2, 0, 1)  # [E,H,W]

        # expand to [B*F, E, H, W]
        sph = sph.unsqueeze(0).expand(B * F, -1, -1, -1)

        print(f"architecture.py - cubediffmodel - forward - before uv += fe + sph  - uv.shape = {uv.shape}, sph.shape = {sph.shape}, fe.shape = {fe.shape}, lat.shape = {lat.shape}, mask.shape = {mask.shape}")
        
        # ─── 4) Concatenate into a single 15‐channel tensor ───
        # order: 4 latent + 1 mask + E (faceID emb + sphere emb + uv emb) 
        # 4 (latent) + 1 (mask) + E (uv channels)  = 4 + 1 + 10 = 15 channels
        uv += fe + sph                  # → [B*F, E, H, W]
        # now cat exactly 4+1+E channels
        lat_input = torch.cat([lat, mask, uv], dim=1) # → shape: [B*F, 4 + 1 + E, H, W] ([12, 15, 64, 64]), where E=10, so 4 + 1  + 10 = 15 channels
        print(f"architecture.py - cubediffmodel - forward - after uv += fe + sph and torch.cat([lat, mask, uv], dim=1)  - lat_input.shape = {lat_input.shape}, encoder_hidden_states.shape = {encoder_hidden_states.shape}")
        
        # ─── 5) Run UNet under mixed precision ───
        with torch.autocast("cuda", dtype=self.model_dtype):
            unet_out = self.base_unet(
                sample=lat_input,  # [B*F,15,H,W]
                timestep=timesteps, # [B*F], sample (size B*F) and timesteps (size B*F) agree,
                encoder_hidden_states=encoder_hidden_states, # [B*F, C_text]
            ).sample  # or `.sample` / `.latent` depending on your version

        # ─── 6) Un-flatten back to [B, F, ...] ───
        out_ch = unet_out.shape[1]
        unet_out = unet_out.view(B, F, out_ch, H, W)

        return unet_out

