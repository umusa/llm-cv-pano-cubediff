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
from cl.model.positional_encoding import CubemapPositionalEncoding
from cl.model.attention        import inflate_attention_layer
from cl.model.normalization    import replace_group_norms
from transformers import BitsAndBytesConfig

class CubeDiffModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_faces: int = 6,
        uv_dim: int   = 9
    ):
        super().__init__()
        # — 1) Load only the U-Net
        # self.base_unet = UNet2DConditionModel.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="unet",
        #     torch_dtype=torch.float32
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.base_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        # print(f"after loading base_unet\n")
        # self.check_dtypes()

        self.model_dtype = torch.bfloat16
        self.device = next(self.base_unet.parameters()).device
        
        # — 2) Inflate its attention to cross‐face
        for name, module in list(self.base_unet.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                parent = self.base_unet
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], inflate_attention_layer(module))
        # print(f"after inflating attention layers\n")
        # self.check_dtypes()

        # — 3) Sync GroupNorm
        replace_group_norms(self.base_unet, in_place=True)
        print(f"after replacing group norms\n")
        # self.check_dtypes()

        # — 4) Positional encoding (adds UV channels)
        self.positional_encoding = CubemapPositionalEncoding(
            num_faces=num_faces,
            embedding_dim=uv_dim
        )
        # print(f"after positional_encoding\n")
        # self.check_dtypes()

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

        # print(f"after replacing conv_in, before self.base_unet.conv_in = new_in, new_in type is {type(new_in)}\n")
        # self.check_dtypes()
        # Explicitly cast to the same dtype BEFORE assigning
        new_in = new_in.to(dtype=self.model_dtype, device=self.device)
        self.base_unet.conv_in = new_in

        # print(f"after replacing conv_in, Patch conv_in to accept the UV channels\n")
        # self.check_dtypes()

        # — 6) Gradient checkpointing on the U-Net
        self.base_unet.enable_gradient_checkpointing()

        # print(f"after enabling gradient checkpointing\n")
        # self.check_dtypes()

        # — 7) Circular padding everywhere
        for m in self.base_unet.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = "circular"

        # print(f"after circular padding everywhere\n")
        # self.check_dtypes()
        # — 8) Spherical positional & face-ID embeddings (Esteves et al 2021)
        self.face_emb = nn.Embedding(num_faces, uv_dim)
        self.sph_emb = nn.Sequential(
            nn.Linear(2, uv_dim),
            nn.SiLU(),
            nn.Linear(uv_dim, uv_dim),
        )
        # print(f"after spherical positional & face-ID embeddings\n")
        # self.check_dtypes()

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
        # print(f"architecture.py - cubeDiffModel - forward() - kwargs is {kwargs}\n")
        B, F, C, H, W = latents.shape
        E = self.positional_encoding.embedding_dim

        # 1) Add UV enc & flatten → [B*6, C+E, H, W]
        # tiling of latents and encoder_hidden_states into a [B*F,…]
        lat = self.positional_encoding(latents)
        lat = lat.view(B * F, C + E, H, W)

        # — 1.5) Inject face-ID & spherical coords into the UV channels only
        # Split into original latent channels vs. UV embedding channels
        orig = lat[:, :C, :, :]              # [B*F, C, H, W]
        uv   = lat[:, C:, :, :]              # [B*F, E, H, W]

        # face-ID embedding (per face index)
        face_ids = torch.arange(F, device=lat.device).unsqueeze(0).repeat(B,1).view(-1)
        fe = self.face_emb(face_ids).view(B*F, E,1,1)

        # spherical coordinate embedding
        theta = (torch.arange(W, device=lat.device)/W)*2*math.pi - math.pi
        phi   = (torch.arange(H, device=lat.device)/H)*math.pi
        th, ph = torch.meshgrid(theta, phi, indexing="xy")
        coords = torch.stack([th,ph], -1).view(-1, 2)      # [H*W,2]
        sph_emb = self.sph_emb(coords)                     # [H*W, E]
        sph = sph_emb.view(1, E, H, W).expand(B*F, -1, -1, -1)

        # add only to the UV slice
        uv = uv + fe + sph

        # recombine to full [C+E] channel map
        lat = torch.cat([orig, uv], dim=1)                 # [B*F, C+E, H, W]

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
        
        # — cast everything *before* it ever hits the attention kernels —
        # Right before calling self.base_unet(...), now ensure both the latent inputs (which become your query) 
        # and the text embeddings (which become key & value) are all torch.bfloat16 on the GPU. 
        # That satisfies xFormers’ requirement that Q, K, and V share the same dtype.

        lat = lat.to(dtype=self.model_dtype, device=self.device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.model_dtype, device=self.device)
        # now we know Q (latents), K,V (k and v are text embedding) all come in as bfloat16
        # wrap the entire call in an autocast context as a last resort
        # print(f"architecture.py - cubeDiffModel - forward() - before unet_out = self.base_unet, lat dtype is {lat.dtype}, encoder_hidden_states dtype is {encoder_hidden_states.dtype}, timesteps dtype is {timesteps.dtype} \n")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            unet_out = self.base_unet(
                sample=lat,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states
            )
        # print(f"architecture.py - cubeDiffModel - forward() - after unet_out = self.base_unet\n")
        out = unet_out.sample  # [B*6, C, H, W]

        # 5) reshape back → [B,6,C,H,W]
        return out.view(B, F, *out.shape[1:])

