import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F           

# class CubemapPositionalEncoding(nn.Module):
#     """
#     Positional encoding for cubemap geometry.
#     """
#     # def __init__(self, embedding_dim=4, max_resolution=64):
#     def __init__(self, embedding_dim=9, max_resolution=64):   # 3 xyz + 6 sincos
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.max_resolution = max_resolution
        
#         # Pre-compute cube face coordinates
#         self.register_buffer('face_coords', self._compute_face_coords())
    
#     def _compute_face_coords(self):
#         """
#         Compute 3D coordinates for each point on the cubemap faces.
        
#         Returns:
#             Tensor of shape [6, max_resolution, max_resolution, 3]
#         """
#         coords = torch.zeros(6, self.max_resolution, self.max_resolution, 3)
        
#         for face_idx in range(6):
#             for y in range(self.max_resolution):
#                 for x in range(self.max_resolution):
#                     # Normalize coordinates to [-1, 1]
#                     x_norm = 2 * (x + 0.5) / self.max_resolution - 1
#                     y_norm = 2 * (y + 0.5) / self.max_resolution - 1
                    
#                     # Face-specific mapping to 3D coordinates
#                     if face_idx == 0:   # Front
#                         vec = [1.0, x_norm, -y_norm]
#                     elif face_idx == 1: # Right
#                         vec = [-x_norm, 1.0, -y_norm]
#                     elif face_idx == 2: # Back
#                         vec = [-1.0, -x_norm, -y_norm]
#                     elif face_idx == 3: # Left
#                         vec = [x_norm, -1.0, -y_norm]
#                     elif face_idx == 4: # Top
#                         vec = [x_norm, y_norm, 1.0]
#                     elif face_idx == 5: # Bottom
#                         vec = [x_norm, -y_norm, -1.0]
                    
#                     # Normalize to unit vector
#                     norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
#                     vec = [v / norm for v in vec]
                    
#                     # Store coordinates
#                     coords[face_idx, y, x, 0] = vec[0]
#                     coords[face_idx, y, x, 1] = vec[1]
#                     coords[face_idx, y, x, 2] = vec[2]
                    
#                     # KEEP existing xyz unit-vector computation
#                     coords[face_idx, y, x] = torch.tensor(vec)   # store (x,y,z)
        
#         return coords
    
#     # def forward(self, x, resolution=None):
#     #     """
#     #     Add positional encodings to input tensor.
        
#     #     Args:
#     #         x: Input tensor of shape [batch, num_faces, channels, height, width]
#     #         resolution: Resolution of input (default: self.max_resolution)
            
#     #     Returns:
#     #         Tensor with positional encodings added
#     #     """
#     #     batch_size, num_faces, C, H, W = x.shape
#     #     resolution = resolution or self.max_resolution
        
#     #     if resolution != self.max_resolution:
#     #         # Resize coordinates to match input resolution
#     #         coords = F.interpolate(
#     #             self.face_coords.permute(0, 3, 1, 2),  # [6, 3, H, W]
#     #             size=(H, W),
#     #             mode='bilinear',
#     #             align_corners=False
#     #         ).permute(0, 2, 3, 1)  # Back to [6, H, W, 3]
#     #     else:
#     #         coords = self.face_coords
        
#     #     # Extract UV coordinates (first 2 dimensions)
#     #     uv_coords = coords[..., :2]  # [6, H, W, 2]
        
#     #     # Reshape and repeat for batch
#     #     uv_coords = uv_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 6, H, W, 2]
        
#     #     # Convert to channels-first format
#     #     uv_coords = uv_coords.permute(0, 1, 4, 2, 3)  # [B, 6, 2, H, W]
        
#     #     # Concatenate with input along channel dimension
#     #     output = torch.cat([x, uv_coords], dim=2)
        
#     #     return output

#     def forward(self, x, resolution=None):
#         """
#         Args
#         ----
#         x : [B,6,C,H,W]   latent blocks
#         resolution : int  optional; defaults to H
#         """
#         B,F,C,H,W = x.shape
#         resolution = resolution or H

#         # --- resize cached (64×64) xyz grid if needed ---
#         if resolution != self.max_resolution:
#             coords = F.interpolate(self.face_coords.permute(0,3,1,2),  # [6,3,H,W]
#                                    size=(H,W), mode="bilinear",
#                                    align_corners=False).permute(0,2,3,1)
#         else:
#             coords = self.face_coords                               # [6,H,W,3]

#         # 1) xyz unit-vector
#         vec = coords.unsqueeze(0).repeat(B,1,1,1,1)                 # [B,6,H,W,3]

#         # 2) single-frequency Fourier (π)  → sin & cos
#         angles  = vec * np.pi
#         sincos  = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [...,6]

#         pe = torch.cat([vec, sincos], dim=-1)                        # [B,6,H,W,9]
#         pe = pe.permute(0,1,4,2,3)                                   # [B,6,9,H,W]

#         return torch.cat([x, pe], dim=2)                             # concat channels


# cl/model/positional_encoding.py

import torch
import torch.nn as nn
from typing import Optional

class CubemapPositionalEncoding(nn.Module):
    def __init__(self,
                 num_faces: int = 6,
                 embedding_dim: int = 9,
                 max_resolution: int = None):
        super().__init__()
        # store params (max_resolution is only for API compatibility)
        self.num_faces      = num_faces
        self.embedding_dim  = embedding_dim
        self.max_resolution = max_resolution

        # face index → vector
        self.face_emb = nn.Embedding(num_faces, embedding_dim)
        # UV coords → vector
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, latents: torch.Tensor):
        """
        latents: either
          [B, F, C, H, W]   or
          [B*F, C, H, W]
        Returns:
          [B*F, C+E, H, W]
        """
        # 1) reshape into [B*F, C, H, W]
        if latents.dim() == 5:
            B, F, C, H, W = latents.shape
            x = latents.reshape(B * F, C, H, W)
        else:
            # assume F= num_faces, batch = first dim // F
            B = latents.shape[0] // self.num_faces
            F, C, H, W = self.num_faces, latents.shape[1], latents.shape[2], latents.shape[3]
            x = latents

        device = x.device

        # 2) build per-face positional embedding of shape [F*H*W, E]
        # 2a) face IDs
        face_ids = torch.arange(F, device=device).view(F,1,1).expand(F, H, W)
        face_ids = face_ids.reshape(-1)                 # [F*H*W]
        fe       = self.face_emb(face_ids)              # [F*H*W, E]

        # 2b) normalized coords in [-1,1]
        ys = torch.linspace(-1,1,H,device=device).view(H,1).expand(H,W)
        xs = torch.linspace(-1,1,W,device=device).view(1,W).expand(H,W)
        coords = torch.stack([xs, ys], dim=-1)          # [H, W, 2]
        coords = coords.reshape(-1,2).unsqueeze(0)      # [1, H*W, 2]
        coords = coords.repeat(F,1,1).reshape(-1,2)     # [F*H*W, 2]
        ce = self.coord_mlp(coords)                     # [F*H*W, E]

        # 3) combine and reshape to [F, E, H, W]
        pe_face = (fe + ce)                             # [F*H*W, E]
        pe_face = pe_face.view(F, H, W, self.embedding_dim)    # [F, H, W, E]
        pe_face = pe_face.permute(0, 3, 1, 2)                   # [F, E, H, W]

        # 4) tile across B to get [B*F, E, H, W]
        pe = pe_face.unsqueeze(0).repeat(B, 1, 1, 1, 1)         # [B, F, E, H, W]
        pe = pe.reshape(B * F, self.embedding_dim, H, W)      # [B*F, E, H, W]

        # 5) concat with original latents
        return torch.cat([x, pe], dim=1)                      # → [B*F, C+E, H, W]

