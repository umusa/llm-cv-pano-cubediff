import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F           

class CubemapPositionalEncoding(nn.Module):
    """
    Positional encoding for cubemap geometry.
    """
    # def __init__(self, embedding_dim=4, max_resolution=64):
    def __init__(self, embedding_dim=9, max_resolution=64):   # 3 xyz + 6 sincos
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_resolution = max_resolution
        
        # Pre-compute cube face coordinates
        self.register_buffer('face_coords', self._compute_face_coords())
    
    def _compute_face_coords(self):
        """
        Compute 3D coordinates for each point on the cubemap faces.
        
        Returns:
            Tensor of shape [6, max_resolution, max_resolution, 3]
        """
        coords = torch.zeros(6, self.max_resolution, self.max_resolution, 3)
        
        for face_idx in range(6):
            for y in range(self.max_resolution):
                for x in range(self.max_resolution):
                    # Normalize coordinates to [-1, 1]
                    x_norm = 2 * (x + 0.5) / self.max_resolution - 1
                    y_norm = 2 * (y + 0.5) / self.max_resolution - 1
                    
                    # Face-specific mapping to 3D coordinates
                    if face_idx == 0:   # Front
                        vec = [1.0, x_norm, -y_norm]
                    elif face_idx == 1: # Right
                        vec = [-x_norm, 1.0, -y_norm]
                    elif face_idx == 2: # Back
                        vec = [-1.0, -x_norm, -y_norm]
                    elif face_idx == 3: # Left
                        vec = [x_norm, -1.0, -y_norm]
                    elif face_idx == 4: # Top
                        vec = [x_norm, y_norm, 1.0]
                    elif face_idx == 5: # Bottom
                        vec = [x_norm, -y_norm, -1.0]
                    
                    # Normalize to unit vector
                    norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                    vec = [v / norm for v in vec]
                    
                    # Store coordinates
                    coords[face_idx, y, x, 0] = vec[0]
                    coords[face_idx, y, x, 1] = vec[1]
                    coords[face_idx, y, x, 2] = vec[2]
                    
                    # KEEP existing xyz unit-vector computation
                    coords[face_idx, y, x] = torch.tensor(vec)   # store (x,y,z)
        
        return coords
    
    # def forward(self, x, resolution=None):
    #     """
    #     Add positional encodings to input tensor.
        
    #     Args:
    #         x: Input tensor of shape [batch, num_faces, channels, height, width]
    #         resolution: Resolution of input (default: self.max_resolution)
            
    #     Returns:
    #         Tensor with positional encodings added
    #     """
    #     batch_size, num_faces, C, H, W = x.shape
    #     resolution = resolution or self.max_resolution
        
    #     if resolution != self.max_resolution:
    #         # Resize coordinates to match input resolution
    #         coords = F.interpolate(
    #             self.face_coords.permute(0, 3, 1, 2),  # [6, 3, H, W]
    #             size=(H, W),
    #             mode='bilinear',
    #             align_corners=False
    #         ).permute(0, 2, 3, 1)  # Back to [6, H, W, 3]
    #     else:
    #         coords = self.face_coords
        
    #     # Extract UV coordinates (first 2 dimensions)
    #     uv_coords = coords[..., :2]  # [6, H, W, 2]
        
    #     # Reshape and repeat for batch
    #     uv_coords = uv_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 6, H, W, 2]
        
    #     # Convert to channels-first format
    #     uv_coords = uv_coords.permute(0, 1, 4, 2, 3)  # [B, 6, 2, H, W]
        
    #     # Concatenate with input along channel dimension
    #     output = torch.cat([x, uv_coords], dim=2)
        
    #     return output

    def forward(self, x, resolution=None):
        """
        Args
        ----
        x : [B,6,C,H,W]   latent blocks
        resolution : int  optional; defaults to H
        """
        B,F,C,H,W = x.shape
        resolution = resolution or H

        # --- resize cached (64×64) xyz grid if needed ---
        if resolution != self.max_resolution:
            coords = F.interpolate(self.face_coords.permute(0,3,1,2),  # [6,3,H,W]
                                   size=(H,W), mode="bilinear",
                                   align_corners=False).permute(0,2,3,1)
        else:
            coords = self.face_coords                               # [6,H,W,3]

        # 1) xyz unit-vector
        vec = coords.unsqueeze(0).repeat(B,1,1,1,1)                 # [B,6,H,W,3]

        # 2) single-frequency Fourier (π)  → sin & cos
        angles  = vec * np.pi
        sincos  = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [...,6]

        pe = torch.cat([vec, sincos], dim=-1)                        # [B,6,H,W,9]
        pe = pe.permute(0,1,4,2,3)                                   # [B,6,9,H,W]

        return torch.cat([x, pe], dim=2)                             # concat channels
