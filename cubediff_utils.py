"""
Utility functions for CubeDiff implementation.
Contains functions for converting between equirectangular and cubemap representations.
"""

import numpy as np
import cv2
from PIL import Image
import torch


def equirect_to_cubemap(equirect_img, face_size=512):
    """
    Convert equirectangular image to six cubemap faces.
    
    Args:
        equirect_img: PIL Image or numpy array in equirectangular format
        face_size: Size of each cubemap face (default: 512)
        
    Returns:
        List of 6 numpy arrays representing the faces of the cubemap in order:
        [front, right, back, left, top, bottom]
    """
    # Convert PIL to numpy if needed
    if isinstance(equirect_img, Image.Image):
        equirect_img = np.array(equirect_img)
    
    # Get equirectangular image dimensions
    h, w = equirect_img.shape[:2]
    
    # Initialize cubemap faces
    faces = []
    
    # Generate mapping for each face
    # Order: front, right, back, left, top, bottom
    faces_info = [
        {'name': 'front',  'rotation': [0, 0, 0]},        # +z
        {'name': 'right',  'rotation': [0, np.pi/2, 0]},  # +x
        {'name': 'back',   'rotation': [0, np.pi, 0]},    # -z
        {'name': 'left',   'rotation': [0, -np.pi/2, 0]}, # -x
        {'name': 'top',    'rotation': [np.pi/2, 0, 0]},  # +y
        {'name': 'bottom', 'rotation': [-np.pi/2, 0, 0]}  # -y
    ]
    
    for face_info in faces_info:
        # Create map for remapping
        map_x = np.zeros((face_size, face_size), dtype=np.float32)
        map_y = np.zeros((face_size, face_size), dtype=np.float32)
        # Create grid of normalized face coordinates (-1 to 1)
        y_range = np.linspace(1, -1, face_size)
        x_range = np.linspace(-1, 1, face_size)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        # Get rotation values
        rx, ry, rz = face_info['rotation']
        
        # Define unit vectors for each face
        if face_info['name'] == 'front':
            # (x, y, 1) normalized
            x = x_grid
            y = y_grid
            z = np.ones_like(x)
        elif face_info['name'] == 'right':
            # (1, y, -x) normalized
            z = -x_grid
            y = y_grid
            x = np.ones_like(z)
        elif face_info['name'] == 'back':
            # (-x, y, -1) normalized
            x = -x_grid
            y = y_grid
            z = -np.ones_like(x)
        elif face_info['name'] == 'left':
            # (-1, y, x) normalized
            z = x_grid
            y = y_grid
            x = -np.ones_like(z)
        elif face_info['name'] == 'top':
            # (x, 1, -y) normalized
            x = x_grid
            z = -y_grid
            y = np.ones_like(x)
        elif face_info['name'] == 'bottom':
            # (x, -1, y) normalized
            x = x_grid
            z = y_grid
            y = -np.ones_like(x)
            
        # Normalize vectors
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm
        
        # Convert to spherical coordinates
        theta = np.arctan2(x, z)  # Longitude (0 to 2π)
        phi = np.arcsin(y)        # Latitude (-π/2 to π/2)
        
        # Map to equirectangular coordinates
        map_x = (theta / (2 * np.pi) + 0.5) * w
        map_y = (0.5 - phi / np.pi) * h
        
        # Handle boundary conditions
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        # Remap with high-quality interpolation
        face = cv2.remap(equirect_img, map_x.astype(np.float32), map_y.astype(np.float32), 
                          interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        faces.append(face)
    
    return faces


def cubemap_to_equirect(cubemap_faces, output_height=1024, output_width=2048):
    """
    Convert 6 cubemap faces to an equirectangular panorama with enhanced smooth seams.
    
    Args:
        cubemap_faces: List of 6 images (numpy arrays) in order [front, right, back, left, top, bottom]
        output_height: Height of the output equirectangular image
        output_width:  Width of the output equirectangular image (typically 2 * height)
    
    Returns:
        Equirectangular image as a numpy array (float32).
    """
    # Check face consistency
    face_size = cubemap_faces[0].shape[0]
    for face in cubemap_faces:
        assert face.shape[0] == face.shape[1] == face_size, (
            "All cubemap faces must be square and of the same size"
        )

    # Prepare output
    if len(cubemap_faces[0].shape) == 3:
        # Color image
        channels = cubemap_faces[0].shape[2]
        equirect = np.zeros((output_height, output_width, channels), dtype=np.float32)
    else:
        # Grayscale image
        equirect = np.zeros((output_height, output_width), dtype=np.float32)
    
    # Weight map for blending
    weight_map = np.zeros((output_height, output_width), dtype=np.float32)

    # For each pixel in the equirectangular image
    for y in range(output_height):
        # Map y to latitude (π/2 to -π/2)
        lat = np.pi * (0.5 - y / output_height)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        
        for x in range(output_width):
            # Map x to longitude (-π to π)
            lon = 2.0 * np.pi * x / output_width - np.pi
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)
            
            # 3D Cartesian coordinates on the unit sphere
            x_cart = cos_lat * sin_lon
            y_cart = sin_lat
            z_cart = cos_lat * cos_lon
            
            # Accumulate from all 6 faces with enhanced blending
            accum_color = 0.0
            accum_weight = 0.0

            for face_idx, face_img in enumerate(cubemap_faces):
                # Convert (x_cart, y_cart, z_cart) to (u, v, face_weight) for this face
                # -------------------------------------------------------
                # Face order: [0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom]
                #
                if face_idx == 0:  # front (+Z)
                    if z_cart <= 0: 
                        continue
                    inv = 1.0 / abs(z_cart)
                    u =  x_cart * inv
                    v = -y_cart * inv
                elif face_idx == 1:  # right (+X)
                    if x_cart <= 0:
                        continue
                    inv = 1.0 / abs(x_cart)
                    u = -z_cart * inv
                    v = -y_cart * inv
                elif face_idx == 2:  # back (-Z)
                    if z_cart >= 0:
                        continue
                    inv = 1.0 / abs(z_cart)
                    u = -x_cart * inv
                    v = -y_cart * inv
                elif face_idx == 3:  # left (-X)
                    if x_cart >= 0:
                        continue
                    inv = 1.0 / abs(x_cart)
                    u =  z_cart * inv
                    v = -y_cart * inv
                elif face_idx == 4:  # top (+Y)
                    if y_cart <= 0:
                        continue
                    inv = 1.0 / abs(y_cart)
                    u =  x_cart * inv
                    v = -z_cart * inv
                else:               # bottom (-Y)
                    if y_cart >= 0:
                        continue
                    inv = 1.0 / abs(y_cart)
                    u =  x_cart * inv
                    v =  z_cart * inv
                
                # Enhanced face weight calculation for smoother transitions
                # Distance from center of face (normalized [0,1])
                # Using max norm (L∞) for better blending at corners
                dist_from_center = max(abs(u), abs(v))
                
                # Weight calculation: higher towards center, lower at edges
                # Enhanced cubic falloff for even smoother transitions
                w = (1.0 - dist_from_center)**3 if dist_from_center < 1.0 else 0.0
                
                if w <= 0.0:
                    continue

                # Map (u,v) from [-1,1] -> [0, face_size-1]
                u_mapped = (u + 1) * (face_size - 1) / 2.0
                v_mapped = (v + 1) * (face_size - 1) / 2.0

                # If pixel is within face bounds, bicubic-interpolate
                if 0 <= u_mapped < face_size and 0 <= v_mapped < face_size:
                    # Bicubic interpolation for higher quality
                    # For simplicity, use bilinear here but OpenCV could be used for bicubic
                    u_floor = int(u_mapped)
                    v_floor = int(v_mapped)
                    u_ceil = min(u_floor + 1, face_size - 1)
                    v_ceil = min(v_floor + 1, face_size - 1)

                    # Interpolation weights
                    wu = u_mapped - u_floor
                    wv = v_mapped - v_floor

                    # Gather the four corners
                    if len(face_img.shape) == 3:
                        # color image
                        c00 = face_img[v_floor, u_floor]
                        c01 = face_img[v_floor, u_ceil]
                        c10 = face_img[v_ceil, u_floor]
                        c11 = face_img[v_ceil, u_ceil]
                    else:
                        # grayscale
                        c00 = face_img[v_floor, u_floor]
                        c01 = face_img[v_floor, u_ceil]
                        c10 = face_img[v_ceil, u_floor]
                        c11 = face_img[v_ceil, u_ceil]
                    
                    # Bilinear interpolate
                    pix = ((1 - wu) * (1 - wv) * c00 +
                           (1 - wu) * wv       * c10 +
                           wu       * (1 - wv) * c01 +
                           wu       * wv       * c11)

                    # Accumulate color * weight with higher weight near face centers
                    accum_color += pix * w
                    accum_weight += w

            # Final blended color with normalization
            if accum_weight > 0.0:
                equirect[y, x] = accum_color / accum_weight
            else:
                # Use a default background color if no face covers this pixel
                # (This shouldn't happen for a complete 360° sphere)
                equirect[y, x] = 0

    return equirect


def add_cubemap_positional_encodings(latents, batch_size=None):
    """
    Add enhanced positional encodings to latents based on cubemap geometry.
    Provides stronger spatial awareness about the 3D cube structure.
    
    Args:
        latents: Latent tensor with shape [B*6, C, H, W] or [6, C, H, W]
        batch_size: Optional explicit batch size
        
    Returns:
        Latent tensor with added positional encodings
    """
    device = latents.device
    dtype = latents.dtype
    B, C, H, W = latents.shape
    
    # Determine if we have a batch of cubemaps or just one
    is_batched = (B % 6 == 0)
    
    if is_batched:
        # Multiple cubemaps in batch
        if batch_size is None:
            batch_size = B // 6
    else:
        # Single cubemap (6 faces)
        assert B == 6, f"Expected either 6 faces or a multiple of 6, got {B}"
        batch_size = 1
    
    # Create positional encoding channels, 2 per face (u, v coordinates)
    num_faces = 6
    pos_enc = torch.zeros(B, 2, H, W, device=device, dtype=dtype)
    
    # Define face orientations in 3D space
    # This helps the model understand the spatial relationships between faces
    face_orientations = [
        (0, 0, 1),    # front:  +Z direction
        (1, 0, 0),    # right:  +X direction
        (0, 0, -1),   # back:   -Z direction
        (-1, 0, 0),   # left:   -X direction
        (0, 1, 0),    # top:    +Y direction
        (0, -1, 0),   # bottom: -Y direction
    ]
    
    # Loop through all images in the batch
    for batch_idx in range(batch_size):
        for face_idx in range(num_faces):
            # Calculate the index in the flattened batch
            if is_batched:
                idx = batch_idx * 6 + face_idx
            else:
                idx = face_idx
            
            # Create normalized grid for this face: u and v in range [-1, 1]
            v_range = torch.linspace(-1, 1, H, device=device, dtype=dtype)
            u_range = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            grid_v, grid_u = torch.meshgrid(v_range, u_range, indexing="ij")
            
            # Get face orientation
            face_orient = face_orientations[face_idx]
            
            # Add orientation information to better distinguish faces
            # Scale and bias to keep in reasonable range
            orient_u = grid_u + face_orient[0] * 0.1
            orient_v = grid_v + face_orient[1] * 0.1
            
            # Save normalized coordinates with orientation information
            pos_enc[idx, 0, :, :] = orient_u
            pos_enc[idx, 1, :, :] = orient_v
    
    # Concatenate positional encodings with latents
    latents_with_pos = torch.cat([latents, pos_enc], dim=1)
    
    return latents_with_pos