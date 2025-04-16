import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from scipy.ndimage import map_coordinates
import time
from matplotlib.colors import LinearSegmentedColormap

def load_image_from_url(url):
    """
    Load an image from a URL.
    
    Args:
        url: URL of the image
        
    Returns:
        numpy array containing the image
    """
    try:
        # Download the image
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad responses
        
        # Convert to numpy array using PIL
        pil_image = Image.open(BytesIO(response.content))
        
        # Convert PIL image to numpy array
        image_np = np.array(pil_image)
        
        # If image is grayscale, convert to 3-channel
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        # If image has alpha channel, remove it
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        print(f"Successfully loaded image: {image_np.shape}")
        return image_np
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def load_local_image(file_path):
    """
    Load an image from a local file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy array containing the image
    """
    try:
        # Load using OpenCV
        image = cv2.imread(file_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Successfully loaded image: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading local image: {e}")
        return None

def add_cubemap_positional_encodings(latents, face_indices=None):
    """
    Add positional encodings to cubemap latents based on 3D geometry.
    
    Args:
        latents: Tensor of latent representations for cubemap faces
                Shape: (batch_size, num_faces, channels, height, width) or
                       (num_faces, channels, height, width)
        face_indices: Optional indices of faces to process (default: all faces)
        
    Returns:
        Tensor with positional encodings added
    """
    # Check if we have a batch dimension
    has_batch = len(latents.shape) == 5
    
    if has_batch:
        batch_size, num_faces, channels, height, width = latents.shape
    else:
        num_faces, channels, height, width = latents.shape
        batch_size = 1
        # Add batch dimension
        latents = latents.unsqueeze(0)
    
    # Process all faces if not specified
    if face_indices is None:
        face_indices = list(range(num_faces))
    # Ensure face_indices is an iterable, not an int
    elif isinstance(face_indices, int):
        face_indices = [face_indices]
    
    # Face directions in 3D
    face_directions = [
        [0, 0, -1],  # Front (negative Z)
        [1, 0, 0],   # Right (positive X)
        [0, 0, 1],   # Back (positive Z)
        [-1, 0, 0],  # Left (negative X)
        [0, -1, 0],  # Top (negative Y)
        [0, 1, 0]    # Bottom (positive Y)
    ]
    
    # Precompute up vectors for each face
    up_vectors = [
        [0, -1, 0],  # Front
        [0, -1, 0],  # Right
        [0, -1, 0],  # Back
        [0, -1, 0],  # Left
        [0, 0, 1],   # Top
        [0, 0, -1]   # Bottom
    ]
    
    # Precompute right vectors for each face
    right_vectors = [
        [1, 0, 0],    # Front
        [0, 0, 1],    # Right
        [-1, 0, 0],   # Back
        [0, 0, -1],   # Left
        [1, 0, 0],    # Top
        [1, 0, 0]     # Bottom
    ]
    
    # Create positional encoding grid for each face
    for face_idx in face_indices:
        # Get the face direction and basis vectors
        face_dir = torch.tensor(face_directions[face_idx], dtype=torch.float32, device=latents.device)
        up = torch.tensor(up_vectors[face_idx], dtype=torch.float32, device=latents.device)
        right = torch.tensor(right_vectors[face_idx], dtype=torch.float32, device=latents.device)
        
        # Create normalized pixel coordinates grid [-1, 1] x [-1, 1]
        y_coords = torch.linspace(-1, 1, height, device=latents.device)
        x_coords = torch.linspace(-1, 1, width, device=latents.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate 3D positions on the cube for each pixel
        # Start with the face center direction
        pos_3d = face_dir.unsqueeze(0).unsqueeze(0).expand(height, width, 3).clone()
        
        # Add offset in the right direction based on x coordinate
        pos_3d += right.unsqueeze(0).unsqueeze(0) * grid_x.unsqueeze(-1)
        
        # Add offset in the up direction based on y coordinate
        pos_3d += up.unsqueeze(0).unsqueeze(0) * grid_y.unsqueeze(-1)
        
        # Normalize to unit vectors (points on the unit sphere)
        norm = torch.norm(pos_3d, dim=2, keepdim=True)
        pos_3d = pos_3d / norm
        
        # Create positional encoding features (x, y, z coordinates as separate channels)
        pos_features = pos_3d.permute(2, 0, 1)  # [3, H, W]
        
        # Optionally, add spherical coordinates (phi, theta)
        phi = torch.acos(torch.clamp(pos_3d[:, :, 1], -1.0, 1.0))  # Polar angle, clamped to avoid numerical issues
        theta = torch.atan2(pos_3d[:, :, 2], pos_3d[:, :, 0])  # Azimuthal angle
        pos_features = torch.cat([
            pos_features,
            phi.unsqueeze(0),
            theta.unsqueeze(0)
        ], dim=0)  # [5, H, W]
        
        # Expand positional features to match channel dimension of latents
        expanded_pos = pos_features.unsqueeze(0)  # [1, 5, H, W]
        
        # Calculate number of repetitions needed
        n_repeat = (channels + 4) // 5
        
        # Repeat and then trim to exact channel count
        expanded_pos = expanded_pos.repeat(1, n_repeat, 1, 1)  # [1, n_repeat*5, H, W]
        expanded_pos = expanded_pos[:, :channels, :, :]  # [1, C, H, W]
        
        # Add to latents for all batches for this face
        for b in range(batch_size):
            latents[b, face_idx] = latents[b, face_idx] + 0.1 * expanded_pos
    
    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        latents = latents.squeeze(0)
    
    return latents


# --------------------------------------------

def equirect_to_cubemap(equirect_img, face_size):
    """
    Convert equirectangular image to cubemap faces with precise mapping
    following the standard cubemap orientation:
    
    - Front:  +X face (positive X direction)
    - Right:  +Y face (positive Y direction)
    - Back:   -X face (negative X direction)
    - Left:   -Y face (negative Y direction)
    - Top:    +Z face (positive Z direction)
    - Bottom: -Z face (negative Z direction)
    
    Args:
        equirect_img: Equirectangular image
        face_size: Size of each face (width=height=face_size)
    
    Returns:
        Dictionary of 6 cubemap faces following the standard orientation
    """
    import numpy as np
    import cv2
    
    equirect_h, equirect_w = equirect_img.shape[:2]
    faces = {}
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    # Generate normalized pixel coordinates
    y_ndc, x_ndc = np.meshgrid(
        np.linspace(-1, 1, face_size, dtype=np.float64),
        np.linspace(-1, 1, face_size, dtype=np.float64),
        indexing='ij'
    )
    
    # For each face in cubemap
    for face_name in face_names:
        # Initialize 3D vectors
        x3d = np.zeros_like(x_ndc)
        y3d = np.zeros_like(y_ndc)
        z3d = np.zeros_like(x_ndc)
        
        # Set coordinates based on face
        # CRITICAL: Use the correct face orientation as per the 3D cubemap visualization
        if face_name == 'front':    # +X
            x3d.fill(1.0)
            y3d = -x_ndc
            z3d = -y_ndc
        elif face_name == 'right':  # +Y
            x3d = x_ndc
            y3d.fill(1.0)
            z3d = -y_ndc
        elif face_name == 'back':   # -X
            x3d.fill(-1.0)
            y3d = x_ndc
            z3d = -y_ndc
        elif face_name == 'left':   # -Y
            x3d = -x_ndc
            y3d.fill(-1.0)
            z3d = -y_ndc
        elif face_name == 'top':    # +Z
            x3d = x_ndc
            y3d = y_ndc
            z3d.fill(1.0)
        elif face_name == 'bottom': # -Z
            x3d = x_ndc
            y3d = -y_ndc
            z3d.fill(-1.0)
        
        # Calculate unit vector length
        norm = np.sqrt(x3d**2 + y3d**2 + z3d**2)
        
        # Convert to spherical coordinates
        # Handle potential division by zero with small epsilon
        eps = 1e-10
        phi = np.arctan2(y3d, x3d)  # Longitude: -π to π
        theta = np.arccos(np.clip(z3d / (norm + eps), -1.0 + eps, 1.0 - eps))  # Latitude: 0 to π
        
        # Convert spherical coordinates to equirectangular pixel coordinates
        # Map φ: [-π, π] → u: [0, width]
        # Map θ: [0, π] → v: [0, height]
        u = ((phi + np.pi) / (2.0 * np.pi)) * equirect_w
        v = (theta / np.pi) * equirect_h
        
        # Special handling for back face to avoid seam artifacts
        if face_name == 'back':
            # Identify pixels near the seam (φ ≈ ±π)
            seam_threshold = 0.01
            near_seam = (np.abs(np.abs(phi) - np.pi) < seam_threshold * np.pi)
            
            # For pixels left of center, use φ = -π (left edge of equirect)
            left_half = (x_ndc < 0) & near_seam
            if np.any(left_half):
                u[left_half] = 0
            
            # For pixels right of center, use φ = π (right edge of equirect)
            right_half = (x_ndc >= 0) & near_seam
            if np.any(right_half):
                u[right_half] = equirect_w - 1
        
        # Create mapping arrays for OpenCV remap
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        
        # Handle edge cases
        map_x = np.remainder(map_x, equirect_w)  # Proper wrapping for longitude
        map_y = np.clip(map_y, 0, equirect_h - 1)  # Clamp to valid range
        
        # Apply remapping with high-quality interpolation
        face = cv2.remap(
            equirect_img,
            map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,  # Highest quality interpolation
            borderMode=cv2.BORDER_WRAP  # Proper wrapping for longitude
        )
        
        # Store the face
        faces[face_name] = face
    
    # Apply edge consistency fixes
    faces = fix_cubemap_edge_artifacts(faces)
    
    return faces


def fix_cubemap_edge_artifacts(cube_faces):
    """
    Fix artifacts at cube face edges and seams by applying
    specialized filtering and edge blending.
    
    Args:
        cube_faces: Dictionary of 6 cubemap faces
        
    Returns:
        Dictionary of fixed faces with reduced artifacts
    """
    import numpy as np
    import cv2
    
    fixed_faces = {k: v.copy() for k, v in cube_faces.items()}
    face_size = fixed_faces['front'].shape[0]
    
    # 1. Special processing for back face seam which has the most issues
    if 'back' in fixed_faces:
        back_face = fixed_faces['back']
        
        # Create a center vertical seam mask
        center_x = face_size // 2
        seam_width = max(3, face_size // 48)  # Thin but effective seam width
        
        # Create vertical gradient mask with smooth falloff
        seam_mask = np.zeros((face_size, face_size), dtype=np.float32)
        for x in range(max(0, center_x - seam_width), min(face_size, center_x + seam_width)):
            # Calculate normalized distance from center [0,1]
            dist = abs(x - center_x) / seam_width
            if dist < 1.0:
                # Apply smoothstep for smooth falloff: 3t² - 2t³
                t = 1.0 - dist
                weight = t * t * (3 - 2 * t)
                seam_mask[:, x] = weight
        
        # Apply multi-stage filtering for high-quality results
        
        # First, bilateral filter preserves edges while reducing noise
        bilateral = cv2.bilateralFilter(
            back_face,
            d=5,           # Diameter
            sigmaColor=25, # Color sigma
            sigmaSpace=5   # Spatial sigma
        )
        
        # Apply directional (vertical) filter to better handle the seam
        kernel_v = np.ones((5, 1), np.float32) / 5
        vert_smoothed = cv2.filter2D(bilateral, -1, kernel_v)
        
        # Blend multiple processed versions at the seam region
        seam_mask_3c = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)
        blend_factor = 0.8  # 80% filtered, 20% original
        
        # Apply blended result
        fixed_faces['back'] = (
            (seam_mask_3c * blend_factor) * vert_smoothed + 
            (1 - (seam_mask_3c * blend_factor)) * back_face
        ).astype(np.uint8)
    
    # 2. Fix top and bottom face distortion (common in equirectangular projections)
    for face_name in ['top', 'bottom']:
        if face_name in fixed_faces:
            face = fixed_faces[face_name]
            
            # Apply subtle bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(face, d=3, sigmaColor=15, sigmaSpace=2)
            
            # Apply very subtle edge enhancement to counteract blur
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  2.0, -0.1],
                              [-0.1, -0.1, -0.1]]) / 1.2
            
            enhanced = cv2.filter2D(filtered, -1, kernel)
            
            # Mix original with enhanced (95% original, 5% enhanced)
            fixed_faces[face_name] = (0.95 * filtered + 0.05 * enhanced).astype(np.uint8)
    
    # 3. Apply very subtle edge consistency between adjacent faces
    # Define the face adjacency relationships
    adjacency = [
        ('front', 'right'),
        ('right', 'back'),
        ('back', 'left'),
        ('left', 'front'),
        ('top', 'front'),
        ('top', 'right'),
        ('top', 'back'),
        ('top', 'left'),
        ('bottom', 'front'),
        ('bottom', 'right'),
        ('bottom', 'back'),
        ('bottom', 'left')
    ]
    
    # For each pair of adjacent faces, apply very subtle edge blending
    edge_width = max(1, face_size // 128)  # Very thin edge region
    
    for face1, face2 in adjacency:
        if face1 not in fixed_faces or face2 not in fixed_faces:
            continue
        
        # Apply minimal blending at edges (3% blend)
        # This is just enough to reduce visible seams without affecting content
        blend_factor = 0.03
        
        # Actual edge blending would depend on the specific edge being processed
        # For simplicity, we'll just apply a very subtle blur at the edges
        f1 = fixed_faces[face1]
        f2 = fixed_faces[face2]
        
        # Apply subtle blur at edges
        f1_edge = cv2.GaussianBlur(f1, (3, 3), 0.5)
        f2_edge = cv2.GaussianBlur(f2, (3, 3), 0.5)
        
        # Create edge masks based on face pairing
        f1_mask = np.zeros_like(f1, dtype=np.float32)
        f2_mask = np.zeros_like(f2, dtype=np.float32)
        
        # Different treatment based on face arrangement
        # (This is simplified - actual implementation would handle each edge specifically)
        if face1 == 'front' and face2 == 'right':
            # Right edge of front connects to left edge of right
            f1_mask[:, -edge_width:] = blend_factor
            f2_mask[:, :edge_width] = blend_factor
            
        # Apply edge blur only at the masked regions
        fixed_faces[face1] = np.where(f1_mask > 0, 
                                     (1-blend_factor) * f1 + blend_factor * f1_edge, 
                                     f1).astype(np.uint8)
        fixed_faces[face2] = np.where(f2_mask > 0, 
                                     (1-blend_factor) * f2 + blend_factor * f2_edge, 
                                     f2).astype(np.uint8)
    
    return fixed_faces


def cubemap_to_equirect(cube_faces, height, width, original_equirect=None):
    """
    Convert cubemap faces to equirectangular panorama with high precision
    and artifact reduction.
    
    Face arrangement follows standard cubemap order:
    - Front:  +X face (positive X direction)
    - Right:  +Y face (positive Y direction)
    - Back:   -X face (negative X direction)
    - Left:   -Y face (negative Y direction)
    - Top:    +Z face (positive Z direction)
    - Bottom: -Z face (negative Z direction)
    
    Args:
        cube_faces: Dictionary or array of 6 cubemap faces
        height: Height of output equirectangular image
        width: Width of output equirectangular image
        original_equirect: Optional reference image for color calibration
        
    Returns:
        Equirectangular image with minimal artifacts
    """
    import numpy as np
    import cv2
    from scipy.ndimage import map_coordinates
    
    # Process input format
    if isinstance(cube_faces, dict):
        face_size = cube_faces['front'].shape[0]
        # Stack faces in the correct order
        faces_array = np.stack([
            cube_faces['front'],  # +X
            cube_faces['right'],  # +Y
            cube_faces['back'],   # -X
            cube_faces['left'],   # -Y
            cube_faces['top'],    # +Z
            cube_faces['bottom']  # -Z
        ])
    else:
        face_size = cube_faces[0].shape[0]
        faces_array = np.stack(cube_faces)
    
    faces_array = faces_array.astype(np.float32)
    
    # Create spherical coordinate grid for output equirectangular image
    phi = np.linspace(-np.pi, np.pi, width, dtype=np.float64)
    theta = np.linspace(0, np.pi, height, dtype=np.float64)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Convert to Cartesian coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # Initialize arrays for face indices and coordinates
    face_idx = np.zeros((height, width), dtype=np.int32)
    u_norm = np.zeros((height, width), dtype=np.float64)
    v_norm = np.zeros((height, width), dtype=np.float64)
    
    # Small epsilon to avoid division by zero
    eps = 1e-10
    
    # Determine which face each ray intersects
    # Find max component of the vector
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    max_comp = np.maximum(np.maximum(abs_x, abs_y), abs_z)
    
    # CRITICAL: Map to the correct cube faces according to the standard orientation
    
    # Front face (+X)
    mask = (abs_x == max_comp) & (x > 0)
    face_idx[mask] = 0
    u_norm[mask] = -y[mask] / (x[mask] + eps)
    v_norm[mask] = -z[mask] / (x[mask] + eps)
    
    # Right face (+Y)
    mask = (abs_y == max_comp) & (y > 0)
    face_idx[mask] = 1
    u_norm[mask] = x[mask] / (y[mask] + eps)
    v_norm[mask] = -z[mask] / (y[mask] + eps)
    
    # Back face (-X)
    mask = (abs_x == max_comp) & (x <= 0)
    face_idx[mask] = 2
    u_norm[mask] = y[mask] / (-x[mask] + eps)
    v_norm[mask] = -z[mask] / (-x[mask] + eps)
    
    # Left face (-Y)
    mask = (abs_y == max_comp) & (y <= 0)
    face_idx[mask] = 3
    u_norm[mask] = -x[mask] / (-y[mask] + eps)
    v_norm[mask] = -z[mask] / (-y[mask] + eps)
    
    # Top face (+Z)
    mask = (abs_z == max_comp) & (z > 0)
    face_idx[mask] = 4
    u_norm[mask] = x[mask] / (z[mask] + eps)
    v_norm[mask] = y[mask] / (z[mask] + eps)
    
    # Bottom face (-Z)
    mask = (abs_z == max_comp) & (z <= 0)
    face_idx[mask] = 5
    u_norm[mask] = x[mask] / (-z[mask] + eps)
    v_norm[mask] = -y[mask] / (-z[mask] + eps)
    
    # Convert normalized coordinates [-1,1] to pixel coordinates [0,size-1]
    # Apply precise pixel center alignment for better sampling
    u_pixel = (u_norm + 1.0) * (face_size - 1) / 2.0
    v_pixel = (v_norm + 1.0) * (face_size - 1) / 2.0
    
    # Initialize output
    equirect = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create mask for seam region (around φ = ±π) for special processing
    seam_mask = np.abs(np.abs(phi_grid) - np.pi) < 0.01
    
    # High-quality sampling from each face
    for f in range(6):
        mask = (face_idx == f)
        if not np.any(mask):
            continue
        
        # Get coordinates for this face
        u_coords = u_pixel[mask]
        v_coords = v_pixel[mask]
        y_idx, x_idx = np.where(mask)
        
        # Stack for map_coordinates
        coords = np.vstack((v_coords, u_coords))
        
        # Sample with high-quality bicubic interpolation
        for c in range(3):
            sampled = map_coordinates(
                faces_array[f, :, :, c],
                coords,
                order=3,        # Bicubic interpolation
                mode='nearest',  # Use nearest at boundaries
                prefilter=True   # Apply prefilter for higher quality
            )
            equirect[y_idx, x_idx, c] = sampled
    
    # Apply special processing for seam regions
    if np.any(seam_mask):
        # Apply bilateral filter to seam region to preserve edges while reducing artifacts
        equirect_uint8 = np.clip(equirect, 0, 255).astype(np.uint8)
        filtered = equirect_uint8.copy()
        
        for y in range(0, height, 50):  # Process in chunks
            chunk_height = min(50, height - y)
            chunk = equirect_uint8[y:y+chunk_height]
            filtered_chunk = cv2.bilateralFilter(chunk, d=5, sigmaColor=20, sigmaSpace=3)
            filtered[y:y+chunk_height] = filtered_chunk
        
        # Apply only to seam region with smooth blend
        seam_mask_3c = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)
        blend_factor = 0.7  # 70% filtered, 30% original
        equirect = np.where(seam_mask_3c, 
                           blend_factor * filtered + (1 - blend_factor) * equirect_uint8, 
                           equirect).astype(np.float32)
    
    # Fix wrap-around (left/right edges)
    edge_width = width // 64
    for y in range(height):
        left_edge = equirect[y, :edge_width].copy()
        right_edge = equirect[y, -edge_width:].copy()
        
        for x in range(edge_width):
            # Smoothstep blend
            t = x / (edge_width - 1)
            weight = t * t * (3 - 2 * t)
            
            # Apply blending
            equirect[y, x] = (1 - weight) * left_edge[x] + weight * right_edge[0]
            equirect[y, width-edge_width+x] = weight * left_edge[0] + (1 - weight) * right_edge[x]
    
    # Apply a final subtle bilateral filter for consistent image quality
    # This preserves edges while reducing remaining noise/artifacts
    equirect_uint8 = np.clip(equirect, 0, 255).astype(np.uint8)
    final_equirect = cv2.bilateralFilter(equirect_uint8, d=3, sigmaColor=10, sigmaSpace=3)
    
    return final_equirect


def diagnose_conversion_issues(original_equirect, reconstructed_equirect):
    """
    Perform detailed diagnostics of equirectangular conversion issues
    to identify specific problem areas and potential fixes.
    
    Args:
        original_equirect: Original equirectangular image
        reconstructed_equirect: Reconstructed equirectangular image
        
    Returns:
        Dictionary of diagnostic information
    """
    import numpy as np
    import cv2
    
    # Calculate basic difference metrics
    diff = cv2.absdiff(original_equirect, reconstructed_equirect)
    diff_norm = np.sqrt(np.sum(diff**2, axis=2) / 3)  # RMS difference
    
    # Create spherical coordinates grid
    h, w = original_equirect.shape[:2]
    phi = np.linspace(-np.pi, np.pi, w)
    theta = np.linspace(0, np.pi, h)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Identify problem areas by longitude (φ) and latitude (θ)
    # Check for seam at φ = ±π
    seam_mask = (np.abs(np.abs(phi_grid) - np.pi) < 0.1)
    seam_diff = np.mean(diff_norm[seam_mask]) if np.any(seam_mask) else 0
    
    # Check poles (θ ≈ 0 or θ ≈ π)
    north_pole_mask = (theta_grid < 0.1 * np.pi)
    south_pole_mask = (theta_grid > 0.9 * np.pi)
    north_pole_diff = np.mean(diff_norm[north_pole_mask]) if np.any(north_pole_mask) else 0
    south_pole_diff = np.mean(diff_norm[south_pole_mask]) if np.any(south_pole_mask) else 0
    
    # Check equator region (θ ≈ π/2)
    equator_mask = (np.abs(theta_grid - np.pi/2) < 0.1 * np.pi)
    equator_diff = np.mean(diff_norm[equator_mask]) if np.any(equator_mask) else 0
    
    # Check for specific longitude bands
    longitude_bands = {
        'Front': (-np.pi/4, np.pi/4),       # Front face region
        'Right': (np.pi/4, 3*np.pi/4),      # Right face region
        'Back': (3*np.pi/4, np.pi),         # Back face region (positive)
        'Back_Neg': (-np.pi, -3*np.pi/4),   # Back face region (negative)
        'Left': (-3*np.pi/4, -np.pi/4)      # Left face region
    }
    
    longitude_diffs = {}
    for name, (phi_min, phi_max) in longitude_bands.items():
        if name == 'Back_Neg':
            mask = (phi_grid >= phi_min) & (phi_grid <= phi_max)
        else:
            mask = (phi_grid >= phi_min) & (phi_grid <= phi_max)
        
        if np.any(mask):
            longitude_diffs[name] = np.mean(diff_norm[mask])
    
    # Combine Back and Back_Neg
    if 'Back' in longitude_diffs and 'Back_Neg' in longitude_diffs:
        back_pixels = np.sum((phi_grid >= longitude_bands['Back'][0]) & 
                            (phi_grid <= longitude_bands['Back'][1]))
        back_neg_pixels = np.sum((phi_grid >= longitude_bands['Back_Neg'][0]) & 
                                (phi_grid <= longitude_bands['Back_Neg'][1]))
        
        total_pixels = back_pixels + back_neg_pixels
        if total_pixels > 0:
            longitude_diffs['Back'] = (
                (longitude_diffs['Back'] * back_pixels + 
                longitude_diffs['Back_Neg'] * back_neg_pixels) / total_pixels
            )
        
        # Remove the separate negative part
        longitude_diffs.pop('Back_Neg', None)
    
    # Find the most problematic areas
    max_long_diff = max(longitude_diffs.items(), key=lambda x: x[1]) if longitude_diffs else (None, 0)
    
    problem_areas = {}
    for name, diff_val in sorted(longitude_diffs.items(), key=lambda x: x[1], reverse=True):
        problem_areas[name] = diff_val
    
    # Create histogram of differences
    diff_hist, diff_bins = np.histogram(diff_norm.flatten(), bins=100, range=(0, 20))
    diff_hist = diff_hist / diff_norm.size * 100  # Convert to percentage
    
    # Create diagnosis report
    diagnoses = []
    
    if seam_diff > 8:
        diagnoses.append("Severe seam issue at longitude ±π (back face center). "
                        "Apply specialized seam handling.")
    
    if north_pole_diff > 8 or south_pole_diff > 8:
        diagnoses.append("Pole distortion detected. Improve sampling near poles (top/bottom faces).")
    
    if problem_areas and max_long_diff[0] == 'Back' and max_long_diff[1] > 8:
        diagnoses.append("Back face has significantly higher error. "
                        "Improve back face handling.")
    
    # Calculate summary statistics
    diagnostic_info = {
        'Seam_Diff': seam_diff,
        'North_Pole_Diff': north_pole_diff,
        'South_Pole_Diff': south_pole_diff,
        'Equator_Diff': equator_diff,
        'Longitude_Diffs': longitude_diffs,
        'Problem_Areas': problem_areas,
        'Diff_Histogram': {
            'bins': diff_bins.tolist()[:-1],  # Remove last bin edge
            'values': diff_hist.tolist()
        },
        'Diagnoses': diagnoses
    }
    
    # Print diagnosis
    print("\nConversion Diagnosis:")
    print(f"Seam at longitude ±π: {seam_diff:.2f} average difference")
    print(f"North pole region: {north_pole_diff:.2f} average difference")
    print(f"South pole region: {south_pole_diff:.2f} average difference")
    print(f"Equator region: {equator_diff:.2f} average difference")
    
    print("\nLongitude band analysis:")
    for name, diff_val in sorted(longitude_diffs.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {diff_val:.2f} average difference")
    
    print("\nDiagnosed issues:")
    if diagnoses:
        for i, diagnosis in enumerate(diagnoses):
            print(f"{i+1}. {diagnosis}")
    else:
        print("No major issues detected.")
    
    return diagnostic_info

# --------------------------------------------

def fix_cubemap_back_face(cube_faces):
    """
    Fix the back face of a cubemap by completely replacing it with a regenerated version
    based on the right and left faces.
    
    Args:
        cube_faces: Dictionary of 6 cubemap faces (front, right, back, left, top, bottom)
    
    Returns:
        Updated cube_faces dictionary with fixed back face
    """
    import numpy as np
    import cv2
    
    # If it's not a dictionary, convert to one
    if not isinstance(cube_faces, dict):
        face_size = cube_faces[0].shape[0]
        face_dict = {
            'front': cube_faces[0],
            'right': cube_faces[1],
            'back': cube_faces[2],
            'left': cube_faces[3],
            'top': cube_faces[4],
            'bottom': cube_faces[5]
        }
        cube_faces = face_dict
    
    face_size = cube_faces['front'].shape[0]
    
    # Create a new back face from scratch
    new_back = np.zeros_like(cube_faces['back'])
    
    # Create grid for the back face
    y_coords, x_coords = np.meshgrid(
        np.arange(face_size), np.arange(face_size), indexing='ij'
    )
    
    # Normalize coordinates to [-1, 1]
    x_ndc = 2.0 * x_coords / (face_size - 1) - 1.0
    y_ndc = 2.0 * y_coords / (face_size - 1) - 1.0
    
    # For each pixel in the back face, determine where to sample from
    for y in range(face_size):
        for x in range(face_size):
            # Current normalized coordinates
            x_norm = x_ndc[y, x]
            y_norm = y_ndc[y, x]
            
            # We'll use right face for the left half of back face
            # and left face for the right half of back face
            if x_norm < 0:
                # Sample from right face
                # Convert back face coordinates to right face coordinates
                # Back(-1, y, z) -> Right(-y, -1, z)
                right_x = int((1-y_norm) * 0.5 * (face_size - 1))
                right_y = int((1+y_norm) * 0.5 * (face_size - 1))
                
                # Ensure coordinates are within bounds
                right_x = max(0, min(face_size-1, right_x))
                right_y = max(0, min(face_size-1, right_y))
                
                new_back[y, x] = cube_faces['right'][right_y, right_x]
            else:
                # Sample from left face
                # Convert back face coordinates to left face coordinates
                # Back(-1, y, z) -> Left(y, 1, z)
                left_x = int((1-y_norm) * 0.5 * (face_size - 1))
                left_y = int((1+y_norm) * 0.5 * (face_size - 1))
                
                # Ensure coordinates are within bounds
                left_x = max(0, min(face_size-1, left_x))
                left_y = max(0, min(face_size-1, left_y))
                
                new_back[y, x] = cube_faces['left'][left_y, left_x]
    
    # Apply Gaussian blurring to smooth the boundary between the two halves
    blurred = cv2.GaussianBlur(new_back, (5, 5), 0)
    
    # Create a blend mask that's strongest at the center (where the two halves meet)
    blend_mask = np.zeros((face_size, 1))
    center_width = face_size // 4
    for y in range(face_size):
        for x in range(max(0, face_size//2 - center_width), min(face_size, face_size//2 + center_width)):
            # Calculate distance from center line
            dist = abs(x - face_size//2)
            # Normalize to [0, 1] and invert so it's 1 at center, 0 at edges
            weight = 1.0 - dist / center_width
            blend_mask[y, 0] = max(blend_mask[y, 0], weight)
    
    # Apply blend mask
    for y in range(face_size):
        for x in range(face_size):
            weight = blend_mask[y, 0] * 0.8  # 80% blending at maximum
            new_back[y, x] = (1-weight) * new_back[y, x] + weight * blurred[y, x]
    
    # Finally, apply a bilateral filter to preserve edges while reducing noise
    new_back = cv2.bilateralFilter(new_back, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Update the back face
    cube_faces['back'] = new_back
    
    return cube_faces

def fix_back_banding(back_face):
    """
    Fix horizontal banding artifacts in a cubemap back face.
    
    Args:
        back_face: The back face image with banding artifacts
    
    Returns:
        Fixed back face image
    """
    import cv2
    import numpy as np
    
    h, w = back_face.shape[:2]
    fixed_face = back_face.copy()
    
    # 1. Detect horizontal bands by looking for abrupt row-to-row changes
    row_diffs = np.mean(np.abs(np.diff(back_face, axis=0)), axis=(1, 2))
    
    # Identify rows with significant differences (potential bands)
    threshold = np.percentile(row_diffs, 95)  # Adapt threshold to image content
    band_rows = np.where(row_diffs > threshold)[0]
    
    if len(band_rows) > 0:
        # 2. For each detected band, apply targeted smoothing
        for row in band_rows:
            # Create a neighborhood for smoothing (5 rows above and below)
            start = max(0, row - 5)
            end = min(h - 1, row + 6)
            
            # Apply horizontal-only median filter to the region
            region = fixed_face[start:end, :, :]
            for c in range(3):
                for y in range(region.shape[0]):
                    # Apply median filter along horizontal direction
                    region[y, :, c] = cv2.medianBlur(region[y, :, c].reshape(-1, 1), 5).reshape(-1)
            
            fixed_face[start:end, :, :] = region
    
    # 3. Apply a bilateral filter to maintain edges while smoothing noise
    fixed_face = cv2.bilateralFilter(fixed_face, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 4. Apply a second pass of bilateral filtering with different parameters
    fixed_face = cv2.bilateralFilter(fixed_face, d=9, sigmaColor=75, sigmaSpace=75)
    
    return fixed_face

# Fix for generating the proper cubemap layout where back is connected to top vertically
def create_cubemap_layout(cube_faces, with_labels=True):
    """
    Layout is:
          [T]
    [L]   [F]   [R]   [B]
          [Bo]
    
    Where F=Front, R=Right, B=Back, L=Left, T=Top, Bo=Bottom
    
    Create a proper cubemap layout with "back" connected to "top" vertically.
    
    Args:
        cube_faces: Dictionary or array of cubemap faces
        with_labels: Whether to add labels to the faces
        
    Returns:
        Cubemap layout as a single image

    """
    import numpy as np
    import cv2  # For text drawing
    
    # Extract faces from dictionary or array
    if isinstance(cube_faces, dict):
        front = cube_faces['front'].copy()  # Make copies to avoid modifying originals
        right = cube_faces['right'].copy()
        back = cube_faces['back'].copy()
        left = cube_faces['left'].copy()
        top = cube_faces['top'].copy()
        bottom = cube_faces['bottom'].copy()
    else:
        front = cube_faces[0].copy()
        right = cube_faces[1].copy() 
        back = cube_faces[2].copy()
        left = cube_faces[3].copy()
        top = cube_faces[4].copy()
        bottom = cube_faces[5].copy()
    
    # Get face dimensions
    face_h, face_w = front.shape[:2]
    
    # Add labels if requested
    if with_labels:
        # Define function to add label
        def add_label(img, text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)  # White text
            
            # Get text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Position text in center
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = (img.shape[0] + text_size[1]) // 2
            
            # Add text
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
            return img
        
        # Add labels to each face
        front = add_label(front, "Front")
        right = add_label(right, "Right")
        back = add_label(back, "Back")
        left = add_label(left, "Left")
        top = add_label(top, "Top")
        bottom = add_label(bottom, "Bottom")
    
    # Create the layout (3 rows x 4 columns)
    layout = np.zeros((3 * face_h, 4 * face_w, 3), dtype=front.dtype)
    
    # Fill in the layout
    # Top row: empty, top, empty, empty
    layout[0:face_h, face_w:2*face_w] = top
    
    # Middle row: left, front, right, back
    layout[face_h:2*face_h, 0:face_w] = left
    layout[face_h:2*face_h, face_w:2*face_w] = front
    layout[face_h:2*face_h, 2*face_w:3*face_w] = right
    layout[face_h:2*face_h, 3*face_w:4*face_w] = back
    
    # Bottom row: empty, bottom, empty, empty
    layout[2*face_h:3*face_h, face_w:2*face_w] = bottom
    
    return layout


def check_face_order(cube_faces):
    """
    Check and print the order of cubemap faces.
    
    Args:
        cube_faces: Dictionary or array of 6 cubemap faces
        
    Returns:
        True if faces are in the expected order
    """
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    print("Cubemap face order:")
    
    # Check if cube_faces is a dictionary
    if isinstance(cube_faces, dict):
        for i, name in enumerate(face_names):
            if name in cube_faces:
                shape_str = str(cube_faces[name].shape)
                print(f"  {i}: {name} - Shape: {shape_str}")
            else:
                print(f"  {i}: {name} - MISSING")
    # If cube_faces is a list, tuple, array, or tensor
    elif hasattr(cube_faces, '__getitem__') and hasattr(cube_faces, '__len__'):
        for i, name in enumerate(face_names):
            if i < len(cube_faces):
                shape_str = str(cube_faces[i].shape)
                print(f"  {i}: {name} - Shape: {shape_str}")
            else:
                print(f"  {i}: {name} - MISSING")
    else:
        print("Error: cube_faces must be a dictionary or array-like object")
        return False
    
    return True

def create_informative_cube_visualization(cube_faces, scale=1.0):
    """
    Create an enhanced 3D visualization of a cubemap with more informative elements.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        scale: Scaling factor for the visualization (default: 1.0)
        
    Returns:
        fig: Matplotlib 3D figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Verify we have exactly 6 faces
    if len(cube_faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(cube_faces)}")
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define vertices of the cube
    vertices = np.array([
        [-1, -1, -1],  # 0: front-bottom-left
        [1, -1, -1],   # 1: front-bottom-right
        [1, 1, -1],    # 2: front-top-right
        [-1, 1, -1],   # 3: front-top-left
        [-1, -1, 1],   # 4: back-bottom-left
        [1, -1, 1],    # 5: back-bottom-right
        [1, 1, 1],     # 6: back-top-right
        [-1, 1, 1]     # 7: back-top-left
    ]) * scale
    
    # Define edges of the cube for wireframe visualization
    edges = [
        # Front face
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        # Back face
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        # Connecting edges
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]
    
    # Define the six faces of the cube using vertex indices
    # Order: Front, Right, Back, Left, Top, Bottom
    face_indices = [
        [0, 1, 2, 3],  # Front (negative Z)
        [1, 5, 6, 2],  # Right (positive X)
        [5, 4, 7, 6],  # Back (positive Z)
        [4, 0, 3, 7],  # Left (negative X)
        [3, 2, 6, 7],  # Top (positive Y)
        [0, 4, 5, 1]   # Bottom (negative Y)
    ]
    
    # Face labels with coordinate axes
    face_labels = [
        'Front (-Z)', 'Right (+X)', 'Back (+Z)', 'Left (-X)', 'Top (-Y)', 'Bottom (+Y)'
    ]
    
    # Face colors
    face_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    # Draw coordinate axes with labels
    axis_length = 1.5 * scale
    
    # X-axis
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax.text(axis_length * 1.1, 0, 0, "X", color='red', fontsize=12)
    
    # Y-axis
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    ax.text(0, axis_length * 1.1, 0, "Y", color='green', fontsize=12)
    
    # Z-axis
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
    ax.text(0, 0, axis_length * 1.1, "Z", color='blue', fontsize=12)
    
    # Draw wireframe cube
    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], 
                [edge[0][1], edge[1][1]], 
                [edge[0][2], edge[1][2]], 'k-', alpha=0.3)
    
    # Add vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=30)
    
    # Add face indices to vertices
    for i, idx in enumerate(vertices):
        ax.text(idx[0], idx[1], idx[2], str(i), fontsize=10, ha='center')
    
    # Draw each face with its label
    for i, (face_idx, face_img, label, color) in enumerate(zip(face_indices, cube_faces, face_labels, face_colors)):
        # Extract vertices for this face
        verts = [vertices[idx] for idx in face_idx]
        
        # Create a polygon
        poly = Poly3DCollection([verts], alpha=0.7)
        
        # Set face color
        poly.set_facecolor(color)
        poly.set_edgecolor('black')
        
        # Add to axes
        ax.add_collection3d(poly)
        
        # Add face label at the center of the face
        face_center = np.mean(verts, axis=0)
        ax.text(face_center[0], face_center[1], face_center[2], label, 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=10, color='black', fontweight='bold')
    
    # Add explanatory annotations
    ax.text(0, 0, -2*scale, "Cubemap Projection System:", 
            fontsize=12, horizontalalignment='center')
    ax.text(0, 0, -2.2*scale, "• Each face is a perspective view with 90° FOV", 
            fontsize=10, horizontalalignment='center')
    ax.text(0, 0, -2.4*scale, "• Vertices are numbered 0-7", 
            fontsize=10, horizontalalignment='center')
    ax.text(0, 0, -2.6*scale, "• Faces are colored to show their position", 
            fontsize=10, horizontalalignment='center')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title and limits
    ax.set_title('3D Cubemap Projection System', fontsize=16)
    
    # Set limits to ensure everything is visible
    limit = 2.0 * scale
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    return fig

def create_cubemap_masks(height, width, device=None):
    """
    Create binary masks for cubemap faces to handle overlapping regions.
    
    Args:
        height: Height of each cubemap face
        width: Width of each cubemap face
        device: Optional torch device for the masks
        
    Returns:
        Tensor of masks for each face with shape (6, 1, height, width)
    """
    import torch
    
    # Create masks for each face
    masks = torch.ones(6, 1, height, width, dtype=torch.float32, device=device)
    
    # Create coordinate grid
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Create a mask that fades out near the edges
    # This helps with seamless transitions between faces
    edge_distance = torch.min(
        torch.min(1 - torch.abs(grid_x), 1 - torch.abs(grid_y)),
        torch.ones_like(grid_x) * 0.2
    )
    
    # Apply smooth falloff near edges
    falloff = torch.nn.functional.sigmoid(edge_distance * 10) * 0.9 + 0.1
    
    # Apply to all faces
    for i in range(6):
        masks[i, 0] = falloff
    
    return masks

def prepare_cubemap_input(image, input_size=None, device=None):
    """
    Prepare an input image for the CubeDiff model.
    
    Args:
        image: Input image (RGB, 0-255)
        input_size: Optional target size for the image
        device: Optional torch device
        
    Returns:
        Normalized image tensor ready for the model
    """
    import torch
    import torch.nn.functional as F
    
    # Default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if it's a numpy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Move to the correct device
    image = image.to(device)
    
    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Resize if needed
    if input_size is not None:
        image = F.interpolate(image, size=input_size, mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1] for model input
    image = 2 * image - 1
    
    return image

def postprocess_cubemap(faces, output_size=None):
    """
    Post-process cubemap faces generated by the model.
    
    Args:
        faces: Tensor of cubemap faces
        output_size: Optional size for the output faces
        
    Returns:
        Processed cubemap faces as numpy arrays
    """
    import torch
    import torch.nn.functional as F
    
    # Clone to avoid modifying the original
    faces = faces.clone()
    
    # Ensure format is correct
    if len(faces.shape) == 5:
        # (B, F, C, H, W) -> assuming batch size 1
        faces = faces.squeeze(0)
    
    # Resize if needed
    if output_size is not None:
        faces = F.interpolate(faces, size=output_size, mode='bilinear', align_corners=False)
    
    # Convert to numpy and correct format
    # First normalize to [0, 1]
    faces = (faces + 1) / 2
    faces = torch.clamp(faces, 0, 1)
    
    # Convert to numpy and change from (F, C, H, W) to list of (H, W, C)
    faces_np = []
    for i in range(faces.shape[0]):
        face = faces[i].permute(1, 2, 0).cpu().numpy()
        face = (face * 255).astype(np.uint8)
        faces_np.append(face)
    
    return faces_np

def synchronized_group_norm(features_list, eps=1e-5):
    """
    Apply synchronized group normalization across multiple cubemap faces.
    
    Args:
        features_list: List of feature tensors for each face
        eps: Small constant for numerical stability
        
    Returns:
        List of normalized feature tensors
    """
    import torch
    
    # Calculate statistics across all faces
    all_features = torch.cat(features_list, dim=0)
    mean = all_features.mean(dim=(2, 3), keepdim=True)
    var = all_features.var(dim=(2, 3), keepdim=True, unbiased=False)
    
    # Normalize each face with the shared statistics
    normalized_features = []
    for features in features_list:
        normalized = (features - mean) / torch.sqrt(var + eps)
        normalized_features.append(normalized)
    
    return normalized_features

def compare_equirectangular(equirect1, equirect2, titles=None):
    """
    Compare two equirectangular images side by side.
    
    Args:
        equirect1: First equirectangular image
        equirect2: Second equirectangular image
        titles: Optional pair of titles for the two images
        
    Returns:
        fig: Matplotlib figure
    """
    if titles is None:
        titles = ['Equirectangular 1', 'Equirectangular 2']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # First equirectangular
    if len(equirect1.shape) > 2:
        axes[0].imshow(equirect1)
    else:
        axes[0].imshow(equirect1, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    # Second equirectangular
    if len(equirect2.shape) > 2:
        axes[1].imshow(equirect2)
    else:
        axes[1].imshow(equirect2, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def debug_cube_faces(cube_faces):
    """
    Debug function to print detailed information about cube_faces.
    
    Args:
        cube_faces: Dictionary or array of cubemap faces
    """
    print("Type of cube_faces:", type(cube_faces))
    
    if isinstance(cube_faces, dict):
        print("cube_faces is a dictionary with keys:", list(cube_faces.keys()))
        for key, value in cube_faces.items():
            print(f"  {key}: type={type(value)}, ", end="")
            if isinstance(value, np.ndarray):
                print(f"shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"value={value}")
    else:
        print("cube_faces is a sequence with length:", len(cube_faces))
        for i, face in enumerate(cube_faces):
            print(f"  {i}: type={type(face)}, ", end="")
            if isinstance(face, np.ndarray):
                print(f"shape={face.shape}, dtype={face.dtype}")
            else:
                print(f"value={face}")

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Compatible with standard OpenCV installations.
    
    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        
    Returns:
        SSIM value
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def visualize_mse(original, reconstructed):
    """
    Enhanced visualization of MSE with percentage-based histogram and industry-standard metrics.
    
    Args:
        original: Original equirectangular image
        reconstructed: Reconstructed equirectangular image
        
    Returns:
        Dictionary of quality metrics
    """
    # Calculate difference and MSE
    diff = cv2.absdiff(original, reconstructed)
    squared_diff = diff.astype(np.float32)**2
    mse = np.mean(squared_diff)
    psnr = 10 * np.log10((255**2) / max(mse, 1e-10))
    
    # Calculate SSIM using a simplified approach
    def calculate_ssim(img1, img2):
        # Convert to grayscale
        if len(img1.shape) > 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) > 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Constants for stability
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Calculate means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        # Calculate variances and covariance
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    # Calculate SSIM
    ssim = calculate_ssim(original, reconstructed)
    
    # Create normalized difference for visualization
    diff_norm = np.sqrt(np.sum(diff**2, axis=2) / 3)  # RMS difference across channels
    max_diff = np.max(diff_norm)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original - direct visualization without color conversion
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Equirectangular")
    axes[0, 0].axis('off')
    
    # Reconstructed - direct visualization without color conversion
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title("Reconstructed Equirectangular")
    axes[0, 1].axis('off')
    
    # Difference heatmap
    im = axes[1, 0].imshow(diff_norm, cmap='jet', vmin=0, vmax=max_diff)
    metrics_text = f"Difference Heatmap\nMSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}"
    axes[1, 0].set_title(metrics_text)
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0], label='Pixel Difference')
    
    # Histogram of differences as percentage
    total_pixels = diff_norm.size
    hist, bins = np.histogram(diff_norm.flatten(), bins=100)
    percentages = (hist / total_pixels) * 100
    
    axes[1, 1].bar(bins[:-1], percentages, width=np.diff(bins), color='skyblue', edgecolor='navy', alpha=0.7)
    axes[1, 1].set_title(f"Histogram of Pixel Differences")
    axes[1, 1].set_xlabel("Difference Value")
    axes[1, 1].set_ylabel("Percentage of Pixels (%)")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add cumulative distribution line
    ax2 = axes[1, 1].twinx()
    cumulative = np.cumsum(percentages)
    ax2.plot(bins[:-1], cumulative, 'r-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate additional metrics
    low_diff_pct = np.sum(diff_norm < 5) / total_pixels * 100
    medium_diff_pct = np.sum((diff_norm >= 5) & (diff_norm < 10)) / total_pixels * 100
    high_diff_pct = np.sum(diff_norm >= 10) / total_pixels * 100
    
    # Compile all metrics
    metrics = {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim,
        'Low_Diff_Percentage': low_diff_pct,
        'Medium_Diff_Percentage': medium_diff_pct,
        'High_Diff_Percentage': high_diff_pct
    }
    
    # Print summary of metrics
    print("Quality Metrics Summary:")
    print(f"MSE: {mse:.2f} (lower is better)")
    print(f"PSNR: {psnr:.2f} dB (higher is better, >30dB is good)")
    print(f"SSIM: {ssim:.4f} (closer to 1.0 is better)")
    print(f"Percentage of pixels with low difference (<5): {low_diff_pct:.2f}%")
    print(f"Percentage of pixels with medium difference (5-10): {medium_diff_pct:.2f}%")
    print(f"Percentage of pixels with high difference (>10): {high_diff_pct:.2f}%")
    
    return metrics

def analyze_conversion_quality(original_equirect, face_size=512):
    """
    Comprehensive analysis of cubemap conversion quality with industry-standard metrics.
    Color preservation is maintained throughout the analysis process.
    
    Args:
        original_equirect: Original equirectangular image
        face_size: Size of cubemap faces to use
        
    Returns:
        Dictionary of quality metrics
    """
    # Convert to cubemap without color manipulation
    cube_faces = equirect_to_cubemap(original_equirect, face_size)
    
    # Convert back to equirectangular
    reconstructed_equirect = cubemap_to_equirect(
        cube_faces, 
        original_equirect.shape[0], 
        original_equirect.shape[1]
    )
    
    # Calculate basic metrics without color manipulation
    metrics = visualize_mse(original_equirect, reconstructed_equirect)
    
    # Advanced metrics: Content-aware difference analysis
    # Convert to grayscale for edge detection (standard practice, doesn't affect original colors)
    original_gray = cv2.cvtColor(original_equirect, cv2.COLOR_BGR2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed_equirect, cv2.COLOR_BGR2GRAY)
    
    # Edge detection on grayscale images
    original_edges = cv2.Canny(original_gray, 50, 150)
    reconstructed_edges = cv2.Canny(reconstructed_gray, 50, 150)
    
    # Calculate edge preservation metric
    edge_difference = cv2.bitwise_xor(original_edges, reconstructed_edges)
    edge_preservation = 1.0 - np.sum(edge_difference) / max(1, np.sum(original_edges))
    
    # Create visualization of edge preservation
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_edges, cmap='gray')
    plt.title('Original Image Edges')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_edges, cmap='gray')
    plt.title('Reconstructed Image Edges')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(edge_difference, cmap='hot')
    plt.title(f'Edge Differences (Preservation: {edge_preservation:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze face-specific metrics
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    face_metrics = {}
    
    # Create back-projection mask for each face
    equirect_h, equirect_w = original_equirect.shape[:2]
    
    # Create spherical coordinates grid
    phi = np.linspace(-np.pi, np.pi, equirect_w)
    theta = np.linspace(0, np.pi, equirect_h)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Convert to cartesian coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # Initialize mask for tracking each face's contribution
    face_masks = {}
    
    # Determine which face each pixel belongs to
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    max_comp = np.maximum(np.maximum(abs_x, abs_y), abs_z)
    
    # Create masks for each face
    face_masks['front'] = (abs_x == max_comp) & (x > 0)
    face_masks['right'] = (abs_y == max_comp) & (y > 0)
    face_masks['back'] = (abs_x == max_comp) & (x <= 0)
    face_masks['left'] = (abs_y == max_comp) & (y <= 0)
    face_masks['top'] = (abs_z == max_comp) & (z > 0)
    face_masks['bottom'] = (abs_z == max_comp) & (z <= 0)
    
    # Calculate face-specific metrics
    for face_name in face_names:
        mask = face_masks[face_name]
        
        # Extract relevant regions from original and reconstructed
        orig_region = original_equirect[mask]
        recon_region = reconstructed_equirect[mask]
        
        # Calculate MSE
        squared_diff = np.mean((orig_region.astype(np.float32) - recon_region.astype(np.float32))**2)
        psnr = 10 * np.log10((255**2) / max(squared_diff, 1e-10))
        
        # Store metrics
        face_metrics[face_name] = {
            'MSE': squared_diff,
            'PSNR': psnr
        }
    
    # Print face-specific metrics
    print("\nFace-Specific Metrics:")
    for face_name, face_metric in face_metrics.items():
        print(f"{face_name.capitalize()} Face - MSE: {face_metric['MSE']:.2f}, PSNR: {face_metric['PSNR']:.2f} dB")
    
    # Check if back face has significantly worse metrics
    back_mse = face_metrics['back']['MSE']
    avg_other_mse = np.mean([m['MSE'] for f, m in face_metrics.items() if f != 'back'])
    
    if back_mse > avg_other_mse * 1.5:
        print("\nWARNING: Back face has significantly higher error compared to other faces.")
        print("Consider additional optimization for the back face conversion.")
    
    # Create consolidated results
    results = {
        'Basic_Metrics': metrics,
        'Advanced_Metrics': {
            'Edge_Preservation': edge_preservation
        },
        'Face_Metrics': face_metrics
    }
    
    # Create a focused visualization of the back face region
    plt.figure(figsize=(18, 6))
    
    # Create a difference mask specific to back face
    diff_img = cv2.absdiff(original_equirect, reconstructed_equirect)
    diff_intensity = np.sum(diff_img, axis=2) / 3
    
    back_diff = np.zeros_like(diff_intensity)
    back_diff[face_masks['back']] = diff_intensity[face_masks['back']]
    
    # Visualize back face in original and reconstructed - WITHOUT color conversion
    plt.subplot(1, 3, 1)
    back_vis_orig = np.zeros_like(original_equirect)
    back_vis_orig[face_masks['back']] = original_equirect[face_masks['back']]
    plt.imshow(back_vis_orig)  # No color conversion
    plt.title('Back Face Region (Original)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    back_vis_recon = np.zeros_like(reconstructed_equirect)
    back_vis_recon[face_masks['back']] = reconstructed_equirect[face_masks['back']]
    plt.imshow(back_vis_recon)  # No color conversion
    plt.title('Back Face Region (Reconstructed)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(back_diff, cmap='hot')
    plt.title(f'Back Face Differences (MSE: {back_mse:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def verify_reconstruction_integrity(equirect_img, face_size):
    """
    Verifies that reconstruction is truly using cubemap faces by modifying a face.
    
    Args:
        equirect_img: Original equirectangular image
        face_size: Size for cubemap faces
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    
    # Create cubemap faces
    cube_faces = equirect_to_cubemap(equirect_img, face_size)
    
    # Display original faces
    print("Original Cubemap Faces:")
    display_faces(cube_faces)
    
    # Modify one face (e.g., add a distinctive pattern to the front face)
    if isinstance(cube_faces, dict):
        # Handle dictionary input
        modified_faces = {}
        for key, face in cube_faces.items():
            modified_faces[key] = face.copy()
        
        # Add a red X pattern to front face
        front_face = modified_faces['front']
        cv2.line(front_face, (0, 0), (face_size-1, face_size-1), (0, 0, 255), 5)
        cv2.line(front_face, (0, face_size-1), (face_size-1, 0), (0, 0, 255), 5)
    else:
        # Handle array input
        modified_faces = [face.copy() for face in cube_faces]
        
        # Add a red X pattern to front face (assume index 0 is front)
        front_face = modified_faces[0]
        cv2.line(front_face, (0, 0), (face_size-1, face_size-1), (0, 0, 255), 5)
        cv2.line(front_face, (0, face_size-1), (face_size-1, 0), (0, 0, 255), 5)
    
    print("Modified Cubemap Faces (red X added to front face):")
    display_faces(modified_faces)
    
    # Generate reconstructions from both original and modified faces
    original_recon = cubemap_to_equirect(cube_faces, equirect_img.shape[0], equirect_img.shape[1])
    modified_recon = cubemap_to_equirect(modified_faces, equirect_img.shape[0], equirect_img.shape[1])
    
    # Calculate difference between the two reconstructions
    diff = cv2.absdiff(original_recon, modified_recon)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_recon)
    plt.title("Original Reconstruction")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(modified_recon)
    plt.title("Modified Reconstruction (should show red X)")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB))
    plt.title("Difference (should highlight X pattern)")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(equirect_img)
    plt.title("Original Equirectangular")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_recon, modified_recon

def display_faces(cube_faces):
    """
    Helper function to display cubemap faces.
    Works with both dictionary and array-like inputs.
    """
    import matplotlib.pyplot as plt
    
    face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Check if cube_faces is a dictionary
    if isinstance(cube_faces, dict):
        dict_keys = ['front', 'right', 'back', 'left', 'top', 'bottom']
        for i, key in enumerate(dict_keys):
            if key in cube_faces:
                axes[i].imshow(cube_faces[key])
            else:
                print(f"Warning: Face '{key}' not found in cube_faces")
                axes[i].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
            axes[i].set_title(face_names[i])
            axes[i].axis('off')
    else:
        # Assume it's a list, tuple, or array
        for i, (face, name) in enumerate(zip(cube_faces, face_names)):
            if i < len(cube_faces):
                axes[i].imshow(face)
            else:
                print(f"Warning: Not enough faces in cube_faces")
                axes[i].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
            axes[i].set_title(name)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()