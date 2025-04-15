import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from scipy.ndimage import map_coordinates
import time


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
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import time

def equirect_to_cubemap(equirect_img, face_size):
    """
    Improved version of equirect_to_cubemap with proper handling of the back face
    to ensure correct content mapping and seamless reconstruction.
    
    Args:
        equirect_img: Equirectangular image
        face_size: Size of each face (width=height=face_size)
    
    Returns:
        Dictionary of 6 cubemap faces (front, right, back, left, top, bottom)
    """
    equirect_h, equirect_w = equirect_img.shape[:2]
    faces = {}
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    # Store the mapping coordinates for later use
    mapping_coords = {}
    
    # First pass: Create standard cubemap faces using proper spherical mapping
    for face_name in face_names:
        # Create grid of coordinates
        y_coords, x_coords = np.meshgrid(
            np.arange(face_size), np.arange(face_size), indexing='ij'
        )
        
        # Convert to normalized device coordinates (-1 to +1)
        x_ndc = 2.0 * x_coords / (face_size - 1) - 1.0
        y_ndc = 2.0 * y_coords / (face_size - 1) - 1.0
        
        # Initialize 3D coordinates for this face
        x3d = np.zeros_like(x_ndc)
        y3d = np.zeros_like(y_ndc)
        z3d = np.zeros_like(x_ndc)
        
        # Set coordinates based on face
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
            y3d = x_ndc  # Standard mapping for initial calculation
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
        
        # Convert to spherical coordinates
        r = np.sqrt(x3d**2 + y3d**2 + z3d**2)
        theta = np.arccos(z3d / r)  # 0 to pi (latitude)
        phi = np.arctan2(y3d, x3d)  # -pi to pi (longitude)
        
        # Store original phi for back face processing
        if face_name == 'back':
            original_phi = phi.copy()
        
        # Convert to equirectangular coordinates
        equirect_x = (phi + np.pi) / (2 * np.pi) * equirect_w
        equirect_y = theta / np.pi * equirect_h
        
        # Store mapping coordinates
        mapping_coords[face_name] = {
            'equirect_x': equirect_x,
            'equirect_y': equirect_y,
            'phi': phi
        }
    
    # Special processing for back face to handle the seam problem
    # Create two separate maps for left and right sides of the back face
    back_map_x = mapping_coords['back']['equirect_x'].copy()
    back_map_y = mapping_coords['back']['equirect_y'].copy()
    original_phi = mapping_coords['back']['phi']
    
    # Find potential seam location (where phi jumps)
    mid_col = face_size // 2
    
    # Create masks for the left and right halves of the back face
    left_mask = np.zeros_like(original_phi, dtype=bool)
    right_mask = np.zeros_like(original_phi, dtype=bool)
    left_mask[:, :mid_col] = True
    right_mask[:, mid_col:] = True
    
    # For left half: Ensure all phi values are negative
    left_phi = original_phi.copy()
    left_phi[left_mask & (left_phi > 0)] -= 2 * np.pi
    left_equirect_x = (left_phi + np.pi) / (2 * np.pi) * equirect_w
    
    # For right half: Ensure all phi values are positive
    right_phi = original_phi.copy()
    right_phi[right_mask & (right_phi < 0)] += 2 * np.pi
    right_equirect_x = (right_phi + np.pi) / (2 * np.pi) * equirect_w
    
    # Update back face mapping with the corrected coordinates
    back_map_x[:, :mid_col] = left_equirect_x[:, :mid_col]
    back_map_x[:, mid_col:] = right_equirect_x[:, mid_col:]
    
    # Process faces with the prepared mapping coordinates
    for face_name in face_names:
        if face_name == 'back':
            # Special handling for back face
            map_x = back_map_x.astype(np.float32)
            map_y = back_map_y.astype(np.float32)
        else:
            # Standard handling for other faces
            map_x = mapping_coords[face_name]['equirect_x'].astype(np.float32)
            map_y = mapping_coords[face_name]['equirect_y'].astype(np.float32)
        
        # Ensure coordinates are within bounds
        map_x = np.clip(map_x, 0, equirect_w - 1)
        map_y = np.clip(map_y, 0, equirect_h - 1)
        
        # Remap using OpenCV
        face = cv2.remap(equirect_img, map_x, map_y, cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_WRAP)
        
        faces[face_name] = face
    
    # Apply smoothing specifically at the center seam of back face
    back_face = faces['back']
    h, w = back_face.shape[:2]
    
    # Define seam region (few pixels around the middle)
    seam_width = 6  # Adjust based on face size
    seam_center = w // 2
    seam_start = max(0, seam_center - seam_width // 2)
    seam_end = min(w, seam_center + seam_width // 2)
    
    # Create a blend mask for the seam region
    blend_mask = np.zeros((h, w), dtype=np.float32)
    for i in range(seam_start, seam_end):
        # Create a smooth weight (0 at edges, 1 at center)
        pos = (i - seam_start) / (seam_end - seam_start - 1)
        weight = 0.5 - 0.5 * np.cos(2 * np.pi * pos)
        blend_mask[:, i] = weight
    
    # Apply guided filter to the seam region for edge-aware smoothing
    seam_region = back_face[:, seam_start:seam_end].copy()
    blurred_region = cv2.GaussianBlur(seam_region, (3, 3), 0)
    
    # Blend original and blurred versions using the blend mask
    for i in range(seam_start, seam_end):
        weight = blend_mask[:, i][:, np.newaxis]
        back_face[:, i] = (1 - weight) * back_face[:, i] + weight * blurred_region[:, i-seam_start]
    
    # Apply bilateral filter to the entire back face for final smoothing
    # while preserving edges
    faces['back'] = cv2.bilateralFilter(back_face, d=5, sigmaColor=25, sigmaSpace=25)
    
    return faces

def cubemap_to_equirect(cube_faces, height, width):
    """
    Convert cubemap faces to equirectangular panorama with correct mapping
    and seam handling for natural blending.
    
    Args:
        cube_faces: Dictionary of 6 cubemap faces (front, right, back, left, top, bottom)
                    or list/array of 6 faces
        height: Height of output equirectangular image
        width: Width of output equirectangular image
    
    Returns:
        Equirectangular panorama image
    """
    start_time = time.time()
    
    # Process input based on type (dict or array)
    if isinstance(cube_faces, dict):
        face_size = cube_faces['front'].shape[0]
        # Create a copy to avoid modifying the original
        original_faces = {k: v.copy() for k, v in cube_faces.items()}
        
        # We'll use the original back face but need to handle its seams
        # Extract all faces for easy indexing
        cube_array = np.stack([
            original_faces['front'], 
            original_faces['right'], 
            original_faces['back'], 
            original_faces['left'], 
            original_faces['top'], 
            original_faces['bottom']
        ])
    else:
        face_size = cube_faces[0].shape[0]
        # Create a copy to avoid modifying the original
        original_faces = [face.copy() for face in cube_faces]
        cube_array = np.stack(original_faces)
    
    # Create equirectangular grid
    phi = np.linspace(-np.pi, np.pi, width)
    theta = np.linspace(0, np.pi, height)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Convert to Cartesian coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # Initialize arrays for face index and coordinates
    face_idx = np.zeros((height, width), dtype=np.int32)
    u_coords = np.zeros((height, width), dtype=np.float32)
    v_coords = np.zeros((height, width), dtype=np.float32)
    
    # Calculate absolute values for determining face
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    # Find maximum component to determine face
    max_comp = np.maximum(np.maximum(abs_x, abs_y), abs_z)
    
    # Define a small epsilon to avoid division by zero
    eps = 1e-10
    
    # Front face (+X)
    mask = (abs_x == max_comp) & (x > 0)
    face_idx[mask] = 0
    u_coords[mask] = -y[mask] / (x[mask] + eps)
    v_coords[mask] = -z[mask] / (x[mask] + eps)
    
    # Right face (+Y)
    mask = (abs_y == max_comp) & (y > 0)
    face_idx[mask] = 1
    u_coords[mask] = x[mask] / (y[mask] + eps)
    v_coords[mask] = -z[mask] / (y[mask] + eps)
    
    # Back face (-X)
    mask = (abs_x == max_comp) & (x <= 0)
    face_idx[mask] = 2
    u_coords[mask] = y[mask] / (-x[mask] + eps)
    v_coords[mask] = -z[mask] / (-x[mask] + eps)
    
    # Left face (-Y)
    mask = (abs_y == max_comp) & (y <= 0)
    face_idx[mask] = 3
    u_coords[mask] = -x[mask] / (-y[mask] + eps)
    v_coords[mask] = -z[mask] / (-y[mask] + eps)
    
    # Top face (+Z)
    mask = (abs_z == max_comp) & (z > 0)
    face_idx[mask] = 4
    u_coords[mask] = x[mask] / (z[mask] + eps)
    v_coords[mask] = y[mask] / (z[mask] + eps)
    
    # Bottom face (-Z)
    mask = (abs_z == max_comp) & (z <= 0)
    face_idx[mask] = 5
    u_coords[mask] = x[mask] / (-z[mask] + eps)
    v_coords[mask] = -y[mask] / (-z[mask] + eps)
    
    # Convert to face pixel coordinates
    u_coords = (u_coords + 1) * 0.5 * (face_size - 1)
    v_coords = (v_coords + 1) * 0.5 * (face_size - 1)
    
    # Clip coordinates to valid range
    u_coords = np.clip(u_coords, 0, face_size - 1)
    v_coords = np.clip(v_coords, 0, face_size - 1)
    
    # Initialize output and back face mask
    equirect = np.zeros((height, width, 3), dtype=np.uint8)
    back_mask = np.zeros((height, width), dtype=bool)
    
    # Store back face boundary coordinates for special handling
    back_boundary_x = []
    back_boundary_y = []
    
    # Process each face
    for f in range(6):
        mask = (face_idx == f)
        if not np.any(mask):
            continue
        
        # Get coordinates for this face
        u_face = u_coords[mask]
        v_face = v_coords[mask]
        coords = np.vstack((v_face, u_face))
        
        # Get indices in the output image
        y_idx, x_idx = np.where(mask)
        
        # Sample each color channel with bilinear interpolation
        for c in range(3):
            sampled = map_coordinates(
                cube_array[f, :, :, c],
                coords,
                order=1,  # bilinear interpolation
                mode='nearest'
            )
            equirect[y_idx, x_idx, c] = sampled
        
        # Mark back face pixels and identify boundary regions
        if f == 2:  # Back face
            back_mask[y_idx, x_idx] = True
            
            # Identify pixels at the left and right edges of the back face in equirect
            edge_width = width // 60  # Adjust based on image width
            
            # Find the leftmost and rightmost columns of the back face in equirect
            unique_x = np.unique(x_idx)
            if len(unique_x) > 0:
                left_edge = unique_x[0]
                right_edge = unique_x[-1]
                
                # Store coordinates of boundary pixels (within edge_width)
                for i in range(len(y_idx)):
                    if x_idx[i] < left_edge + edge_width or x_idx[i] > right_edge - edge_width:
                        back_boundary_x.append(x_idx[i])
                        back_boundary_y.append(y_idx[i])
    
    # Special handling for back face boundaries in equirectangular image
    if len(back_boundary_x) > 0:
        # Convert to numpy arrays for indexing
        back_boundary_x = np.array(back_boundary_x)
        back_boundary_y = np.array(back_boundary_y)
        
        # Apply a specialized filtering only to boundary regions
        # We'll use a bilateral filter to preserve edges while smoothing artifacts
        boundary_region = equirect[back_boundary_y, back_boundary_x].copy()
        
        # Create a temporary image for bilateral filtering
        temp_img = equirect.copy()
        temp_filtered = cv2.bilateralFilter(temp_img, d=5, sigmaColor=25, sigmaSpace=25)
        
        # Replace only the boundary pixels with filtered versions
        equirect[back_boundary_y, back_boundary_x] = temp_filtered[back_boundary_y, back_boundary_x]
    
    # Handle the poles (top and bottom of equirectangular image)
    # These regions often contain distortions
    pole_height = height // 20  # Adjust based on image height
    
    # Top pole
    top_pole = np.zeros((pole_height, width, 3), dtype=np.uint8)
    for x in range(width):
        # Average color of top few rows
        avg_color = np.mean(equirect[pole_height:pole_height*2, x], axis=0).astype(np.uint8)
        top_pole[:, x] = avg_color
    
    # Apply a circular blur to smooth the pole
    top_pole = cv2.GaussianBlur(top_pole, (5, 5), 0)
    
    # Apply a gradual blend to the top
    for y in range(pole_height):
        blend_factor = y / pole_height
        equirect[y, :] = (1 - blend_factor) * top_pole[y, :] + blend_factor * equirect[y, :]
    
    # Similarly for bottom pole
    bottom_pole = np.zeros((pole_height, width, 3), dtype=np.uint8)
    for x in range(width):
        avg_color = np.mean(equirect[height-pole_height*2:height-pole_height, x], axis=0).astype(np.uint8)
        bottom_pole[:, x] = avg_color
    
    bottom_pole = cv2.GaussianBlur(bottom_pole, (5, 5), 0)
    
    for y in range(pole_height):
        blend_factor = (pole_height - y - 1) / pole_height
        equirect[height-pole_height+y, :] = (1 - blend_factor) * bottom_pole[y, :] + blend_factor * equirect[height-pole_height+y, :]
    
    # Handle equirectangular left and right edges (wrap-around seam)
    # This is especially important for the back face
    edge_width = width // 60
    
    # Create left and right edge masks
    left_edge_mask = np.zeros((height, width), dtype=bool)
    right_edge_mask = np.zeros((height, width), dtype=bool)
    left_edge_mask[:, :edge_width] = True
    right_edge_mask[:, width-edge_width:] = True
    
    # Apply a specialized edge blend
    if np.any(left_edge_mask) and np.any(right_edge_mask):
        # Create a blended edge using data from both sides
        for y in range(height):
            # Get colors from both edges
            left_colors = equirect[y, :edge_width].copy()
            right_colors = equirect[y, width-edge_width:].copy()
            
            # Apply a gradual blend
            for x in range(edge_width):
                # Weight increases from left to right edge
                weight = x / (edge_width - 1)
                
                # Apply to left edge
                left_blend = (1 - weight) * left_colors[x] + weight * right_colors[0]
                equirect[y, x] = left_blend.astype(np.uint8)
                
                # Apply to right edge
                right_blend = weight * left_colors[0] + (1 - weight) * right_colors[x]
                equirect[y, width-edge_width+x] = right_blend.astype(np.uint8)
    
    # Final overall smoothing for equirectangular image
    # Apply a very subtle bilateral filter for overall consistency
    # while preserving important details
    final_equirect = cv2.bilateralFilter(equirect, d=3, sigmaColor=15, sigmaSpace=15)
    
    end_time = time.time()
    print(f"Seamless cubemap to equirect conversion took {end_time - start_time:.2f} seconds")
    
    return final_equirect
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

def visualize_mse(original, reconstructed):
    """
    Enhanced visualization of MSE with human-readable colors.
    
    Args:
        original: Original equirectangular image
        reconstructed: Reconstructed equirectangular image
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.colors import LinearSegmentedColormap
    
    # Calculate difference and MSE
    diff = cv2.absdiff(original, reconstructed)
    squared_diff = diff.astype(np.float32)**2
    mse = np.mean(squared_diff)
    psnr = 10 * np.log10((255**2) / max(mse, 1e-10))
    
    # Create normalized difference for visualization
    diff_norm = np.sqrt(np.sum(diff**2, axis=2) / 3)  # RMS difference across channels
    max_diff = np.max(diff_norm)
    
    # Create a custom colormap for better visibility
    colors = ['darkblue', 'blue', 'cyan', 'lime', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Equirectangular")
    axes[0, 0].axis('off')
    
    # Reconstructed
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title("Reconstructed Equirectangular")
    axes[0, 1].axis('off')
    
    # Difference heatmap
    im = axes[1, 0].imshow(diff_norm, cmap=cmap, vmin=0, vmax=max_diff)
    axes[1, 0].set_title(f"Difference Heatmap (MSE: {mse:.2f}, PSNR: {psnr:.2f} dB)")
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0], label='Pixel Difference')
    
    # Histogram of differences
    axes[1, 1].hist(diff_norm.flatten(), bins=100, color='skyblue', edgecolor='navy')
    axes[1, 1].set_title(f"Histogram of Pixel Differences")
    axes[1, 1].set_xlabel("Difference Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show closeup of the most different regions
    plt.figure(figsize=(12, 6))
    
    # Find top 5% most different pixels
    threshold = np.percentile(diff_norm, 95)
    high_diff_mask = diff_norm > threshold
    
    # Create a mask image highlighting these areas
    highlight = np.zeros_like(original)
    highlight[high_diff_mask] = [255, 0, 0]  # Red for high difference areas
    
    # Overlay on reconstructed image
    alpha = 0.7
    highlighted_img = cv2.addWeighted(reconstructed, alpha, highlight, 1-alpha, 0)
    
    plt.imshow(highlighted_img)
    plt.title("Areas with Highest Difference Highlighted")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return mse, psnr


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