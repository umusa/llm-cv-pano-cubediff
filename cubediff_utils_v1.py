import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
import time
import math
from cubediff_utils import *

# 2025-4-18
# MSE reduced from 2132.02 to 153.33 (92.8% reduction)
# PSNR improved from 14.84 dB to 26.27 dB (11.43 dB gain)
# SSIM improved from 0.5320 to 0.7276 (36.8% improvement)
# All face errors substantially reduced (Back face error down 81.3%)


def improved_equirect_to_cubemap(equi, face_size):
    """
    High-precision equirectangular to cubemap conversion.
    """
    H, W = equi.shape[:2]
    faces = {}
    
    for name in ('front', 'right', 'back', 'left', 'top', 'bottom'):
        # Create precise subpixel-aligned grid
        y_coords, x_coords = np.meshgrid(
            (np.arange(face_size) + 0.5) / face_size * 2 - 1,
            (np.arange(face_size) + 0.5) / face_size * 2 - 1,
            indexing='ij'
        )
        
        # Calculate 3D ray direction based on face
        if name == 'front':
            x, y, z = x_coords, -y_coords, np.ones_like(x_coords)
        elif name == 'right':
            x, y, z = np.ones_like(x_coords), -y_coords, -x_coords
        elif name == 'back':
            x, y, z = -x_coords, -y_coords, -np.ones_like(x_coords)
        elif name == 'left':
            x, y, z = -np.ones_like(x_coords), -y_coords, x_coords
        elif name == 'top':
            x, y, z = x_coords, np.ones_like(x_coords), y_coords
        elif name == 'bottom':
            x, y, z = x_coords, -np.ones_like(x_coords), -y_coords
        
        # Normalize with double precision
        norm = np.sqrt(x**2 + y**2 + z**2).astype(np.float64)
        x = (x / norm).astype(np.float64)
        y = (y / norm).astype(np.float64)
        z = (z / norm).astype(np.float64)
        
        # Convert to spherical coordinates
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -0.999999, 0.999999))  # Avoid numerical issues
        
        # Precise pixel coordinate calculation
        map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
        map_y = ((0.5 - lat / np.pi) * H).astype(np.float32)
        
        # Use Lanczos interpolation for higher quality
        face_img = cv2.remap(
            equi, map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_WRAP
        )
        
        faces[name] = face_img
    
    return faces

def optimized_cubemap_to_equirect(cube_faces, H, W):
    """
    Optimized cubemap to equirectangular conversion with high-quality sampling.
    """
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Create a higher-resolution output for downsampling later
    scale_factor = 1.5
    H_hires = int(H * scale_factor)
    W_hires = int(W * scale_factor)
    
    equirect_hires = np.zeros((H_hires, W_hires, 3), dtype=np.uint8)
    
    # Create precise latitude and longitude grid
    lat = np.linspace(np.pi/2, -np.pi/2, H_hires, endpoint=False) + np.pi/(2*H_hires)
    lon = np.linspace(-np.pi, np.pi, W_hires, endpoint=False) + np.pi/W_hires
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Convert to Cartesian coordinates with high precision
    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)
    
    # Determine face for each equirectangular pixel
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    # Create face ID map
    face_id = np.zeros(x.shape, dtype=np.int32)
    
    # Front face (+Z)
    face_id[(abs_z > abs_x) & (abs_z > abs_y) & (z > 0)] = 0
    
    # Right face (+X)
    face_id[(abs_x >= abs_z) & (abs_x > abs_y) & (x > 0)] = 1
    
    # Back face (-Z)
    face_id[(abs_z > abs_x) & (abs_z > abs_y) & (z <= 0)] = 2
    
    # Left face (-X)
    face_id[(abs_x >= abs_z) & (abs_x > abs_y) & (x <= 0)] = 3
    
    # Top face (+Y)
    face_id[(abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)] = 4
    
    # Bottom face (-Y)
    face_id[(abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)] = 5
    
    # Compute local coordinates for each face
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # Use a small epsilon for numerical stability
    eps = 1e-10
    
    # Front face coordinates
    mask = face_id == 0
    u[mask] = x[mask] / (z[mask] + eps)
    v[mask] = -y[mask] / (z[mask] + eps)
    
    # Right face coordinates
    mask = face_id == 1
    u[mask] = -z[mask] / (x[mask] + eps)
    v[mask] = -y[mask] / (x[mask] + eps)
    
    # Back face coordinates
    mask = face_id == 2
    u[mask] = -x[mask] / (-z[mask] + eps)
    v[mask] = -y[mask] / (-z[mask] + eps)
    
    # Left face coordinates
    mask = face_id == 3
    u[mask] = z[mask] / (-x[mask] + eps)
    v[mask] = -y[mask] / (-x[mask] + eps)
    
    # Top face coordinates
    mask = face_id == 4
    u[mask] = x[mask] / (y[mask] + eps)
    v[mask] = z[mask] / (y[mask] + eps)
    
    # Bottom face coordinates
    mask = face_id == 5
    u[mask] = x[mask] / (-y[mask] + eps)
    v[mask] = -z[mask] / (-y[mask] + eps)
    
    # Clip and normalize to pixel coordinates
    u = np.clip(u, -1, 1)
    v = np.clip(v, -1, 1)
    
    # Convert to pixel coordinates with centered sampling
    u_px = ((u + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    v_px = ((v + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    
    # Map face index to face name
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    # Sample from each face
    for face_idx, face_name in enumerate(face_names):
        mask = face_id == face_idx
        if not np.any(mask):
            continue
        
        # Create sampling maps for this face
        map_x = np.zeros((H_hires, W_hires), dtype=np.float32)
        map_y = np.zeros((H_hires, W_hires), dtype=np.float32)
        
        map_x[mask] = u_px[mask]
        map_y[mask] = v_px[mask]
        
        # Get face image
        face_img = cube_faces[face_name]
        
        # Use remap with cubic interpolation for high quality
        sampled = cv2.remap(
            face_img, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Copy only the masked region
        equirect_hires[mask] = sampled[mask]
    
    # Smooth seams in the equirectangular image
    # Left-right wrap seam (±π longitude)
    seam_width = W_hires // 100
    
    for i in range(seam_width):
        alpha = i / seam_width
        equirect_hires[:, i] = (
            alpha * equirect_hires[:, i] + 
            (1-alpha) * equirect_hires[:, -(seam_width-i)]
        )
        equirect_hires[:, -(i+1)] = (
            alpha * equirect_hires[:, -(i+1)] + 
            (1-alpha) * equirect_hires[:, seam_width-i-1]
        )
    
    # Downsample to original resolution
    equirect = cv2.resize(equirect_hires, (W, H), interpolation=cv2.INTER_AREA)
    
    return equirect

def high_quality_conversion(equirect_img, face_size=512):
    """
    End-to-end high-quality conversion using optimized sampling and blending.
    """
    # Step 1: Convert to cubemap with high quality sampling
    cube_faces = high_quality_equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Apply improved seam handling
    cube_faces = improved_seam_handling(cube_faces)
    
    # Step 3: Convert back to equirectangular with optimized sampling
    improved_equirect = optimized_cubemap_to_equirect(cube_faces, *equirect_img.shape[:2])
    
    return improved_equirect, cube_faces


def high_quality_equirect_to_cubemap(equi, face_size):
    """
    High-quality equirectangular to cubemap conversion with oversampling.
    
    Args:
        equi: Equirectangular image
        face_size: Size of cubemap faces
        
    Returns:
        Dictionary of high-quality cubemap faces
    """
    H, W = equi.shape[:2]
    faces = {}
    
    # Step 1: Create higher-resolution faces initially (4x oversampling)
    # This is a key factor in achieving higher quality
    oversample_factor = 4
    target_size = face_size * oversample_factor
    
    for name in ('front', 'right', 'back', 'left', 'top', 'bottom'):
        # Create precise subpixel-aligned grid at higher resolution
        y_coords, x_coords = np.meshgrid(
            (np.arange(target_size) + 0.5) / target_size * 2 - 1,
            (np.arange(target_size) + 0.5) / target_size * 2 - 1,
            indexing='ij'
        )
        
        # Calculate 3D ray direction based on face
        if name == 'front':
            x, y, z = x_coords, -y_coords, np.ones_like(x_coords)
        elif name == 'right':
            x, y, z = np.ones_like(x_coords), -y_coords, -x_coords
        elif name == 'back':
            x, y, z = -x_coords, -y_coords, -np.ones_like(x_coords)
        elif name == 'left':
            x, y, z = -np.ones_like(x_coords), -y_coords, x_coords
        elif name == 'top':
            x, y, z = x_coords, np.ones_like(x_coords), y_coords
        elif name == 'bottom':
            x, y, z = x_coords, -np.ones_like(x_coords), -y_coords
        
        # Normalize to unit vectors with double precision
        norm = np.sqrt(x**2 + y**2 + z**2).astype(np.float64)
        x = (x / norm).astype(np.float64)
        y = (y / norm).astype(np.float64)
        z = (z / norm).astype(np.float64)
        
        # Convert to spherical coordinates
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -0.999999, 0.999999))  # Avoid numerical issues
        
        # Get equirectangular pixel coordinates
        map_x = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
        map_y = ((0.5 - lat / np.pi) * H).astype(np.float32)
        
        # Use Lanczos interpolation for highest quality
        face_img_hires = cv2.remap(
            equi, map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Apply face-specific enhancements at high resolution
        if name == 'back':
            # Special processing for problematic back face
            face_img_hires = cv2.bilateralFilter(face_img_hires, d=9, sigmaColor=50, sigmaSpace=50)
        elif name in ('top', 'bottom'):
            # Special processing for pole faces
            face_img_hires = cv2.bilateralFilter(face_img_hires, d=7, sigmaColor=30, sigmaSpace=30)
        
        # Step 2: Downsample to target size with area interpolation
        # High-quality downsampling is crucial for antialiasing
        face_img = cv2.resize(face_img_hires, (face_size, face_size), 
                              interpolation=cv2.INTER_AREA)
        
        faces[name] = face_img
    
    return faces


def enhance_seams(cube_faces):
    """
    Enhanced seam handling between cubemap faces.
    
    Args:
        cube_faces: Dictionary of cubemap faces
        
    Returns:
        Dictionary with improved seams
    """
    out = {}
    for k, img in cube_faces.items():
        out[k] = img.copy()
    
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Adaptive blend width based on face size
    blend_width = max(4, face_size // 32)
    
    # Create smooth falloff weights for blending
    weights = np.zeros(blend_width)
    for i in range(blend_width):
        # Smooth cubic falloff (smoother than linear)
        x = i / blend_width
        weights[i] = 3*(x**2) - 2*(x**3)  # Smooth step function
    
    # Process each lateral face
    for name in ('front', 'right', 'back', 'left'):
        # Determine adjacent faces
        if name == 'front':
            left_adj = 'left'
            right_adj = 'right'
        elif name == 'right':
            left_adj = 'front'
            right_adj = 'back'
        elif name == 'back':
            left_adj = 'right'
            right_adj = 'left'
        elif name == 'left':
            left_adj = 'back'
            right_adj = 'front'
        
        # Get faces as float for precise blending
        face = out[name].astype(np.float64)
        left_face = out[left_adj].astype(np.float64)
        right_face = out[right_adj].astype(np.float64)
        
        # Enhance left edge
        for i in range(blend_width):
            alpha = weights[blend_width - i - 1]
            face[:, i] = (1 - alpha) * face[:, i] + alpha * left_face[:, -(i+1)]
        
        # Enhance right edge
        for i in range(blend_width):
            alpha = weights[i]
            face[:, -(i+1)] = (1 - alpha) * face[:, -(i+1)] + alpha * right_face[:, i]
        
        # Update the face
        out[name] = face.clip(0, 255).astype(np.uint8)
    
    return out


def ultra_quality_cubemap_to_equirect(cube_faces, H, W):
    """
    Ultra-high quality cubemap to equirectangular conversion.
    
    Args:
        cube_faces: Dictionary of cubemap faces
        H, W: Height and width of output equirectangular image
        
    Returns:
        High-quality equirectangular image
    """
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Create a higher-resolution output for downsampling later
    # This is key to achieving better quality
    scale_factor = 2
    H_hires = int(H * scale_factor)
    W_hires = int(W * scale_factor)
    
    equirect_hires = np.zeros((H_hires, W_hires, 3), dtype=np.uint8)
    
    # Create precise latitude and longitude grid
    lat = np.linspace(np.pi/2, -np.pi/2, H_hires, endpoint=False) + np.pi/(2*H_hires)
    lon = np.linspace(-np.pi, np.pi, W_hires, endpoint=False) + np.pi/W_hires
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Convert to Cartesian coordinates with high precision
    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)
    
    # Determine face for each equirectangular pixel
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    # Create face ID map (0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom)
    face_id = np.zeros(x.shape, dtype=np.int32)
    
    # Determine the face for each pixel based on the dominant axis
    face_id[(abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)] = 0  # Front
    face_id[(abs_x >= abs_z) & (abs_x >= abs_y) & (x > 0)] = 1  # Right
    face_id[(abs_z >= abs_x) & (abs_z >= abs_y) & (z <= 0)] = 2  # Back
    face_id[(abs_x >= abs_z) & (abs_x >= abs_y) & (x <= 0)] = 3  # Left
    face_id[(abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)] = 4  # Top
    face_id[(abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)] = 5  # Bottom
    
    # Compute local coordinates for each face
    u = np.zeros_like(x, dtype=np.float64)
    v = np.zeros_like(x, dtype=np.float64)
    
    # Use a small epsilon for numerical stability
    eps = 1e-10
    
    # Front face coordinates
    mask = face_id == 0
    u[mask] = x[mask] / (z[mask] + eps)
    v[mask] = -y[mask] / (z[mask] + eps)
    
    # Right face coordinates
    mask = face_id == 1
    u[mask] = -z[mask] / (x[mask] + eps)
    v[mask] = -y[mask] / (x[mask] + eps)
    
    # Back face coordinates
    mask = face_id == 2
    u[mask] = -x[mask] / (-z[mask] + eps)
    v[mask] = -y[mask] / (-z[mask] + eps)
    
    # Left face coordinates
    mask = face_id == 3
    u[mask] = z[mask] / (-x[mask] + eps)
    v[mask] = -y[mask] / (-x[mask] + eps)
    
    # Top face coordinates
    mask = face_id == 4
    u[mask] = x[mask] / (y[mask] + eps)
    v[mask] = z[mask] / (y[mask] + eps)
    
    # Bottom face coordinates
    mask = face_id == 5
    u[mask] = x[mask] / (-y[mask] + eps)
    v[mask] = -z[mask] / (-y[mask] + eps)
    
    # Clip and normalize to pixel coordinates
    u = np.clip(u, -0.999999, 0.999999)
    v = np.clip(v, -0.999999, 0.999999)
    
    # Convert to pixel coordinates with centered sampling
    u_px = ((u + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    v_px = ((v + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    
    # Map face index to face name
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    # Sample from each face
    for face_idx, face_name in enumerate(face_names):
        mask = face_id == face_idx
        if not np.any(mask):
            continue
        
        # Create sampling maps for this face
        map_x = np.zeros((H_hires, W_hires), dtype=np.float32)
        map_y = np.zeros((H_hires, W_hires), dtype=np.float32)
        
        map_x[mask] = u_px[mask]
        map_y[mask] = v_px[mask]
        
        # Get face image
        face_img = cube_faces[face_name]
        
        # Use cubic interpolation for higher quality
        sampled = cv2.remap(
            face_img, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Copy only the masked region
        equirect_hires[mask] = sampled[mask]
    
    # Smooth seams in the equirectangular image
    # Left-right wrap seam (±π longitude)
    seam_width = W_hires // 64
    
    # Apply smooth blending at the seam
    for i in range(seam_width):
        # Smooth cubic falloff
        x = i / seam_width
        alpha = 3*(x**2) - 2*(x**3)  # Smooth step function
        
        # Blend left edge
        equirect_hires[:, i] = (
            alpha * equirect_hires[:, i] + 
            (1-alpha) * equirect_hires[:, -(seam_width-i)]
        )
        
        # Blend right edge
        equirect_hires[:, -(i+1)] = (
            alpha * equirect_hires[:, -(i+1)] + 
            (1-alpha) * equirect_hires[:, seam_width-i-1]
        )
    
    # Apply bilateral filter for edge preservation
    equirect_hires = cv2.bilateralFilter(equirect_hires, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Downsample to original resolution with high-quality interpolation
    equirect = cv2.resize(equirect_hires, (W, H), interpolation=cv2.INTER_AREA)
    
    return equirect

def super_high_quality_conversion(equirect_img, face_size=512):
    """
    Super high-quality conversion with comprehensive enhancements.
    
    Args:
        equirect_img: Input equirectangular image
        face_size: Size of cubemap faces
    
    Returns:
        Tuple of (improved_equirect, cube_faces)
    """
    # Step 1: Generate high-quality cubemap faces with oversampling
    cube_faces = high_quality_equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Enhance seams between faces
    cube_faces = enhance_seams(cube_faces)
    
    # Step 3: Convert back to equirectangular with ultra-high quality
    improved_equirect = ultra_quality_cubemap_to_equirect(cube_faces, *equirect_img.shape[:2])
    
    return improved_equirect, cube_faces




def improved_seam_handling(cube_faces):
    """
    Improved seam handling across cubemap faces with gradient blending.
    """
    out = {}
    for k, img in cube_faces.items():
        out[k] = img.copy()
    
    # Process lateral faces (front, right, back, left)
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Prepare Gaussian weights for smooth blending
    blend_width = face_size // 32  # Adaptive width based on face size
    
    # Create a 1D weight function that drops off smoothly
    x = np.arange(blend_width)
    weights = np.exp(-(x**2) / (2 * (blend_width/3)**2))
    weights = weights / weights.max()  # Normalize to [0,1]
    
    # Process each face
    for name in ('front', 'right', 'back', 'left'):
        # Get adjacent faces
        if name == 'front':
            left_adj = 'left'
            right_adj = 'right'
        elif name == 'right':
            left_adj = 'front'
            right_adj = 'back'
        elif name == 'back':
            left_adj = 'right'
            right_adj = 'left'
        elif name == 'left':
            left_adj = 'back'
            right_adj = 'front'
        
        # Get faces
        face = out[name].astype(np.float32)
        left_face = out[left_adj].astype(np.float32)
        right_face = out[right_adj].astype(np.float32)
        
        # Apply left edge blending
        for i in range(blend_width):
            alpha = weights[blend_width - i - 1]
            face[:, i] = (1 - alpha) * face[:, i] + alpha * left_face[:, -(i+1)]
        
        # Apply right edge blending
        for i in range(blend_width):
            alpha = weights[i]
            face[:, -(i+1)] = (1 - alpha) * face[:, -(i+1)] + alpha * right_face[:, i]
        
        # Update the face
        out[name] = face.clip(0, 255).astype(np.uint8)
    
    return out

def minimal_seam_correction(cube_faces):
    """
    Minimal seam correction with accurate blending.
    """
    out = {}
    for k, img in cube_faces.items():
        out[k] = img.copy()
    
    # Process lateral faces only
    for name in ('front', 'right', 'back', 'left'):
        face = out[name].astype(np.float32)
        
        # Get adjacent faces based on cube topology
        if name == 'front':
            left_face = out['left']
            right_face = out['right']
        elif name == 'right':
            left_face = out['front']
            right_face = out['back']
        elif name == 'back':
            left_face = out['right']
            right_face = out['left']
        elif name == 'left':
            left_face = out['back']
            right_face = out['front']
        
        # Use minimal blend width (2 pixels is usually sufficient)
        blend_width = 2
        
        # Blend left edge with right edge of left face
        for i in range(blend_width):
            alpha = (i + 1) / (blend_width + 1)
            face[:, i] = (1 - alpha) * left_face[:, -(blend_width-i)] + alpha * face[:, i]
        
        # Blend right edge with left edge of right face
        for i in range(blend_width):
            alpha = (blend_width - i) / (blend_width + 1)
            face[:, -(i+1)] = alpha * right_face[:, i] + (1 - alpha) * face[:, -(i+1)]
        
        out[name] = face.clip(0, 255).astype(np.uint8)
    
    return out

def minimal_pole_correction(cube_faces):
    """
    Minimal pole correction with adaptive supersampling.
    """
    out = cube_faces.copy()
    
    for pole in ('top', 'bottom'):
        face = cube_faces[pole]
        face_size = face.shape[0]
        
        # Use adaptive factor based on face size
        factor = 3  # 3x supersampling is usually sufficient
        
        # Step 1: Supersample with high-quality interpolation
        big = cv2.resize(face, (0, 0), fx=factor, fy=factor,
                        interpolation=cv2.INTER_LANCZOS4)
        
        # Step 2: Apply edge-preserving filter to maintain sharpness while reducing noise
        big = cv2.edgePreservingFilter(big, flags=cv2.RECURS_FILTER, 
                                      sigma_s=60, sigma_r=0.4)
        
        # Step 3: Downsample with high-quality area interpolation
        out[pole] = cv2.resize(big, (face_size, face_size), 
                              interpolation=cv2.INTER_AREA)
    
    return out

def special_back_face_processing(cube_faces):
    """
    Special processing for the back face which often has the most artifacts.
    """
    out = cube_faces.copy()
    
    if 'back' in cube_faces:
        back = cube_faces['back'].copy()
        
        # Apply bilateral filter to the back face to reduce noise while preserving edges
        back = cv2.bilateralFilter(back, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Apply a gentle sharpening filter
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]]) / 5.0
        back = cv2.filter2D(back, -1, kernel)
        
        out['back'] = back
    
    return out

def advanced_back_face_processing(cube_faces):
    """
    Specialized processing for the back face.
    """
    out = cube_faces.copy()
    
    if 'back' in cube_faces:
        back = cube_faces['back'].copy()
        
        # Step 1: Apply detail-preserving denoising
        denoised = cv2.fastNlMeansDenoisingColored(back, None, 10, 10, 7, 21)
        
        # Step 2: Apply contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Apply detail enhancement
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Blend original and processed to maintain natural look
        alpha = 0.7
        result = cv2.addWeighted(enhanced, alpha, sharpened, 1-alpha, 0)
        
        out['back'] = result
    
    return out

def improved_cubemap_to_equirect(cube_faces, H, W):
    """
    High-precision cubemap to equirectangular conversion.
    """
    face_size = next(iter(cube_faces.values())).shape[0]
    equirect = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Create precise latitude and longitude grid
    lat = np.linspace(np.pi/2, -np.pi/2, H, endpoint=False) + np.pi/(2*H)
    lon = np.linspace(-np.pi, np.pi, W, endpoint=False) + np.pi/W
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Convert to Cartesian coordinates with double precision
    x = (np.cos(lat_grid) * np.sin(lon_grid)).astype(np.float64)
    y = np.sin(lat_grid).astype(np.float64)
    z = (np.cos(lat_grid) * np.cos(lon_grid)).astype(np.float64)
    
    # Determine face for each equirectangular pixel
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    
    # Create face ID map
    face_id = np.zeros(x.shape, dtype=np.int32)
    
    # Front face (+Z)
    mask_front = (abs_z > abs_x) & (abs_z > abs_y) & (z > 0)
    face_id[mask_front] = 0
    
    # Right face (+X)
    mask_right = (abs_x >= abs_z) & (abs_x > abs_y) & (x > 0)
    face_id[mask_right] = 1
    
    # Back face (-Z)
    mask_back = (abs_z > abs_x) & (abs_z > abs_y) & (z <= 0)
    face_id[mask_back] = 2
    
    # Left face (-X)
    mask_left = (abs_x >= abs_z) & (abs_x > abs_y) & (x <= 0)
    face_id[mask_left] = 3
    
    # Top face (+Y)
    mask_top = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    face_id[mask_top] = 4
    
    # Bottom face (-Y)
    mask_bottom = (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)
    face_id[mask_bottom] = 5
    
    # Compute local coordinates for each face
    u = np.zeros_like(x, dtype=np.float64)
    v = np.zeros_like(x, dtype=np.float64)
    
    # Use a small epsilon for numerical stability
    eps = 1e-10
    
    # Front face coordinates
    mask = face_id == 0
    u[mask] = x[mask] / (z[mask] + eps)
    v[mask] = -y[mask] / (z[mask] + eps)
    
    # Right face coordinates
    mask = face_id == 1
    u[mask] = -z[mask] / (x[mask] + eps)
    v[mask] = -y[mask] / (x[mask] + eps)
    
    # Back face coordinates
    mask = face_id == 2
    u[mask] = -x[mask] / (-z[mask] + eps)
    v[mask] = -y[mask] / (-z[mask] + eps)
    
    # Left face coordinates
    mask = face_id == 3
    u[mask] = z[mask] / (-x[mask] + eps)
    v[mask] = -y[mask] / (-x[mask] + eps)
    
    # Top face coordinates
    mask = face_id == 4
    u[mask] = x[mask] / (y[mask] + eps)
    v[mask] = z[mask] / (y[mask] + eps)
    
    # Bottom face coordinates
    mask = face_id == 5
    u[mask] = x[mask] / (-y[mask] + eps)
    v[mask] = -z[mask] / (-y[mask] + eps)
    
    # Clip and normalize to pixel coordinates
    u = np.clip(u, -0.999999, 0.999999)
    v = np.clip(v, -0.999999, 0.999999)
    
    # Convert to pixel coordinates with subpixel precision
    u_px = ((u + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    v_px = ((v + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    
    # Map face index to face name
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    # Sample from each face with optimized approach
    for face_idx, face_name in enumerate(face_names):
        mask = face_id == face_idx
        if not np.any(mask):
            continue
        
        # Create sampling maps for this face
        map_x = np.zeros((H, W), dtype=np.float32)
        map_y = np.zeros((H, W), dtype=np.float32)
        
        map_x[mask] = u_px[mask]
        map_y[mask] = v_px[mask]
        
        # Get face image
        face_img = cube_faces[face_name]
        
        # Use remap with Lanczos interpolation for high quality
        sampled = cv2.remap(
            face_img, map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Copy only the masked region
        equirect[mask] = sampled[mask]
    
    # Apply final smoothing to seams in the equirectangular image
    seam_width = W // 50
    
    # Process left-right wrap seam (±π longitude)
    for i in range(seam_width):
        alpha = i / seam_width
        equirect[:, i] = equirect[:, i] * alpha + equirect[:, -(seam_width-i)] * (1-alpha)
        equirect[:, -(i+1)] = equirect[:, -(i+1)] * alpha + equirect[:, i+1] * (1-alpha)
    
    return equirect

def enhanced_equirect_to_cubemap_to_equirect(equirect_img, face_size=512):
    """
    End-to-end high-quality conversion with comprehensive enhancements.
    """
    # Step 1: Convert to cubemap with high precision
    cube_faces = improved_equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Apply comprehensive enhancements
    # Correct seams between faces
    cube_faces = advanced_seam_correction(cube_faces)
    
    # Improve pole faces
    cube_faces = advanced_pole_correction(cube_faces)
    
    # Special processing for back face
    cube_faces = advanced_back_face_processing(cube_faces)
    
    # Step 3: Convert back to equirectangular with high precision
    improved_equirect = improved_cubemap_to_equirect(cube_faces, *equirect_img.shape[:2])
    
    return improved_equirect, cube_faces

def precise_equirect_to_cubemap_to_equirect(equirect_img, face_size=512):
    """
    Improved end-to-end conversion with minimal corrections.
    """
    # Step 1: Precise equirectangular to cubemap conversion
    cube_faces = precise_equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Apply minimal corrections
    cube_faces = minimal_seam_correction(cube_faces)
    cube_faces = minimal_pole_correction(cube_faces)
    
    # Step 3: Precise cubemap to equirectangular conversion
    improved_equirect = precise_cubemap_to_equirect(cube_faces, *equirect_img.shape[:2])
    
    return improved_equirect, cube_faces

def advanced_seam_correction(cube_faces):
    """
    Advanced seam correction with adaptive blending.
    """
    out = {}
    for k, img in cube_faces.items():
        out[k] = img.copy()
    
    # Process lateral faces
    for name in ('front', 'right', 'back', 'left'):
        face = out[name].astype(np.float64)  # Higher precision
        face_size = face.shape[0]
        
        # Get adjacent faces
        if name == 'front':
            left_face = out['left']
            right_face = out['right']
        elif name == 'right':
            left_face = out['front']
            right_face = out['back']
        elif name == 'back':
            left_face = out['right']
            right_face = out['left']
        elif name == 'left':
            left_face = out['back']
            right_face = out['front']
        
        # Adaptive blend width (wider for back face)
        if name == 'back':
            blend_width = face_size // 16  # More aggressive for back face
        else:
            blend_width = face_size // 32
            
        # Create smooth blending weights with cosine falloff
        left_weights = np.zeros((face_size, blend_width, 1))
        right_weights = np.zeros((face_size, blend_width, 1))
        
        for i in range(blend_width):
            # Cosine falloff for smoother transition
            weight = 0.5 * (1 - np.cos(np.pi * i / blend_width))
            left_weights[:, i, 0] = weight
            right_weights[:, blend_width-i-1, 0] = weight
        
        # Apply directional blending for left edge
        left_face_processed = left_face.astype(np.float64)
        # Apply bilateral filter to smoothen the transition while preserving edges
        if name == 'back' or name == 'left':
            left_face_processed = cv2.bilateralFilter(left_face.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50).astype(np.float64)
        
        face[:, :blend_width] = (
            (1 - left_weights) * face[:, :blend_width] + 
            left_weights * left_face_processed[:, -blend_width:]
        )
        
        # Apply directional blending for right edge
        right_face_processed = right_face.astype(np.float64)
        # Apply bilateral filter to smoothen the transition
        if name == 'back' or name == 'right':
            right_face_processed = cv2.bilateralFilter(right_face.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50).astype(np.float64)
        
        face[:, -blend_width:] = (
            (1 - right_weights) * face[:, -blend_width:] + 
            right_weights * right_face_processed[:, :blend_width]
        )
        
        out[name] = face.clip(0, 255).astype(np.uint8)
    
    return out

def advanced_pole_correction(cube_faces):
    """
    Advanced pole correction with adaptive processing.
    Uses only standard OpenCV functions.
    """
    out = cube_faces.copy()
    
    for pole in ('top', 'bottom'):
        face = cube_faces[pole].copy()
        face_size = face.shape[0]
        
        # Higher supersampling factor
        factor = 4
        
        # Step 1: Supersample with high-quality interpolation
        big = cv2.resize(face, (0, 0), fx=factor, fy=factor,
                        interpolation=cv2.INTER_LANCZOS4)
        
        # Step 2: Apply multiple standard filters
        # First bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(big, d=7, sigmaColor=25, sigmaSpace=25)
        
        # Apply gaussian blur as alternative to guided filter
        blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
        
        # Use addWeighted to enhance details (similar effect to guided filter)
        enhanced = cv2.addWeighted(bilateral, 1.5, blurred, -0.5, 0)
        
        # Step 3: Apply subtle sharpening to enhance details
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Downsample with high-quality area interpolation
        out[pole] = cv2.resize(sharpened, (face_size, face_size), 
                              interpolation=cv2.INTER_AREA)
    
    return out

def enhanced_pole_processing(cube_faces):
    """
    Advanced pole processing with adaptive supersampling and edge-aware filtering.
    """
    out = cube_faces.copy()
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Determine adaptive supersampling factor based on face size
    if face_size <= 256:
        factor = 2
    elif face_size <= 512:
        factor = 3
    else:
        factor = 4
    
    for pole in ('top', 'bottom'):
        face = cube_faces[pole]
        
        # Step 1: Supersample with Lanczos for high-quality upscaling
        big = cv2.resize(face, (0, 0), fx=factor, fy=factor,
                         interpolation=cv2.INTER_LANCZOS4)
        
        # Step 2: Apply edge-preserving bilateral filter to reduce noise while preserving edges
        big_filtered = cv2.bilateralFilter(big, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Step 3: Downsample with area interpolation
        down = cv2.resize(big_filtered, (face_size, face_size),
                          interpolation=cv2.INTER_AREA)
        
        # Step 4: Enhance visual quality with unsharp masking
        gaussian = cv2.GaussianBlur(down, (0, 0), 2)
        unsharp = cv2.addWeighted(down, 1.5, gaussian, -0.5, 0)
        out[pole] = np.clip(unsharp, 0, 255).astype(np.uint8)
    
    return out

def calculate_metrics(img_a, img_b):
    """
    Calculate comprehensive quality metrics between two images.
    
    Args:
        img_a, img_b: Input images (numpy arrays)
        
    Returns:
        Dictionary of metrics
    """
    # Ensure images are in the same format
    if img_a.dtype != np.float32:
        img_a = img_a.astype(np.float32)
    if img_b.dtype != np.float32:
        img_b = img_b.astype(np.float32)
    
    # Calculate difference
    diff = img_a - img_b
    squared_diff = diff ** 2
    mse = np.mean(squared_diff)
    psnr = 10 * np.log10((255.0 ** 2) / max(mse, 1e-10))
    
    # Calculate SSIM
    ssim_val = ssim(img_a, img_b, channel_axis=2, data_range=255.0)
    
    # Calculate specific region metrics
    H, W = img_a.shape[:2]
    
    # Seam at longitude ±π (left and right edges)
    seam_width = W // 50
    seam_diff_left = np.mean(np.abs(diff[:, :seam_width]))
    seam_diff_right = np.mean(np.abs(diff[:, -seam_width:]))
    seam_diff = (seam_diff_left + seam_diff_right) / 2
    
    # North pole region (top 10%)
    north_pole_height = H // 10
    north_pole_diff = np.mean(np.abs(diff[:north_pole_height]))
    
    # South pole region (bottom 10%)
    south_pole_height = H // 10
    south_pole_diff = np.mean(np.abs(diff[-south_pole_height:]))
    
    # Equator region (middle 20%)
    equator_start = int(H * 0.4)
    equator_end = int(H * 0.6)
    equator_diff = np.mean(np.abs(diff[equator_start:equator_end]))
    
    # Longitude bands analysis (front, right, back, left faces)
    longitude_diffs = {}
    longitude_bands = {
        'Front': (W // 4, W // 2),         # 90° to 180°
        'Right': (0, W // 4),              # 0° to 90°
        'Back': (W // 2, 3 * W // 4),      # 180° to 270°
        'Left': (3 * W // 4, W)            # 270° to 360°
    }
    
    for name, (start, end) in longitude_bands.items():
        band_diff = np.mean(np.abs(diff[:, start:end]))
        longitude_diffs[name] = band_diff
    
    # Diagnose issues
    diagnoses = []
    
    if seam_diff > 6.0:
        diagnoses.append("Severe seam issue at longitude ±π (back face center)")
    elif seam_diff > 3.0:
        diagnoses.append("Moderate seam issue at longitude ±π")
    
    if north_pole_diff > 5.0 or south_pole_diff > 5.0:
        diagnoses.append("Pole distortion detected")
    
    if max(longitude_diffs.values()) > 1.5 * min(longitude_diffs.values()):
        worst_band = max(longitude_diffs.items(), key=lambda x: x[1])[0]
        diagnoses.append(f"Uneven quality across longitude bands. {worst_band} face needs improvement")
    
    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "Seam_Diff": seam_diff,
        "North_Pole_Diff": north_pole_diff,
        "South_Pole_Diff": south_pole_diff,
        "Equator_Diff": equator_diff,
        "Longitude_Diffs": longitude_diffs,
        "Problem_Areas": longitude_diffs,
        "Diagnoses": diagnoses
    }

def visualize_conversion_comparison(original, original_recon, improved_recon, cube_faces_orig, cube_faces_improved):
    """
    Visualize the comparison between original and improved conversion.
    
    Args:
        original: Original equirectangular image
        original_recon: Reconstruction using original method
        improved_recon: Reconstruction using improved method
        cube_faces_orig: Original cubemap faces
        cube_faces_improved: Improved cubemap faces
    """
    plt.figure(figsize=(20, 10))
    
    # Row 1: Original and reconstructed equirectangular images
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original Equirectangular")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(original_recon)
    plt.title("Original Reconstruction")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(improved_recon)
    plt.title("Improved Reconstruction")
    plt.axis('off')
    
    # Row 2: Difference maps
    plt.subplot(2, 3, 4)
    diff_orig = np.abs(original.astype(np.float32) - original_recon.astype(np.float32))
    diff_orig = np.mean(diff_orig, axis=2)  # Average across channels
    plt.imshow(diff_orig, cmap='hot', vmin=0, vmax=50)
    plt.title("Original Difference Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    diff_improved = np.abs(original.astype(np.float32) - improved_recon.astype(np.float32))
    diff_improved = np.mean(diff_improved, axis=2)  # Average across channels
    plt.imshow(diff_improved, cmap='hot', vmin=0, vmax=50)
    plt.title("Improved Difference Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Improvement map (how much the error was reduced)
    plt.subplot(2, 3, 6)
    improvement = diff_orig - diff_improved
    plt.imshow(improvement, cmap='coolwarm', vmin=-10, vmax=30)
    plt.title("Improvement Map (Blue = Better)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare the back face specifically
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cube_faces_orig['back'])
    plt.title("Original 'Back' Face")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cube_faces_improved['back'])
    plt.title("Improved 'Back' Face")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    back_diff = np.abs(cube_faces_orig['back'].astype(np.float32) - cube_faces_improved['back'].astype(np.float32))
    back_diff = np.mean(back_diff, axis=2)
    plt.imshow(back_diff, cmap='hot')
    plt.title("Back Face Difference")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def load_equirectangular_image(file_path):
    """
    Load an equirectangular image from a file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy array containing the image in RGB format
    """
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {file_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def visualize_equirectangular(equirect_img, title="Equirectangular Panorama"):
    """
    Visualize an equirectangular panorama image.
    
    Args:
        equirect_img: Equirectangular panorama image
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(equirect_img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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

def compare_face_by_face(cube_faces_orig, cube_faces_improved):
    """
    Compare each face of the cubemap individually.
    
    Args:
        cube_faces_orig: Original cubemap faces
        cube_faces_improved: Improved cubemap faces
    """
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    for face_name in face_names:
        original_face = cube_faces_orig[face_name]
        improved_face = cube_faces_improved[face_name]
        
        # Calculate metrics
        mse = np.mean((original_face.astype(np.float32) - improved_face.astype(np.float32)) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / max(mse, 1e-10))
        ssim_val = ssim(original_face, improved_face, channel_axis=2, data_range=255.0)
        
        print(f"\n{face_name.capitalize()} Face Comparison:")
        print(f"MSE between original and improved: {mse:.2f}")
        print(f"PSNR between original and improved: {psnr:.2f} dB")
        print(f"SSIM between original and improved: {ssim_val:.4f}")
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_face)
        plt.title(f"Original '{face_name}' Face")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(improved_face)
        plt.title(f"Improved '{face_name}' Face")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = np.abs(original_face.astype(np.float32) - improved_face.astype(np.float32))
        diff_img = np.mean(diff, axis=2)
        plt.imshow(diff_img, cmap='hot', vmin=0, vmax=50)
        plt.title(f"Difference (Mean: {np.mean(diff_img):.2f})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def simple_fix_conversion(equirect_img, face_size=512):
    """
    A minimal approach that preserves the strengths of the original method
    while addressing only its key weaknesses.
    """
    # Step 1: Use the original algorithm as the foundation
    cube_faces = equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Apply minimal corrections for specific issues
    
    # Simple fix for the back face (the most problematic face)
    if 'back' in cube_faces:
        back = cube_faces['back'].copy()
        # Apply a mild bilateral filter to reduce noise while preserving edges
        cube_faces['back'] = cv2.bilateralFilter(back, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Simple fix for pole faces
    for pole in ['top', 'bottom']:
        if pole in cube_faces:
            face = cube_faces[pole].copy()
            # Supersample and downsample for better quality
            factor = 2  # Keep this modest
            big = cv2.resize(face, (0, 0), fx=factor, fy=factor, 
                           interpolation=cv2.INTER_LANCZOS4)
            cube_faces[pole] = cv2.resize(big, (face.shape[1], face.shape[0]), 
                                        interpolation=cv2.INTER_AREA)
    
    # Step 3: Use the original algorithm to convert back
    improved_equirect = cubemap_to_equirect(cube_faces, *equirect_img.shape[:2])
    
    return improved_equirect, cube_faces


def test_conversion(image_path, face_size=512):
    """
    Test and compare the original and improved conversion methods.
    
    Args:
        image_path: Path to the equirectangular image
        face_size: Size of cubemap faces
        
    Returns:
        Dictionary containing original and improved results and metrics
    """
    # Load the image
    print(f"Loading image from {image_path}...")
    if image_path.startswith(('http://', 'https://')):
        equirect_img = load_image_from_url(image_path)
    else:
        equirect_img = load_local_image(image_path)
    
    if equirect_img is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    print(f"Loaded image with shape {equirect_img.shape}")
    
    # Benchmark method (previously the high-quality conversion)
    print("\nTesting benchmark conversion method...")
    start_time = time.time()
    
    # Benchmark conversion: high_quality_conversion
    benchmark_equirect, cube_faces_benchmark = high_quality_conversion(equirect_img, face_size)
    
    benchmark_time = time.time() - start_time
    print(f"Benchmark method completed in {benchmark_time:.2f} seconds")
    
    # Calculate metrics for benchmark method
    metrics_benchmark = diagnose_conversion_issues(equirect_img, benchmark_equirect)
    ssim_benchmark = ssim(equirect_img, benchmark_equirect, channel_axis=2, data_range=255.0)
    metrics_benchmark['SSIM'] = ssim_benchmark
    
    print("\nBenchmark Method Metrics:")
    print(f"MSE: {metrics_benchmark['MSE']:.2f}")
    print(f"PSNR: {metrics_benchmark['PSNR']:.2f} dB")
    print(f"SSIM: {metrics_benchmark['SSIM']:.4f}")
    print(f"Seam Difference: {metrics_benchmark['Seam_Diff']:.2f}")
    print(f"North Pole Difference: {metrics_benchmark['North_Pole_Diff']:.2f}")
    print(f"South Pole Difference: {metrics_benchmark['South_Pole_Diff']:.2f}")
    print(f"Equator Difference: {metrics_benchmark['Equator_Diff']:.2f}")
    print("Longitude Differences:", {k: f"{v:.2f}" for k, v in metrics_benchmark['Longitude_Diffs'].items()})
    print("Diagnoses:", metrics_benchmark['Diagnoses'])
    
    # Super-high-quality method
    print("\nTesting super-high-quality conversion method...")
    start_time = time.time()
    
    # Super-high-quality conversion 
    improved_equirect, cube_faces_improved = super_high_quality_conversion(equirect_img, face_size)
    
    improved_time = time.time() - start_time
    print(f"Super-high-quality method completed in {improved_time:.2f} seconds")
    
    # Calculate metrics for improved method
    metrics_improved = diagnose_conversion_issues(equirect_img, improved_equirect)
    ssim_improved = ssim(equirect_img, improved_equirect, channel_axis=2, data_range=255.0)
    metrics_improved['SSIM'] = ssim_improved
    
    print("\nSuper-High-Quality Method Metrics:")
    print(f"MSE: {metrics_improved['MSE']:.2f}")
    print(f"PSNR: {metrics_improved['PSNR']:.2f} dB")
    print(f"SSIM: {metrics_improved['SSIM']:.4f}")
    print(f"Seam Difference: {metrics_improved['Seam_Diff']:.2f}")
    print(f"North Pole Difference: {metrics_improved['North_Pole_Diff']:.2f}")
    print(f"South Pole Difference: {metrics_improved['South_Pole_Diff']:.2f}")
    print(f"Equator Difference: {metrics_improved['Equator_Diff']:.2f}")
    print("Longitude Differences:", {k: f"{v:.2f}" for k, v in metrics_improved['Longitude_Diffs'].items()})
    print("Diagnoses:", metrics_improved['Diagnoses'])
    
    # Compare the improvements
    print("\nImprovement Summary:")
    mse_improvement = metrics_benchmark['MSE'] - metrics_improved['MSE']
    psnr_improvement = metrics_improved['PSNR'] - metrics_benchmark['PSNR']
    ssim_improvement = metrics_improved['SSIM'] - metrics_benchmark['SSIM']
    seam_improvement = metrics_benchmark['Seam_Diff'] - metrics_improved['Seam_Diff']
    
    print(f"MSE Reduction: {mse_improvement:.2f} ({mse_improvement / metrics_benchmark['MSE'] * 100:.1f}%)")
    print(f"PSNR Improvement: {psnr_improvement:.2f} dB")
    print(f"SSIM Improvement: {ssim_improvement:.4f} ({ssim_improvement / metrics_benchmark['SSIM'] * 100:.1f}%)")
    print(f"Seam Difference Reduction: {seam_improvement:.2f} ({seam_improvement / metrics_benchmark['Seam_Diff'] * 100:.1f}%)")
    
    # Per-face metrics
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    face_metrics = {}
    
    print("\nPer-Face Metrics:")
    for face_name in face_names:
        benchmark_face = cube_faces_benchmark[face_name]
        improved_face = cube_faces_improved[face_name]
        
        mse_face = np.mean((benchmark_face.astype(np.float32) - improved_face.astype(np.float32))**2)
        psnr_face = 10 * np.log10((255.0**2) / max(mse_face, 1e-10))
        ssim_face = ssim(benchmark_face, improved_face, channel_axis=2, data_range=255.0)
        
        # Store metrics
        face_metrics[face_name] = {
            'MSE': mse_face,
            'PSNR': psnr_face,
            'SSIM': ssim_face
        }
        
        # Calculate relation to benchmark face metrics
        benchmark_err = metrics_benchmark['Longitude_Diffs'].get(face_name.capitalize(), 0)
        improved_err = metrics_improved['Longitude_Diffs'].get(face_name.capitalize(), 0)
        
        abs_improvement = benchmark_err - improved_err
        rel_improvement = abs_improvement / max(benchmark_err, 1e-10) * 100
        
        print(f"\n{face_name.capitalize()} Face:")
        print(f"  MSE between benchmark and improved: {mse_face:.2f}")
        print(f"  PSNR between benchmark and improved: {psnr_face:.2f} dB")
        print(f"  SSIM between benchmark and improved: {ssim_face:.4f}")
        print(f"  Error in benchmark: {benchmark_err:.2f}")
        print(f"  Error in improved: {improved_err:.2f}")
        print(f"  Absolute improvement: {abs_improvement:.2f}")
        print(f"  Relative improvement: {rel_improvement:.1f}%")
    
    # Visualize the results
    visualize_conversion_comparison(
        equirect_img, 
        benchmark_equirect, 
        improved_equirect, 
        cube_faces_benchmark, 
        cube_faces_improved
    )
    
    return {
        'benchmark_metrics': metrics_benchmark,
        'improved_metrics': metrics_improved,
        'face_metrics': face_metrics,
        'benchmark_cubemap': cube_faces_benchmark,
        'improved_cubemap': cube_faces_improved,
        'benchmark_reconstruction': benchmark_equirect,
        'improved_reconstruction': improved_equirect
    }