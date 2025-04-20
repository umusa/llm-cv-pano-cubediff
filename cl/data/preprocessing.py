import numpy as np
import torch
from PIL import Image
import cv2
import os
from tqdm import tqdm

def equirect_to_cubemap(equirect_img, face_size=512):
    """
    Convert equirectangular panorama to 6 cubemap faces.
    
    Args:
        equirect_img: PIL Image or numpy array of equirectangular panorama
        face_size: Size of each cubemap face
        
    Returns:
        List of 6 numpy arrays representing cube faces
    """
    if isinstance(equirect_img, Image.Image):
        equirect_img = np.array(equirect_img)
    
    # Get height and width of equirectangular image
    h, w = equirect_img.shape[:2]
    
    # Create output cubemap faces
    cube_faces = []
    
    # Define the 6 cube faces (order: front, right, back, left, top, bottom)
    for i in range(6):
        # Create output face
        face = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        
        # Fill face with corresponding pixels from equirectangular image
        for y in range(face_size):
            for x in range(face_size):
                # Convert cube coordinates to 3D vector
                x_norm = 2 * (x + 0.5) / face_size - 1
                y_norm = 2 * (y + 0.5) / face_size - 1
                
                # Map based on which face
                if i == 0:   # Front
                    vec = [1.0, x_norm, -y_norm]
                elif i == 1: # Right
                    vec = [-x_norm, 1.0, -y_norm]
                elif i == 2: # Back
                    vec = [-1.0, -x_norm, -y_norm]
                elif i == 3: # Left
                    vec = [x_norm, -1.0, -y_norm]
                elif i == 4: # Top
                    vec = [x_norm, y_norm, 1.0]
                elif i == 5: # Bottom
                    vec = [x_norm, -y_norm, -1.0]
                
                # Normalize vector
                norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                vec = [v / norm for v in vec]
                
                # Convert 3D vector to equirectangular coordinates
                phi = np.arctan2(vec[1], vec[0])
                theta = np.arcsin(vec[2])
                
                # Map to equirectangular pixel coordinates
                u = (phi / (2 * np.pi) + 0.5) * w
                v = (0.5 - theta / np.pi) * h
                
                # Bilinear interpolation
                u0, v0 = int(u), int(v)
                u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
                
                # Get pixel values
                try:
                    p00 = equirect_img[v0, u0]
                    p01 = equirect_img[v0, u1]
                    p10 = equirect_img[v1, u0]
                    p11 = equirect_img[v1, u1]
                    
                    # Weights
                    du, dv = u - u0, v - v0
                    
                    # Interpolate
                    pixel = (1 - du) * (1 - dv) * p00 + du * (1 - dv) * p01 + \
                            (1 - du) * dv * p10 + du * dv * p11
                    
                    face[y, x] = pixel.astype(np.uint8)
                except IndexError:
                    # Handle edge cases
                    u, v = int(u) % w, int(v) % h
                    face[y, x] = equirect_img[v, u]
        
        cube_faces.append(face)
    
    return cube_faces

def cubemap_to_equirect(cube_faces, out_h, out_w):
    """
    Convert 6 cubemap faces to equirectangular panorama.
    
    Args:
        cube_faces: List of 6 numpy arrays representing cube faces
        out_h: Output height
        out_w: Output width
        
    Returns:
        Numpy array of equirectangular panorama
    """
    # Implementation similar to above but in reverse
    # ...
    
    # Placeholder implementation for now
    equirect = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return equirect

def preprocess_panorama_dataset(input_dir, output_dir, face_size=512, num_samples=None):
    """
    Process a directory of equirectangular panoramas to cubemap faces.
    
    Args:
        input_dir: Directory containing equirectangular panoramas
        output_dir: Output directory for cubemap faces
        face_size: Size of each cubemap face
        num_samples: Number of samples to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List all panorama files
    pano_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if num_samples is not None:
        pano_files = pano_files[:num_samples]
    
    for pano_file in tqdm(pano_files, desc="Processing panoramas"):
        # Load equirectangular panorama
        equirect_path = os.path.join(input_dir, pano_file)
        equirect_img = Image.open(equirect_path).convert('RGB')
        
        # Convert to cubemap
        cube_faces = equirect_to_cubemap(equirect_img, face_size)
        
        # Save cubemap faces
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        base_name = os.path.splitext(pano_file)[0]
        
        for i, face in enumerate(cube_faces):
            face_dir = os.path.join(output_dir, base_name)
            os.makedirs(face_dir, exist_ok=True)
            
            face_path = os.path.join(face_dir, f"{face_names[i]}.jpg")
            Image.fromarray(face).save(face_path)