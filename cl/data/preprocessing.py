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

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import OpenEXR
import Imath
import requests

def read_exr_file(filepath):
    """
    Read an EXR file and convert it to a numpy array with improved tone mapping.
    
    Args:
        filepath: Path to the EXR file
        
    Returns:
        numpy array containing the image data
    """
    # Open the input file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        exr_file = OpenEXR.InputFile(filepath)
    except Exception as e:
        raise IOError(f"Failed to open EXR file: {str(e)}")
    
    # Get the header and determine dimensions
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get available channels
    channels = list(header['channels'].keys())
    
    # Read pixel data for RGB channels as float32
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    try:
        # Try standard RGB channels first
        if 'R' in channels and 'G' in channels and 'B' in channels:
            (red_str, green_str, blue_str) = [exr_file.channel(c, FLOAT) for c in 'RGB']
        # Otherwise try the first three channels available
        elif len(channels) >= 3:
            (red_str, green_str, blue_str) = [exr_file.channel(c, FLOAT) for c in channels[:3]]
        else:
            raise ValueError(f"Could not find enough color channels in EXR file")
    except Exception as e:
        print(f"Error reading channels: {str(e)}")
        raise
    
    # Convert strings to numpy arrays
    red = np.frombuffer(red_str, dtype=np.float32)
    green = np.frombuffer(green_str, dtype=np.float32)
    blue = np.frombuffer(blue_str, dtype=np.float32)
    
    # Reshape and create RGB array
    red.shape = green.shape = blue.shape = (height, width)
    rgb = np.dstack((red, green, blue))
    
    # Improved tone mapping for HDR to LDR conversion
    # Using an exposure-based approach with highlight recovery
    exposure = 1.5  # Adjustable exposure value (higher = brighter)
    rgb_exposed = rgb * exposure
    
    # Apply a filmic tone mapping curve (simplified ACES)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # Apply the tone mapping formula
    rgb_mapped = (rgb_exposed * (a * rgb_exposed + b)) / (rgb_exposed * (c * rgb_exposed + d) + e)
    
    # Ensure values are in proper range
    rgb_mapped = np.clip(rgb_mapped, 0, 1)
    
    # Apply gamma correction for better screen display
    gamma = 1.0 / 2.2
    rgb_gamma = np.power(rgb_mapped, gamma)
    
    # Convert to 8-bit for saving to JPG
    rgb_8bit = np.clip(rgb_gamma * 255, 0, 255).astype(np.uint8)
    
    return rgb_8bit

def download_exr_panoramas(urls, output_dir):
    """
    Download EXR panoramas from URLs and convert them to JPG format.
    
    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for url in tqdm(urls, desc="Downloading panoramas"):
        # Extract filename from URL
        filename = os.path.basename(url)
        filepath = os.path.join(output_dir, filename)
        
        # Download the file if it doesn't exist
        if not os.path.exists(filepath):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                continue
        
        # Convert EXR to JPG if it doesn't exist yet
        jpg_filepath = filepath.replace('.exr', '.jpg')
        if not os.path.exists(jpg_filepath) and filepath.endswith('.exr'):
            try:
                exr_img = read_exr_file(filepath)
                Image.fromarray(exr_img).save(jpg_filepath)
                print(f"Converted: {filename} to {os.path.basename(jpg_filepath)}")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

def preprocess_panorama_dataset(input_dir, output_dir, face_size=512, num_samples=None, visualize=False):
    """
    Process a directory of equirectangular panoramas to cubemap faces.
    Handles both regular image formats and EXR files.
    
    Args:
        input_dir: Directory containing equirectangular panoramas
        output_dir: Output directory for cubemap faces
        face_size: Size of each cubemap face
        num_samples: Number of samples to process (None for all)
        visualize: Whether to visualize the panorama and cubemap faces
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List all panorama files (both standard formats and JPGs converted from EXR)
    pano_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
    
    if num_samples is not None:
        pano_files = pano_files[:num_samples]
    
    for pano_file in tqdm(pano_files, desc="Processing panoramas"):
        # Load equirectangular panorama
        equirect_path = os.path.join(input_dir, pano_file)
        try:
            equirect_img = Image.open(equirect_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {pano_file}: {str(e)}")
            continue
        
        # Convert to cubemap
        try:
            cube_faces = equirect_to_cubemap(equirect_img, face_size)
        except Exception as e:
            print(f"Error converting {pano_file} to cubemap: {str(e)}")
            continue
        
        # Save cubemap faces
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        base_name = os.path.splitext(pano_file)[0]
        
        # Create directory for this panorama's faces
        face_dir = os.path.join(output_dir, base_name)
        os.makedirs(face_dir, exist_ok=True)
        
        # Save each face
        face_paths = []
        for i, face in enumerate(cube_faces):
            face_path = os.path.join(face_dir, f"{face_names[i]}.jpg")
            Image.fromarray(face).save(face_path)
            face_paths.append(face_path)
        
        # Visualize if requested
        if visualize:
            visualize_panorama_and_cubemap(equirect_path, face_paths, face_names)

def visualize_panorama_and_cubemap(panorama_path, face_paths, face_names):
    """
    Visualize the original panorama and the corresponding cubemap faces.
    
    Args:
        panorama_path: Path to the original panorama
        face_paths: List of paths to the cubemap faces
        face_names: List of names for the cubemap faces
    """
    # Load the panorama
    panorama = Image.open(panorama_path)
    
    # Load the faces
    face_images = []
    for path in face_paths:
        face_images.append(np.array(Image.open(path)))
    
    # Create the figure
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    
    # Display the original panorama
    ax_pano = plt.subplot(gs[0, :])
    ax_pano.imshow(panorama)
    ax_pano.set_title(f"Original Panorama: {os.path.basename(panorama_path)}", fontsize=16)
    ax_pano.axis('off')
    
    # Display the cubemap faces
    for i, (face_name, face_img) in enumerate(zip(face_names, face_images)):
        ax = plt.subplot(gs[1 + i//3, i%3])
        ax.imshow(face_img)
        ax.set_title(face_name, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_captions_for_new_panoramas():
    """
    Create captions for new panoramas and save to a JSON file.
    """
    captions = {
        "artist_workshop_2k": "A well-lit artist workshop with painting equipment, natural lighting from windows, and various art tools arranged throughout the space",
        "art_studio_2k": "A bright art studio with large windows, white walls, and various artist supplies and easels arranged around the room",
        "veranda_2k": "An elegant outdoor veranda with stone columns, overlooking a garden landscape with natural lighting and architectural details",
        "modern_buildings_2_2k": "A contemporary urban scene with sleek glass skyscrapers, reflective surfaces, and modern architectural elements under a blue sky",
        "winter_evening_2k": "A serene winter landscape at evening time with snow-covered ground, dramatic sky colors, and soft golden lighting"
    }
    
    # Save captions to JSON file
    os.makedirs("../data/processed", exist_ok=True)
    with open("../data/processed/captions.json", "w") as f:
        json.dump(captions, f, indent=4)
    
    return captions

# Main function to tie everything together
def process_hdri_panoramas(download=True):
    """
    Process HDRI panoramas: download, convert, and create cubemaps.
    
    Args:
        download: Whether to download the panoramas (True) or use existing ones (False)
    """
    # Define the sample URLs
    sample_urls = [
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/artist_workshop_2k.exr",
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/art_studio_2k.exr",
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/veranda_2k.exr",
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/modern_buildings_2_2k.exr",
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/winter_evening_2k.exr"
    ]
    
    # Create directories
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed/cubemaps", exist_ok=True)
    
    # Download panoramas if requested
    if download:
        download_exr_panoramas(sample_urls, "../data/raw")
    
    # Create captions for the panoramas
    create_captions_for_new_panoramas()
    
    # Process the downloaded panoramas to cubemaps
    preprocess_panorama_dataset(
        input_dir="../data/raw",
        output_dir="../data/processed/cubemaps",
        face_size=512,
        visualize=True  # Set to True to visualize the results
    )

# Example usage:
# process_hdri_panoramas(download=True)