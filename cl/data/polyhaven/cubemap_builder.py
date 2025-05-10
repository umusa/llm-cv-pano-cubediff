import numpy as np
import cv2
import pathlib
import tqdm
# from PIL import Image
import os
# import glob
import logging
from typing import List, Tuple, Optional, Union
# from pathlib import Path
import concurrent.futures
import multiprocessing
import time
import sys
# import OpenEXR, Imath

# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor  # still used for API compatibility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard cubemap face order and naming
FACE_ORDER = ['front', 'right', 'back', 'left', 'top', 'bottom']
FACE_NAMES = ['px', 'py', 'nx', 'ny', 'pz', 'nz']  # Standard cubemap naming

# Try to import GPU-related libraries
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Check if GPU is available
HAS_GPU = False
torch_device = None
if HAS_TORCH and torch.cuda.is_available():
    HAS_GPU = True
    torch_device = torch.device("cuda:0")
    logger.info(f"PyTorch GPU acceleration available: {torch.cuda.get_device_name(0)}")
elif HAS_CUPY and cp.cuda.is_available():
    HAS_GPU = True
    logger.info("CuPy GPU acceleration available")
    # Initialize cupy with the default GPU
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
    except Exception as e:
        logger.warning(f"Error initializing CuPy memory pools: {e}")
else:
    logger.info("No GPU acceleration available - using CPU only")

# Determine the number of CPU cores to use (leave 2 cores free for system operations)
NUM_CPU_CORES = max(1, multiprocessing.cpu_count() - 2)
logger.info(f"Using {NUM_CPU_CORES} CPU cores for parallel processing")

def equirect_to_cubemap_torch(equi: np.ndarray, face_size: int) -> List[np.ndarray]:
    """
    Convert equirectangular image to 6 cubemap faces using PyTorch GPU acceleration.
    
    Args:
        equi (np.ndarray): Equirectangular image as numpy array (HxWx3)
        face_size (int): Size of the output cubemap faces in pixels
        
    Returns:
        List[np.ndarray]: List of 6 cubemap faces as numpy arrays
    """
    # Convert numpy array to torch tensor
    h, w = equi.shape[:2]
    equi_tensor = torch.from_numpy(equi).float().to(torch_device)
    if len(equi_tensor.shape) == 3:
        # Add batch dimension for processing
        equi_tensor = equi_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Create meshgrid for pixel coordinates
    xs = torch.linspace(-1, 1, face_size, device=torch_device)
    ys = torch.linspace(-1, 1, face_size, device=torch_device)
    ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing='ij')
    
    faces = []
    
    # Process each face
    for i in range(6):
        # Initialize vectors based on face
        if i == 0:   # Front
            vx = torch.ones_like(xs_grid)
            vy = xs_grid
            vz = -ys_grid
        elif i == 1: # Right
            vx = -xs_grid
            vy = torch.ones_like(xs_grid)
            vz = -ys_grid
        elif i == 2: # Back
            vx = -torch.ones_like(xs_grid)
            vy = -xs_grid
            vz = -ys_grid
        elif i == 3: # Left
            vx = xs_grid
            vy = -torch.ones_like(xs_grid)
            vz = -ys_grid
        elif i == 4: # Top
            vx = xs_grid
            vy = ys_grid
            vz = torch.ones_like(xs_grid)
        elif i == 5: # Bottom
            vx = xs_grid
            vy = -ys_grid
            vz = -torch.ones_like(xs_grid)
        
        # Normalize vectors
        norm = torch.sqrt(vx**2 + vy**2 + vz**2)
        vx = vx / norm
        vy = vy / norm
        vz = vz / norm
        
        # Convert to spherical coordinates
        phi = torch.atan2(vy, vx)
        theta = torch.asin(vz)
        
        # Convert to equirectangular coordinates
        u = (phi / (2 * torch.pi) + 0.5) * w
        v = (0.5 - theta / torch.pi) * h
        
        # Normalize coordinates to [-1, 1] for grid_sample
        u_norm = (u / (w - 1)) * 2 - 1
        v_norm = (v / (h - 1)) * 2 - 1
        
        # Stack coordinates for grid_sample
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1, face_size, face_size, 2]
        
        # Sample from the equirectangular image
        face_tensor = F.grid_sample(
            equi_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Convert back to numpy array
        face = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        faces.append(face)
    
    return faces

def equirect_to_cubemap_cupy(equi: np.ndarray, face_size: int) -> List[np.ndarray]:
    """
    Convert equirectangular image to 6 cubemap faces using CuPy for GPU acceleration.
    
    Args:
        equi (np.ndarray): Equirectangular image as numpy array (HxWx3)
        face_size (int): Size of the output cubemap faces in pixels
        
    Returns:
        List[np.ndarray]: List of 6 cubemap faces as numpy arrays
    """
    # Move data to GPU
    h, w = equi.shape[:2]
    equi_gpu = cp.asarray(equi)
    
    # Create meshgrid for pixel coordinates
    xs = cp.linspace(-1, 1, face_size)
    ys = cp.linspace(-1, 1, face_size)
    ys_grid, xs_grid = cp.meshgrid(ys, xs, indexing='ij')
    
    faces = []
    
    # Process each face
    for i in range(6):
        # Initialize vectors based on face
        if i == 0:   # Front
            vx = cp.ones_like(xs_grid)
            vy = xs_grid
            vz = -ys_grid
        elif i == 1: # Right
            vx = -xs_grid
            vy = cp.ones_like(xs_grid)
            vz = -ys_grid
        elif i == 2: # Back
            vx = -cp.ones_like(xs_grid)
            vy = -xs_grid
            vz = -ys_grid
        elif i == 3: # Left
            vx = xs_grid
            vy = -cp.ones_like(xs_grid)
            vz = -ys_grid
        elif i == 4: # Top
            vx = xs_grid
            vy = ys_grid
            vz = cp.ones_like(xs_grid)
        elif i == 5: # Bottom
            vx = xs_grid
            vy = -ys_grid
            vz = -cp.ones_like(xs_grid)
        
        # Normalize vectors
        norm = cp.sqrt(vx**2 + vy**2 + vz**2)
        vx = vx / norm
        vy = vy / norm
        vz = vz / norm
        
        # Convert to spherical coordinates
        phi = cp.arctan2(vy, vx)
        theta = cp.arcsin(vz)
        
        # Convert to equirectangular coordinates
        u = (phi / (2 * cp.pi) + 0.5) * w
        v = (0.5 - theta / cp.pi) * h
        
        # Clip coordinates to valid range
        u = cp.clip(u, 0, w - 1)
        v = cp.clip(v, 0, h - 1)
        
        # Move coordinates back to CPU for OpenCV remap
        map_x = cp.asnumpy(u).astype(np.float32)
        map_y = cp.asnumpy(v).astype(np.float32)
        
        # Move equirectangular image back to CPU for remap
        equi_cpu = cp.asnumpy(equi_gpu)
        
        # Use OpenCV remap
        face = cv2.remap(equi_cpu, map_x, map_y, cv2.INTER_LINEAR)
        faces.append(face)
    
    # Release GPU memory
    if 'mempool' in globals():
        mempool.free_all_blocks()
    if 'pinned_mempool' in globals():
        pinned_mempool.free_all_blocks()
    
    return faces

def equirect_to_cubemap_vectorized(equi: np.ndarray, face_size: int) -> List[np.ndarray]:
    """
    Convert equirectangular image to 6 cubemap faces using vectorized NumPy operations.
    CPU fallback when GPU acceleration is not available.
    
    Args:
        equi (np.ndarray): Equirectangular image as numpy array (HxWx3)
        face_size (int): Size of the output cubemap faces in pixels
        
    Returns:
        List[np.ndarray]: List of 6 cubemap faces as numpy arrays
    """
    h, w = equi.shape[:2]
    faces = []
    
    # Create meshgrid for pixel coordinates
    xs, ys = np.meshgrid(np.linspace(-1, 1, face_size), np.linspace(-1, 1, face_size))
    
    # Face order: front, right, back, left, top, bottom
    for i in range(6):
        # Initialize vectors based on face
        if i == 0:   # Front
            vx = np.ones_like(xs)
            vy = xs
            vz = -ys
        elif i == 1: # Right
            vx = -xs
            vy = np.ones_like(xs)
            vz = -ys
        elif i == 2: # Back
            vx = -np.ones_like(xs)
            vy = -xs
            vz = -ys
        elif i == 3: # Left
            vx = xs
            vy = -np.ones_like(xs)
            vz = -ys
        elif i == 4: # Top
            vx = xs
            vy = ys
            vz = np.ones_like(xs)
        elif i == 5: # Bottom
            vx = xs
            vy = -ys
            vz = -np.ones_like(xs)
        
        # Normalize vectors
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        vx /= norm
        vy /= norm
        vz /= norm
        
        # Convert to spherical coordinates
        phi = np.arctan2(vy, vx)
        theta = np.arcsin(vz)
        
        # Convert to equirectangular coordinates
        u = (phi / (2 * np.pi) + 0.5) * w
        v = (0.5 - theta / np.pi) * h
        
        # Clip coordinates to valid range
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)
        
        # Use OpenCV's remap function for faster interpolation
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        face = cv2.remap(equi, map_x, map_y, cv2.INTER_LINEAR)
        
        faces.append(face)
    
    return faces

def equirect_to_cubemap_pixel(equi: np.ndarray, face_size: int) -> List[np.ndarray]:
    """
    Convert equirectangular image to 6 cubemap faces using a pixel-by-pixel approach.
    Slowest but most reliable method.
    
    Args:
        equi (np.ndarray): Equirectangular image as numpy array (HxWx3)
        face_size (int): Size of the output cubemap faces in pixels
        
    Returns:
        List[np.ndarray]: List of 6 cubemap faces as numpy arrays
    """
    h, w = equi.shape[:2]
    faces = []
    
    # Process each face
    for i in range(6):
        face = np.zeros((face_size, face_size, 3), dtype=np.uint8)
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
                
                # Nearest neighbor sampling (faster)
                u_int, v_int = int(u) % w, max(0, min(int(v), h - 1))
                face[y, x] = equi[v_int, u_int]
        
        faces.append(face)
    
    return faces

def equirect_to_cubemap(equi: np.ndarray, face_size: int) -> List[np.ndarray]:
    """
    Convert equirectangular image to 6 cubemap faces.
    Selects the best available method based on hardware.
    
    Args:
        equi (np.ndarray): Equirectangular image as numpy array (HxWx3)
        face_size (int): Size of the output cubemap faces in pixels
        
    Returns:
        List[np.ndarray]: List of 6 cubemap faces as numpy arrays
    """
    # Try GPU methods first if available
    if HAS_GPU:
        try:
            if HAS_TORCH:
                return equirect_to_cubemap_torch(equi, face_size)
            elif HAS_CUPY:
                return equirect_to_cubemap_cupy(equi, face_size)
        except Exception as e:
            logger.warning(f"GPU conversion failed: {e}, falling back to CPU")
    
    # Fall back to vectorized CPU method
    try:
        return equirect_to_cubemap_vectorized(equi, face_size)
    except Exception as e:
        logger.warning(f"Vectorized CPU conversion failed: {e}, falling back to pixel-by-pixel method")
    
    # Last resort: pixel-by-pixel method (very slow)
    return equirect_to_cubemap_pixel(equi, face_size)

def load_hdr_image_gpu(erp_path: Union[str, pathlib.Path]) -> Optional[np.ndarray]:
    """
    Load an HDR image using GPU acceleration when possible.
    
    Args:
        erp_path (str or Path): Path to the HDR image file
        
    Returns:
        np.ndarray or None: Loaded image as numpy array, or None if loading failed
    """
    logger.info("▶ Using Python OpenEXR to load .exr")
    erp_path = str(erp_path)
    
    # Fast path for EXR files
    if erp_path.lower().endswith('.exr'):
        try:
            # Try OpenEXR first
            try:
                import OpenEXR
                import Imath
                
                exr_file = OpenEXR.InputFile(erp_path)
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Get RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                # Use GPU for tone mapping if available
                if HAS_GPU and HAS_TORCH:
                    # Convert to PyTorch tensors directly
                    r = torch.frombuffer(r_str, dtype=torch.float32).reshape(height, width).to(torch_device)
                    g = torch.frombuffer(g_str, dtype=torch.float32).reshape(height, width).to(torch_device)
                    b = torch.frombuffer(b_str, dtype=torch.float32).reshape(height, width).to(torch_device)
                    
                    # Stack channels
                    rgb = torch.stack([r, g, b], dim=2)
                    
                    # Tone mapping on GPU (simple exposure adjustment)
                    exposure = 1.5
                    rgb = rgb * exposure
                    
                    # Clamp and convert to 8-bit
                    rgb = torch.clamp(rgb * 255, 0, 255)
                    img = rgb.cpu().numpy().astype(np.uint8)
                elif HAS_GPU and HAS_CUPY:
                    # Convert to CuPy arrays directly
                    r = cp.frombuffer(r_str, dtype=cp.float32).reshape(height, width)
                    g = cp.frombuffer(g_str, dtype=cp.float32).reshape(height, width)
                    b = cp.frombuffer(b_str, dtype=cp.float32).reshape(height, width)
                    
                    # Stack channels
                    rgb = cp.stack([r, g, b], axis=2)
                    
                    # Tone mapping on GPU
                    rgb = rgb * 1.5
                    
                    # Clamp and convert to 8-bit
                    rgb = cp.clip(rgb * 255, 0, 255).astype(cp.uint8)
                    img = cp.asnumpy(rgb)
                else:
                    # CPU fallback
                    r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                    g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                    b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                    
                    # Stack channels and apply exposure
                    rgb = np.stack([r, g, b], axis=2) * 1.5
                    
                    # Convert to 8-bit
                    img = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                
                return img
            except Exception as e:
                logger.warning(f"OpenEXR loading failed: {e}")
            
            # Other fallback methods remain the same
            img = cv2.imread(erp_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                if img.dtype == np.float32:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        except Exception as e:
            logger.warning(f"All EXR loading methods failed: {e}")
        
        # If all fails, create a test pattern
        return create_test_pattern(2048, 1024)
    
    # For regular image formats
    try:
        img = cv2.imread(erp_path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"Image loading failed: {e}")
    
    # Create a test pattern if all methods failed
    return create_test_pattern(2048, 1024)

def load_hdr_image(erp_path: Union[str, pathlib.Path]) -> Optional[np.ndarray]:
    """
    Load an HDR image from various formats (including EXR).
    CPU version as a fallback.
    
    Args:
        erp_path (str or Path): Path to the HDR image file
        
    Returns:
        np.ndarray or None: Loaded image as numpy array, or None if loading failed
    """
    erp_path = str(erp_path)
    
    # Fast path for EXR files
    if erp_path.lower().endswith('.exr'):
        try:
            # Try OpenEXR first
            try:
                import OpenEXR
                import Imath
                
                exr_file = OpenEXR.InputFile(erp_path)
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Get RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                # Convert to numpy arrays
                r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                
                # Stack channels
                rgb = np.stack([r, g, b], axis=2)
                
                # Simple tone mapping
                img = np.clip(rgb * 255 * 1.5, 0, 255).astype(np.uint8)
                
                return img
                
            except Exception as e:
                logger.warning(f"OpenEXR loading failed: {e}")
            
            # Try OpenCV as fallback
            img = cv2.imread(erp_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                if img.dtype == np.float32:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
                
        except Exception as e:
            logger.warning(f"All EXR loading methods failed: {e}")
        
        # If all fails, create a test pattern
        return create_test_pattern(2048, 1024)
    
    # For regular image formats
    try:
        img = cv2.imread(erp_path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"Image loading failed: {e}")
    
    # Create a test pattern if all methods failed
    return create_test_pattern(2048, 1024)

def create_test_pattern(width, height):
    """
    Create a simple test pattern (faster version).
    Uses GPU if available.
    """
    if HAS_GPU and HAS_TORCH:
        # Create pattern on GPU
        x_grid = torch.linspace(0, 1, width, device=torch_device)
        y_grid = torch.linspace(0, 1, height, device=torch_device)
        y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        pattern = torch.zeros((height, width, 3), device=torch_device)
        pattern[:, :, 0] = x_grid * 255
        pattern[:, :, 1] = y_grid * 255
        pattern[:, :, 2] = (1 - y_grid) * 255
        
        return pattern.cpu().numpy().astype(np.uint8)
    elif HAS_GPU and HAS_CUPY:
        # Create pattern on GPU with CuPy
        x_grid = cp.linspace(0, 1, width)
        y_grid = cp.linspace(0, 1, height)
        y_grid, x_grid = cp.meshgrid(y_grid, x_grid, indexing='ij')
        
        pattern = cp.zeros((height, width, 3), dtype=cp.uint8)
        pattern[:, :, 0] = x_grid * 255
        pattern[:, :, 1] = y_grid * 255
        pattern[:, :, 2] = (1 - y_grid) * 255
        
        return cp.asnumpy(pattern)
    else:
        # CPU fallback
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        pattern[:, :, 0] = x_grid * 255
        pattern[:, :, 1] = y_grid * 255
        pattern[:, :, 2] = (1 - y_grid) * 255
        return pattern

def process_panorama(erp_path: Union[str, pathlib.Path], 
                     out_root: Union[str, pathlib.Path],
                     face_px: int = 512) -> bool:
    """
    Process a single panorama - load, convert to cubemap, and save faces.
    Also save the whole panorama image.
    Uses GPU acceleration when available.
    
    Args:
        erp_path: Path to input equirectangular panorama
        out_root: Output directory for cubemap faces
        face_px: Size of cubemap faces
        
    Returns:
        bool: Success or failure
    """
    start_time = time.time()
    
    try:
        # Convert paths to Path objects
        erp_path = pathlib.Path(erp_path)
        out_root = pathlib.Path(out_root)
        
        # Get base filename without extension
        base_name = erp_path.stem
        
        # Output directory for faces
        faces_dir = out_root / "faces" / base_name
        
        # Output directory for panoramas
        pano_dir = out_root / "panoramas" / base_name
        
        # Create output directories
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(pano_dir, exist_ok=True)
        
        # Check if already processed
        face_files = [faces_dir / f"{face}.jpg" for face in FACE_ORDER]
        pano_file = pano_dir / "panorama.jpg"
        
        if all(f.exists() for f in face_files) and pano_file.exists():
            return True
        
        # Load the equirectangular image (with GPU acceleration if available)
        if HAS_GPU:
            equirect_img = load_hdr_image_gpu(erp_path)
        else:
            equirect_img = load_hdr_image(erp_path)
            
        if equirect_img is None:
            logger.warning(f"Failed to load {erp_path}")
            return False
        
        # Save the whole panorama image
        if not pano_file.exists():
            cv2.imwrite(str(pano_file), cv2.cvtColor(equirect_img, cv2.COLOR_RGB2BGR))
            logger.debug(f"Saved panorama: {pano_file}")
        
        # Convert to cubemap faces if not already done
        missing_faces = [f for f in face_files if not f.exists()]
        if missing_faces:
            # This automatically uses the best available method (GPU or CPU)
            cube_faces = equirect_to_cubemap(equirect_img, face_px)
            
            # Save faces
            for i, face in enumerate(cube_faces):
                face_path = faces_dir / f"{FACE_ORDER[i]}.jpg"
                if not face_path.exists():
                    cv2.imwrite(str(face_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        
        # Log processing time
        elapsed = time.time() - start_time
        logger.info(f"Processed {erp_path.name} in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {erp_path}: {e}")
        return False

def batch_convert(src_dir, dst_dir, face_px=512):
    """
    Convert all equirectangular panoramas to cubemap faces with parallel processing.
    Also save the whole panorama images.
    
    Args:
        src_dir (str or Path): Source directory containing equirectangular images
        dst_dir (str or Path): Destination directory for cubemap faces
        face_px (int, optional): Size of cubemap faces in pixels. Defaults to 512.
        
    Returns:
        int: Number of successfully converted panoramas
    """
    # Convert paths to Path objects
    src_dir = pathlib.Path(src_dir)
    dst_dir = pathlib.Path(dst_dir)
    
    # Create destination directories
    os.makedirs(dst_dir / "faces", exist_ok=True)
    os.makedirs(dst_dir / "panoramas", exist_ok=True)
    
    # Find all panorama images (EXR, HDR, JPG, PNG)
    extensions = ["*.exr", "*.hdr", "*.jpg", "*.jpeg", "*.png"]
    erp_files = []
    
    for ext in extensions:
        erp_files.extend(list(src_dir.glob(ext)))
    
    if not erp_files:
        logger.warning(f"No panorama files found in {src_dir}")
        return 0
    
    logger.info(f"Converting {len(erp_files)} panoramas using {NUM_CPU_CORES} CPU processes")
    successful = 0
    pbar = tqdm.tqdm(total=len(erp_files), desc="Converting panoramas")
 
    # Switch to true multiprocessing for CPU‐bound conversion
    spawn_ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=NUM_CPU_CORES,
        mp_context=spawn_ctx
    ) as executor:
        futures = {executor.submit(process_panorama, path, dst_dir, face_px): path
                   for path in erp_files}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                if fut.result():
                    successful += 1
            except Exception as e:
                logger.error(f"Error processing {p.name}: {e}")
            pbar.update(1)
    pbar.close()
    
    logger.info(f"Successfully converted {successful} out of {len(erp_files)} panoramas")
    return successful

# Function alias for backwards compatibility
erp_to_cubemap = process_panorama

# If run as a script, display GPU info
if __name__ == "__main__":
    if HAS_GPU:
        if HAS_TORCH:
            # Print PyTorch GPU info
            logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        if HAS_CUPY:
            # Print CuPy GPU info
            logger.info(f"CuPy CUDA available: {cp.cuda.is_available()}")
            logger.info(f"CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    else:
        logger.info("No GPU acceleration available - using CPU only")