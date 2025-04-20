# cubediff_utils_v3.py (or add to cubediff_utils.py)

import numpy as np
import cv2
import math
import time # Add time for testing
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity as ssim

# --- Constants ---
FACE_NAMES = ['front', 'right', 'back', 'left', 'top', 'bottom']

# --- Coordinate Transformation Helpers (Revised & Verified) ---

def _xyz_to_latlon(xyz):
    """ Convert XYZ array (..., 3) to Lat/Lon (..., 2) """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    # Handle potential division by zero for zero vector
    valid_norm = norm > 1e-10
    lon = np.zeros_like(x)
    lat = np.zeros_like(y)
    # Calculate only for valid norms to avoid warnings/errors
    # Using arcsin for latitude - convention: +90 deg is North Pole (0, 1, 0)
    # Using arctan2(x, z) for longitude - convention: 0 deg at +Z axis, +90 deg at +X axis
    lon[valid_norm] = np.arctan2(x[valid_norm], z[valid_norm])  # [-pi, pi]
    lat[valid_norm] = np.arcsin(np.clip(y[valid_norm] / norm[valid_norm], -1.0, 1.0)) # [-pi/2, pi/2]
    return np.stack([lat, lon], axis=-1) # Shape (..., 2)

def _latlon_to_xyz(latlon):
    """ Convert Lat/Lon array (..., 2) to XYZ (..., 3) """
    lat, lon = latlon[..., 0], latlon[..., 1]
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1) # Shape (..., 3)


def _cubemap_uv_to_xyz(face_idx, uv):
    """
    Convert cubemap UV (0 to 1 range, origin top-left) to XYZ direction vector.
    Matches standard cubemap layouts (e.g., OpenGL).
    """
    u, v = uv[..., 0], uv[..., 1]
    # Map uv [0, 1] to canonical [-1, 1] (center is 0,0)
    u_c = 2.0 * u - 1.0
    v_c = -(2.0 * v - 1.0) # Invert v because image V is top-down, canonical is bottom-up

    # Ensure x, y, z are arrays with the same shape as u_c/v_c
    if face_idx == 0:   # Front (+Z)
        x = u_c
        y = v_c
        z = np.ones_like(u_c) # Use ones_like for correct shape
    elif face_idx == 1: # Right (+X)
        x = np.ones_like(u_c) # Use ones_like
        y = v_c
        z = -u_c
    elif face_idx == 2: # Back (-Z)
        x = -u_c
        y = v_c
        z = -np.ones_like(u_c) # Use ones_like with negative sign
    elif face_idx == 3: # Left (-X)
        x = -np.ones_like(u_c) # Use ones_like with negative sign
        y = v_c
        z = u_c
    elif face_idx == 4: # Top (+Y)
        x = u_c
        y = np.ones_like(u_c) # Use ones_like
        z = -v_c # Matches inverse: Z = -V_canonical
    elif face_idx == 5: # Bottom (-Y)
        x = u_c
        y = -np.ones_like(u_c) # Use ones_like with negative sign
        z = v_c # Matches inverse: Z = V_canonical
    else:
        raise ValueError("Invalid face index")

    # Stack should work now as x, y, z are all arrays of the same shape
    xyz = np.stack([x, y, z], axis=-1)

    # --- CORRECTED NORMALIZATION START ---
    # Normalize vector using np.divide with 'where' argument for safety & broadcasting
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    # Initialize normalized vector as zeros
    xyz_normalized = np.zeros_like(xyz)
    # Define the condition where normalization is safe (norm > epsilon)
    valid_norm_mask = norm > 1e-10
    # Use np.divide specifying the output array and the condition
    # This performs xyz / norm element-wise only where valid_norm_mask is True,
    # leaving xyz_normalized as zeros elsewhere. Broadcasting works correctly here.
    np.divide(xyz, norm, out=xyz_normalized, where=valid_norm_mask)
    # --- CORRECTED NORMALIZATION END ---

    return xyz_normalized # Shape (..., 3)


def _xyz_to_cubemap_uv(xyz):
    """
    Convert XYZ direction vector to the face index and UV coordinates (0 to 1 range, origin top-left).
    Inverse of _cubemap_uv_to_xyz.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)

    max_abs = np.maximum.reduce([abs_x, abs_y, abs_z])
    # Handle edge cases/precision by checking slightly within the max value
    mask_eps = max_abs - 1e-6

    u_c = np.zeros_like(x) # Canonical U [-1, 1]
    v_c = np.zeros_like(x) # Canonical V [-1, 1] (Y-axis up)
    face_idx = np.full(x.shape, -1, dtype=int)

    # Right (+X face): x = max
    mask = (x > mask_eps)
    face_idx[mask] = 1
    u_c[mask] = -z[mask] / x[mask]
    v_c[mask] = y[mask] / x[mask]

    # Left (-X face): -x = max
    mask = (x < -mask_eps)
    face_idx[mask] = 3
    u_c[mask] = z[mask] / x[mask]
    v_c[mask] = y[mask] / x[mask]

    # Top (+Y face): y = max
    mask = (y > mask_eps)
    face_idx[mask] = 4
    u_c[mask] = x[mask] / y[mask]
    v_c[mask] = -z[mask] / y[mask] # Corrected based on _cubemap_uv_to_xyz

    # Bottom (-Y face): -y = max
    mask = (y < -mask_eps)
    face_idx[mask] = 5
    u_c[mask] = x[mask] / y[mask]
    v_c[mask] = z[mask] / y[mask] # Corrected based on _cubemap_uv_to_xyz

    # Front (+Z face): z = max
    mask = (z > mask_eps)
    face_idx[mask] = 0
    u_c[mask] = x[mask] / z[mask]
    v_c[mask] = y[mask] / z[mask]

    # Back (-Z face): -z = max
    mask = (z < -mask_eps)
    face_idx[mask] = 2
    u_c[mask] = -x[mask] / z[mask]
    v_c[mask] = y[mask] / z[mask]

    # Convert canonical u_c, v_c [-1, 1] to image UV [0, 1] (origin top-left)
    u = (u_c + 1.0) * 0.5
    v = (-v_c + 1.0) * 0.5 # Invert canonical V

    # Clamp to [0, 1] for safety
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    return face_idx, np.stack([u, v], axis=-1) # face_idx shape (...), uv shape (..., 2)

# --- Main Conversion Functions V3 ---

def equirect_to_cubemap_v3(equi_img, face_size):
    """ V3: Equirectangular to Cubemap using corrected coordinates and cv2.remap """
    equi_h, equi_w = equi_img.shape[:2]
    cubemap_faces = {}

    # Create pixel center grid for target cubemap face (U, V coordinates 0 to 1)
    v_pix, u_pix = np.meshgrid(np.arange(face_size), np.arange(face_size), indexing='ij')
    u = (u_pix + 0.5) / face_size # Horizontal [0, 1]
    v = (v_pix + 0.5) / face_size # Vertical [0, 1]
    uv = np.stack([u, v], axis=-1) # Shape (face_size, face_size, 2)

    for i, face_name in enumerate(FACE_NAMES):
        # Convert cubemap UV coordinates to XYZ directions
        xyz = _cubemap_uv_to_xyz(i, uv) # Shape (face_size, face_size, 3)

        # Convert XYZ directions to Lat/Lon coordinates
        latlon = _xyz_to_latlon(xyz) # Shape (face_size, face_size, 2)
        lat, lon = latlon[..., 0], latlon[..., 1]

        # Convert Lat/Lon to equirectangular pixel coordinates (map_x, map_y)
        # map_x corresponds to longitude, map_y corresponds to latitude
        # Longitude [-pi, pi] maps to X [0, W-1]
        # Latitude [-pi/2, pi/2] maps to Y [0, H-1] (North pole at Y=0)
        map_x = (lon / (2 * np.pi) + 0.5) * (equi_w -1) # Map [-pi, pi] -> [0, W-1]
        map_y = (-lat / np.pi + 0.5) * (equi_h -1)    # Map [-pi/2, pi/2] -> [0, H-1]

        # Use cv2.remap
        face_img = cv2.remap(
            equi_img,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LANCZOS4,
             # Wrap horizontally, reflect vertically (poles)
            borderMode=cv2.BORDER_WRAP
            # Using BORDER_REFLECT_101 might be better near poles than default BORDER_CONSTANT=0
            # borderMode=cv2.BORDER_REFLECT_101
        )
        cubemap_faces[face_name] = face_img

    return cubemap_faces

def cubemap_to_equirect_v3(cubemap_faces, equi_h, equi_w):
    """ V3: Cubemap to Equirectangular using corrected coordinates and cv2.remap """
    if not isinstance(cubemap_faces, dict):
         # If input is a list/array, convert to dict assuming standard order
         cubemap_faces = {name: face for name, face in zip(FACE_NAMES, cubemap_faces)}

    face_size = cubemap_faces['front'].shape[0]
    output_equi = np.zeros((equi_h, equi_w, cubemap_faces['front'].shape[2]), dtype=cubemap_faces['front'].dtype)

    # Create pixel center grid for the equirectangular output (Lat, Lon coordinates)
    y_pix, x_pix = np.meshgrid(np.arange(equi_h), np.arange(equi_w), indexing='ij')
    lat = -( (y_pix + 0.5) / equi_h - 0.5) * np.pi      # Map Y [0, H] -> Lat [pi/2, -pi/2]
    lon = ( (x_pix + 0.5) / equi_w - 0.5) * 2 * np.pi   # Map X [0, W] -> Lon [-pi, pi]
    latlon = np.stack([lat, lon], axis=-1) # Shape (equi_h, equi_w, 2)

    # Convert Lat/Lon grid to XYZ directions
    xyz = _latlon_to_xyz(latlon) # Shape (equi_h, equi_w, 3)

    # Convert XYZ directions to cubemap face index and UV coordinates [0, 1]
    face_indices, uv = _xyz_to_cubemap_uv(xyz) # face_indices (H, W), uv (H, W, 2)

    # Convert UV coordinates [0, 1] to pixel coordinates [0, face_size-1] for sampling
    # Need to match the coordinate system expected by cv2.remap (map_x = cols, map_y = rows)
    map_x = uv[..., 0] * (face_size - 1) # u maps to horizontal (col)
    map_y = uv[..., 1] * (face_size - 1) # v maps to vertical (row)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Perform remap for each face using the calculated maps and masks
    for i, face_name in enumerate(FACE_NAMES):
        face_img = cubemap_faces[face_name]
        mask = (face_indices == i)

        if np.any(mask): # Only remap if this face is used
            # Remap the *entire* map for the current face's pixels
            remapped_pixels = cv2.remap(
                face_img,
                map_x, # Pass the full maps
                map_y,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101 # Reflect is often good for edges
            )
            # Apply the mask to place pixels in the output image
            output_equi[mask] = remapped_pixels[mask]

    return output_equi

# --- Helper function to run comparison (similar to V2, now V3) ---
def test_conversion_v2(image_path, face_size=512):
    """Loads image, runs V3 conversion, returns metrics."""
    print(f"Loading image: {image_path}")
    equi_img_orig = cv2.imread(image_path)
    if equi_img_orig is None:
        print("Error loading image.")
        return None
    # Ensure image is RGB uint8 for consistent metrics
    if len(equi_img_orig.shape) == 2:
        equi_img_orig = cv2.cvtColor(equi_img_orig, cv2.COLOR_GRAY2RGB)
    elif equi_img_orig.shape[2] == 4:
        equi_img_orig = cv2.cvtColor(equi_img_orig, cv2.COLOR_RGBA2RGB)
    elif equi_img_orig.shape[2] == 3:
         equi_img_orig = cv2.cvtColor(equi_img_orig, cv2.COLOR_BGR2RGB)

    equi_img_orig = equi_img_orig.astype(np.uint8) # Ensure uint8

    equi_h, equi_w = equi_img_orig.shape[:2]
    print(f"Original Image Dimensions: H={equi_h}, W={equi_w}, Dtype={equi_img_orig.dtype}")

    print("Starting V3 Equirectangular -> Cubemap conversion...")
    start_time = time.time()
    cube_faces_v3 = equirect_to_cubemap_v3(equi_img_orig, face_size)
    print(f"V3 E2C completed in {time.time() - start_time:.2f} seconds")

    print("Starting V3 Cubemap -> Equirectangular conversion...")
    start_time = time.time()
    equi_img_reconstructed_v3 = cubemap_to_equirect_v3(cube_faces_v3, equi_h, equi_w)
    equi_img_reconstructed_v3 = equi_img_reconstructed_v3.astype(np.uint8) # Ensure uint8 for metrics
    print(f"V3 C2E completed in {time.time() - start_time:.2f} seconds")
    print(f"Reconstructed Image Dimensions: H={equi_img_reconstructed_v3.shape[0]}, W={equi_img_reconstructed_v3.shape[1]}, Dtype={equi_img_reconstructed_v3.dtype}")


    print("Calculating V3 metrics...")
    mse_v3 = mean_squared_error(equi_img_orig, equi_img_reconstructed_v3)
    # Handle potential zero MSE case for PSNR
    if mse_v3 == 0:
        psnr_v3 = float('inf')
    else:
        # Ensure data_range is appropriate for uint8
        psnr_v3 = peak_signal_noise_ratio(equi_img_orig, equi_img_reconstructed_v3, data_range=255)

    # SSIM parameters might need tuning, win_size must be odd and <= min(H, W) of patches
    # Use a smaller default window size for potentially small faces/images
    win_size = min(7, equi_h, equi_w, face_size)
    if win_size % 2 == 0: win_size -= 1 # Ensure odd
    if win_size < 3: win_size = 3

    ssim_v3 = ssim(equi_img_orig, equi_img_reconstructed_v3, channel_axis=-1, data_range=255, win_size=win_size, gaussian_weights=True)

    print("\nV3 Conversion Metrics:")
    print(f"  MSE: {mse_v3:.2f}")
    print(f"  PSNR: {psnr_v3:.2f} dB")
    print(f"  SSIM: {ssim_v3:.4f}")

    if mse_v3 < 100 and psnr_v3 > 30 and ssim_v3 > 0.9:
        print("\nSUCCESS: Target metrics achieved!")
    else:
        print("\nNOTE: Target metrics not yet fully achieved.")
        if mse_v3 >= 100: print(f"  - MSE target (< 100) missed.")
        if psnr_v3 <= 30: print(f"  - PSNR target (> 30) missed.")
        if ssim_v3 <= 0.9: print(f"  - SSIM target (> 0.9) missed.") # Corrected variable name ssim_v3->ssim_v3

    return {
        'original': equi_img_orig,
        'reconstructed_v3': equi_img_reconstructed_v3,
        'cubemap_v3': cube_faces_v3,
        'metrics_v3': {'MSE': mse_v3, 'PSNR': psnr_v3, 'SSIM': ssim_v3}
    }