import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


# ─────────────── Precompute mapping to speed up cubemap-to-panorama ───────────────
# def init_equirect_mapping(
#     H_face: int, W_face: int,
#     upsample: int = 8,
#     overlap: int = 8,
#     sigma: float = 4.0,
#     device: str = "cuda"
# ):
#     """
#     Precompute all tables and GPU kernel needed for very fast reprojection+blend.
#     Returns:
#       face_idx: [H2,W2] int array
#       ui, vi:   [H2,W2] int arrays
#       blend_kernel: torch.Tensor [3,1,1,2*ov+1] on `device`
#       H2, W2:   ints
#     """
#     # final panorama resolution
#     H2, W2 = H_face * upsample, W_face * upsample

#     # spherical coordinates
#     theta = np.linspace(-np.pi, np.pi, W2)
#     phi   = np.linspace(-np.pi/2, np.pi/2, H2)
#     th, ph = np.meshgrid(theta, phi)

#     # unit sphere
#     x = np.cos(ph) * np.sin(th)
#     y = np.sin(ph)
#     z = np.cos(ph) * np.cos(th)

#     # select face by max abs axis
#     absx, absy, absz = np.abs(x), np.abs(y), np.abs(z)
#     face_idx = np.argmax(np.stack([absx, absy, absz], axis=0), axis=0)

#     # compute UV→pixel indices per face
#     ui = np.zeros_like(face_idx, dtype=np.int64)
#     vi = np.zeros_like(face_idx, dtype=np.int64)
    

#     def get_uv(face_idx, x, y, z):
#         """
#         Map 3D sphere coords (x,y,z) → UV ∈ [0,1] for cubemap face face_idx.
#         """
#         if   face_idx == 0:   # +X
#             u = -z / x;    v = -y / x
#         elif face_idx == 1:   # -X
#             u =  z / -x;   v = -y / -x
#         elif face_idx == 2:   # +Y (top)
#             u =  x / y;    v =  z / y
#         elif face_idx == 3:   # -Y (bottom)
#             u =  x / -y;   v = -z / -y
#         elif face_idx == 4:   # +Z
#             u =  x / z;    v = -y / z
#         elif face_idx == 5:   # -Z
#             u = -x / -z;   v = -y / -z
#         else:
#             raise ValueError(f"Invalid face_idx {face_idx}")

#         # normalize from [-1,1] → [0,1]
#         return (u + 1)/2, (v + 1)/2


#     for f in range(face_idx.max()+1):
#         mask = face_idx == f
#         uf, vf = get_uv(f, x[mask], y[mask], z[mask])
#         uf = np.clip(uf, 0, 1); vf = np.clip(vf, 0, 1)
#         ui[mask] = np.round( uf * (W_face-1) ).astype(int)
#         vi[mask] = np.round((1-vf) * (H_face-1)).astype(int)

#     # build a depthwise conv1d kernel for blending
#     ks = 2*overlap + 1
#     coords = torch.arange(-overlap, overlap+1, dtype=torch.float32)
#     g = torch.exp(-(coords/sigma)**2/2)
#     g = (g / g.sum()).view(1,1,ks)           # [1,1,ks]
#     blend_kernel = g.repeat(3,1,1).unsqueeze(2).to(device)  # [3,1,1,ks]

#     return face_idx, ui, vi, blend_kernel, H2, W2, overlap

# # … call once at import or runtime:
# _face_idx, _ui, _vi, _blend_kernel, H2, W2, _ov = init_equirect_mapping(
#     H_face=64, W_face=64, upsample=8, overlap=8, sigma=4.0, device="cuda"
# )


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

# def cubemap_to_equirect(cube_faces, out_h, out_w):
#     """
#     Convert 6 cubemap faces to equirectangular panorama.
    
#     Args:
#         cube_faces: List of 6 numpy arrays representing cube faces
#         out_h: Output height
#         out_w: Output width
        
#     Returns:
#         Numpy array of equirectangular panorama
#     """
#     # Implementation similar to above but in reverse
#     # Placeholder implementation for now
#     # equirect = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#     """
#     Inverse of `equirect_to_cubemap`.
#     Args
#     ----
#     cube_faces : (6, H, W, 3) uint8/float32 array  – front, right, back, left, top, bottom
#     out_h, out_w : int                              – desired panorama size
#     Returns
#     -------
#     equirect : (out_h, out_w, 3) float32 in [0,1]
#     """
#     cube_faces = np.asarray(cube_faces)

#     # normalise only if the data are uint8
#     if cube_faces.dtype == np.uint8:
#         cube_faces = cube_faces.astype(np.float32) / 255.0
#     else:                                 # already float 0-1
#         cube_faces = cube_faces.astype(np.float32)

#     N  = cube_faces.shape[1]        # face resolution
#     yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")

#     # 1) spherical angles
#     phi   = (xx / out_w  - 0.5) * 2 * np.pi        # [-π, π]
#     theta = (0.5 - yy / out_h) * np.pi             # [-π/2, π/2]

#     # 2) unit vectors
#     cos_t = np.cos(theta)
#     xs = cos_t * np.cos(phi)
#     ys = cos_t * np.sin(phi)
#     zs = np.sin(theta)

#     ax, ay, az = np.abs(xs), np.abs(ys), np.abs(zs)

#     # allocate output
#     equirect = np.empty((out_h, out_w, 3), dtype=np.float32)

#     # helper to sample from a face with bilinear filtering
#     def sample(face_idx, s, t):
#         # (s,t) in [-1,1] ⇒ pixel coords
#         u = (s + 1) * 0.5 * (N - 1)
#         v = (t + 1) * 0.5 * (N - 1)

#         u0 = np.floor(u).astype(np.int32)
#         v0 = np.floor(v).astype(np.int32)
#         u1 = np.clip(u0 + 1, 0, N - 1)
#         v1 = np.clip(v0 + 1, 0, N - 1)

#         du = (u - u0)[..., None]
#         dv = (v - v0)[..., None]

#         # gather
#         c00 = cube_faces[face_idx, v0, u0]
#         c01 = cube_faces[face_idx, v0, u1]
#         c10 = cube_faces[face_idx, v1, u0]
#         c11 = cube_faces[face_idx, v1, u1]

#         return (1 - du) * (1 - dv) * c00 + \
#                du       * (1 - dv) * c01 + \
#                (1 - du) *  dv      * c10 + \
#                du       *  dv      * c11

#     # 3) face masks & projections
#     # --- +X (front) and –X (back)
#     mask = (ax >= ay) & (ax >= az)
#     front = mask & (xs > 0)
#     back  = mask & (xs <= 0)

#     equirect[front] = sample(0,
#                               ys[front] / ax[front],
#                              -zs[front] / ax[front])
#     equirect[back]  = sample(2,
#                              -ys[back] / ax[back],
#                              -zs[back] / ax[back])

#     # --- +Y (right) and –Y (left)
#     mask = (ay > ax) & (ay >= az)
#     right = mask & (ys > 0)
#     left  = mask & (ys <= 0)

#     equirect[right] = sample(1,
#                              -xs[right] / ay[right],
#                              -zs[right] / ay[right])
#     equirect[left]  = sample(3,
#                               xs[left] / ay[left],
#                              -zs[left] / ay[left])

#     # --- +Z (top) and –Z (bottom)
#     mask = (az > ax) & (az > ay)
#     top    = mask & (zs > 0)
#     bottom = mask & (zs <= 0)

#     equirect[top]    = sample(4,
#                                ys[top] / az[top],
#                                xs[top] / az[top])
#     equirect[bottom] = sample(5,
#                                ys[bottom] / az[bottom],
#                               -xs[bottom] / az[bottom])

#     return np.clip(equirect, 0, 1)


def project_to_equirect(faces: list[np.ndarray], out_h=None, out_w=None):
    """
    Very basic cubemap→equirectangular reprojection:
    - faces: list of 6 arrays [H, W, 3]
    - out_h/out_w: target equirect dims. Defaults to H, 2W.
    """
    H, W = faces[0].shape[:2]
    if out_h is None: out_h = H
    if out_w is None: out_w = 2 * W
    # polar coords
    theta = np.linspace(-np.pi, np.pi, out_w)
    phi   = np.linspace(-np.pi/2, np.pi/2, out_h)
    th, ph = np.meshgrid(theta, phi)
    # convert to unit vectors
    x = np.cos(ph) * np.sin(th)
    y = np.sin(ph)
    z = np.cos(ph) * np.cos(th)
    # decide face by largest abs coord
    absx, absy, absz = np.abs(x), np.abs(y), np.abs(z)
    face_idx = np.argmax([absx, absy, absz], axis=0)
    # and sample each face via its UV mapping
    out = np.zeros((out_h, out_w, 3), dtype=faces[0].dtype)
    for f in range(6):
        mask = face_idx == f
        # compute u,v on that face from x,y,z — see any cubemap UV tutorial
        # ... for brevity, call a helper get_uv(f, x,y,z) → (u,v) in [0,1]
        u, v = get_uv(f, x[mask], y[mask], z[mask])
        # ui = (u * (W-1)).round().astype(int)
        # vi = ((1-v) * (H-1)).round().astype(int)
        # out[mask] = faces[f][vi, ui]
        
        # clamp UV to [0,1] and safely convert to integer indices
        u = np.nan_to_num(u, nan=0.5)
        v = np.nan_to_num(v, nan=0.5)
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        ui = (u * (W-1)).round().astype(int)
        vi = ((1 - v) * (H-1)).round().astype(int)
        # ensure we never index out of bounds
        ui = np.clip(ui, 0, W-1)
        vi = np.clip(vi, 0, H-1)

        out[mask] = faces[f][vi, ui]
    return out


# def cubemap_to_equirect(faces, overlap=8, sigma=4):
#     # assume faces: list of 6 [H,W,3] arrays
#     # pad each face horizontally by overlap from its neighbors
#     padded = []
#     for i in range(6):
#         left = faces[(i-1)%6][:, -overlap:, :]
#         right= faces[(i+1)%6][:, :overlap, :]
#         face = np.concatenate([left, faces[i], right], axis=1)  # W+2*ov
#         padded.append(face)
#     # standard projection on these padded faces...
#     eq = project_to_equirect(padded)  # yields Hx2W
#     # then apply horizontal Gaussian smoothing to first and last overlap regions
#     # for x in [0, overlap, eq.shape[1]-overlap, eq.shape[1]]:
#     #     # blend across seam
#     #     eq[:, x-overlap:x+overlap] = gaussian_filter(eq[:, x-overlap:x+overlap],
#     #                                                   sigma=[sigma, sigma, 0])
    
#     # then apply horizontal Gaussian smoothing to the overlap regions:
#     for x in [0, overlap, eq.shape[1] - overlap, eq.shape[1]]:
#         # extract patch
#         patch = eq[:, x - overlap : x + overlap]        # dtype may be float16
#         # cast to float32 for filtering
#         patch32 = patch.astype(np.float32)
#         filtered = gaussian_filter(patch32, sigma=[sigma, sigma, 0])
#         # cast back to original dtype
#         eq[:, x - overlap : x + overlap] = filtered.astype(eq.dtype)
    
    
#     return eq    


# def cubemap_to_equirect(
#     faces: list[np.ndarray],
#     out_h: int = None,
#     out_w: int = None,
# ):
#     """
#     Very fast projection + blend.
#     faces: list of 6 arrays [H2,W2,3] (after VAE upsample).
#     Returns: pano [H2,W2,3] NumPy.
#     out_h/out_w are ignored (precomputed mapping drives the true H2,W2).
#     Returns: pano [H2,W2,3] NumPy.
#     """
#     # 1) stack faces into [6,H2,W2,3]
#     # 1) stack faces into [6,H2,W2,3]
#     arr = np.stack(faces, axis=0)

#     # 2) gather with precomputed indices
#     pano = arr[_face_idx, _vi, _ui]  # [H2,W2,3]

#     # 3) GPU‐blend via depthwise conv1d
#     t = torch.from_numpy(pano).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()

#     # circular pad width by overlap
#     t = F.pad(t, (_ov,_ov,0,0), mode="circular")  # [1,3,H2,W2+2ov]
#     t = F.conv2d(t, _blend_kernel, groups=3)       # [1,3,H2,W2]
#     t = t[..., _ov:-_ov]                           # trim

#     return t.squeeze(0).permute(1,2,0).cpu().numpy()


def get_uv(face_idx: int, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Map 3D sphere coords (x,y,z) → UV ∈ [0,1] for cubemap face face_idx.
    x,y,z: 1D arrays of the same length.
    Returns (u,v) arrays in [0,1].
    """
    if face_idx == 0:    # +X
        u = -z / x
        v = -y / x
    elif face_idx == 1:  # -X
        u =  z / -x
        v = -y / -x
    elif face_idx == 2:  # +Y (top)
        u =  x / y
        v =  z / y
    elif face_idx == 3:  # -Y (bottom)
        u =  x / -y
        v = -z / -y
    elif face_idx == 4:  # +Z
        u =  x / z
        v = -y / z
    elif face_idx == 5:  # -Z
        u = -x / -z
        v = -y / -z
    else:
        raise ValueError(f"Invalid face_idx {face_idx}")

    # normalize from [-1,1] → [0,1]
    return (u + 1) * 0.5, (v + 1) * 0.5


# ── 1) Precompute the cubemap→equirect mapping once ──
_face_idx = _ui = _vi = None
def init_cubemap_map(H, W, out_h=None, out_w=None, device="cuda"):
    global _face_idx, _ui, _vi
    if out_h is None: out_h = H
    if out_w is None: out_w = 2 * W

    θ = torch.linspace(-torch.pi, torch.pi, out_w, device=device)
    φ = torch.linspace(-torch.pi/2, torch.pi/2, out_h, device=device)
    th, ph = torch.meshgrid(θ, φ, indexing="xy")

    x =  torch.cos(ph) * torch.sin(th)
    y =  torch.sin(ph)
    z =  torch.cos(ph) * torch.cos(th)

    absx, absy, absz = x.abs(), y.abs(), z.abs()
    face_idx = torch.argmax(torch.stack([absx, absy, absz], 0), 0)

    ui = torch.zeros_like(x, dtype=torch.long)
    vi = torch.zeros_like(x, dtype=torch.long)
    for f in range(6):
        m = face_idx == f
        xf, yf, zf = x[m], y[m], z[m]
        # spherical → face-UV
        u, v = get_uv(f, xf, yf, zf)    # implement per-face formulas below
        ui[m] = (u.clamp(0,1) * (W-1)).round().long()
        vi[m] = ((1-v).clamp(0,1) * (H-1)).round().long()

    _face_idx, _ui, _vi = face_idx, ui, vi
    

# ── 2) The fast GPU projector + seam-blend ──
def cubemap_to_equirect(faces: torch.Tensor, overlap: int = 8, sigma: float = 4.0):
    """
    faces: Tensor[6, H, W, 3]       (already up→equirect latent→RGB size)
    returns: Tensor[H, 2W, 3]
    """
    device = faces.device
    if _face_idx is None:
        H,W = faces.shape[1], faces.shape[2]
        init_cubemap_map(H, W, device=device)

    # gather
    pano = faces[_face_idx, _vi, _ui]    # [H,2W,3]

    # GPU seam-blend with depthwise conv1d
    # build once per overlap/sigma
    kernel_size = 2*overlap + 1
    half = torch.arange(-overlap, overlap+1, device=device).float()
    gauss = torch.exp(-0.5*(half/sigma)**2)
    gauss /= gauss.sum()
    gauss = gauss.view(1,1,1,-1).repeat(3,1,1,1)  # [3,1,1,K]

    t = pano.permute(2,0,1).unsqueeze(0)           # [1,3,H,2W]
    t = F.pad(t, (overlap,overlap,0,0), mode="circular")
    t = F.conv2d(t, gauss, groups=3)               # smooth horizontally
    t = t[..., overlap:-overlap]                  # trim
    return t.squeeze(0).permute(1,2,0)


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