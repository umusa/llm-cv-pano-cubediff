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
import math
# import py360convert as py360
from skimage.metrics import structural_similarity as ssim

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: centred pixel grid in [-1,1]
def _grid(N):
    jj, ii = np.meshgrid(np.arange(N)+0.5,
                         np.arange(N)+0.5,
                         indexing='ij')
    return 2*ii/N - 1, 2*jj/N - 1     # u (X‑axis), v (Y‑axis)


# ─── pole supersample & down‑filter ──────────────────────────────────────────
def _supersample_pole(face, factor=2):
    big = cv2.resize(face, (face.shape[1]*factor, face.shape[0]*factor),
                     interpolation=cv2.INTER_LANCZOS4)
    # 2× box filter (exact area)
    return cv2.resize(big, (face.shape[1], face.shape[0]),
                      interpolation=cv2.INTER_AREA)

# ─── cubic (Catmull‑Rom) sampler for inverse projector ──────────────────────
# def _cubic_lerp(f, ui, vi):
#     ui0 = np.clip(ui.astype(np.int32)-1, 0, f.shape[1]-1)
#     vi0 = np.clip(vi.astype(np.int32)-1, 0, f.shape[0]-1)
#     patch = f[vi0[:,None]+np.arange(4), ui0[:,None]+np.arange(4)[:,None]]  # 4×4
#     tx = ui - (ui0+1)
#     ty = vi - (vi0+1)
#     def w(t): return (-0.5*t+1.5)*t*t - 1.5*t + 1
#     wx = np.array([w(tx+1), w(tx), w(tx-1), w(tx-2)])
#     wy = np.array([w(ty+1), w(ty), w(ty-1), w(ty-2)])
#     return (wy[:,:,None,None]*wx[:,None,:,None]*patch).sum((0,1))


# def _cubic_lerp(face, u, v):
#     """
#     face : H×H×3  uint8
#     u,v  : 1‑D float arrays of same length  (pixel coords inside this face)

#     Returns an N×3 uint8 array of sampled colours.
#     """
#     H = face.shape[0]
#     ui0 = np.clip(np.floor(u).astype(np.int32) - 1, 0, H-1)
#     vi0 = np.clip(np.floor(v).astype(np.int32) - 1, 0, H-1)

#     # gather 4 rows × 4 cols around each point
#     patch = np.zeros((len(u), 4, 4, 3), dtype=np.float32)
#     for dy in range(4):
#         rows = np.clip(vi0 + dy, 0, H-1)
#         row_pixels = face[rows]
#         for dx in range(4):
#             cols = np.clip(ui0 + dx, 0, H-1)
#             patch[:, dy, dx] = row_pixels[np.arange(len(rows)), cols]

#     tx = u - (ui0 + 1)
#     ty = v - (vi0 + 1)

#     def w(t):
#         return ((-0.5*t + 1.5)*t - 1.5)*t + 1   # Catmull–Rom cubic weight

#     wx = np.stack([w(tx+1), w(tx), w(tx-1), w(tx-2)], axis=1)  # N×4
#     wy = np.stack([w(ty+1), w(ty), w(ty-1), w(ty-2)], axis=1)  # N×4

#     # tensor dot: (N×4)·(N×4×4×3)·(N×4)^T  →  N×3
#     interp = (wy[:, :, None, None] *
#               wx[:, None, :, None] *
#               patch).sum(axis=(1,2))

#     return np.clip(interp, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------
# Equirectangular -> Cubemap  (OpenGL axis convention)
# ---------------------------------------------------------------------
def equirect_to_cubemap(equi, face_size):
    H, W = equi.shape[:2]
    faces = {}
    for name in ('front','right','back','left','top','bottom'):
        u, v = _pixel_grid(face_size)
        if   name == 'front':   x, y, z =  u,   -v,  1
        elif name == 'right':   x, y, z =  1,   -v, -u
        elif name == 'back':    x, y, z = -u,   -v, -1
        elif name == 'left':    x, y, z = -1,   -v,  u
        elif name == 'top':     x, y, z =  u,    1,  v
        elif name == 'bottom':  x, y, z =  u,   -1, -v

        inv = 1./np.sqrt(x*x+y*y+z*z)
        x*=inv; y*=inv; z*=inv
        lon = np.arctan2(x, z) + np.pi
        lat = np.arcsin(y)
        map_x = (lon/(2*np.pi)*W).astype(np.float32)
        map_y = ((0.5 - lat/np.pi)*H).astype(np.float32)
        faces[name] = cv2.remap(equi, map_x, map_y,
                                interpolation=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_WRAP)
    faces = _blend_seam(faces)
    faces = _antialias_poles(faces)
    return faces


# ------------------------------------------------------------
# Equirect  ➜  Cubemap
# ------------------------------------------------------------

# def equirect_to_cubemap(img_equi: np.ndarray, face_size: int):
#     H, W = img_equi.shape[:2]
#     out, names = {}, ['front','right','back','left','top','bottom']

#     for name in names:
#         u, v = _grid(face_size)

#         # right‑handed +X right, +Y up, +Z forward
#         if   name == 'front':   x, y, z =  u,   -v, 1
#         elif name == 'right':   x, y, z =  1,   -v, -u
#         elif name == 'back':    x, y, z = -u,   -v, -1
#         elif name == 'left':    x, y, z = -1,   -v,  u
#         elif name == 'top':     x, y, z =  u,    1,  v
#         elif name == 'bottom':  x, y, z =  u,   -1, -v

#         x = x / np.sqrt(x*x + y*y + z*z)
#         y = y / np.sqrt(x*x + y*y + z*z)
#         z = z / np.sqrt(x*x + y*y + z*z)

#         lon = np.arctan2(x, z) + np.pi          # [0,2π)
#         lat = np.arcsin(y)                      # [‑π/2,π/2]

#         map_x = (lon / (2*np.pi) * W).astype(np.float32)
#         map_y = ((0.5 - lat/np.pi) * H).astype(np.float32)

#         out[name] = cv2.remap(img_equi, map_x, map_y,
#                               interpolation=cv2.INTER_LANCZOS4,
#                               borderMode=cv2.BORDER_WRAP)
#     return out

# ─── substitute the bodies of the two public functions ───────────────────────

# def equirect_to_cubemap(img_equi: np.ndarray, face_size: int):
#     """
#     Wrapper around py360convert.e2c  (equirect → cubemap dictionary).

#     Returns a dict with keys:
#         'front', 'right', 'back', 'left', 'top', 'bottom'
#     in exactly that order.  The images are uint8, shape (face_size, face_size, 3).
#     """
#     #   py360 expects float 32 image in [0,1]
#     # cube = py360.e2c(img_equi.astype(np.float32)/255.,
#     #                  face_w=face_size,
#     #                  cube_format='dict')            # gives a dict of six faces
#     # # convert back to uint8
#     # return {k: (v*255).astype(np.uint8) for k, v in cube.items()}

#     cube = py360.e2c(img_equi.astype(np.float32)/255.,
#                      face_w=face_size,
#                      cube_format='dict')        # keys: F,R,B,L,U,D

#     keymap = {'F':'front', 'R':'right', 'B':'back',
#               'L':'left',  'U':'top',   'D':'bottom'}
#     return {keymap[k]: (v*255).astype(np.uint8) for k, v in cube.items()}

# def equirect_to_cubemap(equi: np.ndarray, face_size: int):
#     H, W = equi.shape[:2]
#     out = {}
#     for name in ('front','right','back','left','top','bottom'):
#         u, v = _grid(face_size)

#         if   name == 'front':  x, y, z =  u,   -v,  1
#         elif name == 'right':  x, y, z =  1,   -v, -u
#         elif name == 'back':   x, y, z = -u,   -v, -1
#         elif name == 'left':   x, y, z = -1,   -v,  u
#         elif name == 'top':    x, y, z =  u,    1,  v
#         elif name == 'bottom': x, y, z =  u,   -1, -v

#         inv_len = 1./np.sqrt(x*x + y*y + z*z)
#         x *= inv_len; y *= inv_len; z *= inv_len

#         lon = np.arctan2(x, z) + np.pi        # 0…2π
#         lat = np.arcsin(y)                    # -π/2…π/2

#         map_x = (lon / (2*np.pi) * W).astype(np.float32)
#         map_y = ((0.5 - lat/np.pi) * H).astype(np.float32)

#         # out[name] = cv2.remap(equi, map_x, map_y,
#         #                       interpolation=cv2.INTER_LANCZOS4,
#         #                       borderMode=cv2.BORDER_WRAP)
        
#         face_img = cv2.remap(equi, map_x, map_y,
#                               interpolation=cv2.INTER_LANCZOS4,
#                               borderMode=cv2.BORDER_WRAP)
#         if name in ('top','bottom'):
#              face_img = _supersample_pole(face_img)   # 2× AA
#         out[name] = face_img

#     return out


def cubemap_to_equirect(cube, H, W):
    F = cube['front'].shape[0]
    lon = (np.arange(W)+0.5)/W*2*np.pi - np.pi
    lat =  np.pi/2 - (np.arange(H)+0.5)/H*np.pi
    lon, lat = np.meshgrid(lon, lat)

    x = np.sin(lon)*np.cos(lat)
    y = np.sin(lat)
    z = np.cos(lon)*np.cos(lat)
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    face_id = np.empty_like(x, int)

    # dominant‐axis test
    face_id[(az>=ax)&(az>=ay)&( z>0)] = 0   # front
    face_id[(ax> az)&(ax>=ay)&( x>0)] = 1   # right
    face_id[(az>=ax)&(az>=ay)&( z<0)] = 2   # back
    face_id[(ax> az)&(ax>=ay)&( x<0)] = 3   # left
    face_id[(ay> ax)&(ay> az)&( y>0)] = 4   # top
    face_id[(ay> ax)&(ay> az)&( y<0)] = 5   # bottom

    u = np.zeros_like(x, np.float32)
    v = np.zeros_like(x, np.float32)
    eps = 1e-8
    m = face_id==0; u[m]= x[m]/(az[m]+eps); v[m]=-y[m]/(az[m]+eps)
    m = face_id==1; u[m]=-z[m]/(ax[m]+eps); v[m]=-y[m]/(ax[m]+eps)
    m = face_id==2; u[m]=-x[m]/(az[m]+eps); v[m]=-y[m]/(az[m]+eps)
    m = face_id==3; u[m]= z[m]/(ax[m]+eps); v[m]=-y[m]/(ax[m]+eps)
    m = face_id==4; u[m]= x[m]/(ay[m]+eps); v[m]= z[m]/(ay[m]+eps)
    m = face_id==5; u[m]= x[m]/(ay[m]+eps); v[m]=-z[m]/(ay[m]+eps)

    u = ((u+1)*0.5*(F-1)).astype(np.float32)
    v = ((v+1)*0.5*(F-1)).astype(np.float32)

    out = np.zeros((H, W, 3), np.uint8)
    order = ['front','right','back','left','top','bottom']
    for fid, name in enumerate(order):
        m = face_id == fid
        if not m.any():
            continue
        ys, xs = np.where(m)
        y0,y1 = ys.min(), ys.max()+1
        x0,x1 = xs.min(), xs.max()+1
        out[y0:y1, x0:x1] = cv2.remap(
            cube[name], u[y0:y1, x0:x1], v[y0:y1, x0:x1],
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_WRAP)
    return out

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _pixel_grid(n):
    jj, ii = np.meshgrid(np.arange(n)+0.5,
                         np.arange(n)+0.5,
                         indexing='ij')
    return 2*ii/n - 1, 2*jj/n - 1                 # u, v in [-1,1]

def _blend_seam(cube):
    """Average first & last two cols of lateral faces for C0 seam."""
    out = cube.copy()
    for k in ('front','right','back','left'):
        f = cube[k].astype(np.float32)
        f[:, 0:2]  = 0.5*(f[:, 0:2] + f[:, -2:])
        f[:, -2:]  = f[:, 0:2]
        out[k] = f.astype(np.uint8)
    return out

def _antialias_poles(cube, factor=2):
    """Supersample U/D faces then downsample with box filter."""
    out = cube.copy()
    for k in ('top','bottom'):
        f = cube[k]
        big = cv2.resize(f, (0,0), fx=factor, fy=factor,
                         interpolation=cv2.INTER_LANCZOS4)
        out[k] = cv2.resize(big, f.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    return out


# ────────────────────────────────────────────────────────────────
# Cubemap  ➜  Equirectangular   (robust OpenCV remap version)
# ────────────────────────────────────────────────────────────────
def cubemap_to_equirect(faces: dict, H: int, W: int):
    """
    faces : dict with keys 'front','right','back','left','top','bottom'
            each of shape (F, F, 3), uint8
    H, W  : desired equirectangular output size
    """

    # 1. create full‑resolution lon/lat grid (pixel centres)
    lon = (np.arange(W)+0.5)/W * 2*np.pi - np.pi
    lat =  np.pi/2 - (np.arange(H)+0.5)/H * np.pi
    lon, lat = np.meshgrid(lon, lat)          # each H×W

    # 2. 3‑D directions
    x = np.sin(lon)*np.cos(lat)
    y = np.sin(lat)
    z = np.cos(lon)*np.cos(lat)
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # 3. which face?
    idx = np.zeros_like(x, dtype=np.int8)
    idx[(az>=ax)&(az>=ay)&( z>0)] = 0     # front  +Z
    idx[(ax>az)&(ax>=ay)&( x>0)] = 1      # right  +X
    idx[(az>=ax)&(az>=ay)&( z<0)] = 2     # back   -Z
    idx[(ax>az)&(ax>=ay)&( x<0)] = 3      # left   -X
    idx[(ay>ax)&(ay> az)&( y>0)] = 4      # top    +Y
    idx[(ay>ax)&(ay> az)&( y<0)] = 5      # bottom -Y

    # 4. build remap coordinates for every pixel *once*
    F  = faces['front'].shape[0]
    u  = np.empty_like(x, np.float32)
    v  = np.empty_like(x, np.float32)
    eps = 1e-8

    # front
    m = idx==0; u[m] =  x[m]/( az[m]+eps); v[m] = -y[m]/( az[m]+eps)
    # right
    m = idx==1; u[m] = -z[m]/( ax[m]+eps); v[m] = -y[m]/( ax[m]+eps)
    # back
    m = idx==2; u[m] = -x[m]/( az[m]+eps); v[m] = -y[m]/( az[m]+eps)
    # left
    m = idx==3; u[m] =  z[m]/( ax[m]+eps); v[m] = -y[m]/( ax[m]+eps)
    # top
    m = idx==4; u[m] =  x[m]/( ay[m]+eps); v[m] =  z[m]/( ay[m]+eps)
    # bottom
    m = idx==5; u[m] =  x[m]/( ay[m]+eps); v[m] = -z[m]/( ay[m]+eps)

    u = ((u+1)*0.5*(F-1)).astype(np.float32)
    v = ((v+1)*0.5*(F-1)).astype(np.float32)

    # 5. allocate output and fill per face via cv2.remap
    equi = np.zeros((H, W, 3), dtype=np.uint8)
    names = ['front','right','back','left','top','bottom']
    for fid, name in enumerate(names):
        m = idx==fid
        if not m.any():            # just in case
            continue
        # remap expects full maps, so build minimal bounding box to save RAM
        ys, xs = np.where(m)
        y0,y1 = ys.min(), ys.max()+1
        x0,x1 = xs.min(), xs.max()+1

        map_x = u[y0:y1, x0:x1]
        map_y = v[y0:y1, x0:x1]
        equi[y0:y1, x0:x1] = cv2.remap(
            faces[name], map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_WRAP
        )
    return equi


# def cubemap_to_equirect(faces: dict, H: int, W: int):
#     """
#     Inverse wrapper around py360convert.c2e  (cubemap → equirect).

#     `faces` is the dict returned by equirect_to_cubemap.
#     """
#     # py360 wants a single (6, H, W, 3) array ordered FRBLUD
#     order = ['front', 'right', 'back', 'left', 'top', 'bottom']
#     cube   = np.stack([faces[k] for k in order]).astype(np.float32)/255.

#     eq = py360.c2e(cube, h=H, w=W, cube_format='dict')      # float32 [0,1]
#     return (eq*255).astype(np.uint8)


# def cubemap_to_equirect(faces: dict, H: int, W: int):
#     """
#     Convert cubemap faces back to an H×W equirectangular image using py360convert.
#     `faces` keys: 'front', 'right', 'back', 'left', 'top', 'bottom'.
#     """
#     # --------------- 1. rename + convert to float32 [0,1] -------------------
#     frblud = {
#         'F': faces['front' ].astype(np.float32) / 255.,
#         'R': faces['right' ].astype(np.float32) / 255.,
#         'B': faces['back'  ].astype(np.float32) / 255.,
#         'L': faces['left'  ].astype(np.float32) / 255.,
#         'U': faces['top'   ].astype(np.float32) / 255.,
#         'D': faces['bottom'].astype(np.float32) / 255.,
#     }

#     # --------------- 2. pad width/height to multiples py360 requires -------
#     W8 = (W + 7) & ~7      # next multiple of 8
#     H8 = (H + 3) & ~3      # next multiple of 4

#     eq = py360.c2e(frblud, h=H8, w=W8, cube_format='dict')   # float32 [0,1]

#     # --------------- 3. resize back if we padded ---------------------------
#     if (H8, W8) != (H, W):
#         eq = cv2.resize(eq, (W, H), interpolation=cv2.INTER_LANCZOS4)

#     return (eq * 255).astype(np.uint8)

# def cubemap_to_equirect(faces: dict, H: int, W: int):
#     F     = next(iter(faces.values())).shape[0]
#     lon   = (np.arange(W)+0.5)/W * 2*np.pi - np.pi
#     lat   =  np.pi/2 - (np.arange(H)+0.5)/H * np.pi
#     lon, lat = np.meshgrid(lon, lat)

#     x = np.sin(lon) * np.cos(lat)
#     y = np.sin(lat)
#     z = np.cos(lon) * np.cos(lat)

#     absX, absY, absZ = np.abs(x), np.abs(y), np.abs(z)
#     isXPositive = x > 0
#     isYPositive = y > 0
#     isZPositive = z > 0

#     u = np.empty_like(x)
#     v = np.empty_like(x)
#     idx = np.empty_like(x, dtype=np.int8)   # 0‑5

#     # +X (right)
#     m = (absX >= absY) & (absX >= absZ) & isXPositive
#     idx[m] = 1
#     u[m] = -z[m] / absX[m]
#     v[m] = -y[m] / absX[m]

#     # –X (left)
#     m = (absX >= absY) & (absX >= absZ) & ~isXPositive
#     idx[m] = 3
#     u[m] =  z[m] / absX[m]
#     v[m] = -y[m] / absX[m]

#     # +Y (top)
#     m = (absY > absX) & (absY >= absZ) & isYPositive
#     idx[m] = 4
#     u[m] =  x[m] / absY[m]
#     v[m] =  z[m] / absY[m]

#     # –Y (bottom)
#     m = (absY > absX) & (absY >= absZ) & ~isYPositive
#     idx[m] = 5
#     u[m] =  x[m] / absY[m]
#     v[m] = -z[m] / absY[m]

#     # +Z (front)
#     m = (absZ > absX) & (absZ > absY) & isZPositive
#     idx[m] = 0
#     u[m] =  x[m] / absZ[m]
#     v[m] = -y[m] / absZ[m]

#     # –Z (back)
#     m = (absZ > absX) & (absZ > absY) & ~isZPositive
#     idx[m] = 2
#     u[m] = -x[m] / absZ[m]
#     v[m] = -y[m] / absZ[m]

#     u_px = ((u+1)*0.5*(F-1)).astype(np.float32)
#     v_px = ((v+1)*0.5*(F-1)).astype(np.float32)

#     equi = np.empty((H, W, 3), dtype=np.uint8)
#     order = ['front','right','back','left','top','bottom']

#     for fid, name in enumerate(order):
#         m = idx == fid
#         if not m.any(): continue
#         ui0 = np.floor(u_px[m]).astype(np.int32)
#         vi0 = np.floor(v_px[m]).astype(np.int32)
#         ui1 = np.clip(ui0+1, 0, F-1)
#         vi1 = np.clip(vi0+1, 0, F-1)
#         du  = (u_px[m]-ui0)[:,None]
#         dv  = (v_px[m]-vi0)[:,None]
#         f   = faces[name]
#         equi[m] = ((1-du)*(1-dv))*f[vi0,ui0] + \
#                   (   du *(1-dv))*f[vi0,ui1] + \
#                   ((1-du)*   dv)*f[vi1,ui0] + \
#                   (   du *   dv)*f[vi1,ui1]

#     return equi


# def equirect_to_cubemap(equirect_img, face_size):
#     """
#     Convert an equirectangular panorama into 6 cubemap faces,
#     using bicubic sampling + oversampling at the poles to reduce distortion.
#     """
#     H, W = equirect_img.shape[:2]
#     faces = {}
#     face_names = ["front", "right", "back", "left", "top", "bottom"]

#     for face in face_names:
#         # — pole faces: oversample then downsample —
#         if face in ("top", "bottom"):
#             os = 2  # oversampling factor
#             sz = face_size * os
#             # NDC grid in [-1,1]
#             x_ndc, y_ndc = np.meshgrid(
#                 np.linspace(-1, 1, sz),
#                 np.linspace(-1, 1, sz),
#                 indexing="xy"
#             )
#             # compute each pixel’s 3D ray for this face
#             dirs = _compute_face_dirs(face, x_ndc, y_ndc)        # (sz,sz,3)
#             map_u, map_v = _dirs_to_equi_uv(dirs, W, H)         # each (sz,sz)
#             tmp = cv2.remap(
#                 equirect_img, map_u, map_v,
#                 interpolation=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_WRAP
#             )
#             # downsample with area‑averaging
#             face_img = cv2.resize(tmp, (face_size, face_size),
#                                   interpolation=cv2.INTER_AREA)

#         else:
#             # standard face: 1× sampling
#             x_ndc, y_ndc = np.meshgrid(
#                 np.linspace(-1, 1, face_size),
#                 np.linspace(-1, 1, face_size),
#                 indexing="xy"
#             )
#             dirs = _compute_face_dirs(face, x_ndc, y_ndc)       # (face_size,face_size,3)
#             map_u, map_v = _dirs_to_equi_uv(dirs, W, H)
#             face_img = cv2.remap(
#                 equirect_img, map_u, map_v,
#                 interpolation=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_WRAP
#             )

#         faces[face] = face_img

#     return faces


# helper: map normalized face coords → 3D ray depending on face

def _compute_face_dirs(face, x, y):
    """
    x,y are each (H,W) arrays in [-1..1].
    Returns (H,W,3) rays where:

      front  = +Z
      right  = +X
      back   = -Z
      left   = -X
      top    = +Y
      bottom = -Y
    """
    ones  = np.ones_like(x)
    neg_x = -x
    neg_y = -y

    if   face == "front":   # +Z
        dirs = np.stack([ x,  -y,  ones ], axis=-1)
    elif face == "right":   # +X
        dirs = np.stack([ ones, -y, -x   ], axis=-1)
    elif face == "back":    # -Z
        dirs = np.stack([ -x, -y, -ones ], axis=-1)
    elif face == "left":    # -X
        dirs = np.stack([ -ones, -y,  x ], axis=-1)
    elif face == "top":     # +Y
        dirs = np.stack([  x,   ones,  y ], axis=-1)
    elif face == "bottom":  # -Y
        dirs = np.stack([  x,  -ones, -y ], axis=-1)
    else:
        raise ValueError(f"Unknown face: {face}")

    # normalize to unit length
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / (norm + 1e-9)



# helper: project 3D dirs back to equirect UV
def _dirs_to_equi_uv(dirs, W, H):
    X, Y, Z = dirs[...,0], dirs[...,1], dirs[...,2]
    lon = np.arctan2(X, Z)            # [–π,π]
    lat = np.arcsin(Y)                # [–π/2,π/2]
    u = (lon + np.pi) / (2*np.pi) * (W-1)
    v = (np.pi/2 - lat) / np.pi * (H-1)
    return u.astype(np.float32), v.astype(np.float32)


def fix_cubemap_edge_artifacts(cube_faces):
    """
    Apply a ramped, widened blend along every vertical seam
    (front↔right, right↔back, etc.) rather than a tiny fixed blur.
    """
    F = cube_faces.copy()
    face_size = F["front"].shape[0]

    # build a 1D linear ramp mask of width ~1/16 face
    w = max(5, face_size//32)
    ramp1d = np.linspace(0, 1, 2*w)           # from 0→1
    mask = np.zeros((face_size,), np.float32)
    mask[:2*w] = ramp1d                       # only along the seam

    # for each of the 4 side seams, blend face A and B
    seams = [("front","right"), ("right","back"),
             ("back","left"),  ("left","front")]
    for A,B in seams:
        # roll mask into 2D
        seam_mask = np.tile(mask[np.newaxis,:], (face_size,1))
        # shift mask so zeros at far‑edge, ones at the shared seam
        seam_mask = np.roll(seam_mask, -w, axis=1)

        # smooth each side before blending
        blur = lambda im: cv2.GaussianBlur(im, (0,0), sigmaX=3, sigmaY=3)

        A_s = blur(F[A])
        B_s = blur(F[B])

        # blend: mask*B_s + (1-mask)*A_s
        F[A] = ( (1-seam_mask[...,None]) * A_s
                + seam_mask[...,None]   * B_s ).astype(np.uint8)
        F[B] = ( seam_mask[...,None]   * A_s
                + (1-seam_mask[...,None])* B_s ).astype(np.uint8)

    return F


# def cubemap_to_equirect(cube_faces, H, W):
#     """
#     1) For each equirect pixel, pick the single best face by dot‐product.
#     2) Sample that face *with BORDER_WRAP* so no black pixels ever appear.
#     """
#     # 6 faces in fixed order
#     face_list = ["front","right","back","left","top","bottom"]
#     normals = np.array([
#         [ 0,0, 1],[ 1,0,0],[ 0,0,-1],[-1,0,0],[ 0,1,0],[ 0,-1,0]
#     ],dtype=np.float32)

#     face_size = cube_faces["front"].shape[0]
#     F_arr = np.stack([cube_faces[f] for f in face_list], axis=0).astype(np.float32)

#     # build ray directions for every equirect pixel
#     j,i = np.meshgrid(np.arange(W), np.arange(H))
#     lon = (j/(W-1))*2*np.pi - np.pi
#     lat = np.pi/2 - (i/(H-1))*np.pi
#     X = np.sin(lon)*np.cos(lat)
#     Y = np.sin(lat)
#     Z = np.cos(lon)*np.cos(lat)
#     rays = np.stack([X,Y,Z], axis=-1).astype(np.float32)

#     # pick face via max(dot)
#     dots = np.stack([(rays*normals[k]).sum(-1) for k in range(6)],axis=0)
#     dots = np.maximum(dots, 0.0)
#     face_map = np.argmax(dots, axis=0)  # (H,W)

#     # precompute UV maps for all faces
#     u_maps, v_maps = [], []
#     for k in range(6):
#         u,v = _face_dir_to_uv(rays, k, face_size)
#         u_maps.append(u); v_maps.append(v)

#     # sample each face only where it's chosen
#     pano = np.zeros((H,W,3), np.uint8)
#     for k in range(6):
#         mask = (face_map==k)
#         if not mask.any(): 
#             continue
#         samp = cv2.remap(
#             F_arr[k], u_maps[k], v_maps[k],
#             interpolation=cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_WRAP  # <-- wrap, not constant
#         ).astype(np.uint8)
#         pano[mask] = samp[mask]

#     return pano


# def cubemap_to_equirect_from_maps(cube_faces, face_map, u_maps, v_maps):
#     """
#     Reconstruct strictly by sampling each pixel from its chosen face,
#     using BORDER_WRAP so there are no black border pixels.
#     """
#     H,W = face_map.shape
#     pano = np.zeros((H,W,3), np.uint8)
#     order = ["front","right","back","left","top","bottom"]

#     for k, name in enumerate(order):
#         mask = (face_map==k)
#         if not mask.any(): continue
#         samp = cv2.remap(
#             cube_faces[name],
#             u_maps[k], v_maps[k],
#             interpolation=cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_WRAP
#         )
#         pano[mask] = samp[mask]
#     return pano


# def cubemap_to_equirect(cube_faces, H, W):
#     """
#     Reconstruct an (H×W) equirectangular panorama from 6 cubemap faces
#     with no warnings, no black bars, and seamless blending.
    
#     cube_faces: dict with keys
#        'front' (+Z), 'right' (+X), 'back' (-Z),
#        'left' (-X),  'top' (+Y),   'bottom'(-Y)
#     """
#     # 1) build per‐pixel world rays
#     #    i=rows (lat), j=cols (lon)
#     j, i = np.meshgrid(np.arange(W), np.arange(H))
#     lon   = (j/(W-1))*2*np.pi - np.pi
#     lat   =  np.pi/2   - (i/(H-1))*np.pi
#     x = np.sin(lon)*np.cos(lat)
#     y = np.sin(lat)
#     z = np.cos(lon)*np.cos(lat)

#     # 2) decide which axis is “major” at each pixel
#     absx, absy, absz = np.abs(x), np.abs(y), np.abs(z)
#     major = np.argmax(np.stack([absx, absy, absz], axis=-1), axis=-1)
#     # major==0 → X‐face, major==1 → Y‐face, major==2 → Z‐face

#     # 3) allocate output
#     pano = np.zeros((H, W, 3), np.uint8)
#     face_size = cube_faces['front'].shape[0]

#     # 4) helper to compute safe u,v for a given face
#     def compute_uv(nx, ny, nd, face_size):
#         """
#         nx,ny = numerators, nd = denominator (can be zero)
#         returns (u,v) in float32 [0…face_size-1], no warnings.
#         """
#         u = np.empty_like(nx, dtype=np.float32)
#         v = np.empty_like(ny, dtype=np.float32)
#         # safe divide (nd==0 → u,v=0)
#         np.divide(nx, nd, out=u, where=(nd!=0))
#         np.divide(ny, nd, out=v, where=(nd!=0))
#         # clamp to [-1,1]
#         np.clip(u, -1, 1, out=u)
#         np.clip(v, -1, 1, out=v)
#         # to pixel coords
#         u = (u + 1)*0.5*(face_size-1)
#         v = (v + 1)*0.5*(face_size-1)
#         return u, v

#     # 5) for each of the six logical faces, build a mask & remap
#     #    and fill pano[mask] with that face’s pixels.
#     for face_name in ['right','left','top','bottom','front','back']:
#         if face_name == 'right':
#             mask = (major==0) & (x>0)
#             # +X plane: x = +|x|
#             u, v = compute_uv( -z, -y,  x, face_size)
#         elif face_name == 'left':
#             mask = (major==0) & (x<0)
#             # -X plane: x = -|x|
#             u, v = compute_uv(  z, -y, -x, face_size)
#         elif face_name == 'top':
#             mask = (major==1) & (y>0)
#             # +Y plane
#             u, v = compute_uv(  x,  z,  y, face_size)
#         elif face_name == 'bottom':
#             mask = (major==1) & (y<0)
#             # -Y plane
#             u, v = compute_uv(  x, -z, -y, face_size)
#         elif face_name == 'front':
#             mask = (major==2) & (z>0)
#             # +Z plane
#             u, v = compute_uv(  x, -y,  z, face_size)
#         else:  # back
#             mask = (major==2) & (z<0)
#             # -Z plane
#             u, v = compute_uv( -x, -y, -z, face_size)

#         if not mask.any():
#             continue

#         # sample that face everywhere, then mask in only the needed pixels
#         samp = cv2.remap(
#             cube_faces[face_name],
#             u, v,
#             interpolation=cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_WRAP
#         )
#         pano[mask] = samp[mask]

#     # 6) erase any single‐column hard seams by averaging neighbors
#     mid = H//2
#     # find seam columns where face‐assign changes 
#     # (i.e. a boundary between two faces)
#     face_map = (
#         (major==0)&(x>0) * 0 + 
#         (major==0)&(x<0) * 1 +
#         (major==1)&(y>0) * 2 +
#         (major==1)&(y<0) * 3 +
#         (major==2)&(z>0) * 4 +
#         (major==2)&(z<0) * 5
#     ).astype(np.int32)
#     seams = np.where(face_map[mid,1:] != face_map[mid,:-1])[0] + 1
#     for j in seams:
#         if 1 <= j < W-1:
#             # column j = average of j–1 and j+1
#             left  = pano[:, j-1].astype(np.float32)
#             right = pano[:, j+1].astype(np.float32)
#             pano[:,j] = ((left + right)*0.5).astype(np.uint8)

#     return pano


def erase_vertical_seams(pano, face_map):
    """
    Find every column where face_map changes (i.e. seams),
    and replace that single column by averaging its left+right neighbors.
    """
    H,W = face_map.shape
    # detect seam columns
    mid = H//2
    js = np.where(face_map[mid,1:] != face_map[mid,:-1])[0] + 1

    for j in js:
        if 0 < j < W-1:
            # average left & right neighbor columns
            left = pano[:, j-1].astype(np.float32)
            right= pano[:, j+1].astype(np.float32)
            pano[:,j] = ((left+right)*0.5).astype(np.uint8)
    return pano


def blend_vertical_seams(pano, cube_faces, face_map, u_maps, v_maps, band_ratio=0.005):
    """
    pano: current reconstruction (H,W,3)
    cube_faces, face_map, u_maps, v_maps as above
    band_ratio: fraction of width to blend over (~0.5%)
    """
    H,W = face_map.shape
    face_list = ["front","right","back","left"]
    face_size = cube_faces["front"].shape[0]

    # detect seam columns by changes in face_map along center row
    mid = H//2
    changes = np.where(face_map[mid,1:] != face_map[mid,:-1])[0] + 1  # seam js

    band_w = max(1, int(W * band_ratio))

    for j_s in changes:
        fL = face_map[mid, j_s-1]
        fR = face_map[mid, j_s]
        # remap both faces
        sampL = cv2.remap(
            cube_faces[face_list[fL]], u_maps[fL], v_maps[fL],
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        sampR = cv2.remap(
            cube_faces[face_list[fR]], u_maps[fR], v_maps[fR],
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )

        # build band mask and linear alpha
        j_grid = np.broadcast_to(np.arange(W)[None,:], (H,W))
        d = np.clip((j_grid - j_s)/band_w, -1, 1)
        alpha = (d+1)*0.5  # −1→0, +1→1

        # blend only inside |d|<=1
        blend_mask = np.abs(j_grid - j_s) <= band_w
        pano[blend_mask] = (
            alpha[blend_mask,None] * sampL[blend_mask] +
            (1-alpha[blend_mask,None]) * sampR[blend_mask]
        ).astype(np.uint8)

    return pano


def prepare_equirect_lookup(H, W, face_size):
    """
    Precompute everything you need to reconstruct an equirect panorama:
     • rays      : (H,W,3) unit direction vectors
     • face_map  : (H,W) int in [0..5], which face each ray points most toward
     • u_maps    : list of six (H,W) float32 arrays giving u coords for each face
     • v_maps    : same for v coords
    """
    import numpy as np

    # 1) Build rays
    j,i = np.meshgrid(np.arange(W), np.arange(H))
    lon = (j/(W-1))*2*np.pi - np.pi
    lat =  np.pi/2 - (i/(H-1))*np.pi
    X = np.sin(lon)*np.cos(lat)
    Y = np.sin(lat)
    Z = np.cos(lon)*np.cos(lat)
    rays = np.stack([X,Y,Z], axis=-1).astype(np.float32)

    # 2) Face‐selection map
    normals = np.array([
      [ 0,0, 1],[ 1,0, 0],[ 0,0,-1],[-1,0, 0],[ 0,1, 0],[ 0,-1,0]
    ],dtype=np.float32)
    dots = np.stack([ (rays*normals[k]).sum(-1) for k in range(6) ], axis=0)
    dots = np.maximum(dots, 0.0)
    face_map = np.argmax(dots, axis=0)   # shape (H,W), values 0..5

    # 3) UV‐maps for each face
    u_maps = []
    v_maps = []
    for k in range(6):
        u,v = _face_dir_to_uv(rays, k, face_size)
        u_maps.append(u)
        v_maps.append(v)

    return face_map, u_maps, v_maps


def _face_dir_to_uv(rays, face_idx, face_size):
    """
    rays: (H,W,3) unit vectors [X,Y,Z]
    face_idx: 0=front(+Z),1=right(+X),2=back(-Z),
              3=left(-X),4=top(+Y),5=bottom(-Y)
    Returns u,v in [0,face_size-1] with NO warnings.
    """
    X = rays[...,0]; Y = rays[...,1]; Z = rays[...,2]

    # pick numerator & denominator
    if   face_idx==0:  num_x, den =  X, Z;    num_y = -Y
    elif face_idx==1:  num_x, den =  Z, X;    num_y = -Y
    elif face_idx==2:  num_x, den = -X, Z;    num_y = -Y
    elif face_idx==3:  num_x, den = -Z, X;    num_y = -Y
    elif face_idx==4:  num_x, den =  X, Y;    num_y =  Z
    elif face_idx==5:  num_x, den =  X, Y;    num_y = -Z
    else: raise ValueError("face_idx must be 0..5")

    # safe divide (no warnings)
    x_ndc = np.empty_like(num_x, dtype=np.float32)
    y_ndc = np.empty_like(num_y, dtype=np.float32)
    # where den!=0, divide; else leave zero
    np.divide(num_x, den, out=x_ndc, where=(den!=0))
    np.divide(num_y, den, out=y_ndc, where=(den!=0))

    # clamp to [-1,1] just in case
    np.clip(x_ndc, -1, 1, out=x_ndc)
    np.clip(y_ndc, -1, 1, out=y_ndc)

    # NDC → pixel coords
    u = (x_ndc + 1)*0.5*(face_size-1)
    v = (y_ndc + 1)*0.5*(face_size-1)

    return u.astype(np.float32), v.astype(np.float32)



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


def diagnose_conversion_issues(original, reconstructed):
    """
    Diagnose issues in the equirectangular to cubemap to equirectangular conversion.
    
    Args:
        original (numpy.ndarray or Tensor): Original equirectangular image
        reconstructed (numpy.ndarray or Tensor): Reconstructed equirectangular image
        
    Returns:
        dict: Diagnostic information about conversion issues
    """
    import math
    import torch
    import numpy as np
    
    # Handle numpy arrays
    is_numpy = isinstance(original, np.ndarray)
    
    if is_numpy:
        # Convert to tensor for processing if numpy
        if len(original.shape) == 3 and original.shape[2] <= 4:  # HWC format
            height, width, channels = original.shape
            original_tensor = torch.from_numpy(original.transpose(2, 0, 1)).float()
            reconstructed_tensor = torch.from_numpy(reconstructed.transpose(2, 0, 1)).float()
        else:
            if len(original.shape) == 3:
                channels, height, width = original.shape
            else:
                # Handle other formats
                height, width = original.shape[:2]
                channels = 3  # Default
            original_tensor = torch.from_numpy(original).float()
            reconstructed_tensor = torch.from_numpy(reconstructed).float()
    else:
        original_tensor = original
        reconstructed_tensor = reconstructed
        if original.dim() == 3:
            channels, height, width = original.shape
        else:
            # Handle batched tensors
            channels, height, width = original.shape[1:4]
    
    # Compute absolute difference between original and reconstructed
    diff = torch.abs(original_tensor - reconstructed_tensor)
    
    # Calculate overall statistics
    mse = torch.mean(diff ** 2).item()
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-10))
    
    # Calculate SSIM
    # Implementation of SSIM using PyTorch
    def calculate_ssim(img1, img2, window_size=11, size_average=True):
        # Constants for stabilization
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Generate a 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/(2*sigma**2) for x in range(window_size)]))
        gauss = gauss / gauss.sum()
        
        # Create 2D kernel by outer product
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        # Expand kernel for multi-channel images
        window = _2D_window.expand(channels, 1, window_size, window_size).to(img1.device)
        
        # Pad images for convolution
        padding = window_size // 2
        
        # Function to calculate mean using convolution
        def conv_mean(x, window):
            return torch.nn.functional.conv2d(x, window, padding=padding, groups=channels)
            
        # Convert to float and calculate means and variances
        mu1 = conv_mean(img1, window)
        mu2 = conv_mean(img2, window)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = conv_mean(img1 * img1, window) - mu1_sq
        sigma2_sq = conv_mean(img2 * img2, window) - mu2_sq
        sigma12 = conv_mean(img1 * img2, window) - mu1_mu2
        
        # SSIM calculation
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    # Ensure images are in the correct range [0, 255]
    # This assumes original range is [0, 255] or [0, 1]
    if original_tensor.max() <= 1.0 and reconstructed_tensor.max() <= 1.0:
        original_tensor_255 = original_tensor * 255
        reconstructed_tensor_255 = reconstructed_tensor * 255
    else:
        original_tensor_255 = original_tensor
        reconstructed_tensor_255 = reconstructed_tensor
    
    # Calculate global SSIM
    ssim_value = calculate_ssim(original_tensor_255, reconstructed_tensor_255).item()
    
    # Calculate regional SSIM values
    # 1. Seam at longitude ±π (left and right edges)
    seam_width = width // 50
    left_seam_ssim = calculate_ssim(
        original_tensor_255[:, :, :seam_width], 
        reconstructed_tensor_255[:, :, :seam_width]
    ).item()
    right_seam_ssim = calculate_ssim(
        original_tensor_255[:, :, -seam_width:], 
        reconstructed_tensor_255[:, :, -seam_width:]
    ).item()
    seam_ssim = (left_seam_ssim + right_seam_ssim) / 2
    
    # 2. North pole region (top 10%)
    north_pole_height = height // 10
    north_pole_ssim = calculate_ssim(
        original_tensor_255[:, :north_pole_height, :], 
        reconstructed_tensor_255[:, :north_pole_height, :]
    ).item()
    
    # 3. South pole region (bottom 10%)
    south_pole_height = height // 10
    south_pole_ssim = calculate_ssim(
        original_tensor_255[:, -south_pole_height:, :], 
        reconstructed_tensor_255[:, -south_pole_height:, :]
    ).item()
    
    # 4. Equator region (middle 20%)
    equator_start = int(height * 0.4)
    equator_end = int(height * 0.6)
    equator_ssim = calculate_ssim(
        original_tensor_255[:, equator_start:equator_end, :], 
        reconstructed_tensor_255[:, equator_start:equator_end, :]
    ).item()
    
    # Analyze specific regions
    # 1. Seam at longitude ±π (left and right edges)
    seam_diff = torch.mean(diff[:, :, :seam_width]).item() + torch.mean(diff[:, :, -seam_width:]).item()
    seam_diff /= 2  # Average of left and right edges
    
    # 2. North pole region (top 10%)
    north_pole_diff = torch.mean(diff[:, :north_pole_height, :]).item()
    
    # 3. South pole region (bottom 10%)
    south_pole_diff = torch.mean(diff[:, -south_pole_height:, :]).item()
    
    # 4. Equator region (middle 20%)
    equator_diff = torch.mean(diff[:, equator_start:equator_end, :]).item()
    
    # 5. Longitude bands analysis (front, right, back, left faces)
    longitude_diffs = {}
    longitude_ssims = {}
    longitude_bands = {
        'Front': (width // 4, width // 2),         # 90° to 180°
        'Right': (0, width // 4),                  # 0° to 90°
        'Back': (width // 2, 3 * width // 4),      # 180° to 270°
        'Left': (3 * width // 4, width)            # 270° to 360°
    }
    
    for name, (start, end) in longitude_bands.items():
        band_diff = torch.mean(diff[:, :, start:end]).item()
        longitude_diffs[name] = band_diff
        
        band_ssim = calculate_ssim(
            original_tensor_255[:, :, start:end], 
            reconstructed_tensor_255[:, :, start:end]
        ).item()
        longitude_ssims[name] = band_ssim
    
    # Find problem areas (regions with highest difference)
    problem_areas = {k: v for k, v in longitude_diffs.items()}
    
    # Find best/worst areas by SSIM
    best_ssim_region = max(longitude_ssims.items(), key=lambda x: x[1])[0]
    worst_ssim_region = min(longitude_ssims.items(), key=lambda x: x[1])[0]
    
    # Diagnose specific issues
    diagnoses = []
    
    if seam_diff > 6.0 or seam_ssim < 0.85:
        diagnoses.append("Severe seam issue at longitude ±π (back face center). Apply specialized seam handling.")
    elif seam_diff > 3.0 or seam_ssim < 0.92:
        diagnoses.append("Moderate seam issue at longitude ±π. Consider improving seam handling.")
    
    if north_pole_ssim < 0.85 or south_pole_ssim < 0.85:
        diagnoses.append("Pole distortion detected. Improve sampling near poles (top/bottom faces).")
    elif north_pole_diff > 5.0 or south_pole_diff > 5.0:
        diagnoses.append("Moderate pole distortion detected. Consider refining pole sampling.")
    
    if min(longitude_ssims.values()) < 0.90:
        diagnoses.append(f"Poor quality in {worst_ssim_region} face (SSIM: {longitude_ssims[worst_ssim_region]:.3f}). Review cubemap face transformation.")
    
    if max(longitude_ssims.values()) - min(longitude_ssims.values()) > 0.1:
        diagnoses.append(f"Uneven quality across faces. {worst_ssim_region} face (SSIM: {longitude_ssims[worst_ssim_region]:.3f}) needs improvement compared to {best_ssim_region} face (SSIM: {longitude_ssims[best_ssim_region]:.3f}).")
    
    # Assemble the diagnostic results
    diagnostics = {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Seam_Diff': seam_diff,
        'Seam_SSIM': seam_ssim,
        'North_Pole_Diff': north_pole_diff,
        'North_Pole_SSIM': north_pole_ssim,
        'South_Pole_Diff': south_pole_diff,
        'South_Pole_SSIM': south_pole_ssim,
        'Equator_Diff': equator_diff,
        'Equator_SSIM': equator_ssim,
        'Longitude_Diffs': longitude_diffs,
        'Longitude_SSIMs': longitude_ssims,
        'Best_SSIM_Region': best_ssim_region,
        'Worst_SSIM_Region': worst_ssim_region,
        'Problem_Areas': problem_areas,
        'Diagnoses': diagnoses
    }
    
    return diagnostics

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

    # --------  EXACTLY the same two fixes the notebook had  -------------------
    cube_faces = fix_vertical_seams(cube_faces)        # blends ±π columns
    cube_faces = antialias_poles(cube_faces, factor=2) # 2× supersample / down‑box
    # --------------------------------------------------------------------------

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

# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive image‑quality metrics
# ─────────────────────────────────────────────────────────────────────────────


# def calculate_metrics(img_a, img_b):
#     """
#     img_a, img_b : H×W×3   NumPy uint8   or   PyTorch CHW uint8 / float

#     Returns dict with keys:
#         mse, psnr, ssim,
#         low_diff_percentage, medium_diff_percentage, high_diff_percentage
#     """
#     # ------------- normalise both inputs to NumPy float32 [0,255] -----------
#     if 'torch' in str(type(img_a)):
#         import torch
#         if img_a.dtype == torch.uint8:
#             arr_a = img_a.permute(1,2,0).cpu().numpy().astype(np.float32)
#         else:                               # assume float 0‑1 or 0‑255
#             arr_a = (img_a.permute(1,2,0)*255.).cpu().numpy().astype(np.float32)
#     else:
#         arr_a = img_a.astype(np.float32)

#     if 'torch' in str(type(img_b)):
#         import torch
#         if img_b.dtype == torch.uint8:
#             arr_b = img_b.permute(1,2,0).cpu().numpy().astype(np.float32)
#         else:
#             arr_b = (img_b.permute(1,2,0)*255.).cpu().numpy().astype(np.float32)
#     else:
#         arr_b = img_b.astype(np.float32)

#     # ------------- basic errors --------------------------------------------
#     diff   = arr_a - arr_b
#     mse    = np.mean(diff**2)
#     psnr   = 10 * np.log10((255.0**2) / mse)

#     # SSIM expects 0‑255 float, channel‑last
#     ssim_val = ssim(arr_a, arr_b, channel_axis=2, data_range=255)

#     # ------------- histogram buckets ---------------------------------------
#     abs_diff = np.mean(np.abs(diff), axis=2)    # mean over RGB
#     total    = abs_diff.size
#     low_pct     = 100 * np.sum(abs_diff < 5)          / total
#     medium_pct  = 100 * np.sum((abs_diff >= 5) &
#                                 (abs_diff < 10))      / total
#     high_pct    = 100 * np.sum(abs_diff >= 10)        / total

#     return {
#         "mse": mse,
#         "psnr": psnr,
#         "ssim": ssim_val,
#         "low_diff_percentage":    low_pct,
#         "medium_diff_percentage": medium_pct,
#         "high_diff_percentage":   high_pct,
#     }

# ---------------------------------------------------------------------
# Quality metrics helper (NumPy / Torch agnostic)
# ---------------------------------------------------------------------
from skimage.metrics import structural_similarity as ssim
def calculate_metrics(a, b):
    import torch
    def to_np(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.uint8:
                return x.permute(1,2,0).cpu().numpy().astype(np.float32)
            return (x.permute(1,2,0)*255.).cpu().numpy().astype(np.float32)
        raise TypeError("unknown type")
    a = to_np(a); b = to_np(b)
    diff = a - b
    mse  = np.mean(diff**2)
    psnr = 10*np.log10((255.0**2)/mse)
    ssim_val = ssim(a, b, channel_axis=2, data_range=255)
    ad = np.mean(np.abs(diff), 2)
    total = ad.size
    return dict(
        mse=mse, psnr=psnr, ssim=ssim_val,
        low_diff_percentage    =100*np.sum(ad<5)/total,
        medium_diff_percentage =100*np.sum((ad>=5)&(ad<10))/total,
        high_diff_percentage   =100*np.sum(ad>=10)/total
    )


# ────────────────────────────────────────────────────────────────
# 1.  Fix the ±π longitude seam       (vertical seam in equirect)
# ────────────────────────────────────────────────────────────────
def fix_vertical_seams(cube):
    """
    Blend the first two and last two columns of *each lateral* face
    (front/right/back/left) so that the equirectangular wrap seam is C0‑continuous.

    cube : {'front','right','back','left','top','bottom'}  ->  H×H×3  uint8
    """
    out = {}
    for k, img in cube.items():
        if k in ('top', 'bottom'):
            out[k] = img.copy()
            continue
        blended = img.copy().astype(np.float32)
        # blend first 2 and last 2 pixel columns
        blended[:, 0:2]   = 0.5*(img[:, 0:2].astype(np.float32) +
                                 img[:, -2:].astype(np.float32))
        blended[:, -2:]   = blended[:, 0:2]
        out[k] = blended.clip(0,255).astype(np.uint8)
    return out

# ────────────────────────────────────────────────────────────────
# 2.  Anti‑alias the pole faces
# ────────────────────────────────────────────────────────────────
def antialias_poles(cube, factor=2):
    """
    Super‑sample the 'top' and 'bottom' faces by <factor>, then box‑filter
    back down (cv2.INTER_AREA).   Default factor=2 is enough for 512‑px faces.
    """
    out = cube.copy()
    for pole in ('top', 'bottom'):
        face = cube[pole]
        big = cv2.resize(face, (0,0), fx=factor, fy=factor,
                         interpolation=cv2.INTER_LANCZOS4)
        out[pole] = cv2.resize(big, (face.shape[1], face.shape[0]),
                               interpolation=cv2.INTER_AREA)
    return out

