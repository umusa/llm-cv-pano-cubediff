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
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

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
    Convert an equirectangular panorama into 6 cubemap faces,
    using bicubic sampling + oversampling at the poles to reduce distortion.
    """
    H, W = equirect_img.shape[:2]
    faces = {}
    face_names = ["front", "right", "back", "left", "top", "bottom"]

    for face in face_names:
        # — pole faces: oversample then downsample —
        if face in ("top", "bottom"):
            os = 2  # oversampling factor
            sz = face_size * os
            # NDC grid in [-1,1]
            x_ndc, y_ndc = np.meshgrid(
                np.linspace(-1, 1, sz),
                np.linspace(-1, 1, sz),
                indexing="xy"
            )
            # compute each pixel’s 3D ray for this face
            dirs = _compute_face_dirs(face, x_ndc, y_ndc)        # (sz,sz,3)
            map_u, map_v = _dirs_to_equi_uv(dirs, W, H)         # each (sz,sz)
            tmp = cv2.remap(
                equirect_img, map_u, map_v,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP
            )
            # downsample with area‑averaging
            face_img = cv2.resize(tmp, (face_size, face_size),
                                  interpolation=cv2.INTER_AREA)

        else:
            # standard face: 1× sampling
            x_ndc, y_ndc = np.meshgrid(
                np.linspace(-1, 1, face_size),
                np.linspace(-1, 1, face_size),
                indexing="xy"
            )
            dirs = _compute_face_dirs(face, x_ndc, y_ndc)       # (face_size,face_size,3)
            map_u, map_v = _dirs_to_equi_uv(dirs, W, H)
            face_img = cv2.remap(
                equirect_img, map_u, map_v,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP
            )

        faces[face] = face_img

    return faces


# helper: map normalized face coords → 3D ray depending on face

import numpy as np

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

def _prepare_lookup(H, W, face_size):
    """
    Precompute face_map, u_maps, v_maps for an (H,W) pano and
    cubemap faces of size face_size.
    """
    import numpy as np

    j,i = np.meshgrid(np.arange(W), np.arange(H))
    lon = (j/(W-1))*2*np.pi - np.pi
    lat = np.pi/2 - (i/(H-1))*np.pi

    X = np.sin(lon)*np.cos(lat)
    Y = np.sin(lat)
    Z = np.cos(lon)*np.cos(lat)
    rays = np.stack([X,Y,Z], axis=-1).astype(np.float32)

    # face selection by max‐axis rule
    absv = np.abs(rays)
    major = np.argmax(absv, axis=-1)

    # build face_map (0..5) same order as normals in your code
    # 0=right,1=left,2=top,3=bottom,4=front,5=back  (match your pipeline)
    face_map = np.zeros((H,W), np.int8)
    face_map[(major==0)&(X>0)] = 0  # right
    face_map[(major==0)&(X<0)] = 1  # left
    face_map[(major==1)&(Y>0)] = 2  # top
    face_map[(major==1)&(Y<0)] = 3  # bottom
    face_map[(major==2)&(Z>0)] = 4  # front
    face_map[(major==2)&(Z<0)] = 5  # back

    # precompute u/v for each of the 6 faces
    u_maps, v_maps = [], []
    for idx in range(6):
        u,v = _face_dir_to_uv(rays, idx, face_size)
        u_maps.append(u); v_maps.append(v)

    return face_map, u_maps, v_maps


def cubemap_to_equirect(cube_faces, H, W, seam_fix=True):
    """
    Drop‑in replacement for your old cubemap_to_equirect.
    Internally calls the precompute, does a single pass remap with wrap,
    then optionally smooths seams.
    """
    # 1) precompute maps
    face_size = cube_faces['front'].shape[0]
    face_map, u_maps, v_maps = _prepare_lookup(H, W, face_size)

    # 2) single‐face remap with wrap
    import numpy as np, cv2
    pano = np.zeros((H, W, 3), np.uint8)
    order = ['right','left','top','bottom','front','back']
    for k,name in enumerate(order):
        m = (face_map==k)
        if not m.any(): continue
        samp = cv2.remap(
            cube_faces[name],
            u_maps[k], v_maps[k],
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        pano[m] = samp[m]

    # 3) optional 1‑px seam fix
    if seam_fix:
        mid = H//2
        seams = np.where(face_map[mid,1:] != face_map[mid,:-1])[0] + 1
        for j in seams:
            if 1 <= j < W-1:
                left  = pano[:, j-1].astype(np.float32)
                right = pano[:, j+1].astype(np.float32)
                pano[:,j] = ((left+right)*0.5).astype(np.uint8)

    return pano

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
    
    # Analyze specific regions
    # 1. Seam at longitude ±π (left and right edges)
    seam_width = width // 50
    seam_diff = torch.mean(diff[:, :, :seam_width]).item() + torch.mean(diff[:, :, -seam_width:]).item()
    seam_diff /= 2  # Average of left and right edges
    
    # 2. North pole region (top 10%)
    north_pole_height = height // 10
    north_pole_diff = torch.mean(diff[:, :north_pole_height, :]).item()
    
    # 3. South pole region (bottom 10%)
    south_pole_height = height // 10
    south_pole_diff = torch.mean(diff[:, -south_pole_height:, :]).item()
    
    # 4. Equator region (middle 20%)
    equator_start = int(height * 0.4)
    equator_end = int(height * 0.6)
    equator_diff = torch.mean(diff[:, equator_start:equator_end, :]).item()
    
    # 5. Longitude bands analysis (front, right, back, left faces)
    longitude_diffs = {}
    longitude_bands = {
        'Front': (width // 4, width // 2),         # 90° to 180°
        'Right': (0, width // 4),                  # 0° to 90°
        'Back': (width // 2, 3 * width // 4),      # 180° to 270°
        'Left': (3 * width // 4, width)            # 270° to 360°
    }
    
    for name, (start, end) in longitude_bands.items():
        band_diff = torch.mean(diff[:, :, start:end]).item()
        longitude_diffs[name] = band_diff
    
    # Find problem areas (regions with highest difference)
    problem_areas = {k: v for k, v in longitude_diffs.items()}
    
    # Diagnose specific issues
    diagnoses = []
    
    if seam_diff > 6.0:
        diagnoses.append("Severe seam issue at longitude ±π (back face center). Apply specialized seam handling.")
    elif seam_diff > 3.0:
        diagnoses.append("Moderate seam issue at longitude ±π. Consider improving seam handling.")
    
    if north_pole_diff > 5.0 or south_pole_diff > 5.0:
        diagnoses.append("Pole distortion detected. Improve sampling near poles (top/bottom faces).")
    
    if max(longitude_diffs.values()) > 1.5 * min(longitude_diffs.values()):
        worst_band = max(longitude_diffs.items(), key=lambda x: x[1])[0]
        diagnoses.append(f"Uneven quality across longitude bands. {worst_band} face needs improvement.")
    
    # Assemble the diagnostic results
    diagnostics = {
        'MSE': mse,
        'PSNR': psnr,
        'Seam_Diff': seam_diff,
        'North_Pole_Diff': north_pole_diff,
        'South_Pole_Diff': south_pole_diff,
        'Equator_Diff': equator_diff,
        'Longitude_Diffs': longitude_diffs,
        'Problem_Areas': problem_areas,
        'Diagnoses': diagnoses
    }
    
    return diagnostics



def compute_metrics(orig, recon):
    """
    Compute a suite of quality metrics between two equirectangular panoramas:
      • MSE              – plain Mean Squared Error
      • PSNR             – Peak Signal‐to‐Noise Ratio
      • SSIM             – Structural Similarity Index (windowed)
      • MSE_spherical    – latitude‐weighted MSE
      • PSNR_spherical   – PSNR derived from MSE_spherical

    Args:
      orig:   (H,W,3) uint8 original panorama
      recon:  (H,W,3) uint8 reconstructed panorama

    Returns:
      dict with keys ['MSE','PSNR','SSIM','MSE_spherical','PSNR_spherical']
    """
    # ensure float32 for calculations
    orig_f = orig.astype(np.float32)
    recon_f = recon.astype(np.float32)

    # 1) Classic MSE & PSNR
    mse = mean_squared_error(orig, recon)
    psnr = peak_signal_noise_ratio(orig, recon, data_range=255)

    # 2) SSIM (windowed)
    ssim = structural_similarity(
        orig, recon,
        data_range=255,
        multichannel=True,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    # 3) Spherical (latitude‐weighted) MSE & PSNR
    H, W = orig.shape[:2]
    # latitude angles from +π/2 (top) to -π/2 (bottom)
    lats = np.linspace(np.pi/2, -np.pi/2, H)[:,None]
    weights = np.cos(lats)
    weights /= weights.sum()

    # per‐pixel squared error (averaged over channels)
    se = ((orig_f - recon_f)**2).mean(axis=2)  # (H,W)
    mse_sph = float((weights * se).sum())      # scalar

    psnr_sph = 10 * np.log10((255.0**2) / mse_sph)

    return {
        'MSE': float(mse),
        'PSNR': float(psnr),
        'SSIM': float(ssim),
        'MSE_spherical': mse_sph,
        'PSNR_spherical': float(psnr_sph),
    }


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