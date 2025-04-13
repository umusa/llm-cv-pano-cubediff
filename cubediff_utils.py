import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

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

def equirect_to_cubemap(equirect_img, face_size=None):
    """
    Convert an equirectangular image to 6 cubemap faces.
    
    Args:
        equirect_img: Equirectangular image in numpy array format (H, W, C)
        face_size: Size of each cubemap face (if None, calculated from equirect size)
        
    Returns:
        List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
    """
    # Get dimensions of the input equirectangular image
    h, w = equirect_img.shape[:2]
    
    # Calculate face size if not provided
    if face_size is None:
        face_size = w // 4  # A common rule of thumb
    
    # Create arrays to store the 6 faces
    faces = []
    
    # Relative positions of the 6 faces in 3D space (normalized vectors)
    # Consistent with the order: Front, Right, Back, Left, Top, Bottom
    face_coordinates = [
        [0, 0, -1],  # Front (negative Z)
        [1, 0, 0],   # Right (positive X)
        [0, 0, 1],   # Back (positive Z)
        [-1, 0, 0],  # Left (negative X)
        [0, -1, 0],  # Top (negative Y)
        [0, 1, 0]    # Bottom (positive Y)
    ]
    
    # For each face
    for face_idx, coords in enumerate(face_coordinates):
        # Create output face
        if len(equirect_img.shape) > 2:
            face = np.zeros((face_size, face_size, equirect_img.shape[2]), dtype=equirect_img.dtype)
        else:
            face = np.zeros((face_size, face_size), dtype=equirect_img.dtype)
        
        # Calculate mapping from face coordinates to equirectangular coordinates
        for y in range(face_size):
            for x in range(face_size):
                # Convert face pixel to 3D direction (normalize [0,size-1] to [-1,1])
                u = 2.0 * x / (face_size - 1) - 1.0
                v = 2.0 * y / (face_size - 1) - 1.0
                
                # Calculate 3D vector based on which face we're mapping
                if face_idx == 0:  # Front (negative Z)
                    vec = [u, v, -1.0]
                elif face_idx == 1:  # Right (positive X)
                    vec = [1.0, v, -u]
                elif face_idx == 2:  # Back (positive Z)
                    vec = [-u, v, 1.0]
                elif face_idx == 3:  # Left (negative X)
                    vec = [-1.0, v, u]
                elif face_idx == 4:  # Top (negative Y)
                    vec = [u, -1.0, -v]
                elif face_idx == 5:  # Bottom (positive Y)
                    vec = [u, 1.0, v]
                
                # Normalize vector to unit length
                norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                vec = [v / norm for v in vec]
                
                # Convert 3D direction to equirectangular coordinates
                # Compute spherical coordinates (phi, theta)
                phi = np.arcsin(vec[1])  # Latitude (-pi/2 to pi/2)
                theta = np.arctan2(vec[2], vec[0])  # Longitude (-pi to pi)
                
                # Map to equirectangular coordinates [0, w-1] x [0, h-1]
                # Map theta from [-pi, pi] to [0, w-1]
                equi_x = (theta + np.pi) / (2.0 * np.pi) * (w - 1)
                # Map phi from [-pi/2, pi/2] to [0, h-1]
                equi_y = (phi + np.pi/2) / np.pi * (h - 1)
                
                # Ensure coordinates are within bounds
                equi_x = max(0, min(w - 1, equi_x))
                equi_y = max(0, min(h - 1, equi_y))
                
                # Sample from equirectangular image with bilinear interpolation
                x0, y0 = int(equi_x), int(equi_y)
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                
                # Calculate interpolation weights
                dx, dy = equi_x - x0, equi_y - y0
                
                # Perform bilinear interpolation
                if len(equirect_img.shape) > 2:  # Color image
                    for c in range(equirect_img.shape[2]):
                        face[y, x, c] = (1 - dx) * (1 - dy) * equirect_img[y0, x0, c] + \
                                       dx * (1 - dy) * equirect_img[y0, x1, c] + \
                                       (1 - dx) * dy * equirect_img[y1, x0, c] + \
                                       dx * dy * equirect_img[y1, x1, c]
                else:  # Grayscale image
                    face[y, x] = (1 - dx) * (1 - dy) * equirect_img[y0, x0] + \
                                dx * (1 - dy) * equirect_img[y0, x1] + \
                                (1 - dx) * dy * equirect_img[y1, x0] + \
                                dx * dy * equirect_img[y1, x1]
        
        faces.append(face)
    
    return faces

def cubemap_to_equirect(cube_faces, equirect_h, equirect_w=None):
    """
    Convert 6 cubemap faces to an equirectangular image.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        equirect_h: Height of output equirectangular image
        equirect_w: Width of output equirectangular image (defaults to 2*height for 2:1 aspect ratio)
        
    Returns:
        Equirectangular image in numpy array format (H, W, C)
    """
    return cubemap_to_equirect_fixed(cube_faces, equirect_h, equirect_w)

def cubemap_to_equirect_fixed(cube_faces, equirect_h, equirect_w=None):
    """
    Convert 6 cubemap faces to an equirectangular image with correct orientation.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        equirect_h: Height of output equirectangular image
        equirect_w: Width of output equirectangular image (defaults to 2*height for 2:1 aspect ratio)
        
    Returns:
        Equirectangular image in numpy array format (H, W, C)
    """
    # Ensure we have exactly 6 faces
    if len(cube_faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(cube_faces)}")
    
    # Default width is twice the height (2:1 aspect ratio for equirectangular)
    if equirect_w is None:
        equirect_w = 2 * equirect_h
    
    # Verify all faces have the same dimensions
    face_shape = cube_faces[0].shape
    for i, face in enumerate(cube_faces):
        if face.shape != face_shape:
            raise ValueError(f"All faces must have the same shape. Face {i} has shape {face.shape}, expected {face_shape}")
    
    # Determine if image is grayscale or color
    if len(face_shape) > 2:
        # Color image
        channels = face_shape[2]
        equirect = np.zeros((equirect_h, equirect_w, channels), dtype=cube_faces[0].dtype)
    else:
        # Grayscale image
        channels = 1
        equirect = np.zeros((equirect_h, equirect_w), dtype=cube_faces[0].dtype)
    
    # Face direction vectors (normalized)
    # Order: Front, Right, Back, Left, Top, Bottom
    face_coords = [
        [0, 0, -1],  # Front (negative Z)
        [1, 0, 0],   # Right (positive X)
        [0, 0, 1],   # Back (positive Z)
        [-1, 0, 0],  # Left (negative X)
        [0, -1, 0],  # Top (negative Y)
        [0, 1, 0]    # Bottom (positive Y)
    ]
    
    # Updated up vectors for correct orientation
    up_vectors = [
        [0, -1, 0],  # Front - Up is negative Y
        [0, -1, 0],  # Right - Up is negative Y
        [0, -1, 0],  # Back - Up is negative Y
        [0, -1, 0],  # Left - Up is negative Y
        [0, 0, 1],   # Top - Up is positive Z (when looking down from top)
        [0, 0, -1]   # Bottom - Up is negative Z (when looking up from bottom)
    ]
    
    # Updated right vectors for correct orientation
    right_vectors = [
        [1, 0, 0],    # Front - Right is positive X
        [0, 0, 1],    # Right - Right is positive Z
        [-1, 0, 0],   # Back - Right is negative X
        [0, 0, -1],   # Left - Right is negative Z
        [1, 0, 0],    # Top - Right is positive X
        [1, 0, 0]     # Bottom - Right is positive X
    ]
    
    # For each pixel in the equirectangular image
    for y in range(equirect_h):
        # Latitude angle (phi) ranges from +π/2 (top) to -π/2 (bottom)
        # Note: We're flipping the y-coordinate calculation to fix the upside-down issue
        phi = np.pi * (0.5 - y / equirect_h)  # Changed from (y / equirect_h - 0.5)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        for x in range(equirect_w):
            # Longitude angle (theta) ranges from -π to π
            theta = 2.0 * np.pi * (x / equirect_w - 0.5)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Convert spherical to 3D Cartesian coordinates (unit vector)
            vec = [
                cos_phi * cos_theta,  # X
                sin_phi,              # Y
                cos_phi * sin_theta   # Z
            ]
            
            # Determine which face this ray hits by finding the largest component
            max_val = -1.0
            selected_face = -1
            
            for i, coords in enumerate(face_coords):
                # Dot product gives projection strength on the face direction
                val = vec[0] * coords[0] + vec[1] * coords[1] + vec[2] * coords[2]
                if val > max_val:
                    max_val = val
                    selected_face = i
            
            # Get the appropriate face, up vector, and right vector
            face_dir = face_coords[selected_face]
            up = up_vectors[selected_face]
            right = right_vectors[selected_face]
            
            # Scale to face coordinate system
            # The max_val is the cosine of the angle between the vector and face direction
            # Dividing by this scales the projection to the face plane
            scale = 1.0 / max_val
            
            # Project 3D position onto the face's 2D coordinate system
            dx = scale * (vec[0] * right[0] + vec[1] * right[1] + vec[2] * right[2])
            dy = scale * (vec[0] * up[0] + vec[1] * up[1] + vec[2] * up[2])
            
            # Map from [-1,1] to [0, face_size-1]
            face_h, face_w = face_shape[:2]
            face_x = (dx + 1.0) * 0.5 * (face_w - 1)
            face_y = (dy + 1.0) * 0.5 * (face_h - 1)
            
            # Ensure coordinates are within bounds
            face_x = max(0, min(face_w - 1, face_x))
            face_y = max(0, min(face_h - 1, face_y))
            
            # Sample from the appropriate cubemap face with bilinear interpolation
            face = cube_faces[selected_face]
            
            # Bilinear interpolation
            x0, y0 = int(face_x), int(face_y)
            x1, y1 = min(x0 + 1, face_w - 1), min(y0 + 1, face_h - 1)
            wx, wy = face_x - x0, face_y - y0
            
            # Apply interpolation
            if channels == 1:  # Grayscale
                equirect[y, x] = (1-wx)*(1-wy)*face[y0, x0] + \
                                 wx*(1-wy)*face[y0, x1] + \
                                 (1-wx)*wy*face[y1, x0] + \
                                 wx*wy*face[y1, x1]
            else:  # Color image
                for c in range(channels):
                    equirect[y, x, c] = (1-wx)*(1-wy)*face[y0, x0, c] + \
                                        wx*(1-wy)*face[y0, x1, c] + \
                                        (1-wx)*wy*face[y1, x0, c] + \
                                        wx*wy*face[y1, x1, c]
    
    return equirect

def check_face_order(cube_faces):
    """
    Utility function to validate and display the order of cubemap faces.
    
    Args:
        cube_faces: List of 6 cubemap faces
        
    Returns:
        True if the correct number of faces is provided, False otherwise
    """
    face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    
    if len(cube_faces) != 6:
        print(f"ERROR: Expected 6 cubemap faces, got {len(cube_faces)}")
        return False
    
    print("Cubemap face order:")
    for i, name in enumerate(face_names):
        shape_str = str(cube_faces[i].shape)
        print(f"  {i}: {name} - Shape: {shape_str}")
    
    return True

def visualize_cubemap_faces(cube_faces, titles=None):
    """
    Visualize all 6 cubemap faces in a grid.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        titles: Optional list of titles for each face
        
    Returns:
        fig: Matplotlib figure
    """
    if titles is None:
        titles = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (face, title) in enumerate(zip(cube_faces, titles)):
        if len(face.shape) > 2:
            axes[i].imshow(face)
        else:
            axes[i].imshow(face, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_cubemap_layout(cube_faces, with_labels=True):
    """
    Create a visual layout of the cubemap faces.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        with_labels: Whether to add face labels
        
    Returns:
        layout: Combined image showing the cubemap layout
    """
    if len(cube_faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(cube_faces)}")
    
    # Get face dimensions
    face_h, face_w = cube_faces[0].shape[:2]
    
    # Determine number of channels
    if len(cube_faces[0].shape) > 2:
        channels = cube_faces[0].shape[2]
    else:
        channels = 1
    
    # Create the layout
    # +---+---+---+
    # |   | T |   |
    # +---+---+---+
    # | L | F | R |
    # +---+---+---+
    # |   | Bo| Ba|
    # +---+---+---+
    
    # Create a 3x3 grid
    if channels == 1:
        layout = np.zeros((face_h * 3, face_w * 3), dtype=cube_faces[0].dtype)
    else:
        layout = np.zeros((face_h * 3, face_w * 3, channels), dtype=cube_faces[0].dtype)
    
    # Position each face in the grid
    # Front (center)
    layout[face_h:face_h*2, face_w:face_w*2] = cube_faces[0]
    
    # Right
    layout[face_h:face_h*2, face_w*2:face_w*3] = cube_faces[1]
    
    # Back
    layout[face_h*2:face_h*3, face_w*2:face_w*3] = cube_faces[2]
    
    # Left
    layout[face_h:face_h*2, 0:face_w] = cube_faces[3]
    
    # Top
    layout[0:face_h, face_w:face_w*2] = cube_faces[4]
    
    # Bottom
    layout[face_h*2:face_h*3, face_w:face_w*2] = cube_faces[5]
    
    # Add labels if requested
    if with_labels:
        # Create a copy for drawing text
        if channels == 3:
            layout_with_labels = layout.copy()
        else:
            # Convert to color for drawing text
            layout_with_labels = cv2.cvtColor(layout, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
        positions = [
            (face_w*1.5, face_h*1.5),  # Front
            (face_w*2.5, face_h*1.5),  # Right
            (face_w*2.5, face_h*2.5),  # Back
            (face_w*0.5, face_h*1.5),  # Left
            (face_w*1.5, face_h*0.5),  # Top
            (face_w*1.5, face_h*2.5),  # Bottom
        ]
        
        for name, pos in zip(face_names, positions):
            cv2.putText(layout_with_labels, name, 
                        (int(pos[0] - 30), int(pos[1])), 
                        font, 0.7, (255, 255, 255), 2)
        
        return layout_with_labels
    
    return layout

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

def visualize_equirectangular(equirect_img, title="Equirectangular Panorama"):
    """
    Visualize an equirectangular panorama with grid lines.
    
    Args:
        equirect_img: Equirectangular image as numpy array
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if len(equirect_img.shape) > 2:
        ax.imshow(equirect_img)
    else:
        ax.imshow(equirect_img, cmap='gray')
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(linestyle='--', alpha=0.3)
    
    # Add longitude/latitude grid lines
    h, w = equirect_img.shape[:2]
    
    # Longitude lines (vertical)
    for lon in range(0, 361, 45):
        x = w * lon / 360
        ax.axvline(x=x, color='white', linestyle='--', alpha=0.3)
        ax.text(x, h-20, f"{lon-180}°", color='white', ha='center')
    
    # Latitude lines (horizontal)
    for lat in range(0, 181, 30):
        y = h * lat / 180
        ax.axhline(y=y, color='white', linestyle='--', alpha=0.3)
        ax.text(10, y, f"{90-lat}°", color='white', va='center')
    
    plt.tight_layout()
    return fig