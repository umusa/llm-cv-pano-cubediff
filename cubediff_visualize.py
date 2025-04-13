"""
Visualization utilities for CubeDiff panorama generation results.
This module provides functions to display individual cubemap faces,
create panoramic views, and visualize 3D cube representations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from mpl_toolkits.mplot3d import Axes3D
    MPLOT3D_AVAILABLE = True
except ImportError:
    MPLOT3D_AVAILABLE = False


def display_faces_with_titles(result, prompt):
    """
    Display individual cubemap faces with descriptive titles.
    
    Args:
        result: Tensor or numpy array of shape [batch_size, 6, channels, height, width]
               or [6, channels, height, width] containing the 6 cubemap faces.
        prompt: The text prompt used to generate the images.
    """
    # Create a figure for the individual faces
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Define descriptive titles for each face
    face_titles = [
        "Front View - Primary Perspective",
        "Right View - Sunset Illumination",
        "Left View - Lake Extension",
        "Back View - Mountain Range",
        "Top View - Sky and Clouds",
        "Bottom View - Ground and Terrain"
    ]
    
    # Get the tensor if not already a numpy array
    if isinstance(result, torch.Tensor):
        result_np = result.detach().cpu().numpy()
    else:
        result_np = result
    
    # Handle different dimensions
    if result_np.ndim == 5:  # [batch, faces, channels, height, width]
        result_np = result_np[0]  # Take first batch
    
    # Display each face with its detailed title
    for i in range(6):
        # Get face from result
        face = result_np[i]
        
        # Convert from [C, H, W] to [H, W, C] for plotting
        face = np.transpose(face, (1, 2, 0))
        
        # Ensure values are in [0, 1] range for imshow
        if face.max() > 1.0:
            face = face / 255.0
        
        # Plot with enhanced title
        axes[i].imshow(face)
        axes[i].set_title(f"{face_titles[i]}", fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle(f"Generated Cubemap for: '{prompt}'", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for suptitle
    plt.show()


def create_and_display_panorama(result, prompt):
    """
    Create and display a panoramic view from cubemap faces.
    
    Args:
        result: Tensor or numpy array of shape [batch_size, 6, channels, height, width]
               or [6, channels, height, width] containing the 6 cubemap faces.
        prompt: The text prompt used to generate the images.
    
    Returns:
        Numpy array of the panorama if successful, None otherwise.
    """
    if not CV2_AVAILABLE:
        print("OpenCV (cv2) not available. Install with: pip install opencv-python")
        return None
    
    # Get the numpy array if it's a tensor
    if isinstance(result, torch.Tensor):
        result_np = result.detach().cpu().numpy()
    else:
        result_np = result
    
    # Handle different dimensions
    if result_np.ndim == 5:  # [batch, faces, channels, height, width]
        result_np = result_np[0]  # Take first batch
    
    # Extract faces and ensure they are in the right format
    faces = []
    for i in range(6):
        face = result_np[i]
        
        # Convert from [C, H, W] to [H, W, C] for processing
        face = np.transpose(face, (1, 2, 0))
        
        # Ensure values are in [0, 1] range
        if face.max() <= 1.0:
            face = (face * 255).astype(np.uint8)
        else:
            face = face.astype(np.uint8)
        
        faces.append(face)
    
    # Create a simple panorama by stitching horizontally
    # For a proper panorama, we would need more complex conversion from cubemap to equirectangular
    
    # Get face dimensions
    h, w = faces[0].shape[:2]
    
    # Order: Front, Right, Back, Left
    panorama = np.hstack([faces[0], faces[1], faces[3], faces[2]])
    
    # Add top and bottom faces to the right side
    side_panel = np.vstack([faces[4], faces[5]])
    
    # Resize side panel to match height of the panorama
    side_panel_resized = cv2.resize(side_panel, (w, panorama.shape[0]))
    
    # Combine main panorama with side panel
    full_panorama = np.hstack([panorama, side_panel_resized])
    
    # Display the panorama
    plt.figure(figsize=(20, 5))
    plt.imshow(full_panorama)
    plt.title(f"Simplified Panoramic View: '{prompt}'", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return full_panorama


def display_3d_cube(result, prompt):
    """
    Display a 3D cube representation with face textures.
    
    Args:
        result: Tensor or numpy array of shape [batch_size, 6, channels, height, width]
               or [6, channels, height, width] containing the 6 cubemap faces.
        prompt: The text prompt used to generate the images.
    """
    if not MPLOT3D_AVAILABLE:
        print("3D plotting not available. Check that mplot3d is properly installed.")
        return
    
    # Create a figure for 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the vertices of the cube
    r = 1  # Cube radius
    vertices = [
        [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
        [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
    ]
    
    # Define the faces of the cube by indices to vertices
    faces = [
        [0, 1, 2, 3],  # Bottom face (negative z)
        [4, 5, 6, 7],  # Top face (positive z)
        [0, 1, 5, 4],  # Front face (negative y)
        [2, 3, 7, 6],  # Back face (positive y)
        [0, 3, 7, 4],  # Left face (negative x)
        [1, 2, 6, 5]   # Right face (positive x)
    ]
    
    # Plot each face
    for i, face in enumerate(faces):
        # Extract vertices for this face
        x = [vertices[j][0] for j in face]
        y = [vertices[j][1] for j in face]
        z = [vertices[j][2] for j in face]
        
        # Create a polygon for this face
        ax.plot_trisurf(x, y, z, color='white', alpha=0.8)
        
        # Calculate center of this face for labeling
        center_x = sum(x) / 4
        center_y = sum(y) / 4
        center_z = sum(z) / 4
        
        # Add a label
        ax.text(center_x*1.2, center_y*1.2, center_z*1.2, f"Face {i+1}", ha='center')
    
    # Set plot parameters
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Cubemap Representation: '{prompt}'", fontsize=14)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=30)
    
    plt.tight_layout()
    plt.show()


def equirectangular_from_cubemap(result, target_width=1024):
    """
    Convert cubemap faces to equirectangular panorama (more advanced method).
    
    Args:
        result: Tensor or numpy array of shape [batch_size, 6, channels, height, width]
               or [6, channels, height, width] containing the 6 cubemap faces.
        target_width: Width of the output equirectangular panorama.
        
    Returns:
        Numpy array of the equirectangular panorama if successful, None otherwise.
    """
    if not CV2_AVAILABLE:
        print("OpenCV (cv2) not available. Install with: pip install opencv-python")
        return None
    
    # Get the numpy array if it's a tensor
    if isinstance(result, torch.Tensor):
        result_np = result.detach().cpu().numpy()
    else:
        result_np = result
    
    # Handle different dimensions
    if result_np.ndim == 5:  # [batch, faces, channels, height, width]
        result_np = result_np[0]  # Take first batch
    
    # Extract faces and ensure they are in the right format
    faces = []
    for i in range(6):
        face = result_np[i]
        
        # Convert from [C, H, W] to [H, W, C] for processing
        face = np.transpose(face, (1, 2, 0))
        
        # Ensure values are in [0, 1] range
        if face.max() <= 1.0:
            face = (face * 255).astype(np.uint8)
        else:
            face = face.astype(np.uint8)
        
        faces.append(face)
    
    # Create equirectangular panorama
    # This is a simplified placeholder implementation
    # For a proper implementation, you would:
    # 1. Create an equirectangular grid of sample points
    # 2. For each point, compute the 3D ray from the center of the view
    # 3. Determine which face of the cube this ray intersects
    # 4. Sample the corresponding pixel from that face
    
    # For now, we'll use the simpler stitching approach as a placeholder
    return create_and_display_panorama(result, "Advanced Equirectangular Panorama")


def visualize_all(result, prompt):
    """
    Run all visualization methods on the result.
    
    Args:
        result: Tensor or numpy array of shape [batch_size, 6, channels, height, width]
               or [6, channels, height, width] containing the 6 cubemap faces.
        prompt: The text prompt used to generate the images.
    """
    print(f"Visualizing results for prompt: '{prompt}'")
    
    # Display individual faces with titles
    display_faces_with_titles(result, prompt)
    
    # Attempt to create panorama
    try:
        if CV2_AVAILABLE:
            create_and_display_panorama(result, prompt)
        else:
            print("OpenCV (cv2) not available. Install with: pip install opencv-python")
    except Exception as e:
        print(f"Could not create panorama visualization: {e}")
    
    # Attempt to create 3D cube visualization
    try:
        if MPLOT3D_AVAILABLE:
            display_3d_cube(result, prompt)
        else:
            print("3D plotting not available. Check that mplot3d is properly installed.")
    except Exception as e:
        print(f"Could not create 3D cube visualization: {e}")
    
    print("Visualization complete.")


# Export main functions
__all__ = [
    'display_faces_with_titles',
    'create_and_display_panorama',
    'display_3d_cube',
    'equirectangular_from_cubemap',
    'visualize_all'
]