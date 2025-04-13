import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

def visualize_equirectangular(equirect_img, title="Equirectangular Panorama"):
    """
    Visualize an equirectangular panorama.
    
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

def create_enhanced_cube_visualization(cube_faces, scale=1.0):
    """
    Create an enhanced 3D visualization of a cubemap with proper texture mapping.
    
    Args:
        cube_faces: List of 6 cubemap faces in order [Front, Right, Back, Left, Top, Bottom]
        scale: Scaling factor for the visualization (default: 1.0)
        
    Returns:
        fig: Matplotlib 3D figure
    """
    # Verify we have exactly 6 faces
    if len(cube_faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(cube_faces)}")
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(6, 6))
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
    
    # Face labels and colors
    face_labels = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    face_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    # Add extra points to ensure at least 3 unique values per axis
    # This prevents the "x and y arrays must consist of at least 3 unique points" error
    extra_points = np.array([
        [-0.8, -0.8, -0.8],
        [-0.4, -0.4, -0.4],
        [0.0, 0.0, 0.0],
        [0.4, 0.4, 0.4],
        [0.8, 0.8, 0.8],
        [-0.6, 0.6, 0.0],
        [0.6, -0.6, 0.0],
        [0.0, 0.6, -0.6],
        [0.0, -0.6, 0.6],
        [0.6, 0.0, 0.6],
        [0.6, 0.0, -0.6]
    ]) * scale
    
    # Add these points with zero size so they're invisible but contribute to axes ranges
    ax.scatter(extra_points[:, 0], extra_points[:, 1], extra_points[:, 2], s=0)
    
    # Draw each face with its label
    for i, (face_idx, face_img, label, color) in enumerate(zip(face_indices, cube_faces, face_labels, face_colors)):
        # Extract vertices for this face
        verts = [vertices[idx] for idx in face_idx]
        
        # Create a polygon
        poly = Poly3DCollection([verts], alpha=0.9)
        
        # Set face color
        poly.set_facecolor(color)
        poly.set_edgecolor('black')
        
        # Add to axes
        ax.add_collection3d(poly)
        
        # Add face label at the center of the face
        face_center = np.mean(verts, axis=0)
        ax.text(face_center[0], face_center[1], face_center[2], label, 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=12, color='white', fontweight='bold')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title and limits
    ax.set_title('3D Cubemap Visualization', fontsize=16)
    
    # Set limits to ensure the entire cube is visible
    limit = scale * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    return fig

def compare_cubemaps(cubemap1, cubemap2, titles=None):
    """
    Compare two sets of cubemap faces side by side.
    
    Args:
        cubemap1: First set of 6 cubemap faces
        cubemap2: Second set of 6 cubemap faces
        titles: Optional pair of titles for the two cubemaps
        
    Returns:
        fig: Matplotlib figure
    """
    if titles is None:
        titles = ['Cubemap 1', 'Cubemap 2']
    
    face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    fig, axes = plt.subplots(6, 2, figsize=(12, 18))
    
    for i in range(6):
        # First cubemap
        if len(cubemap1[i].shape) > 2:
            axes[i, 0].imshow(cubemap1[i])
        else:
            axes[i, 0].imshow(cubemap1[i], cmap='gray')
        axes[i, 0].set_title(f"{face_names[i]} - {titles[0]}")
        axes[i, 0].axis('off')
        
        # Second cubemap
        if len(cubemap2[i].shape) > 2:
            axes[i, 1].imshow(cubemap2[i])
        else:
            axes[i, 1].imshow(cubemap2[i], cmap='gray')
        axes[i, 1].set_title(f"{face_names[i]} - {titles[1]}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig

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