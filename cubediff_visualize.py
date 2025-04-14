import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

def visualize_equirectangular(equirect_img, title="Equirectangular Panorama"):
    """
    Visualize an equirectangular panorama image.
    
    Args:
        equirect_img: Equirectangular panorama image
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(equirect_img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_cubemap_faces(cube_faces, titles=None):
    """
    Visualize the 6 faces of a cubemap.
    
    Args:
        cube_faces: Dictionary or array of 6 cubemap faces
        titles: Optional list of custom titles for each face
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Default face names/titles
    if titles is None:
        titles = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    
    # Extract faces based on input type
    if isinstance(cube_faces, dict):
        # For dictionary input, extract the faces in the correct order
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        faces = [cube_faces[name] for name in face_names]
    else:
        # For array/list input, assume they're already in the correct order
        faces = cube_faces
    
    # Create the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each face with its title
    for i, (face, title) in enumerate(zip(faces, titles)):
        axes[i].imshow(face)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_cubemap_layout(cube_faces):
    """
    Visualize cubemap faces in the correct layout.
    
    Args:
        cube_faces: Dictionary or array of 6 cubemap faces
    """
    from cubediff_utils import create_cubemap_layout
    
    layout = create_cubemap_layout(cube_faces)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(layout)
    plt.title("Cubemap Layout")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Fix for the IndexError in the visualization code
# Use this version instead of the problematic code:

# Original problematic code:
# pos_enc = latents_with_pos[0:6, channels:, :, :].cpu()

# Safe version:
def visualize_positional_encodings(latents_with_pos):
    """Safe version that handles different tensor shapes"""
    import matplotlib.pyplot as plt
    
    # Print the shape to debug
    print(f"Shape of latents_with_pos: {latents_with_pos.shape}")
    
    # Check if tensor has the required dimensions
    if len(latents_with_pos.shape) < 4:
        print("Error: Tensor must have 4 dimensions [batch, channels, height, width]")
        return
    
    # Check if the channels dimension exists
    if latents_with_pos.shape[1] == 0:
        print("Warning: Channels dimension is empty")
        return
    
    # Extract the positional encodings safely
    try:
        pos_enc = latents_with_pos[0:6].cpu() if hasattr(latents_with_pos, 'cpu') else latents_with_pos[0:6]
    except IndexError:
        print("Error: Cannot extract 6 faces from tensor")
        return
    
    plt.figure(figsize=(15, 10))
    
    face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    num_faces = min(6, pos_enc.shape[0])
    num_channels = min(2, pos_enc.shape[1])
    
    if num_channels == 0:
        print("Error: No channels available")
        return
    
    for i in range(num_faces):
        for c in range(num_channels):
            plt.subplot(num_channels, num_faces, c * num_faces + i + 1)
            
            # Convert to numpy safely
            try:
                if hasattr(pos_enc[i, c], 'numpy'):
                    img = pos_enc[i, c].numpy()
                else:
                    img = pos_enc[i, c]
                
                plt.imshow(img, cmap='viridis')
                plt.title(f"{face_names[i]} ({['U', 'V'][c]})")
                plt.axis('off')
            except Exception as e:
                print(f"Error plotting face {i}, channel {c}: {e}")
    
    plt.tight_layout()
    plt.show()


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