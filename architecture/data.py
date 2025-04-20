"""
data.py - Dataset and data handling components for CubeDiff

This module contains classes and utilities for loading, processing, and visualizing
panoramic images and cubemaps used in the CubeDiff model.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the existing equirectangular-cubemap conversion utilities
from cubediff_utils_v1 import improved_equirect_to_cubemap, optimized_cubemap_to_equirect


class CubemapDataset(Dataset):
    """
    Dataset class for loading and processing panoramic images into cubemap format.
    This dataset loads equirectangular panoramas and converts them to cubemaps on-the-fly.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        caption_paths: Optional[List[str]] = None,
        face_size: int = 128,
        transform=None,
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to equirectangular panorama images
            caption_paths: Optional list of paths to caption files for the panoramas
            face_size: Size of each cubemap face in pixels
            transform: Optional transformations to apply to the images
        """
        self.image_paths = image_paths
        self.caption_paths = caption_paths
        self.face_size = face_size
        self.transform = transform
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that all paths exist"""
        for path in self.image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image path not found: {path}")
                
        if self.caption_paths:
            for path in self.caption_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Caption path not found: {path}")
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load an image from a path"""
        # Load using PIL and convert to RGB
        image = Image.open(path).convert("RGB")
        # Convert to numpy array
        return np.array(image)
    
    def _load_caption(self, path: str) -> str:
        """Load a caption from a path"""
        with open(path, 'r') as f:
            return f.read().strip()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a dataset item.
        
        Returns:
            Dictionary containing:
            - 'cubemap': Tensor of shape (6, 3, H, W) containing the 6 cubemap faces
            - 'caption': Text caption if caption_paths was provided
        """
        # Load equirectangular image
        equirect_img = self._load_image(self.image_paths[idx])
        
        # Convert to cubemap
        cube_faces = improved_equirect_to_cubemap(equirect_img, self.face_size)
        
        # Stack faces in a specific order: front, back, left, right, top, bottom
        # The paper uses this specific order for consistency
        face_order = ['front', 'back', 'left', 'right', 'top', 'bottom']
        faces_list = []
        
        for face_name in face_order:
            face = cube_faces[face_name]
            
            # Convert to float and normalize to [0, 1]
            if face.dtype == np.uint8:
                face = face.astype(np.float32) / 255.0
                
            # Apply any additional transformations
            if self.transform:
                face = self.transform(face)
            
            # Convert to PyTorch tensor and rearrange dimensions to (C, H, W)
            if isinstance(face, np.ndarray):
                face = torch.from_numpy(face).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            faces_list.append(face)
        
        # Stack the faces along first dimension to get (6, C, H, W)
        cubemap = torch.stack(faces_list, dim=0)
        
        # Prepare the output dictionary
        output = {'cubemap': cubemap}
        
        # Add caption if available
        if self.caption_paths:
            caption = self._load_caption(self.caption_paths[idx])
            output['caption'] = caption
            
        return output


def create_conditioning_mask(batch_size, num_faces=6, condition_face_idx=0):
    """
    Create a binary mask for conditioning the model on a specific face.
    
    Args:
        batch_size: Number of samples in the batch
        num_faces: Number of faces in the cubemap (typically 6)
        condition_face_idx: Index of the face to use as conditioning (default: 0 = front face)
        
    Returns:
        Binary mask of shape (batch_size, num_faces) where 1 indicates conditioning face
    """
    # Initialize mask with zeros
    mask = torch.zeros(batch_size, num_faces, dtype=torch.bool)
    
    # Set the conditioning face to 1
    mask[:, condition_face_idx] = 1
    
    return mask


def visualize_cubemap(cubemap_tensor, title="Cubemap Visualization"):
    """
    Visualize a cubemap tensor as a grid of faces.
    
    Args:
        cubemap_tensor: Tensor of shape (6, C, H, W) or (B, 6, C, H, W)
        title: Title for the plot
    """
    # Handle batch dimension if present
    if len(cubemap_tensor.shape) == 5:
        # Take the first sample from the batch
        cubemap_tensor = cubemap_tensor[0]
    
    # Convert to numpy and ensure it's in the range [0, 1]
    if isinstance(cubemap_tensor, torch.Tensor):
        cubemap_tensor = cubemap_tensor.detach().cpu().numpy()
    
    # Ensure the tensor is in the range [0, 1]
    if cubemap_tensor.max() > 1.0:
        cubemap_tensor = cubemap_tensor / 255.0
    
    # Create a grid layout for the 6 faces
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    for i, (face, name) in enumerate(zip(cubemap_tensor, face_names)):
        # Transpose to (H, W, C) for display
        if face.shape[0] == 3:  # If channel-first
            face = face.transpose(1, 2, 0)
        
        # Plot the face
        row, col = i // 3, i % 3
        axes[row, col].imshow(face)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def equirectangular_to_cubemap_batch(equirect_batch, face_size=128):
    """
    Convert a batch of equirectangular images to cubemap format.
    
    Args:
        equirect_batch: Batch of equirectangular images (B, C, H, W)
        face_size: Size of each cubemap face
    
    Returns:
        Batch of cubemaps (B, 6, C, face_size, face_size)
    """
    batch_size = equirect_batch.shape[0]
    device = equirect_batch.device
    
    # Process each image in the batch
    cubemap_list = []
    
    for i in range(batch_size):
        # Get equirectangular image and convert to numpy
        equirect = equirect_batch[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        
        # Convert to cubemap
        faces = improved_equirect_to_cubemap(equirect, face_size)
        
        # Arrange faces in order: front, back, left, right, top, bottom
        face_order = ['front', 'back', 'left', 'right', 'top', 'bottom']
        faces_tensor = []
        
        for face_name in face_order:
            # Convert to tensor
            face = faces[face_name]
            face = torch.from_numpy(face).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            if face.max() > 1.0:
                face = face / 255.0
            faces_tensor.append(face)
        
        # Stack faces
        cubemap = torch.stack(faces_tensor, dim=0)  # (6, C, H, W)
        cubemap_list.append(cubemap)
    
    # Stack along batch dimension
    cubemap_batch = torch.stack(cubemap_list, dim=0).to(device)  # (B, 6, C, H, W)
    
    return cubemap_batch


def cubemap_to_equirectangular_batch(cubemap_batch, height=1024, width=2048):
    """
    Convert a batch of cubemaps to equirectangular format.
    
    Args:
        cubemap_batch: Batch of cubemaps (B, 6, C, H, W)
        height: Height of the output equirectangular image
        width: Width of the output equirectangular image
    
    Returns:
        Batch of equirectangular images (B, C, height, width)
    """
    batch_size = cubemap_batch.shape[0]
    device = cubemap_batch.device
    
    # Process each cubemap in the batch
    equirect_list = []
    
    for i in range(batch_size):
        # Get cubemap and convert to numpy
        cubemap = cubemap_batch[i].cpu().numpy()  # (6, C, H, W)
        
        # Rearrange faces from [front, back, left, right, top, bottom] to a dictionary
        face_order = ['front', 'back', 'left', 'right', 'top', 'bottom']
        faces = {}
        
        for j, face_name in enumerate(face_order):
            # Get face and rearrange to (H, W, C)
            face = cubemap[j].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            # Scale to [0, 255] if in [0, 1]
            if face.max() <= 1.0:
                face = (face * 255.0).astype(np.uint8)
            
            faces[face_name] = face
        
        # Convert to equirectangular
        equirect = optimized_cubemap_to_equirect(faces, height, width)
        
        # Convert to tensor
        equirect = torch.from_numpy(equirect).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        if equirect.max() > 1.0:
            equirect = equirect / 255.0
        
        equirect_list.append(equirect)
    
    # Stack along batch dimension
    equirect_batch = torch.stack(equirect_list, dim=0).to(device)  # (B, C, H, W)
    
    return equirect_batch


def test_dataset(dataset, num_samples=2):
    """
    Test the dataset by visualizing a few samples.
    
    Args:
        dataset: CubemapDataset instance
        num_samples: Number of samples to visualize
    """
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        cubemap = sample['cubemap']
        caption = sample.get('caption', 'No caption available')
        
        print(f"Sample {i} - Cubemap shape: {cubemap.shape}")
        print(f"Caption: {caption}")
        
        visualize_cubemap(cubemap, title=f"Sample {i}")