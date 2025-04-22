import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import json

class CubemapDataset(Dataset):
    """
    Dataset for loading cubemap faces with captions.
    """
    def __init__(self, data_dir, captions_file=None, image_size=512, channels_first=True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing panorama cubemap faces
            captions_file: JSON file containing captions for panoramas
            image_size: Size to resize images to (default: 512)
            channels_first: Whether to return images with channels first (NCHW format)
        """
        self.data_dir = data_dir
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        self.image_size = image_size
        self.channels_first = channels_first
        
        # Get all panorama directories
        self.pano_dirs = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
        
        # Load captions if provided
        self.captions = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r') as f:
                self.captions = json.load(f)
    
    def __len__(self):
        """Return the number of panoramas in the dataset."""
        return len(self.pano_dirs)
    
    def preprocess_image(self, image):
        """
        Preprocess a single image to be ready for the model.
        
        Args:
            image: PIL Image or np.ndarray
            
        Returns:
            Processed image as a torch tensor
        """
        if isinstance(image, Image.Image):
            # Resize image if needed
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            # Convert PIL Image to numpy array
            image = np.array(image)
        
        # Ensure image is float32 for processing
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
            
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Move channels to first dimension if needed (HWC -> CHW)
        if self.channels_first and image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        return image
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing:
                - faces: Tensor of shape (6, C, H, W) or (6, H, W, C) containing cubemap faces
                - caption: Caption for the panorama (if available)
        """
        pano_name = self.pano_dirs[idx]
        pano_dir = os.path.join(self.data_dir, pano_name)
        
        # Load the 6 cubemap faces
        faces = []
        for face_name in self.face_names:
            face_path = os.path.join(pano_dir, f"{face_name}.jpg")
            if os.path.exists(face_path):
                # Load and convert to RGB to ensure 3 channels
                face_img = Image.open(face_path).convert('RGB')
                # Preprocess image
                face_tensor = self.preprocess_image(face_img)
                faces.append(face_tensor)
            else:
                # If face doesn't exist, create a blank one
                if self.channels_first:
                    # (C, H, W) format
                    blank = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                else:
                    # (H, W, C) format
                    blank = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
                faces.append(blank)
        
        # Stack faces into a single tensor
        faces_tensor = torch.stack(faces, dim=0)
        
        # Get caption if available (use pano name as key)
        caption = ""
        if pano_name in self.captions:
            caption = self.captions[pano_name]
        
        return {
            'faces': faces_tensor,
            'caption': caption
        }

def get_dataloader(data_dir, captions_file=None, batch_size=2, num_workers=0, shuffle=True):
    """
    Create dataloader for cubemap dataset with memory-efficient settings.
    
    Args:
        data_dir: Directory containing panorama cubemap faces
        captions_file: JSON file containing captions for panoramas
        batch_size: Batch size (reduced to save memory)
        num_workers: Number of workers for data loading (set to 0 to avoid multiprocessing)
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader instance
    """
    # Create dataset with channels first (NCHW format) for compatibility with VAE
    dataset = CubemapDataset(data_dir, captions_file, channels_first=True)
    
    # Print sample shape for debugging
    sample = dataset[0]
    print(f"Dataset sample shape: faces={sample['faces'].shape}, dtype={sample['faces'].dtype}")
    print(f"Sample value range: min={sample['faces'].min():.4f}, max={sample['faces'].max():.4f}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader