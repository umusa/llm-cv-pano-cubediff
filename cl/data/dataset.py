import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import json

class CubemapDataset(Dataset):
    def __init__(self, data_dir, captions_file=None, transform=None):
        """
        Dataset for cubemap faces.
        
        Args:
            data_dir: Directory containing panorama cubemap faces
            captions_file: JSON file containing captions for panoramas
            transform: Transformations to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Load panorama folders
        self.pano_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Load captions if available
        self.captions = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r') as f:
                self.captions = json.load(f)
    
    def __len__(self):
        return len(self.pano_folders)
    
    def __getitem__(self, idx):
        pano_folder = self.pano_folders[idx]
        pano_path = os.path.join(self.data_dir, pano_folder)
        
        # Load all 6 faces
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        faces = []
        
        for face_name in face_names:
            face_path = os.path.join(pano_path, f"{face_name}.jpg")
            if os.path.exists(face_path):
                face_img = Image.open(face_path).convert('RGB')
                face_tensor = self.transform(face_img)
                faces.append(face_tensor)
            else:
                # Handle missing faces (shouldn't happen with proper preprocessing)
                faces.append(torch.zeros(3, 512, 512))
        
        # Stack faces along a new dimension
        faces_tensor = torch.stack(faces, dim=0)  # Shape: [6, C, H, W]
        
        # Get caption if available
        caption = self.captions.get(pano_folder, "A panoramic view")
        
        return {
            "faces": faces_tensor,
            "caption": caption,
            "pano_id": pano_folder
        }

def get_dataloader(data_dir, captions_file=None, batch_size=4, num_workers=4, shuffle=True):
    """
    Create dataloader for cubemap dataset.
    
    Args:
        data_dir: Directory containing panorama cubemap faces
        captions_file: JSON file containing captions for panoramas
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader instance
    """
    dataset = CubemapDataset(data_dir, captions_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader