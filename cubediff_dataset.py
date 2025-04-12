"""
Dataset classes for CubeDiff implementation.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

from cubediff_utils import equirect_to_cubemap


class CubemapDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading panoramic images and converting them to cubemap format.
    
    Args:
        panorama_paths: List of paths to panoramic images
        text_prompts: List of text prompts for each panorama
        face_size: Size of each cubemap face (default: 512)
        transform: Optional transform to apply to cubemap faces
    """
    def __init__(self, panorama_paths, text_prompts, face_size=512, transform=None):
        self.panorama_paths = panorama_paths
        self.text_prompts = text_prompts
        self.face_size = face_size
        self.transform = transform
        
        # Simple validation
        assert len(panorama_paths) == len(text_prompts), "Number of panoramas and prompts must match"
    
    def __len__(self):
        return len(self.panorama_paths)
    
    def __getitem__(self, idx):
        # Load panorama
        panorama_path = self.panorama_paths[idx]
        panorama = Image.open(panorama_path).convert("RGB")
        
        # Convert to cubemap
        cubemap_faces = equirect_to_cubemap(panorama, face_size=self.face_size)
        
        # Convert numpy arrays to PIL Images for transforms
        cubemap_face_pil = [Image.fromarray(face.astype('uint8')) for face in cubemap_faces]
        
        # Apply transforms if specified
        if self.transform:
            cubemap_face_pil = [self.transform(face) for face in cubemap_face_pil]
        else:
            # Default normalization if no transform is provided
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            cubemap_face_pil = [transform(face) for face in cubemap_face_pil]
        
        # Stack faces along a new dimension
        cubemap_tensor = torch.stack(cubemap_face_pil)
        
        return {
            "cubemap": cubemap_tensor,
            "prompt": self.text_prompts[idx],
            "path": panorama_path
        }


def create_dataloader(dataset_path, prompts_file, face_size=512, batch_size=1, 
                   world_size=1, rank=0, num_workers=4, distributed=False):
    """
    Create a dataloader for panorama dataset.
    
    Args:
        dataset_path: Path to directory containing panorama images
        prompts_file: Path to file containing prompts (one per line)
        face_size: Size of each cubemap face
        batch_size: Batch size
        world_size: Number of processes for distributed training
        rank: Process rank for distributed training
        num_workers: Number of worker processes for data loading
        distributed: Whether to use distributed training
        
    Returns:
        dataloader, dataset
    """
    # Get panorama paths
    panorama_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    
    # Make sure we have a prompt for each panorama
    if len(prompts) < len(panorama_paths):
        # Repeat prompts if needed
        prompts = prompts * (len(panorama_paths) // len(prompts) + 1)
    prompts = prompts[:len(panorama_paths)]
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    dataset = CubemapDataset(
        panorama_paths=panorama_paths,
        text_prompts=prompts,
        face_size=face_size,
        transform=transform
    )
    
    # Create sampler for distributed training
    if distributed and world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, dataset