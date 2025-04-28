#!/usr/bin/env python
"""
Verify WebDataset tar files by loading them with a dataloader.
This simulates how the data will be used during training.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import webdataset as wds
import io

def tensor_decoder(key, value):
    """Decode tensor from bytes for WebDataset."""
    if key.endswith('.pt'):
        buffer = io.BytesIO(value)
        return torch.load(buffer)
    return value

def create_dataloader(tar_path, batch_size=4):
    """Create a dataloader from a WebDataset tar file."""
    
    # Define dataset pipeline with proper decoder
    dataset = (
        wds.WebDataset(tar_path)
        .map_dict(tensor_decoder)  # Add custom decoder for tensors
        .to_tuple("__key__", "lat.pt", "txt")
        .batched(batch_size)
    )
    
    return dataset

def verify_with_dataloader(tar_path, max_batches=2):
    """Verify a tar file by loading batches through a dataloader."""
    
    if not os.path.exists(tar_path):
        print(f"Error: Tar file {tar_path} does not exist")
        return False
    
    print(f"Verifying {tar_path} using DataLoader")
    
    try:
        loader = create_dataloader(tar_path)
        
        # Process a few batches
        batch_count = 0
        for batch in loader:
            if batch_count >= max_batches:
                break
                
            keys, latents, captions = batch
            batch_count += 1
            
            print(f"\nBatch {batch_count}:")
            print(f"  Batch size: {len(keys)}")
            print(f"  Keys: {keys}")
            
            # Check latent tensor
            print(f"  Latent tensor shape: {latents.shape}")
            print(f"  Latent tensor type: {latents.dtype}")
            print(f"  Latent min/max values: {latents.min().item():.4f}/{latents.max().item():.4f}")
            
            # Check captions
            print(f"  Caption examples:")
            for i, caption in enumerate(captions[:2]):
                print(f"    {keys[i]}: {caption[:50]}{'...' if len(caption) > 50 else ''}")
        
        if batch_count > 0:
            print("\n✅ Successfully loaded data through DataLoader")
            return True
        else:
            print("\n❌ No batches were loaded from the dataset")
            return False
        
    except Exception as e:
        print(f"\n❌ Error while loading with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify WebDataset tar files with DataLoader")
    parser.add_argument("--train_tar", required=True, help="Path to training tar file")
    parser.add_argument("--val_tar", required=True, help="Path to validation tar file")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches to load")
    args = parser.parse_args()
    
    print("=== Verifying Training Tar with DataLoader ===")
    train_ok = verify_with_dataloader(args.train_tar, args.batches)
    
    print("\n=== Verifying Validation Tar with DataLoader ===")
    val_ok = verify_with_dataloader(args.val_tar, args.batches)
    
    if train_ok and val_ok:
        print("\n✅ Both tar files successfully loaded with DataLoader")
        return 0
    else:
        print("\n❌ Issues found when loading one or both tar files")
        return 1

if __name__ == "__main__":
    main()