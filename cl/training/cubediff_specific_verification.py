#!/usr/bin/env python
"""
Verify CubeDiff-specific properties of the WebDataset tar files.
Checks for cubemap latent structure and appropriate caption content.
"""

import argparse
import os
import torch
import numpy as np
import webdataset as wds
import io
from collections import Counter

def tensor_decoder(key, value):
    """Decode tensor from bytes for WebDataset."""
    if key.endswith('.pt'):
        buffer = io.BytesIO(value)
        return torch.load(buffer)
    return value

def verify_cubediff_data(tar_path, max_samples=10):
    """
    Verify CubeDiff-specific properties of a WebDataset tar file.
    
    Args:
        tar_path: Path to the tar file
        max_samples: Maximum number of samples to check in detail
    """
    if not os.path.exists(tar_path):
        print(f"Error: Tar file {tar_path} does not exist")
        return False
    
    print(f"Verifying CubeDiff-specific properties in {tar_path}")
    
    try:
        # Use WebDataset with proper tensor decoder
        dataset = wds.WebDataset(tar_path).map_dict(tensor_decoder)
        
        sample_count = 0
        valid_samples = 0
        
        # Check for CubeDiff-specific patterns in captions
        caption_topics = Counter()
        
        # Check the expected shape of latent tensors
        latent_shapes = set()
        
        # Process samples
        for sample in dataset:
            sample_count += 1
            is_valid = True
            
            # Check if sample has all required keys
            if not all(k in sample for k in ["__key__", "lat.pt", "txt"]):
                if sample_count <= max_samples:
                    print(f"Sample {sample_count} missing required keys")
                is_valid = False
            
            # Check latent
            if "lat.pt" in sample:
                latent = sample["lat.pt"]
                
                # Check if it's a tensor or numpy array
                if isinstance(latent, torch.Tensor):
                    shape = tuple(latent.shape)
                    latent_shapes.add(shape)
                    
                    # For CubeDiff, we expect a specific shape for cubemap faces
                    # The first dimension should typically be 6 (for the 6 cubemap faces)
                    if shape[0] != 6 and sample_count <= max_samples:
                        print(f"Sample {sample_count} has unexpected latent shape: {shape}")
                        print(f"  Expected first dimension to be 6 for cubemap faces")
                        is_valid = False
                        
                elif isinstance(latent, np.ndarray):
                    shape = latent.shape
                    latent_shapes.add(shape)
                    
                    # Check for cubemap faces
                    if shape[0] != 6 and sample_count <= max_samples:
                        print(f"Sample {sample_count} has unexpected latent shape: {shape}")
                        print(f"  Expected first dimension to be 6 for cubemap faces")
                        is_valid = False
                else:
                    if sample_count <= max_samples:
                        print(f"Sample {sample_count} has latent of unexpected type: {type(latent)}")
                    is_valid = False
            
            # Check caption
            if "txt" in sample:
                caption = sample["txt"]
                
                # Check if caption is not empty
                if not caption:
                    if sample_count <= max_samples:
                        print(f"Sample {sample_count} has empty caption")
                    is_valid = False
                
                # Extract general category from caption (for statistics)
                for keyword in ["abandoned", "factory", "room", "house", "building", 
                               "interior", "hall", "office", "industrial", "nature",
                               "outdoor", "indoor", "church", "temple", "space"]:
                    if keyword in caption.lower():
                        caption_topics[keyword] += 1
            
            if is_valid:
                valid_samples += 1
            
            # Print details for a few samples
            if sample_count <= max_samples:
                print(f"\nSample {sample_count} ({'valid' if is_valid else 'invalid'}):")
                print(f"  ID: {sample.get('__key__', 'Missing')}")
                
                if "lat.pt" in sample:
                    if isinstance(sample["lat.pt"], (torch.Tensor, np.ndarray)):
                        print(f"  Latent: {'Tensor' if isinstance(sample['lat.pt'], torch.Tensor) else 'Numpy array'} with shape {sample['lat.pt'].shape}")
                    else:
                        print(f"  Latent: {type(sample['lat.pt'])} (unexpected type)")
                
                if "txt" in sample:
                    caption = sample["txt"]
                    print(f"  Caption: '{caption[:100]}{'...' if len(caption) > 100 else ''}'")
            
            # Limit the number of samples to check
            if sample_count >= 50:  # Check more than max_samples for statistics
                break
        
        # Print summary
        print("\nSummary:")
        print(f"Total samples examined: {sample_count}")
        print(f"Valid samples: {valid_samples} ({valid_samples/sample_count*100:.1f}% of examined)")
        
        print("\nLatent shapes found:")
        for shape in latent_shapes:
            print(f"  {shape}")
        
        if caption_topics:
            print("\nTop caption topics:")
            for topic, count in caption_topics.most_common(10):
                print(f"  {topic}: {count} ({count/sample_count*100:.1f}%)")
        
        return valid_samples > 0
        
    except Exception as e:
        print(f"Error during CubeDiff verification: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify CubeDiff-specific properties")
    parser.add_argument("--train_tar", required=True, help="Path to training tar file")
    parser.add_argument("--val_tar", required=True, help="Path to validation tar file")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to check in detail")
    args = parser.parse_args()
    
    print("=== Verifying CubeDiff Properties in Training Tar ===")
    train_ok = verify_cubediff_data(args.train_tar, args.samples)
    
    print("\n=== Verifying CubeDiff Properties in Validation Tar ===")
    val_ok = verify_cubediff_data(args.val_tar, args.samples)
    
    if train_ok and val_ok:
        print("\n✅ Both tar files contain valid CubeDiff data")
        return 0
    else:
        print("\n❌ Issues found with CubeDiff-specific properties")
        return 1

if __name__ == "__main__":
    main()