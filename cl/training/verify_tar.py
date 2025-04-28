#!/usr/bin/env python
"""
Verify WebDataset tar files for CubeDiff training.
This script inspects the tar files to ensure they contain the expected data.
"""

import argparse
import webdataset as wds
import torch
import os
import sys

def verify_tar(tar_path, max_samples=5):
    """
    Verify the contents of a WebDataset tar file.
    
    Args:
        tar_path: Path to the tar file
        max_samples: Maximum number of samples to print
    """
    if not os.path.exists(tar_path):
        print(f"Error: Tar file {tar_path} does not exist")
        return False
    
    # Print basic file info
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"Tar file: {tar_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Try to open and iterate through the dataset
    try:
        dataset = wds.WebDataset(tar_path)
        
        # Count samples and check structure
        sample_count = 0
        keys_present = set()
        keys_missing = set()
        latent_shapes = set()
        caption_lengths = []
        
        print("\nSample contents:")
        for i, sample in enumerate(dataset):
            sample_count += 1
            
            # Track which keys are present
            for key in ["__key__", "lat.pt", "txt"]:
                if key in sample:
                    keys_present.add(key)
                else:
                    keys_missing.add(key)
            
            # Check latent shape if present
            if "lat.pt" in sample:
                if isinstance(sample["lat.pt"], torch.Tensor):
                    latent_shapes.add(tuple(sample["lat.pt"].shape))
                else:
                    latent_shapes.add(f"Not a tensor: {type(sample['lat.pt'])}")
            
            # Track caption lengths
            if "txt" in sample:
                caption_lengths.append(len(sample["txt"]))
            
            # Print sample details for a few samples
            if i < max_samples:
                print(f"\nSample {i+1}:")
                print(f"  ID: {sample.get('__key__', 'Missing')}")
                
                if "lat.pt" in sample:
                    if isinstance(sample["lat.pt"], torch.Tensor):
                        print(f"  Latent: Tensor with shape {sample['lat.pt'].shape}")
                    else:
                        print(f"  Latent: {type(sample['lat.pt'])} (not a tensor)")
                else:
                    print("  Latent: Missing")
                
                if "txt" in sample:
                    caption = sample["txt"]
                    print(f"  Caption: '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
                else:
                    print("  Caption: Missing")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total samples: {sample_count}")
        print(f"Keys present in all samples: {keys_present}")
        print(f"Keys missing from some samples: {keys_missing}")
        print(f"Latent shapes found: {latent_shapes}")
        
        if caption_lengths:
            avg_caption_len = sum(caption_lengths) / len(caption_lengths)
            min_caption_len = min(caption_lengths)
            max_caption_len = max(caption_lengths)
            print(f"Caption length - Min: {min_caption_len}, Avg: {avg_caption_len:.1f}, Max: {max_caption_len}")
        
        return sample_count > 0
        
    except Exception as e:
        print(f"Error while verifying tar file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify WebDataset tar files")
    parser.add_argument("--train_tar", required=True, help="Path to training tar file")
    parser.add_argument("--val_tar", required=True, help="Path to validation tar file")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to print")
    args = parser.parse_args()
    
    print("=== Verifying Training Tar ===")
    train_ok = verify_tar(args.train_tar, args.samples)
    
    print("\n=== Verifying Validation Tar ===")
    val_ok = verify_tar(args.val_tar, args.samples)
    
    if train_ok and val_ok:
        print("\n✅ Both tar files appear to be valid WebDatasets")
        return 0
    else:
        print("\n❌ Issues found with one or both tar files")
        return 1

if __name__ == "__main__":
    sys.exit(main())