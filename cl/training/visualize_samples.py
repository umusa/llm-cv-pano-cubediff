#!/usr/bin/env python
"""
Extract and visualize random samples from WebDataset tar files.
This helps visually verify that the data is correct.
"""

import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds
import io

def tensor_decoder(key, value):
    """Decode tensor from bytes for WebDataset."""
    if key.endswith('.pt'):
        try:
            buffer = io.BytesIO(value)
            return torch.load(buffer)
        except Exception as e:
            print(f"Error decoding tensor: {e}")
            return value
    return value

def extract_random_samples(tar_path, output_dir, num_samples=3):
    """
    Extract random samples from a WebDataset tar file and save information.
    
    Args:
        tar_path: Path to the tar file
        output_dir: Directory to save sample information
        num_samples: Number of random samples to extract
    """
    if not os.path.exists(tar_path):
        print(f"Error: Tar file {tar_path} does not exist")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load all samples into memory (with proper tensor decoding)
        dataset = wds.WebDataset(tar_path).map_dict(tensor_decoder)
        samples = list(dataset)
        
        if not samples:
            print(f"Error: No samples found in {tar_path}")
            return False
        
        # Select random samples
        if len(samples) <= num_samples:
            selected_samples = samples
        else:
            selected_samples = random.sample(samples, num_samples)
        
        # Process each selected sample
        for i, sample in enumerate(selected_samples):
            sample_id = sample.get("__key__", f"unknown_{i}")
            print(f"Processing sample: {sample_id}")
            
            # Create a sanitized filename (no problematic characters)
            safe_id = sample_id.replace('/', '_').replace('\\', '_')
            
            # Save sample information
            info_path = os.path.join(output_dir, f"{safe_id}_info.txt")
            with open(info_path, "w") as f:
                f.write(f"Sample ID: {sample_id}\n\n")
                
                # Save caption
                if "txt" in sample:
                    f.write(f"Caption: {sample['txt']}\n\n")
                else:
                    f.write("Caption: Missing\n\n")
                
                # Save latent information
                if "lat.pt" in sample:
                    latent = sample["lat.pt"]
                    
                    # Handle different types of latent data
                    if isinstance(latent, bytes):
                        f.write(f"Latent type: bytes (need to decode)\n")
                        try:
                            # Try to decode bytes to tensor again
                            buffer = io.BytesIO(latent)
                            latent = torch.load(buffer)
                            f.write(f"Successfully decoded to tensor with shape: {latent.shape}\n")
                        except Exception as e:
                            f.write(f"Failed to decode bytes to tensor: {str(e)}\n")
                            continue
                    
                    if isinstance(latent, (torch.Tensor, np.ndarray)):
                        f.write(f"Latent shape: {latent.shape}\n")
                        
                        if isinstance(latent, torch.Tensor):
                            f.write(f"Latent type: torch.Tensor (dtype={latent.dtype})\n")
                            # Convert to numpy for consistent handling
                            latent_np = latent.detach().cpu().numpy()
                        else:
                            f.write(f"Latent type: numpy.ndarray (dtype={latent.dtype})\n")
                            latent_np = latent
                        
                        f.write(f"Latent statistics:\n")
                        f.write(f"  Min: {np.min(latent_np):.6f}\n")
                        f.write(f"  Max: {np.max(latent_np):.6f}\n")
                        f.write(f"  Mean: {np.mean(latent_np):.6f}\n")
                        f.write(f"  Std: {np.std(latent_np):.6f}\n")
                        
                        # For CubeDiff, latent should have shape [6, C, H, W] for 6 cubemap faces
                        if latent_np.shape[0] == 6:
                            f.write("\nCubemap face statistics:\n")
                            for face_idx in range(6):
                                face = latent_np[face_idx]
                                face_name = ["right", "left", "up", "down", "front", "back"][face_idx]
                                f.write(f"  Face {face_idx} ({face_name}):\n")
                                f.write(f"    Min: {np.min(face):.6f}\n")
                                f.write(f"    Max: {np.max(face):.6f}\n")
                                f.write(f"    Mean: {np.mean(face):.6f}\n")
                                f.write(f"    Std: {np.std(face):.6f}\n")
                            
                            # Visualize latent features if they're 2D
                            if len(latent_np.shape) == 4:  # [6, C, H, W]
                                # Create visualization of the first channel of each face
                                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                                axs = axs.flatten()
                                
                                for face_idx in range(6):
                                    # Use the first channel for visualization
                                    face_channel = latent_np[face_idx, 0]
                                    face_name = ["right", "left", "up", "down", "front", "back"][face_idx]
                                    
                                    # Normalize for better visualization
                                    face_vis = (face_channel - np.min(face_channel)) / (np.max(face_channel) - np.min(face_channel) + 1e-8)
                                    
                                    axs[face_idx].imshow(face_vis, cmap='viridis')
                                    axs[face_idx].set_title(f"Face {face_idx} ({face_name})")
                                    axs[face_idx].axis('off')
                                
                                plt.suptitle(f"Sample: {sample_id} - First Channel Visualization")
                                plt.tight_layout()
                                
                                # Save the visualization
                                vis_path = os.path.join(output_dir, f"{safe_id}_visualization.png")
                                plt.savefig(vis_path)
                                plt.close()
                                
                                print(f"  Saved visualization to {vis_path}")
                    else:
                        f.write(f"Latent type: {type(latent)} (unexpected type)\n")
                else:
                    f.write("Latent: Missing\n")
            
            print(f"  Saved information to {info_path}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting samples: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize random samples")
    parser.add_argument("--tar_path", required=True, help="Path to tar file")
    parser.add_argument("--output_dir", required=True, help="Directory to save sample information")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random samples to extract")
    args = parser.parse_args()
    
    print(f"Extracting {args.num_samples} random samples from {args.tar_path}")
    success = extract_random_samples(args.tar_path, args.output_dir, args.num_samples)
    
    if success:
        print(f"\n✅ Successfully extracted and visualized samples to {args.output_dir}")
    else:
        print(f"\n❌ Failed to extract samples")

if __name__ == "__main__":
    main()