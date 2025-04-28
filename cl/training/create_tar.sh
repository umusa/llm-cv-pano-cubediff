#!/bin/bash
# This script provides a complete solution to the WebDataset issues

echo "===== Fixing WebDataset Issues for CubeDiff Data ====="

# Step 1: Create a modified version of make_train_val_tars.py with proper key naming
cat > make_train_val_tars.py << 'EOF'
#!/usr/bin/env python
"""
Create WebDataset train / val tars from CubeDiff latents.
"""

import argparse, glob, json, os, random, re, sys
import numpy as np, torch, webdataset as wds
import io

def load_captions(path):
    if path.endswith(".jsonl"):
        caps = {j["id"]: j["caption"] for j in map(json.loads, open(path))}
    else:                                   # plain JSON object
        caps = json.load(open(path))
        # if the file contains {"root": {...}} pattern (as in VS-Code preview)
        if list(caps.keys()) == ["root"]:
            caps = caps["root"]
    return caps

def load_latent(file_path):
    if file_path.endswith(".pt"):
        return torch.load(file_path, map_location="cpu")
    else:                                   # .npy
        return np.load(file_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lat_root",   required=True,
                   help="directory containing *.pt / *.npy files")
    p.add_argument("--captions",   required=True,
                   help="captions.json or captions.jsonl")
    p.add_argument("--train_tar",  default="cubediff_train.tar")
    p.add_argument("--val_tar",    default="cubediff_val.tar")
    p.add_argument("--val_frac",   type=float, default=0.075)  # ≈50/700
    p.add_argument("--seed",       type=int,   default=1337)
    p.add_argument("--debug",      action="store_true", help="Print debug information")
    args = p.parse_args()

    # Load captions
    caps = load_captions(args.captions)
    if args.debug:
        print(f"Loaded {len(caps)} captions")
        print(f"First 3 caption keys: {list(caps.keys())[:3]}")

    # Get all latent files from the directory
    lat_files = sorted(glob.glob(os.path.join(args.lat_root, "*.pt")))
    if not lat_files:
        lat_files = sorted(glob.glob(os.path.join(args.lat_root, "*.npy")))
    
    if not lat_files:
        sys.exit(f"No *.pt or *.npy files found in {args.lat_root}")
    
    if args.debug:
        print(f"Found {len(lat_files)} latent files")
        print(f"First 3 latent files: {lat_files[:3]}")

    # Extract IDs from filenames
    ids = []
    for f in lat_files:
        base = os.path.basename(f)
        # Remove the file extension (.pt or .npy)
        file_id = os.path.splitext(base)[0]
        # If the ID ends with "_lat", remove it
        if file_id.endswith("_lat"):
            file_id = file_id[:-4]
        ids.append(file_id)
    
    if args.debug:
        print(f"Extracted {len(ids)} IDs")
        print(f"First 3 IDs: {ids[:3]}")
        # Check if these IDs exist in captions
        for id in ids[:3]:
            print(f"ID {id} exists in captions: {id in caps}")

    # Shuffle and split into train/val
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    
    if args.debug:
        print(f"Split into {len(train_ids)} train and {len(val_ids)} val")

    # Process and write the tar files
    process_split(train_ids, args.train_tar, args.lat_root, caps, args.debug, "train")
    process_split(val_ids, args.val_tar, args.lat_root, caps, args.debug, "val")

def process_split(ids, tar_path, lat_root, captions, debug, split_name):
    """Process and write a dataset split (train or val) to a tar file."""
    # Use standard file format that web dataset expects
    sink = wds.TarWriter(tar_path)
    print(f"Processing {split_name} set...")
    
    processed = 0
    missing = 0
    total = len(ids)
    
    for pid in ids:
        # Try different possible filename patterns
        possible_files = [
            os.path.join(lat_root, f"{pid}.pt"),
            os.path.join(lat_root, f"{pid}.npy"),
            os.path.join(lat_root, f"{pid}_lat.pt"),
            os.path.join(lat_root, f"{pid}_lat.npy")
        ]
        
        found = False
        for f in possible_files:
            if os.path.exists(f):
                found = True
                try:
                    # Load the latent
                    latent_data = load_latent(f)
                    
                    # Create a buffer to hold the tensor data
                    buffer = io.BytesIO()
                    if isinstance(latent_data, np.ndarray):
                        latent_data = torch.from_numpy(latent_data)
                    
                    # Save tensor in PyTorch format
                    torch.save(latent_data, buffer)
                    tensor_bytes = buffer.getvalue()
                    
                    # Ensure caption exists
                    caption = captions.get(pid, "")
                    
                    # Write sample with standard WebDataset naming conventions
                    # IMPORTANT: Use standard WebDataset naming like sample_id.pt, sample_id.txt
                    sample = {
                        "__key__": pid,
                        f"{pid}.pt": tensor_bytes,  # Use standard .pt extension
                        f"{pid}.txt": caption       # Use standard .txt extension
                    }
                    sink.write(sample)
                    
                    processed += 1
                    if debug and processed % 50 == 0:
                        print(f"Processed {processed}/{total} in {split_name}")
                except Exception as e:
                    print(f"Error processing {pid}: {str(e)}")
                    continue
                break
        
        if not found:
            if debug:
                print(f"⚠ latent for {pid} not found, tried: {possible_files}")
            missing += 1
    
    sink.close()
    
    # Check file size
    file_size_mb = os.path.getsize(tar_path) / (1024*1024)
    print(f"✓ {split_name}: Processed {processed} samples ({missing} missing)")
    print(f"✓ {split_name}: Created {tar_path} ({file_size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
EOF

# Step 2: Create a simple tar verification script
cat > verify_simple.py << 'EOF'
#!/usr/bin/env python
"""
Simple verification script for WebDataset tar files.
"""

import argparse
import sys
import os
import webdataset as wds
import torch
import io

def verify_tar(path):
    """Verify a WebDataset tar file."""
    print(f"Verifying {path}...")
    
    if not os.path.exists(path):
        print(f"ERROR: File doesn't exist: {path}")
        return False
    
    # Get file size
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"File size: {size_mb:.1f} MB")
    
    # Try to open and iterate through the dataset
    try:
        ds = wds.WebDataset(path)
        
        # Count samples and check keys
        count = 0
        sample_keys = set()
        
        for sample in ds:
            count += 1
            for key in sample.keys():
                sample_keys.add(key)
            
            if count <= 3:
                print(f"\nSample {count}:")
                print(f"  Keys: {list(sample.keys())}")
                
                # Get and print the ID
                if "__key__" in sample:
                    print(f"  ID: {sample['__key__']}")
                
                # Check for tensor data with standard naming
                tensor_keys = [k for k in sample.keys() if k.endswith('.pt')]
                if tensor_keys:
                    tensor_key = tensor_keys[0]
                    # Try to load the tensor
                    try:
                        tensor_data = sample[tensor_key]
                        if isinstance(tensor_data, bytes):
                            buffer = io.BytesIO(tensor_data)
                            tensor = torch.load(buffer)
                            print(f"  Tensor shape: {tensor.shape}")
                        elif isinstance(tensor_data, torch.Tensor):
                            print(f"  Tensor shape: {tensor_data.shape}")
                        else:
                            print(f"  Tensor type: {type(tensor_data)}")
                    except Exception as e:
                        print(f"  Error loading tensor: {str(e)}")
                
                # Check for caption with standard naming
                text_keys = [k for k in sample.keys() if k.endswith('.txt')]
                if text_keys:
                    text_key = text_keys[0]
                    caption = sample[text_key]
                    print(f"  Caption: {caption[:50]}{'...' if len(caption) > 50 else ''}")
        
        print(f"\nFound {count} samples with keys: {sample_keys}")
        return count > 0
    
    except Exception as e:
        print(f"ERROR verifying tar: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple verification for WebDataset tar files")
    parser.add_argument("--train_tar", required=True, help="Path to training tar file")
    parser.add_argument("--val_tar", required=True, help="Path to validation tar file")
    args = parser.parse_args()
    
    train_ok = verify_tar(args.train_tar)
    val_ok = verify_tar(args.val_tar)
    
    if train_ok and val_ok:
        print("\n✅ Both tar files appear valid")
        return 0
    else:
        print("\n❌ Issues found with one or both tar files")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Step 3: Create a WebDataset loading script
cat > load_webdataset.py << 'EOF'
#!/usr/bin/env python
"""
Load and use WebDataset tar files for CubeDiff.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import webdataset as wds
import io
import matplotlib.pyplot as plt
import numpy as np

def tensor_from_bytes(tensor_bytes):
    """Load a tensor from bytes."""
    buffer = io.BytesIO(tensor_bytes)
    tensor = torch.load(buffer)
    return tensor

class CubeDiffDataset:
    """Dataset for loading CubeDiff WebDataset tar files."""
    
    def __init__(self, tar_paths, batch_size=4, shuffle=False):
        """
        Initialize the dataset.
        
        Args:
            tar_paths: Path or list of paths to tar files
            batch_size: Batch size for loading
            shuffle: Whether to shuffle the dataset
        """
        if isinstance(tar_paths, str):
            tar_paths = [tar_paths]
        
        # Create the dataset pipeline
        self.dataset = (
            wds.WebDataset(tar_paths)
            .decode()
            .map(self._process_sample)
            .batched(batch_size, partial=False)
        )
        
        # Create an iterable version that can be reset
        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,  # batching is done by WebDataset
            shuffle=shuffle,
            num_workers=0     # for debugging
        )
    
    def _process_sample(self, sample):
        """Process a WebDataset sample."""
        # Get the sample ID
        sample_id = sample["__key__"]
        
        # Find the tensor data (key ending with .pt)
        tensor_key = next((k for k in sample.keys() if k.endswith('.pt')), None)
        if tensor_key is None:
            raise ValueError(f"No tensor data found in sample {sample_id}")
        
        # Load the tensor from bytes if needed
        tensor_data = sample[tensor_key]
        if isinstance(tensor_data, bytes):
            tensor_data = tensor_from_bytes(tensor_data)
        
        # Find the caption data (key ending with .txt)
        text_key = next((k for k in sample.keys() if k.endswith('.txt')), None)
        if text_key is None:
            raise ValueError(f"No text data found in sample {sample_id}")
        
        text_data = sample[text_key]
        
        # Return processed data
        return {
            "id": sample_id,
            "latent": tensor_data,
            "caption": text_data
        }
    
    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self.loader)

def visualize_cubemap(tensor, sample_id, save_path=None):
    """Visualize a cubemap tensor."""
    if tensor.dim() != 4 or tensor.shape[0] != 6:
        raise ValueError(f"Expected tensor with shape [6, C, H, W], got {tensor.shape}")
    
    # Create a figure with 6 subplots (2x3 grid)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Names for the cubemap faces
    face_names = ["right", "left", "up", "down", "front", "back"]
    
    # Visualize each face
    for i in range(6):
        # Get the first channel of the face
        face = tensor[i, 0].detach().cpu().numpy()
        
        # Normalize for better visualization
        face_min = face.min()
        face_max = face.max()
        if face_max > face_min:
            face = (face - face_min) / (face_max - face_min)
        
        # Display the face
        axs[i].imshow(face, cmap='viridis')
        axs[i].set_title(f"Face {i} ({face_names[i]})")
        axs[i].axis('off')
    
    plt.suptitle(f"Sample: {sample_id}")
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Load and visualize CubeDiff WebDataset")
    parser.add_argument("--tar_path", required=True, help="Path to tar file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--output_dir", default="visualizations", help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.tar_path}...")
    dataset = CubeDiffDataset(args.tar_path, batch_size=args.batch_size)
    
    # Process a few batches
    sample_count = 0
    batch_count = 0
    
    for batch in dataset:
        batch_count += 1
        
        # Get batch data
        ids = batch["id"]
        latents = batch["latent"]
        captions = batch["caption"]
        
        print(f"\nBatch {batch_count}:")
        print(f"  Batch size: {len(ids)}")
        print(f"  Latent shape: {latents.shape}")
        
        # Process each sample in the batch
        for i in range(len(ids)):
            sample_id = ids[i]
            latent = latents[i]
            caption = captions[i]
            
            print(f"\nSample {sample_count + 1}:")
            print(f"  ID: {sample_id}")
            print(f"  Latent shape: {latent.shape}")
            print(f"  Caption: {caption[:50]}{'...' if len(caption) > 50 else ''}")
            
            # Visualize the latent
            save_path = os.path.join(args.output_dir, f"{sample_id}_cubemap.png")
            try:
                visualize_cubemap(latent, sample_id, save_path)
                print(f"  Visualization saved to {save_path}")
            except Exception as e:
                print(f"  Error visualizing latent: {str(e)}")
            
            sample_count += 1
            if sample_count >= args.num_samples:
                break
        
        if sample_count >= args.num_samples:
            break
    
    print(f"\nProcessed {sample_count} samples from {batch_count} batches")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Step 4: Create the main processing script
cat > process_cubediff_data.sh << 'EOF'
#!/bin/bash
# CubeDiff Data Processing Script

# Set variables
DATA_ROOT="../data/dataspace/polyhaven_tiny"
LAT_ROOT="$DATA_ROOT/latents"
CAPTIONS="$DATA_ROOT/raw/captions.json"
TRAIN_TAR="$DATA_ROOT/cubediff_train.tar"
VAL_TAR="$DATA_ROOT/cubediff_val.tar"
VIS_DIR="./visualizations"
VAL_FRAC=0.071  # 50/700

# Create output directories
mkdir -p "$VIS_DIR"

# Print script header
echo "====================================="
echo "CubeDiff Data Processing Pipeline"
echo "====================================="
echo "Data locations:"
echo "- Latents: $LAT_ROOT"
echo "- Captions: $CAPTIONS"
echo "- Output train tar: $TRAIN_TAR"
echo "- Output val tar: $VAL_TAR"
echo "- Validation fraction: $VAL_FRAC"
echo "====================================="

# Check if required directories and files exist
if [ ! -d "$LAT_ROOT" ]; then
    echo "ERROR: Latents directory doesn't exist: $LAT_ROOT"
    exit 1
fi

if [ ! -f "$CAPTIONS" ]; then
    echo "ERROR: Captions file doesn't exist: $CAPTIONS"
    exit 1
fi

# Step 1: Create tar files
echo -e "\n\n===== STEP 1: Creating WebDataset tar files ====="
python make_train_val_tars.py \
       --lat_root   "$LAT_ROOT" \
       --captions   "$CAPTIONS" \
       --train_tar  "$TRAIN_TAR" \
       --val_tar    "$VAL_TAR" \
       --val_frac   "$VAL_FRAC" \
       --debug

# Check if tar files were created successfully
if [ ! -f "$TRAIN_TAR" ] || [ ! -f "$VAL_TAR" ]; then
    echo "ERROR: Tar files were not created successfully."
    exit 1
fi

# Check tar file sizes
TRAIN_SIZE=$(du -h "$TRAIN_TAR" | cut -f1)
VAL_SIZE=$(du -h "$VAL_TAR" | cut -f1)
echo -e "\nTar file sizes:"
echo "- Training tar: $TRAIN_SIZE"
echo "- Validation tar: $VAL_SIZE"

# Step 2: Simple verification
echo -e "\n\n===== STEP 2: Simple Tar Verification ====="
python verify_simple.py \
       --train_tar "$TRAIN_TAR" \
       --val_tar "$VAL_TAR"

# Step 3: Dataset loading and visualization
echo -e "\n\n===== STEP 3: Dataset Loading and Visualization ====="
python load_webdataset.py \
       --tar_path "$TRAIN_TAR" \
       --batch_size 4 \
       --num_samples 3 \
       --output_dir "$VIS_DIR"

echo -e "\n\n===== VERIFICATION COMPLETE ====="
echo "If all verification steps passed, your CubeDiff data is ready for training!"
echo "Training tar: $TRAIN_TAR ($TRAIN_SIZE)"
echo "Validation tar: $VAL_TAR ($VAL_SIZE)"
echo "Sample visualizations: $VIS_DIR"
EOF

# Make the script executable
chmod +x process_cubediff_data.sh

echo "===== Setup Complete ====="
echo "To process your CubeDiff data, run:"
echo "  ./process_cubediff_data.sh"
echo ""
echo "This script will:"
echo "1. Create the WebDataset tar files with proper formatting"
echo "2. Verify the structure of the tar files"
echo "3. Load the data and create visualizations"
echo ""
echo "The key fix is using standard WebDataset naming conventions"
echo "for the files within the tar, which resolves the AssertionError."