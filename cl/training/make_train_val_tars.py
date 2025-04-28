#!/usr/bin/env python
"""
Create WebDataset train / val tars from CubeDiff latents.

It now supports BOTH
  • captions.json   -> {"id": "caption", ...}
  • captions.jsonl  -> {"id": "...", "caption": "..."} per line
"""

import argparse, glob, json, os, random, re, sys
import numpy as np, torch, webdataset as wds
import io

# ---------- util helpers ----------------------------------------------------
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

def encode_tensor(tensor):
    """
    Encode a PyTorch tensor to bytes for WebDataset storage.
    This solves the "lat.pt doesn't map to a bytes after encoding" error.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

# ---------- main ------------------------------------------------------------
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

    # Create tar writers
    writers = {
        "train": wds.TarWriter(args.train_tar),  # Don't use encoder=False
        "val":   wds.TarWriter(args.val_tar),    # Don't use encoder=False
    }

    def write(ids_subset, writer, split_name):
        total = len(ids_subset)
        processed = 0
        missing = 0
        
        for pid in ids_subset:
            # Try different possible filename patterns
            possible_files = [
                os.path.join(args.lat_root, f"{pid}.pt"),
                os.path.join(args.lat_root, f"{pid}.npy"),
                os.path.join(args.lat_root, f"{pid}_lat.pt"),
                os.path.join(args.lat_root, f"{pid}_lat.npy")
            ]
            
            found = False
            for f in possible_files:
                if os.path.exists(f):
                    found = True
                    try:
                        # Load the latent
                        latent_data = load_latent(f)
                        
                        # Manually encode the tensor to bytes
                        if isinstance(latent_data, torch.Tensor):
                            latent_bytes = encode_tensor(latent_data)
                        elif isinstance(latent_data, np.ndarray):
                            # Convert numpy array to torch tensor first
                            latent_tensor = torch.from_numpy(latent_data)
                            latent_bytes = encode_tensor(latent_tensor)
                        else:
                            raise ValueError(f"Unsupported latent type: {type(latent_data)}")
                        
                        # Write to tar with proper encoding
                        writer.write({
                            "__key__": pid,
                            "lat.pt": latent_bytes,  # Use encoded bytes
                            "txt": caps.get(pid, "")
                        })
                        
                        processed += 1
                        if args.debug and processed % 50 == 0:
                            print(f"Processed {processed}/{total} in {split_name}")
                    except Exception as e:
                        print(f"Error processing {pid}: {str(e)}")
                        continue
                    break
                    
            if not found:
                if args.debug:
                    print(f"⚠ latent for {pid} not found, tried: {possible_files}")
                missing += 1
        
        if missing:
            print(f"{missing} IDs had no latent file in {split_name}")
        
        print(f"Successfully processed {processed} files for {split_name}")
        
        # Return size of processed data for verification
        return processed

    print(f"Processing train set...")
    train_count = write(train_ids, writers["train"], "train")
    
    print(f"Processing validation set...")
    val_count = write(val_ids, writers["val"], "val")
    
    for w in writers.values(): 
        w.close()

    # Check file sizes for verification
    if os.path.exists(args.train_tar) and os.path.exists(args.val_tar):
        train_size = os.path.getsize(args.train_tar) / (1024*1024)  # Size in MB
        val_size = os.path.getsize(args.val_tar) / (1024*1024)  # Size in MB
        
        print(f"✔  wrote  {train_count} train  +  {val_count} val panoramas")
        print(f"→ {args.train_tar} ({train_size:.2f} MB)  /  {args.val_tar} ({val_size:.2f} MB)")
    else:
        print("⚠ Warning: One or both tar files were not created successfully.")

if __name__ == "__main__":
    main()