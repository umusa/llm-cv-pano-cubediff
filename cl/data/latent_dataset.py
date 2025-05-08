# -------- the data loader to load the latent tensors and captions from a webdataset ---------
import torch, io
import numpy as np
import io
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import webdataset as wds
import webdataset.compat as wds_compat

# disable empty‐shard checking globally
wds_compat.check_empty = lambda *args, **kwargs: None

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_LEN   = tokenizer.model_max_length

# def get_dataloader(wds_path, batch_size, num_workers=4):


def get_dataloader(
    wds_path,
    batch_size,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
):
    print(f"latent_webdataset.py - get_dataloader - wds_path is {wds_path}, batch_size is {batch_size}, num_workers is {num_workers}")
    
    def preprocess(sample):
        try:
            # Check for None sample early
            if sample is None:
                print("Warning: Received None sample in preprocess")
                return None
                
            # Find tensor and caption keys
            pt_keys = [k for k in sample if k.endswith(".pt")]
            if not pt_keys:
                print(f"No .pt file found in sample with keys {list(sample.keys())}")
                return None
            pt_key = pt_keys[0]
            
            txt_keys = [k for k in sample if k.endswith(".txt")]
            if not txt_keys:
                print(f"No .txt file found in sample with keys {list(sample.keys())}")
                return None
            
            # Load tensor safely
            try:
                bytes_ = sample[pt_key]
                tensor = torch.load(io.BytesIO(bytes_))
                
                # Validate tensor shape
                if isinstance(tensor, torch.Tensor):
                    print(f"Loaded tensor with shape: {tensor.shape}")
                    # Your tensor should be [6, 4, 64, 64] 
                    if tensor.shape != (6, 4, 64, 64):
                        print(f"Warning: Unexpected tensor shape {tensor.shape}, expected (6, 4, 64, 64)")
                else:
                    print(f"Error: Loaded object is not a tensor, got {type(tensor)}")
                    return None
            except Exception as e:
                print(f"Error loading tensor from {pt_key}: {e}")
                return None
                
            # Load and tokenize caption
            try:
                caption = sample[txt_key].decode("utf-8")
                toks = tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt"
                )
            except Exception as e:
                print(f"Error tokenizing caption from {txt_key}: {e}")
                return None
                
            return {
                "latent": tensor,
                "input_ids": toks.input_ids.squeeze(0),
                "attention_mask": toks.attention_mask.squeeze(0),
            }
        except Exception as e:
            print(f"Error in preprocess: {e}")
            return None

    # Create a simpler pipeline that's more robust to errors
    print("Creating WebDataset pipeline...")
    
    # Create dataset with error handling
    try:
        ds = wds.WebDataset(wds_path, handler=wds.warn_and_continue)
        print(f"Created WebDataset with path: {wds_path}")
        
        # Apply preprocessing with better error handling
        ds = ds.map(preprocess, handler=wds.warn_and_continue)
        print("Added preprocess to pipeline")
        
        # Filter out None values
        ds = ds.select(lambda s: s is not None)
        print("Added select filter to pipeline")
        
        # Set a reasonable batch size for debugging if needed
        dataset = ds
        print("Dataset pipeline setup complete")
    except Exception as e:
        print(f"Error setting up WebDataset pipeline: {e}")
        raise
        
    def collate_fn(batch):
        print("collate_fn called with batch size:", len(batch))
        
        # Handle empty batches
        if len(batch) == 0:
            print("Warning: Empty batch encountered")
            # Return empty tensors instead of None
            return {
                "latent": torch.zeros((0, 6, 4, 64, 64)),
                "input_ids": torch.zeros((0, MAX_LEN), dtype=torch.long),
                "attention_mask": torch.zeros((0, MAX_LEN), dtype=torch.long)
            }
        
        try:
            # Stack tensors
            latents = torch.stack([b["latent"] for b in batch], dim=0)
            input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
            attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
            
            result = {
                "latent": latents,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            print(f"Collated batch with shapes: {[(k, v.shape) for k, v in result.items()]}")
            return result
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            if batch:
                print(f"First batch item keys: {list(batch[0].keys())}")
                for k, v in batch[0].items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: type={type(v)}")
            # Return empty tensors instead of None
            return {
                "latent": torch.zeros((0, 6, 4, 64, 64)),
                "input_ids": torch.zeros((0, MAX_LEN), dtype=torch.long),
                "attention_mask": torch.zeros((0, MAX_LEN), dtype=torch.long)
            }
    
    print("Creating DataLoader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # sampling order controlled upstream
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
        print("DataLoader created successfully")
        return loader
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        raise


