#!/usr/bin/env python
"""
CLI entry point that Accelerate will execute on every GPU rank.
It simply:
  • loads the YAML config,
  • instantiates CubeDiffTrainer,
  • calls trainer.train().
"""
import fix_attention_dtype
# Import and apply fixes first, before ANY other imports
# import fix_pytorch_issues 

# import fix_xformers_compat
# import fix_xformers_arguments
# import fix_disable_dyanmo_patch_unet
# # Apply fixes explicitly
# fix_disable_dyanmo_patch_unet.disable_dynamo_thoroughly()
# fix_disable_dyanmo_patch_unet.patch_unet_directly()

# # Add this import near the top of your script
# from fix_unet_xformers_verify import verify_patching

# # Call this function before starting training
# patch_status = verify_patching()

# # Check the results
# if all(patch_status.values()):
#     print("All patches are working correctly!")
# else:
#     print("Warning: Some patches are not applied correctly")

# =======================================================================
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


import ctypes
# adjust to your actual path if needed
ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so", mode=ctypes.RTLD_GLOBAL)

import os
# Update LD_LIBRARY_PATH to include where libcuda.so actually is
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
# Set torch compile backend
os.environ["TORCH_COMPILE_BACKEND"] = "inductor"


import argparse
import yaml
import pathlib
import inspect  # Added this import to fix the NameError
import traceback
import sys

# Add enhanced debugging for PyTorch modules
import torch
# Most modern NVIDIA GPUs (A100, L4, etc.) will run matmuls ~1.5× faster in TF32 with negligible accuracy loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import webdataset as wds

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


import argparse, yaml, pathlib
from cl.training.trainer import CubeDiffTrainer           # <- the class you already have
# Add this to train_cubediff.py after importing peft_patch
from patch_verification import verify_patching

# Run the verification
patching_status = verify_patching()

# Check the results
if all(patching_status.values()):
    print(f"All patching is working correctly - patching_status is {patching_status}, starting training...")
else:
    print("Warning: Some classes are not patched - training may encounter errors")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True,
                    help="Path to tiny_lora.yaml (or any YAML config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.cfg).read_text())
    print(f"train_cubediff.py - cfg is {cfg}")
    trainer = CubeDiffTrainer(
                config  = cfg,
                output_dir = cfg.get("output_dir", "outputs/cubediff_run"),
                mixed_precision = "bf16",
                gradient_accumulation_steps = cfg.get("gradient_accum_steps", 1))
    trainer.train()                       # ← generates samples & checkpoints

if __name__ == "__main__":
    main()

