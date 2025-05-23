#!/usr/bin/env python
"""
CLI entry point that Accelerate will execute on every GPU rank.
It simply:
  • loads the YAML config,
  • instantiates CubeDiffTrainer,
  • calls trainer.train().
"""
# --- Patch HF bug: rename Replicate() → ReplicateParallel() in tensor_parallel.py ---
import os, re, transformers

tp_path = os.path.join(
    os.path.dirname(transformers.__file__),
    "integrations",
    "tensor_parallel.py",
)

# Read the file
with open(tp_path, "r", encoding="utf-8") as f:
    src = f.read()

# Replace any occurrence of Replicate(   ) with ReplicateParallel()
patched = re.sub(r"Replicate\s*\(\s*\)", "ReplicateParallel()", src)

# Write back only if we made a change
if patched != src:
    with open(tp_path, "w", encoding="utf-8") as f:
        f.write(patched)
# ------------------------------------------------------------------------------
import torch
import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")   # <— avoid /dev/shm exhaustion on many workers

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

# Add enhanced debugging for PyTorch modules
import torch
# Most modern NVIDIA GPUs (A100, L4, etc.) will run matmuls ~1.5× faster in TF32 with negligible accuracy loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    print(f"train_cubediff.py - trainer is {trainer}")
    trainer.train()                       # ← generates samples & checkpoints

if __name__ == "__main__":
    main()

