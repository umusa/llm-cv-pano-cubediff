#!/usr/bin/env python
"""
CLI entry point that Accelerate will execute on every GPU rank.
It simply:
  • loads the YAML config,
  • instantiates CubeDiffTrainer,
  • calls trainer.train().
"""

import argparse
import yaml
import pathlib
import inspect  # Added this import to fix the NameError
import traceback
import os
import sys

# Add enhanced debugging for PyTorch modules
import torch

# original_call_impl = torch.nn.Module._call_impl

# def debug_call_impl(self, *input, **kwargs):
#     try:
#         return original_call_impl(self, *input, **kwargs)
#     except TypeError as e:
#         if 'unexpected keyword argument' in str(e):
#             # Get the class name and module path
#             class_name = self.__class__.__name__
#             module_path = self.__class__.__module__
            
#             # Print detailed error info
#             print(f"\n{'='*80}")
#             print(f"TypeError in {module_path}.{class_name}.forward: {e}")
#             print(f"Class defined in: {inspect.getfile(self.__class__)}")
            
#             # Get forward method signature
#             if hasattr(self, 'forward'):
#                 try:
#                     sig = inspect.signature(self.forward)
#                     print(f"Forward method signature: {sig}")
                    
#                     # Check which params are valid for this forward method
#                     valid_params = list(sig.parameters.keys())
#                     extra_kwargs = [k for k in kwargs if k not in valid_params]
                    
#                     print(f"Valid parameters: {valid_params}")
#                     print(f"Extra kwargs causing problems: {extra_kwargs}")
#                 except Exception as sig_err:
#                     print(f"Could not get signature: {sig_err}")
            
#             # Print stack trace with file paths and line numbers
#             print(f"\nDetailed stack trace:")
#             tb = traceback.extract_stack()
#             for frame in tb[:-1]:  # Skip the current frame
#                 filename = os.path.basename(frame.filename)
#                 line = frame.lineno
#                 name = frame.name
#                 print(f"  File '{frame.filename}', line {line}, in {name}")
            
#             # Print kwargs details
#             print(f"\nAll kwargs received ({len(kwargs)}):")
#             for k, v in kwargs.items():
#                 v_type = type(v).__name__
#                 v_shape = getattr(v, 'shape', None) if hasattr(v, 'shape') else None
#                 print(f"  - {k}: {v_type}{' shape=' + str(v_shape) if v_shape else ''}")
            
#             print(f"{'='*80}\n")
            
#             # Now continue with the original error
#             raise
#         else:
#             raise
# torch.nn.Module._call_impl = debug_call_impl

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

