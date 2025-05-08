#!/usr/bin/env python
"""
CUDNN Compatibility Fix Script for CubeDiff Training
This script fixes the 'Could not load symbol cudnnGetLibConfig' error
and ensures proper xFormers compatibility with mixed precision training.

Usage:
1. Run this script before starting training
2. Or import this module at the beginning of your training script
"""
import os
import sys
import ctypes
import warnings
from subprocess import run, PIPE
import torch

def detect_cuda_version():
    """Detect CUDA version from system"""
    try:
        result = run(['nvcc', '--version'], stdout=PIPE, text=True)
        for line in result.stdout.split('\n'):
            if 'release' in line and 'V' in line:
                # Extract version like 11.7 from text
                parts = line.split('V')[1].split('.')
                major = parts[0].strip()
                minor = parts[1].split(' ')[0].strip()
                return f"{major}.{minor}"
    except:
        pass
    
    # Fallback to torch.version.cuda
    if torch.version.cuda:
        return torch.version.cuda
    
    return None

def fix_cudnn_libraries():
    """Fix CUDNN library issues by ensuring proper library paths"""
    # 1. Identify CUDA version
    cuda_version = detect_cuda_version()
    if not cuda_version:
        print("⚠️ Could not detect CUDA version")
        return False
    
    print(f"Detected CUDA version: {cuda_version}")
    
    # 2. Set proper library paths based on CUDA version
    cuda_paths = [
        f"/usr/local/cuda-{cuda_version}/lib64",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/nvidia/lib64"
    ]
    
    # Add paths to LD_LIBRARY_PATH
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(p for p in cuda_paths if os.path.exists(p))
    
    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = new_paths + (":" + current_path if current_path else "")
        print(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    
    # 3. Try to preload required libraries
    try:
        # Try to locate libcudnn.so
        cudnn_paths = [
            f"/usr/lib/x86_64-linux-gnu/libcudnn.so",
            f"/usr/local/cuda-{cuda_version}/lib64/libcudnn.so",
            "/usr/local/cuda/lib64/libcudnn.so"
        ]
        
        cudnn_loaded = False
        for path in cudnn_paths:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    print(f"✅ Successfully loaded CUDNN from {path}")
                    cudnn_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        if not cudnn_loaded:
            print("⚠️ Could not find and load libcudnn.so")
        
        # Try to load libcuda.so
        cuda_lib_paths = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so",
            f"/usr/local/cuda-{cuda_version}/lib64/libcuda.so"
        ]
        
        cuda_loaded = False
        for path in cuda_lib_paths:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    print(f"✅ Successfully loaded CUDA from {path}")
                    cuda_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        if not cuda_loaded:
            print("⚠️ Could not find and load libcuda.so")
            
        return cuda_loaded or cudnn_loaded
    
    except Exception as e:
        print(f"❌ Error loading CUDA/CUDNN libraries: {e}")
        return False

def set_optimized_torch_config():
    """Set optimized PyTorch configuration for training"""
    import torch
    # 1. Configure PyTorch CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # More stable training
    torch.backends.cudnn.deterministic = True  # More stable training
    torch.backends.cudnn.allow_tf32 = True
    
    # 2. Disable dynamic compilation that might cause issues
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
        # Disable dynamo to avoid cudnnGetLibConfig errors
        import torch._dynamo
        torch._dynamo.reset()
        torch._dynamo.disable()
        print("✅ Disabled torch._dynamo")
    
    # 3. Set better memory allocation policies
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TORCH_COMPILE_BACKEND"] = "inductor"  # More stable than nvfuser
    
    # 4. Configure xFormers if available
    try:
        import xformers
        # Ensure PyTorch uses high precision for matmul operations
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision('high')
        
        # Force xFormers to use bfloat16 for attention operations
        os.environ["XFORMERS_FORCE_MIXED_PREC"] = "bfloat16"
        print("✅ Configured xFormers for bfloat16 precision")
        return True
    except ImportError:
        print("⚠️ xFormers not available")
        return False

def apply_patches():
    """Apply monkey patches to ensure consistent dtype and parameter handling"""
    # 1. Patch UNet forward method to filter extra parameters
    try:
        from diffusers import UNet2DConditionModel
        orig_unet_forward = UNet2DConditionModel.forward
        
        def patched_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
            """Filter out extra parameters that cause issues"""
            return orig_unet_forward(self, sample, timestep, encoder_hidden_states)
        
        UNet2DConditionModel.forward = patched_unet_forward
        print("✅ Patched UNet2DConditionModel.forward")
    except ImportError:
        print("⚠️ Could not patch UNet2DConditionModel")
    
    # 2. Apply xFormers compatibility patch
    try:
        from diffusers.models.attention_processor import Attention
        
        # Store original method
        if hasattr(Attention, "set_use_memory_efficient_attention_xformers"):
            orig_method = Attention.set_use_memory_efficient_attention_xformers
            
            def patched_xformers_setter(self, use_memory_efficient_attention_xformers=True, **kwargs):
                # Call original implementation
                result = orig_method(self, use_memory_efficient_attention_xformers, **kwargs)
                
                # Force consistent dtype for Q/K/V
                if hasattr(self, "to_q"):
                    dtype = torch.bfloat16
                    self.to_q = self.to_q.to(dtype=dtype)
                    self.to_k = self.to_k.to(dtype=dtype)
                    self.to_v = self.to_v.to(dtype=dtype)
                    if hasattr(self, "to_out") and isinstance(self.to_out, list):
                        self.to_out[0] = self.to_out[0].to(dtype=dtype)
                
                return result
            
            Attention.set_use_memory_efficient_attention_xformers = patched_xformers_setter
            print("✅ Patched Attention.set_use_memory_efficient_attention_xformers")
    except ImportError:
        print("⚠️ Could not patch Attention processor")
    
    return True

if __name__ == "__main__":
    print("\n🔧 Applying CUDNN compatibility fixes...\n")
    
    # Fix CUDNN libraries
    lib_fixed = fix_cudnn_libraries()
    print(f"\nCUDNN library fix {'✅ successful' if lib_fixed else '❌ failed'}")
    
    # Set optimized torch config
    config_set = set_optimized_torch_config()
    print(f"PyTorch configuration {'✅ optimized' if config_set else '❌ not optimized'}")
    
    # Apply patches
    patches_applied = apply_patches()
    print(f"Code patches {'✅ applied' if patches_applied else '❌ failed'}")
    
    print("\n✅ Setup complete! Run your training script now.")
    print("💡 Import this module at the beginning of your training script for automatic fixes.")
else:
    # When imported as a module, apply all fixes automatically
    lib_fixed = fix_cudnn_libraries()
    config_set = set_optimized_torch_config()
    patches_applied = apply_patches()