"""
System configuration utilities for CubeDiff training.
Provides functions to optimize system settings for training.
"""

import os
import sys
import platform
import torch
import torch.multiprocessing as mp
import shutil
from pathlib import Path
import subprocess

def configure_system_for_training():
    """
    Configure system settings for optimal CubeDiff training.
    Adjusts shared memory settings, multiprocessing options, and provides
    recommendations based on the available hardware.
    
    Returns:
        dict: Dictionary with recommended settings
    """
    print("Configuring system for CubeDiff training...")
    
    # Check system information
    system_info = {}
    system_info['platform'] = platform.system()
    system_info['python_version'] = platform.python_version()
    system_info['cpu_count'] = os.cpu_count()
    
    # Detect if we're running in a container/VM
    is_container = Path('/.dockerenv').exists() or os.environ.get('KUBERNETES_SERVICE_HOST')
    system_info['is_container'] = is_container
    
    # Check for GPU availability
    if torch.cuda.is_available():
        system_info['cuda_available'] = True
        system_info['cuda_version'] = torch.version.cuda
        system_info['gpu_name'] = torch.cuda.get_device_name(0)
        system_info['gpu_count'] = torch.cuda.device_count()
        system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        system_info['cuda_available'] = False
        system_info['gpu_count'] = 0
    
    print(f"System: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    print(f"CPU cores: {system_info['cpu_count']}")
    
    if system_info['cuda_available']:
        print(f"GPU: {system_info['gpu_name']} x{system_info['gpu_count']}")
        print(f"GPU memory: {system_info['gpu_memory_gb']:.1f} GB")
        print(f"CUDA version: {system_info['cuda_version']}")
    else:
        print("No GPU detected - training will be slow!")
    
    # Check shared memory limit on Linux
    shm_size_gb = None
    if system_info['platform'] == 'Linux':
        try:
            result = subprocess.run(['df', '-h', '/dev/shm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                print("\nShared memory status:")
                print(output)
                
                # Extract shared memory size
                lines = output.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        shm_size = parts[1]
                        print(f"Current shared memory size: {shm_size}")
                        
                        # Extract size in GB
                        if 'G' in shm_size:
                            shm_size_gb = float(shm_size.replace('G', ''))
                        elif 'M' in shm_size:
                            shm_size_gb = float(shm_size.replace('M', '')) / 1024
                        
                        # Check if size is less than 8G
                        if shm_size_gb is not None and shm_size_gb < 8:
                            print("\n⚠️ WARNING: Shared memory size is small!")
                            print("To increase shared memory (requires sudo):")
                            print("  sudo mount -o remount,size=16G /dev/shm")
                
                # Print message about shared memory issues
                print("\nIf you encounter 'bus error' or 'shared memory' errors during training:")
                print("1. Consider increasing shared memory with the command above")
                print("2. Reduce batch_size and num_workers in your config")
                print("3. Use persistent_workers=True in DataLoader")
        except Exception as e:
            print(f"Could not check shared memory: {e}")
    
    # Recommended settings based on system
    recommended_settings = {}
    
    # Calculate recommended batch size based on available GPU memory
    if system_info['cuda_available']:
        gpu_mem = system_info['gpu_memory_gb']
        if gpu_mem > 24:  # High-end GPU (e.g., A100, H100)
            recommended_settings['batch_size'] = 8
            recommended_settings['gradient_accumulation_steps'] = 1
        elif gpu_mem > 16:  # Mid-high GPU (e.g., RTX 3090, 4090)
            recommended_settings['batch_size'] = 4
            recommended_settings['gradient_accumulation_steps'] = 2
        elif gpu_mem > 8:  # Mid-range GPU (e.g., RTX 3080, 3070, L4)
            recommended_settings['batch_size'] = 2
            recommended_settings['gradient_accumulation_steps'] = 4
        else:  # Lower-end GPU
            recommended_settings['batch_size'] = 1
            recommended_settings['gradient_accumulation_steps'] = 8
    else:
        # CPU training
        recommended_settings['batch_size'] = 1
        recommended_settings['gradient_accumulation_steps'] = 8
    
    # Calculate recommended workers based on CPU cores
    cpu_count = system_info['cpu_count']
    if cpu_count > 16:
        recommended_settings['num_workers'] = 4
    elif cpu_count > 8:
        recommended_settings['num_workers'] = 2
    elif cpu_count > 4:
        recommended_settings['num_workers'] = 1
    else:
        recommended_settings['num_workers'] = 0
    
    # Check for GCP Vertex AI with L4 GPU
    is_l4_gpu = False
    if system_info['cuda_available']:
        gpu_name = system_info['gpu_name'].lower()
        is_l4_gpu = 'l4' in gpu_name
        
        if is_l4_gpu:
            print("\nDetected NVIDIA L4 GPU - optimizing settings specifically for this hardware")
    
    # Optimized settings for L4 GPU
    if is_l4_gpu:
        l4_settings = {
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'mixed_precision': 'fp16',
        }
        
        # Set workers based on shared memory
        if shm_size_gb is not None:
            # For L4 with 16 vCPUs, optimal workers depends on shared memory
            if shm_size_gb >= 16:
                l4_settings['num_workers'] = 4  # Good with plenty of shared memory
            elif shm_size_gb >= 8:
                l4_settings['num_workers'] = 2  # Safe with moderate shared memory
            else:
                l4_settings['num_workers'] = 1  # Conservative with limited shared memory
        else:
            # If can't determine shared memory, use safe default
            l4_settings['num_workers'] = 2
        
        # Add L4-optimized settings to recommendations
        recommended_settings['l4_gpu'] = l4_settings
        
        print("\nOptimized settings for NVIDIA L4 GPU on GCP Vertex AI:")
        for key, value in l4_settings.items():
            print(f"  {key}: {value}")
        
        if shm_size_gb is not None:
            print(f"  (Shared memory available: {shm_size_gb:.1f} GB)")
            if shm_size_gb < 8:
                print("  ⚠️ Warning: Limited shared memory detected. Consider increasing with:")
                print("    sudo mount -o remount,size=16G /dev/shm")
    
    # Update mini-training settings based on L4 GPU detection
    recommended_settings['mini_training'] = {
        'batch_size': 2 if is_l4_gpu else 1,
        'num_workers': 2 if (is_l4_gpu and (shm_size_gb is None or shm_size_gb >= 8)) else 1,
        'gradient_accumulation_steps': 4, 
        'mixed_precision': 'fp16'
    }
    
    # Add progressive loading settings that start safe and gradually increase
    recommended_settings['progressive_loading'] = {
        'initial_batch_size': 1,
        'initial_num_workers': 0,
        'target_batch_size': 2 if is_l4_gpu else 1,
        'target_num_workers': 2 if is_l4_gpu else 1,
        'gradient_accumulation_steps': 4,
        'mixed_precision': 'fp16'
    }
    
    print("\nRecommended settings for full training:")
    for key, value in recommended_settings.items():
        if key not in ['mini_training', 'progressive_loading', 'l4_gpu']:
            print(f"  {key}: {value}")
    
    print("\nRecommended settings for mini-training:")
    for key, value in recommended_settings['mini_training'].items():
        print(f"  {key}: {value}")
    
    print("\nProgressive loading settings (starts safe, gradually increases):")
    for key, value in recommended_settings['progressive_loading'].items():
        print(f"  {key}: {value}")
    
    # Configure PyTorch settings
    # Set sharing strategy to file_system for more reliable sharing
    mp.set_sharing_strategy('file_system')
    
    # Try to set start method to spawn which is more stable
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass
    
    print(f"\nPyTorch multiprocessing:")
    print(f"  sharing strategy: {mp.get_sharing_strategy()}")
    print(f"  start method: {mp.get_start_method()}")
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nCUDA cache cleared")
    
    return recommended_settings

# Add this to system_config.py
def adaptive_shm_handling(config):
    """
    Adaptively adjust training parameters based on available shared memory
    without requiring manual intervention.
    """
    import subprocess
    import os
    
    # Default safe values when shared memory is limited
    safe_num_workers = 0
    safe_batch_size = 1
    
    try:
        # Check shared memory size on Linux
        result = subprocess.run(['df', '-h', '/dev/shm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            output = result.stdout.decode('utf-8')
            
            # Extract shared memory size
            lines = output.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    shm_size = parts[1]
                    shm_size_gb = None
                    
                    # Convert to GB
                    if 'G' in shm_size:
                        shm_size_gb = float(shm_size.replace('G', ''))
                    elif 'M' in shm_size:
                        shm_size_gb = float(shm_size.replace('M', '')) / 1024
                    
                    print(f"Available shared memory: {shm_size_gb:.2f} GB")
                    
                    # Adjust parameters based on available shared memory
                    if shm_size_gb is not None:
                        if shm_size_gb >= 16:
                            # Plenty of shared memory
                            safe_num_workers = min(4, os.cpu_count() // 2)
                            safe_batch_size = config.batch_size  # Use original batch size
                        elif shm_size_gb >= 8:
                            # Moderate shared memory
                            safe_num_workers = min(2, os.cpu_count() // 4)
                            safe_batch_size = config.batch_size  # Use original batch size
                        elif shm_size_gb >= 2:
                            # Limited shared memory
                            safe_num_workers = 1
                            safe_batch_size = min(config.batch_size, 2)
                        else:
                            # Very limited shared memory
                            safe_num_workers = 0
                            safe_batch_size = 1
                            
                        print(f"Adjusting training parameters based on shared memory:")
                        print(f"  - Original num_workers: {config.num_workers}")
                        print(f"  - Adjusted num_workers: {safe_num_workers}")
                        print(f"  - Original batch_size: {config.batch_size}")
                        print(f"  - Adjusted batch_size: {safe_batch_size}")
                        
                        # Update config with safe values
                        config.num_workers = safe_num_workers
                        config.batch_size = safe_batch_size
                        
                        # Adjust gradient accumulation to maintain effective batch size
                        original_effective_batch = config.batch_size * config.gradient_accumulation_steps
                        if safe_batch_size < config.batch_size:
                            # Increase gradient accumulation to compensate for smaller batch
                            new_grad_accum = max(1, original_effective_batch // safe_batch_size)
                            config.gradient_accumulation_steps = new_grad_accum
                            print(f"  - Adjusted gradient_accumulation_steps: {new_grad_accum}")
    
    except Exception as e:
        print(f"Error checking shared memory: {e}")
        print("Using safe default values")
        
        # Fall back to ultra-safe values
        config.num_workers = safe_num_workers
        config.batch_size = safe_batch_size
        
        # Increase gradient accumulation if we reduced batch size
        if hasattr(config, 'batch_size') and safe_batch_size < config.batch_size:
            config.gradient_accumulation_steps = max(4, config.gradient_accumulation_steps * 2)
    
    # Activate warning and auto-recovery features
    config.enable_memory_monitoring = True
    config.progressive_loading = True
    
    return config