"""
Optimization utilities for CubeDiff implementation.
Specifically for running on GCP Vertex AI with L4 GPUs.
"""

import torch


def optimize_for_vertex_ai(num_gpus=4, memory_per_gpu=22):
    """
    Optimize settings for GCP Vertex AI workbench with NVIDIA L4 GPUs.
    
    Args:
        num_gpus: Number of available GPUs
        memory_per_gpu: Memory per GPU in GB
        
    Returns:
        Dictionary of optimized settings
    """
    # Calculate optimal batch size
    # For L4 GPUs with 22GB memory, a batch size of 1 per GPU is safe
    # Larger batch sizes may be possible with gradient accumulation
    batch_size_per_gpu = 1
    total_batch_size = batch_size_per_gpu * num_gpus
    
    # Calculate gradient accumulation steps to simulate larger batch
    gradient_accumulation_steps = 4  # Simulate batch size of 4
    
    # Estimate maximum training steps for 10-hour constraint
    # Assuming approximately 4 seconds per step with batch size 1
    steps_per_hour = 3600 / 4  # ~900 steps per hour
    max_steps = int(steps_per_hour * 10)  # ~9000 steps for 10 hours
    
    # Calculate learning rate based on batch size
    # Rule of thumb: lr = base_lr * sqrt(batch_size)
    base_lr = 1e-5
    effective_batch_size = total_batch_size * gradient_accumulation_steps
    learning_rate = base_lr * (effective_batch_size ** 0.5)
    
    # Enable mixed precision for efficiency
    mixed_precision = True
    
    # Set optimal number of dataloader workers
    # Rule of thumb: 4 workers per GPU
    num_workers = 4
    
    # Calculate checkpoint frequency
    # Save approximately every 30 minutes
    save_every = int(steps_per_hour / 2)
    
    return {
        "batch_size_per_gpu": batch_size_per_gpu,
        "total_batch_size": total_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "mixed_precision": mixed_precision,
        "num_workers": num_workers,
        "save_every": save_every
    }


def profile_memory_usage(model, sample_batch, 
                         batch_size=1, height=512, width=512,
                         device="cuda"):
    """
    Profile memory usage of the model for a given batch size and resolution.
    
    Args:
        model: Model to profile (UNet)
        sample_batch: Sample batch to use for profiling
        batch_size: Batch size to test
        height: Height of images
        width: Width of images
        device: Device to use for profiling
        
    Returns:
        Dictionary of memory usage statistics
    """
    if device != "cuda":
        print("Memory profiling requires CUDA. Skipping.")
        return {}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Record initial memory
    initial_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    with torch.no_grad():
        _ = model(**sample_batch)
    
    # Record peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Memory usage
    memory_usage = peak_memory - initial_memory
    
    # Calculate memory per batch item
    memory_per_item = memory_usage / batch_size
    
    # Estimate max batch size
    available_memory = torch.cuda.get_device_properties(0).total_memory * 0.8  # 80% of total
    used_memory = torch.cuda.memory_allocated()
    free_memory = available_memory - used_memory
    
    estimated_max_batch_size = int(free_memory // memory_per_item)
    
    # Reset
    torch.cuda.empty_cache()
    
    return {
        "initial_memory_gb": initial_memory / 1e9,
        "peak_memory_gb": peak_memory / 1e9,
        "memory_usage_gb": memory_usage / 1e9,
        "memory_per_item_gb": memory_per_item / 1e9,
        "estimated_max_batch_size": max(1, estimated_max_batch_size)
    }


def enable_memory_efficient_attention(unet):
    """
    Enable memory-efficient attention in the UNet.
    This reduces memory usage during training and inference.
    
    Args:
        unet: UNet model to modify
        
    Returns:
        Modified UNet with memory-efficient attention
    """
    # Try to import xformers
    try:
        import xformers
        import xformers.ops
        print("xformers is available. Enabling memory efficient attention.")
        
        # Enable memory efficient attention
        unet.enable_xformers_memory_efficient_attention()
        return unet
    except ImportError:
        print("xformers is not available. Falling back to standard attention.")
        
        # Enable native memory efficient attention if available
        if hasattr(unet, "enable_slicing"):
            unet.enable_slicing()
            print("Enabled UNet slicing for memory efficiency.")
        
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing for memory efficiency.")
        
        return unet


def calculate_training_cost_estimate(num_gpus=4, training_hours=10, gpu_type="L4"):
    """
    Calculate an estimated cost for training on GCP Vertex AI.
    
    Args:
        num_gpus: Number of GPUs
        training_hours: Number of training hours
        gpu_type: Type of GPU (L4, A100, etc.)
        
    Returns:
        Dictionary with cost estimates
    """
    # Approximate costs per hour for different GPU types (USD)
    gpu_costs = {
        "L4": 1.52,     # L4 price per hour
        "T4": 0.95,     # T4 price per hour
        "A100": 4.28,   # A100-40GB price per hour
        "V100": 2.48    # V100 price per hour
    }
    
    if gpu_type not in gpu_costs:
        print(f"Unknown GPU type: {gpu_type}. Using L4 pricing.")
        gpu_type = "L4"
    
    # Calculate total GPU cost
    gpu_cost_per_hour = gpu_costs[gpu_type]
    total_gpu_cost = gpu_cost_per_hour * num_gpus * training_hours
    
    # Estimate cost of other resources (CPU, memory, etc.)
    additional_cost = 0.2 * total_gpu_cost  # Approximately 20% of GPU cost
    
    # Total cost
    total_cost = total_gpu_cost + additional_cost
    
    return {
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "training_hours": training_hours,
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "total_gpu_cost": total_gpu_cost,
        "additional_cost": additional_cost,
        "total_cost": total_cost
    }
