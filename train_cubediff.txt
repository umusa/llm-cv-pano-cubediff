"""
Training script for CubeDiff model.
Supports single-GPU and multi-GPU training with distributed data parallel.
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import DDPMScheduler

from cubediff_models import load_sd_components, convert_to_inflated_attention
from cubediff_dataset import create_dataloader
from cubediff_trainer import CubeDiffTrainer
from cubediff_optimization import optimize_for_vertex_ai, enable_memory_efficient_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Train CubeDiff model")
    
    # Model parameters
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Model ID for pretrained Stable Diffusion")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to panorama images")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to file containing prompts (one per line)")
    parser.add_argument("--face_size", type=int, default=512,
                        help="Size of each cubemap face")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--mixed_precision", type=bool, default=True,
                        help="Whether to use mixed precision training")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Resource parameters
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Apply memory optimization techniques")
    
    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    return parser.parse_args()


def setup_distributed_training(local_rank):
    """
    Setup for distributed training.
    
    Args:
        local_rank: Local process rank
        
    Returns:
        device, is_master, world_size
    """
    if local_rank != -1:
        # Initialize process group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_master = rank == 0
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single GPU training
        world_size = 1
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device, is_master, world_size


def main():
    args = parse_args()
    
    # Setup distributed training
    device, is_master, world_size = setup_distributed_training(args.local_rank)
    
    # Only log on master process
    if is_master:
        print(f"Starting training with {world_size} GPUs")
        print(f"Training parameters: {args}")
    
    # Optimize settings for Vertex AI if requested
    if args.optimize_memory:
        settings = optimize_for_vertex_ai(num_gpus=world_size)
        if is_master:
            print("Optimized settings for Vertex AI:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        # Update args with optimized settings
        args.batch_size = settings["batch_size_per_gpu"]
        args.save_every = settings["save_every"]
        args.learning_rate = settings["learning_rate"]
        args.num_workers = settings["num_workers"]
        args.mixed_precision = settings["mixed_precision"]
        args.grad_accum_steps = settings["gradient_accumulation_steps"]
    
    # Load model components
    vae, text_encoder, tokenizer, unet = load_sd_components(
        model_id=args.model_id,
        use_sync_gn=True
    )
    
    # Convert UNet to use inflated attention
    unet = convert_to_inflated_attention(unet)
    
    # Apply memory optimizations if requested
    if args.optimize_memory:
        unet = enable_memory_efficient_attention(unet)
        unet.enable_gradient_checkpointing()
    
    # Move models to device
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        num_train_timesteps=1000,
        prediction_type="epsilon"
    )
    
    # Create dataloader
    dataloader, dataset = create_dataloader(
        dataset_path=args.dataset_path,
        prompts_file=args.prompts_file,
        face_size=args.face_size,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=args.local_rank if args.local_rank != -1 else 0,
        num_workers=args.num_workers,
        distributed=args.local_rank != -1
    )
    
    # Create trainer
    trainer = CubeDiffTrainer(
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        device=device
    )
    
    # Wrap UNet with DDP for distributed training
    if args.local_rank != -1:
        trainer.unet = DDP(
            trainer.unet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Train model
    if is_master:
        print(f"Starting training for {args.num_epochs} epochs")
    
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        save_every=args.save_every
    )
    
    if is_master:
        print("Training complete!")


if __name__ == "__main__":
    main()