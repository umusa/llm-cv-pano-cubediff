import os
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import time
from pathlib import Path
import psutil
import threading
import types

from cl.model.architecture import CubeDiffModel
from cl.data.dataset import CubemapDataset
from cl.training.lora import add_lora_to_model

class CubeDiffTrainer:
    """
    Trainer for CubeDiff model.
    """
    def __init__(
        self,
        config,
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        output_dir="./outputs",
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    ):
        self.config = config
        self.pretrained_model_name = pretrained_model_name
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create output directory and subdirectories
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logs directory for wandb offline mode
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create images directory for saving samples
        self.images_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Configure dataloader memory settings
        self.configure_dataloader_memory()
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # Initialize wandb in offline mode if enabled
        self.wandb_run = None
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
            self._init_wandb()
        
        # Set up model and noise scheduler
        self.setup_model()
    
    def _init_wandb(self):
        """Initialize Weights & Biases in offline mode."""
        wandb_dir = os.path.join(self.logs_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        # Format run name with timestamp if not provided
        if not hasattr(self.config, 'wandb_run_name') or not self.config.wandb_run_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"cubediff_{timestamp}"
        else:
            run_name = self.config.wandb_run_name
        
        # Initialize wandb in offline mode
        self.wandb_run = wandb.init(
            dir=wandb_dir,
            project=getattr(self.config, 'wandb_project', 'cubediff'),
            name=run_name,
            config=vars(self.config),
            mode="offline"  # This is the key setting for offline mode
        )
        
        print(f"Wandb initialized in offline mode. Logs will be saved to {wandb_dir}")
        print("To view logs in JupyterLab, use: wandb.jupyter.show()")
    
    def configure_dataloader_memory(self):
        """
        Configure DataLoader to prevent shared memory issues.
        This method adjusts settings to help prevent "out of shared memory" errors.
        """
        # Determine optimal number of workers based on system resources
        suggested_workers = max(1, os.cpu_count() // 2)
        
        # Override the configured number of workers if needed
        if hasattr(self.config, 'num_workers') and self.config.num_workers > suggested_workers:
            print(f"Reducing num_workers from {self.config.num_workers} to {suggested_workers} to prevent memory issues")
            self.config.num_workers = suggested_workers
        elif not hasattr(self.config, 'num_workers') or self.config.num_workers == 0:
            # Set default if not provided
            self.config.num_workers = 2
            print(f"Setting num_workers to {self.config.num_workers}")
        
        # Configure PyTorch dataloader behavior
        # Set sharing strategy to file_system for more reliable sharing
        mp.set_sharing_strategy('file_system')
        
        # Ensure we're not using too many threads overall
        if hasattr(self.config, 'batch_size') and self.config.batch_size > 8:
            print(f"Large batch size detected ({self.config.batch_size}). Consider reducing if memory issues persist.")
        elif not hasattr(self.config, 'batch_size'):
            # Set default if not provided
            self.config.batch_size = 32 # 4
            print(f"Setting batch_size to {self.config.batch_size}")
        
        # Set multiprocessing start method to 'spawn' which is more stable 
        # than the default 'fork' method for complex applications
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # May already be set
            pass
        
        print(f"DataLoader configured with {self.config.num_workers} workers")
        print(f"Sharing strategy: {mp.get_sharing_strategy()}")
        print(f"Multiprocessing start method: {mp.get_start_method()}")
        
    
    def setup_model(self):
        """
        Set up model, tokenizer, and noise scheduler with consistent precision.
        """
        # Initialize diffusion pipeline to get all components
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name,
            torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32,
        )
        
        # Extract components
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        
        # Get the target dtype
        dtype = torch.float16 if self.mixed_precision == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Explicitly convert model components to the correct dtype
        self.vae = self.vae.to(device=device, dtype=dtype)
        self.text_encoder = self.text_encoder.to(device=device, dtype=dtype)
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Set up noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name,
            subfolder="scheduler",
        )
        # align scheduler buffers with chosen dtype
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(dtype))

        # Initialize CubeDiff model (ALWAYS in float32 for training)
        self.model = CubeDiffModel(self.pretrained_model_name)
        
        # IMPORTANT: First initialize parameters in float32 for the optimizer
        self.model = self.model.to(device=device, dtype=torch.float32)
        
        # Add LoRA for efficient fine-tuning (in float32)
        self.lora_params = add_lora_to_model(
            self.model.base_unet,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )
        
        # Force LoRA params to float32 for stable training
        for p in self.lora_params:
            p.data = p.data.to(device=device, dtype=torch.float32)
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"LoRA parameters: {sum(p.numel() for p in self.lora_params):,}")
        
        # Verify all components dtypes
        vae_dtype = next(self.vae.parameters()).dtype
        text_encoder_dtype = next(self.text_encoder.parameters()).dtype
        model_dtype = next(self.model.parameters()).dtype
        
        print(f"VAE dtype: {vae_dtype}")
        print(f"Text encoder dtype: {text_encoder_dtype}")
        print(f"Model dtype: {model_dtype}")

    def fix_device_mismatch(self):
        """
        Update the model components to ensure consistent precision settings.
        Handles both data type and device consistency issues.
        """
        device = self.accelerator.device
        # Keep mixed precision (fp16) for memory optimization
        dtype = torch.float16 if self.mixed_precision == "fp16" else torch.float32
        
        print(f"Moving components to device {device} with dtype {dtype}")
        
        # Move model components to the correct device and precision
        self.vae = self.vae.to(device=device, dtype=dtype)
        self.text_encoder = self.text_encoder.to(device=device, dtype=dtype)
        self.model = self.model.to(device=device, dtype=dtype)
        
        # Fix specific UNet components for consistent dtypes
        for name, module in self.model.base_unet.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias') and module.bias is not None:
                # Ensure both weight and bias use same dtype
                current_dtype = module.weight.dtype
                if module.bias.dtype != current_dtype:
                    print(f"Fixing dtype mismatch in {name}: bias {module.bias.dtype} -> {current_dtype}")
                    module.bias.data = module.bias.data.to(dtype=current_dtype)
        
        print(f"Model components updated for consistent dtypes")
        
        # Verify devices and dtypes
        vae_param = next(self.vae.parameters())
        text_param = next(self.text_encoder.parameters())
        model_param = next(self.model.parameters())
        
        print(f"VAE device: {vae_param.device}, dtype: {vae_param.dtype}")
        print(f"Text encoder device: {text_param.device}, dtype: {text_param.dtype}")
        print(f"Model device: {model_param.device}, dtype: {model_param.dtype}")


    def prepare_dataloaders(self, train_dataset, val_dataset=None):
        """
        Prepare training and validation dataloaders with optimized memory handling.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Dataloaders prepared with Accelerator
        """
        # Make sure we're using the right number of workers based on the configuration
        num_workers = self.config.num_workers
        
        # Configure DataLoader with memory-optimized settings
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,  # Only use pin_memory with workers
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,  # Avoid issues with partial batches
        )
        
        val_dataloader = None
        if val_dataset:
            # Use fewer workers for validation to avoid memory issues
            val_workers = max(0, num_workers // 2)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True if val_workers > 0 else False,
                persistent_workers=True if val_workers > 0 else False,
                prefetch_factor=2 if val_workers > 0 else None,
                drop_last=False,  # We want to evaluate on all validation samples
            )
        
        print(f"DataLoader configured with {num_workers} workers")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Prefetch factor: 2")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        return train_dataloader, val_dataloader

    def train(self, train_dataset, val_dataset=None, num_train_epochs=30000):
        """
        Train the CubeDiff model with improved memory handling.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_train_epochs: Number of training epochs
        """
        print(f"trainer.py train() started ... \n")
        # Memory error recovery mechanism
        max_memory_retries = 3
        memory_retry_count = 0
        
        # Empty CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders(train_dataset, val_dataset)
        
        # Initialize optimizer
        # optimizer = AdamW(
        #     filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.weight_decay,
        #     betas=(0.9, 0.999),
        # )
        
        # In the train method, when initializing the optimizer:
        # optimizer = AdamW(
        #     [p.float() for p in self.model.parameters() if p.requires_grad],  # Ensure float32 for optimizer
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.weight_decay,
        #     betas=(0.9, 0.999),
        # )

        # Initialize optimizer with parameters directly
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # Initialize optimizer with properly collected parameters
        optimizer = AdamW(
            trainable_params,  # Use collected parameters instead of filter
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )


        # Set up scheduler
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_train_epochs,
            eta_min=self.config.min_learning_rate,
        )
        
        # Prepare everything with accelerator
        self.model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, lr_scheduler
        )
        
        if val_dataloader:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        
        # Setup for generating sample images
        if hasattr(self.config, 'sample_prompts') and self.config.sample_prompts:
            sample_prompts = self.config.sample_prompts
        else:
            sample_prompts = [
                "A beautiful mountain lake at sunset with snow-capped peaks",
                "A futuristic cityscape with tall skyscrapers and flying vehicles"
            ]
        
        # Training loop
        global_step = 0
        start_time = datetime.datetime.now()
        
        # Create a local training log file
        log_file = os.path.join(self.logs_dir, "training_log.csv")
        with open(log_file, 'w') as f:
            f.write("step,loss,learning_rate,time_elapsed\n")
        
        # Create a simple matplotlib figure for live loss plotting
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.losses = []
            self.steps = []
        
        # Create a metrics dictionary for tracking
        metrics = {
            'loss': [],
            'step': [],
            'lr': [],
            'time': []
        }
        
        self.model.train()
        progress_bar = tqdm(range(num_train_epochs), desc="Training")
        
        # Main training loop with error handling
        print(f"trainer.py train() training loop started with num_train_epochs as {num_train_epochs}... \n")
        for epoch in range(num_train_epochs):
            print(f"------------- trainer.py - train() - training loop started with epoch {epoch} ---------------- \n")
            try:
                # Add this check at the beginning of each epoch in the train method
                # (inside the for epoch in range(num_train_epochs): loop)
                if hasattr(self, 'adapt_to_memory_pressure') and self.adapt_to_memory_pressure():
                    # If we changed settings, recreate dataloaders
                    train_dataloader, val_dataloader = self.prepare_dataloaders(train_dataset, val_dataset)
                    train_dataloader = self.accelerator.prepare(train_dataloader)
                    if val_dataloader:
                        val_dataloader = self.accelerator.prepare(val_dataloader)
                        
                for step, batch in enumerate(train_dataloader):
                    # Reset memory error flag if we've successfully loaded a batch
                    memory_retry_count = 0
                    
                    with self.accelerator.accumulate(self.model):
                        # Get data from batch
                        faces = batch["faces"]  # [batch, 6, C, H, W]
                        captions = batch["caption"]
                        
                        # Encode text
                        tokens = self.tokenizer(
                            captions, 
                            padding="max_length", 
                            max_length=self.tokenizer.model_max_length, 
                            truncation=True, 
                            return_tensors="pt"
                        ).input_ids.to(self.accelerator.device)
                        
                        # Get text embeddings
                        with torch.no_grad():
                            tokens = tokens.to(self.accelerator.device)
                            text_embeddings = self.text_encoder(tokens)[0]
                            # Ensure text embeddings have the right dtype
                            text_embeddings = self.ensure_dtype(text_embeddings)
                        
                        # Process each face with the safe encoder
                        # Get VAE parameters to match dtype and device
                        vae_param = next(self.vae.parameters())
                        vae_dtype = vae_param.dtype
                        vae_device = vae_param.device
                        
                        latents = []
                        for i in range(faces.shape[1]):
                            # Get single face batch
                            face = faces[:, i]  # [batch, C, H, W]
                            
                            # Ensure face is in the correct format
                            if face.shape[-1] == 3:  # If channels are last
                                face = face.permute(0, 3, 1, 2)  # NHWC -> NCHW
                            
                            # CRITICAL: Convert face to same dtype as VAE BEFORE encoding
                            face = face.to(dtype=vae_dtype, device=vae_device)
                            
                            # Normalize if needed
                            if face.max() > 1.0:
                                face = face / 127.5 - 1.0
                            
                            # Use safe encoding
                            face_latent = self.safe_encode_with_vae(face)
                            latents.append(face_latent)
                        
                        # Stack latents along face dimension
                        latents = torch.stack(latents, dim=1)  # [batch, 6, C, H, W]
                        
                        # IMPORTANT FIX: Ensure latents have the correct dtype
                        latents = self.ensure_dtype(latents)

                        # Add noise to latents
                        noise = torch.randn_like(latents)
                        
                        # IMPORTANT FIX: Ensure noise has the correct dtype
                        noise = self.ensure_dtype(noise)
                        
                        # Sample random timesteps
                        timesteps = torch.randint(
                            0, 
                            self.noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],),
                            device=self.accelerator.device,
                        )
                        # Debug tensors before adding noise
                        # get_tensor_info(latents, "latents before noise")
                        # get_tensor_info(noise, "noise tensor")
                        # get_tensor_info(timesteps, "timesteps")

                        # Add noise to latents
                        # noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                        # run the scheduler in fp32, then convert back
                        print(f"trainer.py train() training loop, before self.noise_scheduler.add_noise ... \n")
                        latents_f32 = latents.float()
                        noise_f32   = noise.float()
                        noisy_latents = self.noise_scheduler.add_noise(
                            latents_f32, noise_f32, timesteps
                        ).to(latents.dtype)          # ← back to fp16
                        print(f"trainer.py train() training loop, after self.noise_scheduler.add_noise \n")

                        # Get model prediction
                        print(f"trainer.py train() training loop, before self.model(noisy_latents, timesteps, text_embeddings)  \n")
                        noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                        print(f"trainer.py train() training loop, after self.model(noisy_latents, timesteps, text_embeddings)  \n")

                        # Compute loss
                        print(f"trainer.py train() training loop, before compute loss  \n")
                        if self.config.prediction_type == "epsilon":
                            target = noise  # Already has the correct dtype
                        elif self.config.prediction_type == "v_prediction":
                            # Both inputs already have the correct dtype from ensure_dtype
                            # target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                            print(f"trainer.py train() training loop, before self.noise_scheduler.get_velocity\n")
                            target = self.noise_scheduler.get_velocity(
                                        latents.float(),      # scheduler wants fp32
                                        noise.float(),
                                        timesteps
                                    )       # keep it fp32
                            print(f"trainer.py train() training loop, after self.noise_scheduler.get_velocity\n")
                        else:
                            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
                        
                        # Ensure both tensors have the same dtype for loss computation
                        # If noise_pred is float32, convert target to float32
                        # If target is in a different dtype, convert it to match noise_pred
                        # if target.dtype != noise_pred.dtype:
                        #     target = target.to(dtype=noise_pred.dtype)
                        #     print(f"Converted target dtype to {target.dtype} to match noise_pred as {noise_pred.dtype} \n")
                        
                        # Make sure both have the same dtype for loss computation
                        # Always compute loss in float32 for stability and gradient scaling
                        noise_pred = noise_pred.float()  # Ensure float32 for loss computation
                        target = target.float()          # Ensure float32 for loss computation

                        print(f"trainer.py train() training loop, after compute loss, before mse_loss, noise_pred dtype is {noise_pred.dtype}\
                              target dtype is {target.dtype} \n")
                        loss = torch.nn.functional.mse_loss(noise_pred, target)
                        print(f"trainer.py train() training loop, after compute loss, after mse_loss \n")
                        
                        # Backward pass
                        print(f"trainer.py train() training loop, before backward \n")
                        self.accelerator.backward(loss)
                        print(f"trainer.py train() training loop, after backward \n")

                        # Clip gradients
                        # if self.config.max_grad_norm > 0:
                        #     self.accelerator.clip_grad_norm_(
                        #         filter(lambda p: p.requires_grad, self.model.parameters()),
                        #         self.config.max_grad_norm,
                        #     )
                        # print(f"trainer.py train() training loop, after Clip gradients \n")


                        # Clip gradients & update params only on the last micro‑batch of the accumulation window
                        if self.accelerator.sync_gradients:          # True only when gradients are synchronized
                            if self.config.max_grad_norm > 0:
                                self.accelerator.clip_grad_norm_(
                                    filter(lambda p: p.requires_grad, self.model.parameters()),
                                    self.config.max_grad_norm,
                                )
                            print("trainer.py train() training loop, after Clip gradients\n")
                            
                            # update model parameters + scaler in one call
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)              # clear grads for next accumulation window
                                            
                    # Log metrics
                    if global_step % self.config.log_every_n_steps == 0:
                        # Get current time elapsed
                        time_elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60.0  # minutes
                        
                        # Log metrics
                        current_lr = lr_scheduler.get_last_lr()[0]
                        loss_value = loss.item()
                        
                        # Update metrics dictionary
                        metrics['loss'].append(loss_value)
                        metrics['step'].append(global_step)
                        metrics['lr'].append(current_lr)
                        metrics['time'].append(time_elapsed)
                        
                        # Write to local log file
                        with open(log_file, 'a') as f:
                            f.write(f"{global_step},{loss_value},{current_lr},{time_elapsed}\n")
                        
                        # Update progress bar
                        progress_bar.set_postfix(loss=loss_value, lr=f"{current_lr:.2e}")
                        
                        # Log to wandb if enabled
                        if self.wandb_run:
                            self.wandb_run.log({
                                "train/loss": loss_value,
                                "train/lr": current_lr,
                                "train/epoch": epoch,
                                "train/step": global_step,
                                "train/time_minutes": time_elapsed
                            })
                        
                        # Update local plot
                        self.steps.append(global_step)
                        self.losses.append(loss_value)
                        
                        self.ax.clear()
                        self.ax.plot(self.steps, self.losses)
                        self.ax.set_xlabel('Step')
                        self.ax.set_ylabel('Loss')
                        self.ax.set_title('Training Loss')
                        self.ax.grid(True)
                        
                        # Save the plot
                        plt.savefig(os.path.join(self.logs_dir, 'loss_curve.png'))
                    
                    # Save model checkpoint
                    if global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
                    
                    # Generate samples
                    if hasattr(self.config, 'sample_every_n_steps') and \
                    self.config.sample_every_n_steps > 0 and \
                    global_step % self.config.sample_every_n_steps == 0:
                        self.generate_samples(sample_prompts, global_step)
                    
                    # Evaluate model
                    if val_dataloader and global_step % self.config.eval_every_n_steps == 0:
                        eval_loss = self.evaluate(val_dataloader)
                        
                        # Log validation loss
                        if self.wandb_run:
                            self.wandb_run.log({"val/loss": eval_loss})
                        
                        # Write to local log file
                        with open(os.path.join(self.logs_dir, "eval_log.csv"), 'a') as f:
                            f.write(f"{global_step},{eval_loss}\n")
                    
                    global_step += 1
                    progress_bar.update(1)
                    
                    # Break if maximum steps reached
                    if global_step >= num_train_epochs:
                        break
                
                # Break if maximum steps reached
                if global_step >= num_train_epochs:
                    break
                    
            except RuntimeError as e:
                # Check if this is a memory-related error
                error_msg = str(e).lower()
                is_memory_error = any(term in error_msg for term in [
                    "out of memory", "shared memory", "bus error", 
                    "dataloader worker", "connection refused"
                ])
                
                if is_memory_error:
                    memory_retry_count += 1
                    
                    if memory_retry_count <= max_memory_retries:
                        print(f"\nMemory error encountered ({memory_retry_count}/{max_memory_retries}): {str(e)}")
                        print("Attempting to recover by clearing caches and reducing batch size...")
                        
                        # Clear CUDA cache if using GPU
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Reduce batch size temporarily
                        if hasattr(self.config, 'batch_size') and self.config.batch_size > 1:
                            original_batch_size = self.config.batch_size
                            self.config.batch_size = max(1, self.config.batch_size // 2)
                            print(f"Temporarily reducing batch size from {original_batch_size} to {self.config.batch_size}")
                            
                            # Recreate dataloaders with smaller batch size
                            train_dataloader, val_dataloader = self.prepare_dataloaders(train_dataset, val_dataset)
                            train_dataloader = self.accelerator.prepare(train_dataloader)
                            if val_dataloader:
                                val_dataloader = self.accelerator.prepare(val_dataloader)
                        
                        # Reduce number of workers
                        if hasattr(self.config, 'num_workers') and self.config.num_workers > 0:
                            original_workers = self.config.num_workers
                            self.config.num_workers = max(0, self.config.num_workers - 1)
                            print(f"Reducing workers from {original_workers} to {self.config.num_workers}")
                        
                        # Sleep briefly to let system recover
                        time.sleep(5)
                        
                        # Skip to next epoch
                        print("Skipping to next epoch...")
                        continue
                    else:
                        print(f"\nFailed to recover after {max_memory_retries} attempts. Memory issues persist.")
                        print("Try manually reducing batch size, number of workers, or increasing system shared memory.")
                        print(f"Error details: {str(e)}")
                        
                        # Create error log
                        with open(os.path.join(self.logs_dir, "error_log.txt"), 'w') as f:
                            f.write(f"Error at step {global_step}:\n{str(e)}\n")
                            f.write(f"Current settings: batch_size={self.config.batch_size}, num_workers={self.config.num_workers}\n")
                        
                        # Save checkpoint before exiting
                        try:
                            self.save_checkpoint(os.path.join(self.output_dir, f"error_checkpoint-{global_step}"))
                        except:
                            pass
                        
                        raise
                else:
                    # This is not a memory error, so re-raise
                    print(f"\nNon-memory error encountered: {str(e)}")
                    
                    # Create error log
                    with open(os.path.join(self.logs_dir, "error_log.txt"), 'w') as f:
                        f.write(f"Error at step {global_step}:\n{str(e)}\n")
                    
                    raise
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, "final_model"))
        
        # Save final metrics plot
        self.ax.clear()
        self.ax.plot(metrics['step'], metrics['loss'])
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss')
        self.ax.grid(True)
        plt.savefig(os.path.join(self.logs_dir, 'final_loss_curve.png'))
        
        # Save metrics as numpy arrays for later analysis
        np.save(os.path.join(self.logs_dir, 'metrics.npy'), metrics)
        
        # Finalize wandb logging
        if self.wandb_run:
            # Save the final plot to wandb
            self.wandb_run.log({"final_loss_curve": wandb.Image(os.path.join(self.logs_dir, 'final_loss_curve.png'))})
            
            # Create a summary table of the training process
            total_time = (datetime.datetime.now() - start_time).total_seconds() / 3600.0  # hours
            final_loss = metrics['loss'][-1] if metrics['loss'] else float('nan')
            
            summary = {
                "final_loss": final_loss,
                "total_training_hours": total_time,
                "total_steps": global_step,
                "steps_per_hour": global_step / total_time if total_time > 0 else 0,
            }
            
            for key, value in summary.items():
                self.wandb_run.summary[key] = value
            
            print("\nTraining completed successfully!")
            print(f"Total time: {total_time:.2f} hours")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Results saved to {self.output_dir}")
            print(f"To view training visualizations in JupyterLab: wandb.jupyter.show()")
        
        return global_step

    def evaluate(self, val_dataloader):
        """
        Evaluate the model on validation set with proper dtype handling.
        
        Args:
            val_dataloader: Validation dataloader
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                # Get data from batch
                faces = batch["faces"]  # [batch, 6, C, H, W]
                captions = batch["caption"]
                
                # Encode text
                tokens = self.tokenizer(
                    captions, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(self.accelerator.device)
                
                # Get text embeddings
                tokens = tokens.to(self.accelerator.device)
                text_embeddings = self.text_encoder(tokens)[0]
                
                # Ensure text embeddings have the right dtype
                text_embeddings = self.ensure_dtype(text_embeddings)
                
                # Process images to latent space using safe encoder
                latents = []
                
                # Get VAE dtype and device
                vae_param = next(self.vae.parameters())
                vae_dtype = vae_param.dtype
                vae_device = vae_param.device
                
                for i in range(faces.shape[1]):
                    face = faces[:, i]
                    if face.shape[1] == 4 and face.shape[2] == 64 and face.shape[3] == 64:
                        # Already a latent, no need to encode
                        face_latent = face
                    else: 
                        # Handle channel dimension ordering
                        if face.shape[-1] == 3:  # If channels are last
                            face = face.permute(0, 3, 1, 2)  # NHWC -> NCHW
                        
                        # Convert to VAE dtype
                        face = face.to(dtype=vae_dtype, device=vae_device)
                        
                        # Normalize if needed
                        if face.max() > 1.0:
                            face = face / 127.5 - 1.0
                        
                        # Encode safely
                        face_latent = self.safe_encode_with_vae(face)
                    latents.append(face_latent)
                
                # Stack latents along face dimension
                latents = torch.stack(latents, dim=1)  # [batch, 6, C, H, W]
                
                # Ensure latents have the correct dtype
                latents = self.ensure_dtype(latents)
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                
                # Ensure noise has the correct dtype
                noise = self.ensure_dtype(noise)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.accelerator.device,
                )
                # Debug tensors before adding noise
                # get_tensor_info(latents, "latents before noise")
                # get_tensor_info(noise, "noise tensor")
                # get_tensor_info(timesteps, "timesteps")

                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get model prediction
                noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                
                # Compute loss
                if self.config.prediction_type == "epsilon":
                    target = noise
                elif self.config.prediction_type == "v_prediction":
                    # Both inputs already have the correct dtype
                    # target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    target = self.noise_scheduler.get_velocity(
                            latents.float(),      # scheduler wants fp32
                            noise.float(),
                            timesteps
                        ).to(latents.dtype)       # ← back to fp16
                else:
                    raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
                
                loss = torch.nn.functional.mse_loss(noise_pred, target)
                total_loss += loss.item()
            
            # Compute average loss
            avg_loss = total_loss / len(val_dataloader)
            
            print(f"Validation loss: {avg_loss:.6f}")
            
            self.model.train()
            
            return avg_loss

    def generate_samples(self, prompts, step):
        """
        Generate and save sample images.
        
        Args:
            prompts: List of text prompts
            step: Current training step
        """
        print(f"trainer.py - generate_samples - started with prompts {prompts}, step as {step} ...")
        # We need a minimal inference pipeline
        # This is a simplified version just for visualization
        from ..inference.pipeline import CubeDiffPipeline
        
        # Create samples directory for this step
        samples_dir = os.path.join(self.images_dir, f"step_{step}")
        os.makedirs(samples_dir, exist_ok=True)
        print(f"trainer.py - generate_samples - samples_dir is {samples_dir}")

        # Create temporary checkpoint for the current model
        temp_ckpt = os.path.join(self.output_dir, "temp_model.pt")
        print(f"trainer.py - generate_samples - temp_ckpt dir is {temp_ckpt}")

        self.accelerator.wait_for_everyone()
        print(f"trainer.py - generate_samples - after self.accelerator.wait_for_everyone()")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        print(f"trainer.py - generate_samples - after self.accelerator.unwrap_model")
        self.accelerator.save(unwrapped_model.state_dict(), temp_ckpt)
        print(f"trainer.py - generate_samples - after self.accelerator.save(unwrapped_model.state_dict(), temp_ckpt)")

        # Initialize inference pipeline
        pipeline = CubeDiffPipeline(
            pretrained_model_name=self.pretrained_model_name,
            checkpoint_path=temp_ckpt,
            device=self.accelerator.device,
        )
        
        # Generate samples for each prompt
        print(f"trainer.py - generate_samples - generate panorama with prompts started ...")
        for i, prompt in enumerate(prompts):
            try:
                with torch.cuda.amp.autocast():
                    panorama = pipeline.generate(
                        prompt=prompt,
                        negative_prompt="low quality, blurry, distorted",
                        num_inference_steps=30,  # Use fewer steps for faster visualization
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        output_type="pil"
                    )
                
                # Save the panorama
                output_path = os.path.join(samples_dir, f"sample_{i}.jpg")
                panorama.save(output_path)
                print(f"trainer.py - generate_samples - saved panorama {i} to {output_path}")

                # Log to wandb
                if self.wandb_run:
                    self.wandb_run.log({
                        f"samples/prompt_{i}": wandb.Image(
                            panorama, 
                            caption=prompt
                        )
                    })
                
                print(f"Generated sample for prompt: {prompt}")
            except Exception as e:
                print(f"Error generating sample for prompt '{prompt}': {e}")
        
        # Clean up temporary checkpoint
        if os.path.exists(temp_ckpt):
            os.remove(temp_ckpt)
            print(f"trainer.py - generate_samples - removed {temp_ckpt}")
        print(f"trainer.py - generate_samples - done retunred")

    def save_checkpoint(self, checkpoint_dir):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.accelerator.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            
            # Save config
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                import json
                json.dump(vars(self.config), f, indent=2)
            
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        
        print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def show_training_visualizations(self):
        """
        Display training visualizations in Jupyter notebook.
        """
        if self.wandb_run:
            import wandb.jupyter as wandb_jupyter
            wandb_jupyter.show()
        else:
            # Display local visualizations
            from IPython.display import display, Image
            
            loss_curve_path = os.path.join(self.logs_dir, 'loss_curve.png')
            if os.path.exists(loss_curve_path):
                display(Image(filename=loss_curve_path))
            else:
                print("No loss curve found. Training may not have started or logged yet.")
            
            # Display sample images if available
            sample_dirs = [d for d in os.listdir(self.images_dir) if d.startswith('step_')]
            if sample_dirs:
                latest_sample_dir = sorted(sample_dirs, key=lambda x: int(x.split('_')[1]))[-1]
                samples = [f for f in os.listdir(os.path.join(self.images_dir, latest_sample_dir)) if f.endswith('.jpg')]
                
                print(f"Latest samples from {latest_sample_dir}:")
                for sample in samples[:3]:  # Show up to 3 samples
                    display(Image(filename=os.path.join(self.images_dir, latest_sample_dir, sample)))
            else:
                print("No sample images found.")

    # Add this function to your CubeDiffTrainer class to fix dimension issues
    def process_faces(self, faces):
        """
        Pre-process face images to ensure correct dimensions and dtype before encoding.
        
        Args:
            faces: Tensor of face images [batch, 6, H, W, C] or [batch, 6, C, H, W]
                
        Returns:
            Processed faces with correct dimensions for VAE encoding
        """
        # Print original shape for debugging
        print(f"Original faces shape: {faces.shape}")
        
        # Get VAE dtype and device
        vae_param = next(self.vae.parameters())
        vae_dtype = vae_param.dtype
        vae_device = vae_param.device
        
        # Check if we need to permute dimensions
        # VAE expects images in [B, C, H, W] format
        if faces.shape[-1] == 3:  # If channels are last (NHWC format)
            # Convert NHWC -> NCHW format
            faces_processed = faces.permute(0, 1, 4, 2, 3)
            print(f"Permuted faces from NHWC to NCHW: {faces_processed.shape}")
        else:
            faces_processed = faces
        
        # CRITICAL: Ensure dtype and device match VAE
        faces_processed = faces_processed.to(dtype=vae_dtype, device=vae_device)
        
        # Normalize images to [-1, 1] if not already
        if faces_processed.max() > 1.0:
            print("Normalizing images from [0, 255] to [-1, 1]")
            faces_processed = faces_processed / 127.5 - 1.0
        
        return faces_processed

    # Add this function to your CubeDiffTrainer class to encode faces to latents    
    def encode_faces_to_latents(self, faces_batch):
        """
        Safely encode face images to latent representations.
        Handles data type conversion and dimension ordering.
        Optimized for L4 GPU mixed precision training.
        
        Args:
            faces_batch: Tensor of shape [batch, 6, C, H, W] or [batch, 6, H, W, C]
            
        Returns:
            Latent representations of shape [batch, 6, latent_C, latent_H, latent_W]
        """
        with torch.no_grad():
            # Get VAE parameters to match dtype and device
            vae_param = next(self.vae.parameters())
            vae_dtype = vae_param.dtype
            vae_device = vae_param.device
            
            # CRITICAL: Convert faces to match VAE dtype and device
            faces_processed = faces_batch.to(dtype=vae_dtype, device=vae_device)
            
            # Check if we need to permute dimensions (VAE expects NCHW format)
            if faces_processed.shape[-1] == 3:  # If channels are last (NHWC format)
                # Convert NHWC -> NCHW format
                faces_processed = faces_processed.permute(0, 1, 4, 2, 3)
                print(f"Permuted faces from NHWC to NCHW: {faces_processed.shape}")
            
            # Normalize images to [-1, 1] if not already
            if faces_processed.max() > 1.0:
                print("Normalizing images from [0, 255] to [-1, 1]")
                faces_processed = faces_processed / 127.5 - 1.0
            
            # For L4 GPU, we can try batched encoding to improve performance
            try:
                # First try to encode all faces in a single batch
                # This is more efficient but might fail if memory is tight
                faces_reshaped = faces_processed.reshape(-1, *faces_processed.shape[2:])
                latents_all = self.vae.encode(faces_reshaped).latent_dist.sample() * 0.18215
                
                # Reshape back to [batch, 6, latent_C, latent_H, latent_W]
                latents = latents_all.reshape(
                    faces_processed.shape[0], faces_processed.shape[1], *latents_all.shape[1:]
                )
                
                print("Successfully encoded all faces in a single batch")
                
            except RuntimeError as e:
                # Fallback to per-face encoding
                print(f"Batch encoding failed, falling back to per-face encoding: {e}")
                
                # Encode each face to latent space using safe_encode_with_vae
                latents = []
                for i in range(faces_processed.shape[1]):
                    # Get single face batch
                    face = faces_processed[:, i]  # [batch, C, H, W]
                    face_latent = self.safe_encode_with_vae(face)
                    latents.append(face_latent)
                
                # Stack latents along face dimension
                latents = torch.stack(latents, dim=1)  # [batch, 6, C, H, W]
            
            return latents

    def enable_progressive_loading(self, enable=True):
        """
        Enable progressive loading to gracefully handle memory issues.
        Starts with minimal settings and gradually increases to target values.
        
        Args:
            enable: Whether to enable progressive loading
        """
        self.use_progressive_loading = enable
        if enable:
            # Store original values
            self._original_batch_size = self.config.batch_size
            self._original_num_workers = self.config.num_workers
            
            # Start with conservative values
            self.config.batch_size = self.config.batch_size # 1
            self.config.num_workers = self.config.num_workers # 0
            
            print(f"Progressive loading enabled. Starting with batch_size={self.config.batch_size}, num_workers={self.config.num_workers}")
            print(f"Will gradually increase to batch_size={self._original_batch_size}, num_workers={self._original_num_workers}")

    def progressive_loading_step(self, global_step):
        """
        Update batch size and workers progressively as training proceeds.
        
        Args:
            global_step: Current training step
        """
        if not hasattr(self, 'use_progressive_loading') or not self.use_progressive_loading:
            return
        
        # Only make changes during early steps
        if global_step > 50:
            # Restore original values if we're past early stages
            if self.config.batch_size != self._original_batch_size or self.config.num_workers != self._original_num_workers:
                print(f"Progressive loading complete. Restoring original settings:")
                print(f"  batch_size: {self.config.batch_size} -> {self._original_batch_size}")
                print(f"  num_workers: {self.config.num_workers} -> {self._original_num_workers}")
                
                self.config.batch_size = self._original_batch_size
                self.config.num_workers = self._original_num_workers
                
                # Recreate dataloaders with final settings
                self.train_dataloader, self.val_dataloader = self.prepare_dataloaders(
                    self.train_dataset, self.val_dataset
                )
                
                # Prepare with accelerator
                self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
                if self.val_dataloader:
                    self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
            
            # Disable progressive loading to avoid further changes
            self.use_progressive_loading = False
            return
            
        # Step 1: At step 10, try increasing workers if batch size is still 1
        if global_step == 10 and self.config.batch_size == 1 and self._original_num_workers > 0:
            new_workers = min(2, self._original_num_workers)
            print(f"Progressive loading: Increasing num_workers: 0 -> {new_workers}")
            self.config.num_workers = new_workers
            
            # Recreate dataloaders with new settings
            self.train_dataloader, self.val_dataloader = self.prepare_dataloaders(
                self.train_dataset, self.val_dataset
            )
            
            # Prepare with accelerator
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
            if self.val_dataloader:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
            
        # Step 2: At step 25, try increasing batch size if it's still 1
        elif global_step == 25 and self.config.batch_size == 1 and self._original_batch_size > 1:
            new_batch_size = min(2, self._original_batch_size)
            print(f"Progressive loading: Increasing batch_size: 1 -> {new_batch_size}")
            self.config.batch_size = new_batch_size
            
            # Recreate dataloaders with new settings
            self.train_dataloader, self.val_dataloader = self.prepare_dataloaders(
                self.train_dataset, self.val_dataset
            )
            
            # Prepare with accelerator
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
            if self.val_dataloader:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

    # Add this function to the CubeDiffTrainer class in trainer.py
    def safe_encode_with_vae(self, face):
        """
        Safely encode a face with VAE handling dtype mismatches gracefully.
        Always ensures the input tensor has the same dtype as the VAE.
        
        Args:
            face: Image tensor of shape [batch, C, H, W]
            
        Returns:
            Latent representation
        """
        # Get VAE parameter dtype and device
        vae_param = next(self.vae.parameters())
        vae_dtype = vae_param.dtype
        vae_device = vae_param.device
        
        # CRITICAL: Ensure tensor has same dtype and device as VAE
        if face.dtype != vae_dtype or face.device != vae_device:
            face = face.to(dtype=vae_dtype, device=vae_device)
        
        try:
            # Try to encode with matching dtype
            face_latent = self.vae.encode(face).latent_dist.sample()
            return face_latent * 0.18215
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "dtype" in error_msg or "type" in error_msg:
                # Something still went wrong with types - try an alternative approach
                # Make sure tensor is contiguous
                face = face.contiguous()
                face_latent = self.vae.encode(face).latent_dist.sample()
                return face_latent * 0.18215
            else:
                # Not a dtype issue, re-raise
                raise
        
    def ensure_dtype(self, tensor, reference_tensor=None):
        """
        Ensure tensor has correct dtype matching model or reference tensor
        
        Args:
            tensor: The tensor to convert
            reference_tensor: Optional reference tensor whose dtype to match
            
        Returns:
            Tensor with the correct dtype
        """
        if reference_tensor is not None:
            return tensor.to(dtype=reference_tensor.dtype)
        else:
            model_dtype = next(self.model.parameters()).dtype
            return tensor.to(dtype=model_dtype)

# Add this function to trainer.py
def add_memory_monitoring(trainer):
    """
    Add memory monitoring capabilities to the trainer to detect and
    recover from shared memory issues automatically.
    """
    # Store original trainer train method
    original_train_method = trainer.train
    
    def memory_monitor_thread(trainer):
        """Background thread that monitors memory usage"""
        try:
            while getattr(trainer, '_monitoring_active', True):
                # Check system memory usage
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Check if memory usage is critical (>90%)
                if memory.percent > 90 or swap.percent > 90:
                    print("\nCritical memory usage detected!")
                    print(f"RAM: {memory.percent}% used, Swap: {swap.percent}% used")
                    
                    # Set trainer flag to reduce batch size/workers
                    trainer._reduce_memory_usage = True
                
                # Check specifically for shared memory on Linux
                try:
                    if os.path.exists('/dev/shm'):
                        shm_stats = os.statvfs('/dev/shm')
                        shm_total = shm_stats.f_blocks * shm_stats.f_frsize
                        shm_free = shm_stats.f_bfree * shm_stats.f_frsize
                        shm_used_percent = 100 - (shm_free / shm_total * 100)
                        
                        if shm_used_percent > 85:
                            print(f"\n/dev/shm usage critical: {shm_used_percent:.1f}%")
                            trainer._reduce_memory_usage = True
                except:
                    pass
                
                # Sleep for a few seconds before checking again
                time.sleep(10)
        except:
            pass
    
    def wrapped_train(*args, **kwargs):
        """Wrapper around train method with memory monitoring"""
        # Start monitoring thread
        trainer._monitoring_active = True
        trainer._reduce_memory_usage = False
        
        monitor_thread = threading.Thread(
            target=memory_monitor_thread,
            args=(trainer,),
            daemon=True
        )
        monitor_thread.start()
        
        try:
            # Call original train method
            result = original_train_method(*args, **kwargs)
            return result
        finally:
            # Stop monitoring thread
            trainer._monitoring_active = False
            monitor_thread.join(timeout=1.0)
    
    # Replace train method with wrapped version
    trainer.train = wrapped_train
    
    # Add dynamic memory adaptation method
    def adapt_to_memory_pressure(self):
        """Dynamically reduce memory usage during training"""
        if getattr(self, '_reduce_memory_usage', False):
            print("\nReducing memory usage due to system pressure...")
            
            # Already at minimum?
            if self.config.batch_size <= 1 and self.config.num_workers <= 0:
                print("Already at minimum memory usage settings")
                return False
            
            # Store original values
            original_batch_size = self.config.batch_size
            original_workers = self.config.num_workers
            
            # Reduce batch size if possible
            if self.config.batch_size > 1:
                self.config.batch_size = max(1, self.config.batch_size // 2)
            
            # Reduce workers if possible
            if self.config.num_workers > 0:
                self.config.num_workers = max(0, self.config.num_workers - 1)
            
            # Reset flag
            self._reduce_memory_usage = False
            
            print(f"Adjusted batch_size: {original_batch_size} -> {self.config.batch_size}")
            print(f"Adjusted num_workers: {original_workers} -> {self.config.num_workers}")
            
            # Increase gradient accumulation to compensate
            self.config.gradient_accumulation_steps *= 2
            print(f"Increased gradient_accumulation_steps to {self.config.gradient_accumulation_steps}")
            
            return True
        
        return False
    
    # Add method to trainer
    trainer.adapt_to_memory_pressure = types.MethodType(adapt_to_memory_pressure, trainer)
    
    return trainer 


# Define the function in a cell before training
def get_tensor_info(tensor, name="tensor"):
    """Print debug info for a tensor"""
    print(f"{name} - Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
    print(f"{name} - Min: {tensor.min().item()}, Max: {tensor.max().item()}")
    try:
        print(f"{name} - Requires grad: {tensor.requires_grad}")
        if tensor.requires_grad:
            print(f"{name} - Grad fn: {tensor.grad_fn}")
    except:
        pass


def fix_loss_curve(trainer):
    """
    Minimal fix for loss curve visualization that produces just a single figure.
    This solution intercepts wandb logging to capture and plot loss data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Initialize empty arrays for tracking loss data
    trainer.steps = []
    trainer.losses = []
    
    # Create a single figure that will be updated
    trainer.fig, trainer.ax = plt.subplots(figsize=(10, 6))
    trainer.ax.set_xlabel('Step')
    trainer.ax.set_ylabel('Loss')
    trainer.ax.set_title('Training Loss')
    trainer.ax.grid(True)
    trainer.ax.set_xlim([0, 10])  # Initial view until we get data
    trainer.ax.set_ylim([0, 2])   # Initial y-range until we get data
    
    # Store original wandb log method to intercept data
    if hasattr(trainer, 'wandb_run') and trainer.wandb_run is not None:
        original_wandb_log = trainer.wandb_run.log
        
        def intercept_wandb_log(log_dict, *args, **kwargs):
            """Intercept wandb logging to capture loss data"""
            # Extract loss and step if present
            if 'train/loss' in log_dict and 'train/step' in log_dict:
                loss_val = log_dict['train/loss']
                step_val = log_dict['train/step']
                
                # Store the values without duplicates
                if step_val not in trainer.steps:
                    trainer.steps.append(step_val)
                    trainer.losses.append(loss_val)
                    
                    # Update the plot
                    trainer.ax.clear()
                    trainer.ax.plot(trainer.steps, trainer.losses, 'b-', linewidth=2)
                    trainer.ax.set_xlabel('Step')
                    trainer.ax.set_ylabel('Loss')
                    trainer.ax.set_title('Training Loss')
                    trainer.ax.grid(True)
                    
                    # Set proper axis limits
                    trainer.ax.set_xlim([0, max(trainer.steps) * 1.05])
                    
                    if len(trainer.losses) > 1:
                        min_loss = min(trainer.losses)
                        max_loss = max(trainer.losses)
                        margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
                        trainer.ax.set_ylim([max(0, min_loss - margin), max_loss + margin])
                    
                    # Save the updated plot, overwriting previous version
                    plt.tight_layout()
                    plt.savefig(os.path.join(trainer.logs_dir, 'loss_curve.png'))
                    
                    # Save raw data as backup
                    np.save(os.path.join(trainer.logs_dir, 'loss_data.npy'), 
                          {'steps': trainer.steps, 'losses': trainer.losses})
            
            # Call the original log method
            return original_wandb_log(log_dict, *args, **kwargs)
        
        # Replace wandb log method to intercept data
        trainer.wandb_run.log = intercept_wandb_log
    
    # Override existing plot method if it exists
    if hasattr(trainer, 'steps') and hasattr(trainer, 'losses'):
        # Don't create multiple figures - just update the existing one
        original_plot = plt.figure
        
        def single_figure_plot(*args, **kwargs):
            """Ensure only one figure is created"""
            if hasattr(trainer, 'fig') and plt.fignum_exists(trainer.fig.number):
                # Return existing figure instead of creating a new one
                return trainer.fig
            else:
                # If no figure exists, create one
                return original_plot(*args, **kwargs)
        
        # Replace plt.figure with our version
        plt.figure = single_figure_plot
    
    # Save the initial empty plot
    plt.tight_layout()
    os.makedirs(trainer.logs_dir, exist_ok=True)
    plt.savefig(os.path.join(trainer.logs_dir, 'loss_curve.png'))
    
    return trainer


# def direct_fix_loss_curve(trainer):
#     """
#     Direct fix for the loss curve issue that accesses wandb log data
#     and creates a proper visualization without modifying training methods.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os
#     import types
    
#     # Initialize empty arrays for tracking loss data
#     trainer.steps = []
#     trainer.losses = []
    
#     # Store original wandb log method to intercept data
#     if hasattr(trainer, 'wandb_run') and trainer.wandb_run is not None:
#         original_wandb_log = trainer.wandb_run.log
        
#         def intercept_wandb_log(log_dict, *args, **kwargs):
#             """Intercept wandb logging to capture loss data"""
#             # Extract loss and step if present
#             if 'train/loss' in log_dict and 'train/step' in log_dict:
#                 loss_val = log_dict['train/loss']
#                 step_val = log_dict['train/step']
                
#                 # Store the values
#                 if step_val not in trainer.steps:
#                     trainer.steps.append(step_val)
#                     trainer.losses.append(loss_val)
                    
#                     # Create a proper plot with the data
#                     if not hasattr(trainer, 'fig') or trainer.fig is None:
#                         trainer.fig, trainer.ax = plt.subplots(figsize=(10, 6))
                    
#                     trainer.ax.clear()
#                     trainer.ax.plot(trainer.steps, trainer.losses, 'b-', linewidth=2)
#                     trainer.ax.set_xlabel('Step')
#                     trainer.ax.set_ylabel('Loss')
#                     trainer.ax.set_title('Training Loss')
#                     trainer.ax.grid(True)
                    
#                     # Set proper axis limits
#                     trainer.ax.set_xlim([0, max(trainer.steps) * 1.05])
                    
#                     if len(trainer.losses) > 1:
#                         min_loss = min(trainer.losses)
#                         max_loss = max(trainer.losses)
#                         margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
#                         trainer.ax.set_ylim([max(0, min_loss - margin), max_loss + margin])
                    
#                     # Save the plot
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(trainer.logs_dir, 'loss_curve.png'))
                    
#                     # Also save raw data as backup
#                     np.save(os.path.join(trainer.logs_dir, 'loss_data.npy'), 
#                           {'steps': trainer.steps, 'losses': trainer.losses})
            
#             # Call the original log method
#             return original_wandb_log(log_dict, *args, **kwargs)
        
#         # Replace wandb log method to intercept data
#         trainer.wandb_run.log = intercept_wandb_log
    
#     # Create the initial plot figure
#     if not hasattr(trainer, 'fig') or trainer.fig is None:
#         trainer.fig, trainer.ax = plt.subplots(figsize=(10, 6))
#         trainer.ax.set_xlabel('Step')
#         trainer.ax.set_ylabel('Loss')
#         trainer.ax.set_title('Training Loss')
#         trainer.ax.grid(True)
#         trainer.ax.set_xlim([0, 10])  # Initial view until we get data
#         trainer.ax.set_ylim([0, 2])   # Initial y-range until we get data
    
#     # Save the initial empty plot
#     plt.tight_layout()
#     os.makedirs(trainer.logs_dir, exist_ok=True)
#     plt.savefig(os.path.join(trainer.logs_dir, 'loss_curve_init.png'))
    
#     return trainer
