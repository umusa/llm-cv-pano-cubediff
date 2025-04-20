"""
trainer.py - Training and evaluation utilities for CubeDiff

This module contains functions for training, validating, and evaluating the CubeDiff model.
"""

import os
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Union, Tuple

# Import model components
from model import CubeDiff, CubeDiffPipeline
from data import visualize_cubemap, equirectangular_to_cubemap_batch, cubemap_to_equirectangular_batch


class CubeDiffTrainer:
    """
    Trainer for the CubeDiff model.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        device=None,
        output_dir="output",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CubeDiff model
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer: Optional optimizer (if not provided, will be created)
            lr_scheduler: Optional learning rate scheduler
            device: Device to use
            output_dir: Directory to save outputs
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Create optimizer if not provided
        if optimizer is None:
            # Only optimize trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        else:
            self.optimizer = optimizer
        
        # Set learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def train(
        self,
        num_epochs=30,
        gradient_accumulation_steps=1,
        save_interval=1000,
        eval_interval=1000,
        log_interval=100,
        max_grad_norm=1.0,
        sample_interval=5000,
        sample_prompts=None,
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            save_interval: Interval (in steps) to save model checkpoints
            eval_interval: Interval (in steps) to evaluate model
            log_interval: Interval (in steps) to log progress
            max_grad_norm: Maximum gradient norm for gradient clipping
            sample_interval: Interval (in steps) to generate samples
            sample_prompts: Prompts to use for generating samples
        """
        # Set default sample prompts if not provided
        if sample_prompts is None:
            sample_prompts = [
                "A beautiful sunset over the ocean",
                "A cozy log cabin in a snowy forest"
            ]
        
        # Set model to training mode
        self.model.train()
        
        # Print training information
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Main training loop
        for epoch in range(self.current_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            # Progress bar for this epoch
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=True
            )
            
            # Training loop for this epoch
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                # Perform a single training step
                loss = self.training_step(batch, step)
                
                # Accumulate gradients if needed
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Only update every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_grad_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update learning rate
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                
                # Update progress
                epoch_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Update global step
                self.global_step += 1
                
                # Log progress
                if self.global_step % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.history['learning_rate'].append(lr)
                    self.history['train_loss'].append(loss.item())
                    
                    print(f"Step {self.global_step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}")
                
                # Evaluate model
                if self.val_dataloader is not None and self.global_step % eval_interval == 0:
                    val_loss = self.evaluate()
                    self.history['val_loss'].append(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                    
                    # Resume training
                    self.model.train()
                
                # Generate samples
                if self.global_step % sample_interval == 0:
                    self.generate_samples(sample_prompts)
                    # Resume training
                    self.model.train()
            
            # End of epoch
            progress_bar.close()
            
            # Calculate average loss for this epoch
            avg_loss = epoch_loss / len(self.train_dataloader)
            
            # Update epoch counter
            self.current_epoch += 1
            
            # Evaluate at the end of each epoch
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                self.history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model")
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}")
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {self.current_epoch}/{num_epochs} completed in {epoch_time:.2f}s, Avg loss: {avg_loss:.4f}")
            
            # Generate samples at the end of each epoch
            self.generate_samples(sample_prompts)
            
            # Plot training history
            self.plot_training_history()
            
            # Resume training
            self.model.train()
        
        # End of training
        print(f"Training completed after {self.current_epoch} epochs")
        
        # Save final model
        self.save_checkpoint("final_model")
        
        # Plot final training history
        self.plot_training_history()
        
        return self.history
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        # Forward pass
        loss = self.model.training_step(batch, batch_idx)
        
        return loss
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Validation loop
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Forward pass
                loss = self.model.validation_step(batch, batch_idx)
                
                # Accumulate loss
                val_loss += loss.item()
        
        # Calculate average loss
        avg_val_loss = val_loss / len(self.val_dataloader)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def generate_samples(self, prompts, guidance_scale=7.5, num_inference_steps=50):
        """
        Generate samples using the model.
        
        Args:
            prompts: List of text prompts
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Generated images
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create pipeline
        pipeline = CubeDiffPipeline(self.model, self.device)
        
        # Generate samples for each prompt
        for i, prompt in enumerate(prompts):
            print(f"Generating sample for prompt: {prompt}")
            
            # Generate image
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="equirectangular",
                    height=512,
                    width=1024,
                )
            
            # Convert to PIL image
            if isinstance(output, torch.Tensor):
                # If output is a tensor, convert to PIL image
                if output.ndim == 4:  # Batch of images
                    output = output[0]  # Take first image
                
                # Move to CPU
                output = output.cpu()
                
                # Convert to numpy
                output = output.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                
                # Ensure values are in [0, 1]
                if output.max() > 1.0:
                    output = output / 255.0
                
                # Scale to [0, 255]
                output = (output * 255.0).astype(np.uint8)
                
                # Convert to PIL image
                output = Image.fromarray(output)
            
            # Save image
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            output.save(
                os.path.join(
                    self.output_dir,
                    "samples",
                    f"sample_{self.global_step}_{i}.png"
                )
            )
    
    def save_checkpoint(self, name):
        """
        Save a model checkpoint.
        
        Args:
            name: Name of the checkpoint
        """
        # Create checkpoint directory
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Save model
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", f"{name}.pt")
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'history': self.history,
            },
            checkpoint_path
        )
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if checkpoint['lr_scheduler_state_dict'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        
        # Plot validation loss if available
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate During Training')
        plt.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        plt.savefig(
            os.path.join(
                self.output_dir,
                "plots",
                f"training_history_{self.global_step}.png"
            )
        )
        
        # Close figure
        plt.close()