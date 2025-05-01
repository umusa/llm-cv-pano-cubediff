"""
Trainer class for CubeDiff model.
"""

import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from einops import rearrange

from cubediff_utils import add_cubemap_positional_encodings


class CubeDiffTrainer_bk:
    """
    Trainer class for CubeDiff model.
    
    Args:
        vae: VAE model with synchronized GroupNorm
        unet: UNet model with inflated attention
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save model checkpoints
        mixed_precision: Whether to use mixed precision training
        device: Device to use for training
    """
    def __init__(
        self,
        vae,
        unet,
        text_encoder,
        tokenizer,
        noise_scheduler,
        learning_rate=1e-5,
        output_dir="./output",
        mixed_precision=True,
        device="cuda"
    ):
        self.vae = vae.to(device)
        self.unet = unet.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Only train attention layers in UNet to prevent overfitting
        for name, param in self.unet.named_parameters():
            if "attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Setup optimizer with parameter groups
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.unet.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        
        # Setup gradient scaler for mixed precision training
        self.scaler = GradScaler(enabled=mixed_precision)
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=learning_rate * 0.1
        )
    
    def train_step(self, batch, conditioning_face_idx=0):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data from the dataloader
            conditioning_face_idx: Index of the face to use as conditioning (0=front)
            
        Returns:
            Dictionary of loss values
        """
        # Get data from batch
        cubemap_images = batch["cubemap"].to(self.device)
        prompts = batch["prompt"]
        
        batch_size = len(prompts)
        
        # Encode text
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        # Add unconditional embeddings for classifier-free guidance during inference
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        
        # Concatenate conditional and unconditional embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Reshape cubemap for processing
        # [B, 6, C, H, W] -> [B*6, C, H, W]
        cubemap_flat = rearrange(cubemap_images, 'b f c h w -> (b f) c h w')
        
        # Encode images to latents
        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                latents = self.vae.encode(cubemap_flat).latent_dist.sample() * 0.18215
        
        # Create conditioning mask
        cond_mask = torch.zeros_like(latents[:, :1, :, :])
        for i in range(batch_size):
            cond_mask[i * 6 + conditioning_face_idx] = 1.0
        
        # Add positional encodings
        latents = add_cubemap_positional_encodings(latents, batch_size)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=self.device
        )
        # Repeat timesteps for all 6 faces of each panorama
        timesteps = timesteps.repeat_interleave(6)
        
        # Add noise to latents based on timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Concatenate conditioning mask
        model_input = torch.cat([noisy_latents, cond_mask], dim=1)
        
        # Forward pass
        with autocast(enabled=self.mixed_precision):
            # Predict noise
            noise_pred = self.unet(
                model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # L2 loss between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)
        
        # Backpropagate and optimize
        self.optimizer.zero_grad()
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.optimizer.step()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return {"loss": loss.item()}
    
    def save_checkpoint(self, step):
        """
        Save model checkpoint.
        
        Args:
            step: Current training step
        """
        # Create checkpoint directory
        os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)
        
        # Save UNet (only attention layers were trained)
        self.unet.save_pretrained(f"{self.output_dir}/checkpoints/unet_{step}")
        
        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.mixed_precision else None,
                "step": step
            },
            f"{self.output_dir}/checkpoints/optimizer_{step}.pt"
        )
    
    def train(self, dataloader, num_epochs, save_every=1000):
        """
        Train the model.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N steps
            
        Returns:
            Dictionary of training metrics
        """
        total_steps = len(dataloader) * num_epochs
        global_step = 0
        losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            self.unet.train()
            epoch_losses = []
            
            # Create progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Train step
                loss_dict = self.train_step(batch)
                loss = loss_dict["loss"]
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss)
                
                # Save metrics
                epoch_losses.append(loss)
                losses.append(loss)
                
                # Save checkpoint
                if global_step % save_every == 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
            
            # Print epoch metrics
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save final model for this epoch
            self.save_checkpoint(global_step)
        
        return {"losses": losses}