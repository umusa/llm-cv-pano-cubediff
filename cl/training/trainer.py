import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from tqdm.auto import tqdm
import wandb

from ..model.architecture import CubeDiffModel
from ..data.dataset import CubemapDataset
from .lora import add_lora_to_model

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
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # Set up model and noise scheduler
        self.setup_model()
        
    def setup_model(self):
        """
        Set up model, tokenizer, and noise scheduler.
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
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Set up noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name,
            subfolder="scheduler",
        )
        
        # Initialize CubeDiff model
        self.model = CubeDiffModel(self.pretrained_model_name)
        
        # Add LoRA for efficient fine-tuning
        self.lora_params = add_lora_to_model(
            self.model.base_unet,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"LoRA parameters: {sum(p.numel() for p in self.lora_params):,}")
    
    def prepare_dataloaders(self, train_dataset, val_dataset=None):
        """
        Prepare training and validation dataloaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Dataloaders prepared with Accelerator
        """
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        
        return train_dataloader, val_dataloader
    
    def train(self, train_dataset, val_dataset=None, num_train_epochs=30000):
        """
        Train the CubeDiff model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_train_epochs: Number of training epochs
        """
        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders(train_dataset, val_dataset)
        
        # Initialize optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
        
        # Initialize wandb for tracking (optional)
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
            )
        
        # Training loop
        global_step = 0
        for epoch in range(num_train_epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
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
                        text_embeddings = self.text_encoder(tokens)[0]
                    
                    # Process images to latent space
                    with torch.no_grad():
                        # Encode each face to latent space
                        latents = []
                        for i in range(faces.shape[1]):
                            face_latent = self.vae.encode(faces[:, i]).latent_dist.sample()
                            face_latent = face_latent * 0.18215  # Scale factor for SD
                            latents.append(face_latent)
                        
                        # Stack latents along face dimension
                        latents = torch.stack(latents, dim=1)  # [batch, 6, C, H, W]
                    
                    # Add noise to latents
                    noise = torch.randn_like(latents)
                    
                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=self.accelerator.device,
                    )
                    
                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get model prediction
                    noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                    
                    # Compute loss
                    if self.config.prediction_type == "epsilon":
                        target = noise
                    elif self.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
                    
                    loss = torch.nn.functional.mse_loss(noise_pred, target)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            self.config.max_grad_norm,
                        )
                    
                    # Update model parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Log progress
                progress_bar.set_postfix(loss=loss.item())
                
                # Log metrics
                if self.config.use_wandb and global_step % self.config.log_every_n_steps == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    })
                
                # Save model checkpoint
                if global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
                
                # Evaluate model
                if val_dataloader and global_step % self.config.eval_every_n_steps == 0:
                    self.evaluate(val_dataloader)
                
                global_step += 1
                
                # Break if maximum steps reached
                if global_step >= num_train_epochs:
                    break
            
            # Break if maximum steps reached
            if global_step >= num_train_epochs:
                break
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, "final_model"))
        
        return global_step
    
    def evaluate(self, val_dataloader):
        """
        Evaluate the model on validation set.
        
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
                text_embeddings = self.text_encoder(tokens)[0]
                
                # Process images to latent space
                latents = []
                for i in range(faces.shape[1]):
                    face_latent = self.vae.encode(faces[:, i]).latent_dist.sample()
                    face_latent = face_latent * 0.18215  # Scale factor for SD
                    latents.append(face_latent)
                
                # Stack latents along face dimension
                latents = torch.stack(latents, dim=1)  # [batch, 6, C, H, W]
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.accelerator.device,
                )
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get model prediction
                noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                
                # Compute loss
                if self.config.prediction_type == "epsilon":
                    target = noise
                elif self.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
                
                loss = torch.nn.functional.mse_loss(noise_pred, target)
                total_loss += loss.item()
        
        # Compute average loss
        avg_loss = total_loss / len(val_dataloader)
        
        # Log metrics
        if self.config.use_wandb:
            wandb.log({"val/loss": avg_loss})
        
        print(f"Validation loss: {avg_loss:.4f}")
        
        self.model.train()
        
        return avg_loss
    
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
            torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            
            # Save config
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                import json
                json.dump(self.config.__dict__, f, indent=2)
            
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