"""
model.py - Implementation of the CubeDiff model architecture

This module contains the complete CubeDiff model implementation, including the
latent diffusion model, VAE integration, and text-conditioning components.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Import custom modules
from modules import (
    CubemapPositionalEncoding,
    GroupNormalizationSync,
    InflatedAttention,
    OverlappingEdgeProcessor,
    adapt_unet_for_cubemap
)

# Import conversion utilities
from cubediff_utils_v1 import improved_equirect_to_cubemap, optimized_cubemap_to_equirect


class CubeDiff(nn.Module):
    """
    CubeDiff model for generating high-quality 360° panoramas from text prompts
    or normal field-of-view (NFoV) images.
    
    This implements the architecture described in the paper:
    "CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation"
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base",
        use_fp16=True,
        device=None,
    ):
        """
        Initialize the CubeDiff model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            use_fp16: Whether to use mixed precision
            device: Device to use (default: auto-detect)
        """
        super().__init__()
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Set precision
        self.dtype = torch.float16 if use_fp16 else torch.float32
        print(f"Initializing CubeDiff with {self.dtype} precision on {self.device}")
        
        # Stage 1: Load pretrained components
        print("Loading pretrained model components...")
        self._load_pretrained_components(pretrained_model_name_or_path)
        
        # Stage 2: Adapt the UNet for cubemap inputs
        print("Adapting UNet for cubemap processing...")
        self.unet = adapt_unet_for_cubemap(self.unet)
        
        # Stage 3: Add cubemap-specific modules
        print("Adding cubemap-specific modules...")
        latent_channels = self.vae.config.latent_channels
        self.cubemap_pos_encoding = CubemapPositionalEncoding(latent_channels)
        self.group_norm_sync = GroupNormalizationSync(4, latent_channels)  # Use 4 or 2 or 1 as the group count
        self.edge_processor = OverlappingEdgeProcessor(overlap_size=4)
        
        # Stage 4: Setup model for training/inference
        self._setup_model()
        
        # Stage 5: Move model to device
        self.to(device)
        
        print("CubeDiff model initialization complete!")
    
    def _load_pretrained_components(self, pretrained_model_name_or_path):
        """
        Load pretrained model components.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
        """
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="tokenizer"
        )
        print("✓ Loaded tokenizer")
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )
        # Freeze text encoder as it doesn't need to be fine-tuned
        self.text_encoder.requires_grad_(False)
        print("✓ Loaded text encoder (frozen)")
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="vae"
        )
        # Freeze VAE as it doesn't need to be fine-tuned
        self.vae.requires_grad_(False)
        print("✓ Loaded VAE (frozen)")
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="unet"
        )
        print("✓ Loaded UNet")
        
        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        print("✓ Loaded scheduler")
    
    def _setup_model(self):
        """
        Setup the model for training or inference.
        """
        # Ensure only the attention layers in the UNet are trainable
        for name, param in self.unet.named_parameters():
            if "attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Set the model to training mode
        self.train()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    
    def encode_text(self, text_prompts: List[str]):
        """
        Encode text prompts to text embeddings.
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            Text embeddings tensor
        """
        # Tokenize text
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        text_inputs = text_inputs.to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        return text_embeddings
    
    def encode_image_to_latents(self, image):
        """
        Encode an image to latent space using the VAE.
        
        Args:
            image: Image tensor of shape (B, 3, H, W) or (B, 6, 3, H, W) for cubemap
            
        Returns:
            Latent tensor
        """
        with torch.no_grad():
            # Handle cubemap input with 6 faces
            if len(image.shape) == 5 and image.shape[1] == 6:
                # Process each face separately
                batch_size = image.shape[0]
                num_faces = image.shape[1]
                face_latents = []
                
                for face_idx in range(num_faces):
                    # Get face and ensure it's in range [-1, 1]
                    face = image[:, face_idx]
                    if face.min() >= 0 and face.max() <= 1:
                        face = 2 * face - 1
                    
                    # Encode face
                    face_latent = self.vae.encode(face).latent_dist.sample()
                    face_latent = face_latent * self.vae.config.scaling_factor
                    face_latents.append(face_latent)
                
                # Stack along face dimension
                latents = torch.stack(face_latents, dim=1)
            else:
                # Handle normal image input (single face)
                # Ensure image is in range [-1, 1]
                if image.min() >= 0 and image.max() <= 1:
                    image = 2 * image - 1
                
                # Encode image
                latents = self.vae.encode(image).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def decode_latents_to_image(self, latents):
        """
        Decode latents to images using the VAE.
        
        Args:
            latents: Latent tensor of shape (B, 4, H, W) or (B, 6, 4, H, W) for cubemap
            
        Returns:
            Decoded image tensor
        """
        with torch.no_grad():
            # Handle cubemap latents with 6 faces
            if len(latents.shape) == 5 and latents.shape[1] == 6:
                # Process each face separately
                batch_size = latents.shape[0]
                num_faces = latents.shape[1]
                face_images = []
                
                for face_idx in range(num_faces):
                    # Get face latents and scale
                    face_latent = latents[:, face_idx]
                    face_latent = face_latent / self.vae.config.scaling_factor
                    
                    # Decode face
                    face_image = self.vae.decode(face_latent).sample
                    # Convert to range [0, 1]
                    face_image = (face_image + 1) / 2
                    face_image = torch.clamp(face_image, 0, 1)
                    face_images.append(face_image)
                
                # Stack along face dimension
                images = torch.stack(face_images, dim=1)
            else:
                # Handle normal latents (single image)
                # Scale latents
                latents = latents / self.vae.config.scaling_factor
                
                # Decode latents
                images = self.vae.decode(latents).sample
                # Convert to range [0, 1]
                images = (images + 1) / 2
                images = torch.clamp(images, 0, 1)
        
        return images
    
    def prepare_latents(
        self,
        batch_size,
        num_faces=6,
        height=512,
        width=512,
        generator=None,
        latents=None,
    ):
        """
        Prepare initial random noise as latents.
        
        Args:
            batch_size: Number of samples in the batch
            num_faces: Number of faces in the cubemap
            height: Height of the output image
            width: Width of the output image
            generator: Optional random generator for reproducibility
            latents: Optional pre-generated latents
            
        Returns:
            Initial latents for the diffusion process
        """
        # Height and width in the latent space
        latent_height = height // 8
        latent_width = width // 8
        
        # Generate random latents if not provided
        if latents is None:
            # Standard normal distribution
            shape = (batch_size, num_faces, self.vae.config.latent_channels, latent_height, latent_width)
            latents = torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype
            )
        else:
            # Check if the provided latents have the right shape
            if latents.shape[1:] != (num_faces, self.vae.config.latent_channels, latent_height, latent_width):
                raise ValueError(f"Provided latents have an incorrect shape: {latents.shape}")
            # Ensure the latents are on the correct device and have the right dtype
            latents = latents.to(device=self.device, dtype=self.dtype)
        
        return latents
    
    def prepare_mask(self, batch_size, condition_face=0, num_faces=6):
        """
        Prepare a mask for conditional generation.
        
        Args:
            batch_size: Number of samples in the batch
            condition_face: Index of the face to condition on
            num_faces: Number of faces in the cubemap
            
        Returns:
            Binary mask tensor
        """
        # Create a mask tensor with shape (batch_size, num_faces)
        mask = torch.zeros(batch_size, num_faces, dtype=torch.bool, device=self.device)
        
        # Set the conditioning face to True
        mask[:, condition_face] = True
        
        return mask
    
    def denoise_latents(
        self,
        latents,
        text_embeddings,
        condition_mask=None,
        condition_latents=None,
        timesteps=None,
        guidance_scale=7.5,
    ):
        """
        Perform the denoising diffusion process.
        
        Args:
            latents: Initial noisy latents
            text_embeddings: Text embeddings for conditioning
            condition_mask: Optional mask indicating which faces are conditioning
            condition_latents: Optional latents to use as conditioning
            timesteps: Optional list of specific timesteps to use
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Denoised latents
        """
        # Set timesteps
        if timesteps is None:
            self.scheduler.set_timesteps(50, device=self.device)
            timesteps = self.scheduler.timesteps
        
        # Apply cubemap-specific processing to initial latents
        latents = self.cubemap_pos_encoding(latents)
        latents = self.group_norm_sync(latents)
        latents = self.edge_processor(latents)
        
        # Extract dimensions
        batch_size, num_faces, latent_channels, height, width = latents.shape
        
        # Prepare text embeddings for classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Duplicate text embeddings for conditional and unconditional paths
        if do_classifier_free_guidance:
            # Ensure text_embeddings include unconditional embeddings
            if text_embeddings.shape[0] != batch_size * 2:
                # Create unconditional embeddings
                uncond_embeddings = self.encode_text([""] * batch_size)
                # Concatenate with conditional embeddings
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare masking for conditional generation if needed
        if condition_mask is not None and condition_latents is not None:
            # Prepare a masked version of the latents
            masked_latents = latents.clone()
            
            # Replace the conditioning faces with the provided latents
            for i in range(batch_size):
                for j in range(num_faces):
                    if condition_mask[i, j]:
                        masked_latents[i, j] = condition_latents[i, j]
        else:
            masked_latents = latents
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand noisy latents for classifier-free guidance
            # latents shape: (B, 6, C, H, W)
            # masked_latents shape: (B, 6, C, H, W)
            latent_model_input = torch.cat([masked_latents] * 2) if do_classifier_free_guidance else masked_latents
            
            # Scale the latents (important for the DDIM scheduler)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Apply cubemap-specific processing TO THE UNET INPUT
            # Note: Check if group_norm_sync should be here or only in VAE
            latent_model_input = self.cubemap_pos_encoding(latent_model_input)
            latent_model_input = self.group_norm_sync(latent_model_input)

            # Reshape for UNet: (B*num_faces, C, H, W)
            # b_times_f, c, h, w = latent_model_input.reshape(-1, latent_channels, height, width).shape
            b_cfg, num_faces_cfg, c, h, w = latent_model_input.shape # Should be B*2, 6, C, H, W
            latent_model_input_reshaped = latent_model_input.view(b_cfg * num_faces_cfg, c, h, w)
            # b_times_f is now b_cfg * num_faces_cfg (e.g., 1*2*6 = 12 for batch size 1)
            b_times_f = latent_model_input_reshaped.shape[0]

            # Ensure text_embeddings batch dim matches latent_model_input_reshaped batch dim
            # text_embeddings shape: (B*2, text_seq, dim)
            text_embeddings_for_unet = text_embeddings
            if text_embeddings.shape[0] != b_times_f:
                # Expected latent batch is B*2*6, expected text batch is B*2
                # We need to repeat text embeddings by num_faces (6)
                num_faces = 6 # Assuming cubemap
                if text_embeddings.shape[0] * num_faces == b_times_f:
                    text_embeddings_for_unet = text_embeddings.repeat_interleave(num_faces, dim=0)
                else:
                    # This case should ideally not happen if inputs are prepared correctly
                    print(f"Warning: Unexpected shape mismatch between latents ({b_times_f}) and text embeddings ({text_embeddings.shape[0]}). Using original text embeddings.")


            # Predict noise
            with torch.no_grad():
                # Forward pass through the UNet
                noise_pred = self.unet(
                    # latent_model_input.reshape(b_times_f, c, h, w),
                    latent_model_input_reshaped, # Use the reshaped version
                    t,
                    # encoder_hidden_states=text_embeddings
                    encoder_hidden_states=text_embeddings_for_unet # Use the potentially repeated embeddings
                ).sample
                
                # Reshape back to (B, num_faces, C, H, W)
                # Reshape noise_pred output back to (B*2, 6, C, H, W) for CFG splitting
                noise_pred = noise_pred.reshape(
                    # -1, num_faces, latent_channels, height, width
                    b_cfg, num_faces_cfg, latent_channels, height, width # Should be B*2, 6, C, H, W
                )
            
            # Apply classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step expects noise_pred and latents to have same shape (B, 6, C, H, W)
            # Current latents shape: (B, 6, C, H, W)
            # Current noise_pred shape after CFG: (B, 6, C, H, W)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Apply cubemap-specific processing
            latents = self.cubemap_pos_encoding(latents)
            latents = self.group_norm_sync(latents)
            latents = self.edge_processor(latents)
            
            # Apply conditioning if needed
            if condition_mask is not None and condition_latents is not None:
                for i in range(batch_size):
                    for j in range(num_faces):
                        if condition_mask[i, j]:
                            latents[i, j] = condition_latents[i, j]
        
        return latents
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt=None,
        input_image=None,
        condition_face=0,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
        output_type="equirectangular",
        height=512,
        width=512,
    ):
        """
        Generate a 360° panorama from a text prompt and optionally an input image.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Optional negative prompt for generation
            input_image: Optional input image for image-to-image generation
            condition_face: Index of the face to condition on (if input_image is provided)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            generator: Optional random generator for reproducibility
            output_type: Type of output ("cubemap" or "equirectangular")
            height: Height of the output image
            width: Width of the output image
            
        Returns:
            Generated panorama
        """
        # Process input prompt
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        # Process negative prompt
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        # Encode text prompts
        text_embeddings = self.encode_text(prompt)
        uncond_embeddings = self.encode_text(negative_prompt)
        
        # Concatenate embeddings for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Prepare initial latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_faces=6,
            height=height,
            width=width,
            generator=generator,
        )
        
        # Process input image if provided
        condition_mask = None
        condition_latents = None
        
        if input_image is not None:
            # Check if input is equirectangular or a single face
            if len(input_image.shape) == 4:  # (B, C, H, W)
                # Convert equirectangular to cubemap
                face_size = height // 4  # Default face size based on height
                
                # Process each image in batch
                cubemap_images = []
                for i in range(input_image.shape[0]):
                    img = input_image[i].permute(1, 2, 0).cpu().numpy()
                    faces = improved_equirect_to_cubemap(img, face_size)
                    
                    # Convert to tensor format (6, 3, H, W)
                    faces_tensor = []
                    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
                        face = torch.tensor(faces[face_name], device=self.device).float()
                        face = face.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                        if face.max() > 1.0:
                            face = face / 255.0
                        faces_tensor.append(face)
                    
                    # Stack faces
                    cubemap = torch.stack(faces_tensor, dim=0)
                    cubemap_images.append(cubemap)
                
                # Stack to batch
                input_cubemap = torch.stack(cubemap_images, dim=0).to(self.device)
                
            elif len(input_image.shape) == 5:  # (B, 6, C, H, W)
                # Already in cubemap format
                input_cubemap = input_image
            
            # Encode image to latent space
            condition_latents = self.encode_image_to_latents(input_cubemap)
            
            # Create conditioning mask
            condition_mask = self.prepare_mask(batch_size, condition_face, num_faces=6)
        
        # Denoise latents with the diffusion model
        latents = self.denoise_latents(
            latents=latents,
            text_embeddings=text_embeddings,
            condition_mask=condition_mask,
            condition_latents=condition_latents,
            timesteps=self.scheduler.timesteps,
            guidance_scale=guidance_scale,
        )
        
        # Decode the latents to images
        cubemap_images = self.decode_latents_to_image(latents)
        
        # Convert to equirectangular if requested
        if output_type == "equirectangular":
            # Process each image in batch
            equirect_images = []
            for i in range(cubemap_images.shape[0]):
                cubemap = cubemap_images[i].cpu().numpy()
                
                # Create dictionary of faces
                faces = {}
                face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
                for j, name in enumerate(face_names):
                    face = cubemap[j].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    # Scale to [0, 255]
                    face = (face * 255.0).astype(np.uint8)
                    faces[name] = face
                
                # Convert to equirectangular
                # equirect = optimized_cubemap_to_equirect(faces, height=height, width=width)
                equirect = optimized_cubemap_to_equirect(faces, H=height, W=width) # Use positional args

                # Convert back to tensor
                equirect = torch.tensor(equirect, device=self.device).float()
                equirect = equirect.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                if equirect.max() > 1.0:
                    equirect = equirect / 255.0
                equirect_images.append(equirect)
            
            # Stack to batch
            result = torch.stack(equirect_images, dim=0)
        else:
            # Return cubemap format
            result = cubemap_images
        
        return result
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data containing:
                  - 'cubemap': Cubemap tensor (B, 6, 3, H, W)
                  - 'caption': Text captions
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        # Extract data from batch
        cubemap = batch['cubemap'].to(self.device)
        captions = batch['caption']
        
        # Set timesteps
        batch_size = cubemap.shape[0]
        noise_scheduler = self.scheduler
        noise_scheduler.set_timesteps(1000)
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device).long()
        
        # Encode cubemap to latent space
        latents = self.encode_image_to_latents(cubemap)
        
        # Apply cubemap-specific processing
        latents = self.cubemap_pos_encoding(latents)
        latents = self.group_norm_sync(latents)
        latents = self.edge_processor(latents)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        text_embeddings = self.encode_text(captions)
        
        # Create conditional and unconditional embeddings for classifier-free guidance
        # This is only needed during training if using classifier-free guidance
        uncond_embeddings = self.encode_text([""] * batch_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Predict noise
        # Reshape for UNet: (B*num_faces, C, H, W)
        b, f, c, h, w = noisy_latents.shape
        model_pred = self.unet(
            noisy_latents.view(b * f, c, h, w),
            timesteps,
            encoder_hidden_states=text_embeddings[batch_size:],  # Use only the conditional embeddings during training
        ).sample
        
        # Reshape back to (B, num_faces, C, H, W)
        model_pred = model_pred.reshape(b, f, c, h, w)
        
        # Calculate loss
        loss = F.mse_loss(model_pred, noise, reduction="mean")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            Optimizer and scheduler
        """
        # Only optimize the trainable parameters (attention layers)
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5)
        
        return optimizer


class CubeDiffPipeline:
    """
    Pipeline for generating panoramic images using the CubeDiff model.
    This provides a convenient interface similar to the diffusers library.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize the pipeline.
        
        Args:
            model: CubeDiff model
            device: Device to use
        """
        self.model = model
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Ensure model is on the correct device
        self.model.to(device)
        
        # Set to eval mode
        self.model.eval()
    
    def __call__(
        self,
        prompt,
        negative_prompt=None,
        input_image=None,
        condition_face=0,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
        output_type="equirectangular",
        height=512,
        width=2048,
    ):
        """
        Generate a panoramic image.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Optional negative prompt for generation
            input_image: Optional input image for image-to-image generation
            condition_face: Index of the face to condition on (if input_image is provided)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            generator: Optional random generator for reproducibility
            output_type: Type of output ("cubemap" or "equirectangular")
            height: Height of the output image
            width: Width of the output image
            
        Returns:
            Generated panoramic image
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Call the generate method of the model
        with torch.no_grad():
            output = self.model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                condition_face=condition_face,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type=output_type,
                height=height,
                width=width,
            )
        
        return output