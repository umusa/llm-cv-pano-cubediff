import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


class CubeDiffInference:
    def __init__(self, vae, unet, text_encoder, tokenizer, scheduler, device="cuda"):
        """
        Initialize the CubeDiff inference pipeline.
        
        Args:
            vae: VAE model for encoding/decoding images
            unet: UNet model for diffusion
            text_encoder: Text encoder model
            tokenizer: Tokenizer for text processing
            scheduler: Diffusion scheduler
            device: Device to run inference on (cuda or cpu)
        """
        self.device = device
        
        # Store components
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        
        # Set evaluation mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        print("CubeDiff pipeline initialized successfully")
    
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None, return_faces=True):
        """
        Generate a panorama from a text prompt using a memory-efficient approach.
        
        Args:
            prompt (str): Text prompt for image generation
            num_inference_steps (int): Number of diffusion steps
            guidance_scale (float): Guidance scale for classifier-free guidance
            seed (int, optional): Random seed for reproducibility
            return_faces (bool): Whether to return individual faces or combined panorama
        
        Returns:
            torch.Tensor: Generated panorama or faces
        """
        print(f"Starting panorama generation for prompt: '{prompt}'")
        
        # Set seed for reproducibility if provided
        if seed is not None:
            print(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Encode the prompt to get text embeddings
        print("Encoding text prompt...")
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input = text_input.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        # For classifier-free guidance, we need an unconditional embedding (empty prompt)
        if guidance_scale > 1.0:
            print(f"Setting up classifier-free guidance with scale: {guidance_scale}")
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input = uncond_input.to(self.device)
            
            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
            
            # Concatenate unconditional and conditional embeddings
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Memory-efficient approach: Process each face independently
        print("Using memory-efficient approach, processing each face independently...")
        batch_size = 1
        height = width = 96  # Assuming 96×96 resolution per face
        channels = self.unet.config.in_channels
        
        # Create storage for the final result
        all_face_tensors = []
        
        # Process each face separately to save memory
        for face_idx in range(6):
            print(f"Generating face {face_idx+1}/6")
            
            # Generate random noise for this face
            face_latent = torch.randn(
                batch_size, 
                channels, 
                height, 
                width
            ).to(self.device)
            
            # Set up the diffusion scheduler
            if face_idx == 0:
                print(f"Setting up diffusion process with {num_inference_steps} steps...")
                self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            
            # Diffusion process for this face
            print(f"Starting diffusion process for face {face_idx+1}...")
            total_steps = len(self.scheduler.timesteps)
            
            # Clear CUDA cache to free up memory
            torch.cuda.empty_cache()
            
            for i, t in enumerate(self.scheduler.timesteps):
                # Print progress
                if i % 10 == 0 or i == total_steps - 1:
                    print(f"Face {face_idx+1}, diffusion step {i+1}/{total_steps} ({((i+1)/total_steps*100):.1f}%)")
                
                # Prepare latent model input
                if guidance_scale > 1.0:
                    # For classifier-free guidance, repeat latents
                    latent_model_input = torch.cat([face_latent] * 2)
                    timestep_batch = torch.cat([t.unsqueeze(0)] * 2).to(self.device)
                else:
                    latent_model_input = face_latent
                    timestep_batch = t.unsqueeze(0).to(self.device)
                
                # Process this face through the UNet
                unet_output = self.unet(
                    latent_model_input, 
                    timestep_batch, 
                    encoder_hidden_states=text_embeddings
                )
                noise_pred = unet_output.sample
                
                # Handle classifier-free guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Update latents with scheduler step
                face_latent = self.scheduler.step(noise_pred, t, face_latent).prev_sample
            
            # Decode the latent for this face
            print(f"Decoding face {face_idx+1}/6")
            with torch.no_grad():
                face_tensor = self.decode_latent_face(face_latent)
                all_face_tensors.append(face_tensor)
                
                # Clear CUDA cache after processing each face
                del face_latent
                torch.cuda.empty_cache()
        
        # Stack all face tensors together
        print("Combining all faces...")
        with torch.no_grad():
            # Move tensors to CPU first to save GPU memory
            cpu_tensors = [tensor.cpu() for tensor in all_face_tensors]
            # Stack along a new dimension to get [batch_size, 6, C, H, W]
            images = torch.stack(cpu_tensors, dim=1)
        
        print("Image generation complete.")
        
        # Return the individual faces
        if return_faces:
            print("Returning individual faces.")
            return images
        else:
            print("Converting faces to panorama.")
            # For now, just return the individual faces
            return images
    
    def decode_latent_face(self, latent):
        """
        Decode a single face latent to an image tensor.
        
        Args:
            latent (torch.Tensor): Latent tensor for a single face
            
        Returns:
            torch.Tensor: Decoded image tensor
        """
        # Scale the latents (required for the VAE)
        latent = 1 / 0.18215 * latent
        
        # Decode the latent with the VAE
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        
        # Process the image but keep as tensor
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image  # Returns tensor of shape [1, 3, H, W]