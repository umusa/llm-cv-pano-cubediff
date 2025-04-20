import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
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
            vae (AutoencoderKL): VAE model for encoding/decoding images
            unet (UNet2DConditionModel): UNet model for diffusion
            text_encoder (CLIPTextModel): Text encoder model
            tokenizer (CLIPTokenizer): Tokenizer for text processing
            scheduler (DDIMScheduler): Diffusion scheduler
            device (str): Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        
        # Choose half precision if on GPU, else float32 on CPU
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Store components
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        
        # Put all models in eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        # Move them to the selected device *and* dtype
        self.text_encoder.to(device=self.device, dtype=self.dtype)
        self.vae.to(device=self.device, dtype=self.dtype)
        self.unet.to(device=self.device, dtype=self.dtype)

        # Use the new attention processor
        self.unet.set_attn_processor(AttnProcessor2_0())

        # Enable xFormers memory‐efficient attention (if installed)
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("xFormers memory‐efficient attention is enabled.")
        except Exception as e:
            print(f"Could not enable xFormers memory‐efficient attention: {e}")

        # Enable attention slicing (splits up the attention computation to reduce peak memory)
        try:
            # Some diffusers versions call this API differently. 
            # For modern versions: unet.set_attention_slice("auto") or pipeline.enable_attention_slicing("auto")
            self.unet.set_attention_slice("auto")
            print("Attention slicing is enabled (auto).")
        except Exception as e:
            print(f"Could not enable attention slicing: {e}")
        
        # Optionally slice VAE decoding as well (helps if decoding is large):
        try:
            self.vae.enable_slicing()
            print("VAE slicing is enabled.")
        except Exception as e:
            # Not all diffusers versions have VAE slicing
            print(f"Could not enable VAE slicing: {e}")

        # Print model parameter counts
        print(f"VAE parameters:  {sum(p.numel() for p in self.vae.parameters()):,}")
        print(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"Text Encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        
        print("CubeDiff pipeline initialized successfully")
    
    def generate(
        self, 
        prompt, 
        num_inference_steps=50, 
        guidance_scale=7.5, 
        seed=None, 
        return_faces=True
    ):
        """
        Generate a panorama from a text prompt using a memory-efficient approach.
        
        Args:
            prompt (str): Text prompt for image generation
            num_inference_steps (int): Number of diffusion steps
            guidance_scale (float): Guidance scale for classifier-free guidance
            seed (int, optional): Random seed for reproducibility
            return_faces (bool): Whether to return individual faces or the combined panorama
        
        Returns:
            torch.Tensor: Generated panorama or faces
        """
        print(f"Starting panorama generation for prompt: '{prompt}'")
        
        # Set seed for reproducibility if provided
        if seed is not None:
            print(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # We wrap the entire generation in no_grad + autocast to minimize memory usage
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device == "cuda")):

            # 1) Tokenize & encode the prompt -> text embeddings
            print("Encoding text prompt...")
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input = text_input.to(self.device)
            
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

            # 2) For classifier-free guidance, create an unconditional embedding
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
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
                
                # Concatenate unconditional and conditional embeddings
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            
            # 3) We generate 6 "faces" of a panorama, each 96×96 by default
            print("Using memory-efficient approach, processing each face independently...")
            batch_size = 1
            height = width = 96  # face resolution
            channels = self.unet.config.in_channels
            
            # Create storage for the final result
            all_face_tensors = []
            
            # 4) Process each face separately
            for face_idx in range(6):
                print(f"\nGenerating face {face_idx+1}/6")

                # Clear CUDA cache before each face to help with memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Generate random noise for this face
                face_latent = torch.randn(
                    batch_size, 
                    channels, 
                    height, 
                    width,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Set up the diffusion scheduler once
                if face_idx == 0:
                    print(f"Setting scheduler with {num_inference_steps} steps...")
                    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
                
                print(f"Starting diffusion process for face {face_idx+1}...")
                total_steps = len(self.scheduler.timesteps)
                
                for i, t in enumerate(self.scheduler.timesteps):
                    if i % 10 == 0 or i == total_steps - 1:
                        print(f"Face {face_idx+1}, step {i+1}/{total_steps} "
                              f"({(i+1)/total_steps*100:.1f}%)")

                    # Classifier-free guidance => latents must be repeated
                    if guidance_scale > 1.0:
                        latent_model_input = torch.cat([face_latent] * 2)
                    else:
                        latent_model_input = face_latent

                    # Forward pass
                    unet_out = self.unet(
                        latent_model_input, 
                        t, 
                        encoder_hidden_states=text_embeddings
                    )
                    noise_pred = unet_out.sample
                    
                    # Guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Scheduler step
                    face_latent = self.scheduler.step(noise_pred, t, face_latent).prev_sample

                # Decode the latent for this face
                print(f"Decoding face {face_idx+1}/6 ...")
                face_tensor = self.decode_latent_face(face_latent)
                all_face_tensors.append(face_tensor)
                
                # Clean up
                del face_latent
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # 5) Combine all faces on CPU
        print("\nCombining all faces into a single tensor...")
        cpu_tensors = [tensor.cpu() for tensor in all_face_tensors]
        # shape => [batch_size, 6, 3, H, W]
        images = torch.stack(cpu_tensors, dim=1)
        
        print("Image generation complete!")
        if return_faces:
            print("Returning individual face tensors (6 faces).")
            return images
        else:
            print("Returning stacked faces (still individual, but shape [1,6,3,H,W]).")
            return images
    
    def decode_latent_face(self, latent):
        """
        Decode a single face latent to an image tensor.
        
        Args:
            latent (torch.Tensor): Latent tensor for a single face
            
        Returns:
            torch.Tensor: Decoded image tensor in [1, 3, H, W], range [0,1]
        """
        # VAE scaling factor
        latent = 1 / 0.18215 * latent
        
        image = self.vae.decode(latent).sample  # shape: [1, 3, H, W]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
