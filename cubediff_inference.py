"""
Inference classes for CubeDiff model with enhanced image quality.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from cubediff_utils import add_cubemap_positional_encodings, cubemap_to_equirect


class CubeDiffInference:
    """
    Enhanced inference class for CubeDiff model.
    
    Args:
        vae: VAE model with synchronized GroupNorm
        unet: UNet model with inflated attention
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        scheduler: Noise scheduler for sampling
        device: Device to use for inference
    """
    def __init__(
        self,
        vae,
        unet,
        text_encoder,
        tokenizer,
        scheduler,
        device="cuda"
    ):
        self.vae = vae.to(device)
        self.unet = unet.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        
        # Set models to evaluation mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        # Set precision - use half precision for faster inference if available
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda":
            self.vae.to(dtype=self.dtype)
            self.unet.to(dtype=self.dtype)
        
        # Print model parameters to verify setup
        print(f"VAE parameters: {sum(p.numel() for p in self.vae.parameters())}")
        print(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters())}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt="blurry, ugly, distorted, low quality, low resolution, bad anatomy, worst quality, unrealistic, text, watermark",
        conditioning_image=None,
        height=768,  # Increased resolution
        width=768,   # Increased resolution
        num_inference_steps=100,  # More steps for better quality
        guidance_scale=9.5,  # Increased for better prompt adherence
        seed=None,
        return_faces=True,
        use_pos_encodings=True  # Explicitly control positional encodings
    ):
        """
        Generate cubemap and panorama from text prompt with enhanced quality.
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Calculate latent dimensions
        latent_height, latent_width = height // 8, width // 8
        
        # Process prompt by adding details to enhance quality
        enhanced_prompt = self._enhance_prompt(prompt)
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        # Encode text condition
        text_input = self.tokenizer(
            [enhanced_prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode negative prompt
        uncond_input = self.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            cond_embeddings = self.text_encoder(text_input.input_ids)[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
            
            # Make sure embeddings have the correct dtype
            cond_embeddings = cond_embeddings.to(dtype=self.dtype)
            uncond_embeddings = uncond_embeddings.to(dtype=self.dtype)
        
        # Full batch of embeddings for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        
        # Initialize 6 faces of random noise for the cubemap
        # The 6 faces are in order: [front, right, back, left, top, bottom]
        latents = torch.randn(
            (6, 4, latent_height, latent_width),
            device=self.device,
            dtype=self.dtype
        )
        
        # Process conditioning image if provided
        cond_mask = torch.zeros((6, 1, latent_height, latent_width), device=self.device, dtype=self.dtype)
        if conditioning_image is not None:
            if isinstance(conditioning_image, Image.Image):
                conditioning_image = transforms.Compose([
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])(conditioning_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                conditioning_latent = self.vae.encode(conditioning_image).latent_dist.sample() * 0.18215
            
            latents[0] = conditioning_latent[0]
            cond_mask[0] = 1.0
        
        # Calculate positional encodings for cross-face awareness
        if use_pos_encodings:
            pos_encodings = add_cubemap_positional_encodings(latents)
            orig_latents = latents.clone()
            latents = pos_encodings  # Use the full tensor with positional encodings
            print(f"Using positional encodings. Shape: {latents.shape}")
        
        # Scale latents (magic number from Stable Diffusion)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            # For UNet input, separate latents and positional encodings
            if use_pos_encodings and latents.shape[1] > 4:
                latent_model_input = latents[:, :4]  # First 4 channels are the actual latents
                pos_channels = latents[:, 4:]  # Remaining channels are positional encodings
            else:
                latent_model_input = latents
            
            # Expand latents for classifier-free guidance
            # [6, 4, H, W] -> [12, 4, H, W]
            latent_model_input = torch.cat([latent_model_input] * 2)
            
            # Create timestep tensor
            timesteps = torch.full(
                (latent_model_input.shape[0],),
                t,
                device=self.device,
                dtype=torch.long
            )
            
            # Forward through UNet - use safe autocast approach
            noise_pred = None
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    noise_pred = self.unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample (only on the latent part if using positional encodings)
            if use_pos_encodings and latents.shape[1] > 4:
                # Update only the latent part, keep positional encodings constant
                updated_latents = self.scheduler.step(noise_pred, t, latents[:, :4]).prev_sample
                latents = torch.cat([updated_latents, pos_channels], dim=1)
            else:
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Extract actual latents if using positional encodings
        if use_pos_encodings and latents.shape[1] > 4:
            latents = latents[:, :4]
        
        # Decode latents
        # Scale latents for VAE
        latents = 1 / 0.18215 * latents
        
        # Process each face with VAE to get images
        faces = []
        for i in range(6):
            face_latent = latents[i:i+1]
            
            # Safe decoding with proper autocast handling
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    face_image = self.vae.decode(face_latent).sample
            else:
                face_image = self.vae.decode(face_latent).sample
                
            faces.append(face_image)
        
        # Stack all faces
        faces = torch.cat(faces, dim=0)
        
        # Convert to pixel values [0, 1]
        faces = (faces / 2 + 0.5).clamp(0, 1)
        
        # Move to CPU
        faces_np = faces.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Generate panorama with improved blending
        print("Converting cubemap to equirectangular panorama...")
        panorama = cubemap_to_equirect(faces_np, output_height=height*2, output_width=width*4)
        
        # Convert to PIL
        faces_pil = [Image.fromarray((face * 255).astype(np.uint8)) for face in faces_np]
        panorama_pil = Image.fromarray((panorama * 255).astype(np.uint8))
        
        if return_faces:
            return {"panorama": panorama_pil, "faces": faces_pil}
        else:
            return {"panorama": panorama_pil}
    
    def _enhance_prompt(self, prompt):
        """
        Enhance a basic prompt with additional details for better image quality.
        """
        # If the prompt already seems detailed enough, return it as is
        if len(prompt.split()) > 15:
            return prompt
        
        # Extract key elements
        if "mountain" in prompt.lower() and "lake" in prompt.lower():
            return f"Highly detailed professional photograph of {prompt}. 8k resolution, photorealistic, masterpiece, sharp focus, dramatic lighting, trending on artstation"
        elif "landscape" in prompt.lower():
            return f"Breathtaking professional landscape photograph of {prompt}. 8k resolution, photorealistic, masterpiece, sharp focus, hyperdetailed"
        else:
            return f"Ultra detailed professional photograph of {prompt}. 8k resolution, photorealistic, masterpiece, perfect composition, dramatic lighting"