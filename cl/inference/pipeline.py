import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np

from ..model.architecture import CubeDiffModel
from ..data.preprocessing import cubemap_to_equirect

class CubeDiffPipeline:
    """
    Inference pipeline for CubeDiff model.
    """
    def __init__(
        self,
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        checkpoint_path=None,
        device="cuda",
        strict_loading=True
    ):
        self.device = device
        
        # Load base SD pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.pipeline.to(device)
        
        # Extract components
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        
        # Set up scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler",
        )
        self.scheduler.set_timesteps(50)  # Use 50 steps by default
        
        # Initialize CubeDiff model
        self.model = CubeDiffModel(pretrained_model_name)
        
        # Load checkpoint if provided
        # if checkpoint_path:
        #     self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            try:
                self.model.load_state_dict(state_dict, strict=strict_loading)
            except RuntimeError as e:
                # If strict loading fails, try with a more permissive approach
                if strict_loading:
                    print(f"Strict loading failed, attempting non-strict loading: {e}")
                    self.model.load_state_dict(state_dict, strict=False)
                    print("Non-strict loading succeeded")
                else:
                    # If non-strict loading also fails, raise the error
                    raise e
                
        self.model.to(device)
        self.model.eval()


    
    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
        output_type="pil",
    ):
        """
        Generate a panorama from a text prompt.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            height: Height of each cubemap face
            width: Width of each cubemap face
            output_type: Output type, one of ["pil", "np", "latent"]
            
        Returns:
            Generated panorama (equirectangular format)
        """
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Encode text
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        
        # Get unconditional embeddings for guidance
        if guidance_scale > 1.0:
            uncond_input = self.tokenizer(
                [negative_prompt or ""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
            
            # For classifier-free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Generate random latent vectors for each face
        num_faces = 6
        latents = torch.randn(
            (1, num_faces, 4, height // 8, width // 8),
            device=self.device,
            dtype=text_embeddings.dtype,
        )
        
        # Denoise latents
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = latents.repeat(2, 1, 1, 1, 1) if guidance_scale > 1.0 else latents
            
            # Get model prediction
            with torch.no_grad():
                noise_pred = self.model(
                    latent_model_input,
                    torch.tensor([t] * latent_model_input.shape[0], device=self.device),
                    text_embeddings,
                )
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        with torch.no_grad():
            cube_faces = []
            for i in range(num_faces):
                face_latent = latents[0, i]
                face_image = self.vae.decode(face_latent / 0.18215).sample
                cube_faces.append(face_image)
            
            # Stack cube faces
            cube_faces = torch.stack(cube_faces, dim=0)  # [6, 3, H, W]
        
        # Convert to PIL images
        if output_type == "latent":
            return latents
        
        # Normalize and convert to numpy
        cube_faces = (cube_faces / 2 + 0.5).clamp(0, 1)
        cube_faces = cube_faces.cpu().permute(0, 2, 3, 1).numpy()  # [6, H, W, 3]
        
        # Convert to equirectangular
        equirect = cubemap_to_equirect(cube_faces, height * 2, width * 4)
        
        if output_type == "np":
            return equirect
        
        # Convert to PIL image
        equirect = (equirect * 255).astype(np.uint8)
        equirect_pil = Image.fromarray(equirect)
        
        return equirect_pil