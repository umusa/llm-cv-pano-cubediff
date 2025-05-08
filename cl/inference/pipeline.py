import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
import time
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
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            try:
                self.model.load_state_dict(state_dict, strict=strict_loading)
                print(f"pipeline.py - CubeDiffPipeline - __init__() - loading checkpoint OK\n")
            except RuntimeError as e:
                # If strict loading fails, try with a more permissive approach
                if strict_loading:
                    print(f"Strict loading failed, attempting non-strict loading: {e}")
                    self.model.load_state_dict(state_dict, strict=False)
                    print("Non-strict loading succeeded")
                else:
                    # If non-strict loading also fails, raise the error
                    print(f"pipeline.py - CubeDiffPipeline - __init__() - checkpoint non-strict loading failed due to {e}\n")
                    raise e
        
        # Move model to device and set to evaluation mode
        self.model.to(device)
        self.model.eval()
        
        # Apply fixes for mixed precision inference
        self._fix_for_mixed_precision()

    def _fix_for_mixed_precision(self):
        """
        Apply fixes to handle mixed precision during inference.
        This ensures consistent dtype handling between FP16 and FP32 tensors.
        """
        # Get model's dtype
        # model_dtype = next(self.vae.parameters()).dtype
        # print(f"Fixing pipeline for inference with dtype: {model_dtype}")
        
        # Fix the model's forward method
        original_forward = self.model.forward
        
        # use the U-Net’s dtype
        unet_dtype = next(self.model.base_unet.parameters()).dtype
        print(f"Fixing pipeline: casting inputs to {unet_dtype}")

        def patched_forward(self_model, latent_model_input, t, encoder_hidden_states=None):
            """
            Ensure consistent dtypes in forward pass
            """
            # Convert inputs to the model's dtype
            # cast inputs to the UNet’s dtype, not the VAE’s:
            latent_model_input = latent_model_input.to(dtype=unet_dtype) # not model_dtype
            # Note: timesteps t should remain int64
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype) # not model_dtype to keep U-Net in float32 and the inputs float32.
            
            # Call the original forward method with type-matched inputs
            return original_forward(latent_model_input, t, encoder_hidden_states)        

        # Replace the forward method
        import types
        self.model.forward = types.MethodType(patched_forward, self.model)
        
        print("pipeline.py - CubePipeline - Pipeline fixed for mixed precision inference")
    
    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
        output_type="pil",
        *,
        preview_hw=(512, 1024),                 # NEW – (h,w) you want to see
        preview_interpolation=Image.BILINEAR # or LANCZOS / NEAREST
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
        # Get model's dtype for consistent precision
        model_dtype = next(self.vae.parameters()).dtype
        
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
        )[0].to(dtype=model_dtype)
        
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
            )[0].to(dtype=model_dtype)
            
            # For classifier-free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Generate random latent vectors for each face
        num_faces = 6
        latents = torch.randn(
            (1, num_faces, 4, height // 8, width // 8),
            device=self.device,
            dtype=model_dtype,
        )
        print(f"pipeline.py - CubeDiffPipeline - generate() - before Denoise latents, latents shape is {latents.shape}, latents type is {type(latents)}\n")

        # Denoise latents
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = latents.repeat(2, 1, 1, 1, 1) if guidance_scale > 1.0 else latents
            
            # Get model prediction
            with torch.no_grad():
                # Use int64 for timesteps as required by the model
                timesteps = torch.tensor([t] * latent_model_input.shape[0], device=self.device, dtype=torch.int64)
                # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - before noise_pred = self.model\n")
                noise_pred = self.model(
                    latent_model_input,
                    timesteps,
                    text_embeddings,
                )
                # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - after model inference, latent_model_input shape is {latent_model_input.shape}, latent_model_input is {latent_model_input}\n")
                # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - text_embeddings shape is {text_embeddings.shape}, text_embeddings is {text_embeddings}\n")
                # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - timesteps shape is {timesteps.shape}, timesteps is {timesteps}\n")
                # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - text_embeddings shape is {text_embeddings.shape}, text_embeddings is {text_embeddings}\n")

            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - noise_pred_text shape is {noise_pred_text.shape}, noise_pred_text is {noise_pred_text}\n")
            # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - noise_pred_uncond shape is {noise_pred_uncond.shape}, noise_pred_uncond is {noise_pred_text}\n")
            # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - guidance_scale is {guidance_scale}\n")
            # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - noise_pred shape is {noise_pred.shape}, noise_pred is {noise_pred}\n")
            
            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # print(f"pipeline.py - CubeDiffPipeline - generate() - Get model prediction - after update latents - latents shape is {latents.shape}, latents is {latents}\n")

        print(f"pipeline.py - CubeDiffPipeline - generate() - after Denoise latents, latents shape is {latents.shape}, latents type is {type(latents)}\n")
        # Decode latents
        with torch.no_grad():
            cube_faces = []
            for i in range(num_faces):
                # face_latent = latents[0, i]
                # with torch.cuda.amp.autocast(enabled=True):
                #     face_image = self.vae.decode(face_latent / 0.18215).sample
                # cube_faces.append(face_image)
                
                # 1) add the missing batch dim
                face_latent = latents[0, i].unsqueeze(0)           # → [1, C, H, W]
            
                # 2) decode with batch dim, then strip it off
                with torch.amp.autocast(enabled=True, device_type="cuda"):
                    out = self.vae.decode(face_latent / 0.18215)
                    sample = out.sample                            # → [1, 3, h, w]
                cube_faces.append(sample[0])                       # → [3, h, w]
            
            # Stack cube faces
            cube_faces = torch.stack(cube_faces, dim=0)  # [6, 3, H, W]
        
        # Convert to PIL images
        if output_type == "latent":
            return latents
        
        # Normalize and convert to numpy
        cube_faces = (cube_faces / 2 + 0.5).clamp(0, 1)
        cube_faces = cube_faces.cpu().permute(0, 2, 3, 1).numpy()  # [6, H, W, 3]
        
        # Convert to equirectangular (panorama)
        print(f"pipeline.py - CubeDiffPipeline - generate() - before cubemap_to_equirect, cube_faces shape is {cube_faces.shape}\n")
        equi_tm = time.time()
        equirect = cubemap_to_equirect(cube_faces, height * 2, width * 4)
        equi_end_tm = time.time()
        print(f"pipeline.py - CubeDiffPipeline - generate() - cubemap_to_equirect cost {equi_end_tm-equi_tm:.2f} seconds\n")

        if output_type == "np":
            return equirect
        
        # Convert to PIL image
        print(f"pipeline.py - CubeDiffPipeline - generate() - before converted to PIL image, equirect shape is {equirect.shape}\n")
        # equirect = (equirect * 255).astype(np.uint8)
        # Fix with tensor method:
        if isinstance(equirect, torch.Tensor):
            equirect = (equirect * 255).to(torch.uint8)
        elif isinstance(equirect, np.ndarray):
            equirect = (equirect * 255).astype(np.uint8)
        print(f"pipeline.py - CubeDiffPipeline - generate() - after equirect * 255\n")
        # equirect_pil = Image.fromarray(equirect)
        
        if isinstance(equirect, torch.Tensor):
            # Convert PyTorch tensor to NumPy array first
            equirect_np = equirect.detach().cpu().numpy()
            equirect_pil = Image.fromarray(equirect_np)
        elif isinstance(equirect, np.ndarray):
            # NumPy array can go directly to PIL
            equirect_pil = Image.fromarray(equirect)
        else:
            raise TypeError(f"equirect must be either torch.Tensor or np.ndarray, got {type(equirect)}")

        print(f"pipeline.py - CubeDiffPipeline - generate() - after converted to PIL image\n")
        
        # ───────── NEW: optional down-scale preview ─────────
        if preview_hw is not None and output_type == "pil":
            equirect_pil = equirect_pil.resize(preview_hw[::-1],  # PIL expects (w,h)
                            resample=preview_interpolation)
            
        return equirect_pil
# EOF -------------------