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
        # self.pipeline = StableDiffusionPipeline.from_pretrained(
        #     pretrained_model_name,
        #     # torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        #     torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        # )
        # self.pipeline.to(device)        

        def safe_pipeline_from_pretrained(repo_id, **kwargs):
            # 1) try *only* local cache
            try:
                return StableDiffusionPipeline.from_pretrained(
                    repo_id,
                    local_files_only=True,
                    **kwargs
                )
            except (OSError, ValueError):
                # 2) fallback to the normal download→cache path
                return StableDiffusionPipeline.from_pretrained(repo_id, **kwargs)

        self.pipeline = safe_pipeline_from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        self.pipeline.to(device)   

        print(f"pipeline.py - CubeDiffPipeline - __init__() - pipeline loaded from {pretrained_model_name} to {device}\n")
        # Extract components
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        
        # Set up scheduler
        # self.scheduler = DDIMScheduler.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="scheduler",
        # )
        # self.scheduler.set_timesteps(50)  # Use 50 steps by default

        def safe_sched_from_pretrained(repo_id, subfolder, **kwargs):
            # 1) try *only* local cache
            try:
                return DDIMScheduler.from_pretrained(
                    repo_id,
                    local_files_only=True,
                    subfolder=subfolder,
                    **kwargs
                )
            except (OSError, ValueError):
                # 2) fallback to the normal download→cache path
                return DDIMScheduler.from_pretrained(repo_id, subfolder=subfolder, **kwargs)

        self.scheduler = safe_sched_from_pretrained(
            repo_id=pretrained_model_name,
            subfolder="scheduler",
        )
        self.scheduler.set_timesteps(50)  # Use 50 steps by default
        print(f"pipeline.py - CubeDiffPipeline - __init__() - DDIMScheduler loaded from {pretrained_model_name}\n")

        # Initialize CubeDiff model
        self.model = CubeDiffModel(pretrained_model_name)
        print(f"pipeline.py - CubeDiffPipeline - __init__() - CubeDiffModel loaded from {pretrained_model_name}\n")
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
        print(f"pipeline.py - CubeDiffPipeline - __init__() - CubeDiffPipeline initialized\n")
        
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
        
        # Generate random latent vectors for each face (4 latent dims)
        num_faces = 6
        latents = torch.randn(
            (1, num_faces, 4, height // 8, width // 8),
            device=self.device,
            dtype=model_dtype,
        )
        # ── Append mask channel so conv_in sees 4+9+1=14 channels ──
        # CFG: Classifier-free guidance mask: 1 = text‐conditioned
        # mask = torch.ones(
        #     (1, num_faces, 1, height // 8, width // 8),
        #     device=self.device,
        #     dtype=model_dtype,
        # )
        # latents = torch.cat([latents, mask], dim=2)  # → [1,6,5,H/8,W/8] ([1, 6, 4, 64, 64])
        print(f"pipeline.py - CubeDiffPipeline - generate() - before Denoise latents, latents shape is {latents.shape}, latents type is {type(latents)}\n")

        # Denoise latents
        for t in self.scheduler.timesteps:
            # — 1) Prepare input to U-Net (with mask still attached) —
            latent_model_input = latents.repeat(2,1,1,1,1) if guidance_scale>1.0 else latents
            # latent_model_input = latents
            # — 2) Predict noise on the 4 real channels (model strips mask internally) —
            with torch.no_grad():
                timesteps = torch.tensor([t] * latent_model_input.shape[0],
                                        device=self.device,
                                        dtype=torch.int64)
                noise_pred = self.model(latent_model_input, timesteps, text_embeddings)

            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
            # — 3) Do DDIM step on just the 4 real channels —
            #    split latents → [B,6,4,H/8,W/8] + [B,6,1,H/8,W/8]
            # real_lat, mask = latents[:, :, :4], latents[:, :, 4:].clone()
            real_lat = latents[:, :, :4]
            latents = self.scheduler.step(noise_pred, t, real_lat).prev_sample
            
            # scheduler.step requires model_output and sample to share the same shape. By slicing off the mask channel, ensure both are [B,6,4,H/8,W/8]
            # updated_real = self.scheduler.step(noise_pred, t, real_lat).prev_sample

            # — 4) re-attach the mask for the next iteration —
            # Maintaining mask: re-attach the mask each iteration so the next U-Net call still sees the 5-channel input it expects.
            # latents = torch.cat([updated_real, mask], dim=2)
    
        # At this point `latents` is [1,6,5,H/8,W/8]
        print(f"pipeline.py - CubeDiffPipeline - generate() - after Denoise latents, latents shape is {latents.shape}, latents type is {type(latents)}\n")
        
        # 5) Drop mask before decoding with VAE
        # --------------------------------------------------------
        with torch.no_grad():
            cube_faces = []
            for i in range(num_faces):
                # keep only the 4 latent channels, drop the mask
                # VAE decode: The VAE always expects exactly 4 latent channels; dropping the mask channel here prevents another shape mismatch.
                face_latent = latents[0, i, :4].unsqueeze(0)     # → [1,4,H/8,W/8]
                with torch.amp.autocast(enabled=True, device_type="cuda"):
                    out = self.vae.decode(face_latent / 0.18215) 
                cube_faces.append(out.sample[0])                 # → [4 h, w]
        # --------------------------------------------------------
        # Batch decode rather than Python loop for speed:
        # assume latents: [1,6,4,H/8,W/8]
        # faces_lat = latents[0, :, :4]            # [6,4,H/8,W/8]
        # faces_lat = latents[0] # no mask because mask weas added by CubeDiffModel.forward()
        # faces_lat = faces_lat / 0.18215          # (the standard LDM latent scaling) before decoding
        # with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        #     out = self.vae.decode(faces_lat)     # [6,4,h,w]
        # cube_faces = [out.sample[i] for i in range(6)]
        # --------------------------------------------------------

        #     # Stack cube faces
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