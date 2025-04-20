"""
CubeDiff pipeline initialization and management.
"""

import torch
from diffusers import DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler

from cubediff_models import load_sd_components, convert_attention_modules
from cubediff_inference import CubeDiffInference


def create_cubediff_pipeline(
    pretrained_model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="pndm",
    use_enhanced_attention=True,
    device="cuda"
):
    """
    Create a CubeDiff pipeline with all components properly initialized.
    
    Args:
        pretrained_model_id: HuggingFace model ID for pretrained Stable Diffusion
        scheduler_type: Type of scheduler to use ("ddim", "pndm", or "dpm")
        use_enhanced_attention: Whether to use enhanced attention mechanisms
        device: Device to run inference on
        
    Returns:
        Initialized CubeDiff pipeline
    """
    print(f"Initializing CubeDiff with {pretrained_model_id} on {device}")
    
    # Load model components with synchronized GroupNorm
    vae, text_encoder, tokenizer, unet = load_sd_components(
        model_id=pretrained_model_id, 
        use_sync_gn=True,
        device=device
    )
    
    # Convert UNet's attention to inflated attention
    print("Converting UNet attention modules...")
    unet = convert_attention_modules(unet, use_enhanced_attn=use_enhanced_attention)
    
    # Set up scheduler
    print(f"Using {scheduler_type} scheduler...")
    if scheduler_type.lower() == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_id,
            subfolder="scheduler"
        )
    elif scheduler_type.lower() == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            pretrained_model_id,
            subfolder="scheduler"
        )
    else:  # default to pndm
        scheduler = PNDMScheduler.from_pretrained(
            pretrained_model_id,
            subfolder="scheduler"
        )
    
    # Adjust scheduler settings for better quality
    scheduler.set_timesteps(100)  # Higher number of steps by default
    
    # Create inference pipeline
    pipeline = CubeDiffInference(
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        device=device
    )
    
    print("CubeDiff pipeline initialized successfully")
    return pipeline


def generate_panorama(
    prompt,
    negative_prompt="blurry, ugly, distorted, low quality, low resolution, bad anatomy, worst quality, text, watermark",
    height=768,
    width=768,
    num_inference_steps=100,
    guidance_scale=9.5,
    seed=None,
    pipeline=None,
    return_faces=True,
    use_pos_encodings=True
):
    """
    Generate a high-quality panorama from a text prompt using CubeDiff.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative text prompt
        height: Height of each cubemap face
        width: Width of each cubemap face
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducibility
        pipeline: Existing pipeline (created if None)
        return_faces: Whether to return individual cubemap faces
        use_pos_encodings: Whether to use positional encodings
        
    Returns:
        Generated panorama and faces
    """
    # Create pipeline if not provided
    if pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = create_cubediff_pipeline(device=device)
    
    # Generate panorama
    print(f"Generating panorama with prompt: {prompt}")
    print(f"Using {num_inference_steps} steps and guidance scale {guidance_scale}")
    
    result = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_faces=return_faces,
        use_pos_encodings=use_pos_encodings
    )
    
    return result


def debug_cubediff_setup():
    """
    Test and debug the CubeDiff setup with minimal parameters.
    
    Returns:
        Test generation results
    """
    # Load pipeline on CPU if needed for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = create_cubediff_pipeline(
        device=device,
        scheduler_type="dpm"  # DPM solver for faster testing
    )
    
    # Test with a simple prompt
    simple_prompt = "A blue sky with white clouds"
    
    # Generate with minimal steps for quick test
    print("Testing generation with minimal steps...")
    result = pipeline.generate(
        prompt=simple_prompt,
        num_inference_steps=10,  # Minimal steps for testing
        guidance_scale=7.5,
        height=256,  # Small size for quick test
        width=256,
        return_faces=True,
        use_pos_encodings=True
    )
    
    # Debug output
    print(f"Generated {len(result['faces'])} faces")
    
    # Return test result
    return result