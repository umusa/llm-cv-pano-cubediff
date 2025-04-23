# panorama_generator.py

import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDPMScheduler
import types

def load_model(checkpoint_path, pretrained_model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Load the trained CubeDiff model from checkpoint
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        pretrained_model_name: Base model name
        device: Device to load the model on
    
    Returns:
        Loaded pipeline ready for inference
    """
    from cl.inference.pipeline import CubeDiffPipeline
    
    print(f"Loading model from {checkpoint_path}")
    
    try:
        pipeline = CubeDiffPipeline(
            pretrained_model_name=pretrained_model_name,
            checkpoint_path=checkpoint_path,
            device=device,
            strict_loading=False  # Use non-strict loading by default
        )
        print("Model loaded successfully")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_panorama(pipeline, prompt, negative_prompt="low quality, blurry, distorted", 
                     num_inference_steps=30, guidance_scale=7.5, height=512, width=512):
    """
    Generate a panorama from a text prompt using the trained model
    
    Args:
        pipeline: Loaded CubeDiffPipeline
        prompt: Text prompt describing the desired panorama
        negative_prompt: Prompt describing what to avoid
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        height: Height of the output image
        width: Width of the output image
    
    Returns:
        Generated panorama image
    """
    print(f"Generating panorama for prompt: '{prompt}'")
    
    try:
        # Get model's dtype for consistent precision handling
        model_dtype = next(pipeline.vae.parameters()).dtype
        
        # Use automatic mixed precision to handle dtype conversions
        with torch.cuda.amp.autocast(enabled=True):
            # Use the pipeline's generate method directly
            panorama = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                output_type="pil"
            )
        
        print("Panorama generated successfully")
        return panorama
    except Exception as e:
        print(f"Error generating panorama: {e}")
        # Create a placeholder image instead of raising an exception
        placeholder = Image.new('RGB', (width*4, height*2), color=(100, 100, 100))
        return placeholder

def display_panorama(panorama, prompt, save_path=None):
    """
    Display and optionally save the generated panorama
    
    Args:
        panorama: Generated panorama image
        prompt: Text prompt used to generate the image
        save_path: Path to save the image (optional)
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(panorama))
    plt.title(f"Prompt: {prompt}")
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        panorama.save(save_path)
        print(f"Panorama saved to {save_path}")
    
    plt.show()

def test_panorama_generation(checkpoint_path, prompts, output_dir=None, 
                            pretrained_model_name="runwayml/stable-diffusion-v1-5"):
    """
    Run panorama generation test with the provided prompts
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        prompts: List of text prompts to test
        output_dir: Directory to save the generated panoramas
        pretrained_model_name: Base model name
    
    Returns:
        Generated panorama images
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    pipeline = load_model(
        checkpoint_path=checkpoint_path, 
        pretrained_model_name=pretrained_model_name, 
        device=device
    )
    
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            # Generate the panorama
            panorama = generate_panorama(pipeline, prompt)
            results.append(panorama)
            
            # Save if output directory is specified
            if output_dir:
                save_path = os.path.join(output_dir, f"panorama_{i}.jpg")
                panorama.save(save_path)
                print(f"Saved panorama to {save_path}")
            
            # Display the panorama
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(panorama))
            plt.title(f"Prompt: {prompt}")
            plt.axis('off')
            plt.show()
                
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            # Create a placeholder image
            placeholder = Image.new('RGB', (512*4, 512*2), color=(100, 100, 100))
            results.append(placeholder)
            
            if output_dir:
                save_path = os.path.join(output_dir, f"panorama_error_{i}.jpg")
                placeholder.save(save_path)
                print(f"Generated placeholder for prompt: {prompt}")
    
    return results

def get_latents(pipeline, prompt, negative_prompt="", num_inference_steps=30, 
               guidance_scale=7.5, height=512, width=512):
    """
    Get latents for a text prompt
    
    Args:
        pipeline: Loaded CubeDiffPipeline
        prompt: Text prompt
        negative_prompt: Negative prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        height: Height of each face
        width: Width of each face
        
    Returns:
        Latents
    """
    # Get model's dtype for consistent precision
    model_dtype = next(pipeline.vae.parameters()).dtype
    
    # Set up scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps)
    
    # Encode text
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings = pipeline.text_encoder(
        text_input.input_ids.to(pipeline.device)
    )[0].to(dtype=model_dtype)
    
    # Get unconditional embeddings for guidance
    if guidance_scale > 1.0:
        uncond_input = pipeline.tokenizer(
            [negative_prompt or ""],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_embeddings = pipeline.text_encoder(
            uncond_input.input_ids.to(pipeline.device)
        )[0].to(dtype=model_dtype)
        
        # For classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Generate random latent vectors for each face
    num_faces = 6
    latents = torch.randn(
        (1, num_faces, 4, height // 8, width // 8),
        device=pipeline.device,
        dtype=model_dtype,
    )
    
    # Denoise latents
    for t in pipeline.scheduler.timesteps:
        # Expand latents for classifier-free guidance
        latent_model_input = latents.repeat(2, 1, 1, 1, 1) if guidance_scale > 1.0 else latents
        
        # Get model prediction
        with torch.no_grad():
            # Use int64 for timesteps as required by the model
            timesteps = torch.tensor([t] * latent_model_input.shape[0], device=pipeline.device, dtype=torch.int64)
            
            noise_pred = pipeline.model(
                latent_model_input,
                timesteps,
                text_embeddings,
            )
        
        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Update latents
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents

def faces_to_panorama(cube_faces, output_type="pil"):
    """
    Convert cube faces to panorama
    
    Args:
        cube_faces: List of 6 cube face images
        output_type: Output type ("pil" or "np")
        
    Returns:
        Panorama image
    """
    # Simple implementation for testing
    # Just place faces side by side
    if isinstance(cube_faces[0], torch.Tensor):
        # Convert tensors to numpy
        faces_np = []
        for face in cube_faces:
            # Remove batch dimension if present
            if len(face.shape) == 4:
                face = face[0]
            # Normalize
            face_np = (face / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            faces_np.append(face_np)
    else:
        faces_np = cube_faces
    
    # Get dimensions
    height, width = faces_np[0].shape[:2]
    
    # Create empty panorama
    panorama = np.zeros((height * 2, width * 3, 3), dtype=np.float32)
    
    # Place faces in a grid
    # [0][1][2]
    # [3][4][5]
    for i, face in enumerate(faces_np[:6]):  # Only use first 6 faces
        row = i // 3
        col = i % 3
        panorama[row * height:(row + 1) * height, col * width:(col + 1) * width] = face
    
    if output_type == "np":
        return panorama
    
    # Convert to PIL
    panorama = (panorama * 255).astype(np.uint8)
    return Image.fromarray(panorama)

def fix_model_for_mixed_precision(pipeline):
    """
    Apply fixes to handle mixed precision inference by ensuring consistent dtypes
    
    Args:
        pipeline: The CubeDiffPipeline to fix
        
    Returns:
        Fixed pipeline
    """
    # Get model's dtype
    model_dtype = next(pipeline.vae.parameters()).dtype
    device = pipeline.device
    
    print(f"Fixing model for inference with dtype: {model_dtype}")
    
    # Fix the model's forward method
    original_forward = pipeline.model.forward
    
    def patched_forward(self, latent_model_input, t, encoder_hidden_states=None):
        """
        Ensure consistent dtypes in forward pass
        """
        # Convert inputs to the model's dtype
        latent_model_input = latent_model_input.to(dtype=model_dtype)
        # Note: timestep t should remain int64
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        
        # Call the original forward method with type-matched inputs
        return original_forward(latent_model_input, t, encoder_hidden_states)
    
    # Replace the forward method
    pipeline.model.forward = types.MethodType(patched_forward, pipeline.model)
    
    print("Model fixed for inference with mixed precision")
    return pipeline



# (other functions above)

def fix_model_for_mixed_precision(pipeline):
    """
    Apply fixes to handle mixed precision inference by ensuring consistent dtypes
    
    Args:
        pipeline: The CubeDiffPipeline to fix
        
    Returns:
        Fixed pipeline
    """
    # Get model's dtype
    model_dtype = next(pipeline.vae.parameters()).dtype
    device = pipeline.device
    
    print(f"Fixing model for inference with dtype: {model_dtype}")
    
    # Fix the model's forward method
    original_forward = pipeline.model.forward
    
    def patched_forward(self, latent_model_input, t, encoder_hidden_states=None):
        """
        Ensure consistent dtypes in forward pass
        """
        # Convert inputs to the model's dtype
        latent_model_input = latent_model_input.to(dtype=model_dtype)
        # Note: timestep t should remain int64
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        
        # Call the original forward method with type-matched inputs
        return original_forward(latent_model_input, t, encoder_hidden_states)
    
    # Replace the forward method
    import types
    pipeline.model.forward = types.MethodType(patched_forward, pipeline.model)
    
    print("Model fixed for inference with mixed precision")
    return pipeline





