"""
Inference script for CubeDiff model.
Run text-to-panorama or image-to-panorama generation with trained models.
"""

import os
import argparse
import torch
from PIL import Image
from diffusers import DDIMScheduler

from cubediff_models import load_sd_components, convert_to_inflated_attention
from cubediff_inference import CubeDiffInference


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with CubeDiff model")
    
    # Model parameters
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base model ID for pretrained Stable Diffusion")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to trained UNet checkpoint (optional)")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--input_image", type=str, default=None,
                        help="Path to input image for image-to-panorama generation (optional)")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="Directory to save generated panorama")
    parser.add_argument("--face_size", type=int, default=512,
                        help="Size of each cubemap face")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_faces", action="store_true",
                        help="Save individual cubemap faces")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model components
    print(f"Loading base model from {args.model_id}...")
    vae, text_encoder, tokenizer, unet = load_sd_components(
        model_id=args.model_id,
        use_sync_gn=True
    )
    
    # Convert UNet to use inflated attention
    unet = convert_to_inflated_attention(unet)
    
    # Load trained checkpoint if provided
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading trained weights from {args.checkpoint_path}...")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        
        # If state dict has 'model' key, use that
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        unet.load_state_dict(state_dict)
    
    # Create scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        num_train_timesteps=1000,
        clip_sample=False,
        prediction_type="epsilon"
    )
    
    # Create inference pipeline
    pipeline = CubeDiffInference(
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        device=device
    )
    
    # Parse seed
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {seed}")
    
    # Generate panorama
    if args.input_image is not None:
        # Image-to-panorama generation
        print(f"Generating panorama from image: {args.input_image}")
        image = Image.open(args.input_image).convert("RGB")
        
        result = pipeline.generate_from_image(
            image=image,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=seed
        )
    else:
        # Text-to-panorama generation
        print(f"Generating panorama from text prompt: {args.prompt}")
        
        result = pipeline.generate(
            prompt=args.prompt,
            height=args.face_size,
            width=args.face_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=seed,
            return_faces=args.save_faces
        )
    
    # Save outputs
    panorama = result["panorama"]
    panorama_path = os.path.join(args.output_dir, "panorama.png")
    panorama.save(panorama_path)
    print(f"Panorama saved to: {panorama_path}")
    
    if args.save_faces and "faces" in result:
        faces_dir = os.path.join(args.output_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        for i, (face, name) in enumerate(zip(result["faces"], face_names)):
            face_path = os.path.join(faces_dir, f"{name}.png")
            face.save(face_path)
        
        print(f"Cubemap faces saved to: {faces_dir}")


if __name__ == "__main__":
    main()