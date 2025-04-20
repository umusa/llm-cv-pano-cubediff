"""
Simplified CubeDiff implementation for generating 360° panoramas from text prompts.
This version avoids complex tensor manipulations to get reliable results.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from typing import List, Dict, Tuple, Union, Optional
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the cubemap conversion utilities
from cubediff_utils_v1 import improved_equirect_to_cubemap, optimized_cubemap_to_equirect

class SimplifiedCubeDiff:
    """
    Simplified implementation of CubeDiff for generating 360° panoramas from text prompts.
    This version processes each face of the cubemap separately to avoid tensor compatibility issues.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base",
        device=None,
    ):
        """
        Initialize the SimplifiedCubeDiff model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            device: Device to use (default: auto-detect)
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        print(f"Initializing SimplifiedCubeDiff on {device}")
        
        # Load the standard Stable Diffusion pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path
        ).to(device)
        
        # Define face order for cubemap
        self.face_order = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        # Define face weights for prompt engineering
        # Each face gets a specific emphasis in the prompt
        self.face_descriptions = {
            'front': "front view, looking straight ahead",
            'back': "back view, looking behind",
            'left': "left side view, looking to the left",
            'right': "right side view, looking to the right",
            'top': "top view, looking upward at the sky/ceiling",
            'bottom': "bottom view, looking downward at the ground/floor"
        }
        
        print("SimplifiedCubeDiff initialization complete!")
    
    def generate_panorama(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        face_size: int = 512,
        output_height: int = 1024,
        output_width: int = 2048,
        seed: Optional[int] = None,
        output_type: str = "equirectangular", # "equirectangular" or "cubemap"
    ):
        """
        Generate a panorama from a text prompt by creating six faces of a cubemap
        and then converting to equirectangular format.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Optional negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            face_size: Size of each cubemap face
            output_height: Height of output equirectangular image
            output_width: Width of output equirectangular image
            seed: Random seed for reproducibility
            output_type: Output format ("equirectangular" or "cubemap")
            
        Returns:
            Generated panorama (equirectangular format) and cubemap faces
        """
        print(f"Generating panorama for prompt: '{prompt}'")
        start_time = time.time()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate each face of the cubemap separately
        face_images = {}
        
        for face_name in self.face_order:
            # Create a direction-specific prompt for each face
            face_prompt = f"{prompt}, {self.face_descriptions[face_name]}"
            print(f"Generating {face_name} face with prompt: '{face_prompt}'")
            
            # Generate the image for this face
            face_image = self.pipeline(
                prompt=face_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=face_size,
                width=face_size,
                generator=generator,
            ).images[0]
            
            # Convert to numpy array
            face_images[face_name] = np.array(face_image)
        
        # Return just the cubemap if requested
        if output_type == "cubemap":
            return face_images
        
        # Convert the cubemap to equirectangular format
        print("Converting cubemap to equirectangular format...")
        equirect = optimized_cubemap_to_equirect(face_images, output_height, output_width)
        
        # Report timing
        total_time = time.time() - start_time
        print(f"Panorama generation completed in {total_time:.2f} seconds")
        
        return equirect, face_images
    
    def generate_from_image(
        self,
        input_image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        face_size: int = 512,
        output_height: int = 1024,
        output_width: int = 2048,
        seed: Optional[int] = None,
        output_type: str = "equirectangular", # "equirectangular" or "cubemap"
    ):
        """
        Generate a panorama from an input image and text prompt.
        
        Args:
            input_image: Input image (PIL Image or numpy array)
            prompt: Text prompt for generation
            negative_prompt: Optional negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            strength: Strength of conditioning (0-1)
            face_size: Size of each cubemap face
            output_height: Height of output equirectangular image
            output_width: Width of output equirectangular image
            seed: Random seed for reproducibility
            output_type: Output format ("equirectangular" or "cubemap")
            
        Returns:
            Generated panorama (equirectangular format) and cubemap faces
        """
        print(f"Generating panorama from input image and prompt: '{prompt}'")
        start_time = time.time()
        
        # Convert input image to PIL if it's numpy array
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Check if input is already equirectangular or a single view
        if input_image.width > input_image.height * 1.5:
            # Likely equirectangular - convert to cubemap
            print("Converting input equirectangular image to cubemap...")
            input_cubemap = improved_equirect_to_cubemap(np.array(input_image), face_size)
        else:
            # Single view - use as front face and generate the rest
            print("Using input as front face only...")
            # Resize to face_size if needed
            input_image_resized = input_image.resize((face_size, face_size))
            
            # Create empty cubemap with front face from input
            input_cubemap = {}
            input_cubemap['front'] = np.array(input_image_resized)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate or complete the cubemap
        face_images = {}
        
        for face_name in self.face_order:
            if face_name in input_cubemap:
                # Use the existing face from input
                face_images[face_name] = input_cubemap[face_name]
                print(f"Using input image for {face_name} face")
            else:
                # Generate the missing face
                face_prompt = f"{prompt}, {self.face_descriptions[face_name]}"
                print(f"Generating {face_name} face with prompt: '{face_prompt}'")
                
                # Generate the image for this face
                face_image = self.pipeline(
                    prompt=face_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=face_size,
                    width=face_size,
                    generator=generator,
                ).images[0]
                
                # Convert to numpy array
                face_images[face_name] = np.array(face_image)
        
        # Return just the cubemap if requested
        if output_type == "cubemap":
            return face_images
        
        # Convert the cubemap to equirectangular format
        print("Converting cubemap to equirectangular format...")
        equirect = optimized_cubemap_to_equirect(face_images, output_height, output_width)
        
        # Report timing
        total_time = time.time() - start_time
        print(f"Panorama generation completed in {total_time:.2f} seconds")
        
        return equirect, face_images

    def visualize_cubemap(self, cubemap, title="Cubemap Visualization"):
        """
        Visualize a cubemap as a grid of faces.
        
        Args:
            cubemap: Dictionary of cubemap faces
            title: Title for the plot
        """
        # Create a grid layout for the 6 faces
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Plot each face
        for i, face_name in enumerate(self.face_order):
            row, col = i // 3, i % 3
            axes[row, col].imshow(cubemap[face_name])
            axes[row, col].set_title(face_name)
            axes[row, col].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def test_simplified_cubediff():
    """
    Test function to demonstrate the SimplifiedCubeDiff implementation.
    """
    # Initialize the model
    model = SimplifiedCubeDiff()
    
    # Generate a panorama from a text prompt
    prompt = "A beautiful mountain landscape with a lake, snowy peaks, and pine trees"
    equirect, cubemap = model.generate_panorama(
        prompt=prompt,
        num_inference_steps=30,  # Reduced for faster testing
        face_size=256,  # Smaller faces for faster testing
        output_height=512,
        output_width=1024,
    )
    
    # Visualize the cubemap
    model.visualize_cubemap(cubemap, title=f"Cubemap: {prompt}")
    
    # Visualize the equirectangular panorama
    plt.figure(figsize=(15, 7.5))
    plt.imshow(equirect)
    plt.title(f"Equirectangular Panorama: {prompt}")
    plt.axis('off')
    plt.show()
    
    return equirect, cubemap


if __name__ == "__main__":
    # Run the test function if the script is executed directly
    test_simplified_cubediff()