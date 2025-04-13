"""
This module provides functions for generating and visualizing cubemap faces for the CubeDiff model.
It contains code extracted from notebook cell [16] for better maintainability.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cubediff_utils import check_face_order, cubemap_to_equirect
from cubediff_visualize import (create_cubemap_layout, create_enhanced_cube_visualization, 
                               compare_equirectangular)

def generate_and_visualize_cubemap(prompt, model_inference_function, face_size=512, 
                                   face_order_correction=None, equirect_size=(1024, 2048),
                                   original_equirect=None):
    """
    Generate cubemap faces from a text prompt and visualize them in various ways.
    
    Args:
        prompt: Text prompt for the generation model
        model_inference_function: Function that generates cubemap faces from a prompt
        face_size: Size of each cubemap face (default: 512)
        face_order_correction: Optional list to reorder faces if needed (default: None)
        equirect_size: Size of the output equirectangular panorama as (height, width) (default: (1024, 2048))
        original_equirect: Optional original panorama for comparison (default: None)
        
    Returns:
        Dict containing:
        - 'faces': Generated cubemap faces
        - 'equirect': Generated equirectangular panorama
        - 'metrics': Reconstruction quality metrics (if applicable)
    """
    # Generate cubemap faces with the model
    print(f"Generating cubemap faces from prompt: \"{prompt}\"...")
    generated_faces = model_inference_function(prompt, face_size=face_size)
    
    # Apply face order correction if provided
    if face_order_correction is not None:
        if len(face_order_correction) != len(generated_faces):
            print(f"Warning: face_order_correction length ({len(face_order_correction)}) "
                  f"doesn't match number of faces ({len(generated_faces)})")
        else:
            generated_faces = [generated_faces[i] for i in face_order_correction]
    
    # Verify face order
    print("\nVerifying face order...")
    face_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
    is_valid = check_face_order(generated_faces)
    
    if not is_valid:
        print("WARNING: Face order validation failed!")
    
    # Display the generated faces
    print("\nDisplaying generated faces...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (face, name) in enumerate(zip(generated_faces, face_names)):
        axes[i].imshow(face)
        axes[i].set_title(name)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display cubemap layout
    print("\nDisplaying cubemap layout...")
    cubemap_layout = create_cubemap_layout(generated_faces, with_labels=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(cubemap_layout)
    plt.title('Generated Cubemap Layout')
    plt.axis('off')
    plt.show()
    
    # Create a 3D cube visualization of the generated faces
    print("\nCreating 3D cube visualization...")
    try:
        fig = create_enhanced_cube_visualization(generated_faces)
        plt.show()
        print("3D cube visualization created successfully.")
    except Exception as e:
        print(f"Could not create 3D cube visualization: {str(e)}")
        # If the above fails, try the alternative visualization
        print("Displaying alternative cubemap layout...")
        cubemap_layout = create_cubemap_layout(generated_faces, with_labels=True)
        plt.figure(figsize=(12, 12))
        plt.imshow(cubemap_layout)
        plt.title('Generated Cubemap Layout (Alternative View)')
        plt.axis('off')
        plt.show()
    
    # Convert the generated cubemap to equirectangular panorama
    print("\nConverting cubemap to equirectangular panorama...")
    equirect_h, equirect_w = equirect_size
    generated_equirect = cubemap_to_equirect(generated_faces, equirect_h, equirect_w)
    
    # Display the generated equirectangular panorama
    plt.figure(figsize=(15, 7.5))
    plt.imshow(generated_equirect)
    plt.title('Generated Equirectangular Panorama')
    plt.axis('off')
    plt.show()
    
    # Optional: Compare with an original panorama if available
    metrics = {}
    if original_equirect is not None:
        print("\nComparing with original panorama...")
        # Resize original to match generated size for fair comparison
        original_resized = cv2.resize(original_equirect, (equirect_w, equirect_h))
        
        # Use the comparison function from cubediff_visualize
        compare_equirectangular(original_resized, generated_equirect, 
                              titles=['Original Panorama', 'Generated Panorama'])
    
    # Final validation: Check if cubemap reconstruction preserves the generated content
    print("\nValidating reconstruction consistency...")
    reconstructed_from_generated = cubemap_to_equirect(generated_faces, 
                                                     equirect_h=equirect_h,
                                                     equirect_w=equirect_w)
    
    # Calculate similarity to the direct conversion
    if reconstructed_from_generated.shape == generated_equirect.shape:
        mse = np.mean((reconstructed_from_generated.astype(np.float32) - 
                      generated_equirect.astype(np.float32))**2)
        psnr = 10 * np.log10((255**2) / mse) if mse > 0 else float('inf')
        
        metrics['mse'] = mse
        metrics['psnr'] = psnr
        
        print(f"Reconstruction consistency - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
        
        # Evaluate consistency
        if mse < 1.0 or psnr > 40:
            print("Excellent reconstruction consistency! The cubemap faces are coherent.")
        elif mse < 10.0 or psnr > 30:
            print("Good reconstruction consistency. Minor adjustments might improve results.")
        else:
            print("Poor reconstruction consistency. The cubemap faces may not be logically coherent.")
    
    return {
        'faces': generated_faces,
        'equirect': generated_equirect,
        'metrics': metrics
    }


def batch_generate_cubemaps(prompts, model_inference_function, face_size=512, 
                            face_order_correction=None, equirect_size=(1024, 2048)):
    """
    Generate multiple cubemaps from a list of prompts.
    
    Args:
        prompts: List of text prompts for the generation model
        model_inference_function: Function that generates cubemap faces from a prompt
        face_size: Size of each cubemap face (default: 512)
        face_order_correction: Optional list to reorder faces if needed (default: None)
        equirect_size: Size of the output equirectangular panorama as (height, width) (default: (1024, 2048))
        
    Returns:
        List of dicts containing results for each prompt
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n\n{'='*50}")
        print(f"Processing prompt {i+1}/{len(prompts)}: \"{prompt}\"")
        print(f"{'='*50}\n")
        
        result = generate_and_visualize_cubemap(
            prompt, 
            model_inference_function, 
            face_size=face_size,
            face_order_correction=face_order_correction,
            equirect_size=equirect_size
        )
        
        results.append(result)
    
    return results


# Example usage in the notebook:
"""
# Cell [16]: Generate cubemap faces from a text prompt using your CubeDiff model
from cubediff_inference import generate_cubemap_from_prompt
from generate_and_visualize import generate_and_visualize_cubemap

# Generate and visualize cubemap faces
prompt = "A scenic mountain landscape with a lake and forest"

# If your model outputs faces in a different order than expected,
# provide a correction mapping. For example:
# face_order_correction = [3, 1, 0, 2, 4, 5] 
# Replace with None if no correction is needed
face_order_correction = None

# Run the generation and visualization
results = generate_and_visualize_cubemap(
    prompt=prompt,
    model_inference_function=generate_cubemap_from_prompt,
    face_size=512,
    face_order_correction=face_order_correction
)

# Access the generated faces and panorama
generated_faces = results['faces']
generated_equirect = results['equirect']
"""