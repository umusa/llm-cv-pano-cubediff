"""
Build a tiny dataset for CubeDiff from Polyhaven HDRIs.

Run once:

$ accelerate config   # set python -m torch.distributed.run etc.
$ python -m cl.data.polyhaven.build_tiny_set --out /workspace/polyhaven_tiny

The script:
1. Downloads HDRIs from Polyhaven
2. Converts them to cubemap faces
3. Optionally encodes faces to latent space using Stable Diffusion VAE
"""
import pathlib
import argparse
import json
import torch
import tqdm
import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use absolute imports instead of relative
from cl.data.polyhaven.api_client import list_hdris, parallel_download
from cl.data.polyhaven.cubemap_builder import batch_convert

def encode_faces(face_dir, latent_dir, batch=6):
    """
    Encode cubemap faces to latent space using VAE from Stable Diffusion.
    Only runs if the user wants to generate latents.
    
    Args:
        face_dir (str or Path): Directory containing cubemap face images
        latent_dir (str or Path): Directory to save latent representations
        batch (int, optional): Batch size for encoding. Defaults to 6.
    
    Returns:
        bool: True if encoding was successful, False otherwise
    """
    # First attempt to import modules without the try/except so we can get specific errors
    logger.info("Attempting to import required modules for latent encoding...")
    
    try:
        import torch
        logger.info("Successfully imported torch")
    except ImportError as e:
        logger.error(f"Failed to import torch: {e}")
        logger.warning("Cannot continue with latent encoding without PyTorch")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline
        logger.info("Successfully imported diffusers")
    except ImportError as e:
        logger.error(f"Failed to import diffusers: {e}")
        logger.warning("Please install diffusers: pip install diffusers transformers")
        return False
    
    # Main try block for the core functionality
    try:
        # Try importing CubemapDataset even if CubeDiffModel might not be available
        try:
            from cl.data.dataset import CubemapDataset
            logger.info("Successfully imported CubemapDataset")
        except ImportError as e:
            logger.error(f"Failed to import CubemapDataset: {e}")
            # Create a minimal implementation of CubemapDataset
            logger.info("Creating a minimal implementation of CubemapDataset")
            
            # Import PIL for the fallback implementation
            from PIL import Image
            import numpy as np
            
            class CubemapDataset(torch.utils.data.Dataset):
                """Minimal implementation of CubemapDataset."""
                def __init__(self, root_dir, channels_first=False, image_size=512):
                    self.root_dir = Path(root_dir)
                    self.channels_first = channels_first
                    self.image_size = image_size
                    
                    # Find all panorama directories
                    self.pano_dirs = []
                    for d in self.root_dir.iterdir():
                        if d.is_dir() and all((d / f"{face}.jpg").exists() for face in FACE_ORDER):
                            self.pano_dirs.append(d.name)
                    
                    logger.info(f"Found {len(self.pano_dirs)} valid panoramas in {root_dir}")
                
                def __len__(self):
                    return len(self.pano_dirs)
                
                def __getitem__(self, idx):
                    pano_name = self.pano_dirs[idx]
                    pano_dir = self.root_dir / pano_name
                    
                    # Load all 6 faces
                    faces = []
                    for face in FACE_ORDER:
                        img_path = pano_dir / f"{face}.jpg"
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                        img = np.array(img) / 255.0  # Normalize to [0, 1]
                        
                        if self.channels_first:
                            img = np.transpose(img, (2, 0, 1))
                        
                        faces.append(img)
                    
                    # Stack faces into a single tensor
                    faces = np.stack(faces, axis=0)
                    faces = torch.from_numpy(faces).float()
                    
                    return {"faces": faces, "name": pano_name}
        
        try:
            from cl.model.architecture import CubeDiffModel
            logger.info("Successfully imported CubeDiffModel")
        except ImportError as e:
            logger.warning(f"Failed to import CubeDiffModel: {e}")
            logger.info("This is expected and won't affect latent generation")
        
        # Import FACE_ORDER from cubemap_builder
        try:
            from cl.data.polyhaven.cubemap_builder import FACE_ORDER
        except ImportError as e:
            # Define FACE_ORDER directly if import fails
            FACE_ORDER = ['front', 'right', 'back', 'left', 'top', 'bottom']
            logger.warning(f"Could not import FACE_ORDER, using default: {FACE_ORDER}")
        
        latent_dir = Path(latent_dir)
        latent_dir.mkdir(exist_ok=True)
        
        # Check if latents already exist
        existing_latents = list(latent_dir.glob("*.pt"))
        if existing_latents:
            logger.info(f"Found {len(existing_latents)} existing latents in {latent_dir}, skipping encoding")
            return True

        # Load Stable Diffusion VAE
        logger.info("Loading Stable Diffusion VAE...")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16, 
                device_map="auto"
            )
            vae = pipe.vae.eval().half()
            logger.info("Successfully loaded Stable Diffusion VAE")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion VAE: {e}")
            return False

        # Set up dataset and loader
        face_dir = Path(face_dir)
        logger.info(f"Loading faces from {face_dir}")
        
        ds = CubemapDataset(face_dir, channels_first=True, image_size=512)
        
        if len(ds) == 0:
            logger.warning(f"No faces found in {face_dir}")
            return False
        
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
        logger.info(f"Found {len(ds)} panoramas with faces to encode")

        # Encode faces to latent space
        successful_encodes = 0
        for i, b in enumerate(tqdm.tqdm(loader, desc="Encoding latents")):
            try:
                lat = []
                with torch.no_grad():
                    for j in range(6):  # Six faces of the cubemap
                        f = b["faces"][:, j].half().to("cuda")
                        lat.append(vae.encode(f).latent_dist.sample() * 0.18215)
                
                lat = torch.stack(lat, 1).cpu()  # Shape: (B,6,4,64,64)
                
                # Save each latent tensor
                for k, l in enumerate(lat):
                    idx = i * batch + k
                    if idx < len(ds.pano_dirs):
                        pano_name = ds.pano_dirs[idx]
                        torch.save(l, latent_dir / f"{pano_name}.pt")
                        successful_encodes += 1
            except Exception as e:
                logger.error(f"Error encoding batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"Successfully encoded {successful_encodes} panoramas to latent space")
        return successful_encodes > 0
    
    except Exception as e:
        logger.error(f"Error during latent encoding: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to build the tiny dataset."""
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Build a tiny dataset for CubeDiff from Polyhaven HDRIs")
    ap.add_argument("--out", required=True, help="Dataset root directory")
    ap.add_argument("--n", type=int, default=700, help="Number of HDRIs to download")
    ap.add_argument("--skip_latents", action="store_true", help="Skip latent encoding")
    args = ap.parse_args()
    
    # Create output directories
    try:
        logger.info(f"Creating output directories in {args.out}")
        tmp = pathlib.Path(args.out) / "raw"
        erp_dir = tmp / "erp"
        faces_dir = tmp / "faces"
        latent_dir = pathlib.Path(args.out) / "latents"
        
        os.makedirs(tmp, exist_ok=True)
        os.makedirs(erp_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        sys.exit(1)
    
    # Download HDRIs
    try:
        logger.info(f"Listing HDRIs from Polyhaven (limit: {args.n})...")
        pairs = list_hdris(args.n)
        if not pairs:
            logger.error("No HDRIs found. Exiting.")
            sys.exit(1)
            
        logger.info(f"Found {len(pairs)} HDRIs, downloading to {erp_dir}...")
        successful_downloads = parallel_download(pairs, erp_dir)
        
        if successful_downloads == 0:
            logger.error("No HDRIs were downloaded successfully. Check network connection or Polyhaven API.")
            sys.exit(1)
        else:
            logger.info(f"Successfully downloaded {successful_downloads} HDRIs")
    except Exception as e:
        logger.error(f"Error during HDRI download: {e}")
        sys.exit(1)
        
    # Convert to cubemap faces
    try:
        logger.info("Converting equirectangular panoramas to cubemap faces...")
        num_converted = batch_convert(erp_dir, faces_dir, face_px=512)
        logger.info(f"Converted {num_converted} panoramas to cubemap faces")
    except Exception as e:
        logger.error(f"Error during cubemap conversion: {e}")
        sys.exit(1)
    
    # Generate dummy captions (asset_id as caption)
    try:
        captions_file = tmp / "captions.json"
        if not captions_file.exists():
            logger.info(f"Generating captions file at {captions_file}")
            captions = {}
            for pair in pairs:
                filename = pair[0]  # This is the asset_id
                captions[filename] = filename.replace("_", " ")
            
            with open(captions_file, "w") as f:
                json.dump(captions, f, indent=2)
            logger.info(f"Created captions for {len(captions)} panoramas")
    except Exception as e:
        logger.error(f"Error generating captions: {e}")
        # Not critical, continue execution
    
    # Encode latents if not skipped
    if not args.skip_latents:
        try:
            logger.info("Encoding cubemap faces to latent space...")
            success = encode_faces(faces_dir, latent_dir)
            if success:
                logger.info("Successfully encoded faces to latent space")
            else:
                logger.warning("Failed to encode faces to latent space")
        except Exception as e:
            logger.error(f"Error during latent encoding: {e}")
            logger.warning("Consider running with --skip_latents if you just need the cubemap faces")
    else:
        logger.info("Skipping latent encoding as requested")
    
    logger.info(f"✅ Tiny dataset ready at: {args.out}")

if __name__ == "__main__":
    main()