"""
Download sample panorama datasets for CubeDiff training.
"""

import os
import requests
import argparse
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Download panorama datasets for CubeDiff")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["polyhaven", "sun360", "matterport3d_sample"], 
        default="polyhaven",
        help="Dataset to download"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=20,
        help="Number of panoramas to download (for polyhaven)"
    )
    
    return parser.parse_args()

def download_file(url, filepath, desc=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_polyhaven(output_dir, num_samples=20):
    """Download panoramas from Polyhaven."""
    # Get list of available HDRIs from Polyhaven API
    print("Fetching HDRI list from Polyhaven...")
    response = requests.get("https://api.polyhaven.com/hdris")
    hdris = response.json()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download HDRIs
    count = 0
    captions = {}
    
    for hdri_id, hdri_info in hdris.items():
        if count >= num_samples:
            break
        
        # Get HDRI info
        hdri_url = f"https://dl.polyhaven.com/file/ph-assets/HDRIs/exr/2k/{hdri_id}_2k.exr"
        hdri_path = os.path.join(output_dir, f"{hdri_id}.exr")
        
        # Download HDRI
        try:
            download_file(hdri_url, hdri_path, desc=f"Downloading {hdri_id}")
            
            # Create caption for the HDRI
            caption = hdri_info.get('name', hdri_id)
            if 'tags' in hdri_info:
                tags = ', '.join(hdri_info['tags'])
                caption += f". {tags}"
            
            captions[hdri_id] = caption
            count += 1
        except Exception as e:
            print(f"Error downloading {hdri_id}: {e}")
    
    # Save captions
    captions_path = os.path.join(output_dir, "captions.json")
    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=4)
    
    print(f"Downloaded {count} HDRIs to {output_dir}")
    print(f"Captions saved to {captions_path}")

def download_sun360(output_dir):
    """Download sample from SUN360 dataset."""
    # The SUN360 dataset is no longer publicly available in full
    # Here we download a small sample that's still available
    sample_url = "http://vision.princeton.edu/projects/2012/SUN360/data/SUN360_subset.zip"
    zip_path = os.path.join(output_dir, "sun360_subset.zip")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and extract sample
    print("Downloading SUN360 sample...")
    download_file(sample_url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Clean up
    os.remove(zip_path)
    
    print(f"SUN360 sample extracted to {output_dir}")

def download_matterport3d_sample(output_dir):
    """Download Matterport3D sample (for demonstration only)."""
    # The full Matterport3D dataset requires registration
    # Here we create a placeholder for demonstration
    os.makedirs(output_dir, exist_ok=True)
    
    placeholder_text = """
    The Matterport3D dataset requires registration at:
    https://niessner.github.io/Matterport/
    
    After registration, you can download the dataset and place
    the panorama images in this directory.
    
    For CubeDiff training, you'll need the equirectangular RGB
    panoramas from the dataset.
    """
    
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(placeholder_text)
    
    print("Created placeholder for Matterport3D dataset.")
    print("Please register at https://niessner.github.io/Matterport/ to access the full dataset.")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download selected dataset
    if args.dataset == "polyhaven":
        download_polyhaven(os.path.join(args.output_dir, "polyhaven"), args.num_samples)
    elif args.dataset == "sun360":
        download_sun360(os.path.join(args.output_dir, "sun360"))
    elif args.dataset == "matterport3d_sample":
        download_matterport3d_sample(os.path.join(args.output_dir, "matterport3d"))
    
    print("\nData download complete!")
    print("Next steps:")
    print("1. Process the data: python main.py --mode preprocess --input_dir data/raw --output_dir data/processed")
    print("2. Train the model: python main.py --mode train --config config/training_config.yaml")

if __name__ == "__main__":
    main()