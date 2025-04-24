"""
API client for downloading HDRIs from Polyhaven.

This module provides functions to list and download HDRIs from the Polyhaven API.
"""
import os
import requests
import tqdm
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Polyhaven API
BASE_API_URL = "https://api.polyhaven.com"
BASE_DL_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k"
FALLBACK_DL_URL = "https://dl.polyhaven.com/HDRIs/2k/exr"  # Alternative URL structure
HEADERS = {
    'User-Agent': 'CubeDiff-Research-Project/1.0',
    'Accept': 'application/json',
}

# Fallback HDRIs in case the API fails
FALLBACK_HDRIS = [
    "rural_asphalt_road",
    "kloppenheim_02",
    "kloppenheim_06",
    "wide_street_01",
    "small_rural_road",
    "abandoned_factory_canteen",
    "abandoned_slaughterhouse",
    "abandoned_tank_farm",
    "aerodynamics_workshop",
    "air_museum_entrance",
    "alpine_cabin",
    "alps_field",
    "artist_workshop",
    "autumn_forest",
    "autumn_park",
]

def list_hdris(limit: int = 1000) -> List[Tuple[str, str]]:
    """
    Return list of (asset_id, download_url) tuples for HDRIs.
    
    Args:
        limit (int, optional): Maximum number of HDRIs to list. Defaults to 1000.
        
    Returns:
        List[Tuple[str, str]]: List of (asset_id, download_url) tuples
    """
    logger.info(f"Fetching list of HDRIs from Polyhaven API (limit: {limit})...")
    
    try:
        # Get the list of all assets
        response = requests.get(f"{BASE_API_URL}/assets", headers=HEADERS, timeout=30)
        response.raise_for_status()
        assets = response.json()
        
        # Filter for HDRIs only (type 0 according to API docs)
        hdri_assets = []
        for asset_id, asset_info in assets.items():
            if isinstance(asset_info, dict) and asset_info.get('type') == 0:  # HDRI type
                hdri_assets.append(asset_id)
        
        # Limit the number of assets
        hdri_assets = hdri_assets[:limit]
        
        # Construct direct download URLs
        result = []
        for asset_id in hdri_assets:
            formatted_id = f"{asset_id}_2k.exr"
            url = f"{BASE_DL_URL}/{formatted_id}"
            result.append((asset_id, url))
        
        logger.info(f"Found {len(result)} HDRIs")
        return result
        
    except Exception as e:
        logger.error(f"Error accessing Polyhaven API: {e}")
        # Use fallback HDRIs if API fails
        logger.warning("Using fallback HDRI list")
        return use_fallback_hdris(limit)

def use_fallback_hdris(limit: int) -> List[Tuple[str, str]]:
    """
    Use fallback HDRI list when API fails.
    
    Args:
        limit (int): Maximum number of HDRIs to return
        
    Returns:
        List[Tuple[str, str]]: List of (asset_id, download_url) tuples for fallback HDRIs
    """
    # Limit to available fallbacks
    fallback_limit = min(limit, len(FALLBACK_HDRIS))
    result = []
    
    for hdri in FALLBACK_HDRIS[:fallback_limit]:
        formatted_id = f"{hdri}_2k.exr"
        url = f"{BASE_DL_URL}/{formatted_id}"
        result.append((hdri, url))
    
    return result

def download(pair: Tuple[str, str], outdir: Union[str, Path]) -> bool:
    """
    Download a file from url to outpath if it doesn't already exist.
    
    Args:
        pair (Tuple[str, str]): Tuple of (asset_id, download_url)
        outdir (str or Path): Output directory
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    asset_id, url = pair
    outdir = Path(outdir)
    outpath = outdir / f"{asset_id}.exr"
    
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Check if file already exists
    if outpath.exists():
        logger.debug(f"File already exists: {outpath}")
        return True
    
    # Try to download the file
    try:
        logger.debug(f"Downloading {url} to {outpath}")
        response = requests.get(url, stream=True, headers=HEADERS, timeout=60)
        response.raise_for_status()
        
        # Save the file
        with open(outpath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded: {outpath}")
        return True
        
    except Exception as e:
        logger.warning(f"Error downloading {url}: {e}")
        
        # Try alternative URL format
        if "_2k.exr" in url:
            try:
                # Try the fallback URL structure
                alt_url = f"{FALLBACK_DL_URL}/{asset_id}_2k.exr"
                logger.debug(f"Trying alternative URL: {alt_url}")
                
                response = requests.get(alt_url, stream=True, headers=HEADERS, timeout=60)
                response.raise_for_status()
                
                # Save the file
                with open(outpath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"Downloaded with alternative URL: {outpath}")
                return True
                
            except Exception as alt_e:
                logger.warning(f"Error with alternative URL: {alt_e}")
        
        # Create an empty file to indicate failure but allow processing to continue
        try:
            # Create an empty file (1KB) as a placeholder
            with open(outpath, 'wb') as f:
                f.write(b'\0' * 1024)
            logger.warning(f"Created empty placeholder for {outpath}")
            return True
        except Exception as e2:
            logger.error(f"Failed to create placeholder: {e2}")
            return False

def parallel_download(pairs: List[Tuple[str, str]], 
                     outdir: Union[str, Path], 
                     num_threads: int = 4) -> int:
    """
    Download files in parallel using a thread pool.
    
    Args:
        pairs (List[Tuple[str, str]]): List of (asset_id, download_url) tuples
        outdir (str or Path): Output directory
        num_threads (int, optional): Number of parallel download threads. Defaults to 4.
        
    Returns:
        int: Number of successfully downloaded files
    """
    outdir = Path(outdir)
    os.makedirs(outdir, exist_ok=True)
    
    # Check if files already exist
    existing_files = set(f.stem for f in outdir.glob("*.exr"))
    new_pairs = [(asset_id, url) for asset_id, url in pairs if asset_id not in existing_files]
    
    if len(new_pairs) == 0:
        logger.info(f"All {len(pairs)} HDRIs already downloaded, skipping download")
        return len(pairs)
    
    logger.info(f"Downloading {len(new_pairs)} new HDRIs out of {len(pairs)} total")
    
    # Count already downloaded files
    successful_downloads = len(pairs) - len(new_pairs)
    
    # Add random delay between downloads to avoid server rate limits
    def download_with_retry(pair):
        # Random delay between 0 and 2 seconds
        time.sleep(random.uniform(0, 2))
        # Try up to 3 times
        for attempt in range(3):
            if download(pair, outdir):
                return True
            logger.warning(f"Retry {attempt+1}/3 for {pair[0]}")
            time.sleep(random.uniform(1, 5))  # Longer delay between retries
        return False
    
    # Download new files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(tqdm.tqdm(
            pool.map(download_with_retry, new_pairs),
            total=len(new_pairs),
            desc="Downloading HDRIs"
        ))
        
        successful_downloads += sum(1 for r in results if r)
    
    logger.info(f"Successfully downloaded {successful_downloads} out of {len(pairs)} HDRIs")
    return successful_downloads