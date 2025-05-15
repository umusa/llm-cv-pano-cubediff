"""
2025-5-13 
the original data was added one mask channel, which should be done in CubbeDiffModel/forward
so, remove the mask channel with this script

Build a tiny dataset for CubeDiff from Polyhaven HDRIs.

Usage:
  # Only convert/encode new panoramas, skip re-download:
  python -m build_tiny_set \
      --out /home/jupyter/mluser/git/llm-cv-pano-cubediff/cl/data/dataspace/polyhaven_tiny \
      --skip_download \
      --skip_convert

  # To re-run latent encoding on missing ones:
  python -m build_tiny_set \
      --out /home/jupyter/mluser/git/llm-cv-pano-cubediff/cl/data/dataspace/polyhaven_tiny \
      --skip_download \
      --skip_convert
"""

import argparse, pathlib, sys, os, json, logging, tqdm
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import torch
import os
os.environ["PYTHONPATH"] = "/home/jupyter/mluser/git/llm-cv-pano-cubediff"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from diffusers import AutoencoderKL

# imports for cubemap conversion
from cl.data.polyhaven.api_client import list_hdris, parallel_download
from cl.data.polyhaven.cubemap_builder import batch_convert
# import v2 model tools
# from cl.model.normalization import replace_group_norms


def needs_encoding(pano_name: str,
                   faces_dir: Path,
                   latent_dir: Path,
                   expected_faces: int = 6,
                   expected_channels: int = 4) -> bool:
    """
    Return True if we should (re-)encode this panorama:
      • No .pt exists
      • OR its stored tensor has the wrong shape
      • OR any face.jpg is newer than the latent file
    """
    latent_path = latent_dir / f"{pano_name}.pt"
    # 1) missing → must encode
    if not latent_path.exists():
        return True

    # 2) shape mismatch → must re-encode
    try:
        # only load metadata (fast) by map_location='cpu'
        lt = torch.load(latent_path, map_location="cpu")
        # correct shape is [F, C, H, W]
        if lt.ndim != 4 or lt.shape[0] != expected_faces or lt.shape[1] != expected_channels:
            return True
    except Exception:
        # if loading fails, just re-encode
        return True

    # 3) out-of-date → must re-encode
    latent_mtime = latent_path.stat().st_mtime
    face_folder  = faces_dir / pano_name
    for img in face_folder.glob("*.jpg"):
        if img.stat().st_mtime > latent_mtime:
            return True

    # otherwise it’s fresh & correct
    return False


def encode_faces(face_dir, latent_dir, batch=6):
    """Encode only *missing* cubemap latents, skip existing .pt files."""
    import torch
    from diffusers import StableDiffusionPipeline
    from cl.data.dataset import CubemapDataset
    from cl.model.normalization import replace_group_norms
    from cl.data.polyhaven.cubemap_builder import FACE_ORDER

    latent_dir = Path(latent_dir); latent_dir.mkdir(exist_ok=True)
    # Load only VAE in FP16 on GPU
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=torch.float16,
        revision=None
    ).to("cuda").eval()

    # SyncGroupNorm for cubemap consistency
    #*do not* sync‐GN the VAE but only the UNet needs that
    # replace_group_norms(vae, in_place=True)

    # Dataset of *all* faces
    ds = CubemapDataset(face_dir, channels_first=True, image_size=512)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
    new_count, no_need_cencode = 0, 0
    # Rename loop vars so we don’t shadow `batch_size`
    for batch_idx, batch_data in enumerate(tqdm.tqdm(loader, desc="Encoding latents")):
        faces = batch_data['faces'].to('cuda').half()  # [B,6,3,512,512]
        lat_batch = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            lat_samples = []
            for f in faces:  # f.shape == [6,3,512,512]
                per_face_lat = []
                for j in range(6):
                    img = f[j].unsqueeze(0)            # [1,3,512,512]
                    lat = vae.encode(img).latent_dist.sample() * 0.18215
                    per_face_lat.append(lat)          # each lat is [1,4,64,64]
                # ✅ stack into a true face-dimension → [1,6,4,64,64]
                # stack → [1,6,4,64,64], cat → [1,24,64,64]
                sample_lat = torch.stack(per_face_lat, dim=1)
                lat_samples.append(sample_lat)
            # now lat_samples is a list of [1,6,4,64,64]; concat into [B,6,4,64,64]
            lat_batch = torch.cat(lat_samples, dim=0).cpu()
            print(f"encode_faces: lat_batch.shape={lat_batch.shape}")

        # Compute the start offset for this minibatch
        start = batch_idx * batch
        for idx, latent in enumerate(lat_batch):
            # pano = ds.pano_dirs[start + idx]
            # if pano in done:
            #     continue

            # now per-sample:
            pano = ds.pano_dirs[start + idx]
            # skip only if latent exists and shape is correct or faces aren’t newer
            if not needs_encoding(pano, face_dir, latent_dir):
                no_need_cencode += 1
                continue

            torch.save(latent, Path(latent_dir) / f"{pano}.pt")
            new_count += 1

    logger.info(f"Encoded {new_count} new panoramas to generate latents by vae but (skipped {no_need_cencode})")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        required=True, help="Dataset root dir")
    ap.add_argument("--n",    type=int, default=700,      help="Max HDRIs")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_convert",  action="store_true")
    ap.add_argument("--skip_latents",  action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    erp_dir   = out / "raw" / "erp"
    faces_dir = out / "raw" / "faces"
    latent_dir= out / "latents"

    erp_dir.parent.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    # 1) download
    if args.skip_download or any(erp_dir.glob("*.exr")):
        logger.info("Skipping HDRI download")
    else:
        pairs = list_hdris(args.n)
        successful = parallel_download(pairs, erp_dir)
        logger.info(f"Downloaded {successful} HDRIs")

    # 2) convert
    if args.skip_convert or any(faces_dir.glob("*/*.jpg")):
        logger.info("Skipping cubemap conversion")
    else:
        num = batch_convert(erp_dir, faces_dir, face_px=512)
        logger.info(f"Converted {num} panoramas to faces")

    # 3) captions
    caps = out / "raw" / "captions.json"
    if not caps.exists():
        hd = {p.stem: p.stem.replace("_"," ") for p in erp_dir.glob("*.exr")}
        caps.write_text(json.dumps(hd,indent=2))
        logger.info(f"Wrote {len(hd)} captions")

    # 4) encode latents
    if args.skip_latents:
        logger.info("Skipping latent encoding")
    else:
        encode_faces(faces_dir, latent_dir)

if __name__ == "__main__":
    main()
