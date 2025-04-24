import torch, pathlib, json, numpy as np
from torch.utils.data import Dataset
from .dataset import CubemapDataset

class LatentCubemapDataset(Dataset):
    """
    Loads pre-encoded VAE latents stored as *.pt tensors.
    Shape: (6,4,64,64) for SD-1.5 @512px.
    """
    def __init__(self, latent_dir, caption_json, channels_first=True):
        self.latent_dir = pathlib.Path(latent_dir)
        self.items = sorted(self.latent_dir.glob("*.pt"))
        self.captions = json.load(open(caption_json))
        self.channels_first = channels_first

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        lat = torch.load(self.items[idx])               # already float16
        return {"faces": lat, "caption": self.captions[self.items[idx].stem]}
