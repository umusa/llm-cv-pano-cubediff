# -------- the data loader to load the latent tensors and captions from a webdataset ---------
import os
import io
import torch
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_LEN = tokenizer.model_max_length

def get_dataloader(wds_path, batch_size, data_size, num_workers=8):
    # ==== EXPLICIT NODE SPLITTER FOR DDP ====
    # Accelerate or torchrun will set RANK and WORLD_SIZE
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    def nodesplitter(urls):
        urls = list(urls)
        # Round-robin assign shards to this process
        return [u for idx, u in enumerate(urls) if (idx % world_size) == rank]

    def preprocess(sample):
        # 1) load the latent tensor
        pt_key = next(k for k in sample if k.endswith(".pt"))
        tensor = torch.load(io.BytesIO(sample[pt_key]))  # -> [6, C, H, W]
        # 2) load and tokenize the caption
        txt_key = next(k for k in sample if k.endswith(".txt"))
        caption = sample[txt_key].decode("utf-8")
        toks = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "latent":         tensor,
            "input_ids":      toks.input_ids.squeeze(0),
            "attention_mask": toks.attention_mask.squeeze(0),
        }

    # Build WebDataset pipeline with explicit nodesplitter
    dataset = (
        wds.WebDataset(
            urls=wds_path,
            nodesplitter=nodesplitter,
            handler=wds.warn_and_continue,
            empty_check=False
        )
        .shuffle(1000, initial=100)
        .map(preprocess)
        .slice(data_size)
    )

    def collate_fn(batch):
        latents        = torch.stack([b["latent"]         for b in batch], dim=0)
        input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        return {
            "latent":         latents,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # overlap host→device copies
        persistent_workers=True, # keep workers alive between batches
        prefetch_factor=8,  # # how many batches each worker preloads
        collate_fn=collate_fn,
    )
