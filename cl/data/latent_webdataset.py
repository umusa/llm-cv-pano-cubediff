# -------- the data loader to load the latent tensors and captions from a webdataset ---------
import torch, webdataset as wds, io
import numpy as np
import io
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_LEN   = tokenizer.model_max_length

def get_dataloader(wds_path, batch_size, num_workers=4):
    def preprocess(sample):
        # sample: dict with keys "__key__", "<id>.pt", "<id>.txt"
        
        # 1) load the latent tensor
        pt_key = next(k for k in sample if k.endswith(".pt"))
        bytes_ = sample[pt_key]
        tensor = torch.load(io.BytesIO(bytes_))  # -> [6, C, H, W]
        
        # 2) load and tokenize the caption
        txt_key = next(k for k in sample if k.endswith(".txt"))
        caption = sample[txt_key].decode("utf-8")  # raw bytes → str
        toks = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        
        # squeeze off the extra batch dim
        return {
            "latent":        tensor,
            "input_ids":     toks.input_ids.squeeze(0),
            "attention_mask":toks.attention_mask.squeeze(0),
        }
    
    dataset = (
        wds.WebDataset(wds_path, handler=wds.warn_and_continue, empty_check=False)
          .shuffle(1000, initial=100)   # optional
          .map(preprocess)              # now returns dict[str,Tensor]
    )
    
    def collate_fn(batch):
        # batch is a list of dicts, so we stack each field
        latents       = torch.stack([b["latent"]        for b in batch], dim=0)
        input_ids     = torch.stack([b["input_ids"]     for b in batch], dim=0)
        attention_mask= torch.stack([b["attention_mask"]for b in batch], dim=0)
        return {
            "latent":        latents,
            "input_ids":     input_ids,
            "attention_mask":attention_mask,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # keep workers alive
        prefetch_factor=2,         # each worker preloads 2 batches
        collate_fn=collate_fn,
    )
