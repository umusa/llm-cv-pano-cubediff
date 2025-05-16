# -------- the data loader to load the latent tensors and captions from a webdataset ---------
# import os
# import io
# import torch
# import numpy as np
# import webdataset as wds
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer

# # Initialize tokenizer once
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# MAX_LEN = tokenizer.model_max_length


# def get_dataloader(wds_path, batch_size, data_size, num_workers=8):
#     # ==== EXPLICIT NODE SPLITTER FOR DDP ====
#     # Accelerate or torchrun will set RANK and WORLD_SIZE
#     rank = int(os.environ.get("RANK", 0))
#     world_size = int(os.environ.get("WORLD_SIZE", 1))

#     def nodesplitter(urls):
#         urls = list(urls)
#         # Round-robin assign shards to this process
#         return [u for idx, u in enumerate(urls) if (idx % world_size) == rank]

#     def preprocess(sample):
#         # 1) load the latent tensor
#         pt_key = next(k for k in sample if k.endswith(".pt"))
#         tensor = torch.load(io.BytesIO(sample[pt_key]))  # -> [6, C, H, W]
#         # 2) load and tokenize the caption
#         txt_key = next(k for k in sample if k.endswith(".txt"))
#         caption = sample[txt_key].decode("utf-8")
#         toks = tokenizer(
#             caption,
#             padding="max_length",
#             truncation=True,
#             max_length=MAX_LEN,
#             return_tensors="pt"
#         )
#         return {
#             "latent":         tensor,   # [6,4,H,W]
#             "input_ids":      toks.input_ids.squeeze(0),
#             "attention_mask": toks.attention_mask.squeeze(0),
#         }

#     # Build WebDataset pipeline with explicit nodesplitter
#     dataset = (
#         wds.WebDataset(
#             urls=wds_path,
#             nodesplitter=nodesplitter,
#             handler=wds.warn_and_continue,
#             empty_check=False
#         )
#         .shuffle(1000, initial=100)
#         .map(preprocess)
#         .slice(data_size)
#     )

#     # def collate_fn(batch):
#     #     latents        = torch.stack([b["latent"]         for b in batch], dim=0)  # → [B, 6, 4, H, W]
#     #     # ── CubeDiff requires a 1-channel “mask” (1=clean/text conditioning)
#     #     #     so that conv_in (patched for uv_dim=9 + mask_ch=1) sees 14 channels
#     #     B, F, C, H, W = latents.shape
#     #     mask = torch.ones((B, F, 1, H, W), 
#     #                       dtype=latents.dtype, 
#     #                       device=latents.device)
#     #     latents = torch.cat([latents, mask], dim=2)             # → [B, 6, 5, H, W]
#     #     input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0)
#     #     attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
#     #     return {
#     #         "latent":         latents,
#     #         "input_ids":      input_ids,
#     #         "attention_mask": attention_mask,
#     #     }

#     def collate_fn(batch):
#         # Stack per-sample tensors into a batch
#         latents        = torch.stack([b["latent"]         for b in batch], dim=0)  # → [B, 6, 4, H, W]

#         # ── CubeDiff requires a 1-channel “mask” indicating which latents
#         #     are clean (text-conditioned) vs. noisy.  Without it, the U-Net
#         #     falls back to an unconditional prior and you get pure noise.
#         B, F, C, H, W = latents.shape
#         # make a mask of all ones (text-only setting)
#         mask = torch.ones((B, F, 1, H, W),
#                           dtype=latents.dtype,
#                           device=latents.device)
#         # append as a 5th channel per face → [B, 6, 5, H, W]
#         latents = torch.cat([latents, mask], dim=2)

#         input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0)
#         attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

#         return {
#             "latent":         latents,        # [B,6,5,H,W]
#             "input_ids":      input_ids,
#             "attention_mask": attention_mask,
#         }


#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,  # overlap host→device copies
#         persistent_workers=True, # keep workers alive between batches
#         prefetch_factor=8,  # # how many batches each worker preloads
#         collate_fn=collate_fn,
#     )


import os, io, torch, webdataset as wds
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
MAX_LEN  = tokenizer.model_max_length

def get_dataloader(wds_path, batch_size, data_size, num_workers=8):
    rank       = int(os.environ.get("RANK",0))
    world_size = int(os.environ.get("WORLD_SIZE",1))

    def nodesplitter(urls):
        return [u for idx,u in enumerate(urls) if idx % world_size == rank]

    def preprocess(sample):
        # load latent: [6,4,H,W]
        pt_key = next(k for k in sample if k.endswith(".pt"))
        lat = torch.load(io.BytesIO(sample[pt_key]))  # -> [6,4,H,W]

        # load caption
        txt_key = next(k for k in sample if k.endswith(".txt"))
        cap     = sample[txt_key].decode("utf-8")
        toks    = tokenizer(cap, padding="max_length",
                            truncation=True,
                            max_length=MAX_LEN,
                            return_tensors="pt")
        return {
            "latent":         lat,
            "input_ids":      toks.input_ids.squeeze(0),
            "attention_mask": toks.attention_mask.squeeze(0),
        }

    def collate_fn(batch):
        latents = torch.stack([b["latent"]         for b in batch], dim=0)  # [B,6,4,H,W]
        print(f"latent_webdataset.py - get_dataloader - collate_fn - latents.shape: {latents.shape}")
        input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0) # [B,seq_len]
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0) # [B,seq_len]
        return {
            "latent":         latents,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

    ds = (
        wds.WebDataset(
            urls=wds_path, nodesplitter=nodesplitter,
            handler=wds.warn_and_continue
        )
        .shuffle(1000, initial=100)
        .map(preprocess)
        .slice(data_size)
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )



