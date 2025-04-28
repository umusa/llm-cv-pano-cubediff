# import torch, webdataset as wds
import torch, webdataset as wds, io
import numpy as np
# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     ds = (wds.WebDataset(wds_path)
#             .decode()          # no image decoding, just bytes
#             .to_tuple("*.pt", "*.txt")
#             .map_tuple(
#                  lambda x: torch.from_numpy(x).float(),  # (6,4,64,64)
#                  lambda t: t.decode()))
#     return torch.utils.data.DataLoader(
#             ds.batched(batch_per_gpu, partial=True),
#             num_workers=workers, pin_memory=True, persistent_workers=True)

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     ds = (wds.WebDataset(wds_path)
#           .decode()          # no image decoding, just bytes
#           # Rename keys to consistent names for downstream processing
#           .map(lambda sample: {
#               "lat.pt": sample[next(key for key in sample.keys() if key.endswith('.pt'))],
#               "txt": sample[next(key for key in sample.keys() if key.endswith('.txt'))]
#           })
#           .to_tuple("lat.pt", "txt")
#           .map_tuple(
#               lambda x: torch.from_numpy(x).float(),  # (6,4,64,64)
#               lambda t: t.decode()))
#     return torch.utils.data.DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         num_workers=workers, pin_memory=True, persistent_workers=True)

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     ds = (wds.WebDataset(wds_path)
#           # Don't decode here since we need to handle PT files specially
#           .map(lambda sample: {
#               # Find keys ending with .pt and .txt
#               "lat.pt": torch.load(io.BytesIO(sample[next(k for k in sample.keys() if k.endswith('.pt'))])),
#               "txt": sample[next(k for k in sample.keys() if k.endswith('.txt'))].decode('utf-8')
#           })
#           # Convert to tuple format after proper processing
#           .to_tuple("lat.pt", "txt"))
    
#     return torch.utils.data.DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         num_workers=workers, pin_memory=True, persistent_workers=True)

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     # Helper function to safely load numpy data
#     def load_pt_file(sample):
#         # Find the .pt key
#         pt_key = next(k for k in sample.keys() if k.endswith('.pt'))
#         # Get the binary data
#         pt_data = sample[pt_key]
        
#         # Load the data with a BytesIO buffer
#         with io.BytesIO(pt_data) as f:
#             # Try loading as NPZ first
#             try:
#                 npz = np.load(f)
#                 # If it's an NPZ file, get the first array
#                 if isinstance(npz, np.lib.npyio.NpzFile):
#                     # Use the first array in the NPZ file
#                     array_name = npz.files[0]
#                     return torch.from_numpy(npz[array_name]).float()
#             except:
#                 # Reset the buffer position
#                 f.seek(0)
#                 # Try loading as regular NPY
#                 return torch.from_numpy(np.load(f)).float()
    
#     # Process the dataset
#     ds = (wds.WebDataset(wds_path)
#           .map(lambda sample: {
#               "lat.pt": load_pt_file(sample),
#               "txt": sample[next(k for k in sample.keys() if k.endswith('.txt'))].decode('utf-8')
#           }))
    
#     loader = torch.utils.data.DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         num_workers=workers, pin_memory=True, persistent_workers=True)
    
#     return loader

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     # WebDataset for the tar file containing PT tensors
#     ds = (wds.WebDataset(wds_path)
#           .map(lambda sample: {
#               # Find the PT files and load them with torch.load
#               "lat.pt": torch.load(io.BytesIO(sample[next(k for k in sample.keys() if k.endswith('.pt'))])),
#               "txt": sample[next(k for k in sample.keys() if k.endswith('.txt'))].decode('utf-8')
#           }))
    
#     # Create DataLoader with appropriate settings
#     loader = torch.utils.data.DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         num_workers=workers, pin_memory=True, persistent_workers=True)
    
#     return loader

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     # WebDataset for the tar file containing PT tensors
#     ds = (wds.WebDataset(wds_path)
#         .map(lambda sample: {
#             # Find the PT files and load them with torch.load
#             # Make sure we're only grabbing one .pt file per sample
#             "lat.pt": torch.load(io.BytesIO(sample[next(k for k in sample.keys() if k.endswith('.pt'))])),
#             # Ensure text files are properly found and decoded
#             "txt": sample[next(k for k in sample.keys() if k.endswith('.txt'))].decode('utf-8') if any(k.endswith('.txt') for k in sample.keys()) else ""
#         })
#     )
    
#     # Create DataLoader with appropriate settings
#     loader = torch.utils.data.DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         num_workers=workers, pin_memory=True, persistent_workers=True
#     )
    
#     return loader


# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     import webdataset as wds
#     import io
#     import torch
#     from torch.utils.data import DataLoader
    
#     # Define a decoding function that handles potential errors
#     def decoder(sample):
#         try:
#             # Find a PT file in the sample
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
#             # Skip samples without both PT and TXT files
#             if not pt_keys or not txt_keys:
#                 return None
                
#             # Get the first PT and TXT files
#             pt_key = pt_keys[0]
#             txt_key = txt_keys[0]
            
#             # Load the PT tensor
#             tensor = torch.load(io.BytesIO(sample[pt_key]))
#             # Decode the text
#             text = sample[txt_key].decode('utf-8')
            
#             # Return a dictionary with the processed data
#             return {
#                 "latent": tensor,
#                 "text": text
#             }
#         except Exception as e:
#             print(f"Error decoding sample: {e}")
#             return None
    
#     # Create the WebDataset with simpler pipeline
#     # Instead of trying to filter, we'll handle invalid samples in the training loop
#     ds = wds.WebDataset(wds_path).map(decoder)
    
#     # Create a DataLoader
#     loader = DataLoader(
#         ds.batched(batch_per_gpu, partial=True),
#         batch_size=None,  # Important with WebDataset batched
#         num_workers=workers,
#         pin_memory=True,
#         persistent_workers=True if workers > 0 else False
#     )
    
#     return loader

# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     import webdataset as wds
#     import io
#     import torch
#     from torch.utils.data import DataLoader
    
#     # Custom collate function to handle mixed tensor and string types
#     def collate_fn(batch):
#         # Separate latents and texts
#         latents = [item["latent"] for item in batch]
#         texts = [item["text_str"] for item in batch]
        
#         # Stack tensors
#         latents = torch.stack(latents)
        
#         # Return without trying to stack texts
#         return {
#             "latent": latents,
#             "text_str": texts  # List of strings
#         }
    
#     # Load and process samples
#     def decode_sample(sample):
#         try:
#             # Find PT and TXT files
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
#             if not pt_keys or not txt_keys:
#                 return None
                
#             pt_key = pt_keys[0]
#             txt_key = txt_keys[0]
            
#             # Load the PT tensor
#             tensor = torch.load(io.BytesIO(sample[pt_key]))
#             text = sample[txt_key].decode('utf-8')
            
#             return {
#                 "latent": tensor,
#                 "text_str": text
#             }
#         except Exception as e:
#             print(f"Error decoding sample: {e}")
#             return None
    
#     # Create dataset with filter
#     ds = wds.WebDataset(wds_path).map(decode_sample) #.filter(lambda x: x is not None)
    
#     # Create DataLoader with custom collate function
#     loader = DataLoader(
#         ds,  # Note: No batching at WebDataset level
#         batch_size=batch_per_gpu,  # Use PyTorch's batching instead
#         collate_fn=collate_fn,  # Custom collate function
#         num_workers=workers,
#         pin_memory=True,
#         persistent_workers=True if workers > 0 else False
#     )
    
#     return loader

# def get_dataloader(wds_path, batch_per_gpu=1, workers=4):
#     import webdataset as wds
#     import torch
#     from torch.utils.data import DataLoader
#     print(f"latent_webdataset.py - line 231 - get_dataloader - wds_path: {wds_path} , batch_per_gpu is {batch_per_gpu}, workers is {workers}")
#     # Define how to decode samples from WebDataset format
#     def decode_fn(sample):
#         """Decode a WebDataset sample to a standardized format."""
#         # Find the tensor and text files using standard WebDataset naming
#         pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#         txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
        
#         if not pt_keys or not txt_keys:
#             # Skip samples without both tensor and text
#             return None
            
#         # Get the first tensor file and text file
#         tensor_key = pt_keys[0]
#         text_key = txt_keys[0]
        
#         # The tensor is already loaded by WebDataset's decode method
#         tensor = sample[tensor_key]
        
#         # The text is also already decoded
#         text = sample[text_key]
        
#         return {
#             "latent": tensor,
#             "text": text
#         }
    
#    # Custom collate function to handle WebDataset samples
#     def collate_fn(batch):
#         """Custom collation function that properly converts byte data to tensors."""
#         latents = []
#         texts = []
        
#         for sample in batch:
#             # Process tensor data
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
#             if not pt_keys or not txt_keys:
#                 continue
            
#             pt_key = pt_keys[0]
#             txt_key = txt_keys[0]
            
#             # Handle tensor data - could be bytes or already a tensor
#             tensor_data = sample[pt_key]
#             if isinstance(tensor_data, bytes):
#                 # Convert bytes to tensor
#                 buffer = io.BytesIO(tensor_data)
#                 tensor = torch.load(buffer)
#             elif isinstance(tensor_data, torch.Tensor):
#                 tensor = tensor_data
#             else:
#                 # Skip this sample if tensor data is invalid
#                 continue
                
#             # Handle text data
#             text = sample[txt_key]
#             if isinstance(text, bytes):
#                 text = text.decode('utf-8')
                
#             # Add to batch lists
#             latents.append(tensor)
#             texts.append(text)
        
#         if not latents:
#             # Return empty batch if no valid samples
#             return {"latent": torch.empty(0, 6, 4, 64, 64), "text": []}
            
#         # Stack tensors into a batch
#         latents_batch = torch.stack(latents)
        
#         return {
#             "latent": latents_batch,
#             "text": texts
#         }
    
#     # Create a WebDataset pipeline
#     dataset = wds.WebDataset(wds_path)
    
#     # Create DataLoader with custom collate function
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_per_gpu,
#         collate_fn=collate_fn,
#         num_workers=workers,
#         pin_memory=True,
#         persistent_workers=True if workers > 0 else False
#     )
    
#     return loader


# def get_dataloader(wds_path, batch_per_gpu, workers=4):
#     import webdataset as wds
#     import torch
#     import io
#     from torch.utils.data import DataLoader
    
#     # First, define a pre-processing function that ensures all samples have the right format
#     def preprocess_sample(sample):
#         """Convert raw WebDataset sample to properly structured data."""
#         try:
#             # Find tensor and text files
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
#             if not pt_keys or not txt_keys:
#                 return None
                
#             pt_key = pt_keys[0]
#             txt_key = txt_keys[0]
            
#             # Load tensor from bytes if needed
#             tensor_data = sample[pt_key]
#             if isinstance(tensor_data, bytes):
#                 buffer = io.BytesIO(tensor_data)
#                 tensor = torch.load(buffer)
#             elif isinstance(tensor_data, torch.Tensor):
#                 tensor = tensor_data
#             else:
#                 return None
                
#             # Get text data
#             text_data = sample[txt_key]
#             if isinstance(text_data, bytes):
#                 text = text_data.decode('utf-8')
#             else:
#                 text = str(text_data)
                
#             # Return as dictionary with properly typed elements
#             return {
#                 "latent": tensor,
#                 "text_idx": 0  # Placeholder - will be replaced in collate_fn
#             }
            
#         except Exception as e:
#             print(f"Error preprocessing sample: {e}")
#             return None
    
#     # Define a collate function that handles string data separately
#     def collate_fn(batch):
#         """Custom collate function that separates tensor and string data."""
#         # Filter out None samples
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return {"latent": torch.empty(0, 6, 4, 64, 64), "text": []}
        
#         # Extract and process tensors
#         latents = [item["latent"] for item in batch]
        
#         # Collect text data separately - not for concatenation
#         text_data = []
#         for i, item in enumerate(batch):
#             # Get original text from the sample
#             if hasattr(item, "text"):
#                 text_data.append(item["text"])
#             else:
#                 # Default empty string if missing
#                 text_data.append("")
        
#         # Stack tensors
#         latents_batch = torch.stack(latents)
        
#         # Return properly structured data for accelerate
#         # Keep text_data as a separate list, not in the main tensor dictionary
#         return {
#             "latent": latents_batch,  # Tensor of shape [B, 6, 4, 64, 64]
#             "text": text_data  # Separate list, not part of tensor operations
#         }
    
#     # Create the WebDataset with proper preprocessing
#     dataset = (
#         wds.WebDataset(wds_path)
#         .decode()
#         .map(preprocess_sample)
#         .select(lambda x: x is not None)
#     )
    
#     # Create the DataLoader
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_per_gpu,
#         collate_fn=collate_fn,
#         num_workers=workers,
#         pin_memory=True,
#         persistent_workers=True if workers > 0 else False
#     )
    
#     return loader

# def get_dataloader(wds_path, batch_per_gpu, workers=0):
#     import webdataset as wds
#     import torch
#     import io
#     from torch.utils.data import DataLoader
    
#     # Process samples to extract tensors and return them in a dictionary
#     def preprocess_sample(sample):
#         try:
#             # Find tensor files
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
#             if not pt_keys:
#                 return None
                
#             # Load tensor data
#             tensor_data = sample[pt_keys[0]]
#             if isinstance(tensor_data, bytes):
#                 buffer = io.BytesIO(tensor_data)
#                 tensor = torch.load(buffer)
#             else:
#                 tensor = tensor_data
            
#             # Get caption if available
#             caption = ""
#             if txt_keys:
#                 text_data = sample[txt_keys[0]]
#                 if isinstance(text_data, bytes):
#                     caption = text_data.decode('utf-8')
#                 else:
#                     caption = str(text_data)
            
#             # Return a dictionary with the tensor to match expected structure
#             return {
#                 "latent": tensor,
#                 "caption": caption
#             }
            
#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             return None
    
#     # Custom collate function that maintains dictionary structure
#     def collate_fn(batch):
#         # Filter out None values
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return {"latent": torch.empty(0)}
        
#         # Extract tensors and captions
#         tensors = [item["latent"] for item in batch]
#         captions = [item["caption"] for item in batch]
        
#         # Stack tensors
#         try:
#             tensor_batch = torch.stack(tensors)
#         except Exception as e:
#             print(f"Error stacking batch: {e}")
#             return {"latent": torch.empty(0)}
        
#         # Return dictionary structure that matches what the trainer expects
#         return {
#             "latent": tensor_batch,
#             "caption": captions
#         }
    
#     # Create WebDataset
#     dataset = (
#         wds.WebDataset(
#             wds_path,
#             shardshuffle=False,
#             handler=wds.handlers.warn_and_continue
#         )
#         .decode()
#         .map(preprocess_sample)
#         .select(lambda x: x is not None)
#     )
    
#     # Create DataLoader
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_per_gpu,
#         collate_fn=collate_fn,
#         num_workers=workers,
#         pin_memory=True
#     )
    
#     return loader

# import torch
# import io
# import webdataset as wds
# from torch.utils.data import DataLoader

# def get_dataloader(wds_path, batch_per_gpu, workers=0):
#     # Process samples while preserving the original filename keys
#     def preprocess_sample(sample):
#         try:
#             # Get the key that ends with .pt (the original filename)
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
            
#             if not pt_keys:
#                 return None
                
#             # Load tensor data
#             pt_key = pt_keys[0]  # Original .pt filename
#             tensor_data = sample[pt_key]
            
#             if isinstance(tensor_data, bytes):
#                 buffer = io.BytesIO(tensor_data)
#                 tensor = torch.load(buffer)
#             else:
#                 tensor = tensor_data
            
#             # Create a dictionary with the original key preserved
#             # This is critical - the training code expects keys ending with .pt
#             result = {pt_key: tensor}
            
#             # Also get caption if available
#             txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
#             if txt_keys:
#                 txt_key = txt_keys[0]
#                 text_data = sample[txt_key]
#                 if isinstance(text_data, bytes):
#                     caption = text_data.decode('utf-8')
#                 else:
#                     caption = str(text_data)
                    
#                 result[txt_key] = caption
            
#             return result
            
#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             return None
    
#     # Simple collate function that merges dictionaries
#     def collate_fn(batch):
#         # Filter out None values
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return {}
        
#         # Merge all dictionaries, preserving original keys
#         result = {}
#         for item in batch:
#             for k, v in item.items():
#                 if k not in result:
#                     result[k] = []
#                 result[k].append(v)
        
#         return result
    
#     # Create WebDataset
#     dataset = (
#         wds.WebDataset(
#             wds_path,
#             shardshuffle=False,
#             handler=wds.handlers.warn_and_continue
#         )
#         .decode()
#         .map(preprocess_sample)
#         .select(lambda x: x is not None)
#     )
    
#     # Create DataLoader
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_per_gpu,
#         collate_fn=collate_fn,
#         num_workers=workers,
#         pin_memory=True
#     )
    
#     return loader

# import torch
# import io
# import webdataset as wds
# from torch.utils.data import DataLoader

# # Define at module level so it can be properly pickled
# class TensorBatchWrapper(dict):
#     """A dictionary wrapper that provides .pt keys for tensors."""
#     def __init__(self, tensors=None):
#         super().__init__()
#         if tensors is not None:
#             for i, tensor in enumerate(tensors):
#                 self[f"tensor_{i}.pt"] = tensor

# def get_dataloader(wds_path, batch_per_gpu, workers=0):
#     # Process samples to extract only tensors
#     def preprocess_sample(sample):
#         try:
#             # Find tensor files
#             pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
            
#             if not pt_keys:
#                 return None
                
#             # Load tensor data
#             tensor_data = sample[pt_keys[0]]
#             if isinstance(tensor_data, bytes):
#                 buffer = io.BytesIO(tensor_data)
#                 tensor = torch.load(buffer)
#             else:
#                 tensor = tensor_data
            
#             # Return only the tensor
#             return tensor
            
#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             return None
    
#     # Collate function that creates a wrapper with .pt keys
#     def collate_fn(batch):
#         # Filter out None values
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return TensorBatchWrapper()
        
#         # Return a dictionary with .pt keys for each tensor
#         return TensorBatchWrapper(batch)
    
#     # Create WebDataset with minimal configuration
#     dataset = (
#         wds.WebDataset(
#             wds_path,
#             shardshuffle=False
#         )
#         .decode()
#         .map(preprocess_sample)
#         .select(lambda x: x is not None)
#     )
    
#     # Create DataLoader with single worker initially
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_per_gpu,
#         collate_fn=collate_fn,
#         num_workers=0,  # Start with 0 to avoid multiprocessing issues
#         pin_memory=True
#     )
    
#     return loader

import io
import torch
import webdataset as wds
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
        wds.WebDataset(wds_path)
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
        collate_fn=collate_fn,
    )

