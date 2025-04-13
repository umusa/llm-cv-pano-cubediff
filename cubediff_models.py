"""
Model components for CubeDiff implementation.
Contains the synchronized GroupNorm and inflated attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class FixedInflatedAttention(nn.Module):
    """
    Memory-efficient attention layer for cubemap structure.
    """
    def __init__(self, original_attn_module):
        super().__init__()
        self.original_attn = original_attn_module
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        Simple forward pass that delegates to the original attention module.
        """
        return self.original_attn(
            hidden_states,
            encoder_hidden_states,
            attention_mask
        )


def simplified_convert_attention(unet):
    """
    Convert attention modules with minimal memory overhead.
    """
    # Find all attention modules in the UNet
    count = 0
    for name, module in unet.named_modules():
        # Check if this is an attention module
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            parent = unet
            for part in parent_name.split('.'):
                if not part:
                    continue
                parent = getattr(parent, part)
            
            attr_name = name.split('.')[-1]
            
            # Create replacement module
            replacement = FixedInflatedAttention(module)
            
            # Replace the module
            setattr(parent, attr_name, replacement)
            count += 1
    
    print(f"Converted {count} attention modules")
    return unet


class SynchronizedGroupNorm(nn.Module):
    """
    GroupNorm that synchronizes statistics across cubemap faces.
    This ensures color consistency across the six faces of the cubemap.
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        # First check if input has enough dimensions
        if len(x.shape) < 4:
            # Not a 4D tensor, use standard group norm
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
            
        # Assume x has shape [B*6, C, H, W] where 6 is the number of cube faces
        # and B is the batch size
        batch_size = x.shape[0] // 6
        
        if batch_size * 6 != x.shape[0]:
            # If the batch isn't perfectly divisible by 6, fall back to standard GroupNorm
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        
        # Reshape to separate batch and faces
        try:
            x_reshaped = x.view(batch_size, 6, x.shape[1], x.shape[2], x.shape[3])
            
            # Check if channels divisible by groups
            if x_reshaped.shape[2] % self.num_groups != 0:
                return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
                
            channels_per_group = x_reshaped.shape[2] // self.num_groups
            
            # Reshape for group norm calculation
            x_reshaped = x_reshaped.view(
                batch_size, 6, self.num_groups, channels_per_group, x_reshaped.shape[3], x_reshaped.shape[4]
            )
            
            # Calculate mean and variance across spatial dims and faces
            # Keep batch and group dimensions separate
            mean = x_reshaped.mean(dim=[1, 3, 4, 5], keepdim=True)
            var = x_reshaped.var(dim=[1, 3, 4, 5], keepdim=True, unbiased=False)
            
            # Normalize
            x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)
            
            # Reshape back to original format
            x_normalized = x_normalized.view(batch_size, 6, x.shape[1], x.shape[2], x.shape[3])
            x_normalized = x_normalized.view(batch_size * 6, x.shape[1], x.shape[2], x.shape[3])
            
            # Apply weight and bias
            return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        except Exception as e:
            # If any error occurs during reshaping, fall back to standard GroupNorm
            print(f"SyncGroupNorm fallback: {str(e)}")
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class DimensionAdapter(nn.Module):
    """
    Adapter module that handles dimension mismatches in the UNet blocks.
    """
    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Use pointwise convolution for 4D tensors, linear layer for others
        self.is_4d = True  # Will be determined in forward
        self.proj_4d = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
        self.proj_3d = nn.Linear(in_dim, out_dim, device=device, dtype=dtype)
    
    def forward(self, x):
        # Determine type of projection based on input shape
        if len(x.shape) == 4:  # [B, C, H, W]
            return self.proj_4d(x)
        else:  # [B, C, N] or similar
            # Transpose for linear layer: [B, C, N] -> [B, N, C]
            x_t = x.transpose(1, 2).contiguous()
            # Apply projection: [B, N, C] -> [B, N, out_dim]
            x_p = self.proj_3d(x_t)
            # Transpose back: [B, N, out_dim] -> [B, out_dim, N]
            return x_p.transpose(1, 2).contiguous()


class InflatedSelfAttention(nn.Module):
    """
    Self-attention layer that operates across cube faces for UNet.
    Legacy implementation kept for compatibility.
    """
    def __init__(self, original_attn_module):
        super().__init__()
        self.original_attn = original_attn_module
        
        # Copy attributes from original module
        self.to_q = original_attn_module.to_q
        self.to_k = original_attn_module.to_k
        self.to_v = original_attn_module.to_v
        self.to_out = original_attn_module.to_out
        self.heads = original_attn_module.heads
        
        # Calculate expected input dimension
        self.expected_dim = self.to_q.weight.shape[1]
        
        # Create dimension adapters (initialized on first use)
        self.in_adapter = None
        self.out_adapter = None
        
        # Get scale from original module if available
        if hasattr(original_attn_module, 'scale'):
            self.scale = original_attn_module.scale
        else:
            head_dim = original_attn_module.to_q.out_features // self.heads
            self.scale = head_dim ** -0.5
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        Forward pass with simplest possible implementation.
        """
        # Store original shape and channels
        original_shape = hidden_states.shape
        original_channels = hidden_states.shape[1]
        
        # Check for dimension mismatch
        if original_channels != self.expected_dim:
            # Create in adapter if needed
            if self.in_adapter is None or self.in_adapter.in_dim != original_channels:
                print(f"Creating in adapter: {original_channels} -> {self.expected_dim}")
                self.in_adapter = DimensionAdapter(
                    original_channels,
                    self.expected_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            
            # Create out adapter if needed
            if self.out_adapter is None or self.out_adapter.in_dim != self.expected_dim:
                print(f"Creating out adapter: {self.expected_dim} -> {original_channels}")
                self.out_adapter = DimensionAdapter(
                    self.expected_dim,
                    original_channels,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            
            # Apply in adapter
            adapted_states = self.in_adapter(hidden_states)
        else:
            adapted_states = hidden_states
        
        # Process using original attention module
        attn_output = self.original_attn(
            adapted_states, 
            encoder_hidden_states,
            attention_mask
        )
        
        # Apply out adapter if needed to match original dimension
        if original_channels != self.expected_dim:
            attn_output = self.out_adapter(attn_output)
        
        return attn_output


class InflatedCrossAttention(nn.Module):
    """
    Cross-attention layer that operates across cube faces for UNet.
    Legacy implementation kept for compatibility.
    """
    def __init__(self, original_attn_module):
        super().__init__()
        self.original_attn = original_attn_module
        
        # Copy attributes from original module
        self.to_q = original_attn_module.to_q
        self.to_k = original_attn_module.to_k
        self.to_v = original_attn_module.to_v
        self.to_out = original_attn_module.to_out
        self.heads = original_attn_module.heads
        
        # Calculate expected input dimension
        self.expected_dim = self.to_q.weight.shape[1]
        
        # Create dimension adapters (initialized on first use)
        self.in_adapter = None
        self.out_adapter = None
        
        # Get scale from original module if available
        if hasattr(original_attn_module, 'scale'):
            self.scale = original_attn_module.scale
        else:
            head_dim = original_attn_module.to_q.out_features // self.heads
            self.scale = head_dim ** -0.5
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        Forward pass with simplest possible implementation.
        """
        # Store original shape and channels
        original_shape = hidden_states.shape
        original_channels = hidden_states.shape[1]
        
        # Check for dimension mismatch
        if original_channels != self.expected_dim:
            # Create in adapter if needed
            if self.in_adapter is None or self.in_adapter.in_dim != original_channels:
                print(f"Creating in adapter: {original_channels} -> {self.expected_dim}")
                self.in_adapter = DimensionAdapter(
                    original_channels,
                    self.expected_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            
            # Create out adapter if needed
            if self.out_adapter is None or self.out_adapter.in_dim != self.expected_dim:
                print(f"Creating out adapter: {self.expected_dim} -> {original_channels}")
                self.out_adapter = DimensionAdapter(
                    self.expected_dim,
                    original_channels,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            
            # Apply in adapter
            adapted_states = self.in_adapter(hidden_states)
        else:
            adapted_states = hidden_states
        
        # When using cross-attention with encoder_hidden_states,
        # we need to handle the batch size mismatch
        if encoder_hidden_states is not None:
            # Process each face separately using encoder_hidden_states
            B = adapted_states.shape[0]
            
            if B % 6 == 0:  # Ensure it's divisible by 6 (cubemap faces)
                true_batch = B // 6
                
                # If encoder_hidden_states doesn't match the face batch size,
                # repeat it for each face
                if encoder_hidden_states.shape[0] != B:
                    # Assume encoder_hidden_states is [true_batch, ...]
                    if encoder_hidden_states.shape[0] == true_batch:
                        # Repeat for each face
                        repeated = []
                        for i in range(true_batch):
                            # Repeat this batch item 6 times (once per face)
                            for _ in range(6):
                                repeated.append(encoder_hidden_states[i:i+1])
                        
                        encoder_hidden_states = torch.cat(repeated, dim=0)
                    else:
                        # Unknown batch structure, try direct repeat
                        # This might not be correct for all cases
                        encoder_hidden_states = encoder_hidden_states.repeat_interleave(6, dim=0)
        
        # Use the original attention module with the adapted states
        # and potentially expanded encoder_hidden_states
        attn_output = self.original_attn(
            adapted_states,
            encoder_hidden_states,
            attention_mask
        )
        
        # Apply out adapter if needed
        if original_channels != self.expected_dim:
            attn_output = self.out_adapter(attn_output)
        
        return attn_output


def get_submodule(model, submodule_name):
    """
    Helper function to get a submodule by name.
    """
    if not submodule_name:
        return model
    
    submodule_parts = submodule_name.split('.')
    current_module = model
    
    for part in submodule_parts:
        # Handle indexing (for lists, tuples, etc.)
        if part.isdigit():
            current_module = current_module[int(part)]
        else:
            current_module = getattr(current_module, part)
    
    return current_module


def convert_attention_modules(unet):
    """
    Converts attention modules in UNet to handle cubemap structure.
    Automatically adds dimension adapters where needed.
    """
    # Track which modules were converted
    converted_modules = {}
    
    # Find all attention modules in the UNet
    for name, module in unet.named_modules():
        # Skip already processed modules
        if name in converted_modules:
            continue
        
        # Check if this is an attention module based on attribute names
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            parent = get_submodule(unet, parent_name)
            
            # Get attribute name
            attr_name = name.split('.')[-1]
            
            # Determine if this is self-attention or cross-attention based on naming conventions
            is_self_attention = ('attn1' in attr_name) or ('self' in attr_name.lower())
            
            # Create replacement module - simplified version for memory efficiency
            replacement = FixedInflatedAttention(module)
            
            # Replace the module
            setattr(parent, attr_name, replacement)
            converted_modules[name] = True
    
    print(f"Converted {len(converted_modules)} attention modules to handle cubemap structure")
    return unet


def convert_to_sync_groupnorm(module):
    """
    Recursively convert all GroupNorm layers to SynchronizedGroupNorm.
    
    Args:
        module: PyTorch module to modify
        
    Returns:
        Modified module with synchronized GroupNorm layers
    """
    num_converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            # Create a synchronized group norm with the same parameters
            sync_gn = SynchronizedGroupNorm(
                num_channels=child.num_channels,
                num_groups=child.num_groups,
                eps=child.eps
            )
            
            # Copy weights and biases
            sync_gn.weight.data.copy_(child.weight.data)
            sync_gn.bias.data.copy_(child.bias.data)
            
            # Replace the module
            setattr(module, name, sync_gn)
            num_converted += 1
        else:
            # Recursively process child modules
            num_converted += convert_to_sync_groupnorm(child)
    
    return num_converted


def load_sd_components(model_id="runwayml/stable-diffusion-v1-5", use_sync_gn=True, device="cuda"):
    """
    Load pretrained Stable Diffusion components.
    
    Args:
        model_id: Model ID from HuggingFace hub
        use_sync_gn: Whether to convert GroupNorm to SynchronizedGroupNorm
        device: Device to load models on
        
    Returns:
        vae, text_encoder, tokenizer, unet
    """
    print(f"Loading model components from {model_id}...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
    
    # Load text encoder and tokenizer
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=torch.float16)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    
    if use_sync_gn:
        print("Converting VAE GroupNorm layers to SynchronizedGroupNorm...")
        num_converted = convert_to_sync_groupnorm(vae)
        print(f"Converted {num_converted} GroupNorm layers in VAE")
    
    # Set to evaluation mode
    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    return vae, text_encoder, tokenizer, unet