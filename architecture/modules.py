"""
modules.py - Customized model components for CubeDiff

This module contains specialized neural network components designed for 
processing cubemap representations in the CubeDiff architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union


class CubemapPositionalEncoding(nn.Module):
    """
    Positional encoding for cubemap faces that encodes the 3D geometric relationships.
    
    This module adds positional information to help the model understand the 
    spatial relationships between different faces of the cubemap.
    """
    
    def __init__(self, embedding_dim, max_seq_len=128):
        """
        Initialize the positional encoding.
        
        Args:
            embedding_dim: Dimension of the embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Initialize face-specific embeddings
        self.face_embeddings = nn.Embedding(6, embedding_dim)
        
        # Create a standard sinusoidal position encoding
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            -(math.log(10000.0) / embedding_dim)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter to be optimized)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Initialize face embeddings with geometric understanding
        self._init_face_embeddings()
    
    def _init_face_embeddings(self):
        """
        Initialize face embeddings with geometric understanding of the cubemap structure.
        
        Each face is initialized with information about its position in 3D space,
        which helps the model understand the spatial relationships.
        """
        # Define 3D directions for each face
        directions = [
            [0, 0, 1],   # Front (+Z)
            [0, 0, -1],  # Back (-Z)
            [-1, 0, 0],  # Left (-X)
            [1, 0, 0],   # Right (+X)
            [0, 1, 0],   # Top (+Y)
            [0, -1, 0],  # Bottom (-Y)
        ]
        
        with torch.no_grad():
            # Initialize each face embedding
            for i, direction in enumerate(directions):
                # Convert to tensor
                dir_tensor = torch.tensor(direction, dtype=torch.float32)
                
                # Expand to full embedding dimension
                embedding_dim = self.face_embeddings.embedding_dim
                expanded = dir_tensor.repeat(embedding_dim // 3 + 1)[:embedding_dim]
                
                # Set the embedding
                self.face_embeddings.weight[i] = expanded
    
    def forward(self, x):
        """
        Forward pass to add positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, num_faces, channels, height, width)
            
        Returns:
            Tensor with positional encoding added
        """
        # Extract dimensions
        batch_size, num_faces, channels, height, width = x.shape
        
        # Create a position encoding for each spatial location
        y_pos = torch.linspace(-1, 1, height, device=x.device)
        x_pos = torch.linspace(-1, 1, width, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Stack grid coordinates
        pos_grid = torch.stack([x_grid, y_grid], dim=-1).to(x.device)  # (H, W, 2)
        
        # Create sinusoidal encoding for positions
        pos_encoding = torch.zeros(height, width, channels, device=x.device)
        
        for i in range(0, channels, 2):
            if i < channels - 1:  # Ensure we don't go out of bounds
                freq = 1.0 / (10000 ** (i / channels))
                pos_encoding[..., i] = torch.sin(pos_grid[..., 0] * freq)
                pos_encoding[..., i+1] = torch.cos(pos_grid[..., 1] * freq)
        
        # Permute to (C, H, W) for broadcasting
        pos_encoding = pos_encoding.permute(2, 0, 1)
        
        # Get face-specific embeddings
        face_indices = torch.arange(num_faces, device=x.device)
        face_emb = self.face_embeddings(face_indices)  # (num_faces, channels)
        
        # Reshape for broadcasting
        face_emb = face_emb.view(1, num_faces, channels, 1, 1)
        pos_encoding = pos_encoding.view(1, 1, channels, height, width)
        
        # Apply positional encoding with a small weight to avoid dominating the signal
        return x + 0.1 * face_emb + 0.1 * pos_encoding


class GroupNormalizationSync(nn.Module):
    """
    Synchronized Group Normalization for cubemap faces.
    """
    
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        # Adjust num_groups to be compatible with num_channels
        if num_channels % num_groups != 0:
            # Find the largest divisor of num_channels that's <= num_groups
            for i in range(min(num_groups, num_channels), 0, -1):
                if num_channels % i == 0:
                    num_groups = i
                    print(f"Adjusted num_groups to {num_groups} to be compatible with {num_channels} channels")
                    break
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # Learnable parameters for scaling and shifting
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        """
        Forward pass for synchronized normalization.
        """
        # Extract dimensions
        batch_size, num_faces, channels, height, width = x.shape
        
        # Ensure the channel dimension matches what we expect
        if channels != self.num_channels:
            raise ValueError(f"Input has {channels} channels, but expected {self.num_channels}")
        
        # Compute channels per group
        channels_per_group = self.num_channels // self.num_groups
        
        # Reshape to (batch_size * num_faces, channels, height, width) for group norm
        x_reshaped = x.view(batch_size * num_faces, channels, height, width)
        
        # Apply group normalization
        x_normalized = torch.nn.functional.group_norm(
            x_reshaped, 
            num_groups=self.num_groups,
            weight=None,
            bias=None,
            eps=self.eps
        )
        
        # Reshape back to original dimensions
        x_normalized = x_normalized.view(batch_size, num_faces, channels, height, width)
        
        # Apply learnable parameters
        weight = self.weight.view(1, 1, -1, 1, 1)
        bias = self.bias.view(1, 1, -1, 1, 1)
        
        return x_normalized * weight + bias


def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
    """
    Forward pass of the inflated attention.
    
    This matches the signature of the diffusers attention modules.
    """
    # For cross-attention with encoder_hidden_states, it's safer 
    # to fall back to the original implementation
    if encoder_hidden_states is not None:
        return self.base_module(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
    
    # Try to reshape for cubemap processing (only for self-attention)
    reshaped_hidden_states, is_cubemap = self.reshape_for_cubemap(hidden_states)
    
    # If not in cubemap format, fall back to the base attention module
    if not is_cubemap:
        return self.base_module(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
    
    # Process in cubemap format for self-attention
    batch_size, num_faces = reshaped_hidden_states.shape[:2]
    
    # Process each face
    outputs = []
    
    for face_idx in range(num_faces):
        # Get current face features
        face_hidden = reshaped_hidden_states[:, face_idx]  # (B, ...)
        
        # Compute query for this face
        query = self.to_q(face_hidden)
        
        # Compute key and value for all faces with learned relationships
        keys = []
        values = []
        
        for other_face_idx in range(num_faces):
            other_face = reshaped_hidden_states[:, other_face_idx]
            
            # Weight the contribution using learned relationships
            relation_weight = torch.sigmoid(self.face_interactions[face_idx, other_face_idx])
            
            key = self.to_k(other_face)
            value = self.to_v(other_face)
            
            # Store with relationship weighting
            keys.append(key * relation_weight)
            values.append(value * relation_weight)
        
        # Combine all contributions
        key = torch.stack(keys, dim=1).mean(dim=1)  # Average across faces
        value = torch.stack(values, dim=1).mean(dim=1)  # Average across faces
        
        # Attention calculation
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        face_output = torch.matmul(attention_probs, value)
        
        # Apply output projection
        face_output = self.to_out[0](face_output)
        face_output = self.to_out[1](face_output)
        
        outputs.append(face_output)
    
    # Stack outputs and reshape back to original format
    stacked_output = torch.stack(outputs, dim=1)  # (B, 6, ...)
    
    # Reshape back to match the input format expected by the next layer
    final_output = stacked_output.view(*hidden_states.shape)
    
    return final_output


class OverlappingEdgeProcessor(nn.Module):
    """
    Process overlapping edges between cubemap faces to ensure seamless transitions.
    
    This module helps create smooth transitions at the boundaries between faces,
    which is crucial for a realistic and coherent 360° panorama.
    """
    
    def __init__(self, overlap_size=4):
        """
        Initialize the edge processor.
        
        Args:
            overlap_size: Size of the overlapping region in pixels
        """
        super().__init__()
        self.overlap_size = overlap_size
        
        # Define face adjacency (which faces are connected)
        # Format: (face_idx, edge, adjacent_face_idx, adjacent_edge)
        # Edges: 0=left, 1=right, 2=top, 3=bottom
        self.adjacency = [
            # Front face connections
            (0, 0, 2, 1),  # Front-Left
            (0, 1, 3, 0),  # Front-Right
            (0, 2, 4, 2),  # Front-Top
            (0, 3, 5, 2),  # Front-Bottom
            
            # Back face connections
            (1, 0, 3, 1),  # Back-Left
            (1, 1, 2, 0),  # Back-Right
            (1, 2, 4, 0),  # Back-Top
            (1, 3, 5, 0),  # Back-Bottom
            
            # Left face connections
            (2, 0, 1, 1),  # Left-Back
            (2, 1, 0, 0),  # Left-Front
            (2, 2, 4, 3),  # Left-Top
            (2, 3, 5, 3),  # Left-Bottom
            
            # Right face connections
            (3, 0, 0, 1),  # Right-Front
            (3, 1, 1, 0),  # Right-Back
            (3, 2, 4, 1),  # Right-Top
            (3, 3, 5, 1),  # Right-Bottom
            
            # Top face connections
            (4, 0, 1, 2),  # Top-Back
            (4, 1, 3, 2),  # Top-Right
            (4, 2, 0, 2),  # Top-Front
            (4, 3, 2, 2),  # Top-Left
            
            # Bottom face connections
            (5, 0, 1, 3),  # Bottom-Back
            (5, 1, 3, 3),  # Bottom-Right
            (5, 2, 0, 3),  # Bottom-Front
            (5, 3, 2, 3),  # Bottom-Left
        ]
    
    def forward(self, x):
        """
        Forward pass to process overlapping edges.
        
        Args:
            x: Input tensor of shape (batch_size, num_faces, channels, height, width)
            
        Returns:
            Processed tensor with smoothed edges
        """
        batch_size, num_faces, channels, height, width = x.shape
        
        # Create a copy of the input to modify
        output = x.clone()
        
        # Process each adjacency relationship
        for face_idx, edge, adj_face_idx, adj_edge in self.adjacency:
            # Extract the faces
            face = x[:, face_idx]  # (batch_size, channels, height, width)
            adj_face = x[:, adj_face_idx]  # (batch_size, channels, height, width)
            
            # Define edge regions based on the edge index
            if edge == 0:  # Left edge
                edge_region = face[:, :, :, :self.overlap_size]
                weights = torch.linspace(0, 1, self.overlap_size, device=x.device)
                # Reshape properly to match dimensions
                weights = weights.view(1, 1, 1, -1).expand(batch_size, channels, height, self.overlap_size)
            elif edge == 1:  # Right edge
                edge_region = face[:, :, :, -self.overlap_size:]
                weights = torch.linspace(1, 0, self.overlap_size, device=x.device)
                # Reshape properly to match dimensions
                weights = weights.view(1, 1, 1, -1).expand(batch_size, channels, height, self.overlap_size)
            elif edge == 2:  # Top edge
                edge_region = face[:, :, :self.overlap_size, :]
                weights = torch.linspace(0, 1, self.overlap_size, device=x.device)
                # Reshape properly to match dimensions
                weights = weights.view(1, 1, -1, 1).expand(batch_size, channels, self.overlap_size, width)
            elif edge == 3:  # Bottom edge
                edge_region = face[:, :, -self.overlap_size:, :]
                weights = torch.linspace(1, 0, self.overlap_size, device=x.device)
                # Reshape properly to match dimensions
                weights = weights.view(1, 1, -1, 1).expand(batch_size, channels, self.overlap_size, width)
            
            # Define adjacent edge regions based on the adjacent edge index
            if adj_edge == 0:  # Left edge
                adj_edge_region = adj_face[:, :, :, :self.overlap_size]
                if edge == 2 or edge == 3:  # Need to rotate/flip
                    adj_edge_region = torch.flip(adj_edge_region, dims=[-1])
            elif adj_edge == 1:  # Right edge
                adj_edge_region = adj_face[:, :, :, -self.overlap_size:]
                if edge == 2 or edge == 3:  # Need to rotate/flip
                    adj_edge_region = torch.flip(adj_edge_region, dims=[-1])
            elif adj_edge == 2:  # Top edge
                adj_edge_region = adj_face[:, :, :self.overlap_size, :]
                if edge == 0 or edge == 1:  # Need to rotate/flip
                    adj_edge_region = torch.flip(adj_edge_region, dims=[-2])
            elif adj_edge == 3:  # Bottom edge
                adj_edge_region = adj_face[:, :, -self.overlap_size:, :]
                if edge == 0 or edge == 1:  # Need to rotate/flip
                    adj_edge_region = torch.flip(adj_edge_region, dims=[-2])
            
            # Check dimensions match before blending
            if edge_region.shape != adj_edge_region.shape:
                # This is a safety check - dimensions must match for blending
                print(f"Dimension mismatch: edge_region {edge_region.shape}, adj_edge_region {adj_edge_region.shape}")
                continue
                
            # Ensure weights dimensions match edge_region for proper broadcasting
            if weights.shape != edge_region.shape:
                # Reshape weights to match exactly
                if edge == 0 or edge == 1:  # Left or right edge
                    weights = weights.view(1, 1, 1, self.overlap_size).expand_as(edge_region)
                else:  # Top or bottom edge
                    weights = weights.view(1, 1, self.overlap_size, 1).expand_as(edge_region)
            
            # Blend the edges using the weights
            blended = edge_region * weights + adj_edge_region * (1 - weights)
            
            # Update the output tensor
            if edge == 0:  # Left edge
                output[:, face_idx, :, :, :self.overlap_size] = blended
            elif edge == 1:  # Right edge
                output[:, face_idx, :, :, -self.overlap_size:] = blended
            elif edge == 2:  # Top edge
                output[:, face_idx, :, :self.overlap_size, :] = blended
            elif edge == 3:  # Bottom edge
                output[:, face_idx, :, -self.overlap_size:, :] = blended
        
        return output


def adapt_unet_for_cubemap(unet_model):
    """
    Adapt a UNet model for processing cubemap inputs by ONLY modifying
    self-attention layers (attn1) and leaving cross-attention (attn2) untouched.
    """
    # Track which layers are modified
    modified_layers = []
    
    # Replace ONLY self-attention layers with inflated versions
    for name, module in unet_model.named_modules():
        # Only target self-attention (attn1) modules, NOT cross-attention (attn2)
        # if "attn1" in name and hasattr(module, "to_q"):
        #     print(f"Adapting self-attention layer: {name}")
        # Condition to find attention blocks (adjust if needed based on diffusers version)
        is_attention = "attn1" in name or "attn2" in name
        is_transformer_block = "transformer_blocks" in name
        
        if "attn1" in name and not "attn2" in name and hasattr(module, "to_q"): # Target only attn1
            print(f"Adapting self-attention layer: {name}")
            
            # Get the parent module
            parent_name = name.rsplit(".", 1)[0]
            attr_name = name.split(".")[-1]
            
            # Navigate to the parent module
            parent = unet_model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            
            # Create inflated attention with the original module
            inflated_attn = InflatedAttention(module)
            
            # Replace the original attention module
            setattr(parent, attr_name, inflated_attn)
            
            # Keep track of modified layers
            modified_layers.append(name)
    
    # print(f"Modified {len(modified_layers)} self-attention layers to support cubemap inputs")
    print(f"Modified {len(modified_layers)} attention layers to support cubemap inputs") # Adjust print message back
    return unet_model
    


class InflatedAttention(nn.Module):
    """
    Simplified inflated attention module that only handles self-attention.
    Cross-attention is left to the original implementation.
    """
    
    def __init__(self, original_attn_module):
        super().__init__()
        
        # Keep a reference to the original module for fallback
        self.original_module = original_attn_module
        
        # Copy parameters
        self.to_q = original_attn_module.to_q
        self.to_k = original_attn_module.to_k
        self.to_v = original_attn_module.to_v
        self.to_out = original_attn_module.to_out
        
        # Copy other attributes
        if hasattr(original_attn_module, 'scale'):
            self.scale = original_attn_module.scale
        else:
            self.scale = 1.0
            
        if hasattr(original_attn_module, 'heads'):
            self.heads = original_attn_module.heads
        else:
            self.heads = 8  # Default value
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        Forward pass with safety checks.
        """
        # If this is cross-attention, use the original module
        if encoder_hidden_states is not None:
            # Attempting the previous fix here didn't work, let's handle it before UNet call
            #  return self.original_module(hidden_states, encoder_hidden_states, attention_mask, **kwargs)

            # Check if hidden_states batch dim is num_faces times encoder_hidden_states batch dim / 2 (for CFG)
            # B*6 vs B*2 -> factor is 3
            batch_size_hidden = hidden_states.shape[0]
            batch_size_encoder = encoder_hidden_states.shape[0]

            # Check if this could be cubemap input (divisible by 6)
            is_cubemap = hidden_states.shape[0] % 6 == 0
        
            num_faces = 6 # Assuming cubemap
            cfg_multiplier = 2 # Assuming CFG

            # Check if hidden_states batch size is consistent with cubemap faces + CFG applied to encoder
            if batch_size_hidden == (batch_size_encoder // cfg_multiplier) * num_faces:
                 # Repeat encoder_hidden_states to match hidden_states batch dimension
                 # Each text embedding (uncond, cond) needs to be repeated num_faces / cfg_multiplier times
                 repeat_factor = num_faces // cfg_multiplier # 6 // 2 = 3
                 encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeat_factor, dim=0)

            # Call the original module with potentially repeated encoder_hidden_states
            return self.original_module(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        
        # Check if this could be cubemap input (divisible by 6)
        is_cubemap = hidden_states.shape[0] % 6 == 0
        
        if not is_cubemap:
            # Not cubemap input, use original module
            return self.original_module(hidden_states, None, attention_mask, **kwargs)
        
        # This is cubemap self-attention
        batch_size = hidden_states.shape[0] // 6
        num_faces = 6
        
        # Simple implementation: just use the original module's computation
        # but reshape input/output appropriately
        # Fallback to original module for self-attention too for now to isolate issue
        result = self.original_module(hidden_states, None, attention_mask, **kwargs)
        
        # No additional cubemap-specific processing for now, just to get it working
        
        return result






