import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention

class InflatedAttention(nn.Module):
    """
    Inflated attention mechanism for cubemap faces.
    This extends the standard attention to work across cube faces.
    """
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None, mask=None):
        """
        Args:
            x: Input tensor of shape [batch, num_faces, seq_len, channels]
            context: Context tensor (optional)
            mask: Attention mask (optional)
            
        Returns:
            Attention output of same shape as input
        """
        # If no context is provided, use x
        context = context if context is not None else x
        
        # Get batch size, number of faces, and sequence length
        batch_size, num_faces, seq_len, _ = x.shape
        
        # Reshape to process all faces together
        x_reshaped = x.reshape(batch_size * num_faces, seq_len, -1)
        
        # If context is provided with face dimension, reshape it similarly
        if context.ndim == 4:  # [batch, num_faces, seq_len, channels]
            context_reshaped = context.reshape(batch_size * num_faces, context.shape[2], -1)
        else:  # Context without face dimension
            context_reshaped = context
        
        # Standard attention computation first
        q = self.to_q(x_reshaped)
        k = self.to_k(context_reshaped)
        v = self.to_v(context_reshaped)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size * num_faces, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(batch_size * num_faces, k.shape[1], self.heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(batch_size * num_faces, v.shape[1], self.heads, -1).permute(0, 2, 1, 3)
        
        # Reshape again to enable cross-face attention
        q = q.reshape(batch_size, num_faces, self.heads, seq_len, -1)
        k = k.reshape(batch_size, num_faces, self.heads, k.shape[2], -1)
        v = v.reshape(batch_size, num_faces, self.heads, v.shape[2], -1)
        
        # Stack across faces to allow cross-face attention
        q_stacked = q.reshape(batch_size, num_faces * self.heads, seq_len, -1)
        k_stacked = k.reshape(batch_size, num_faces * self.heads, k.shape[3], -1)
        v_stacked = v.reshape(batch_size, num_faces * self.heads, v.shape[3], -1)
        
        # Compute attention scores
        attn_weights = torch.matmul(q_stacked, k_stacked.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        # Softmax for attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v_stacked)
        
        # Reshape back to original format
        out = out.reshape(batch_size, num_faces, self.heads, seq_len, -1)
        out = out.permute(0, 1, 3, 2, 4).reshape(batch_size, num_faces, seq_len, -1)
        
        # Apply output projection
        out = self.to_out(out.reshape(batch_size * num_faces, seq_len, -1))
        out = out.reshape(batch_size, num_faces, seq_len, -1)
        
        return out

def inflate_attention_layer(original_attn):
    """
    Inflate a standard attention layer to work with cubemap faces.
    
    Args:
        original_attn: Original attention layer from Stable Diffusion
        
    Returns:
        Inflated attention layer
    """
    # Create inflated attention with same parameters
    inflated_attn = InflatedAttention(
        query_dim=original_attn.to_q.in_features,
        heads=original_attn.heads,
        dim_head=original_attn.to_q.out_features // original_attn.heads,
        dropout=original_attn.to_out[1].p if len(original_attn.to_out) > 1 else 0.0,
    )
    
    # Copy weights from original attention
    inflated_attn.to_q.weight.data.copy_(original_attn.to_q.weight.data)
    inflated_attn.to_k.weight.data.copy_(original_attn.to_k.weight.data)
    inflated_attn.to_v.weight.data.copy_(original_attn.to_v.weight.data)
    inflated_attn.to_out[0].weight.data.copy_(original_attn.to_out[0].weight.data)
    
    if hasattr(original_attn.to_out[0], 'bias') and original_attn.to_out[0].bias is not None:
        inflated_attn.to_out[0].bias.data.copy_(original_attn.to_out[0].bias.data)
    
    return inflated_attn