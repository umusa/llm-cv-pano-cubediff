# File: cl/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention  # type: ignore

class InflatedAttention(nn.Module):
    """
    Inflated multi-head attention for cubemap faces.
    Takes input [B, N, C] where N = 6 * num_patches_per_face,
    splits N→(6 faces × L patches), attends across all faces,
    and returns [B, N, C].
    """
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        # Q/K/V projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,              # [B×6, L, C]
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.BoolTensor    = None,
        **kwargs
    ) -> torch.FloatTensor:
        # 1) Recover original batch B and 6 faces
        total, L, C = hidden_states.shape
        FACES = 6
        if total % FACES != 0:
            raise ValueError(f"Batch size {total} not divisible by {FACES} faces")
        B = total // FACES

        # 2) Reshape into (B,6,L,C)
        x = hidden_states.view(B, FACES, L, C)
        if encoder_hidden_states is not None:
            ctx = encoder_hidden_states.view(B, FACES, L, C)
        else:
            ctx = x

        # 3) Merge faces back into tokens: [B, 6×L, C]
        x   = x.reshape(B, FACES * L, C)
        ctx = ctx.reshape(B, FACES * L, C)

        # 4) Standard multi-head attention on [B, N=6L, C]
        # replace the explicit q @ k^T → softmax → (attn @ v) with PyTorch’s fused, block-wise attention. 
        # This never materializes the full [B, H, N, N] tensor on GPU.
        # 4) Memory-efficient scaled dot-product attention
        q = self.to_q(x).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)
        k = self.to_k(ctx).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)
        v = self.to_v(ctx).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)

        # PyTorch 2.0+ fused attention — blocks internally, avoids OOM
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.to_out[1].p if len(self.to_out)>1 else 0.0,
            is_causal=False,
        )  # → [B, H, N, D]
        out = out.permute(0,2,1,3).reshape(B, FACES * L, C)

        # 5) Project back to C and split into batch of 6 faces again
        out = self.to_out(out)                         # [B,6L,C]
        return out.reshape(B * FACES, L, C)            # [B×6,L,C]


def inflate_attention_layer(
    original_attn: Attention,
    skip_copy: bool = False
) -> InflatedAttention:
    """
    Replace a HuggingFace SD Attention with our InflatedAttention.
    When skip_copy=False, dequantize() and copy the 320×320 on-face weights
    into the InflatedAttention’s to_q/to_k/to_v/to_out[0] for the diagonal blocks.
    When skip_copy=True, leave them at random init (fast 4-bit path).
    """
    # Build the inflated module
    query_dim = original_attn.to_q.in_features
    dropout   = getattr(original_attn.to_out[1], "p", 0.0)
    heads     = original_attn.heads
    dim_head  = query_dim // heads

    inflated = InflatedAttention(
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
    )

    if not skip_copy:
        # helper to extract the full [320×320] weight for both FP and 4-bit layers
        def get_weight(module: nn.Module):
            if hasattr(module, "dequantize"):
                return module.dequantize()
            if hasattr(module, "weight"):
                return module.weight.data
            if hasattr(module, "qweight"):
                return module.qweight.data
            raise RuntimeError(f"No weight found in {module}")

        # Extract and copy
        wq = get_weight(original_attn.to_q)
        wk = get_weight(original_attn.to_k)
        wv = get_weight(original_attn.to_v)
        wo = get_weight(original_attn.to_out[0])

        # Transpose if shapes mismatch
        if inflated.to_q.weight.shape != wq.shape:
            wq = wq.t().view_as(inflated.to_q.weight)

        inflated.to_q.weight.data.copy_(wq)
        inflated.to_k.weight.data.copy_(wk)
        inflated.to_v.weight.data.copy_(wv)
        inflated.to_out[0].weight.data.copy_(wo)

        # Copy bias if present
        b = getattr(original_attn.to_out[0], "bias", None)
        if b is not None:
            inflated.to_out[0].bias.data.copy_(b.data)

    return inflated
