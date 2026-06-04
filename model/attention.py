import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product attention with multiple heads.
    Supports causal masking and padding mask.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Single projection layer for Query, Key, Value
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,                           # (B, T, d_model)
        key_padding_mask: Optional[Tensor] = None,  # (B, T) bool
        attn_mask: Optional[Tensor] = None,         # (T, T) causal
    ) -> Tuple[Tensor, Tensor]:              # output, attn_weights
        # TODO: Implement scaled dot-product and split-head attention
        # For now, return dummy outputs matching shapes
        B, T, _ = x.shape
        attn_weights = torch.zeros(B, self.num_heads, T, T, device=x.device)
        return x, attn_weights
