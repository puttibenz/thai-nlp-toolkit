import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or GeLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # LayerNorm -> Attention -> residual -> LayerNorm -> FFN -> residual
        # TODO: Implement forward pass with residuals
        return x
