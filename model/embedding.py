import torch
import torch.nn as nn
from torch import Tensor

class TokenEmbedding(nn.Module):
    """Token Embedding lookup layer."""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # TODO: Implement sinusoidal pos encoding weights matrix
        self.register_buffer('pe', torch.zeros(max_seq_len, d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, T, d_model)
        # Adds PE to input tensor and applies dropout
        return self.dropout(x)
