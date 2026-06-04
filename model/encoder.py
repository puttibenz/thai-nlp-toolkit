import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1

class ThaiTransformerEncoder(nn.Module):
    """
    Shared encoder backbone stacking multiple Transformer blocks.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO: Implement token and position embedding layers, and transformer layers
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Input shape: (B, T)
        # Returns hidden states of shape: (B, T, d_model)
        B, T = input_ids.shape
        return torch.zeros(B, T, self.config.d_model, device=input_ids.device)
