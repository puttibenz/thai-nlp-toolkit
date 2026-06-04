import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class QAHead(nn.Module):
    """
    Extractive QA head to predict start and end token positions.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Outputs start and end logits for each token
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        # Input shape: (B, T, d_model)
        # Returns shape: (B, T) start_logits, (B, T) end_logits
        logits = self.qa_outputs(hidden_states) # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
