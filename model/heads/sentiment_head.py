import torch
import torch.nn as nn
from torch import Tensor

class SentimentHead(nn.Module):
    """
    Sentence-level classification head using CLS token or mean pooling.
    Classes: positive, neutral, negative
    """
    def __init__(self, d_model: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Input shape: (B, T, d_model)
        # Returns shape: (B, num_classes)
        # TODO: Implement pooling (e.g., mean pooling or [CLS] token)
        B, _, d_model = hidden_states.shape
        pooled = torch.zeros(B, d_model, device=hidden_states.device)
        return self.classifier(pooled)
