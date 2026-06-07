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
        # Mean pooling — เฉลี่ยเฉพาะ real tokens ไม่รวม padding
        # attention_mask: (B, T) — 1=real, 0=padding
        mask = attention_mask.unsqueeze(-1).float()     # (B, T, 1)
        sum_hidden = (hidden_states * mask).sum(dim=1)  # (B, d_model)
        lengths = mask.sum(dim=1).clamp(min=1)          # (B, 1) ป้องกัน หารด้วยศูนย์
        pooled = sum_hidden / lengths                   # (B, d_model)                   # (B, d_model)
        return self.classifier(pooled)                  # (B, num_classes)
