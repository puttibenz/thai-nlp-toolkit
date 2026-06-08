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
        # [CLS] อยู่ที่ position 0 เสมอ — ใช้ตรงๆ ได้เลย
        # ดีกว่า mean pooling เพราะ [CLS] ถูก design มาเพื่อ aggregate ทั้ง sequence
        cls_output = hidden_states[:, 0, :]     # (B, d_model)
        return self.classifier(cls_output)