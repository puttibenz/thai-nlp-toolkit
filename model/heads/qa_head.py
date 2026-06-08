import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class QAHead(nn.Module):
    """
    Extractive QA head to predict start and end token positions.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Outputs start and end logits for each token
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(
        self,
        hidden_states: Tensor,
        context_start: Optional[int] = None,   # รับ context_start จาก encode_qa()
    ) -> Tuple[Tensor, Tensor]:
        logits = self.qa_outputs(hidden_states)            # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).clone()            # (B, T)
        end_logits   = end_logits.squeeze(-1).clone()              # (B, T)

        # mask question positions ออก — คำตอบต้องอยู่ใน context เท่านั้น
        if context_start is not None:
            mask_val = torch.finfo(start_logits.dtype).min   # -inf
            start_logits[:, :context_start] = mask_val
            end_logits  [:, :context_start] = mask_val

        return start_logits, end_logits
