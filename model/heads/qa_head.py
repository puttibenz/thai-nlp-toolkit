import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Any

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
        context_start: Optional[Any] = None,   # รับ context_start จาก encode_qa()
    ) -> Tuple[Tensor, Tensor]:
        logits = self.qa_outputs(hidden_states)            # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).clone()            # (B, T)
        end_logits   = end_logits.squeeze(-1).clone()              # (B, T)

        # mask question positions ออก — คำตอบต้องอยู่ใน context เท่านั้น
        if context_start is not None:
            mask_val = torch.finfo(start_logits.dtype).min   # -inf
            if isinstance(context_start, torch.Tensor):
                B, T = start_logits.shape
                # context_start shape: (B,)
                # ย้าย context_start ไปยัง device เดียวกับ start_logits
                ctx_start = context_start.to(start_logits.device)
                rng = torch.arange(T, device=start_logits.device).unsqueeze(0) # (1, T)
                mask = rng < ctx_start.unsqueeze(1) # (B, T)
                start_logits = start_logits.masked_fill(mask, mask_val)
                end_logits = end_logits.masked_fill(mask, mask_val)
            else:
                start_logits[:, :context_start] = mask_val
                end_logits  [:, :context_start] = mask_val

        return start_logits, end_logits
