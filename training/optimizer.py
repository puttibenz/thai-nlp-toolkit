import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, List, Optional


def get_optimizer(
    model:          Any,
    learning_rate:  float,
    weight_decay:   float = 0.01,
    betas:          tuple = (0.9, 0.999),   # AdamW defaults
    eps:            float = 1e-8,
) -> AdamW:
    """
    AdamW พร้อม weight decay exclusion สำหรับ bias และ LayerNorm

    ทำไมต้อง exclude:
    - bias terms  — เป็น scalar offset ไม่ใช่ weight matrix
                    การ decay bias ทำให้ model bias ต่อ zero โดยไม่จำเป็น
    - LayerNorm   — weight (gamma) และ bias (beta) ใช้ normalize distribution
                    ถ้า decay จะรบกวน normalization ทำให้ training ไม่เสถียร
    """
    # แยก parameters เป็น 2 กลุ่ม
    decay_params     = []
    no_decay_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue   # skip frozen params

        # ชื่อที่ลงท้ายด้วย "bias" หรือ มี "norm" อยู่ในชื่อ → no decay
        if name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    # log สรุปให้รู้ว่า group ไหนมีกี่ params
    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    total      = n_decay + n_no_decay
    print(
        f"optimizer: {total:,} params total | "
        f"{n_decay:,} with decay | "
        f"{n_no_decay:,} no decay"
    )

    return optimizer


def get_scheduler(
    optimizer:    Any,
    warmup_steps: int,
    max_steps:    int,
    schedule:     str = "cosine",   # "linear" หรือ "cosine"
    min_lr_ratio: float = 0.1,      # lr ต่ำสุด = min_lr_ratio * peak_lr
) -> LambdaLR:
    """
    Learning rate scheduler พร้อม linear warmup

    2 ตัวเลือก:
    ┌─────────┬──────────────────────────────────────────────────┐
    │ linear  │ warmup → decay เส้นตรงไปถึง min_lr                 │
    │ cosine  │ warmup → decay แบบ cosine curve (แนะนำ)          │
    └─────────┴──────────────────────────────────────────────────┘

    Cosine ดีกว่า linear เพราะ decay ช้าตอนแรกแล้วค่อยเร็วขึ้น
    ทำให้ model มีเวลา fine-tune ก่อน lr จะเล็กมาก
    """
    assert warmup_steps >= 0,             "warmup_steps ต้องเป็น non-negative"
    assert max_steps > warmup_steps,      "max_steps ต้องมากกว่า warmup_steps"
    assert 0.0 <= min_lr_ratio <= 1.0,    "min_lr_ratio ต้องอยู่ใน [0, 1]"

    def lr_lambda(current_step: int) -> float:
        # ── Phase 1: Warmup ──────────────────────────────────────
        # เพิ่ม lr จาก 0 → 1.0 เป็น linear ใน warmup_steps แรก
        # warmup ป้องกัน gradient explosion ตอนเริ่ม training
        # เพราะ weight ยัง random อยู่ และ gradient จะใหญ่มาก
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # ── Phase 2: Decay ───────────────────────────────────────
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(1.0, progress)   # clamp ไม่ให้เกิน 1.0

        if schedule == "linear":
            # decay เส้นตรง: 1.0 → min_lr_ratio
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        elif schedule == "cosine":
            # cosine decay: 1.0 → min_lr_ratio
            # cos(0) = 1.0, cos(π) = -1.0 → normalize ให้อยู่ใน [min_lr_ratio, 1.0]
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        else:
            raise ValueError(f"schedule ไม่รู้จัก: '{schedule}' — ใช้ 'linear' หรือ 'cosine'")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)