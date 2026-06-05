import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model ต้องหาร num_heads ลงตัว"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # ขนาดต่อ head

        # Single projection สำหรับ Q, K, V พร้อมกัน (efficient กว่า 3 linear แยก)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,                                    # (B, T, d_model)
        key_padding_mask: Optional[Tensor] = None,    # (B, T) True = padding token
        attn_mask: Optional[Tensor] = None,           # (T, T) สำหรับ causal masking
    ) -> Tuple[Tensor, Tensor]:

        B, T, _ = x.shape

        # ── Step 1: Project → Q, K, V ──────────────────────────────────────
        # (B, T, d_model*3) แล้ว chunk เป็น 3 ส่วน
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)               # แต่ละตัว (B, T, d_model)

        # reshape เป็น multi-head: (B, num_heads, T, d_k)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # ── Step 2: Scaled dot-product attention ───────────────────────────
        # scores shape: (B, num_heads, T, T)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # optional: causal mask (decoder-style, ใช้ถ้าต้องการ)
        if attn_mask is not None:
            scores = scores + attn_mask  # attn_mask เป็น -inf ที่ตำแหน่งที่ mask

        # optional: padding mask — ปิด padding tokens ไม่ให้ถูก attend
        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T) เพื่อ broadcast ข้าม heads และ query positions
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float('-inf'))

        # ── Step 3: Softmax + dropout ───────────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)

        # ป้องกัน NaN ถ้า row ทั้งหมด -inf (เช่น padding token เป็น query)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # ── Step 4: Weighted sum of V ────────────────────────────────────────
        out = torch.matmul(attn_weights, V)         # (B, num_heads, T, d_k)

        # merge heads กลับ: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # final projection
        out = self.out_proj(out)

        # return attn_weights averaged across heads สำหรับ visualization
        return out, attn_weights.mean(dim=1)        # (B, T, d_model), (B, T, T)

if __name__ == "__main__":
    # quick sanity check
    mha = MultiHeadSelfAttention(d_model=256, num_heads=8)
    mha.eval()
    x = torch.randn(2, 16, 256)   # batch=2, seq_len=16
    out, weights = mha(x)

    assert out.shape == (2, 16, 256), f"wrong output shape: {out.shape}"
    assert weights.shape == (2, 16, 16), f"wrong weights shape: {weights.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    assert abs(weights[0, 0].sum().item() - 1.0) < 1e-5, "weights ไม่ sum to 1!"
    print("attention OK")