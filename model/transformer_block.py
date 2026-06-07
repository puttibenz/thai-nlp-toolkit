import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import sys
import pathlib

# Add project root to sys.path
root = pathlib.Path(__file__).resolve().parent
while root.parent != root:
    if (root / "requirements.txt").exists() or (root / "README.md").exists():
        sys.path.append(str(root))
        break
    root = root.parent

from model.attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block.
    
    Pre-norm (LN → sublayer → residual) มีเสถียรภาพ training
    ดีกว่า post-norm โดยเฉพาะตอน network ลึกหรือ lr สูง
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. เรียกใช้งานชิ้นส่วนประกอบย่อยที่เราทำไว้
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)

        # Pre-Norm ต้องการ LayerNorm 2 ชุด (ก่อนเข้า Attention และ ก่อนเข้า FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network (FFN) ขยายมิติเป็น d_ff แล้วกลับมาเป็น d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # โค้ดภาษาไทยแนะนำใช้ GELU จะช่วยให้ลื่นไหลและเสถียรกว่า ReLU ครับ
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Input shaper x: (B, T, d_model)
        # Input shape padding_mask: (B, T)

        # ── Step 1: Pre-Norm & Multi-Head Self-Attention ──────────────────
        # ทำตาม Pre-Norm: ส่ง ไปนวดด้วย norm1 ก่อนเข้า Attention
        residual = x
        normed_x = self.norm1(x)

        # โยนเข้า Attention พร้อมส่งต่อ padding_mask (ดึงมาเฉพาะตำแหน่งแรกที่เป็นตัวเอาต์พุตเวกเตอร์)
        attn_out, attn_weights = self.attn(normed_x, key_padding_mask=padding_mask)

        # Residual Connection ชุดที่ 1: เอาผลลัพธ์จาก Attention มาบวกกับ x เดิม
        x = residual + attn_out

        # ── Step 2: Pre-Norm & Feed-Forward Network ───────────────────────
        # ส่ง x ไปนวดด้วย norm2 ก่อนเข้า FFN
        residual = x
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)

        # Residual Connection ชุดที่ 2: เอาผลลัพธ์จาก FFN มาบวกกับ x เดิม
        x = residual + ffn_out
        return x, attn_weights

if __name__ == "__main__":
    # Quick sanity check
    block = TransformerBlock(d_model=256, num_heads=8, d_ff=1024)
    block.eval()  # ตั้งเป็นโหมดประเมินผลเพื่อปิด dropout

    # Sample input: batch size 2, sequence length 16, model dimension 256
    dummy_x = torch.randn(2, 16, 256)
    dummy_mask = torch.zeros(2, 16, dtype=torch.bool)  # False means no padding
    dummy_mask[0, 12:] = True  # ตัวอย่าง: ลองสมมติให้ประโยคแรกมี padding ท้ายประโยค

    out, attn_weights = block(dummy_x, padding_mask=dummy_mask)

    assert out.shape == (2, 16, 256), f"Wrong output shape: {out.shape}"
    assert attn_weights.shape == (2, 8, 16, 16), f"Wrong attention weights shape: {attn_weights.shape}"
    assert not torch.isnan(out).any(), "พบค่า NaN ในโมดูล"
    assert not torch.isnan(attn_weights).any(), "พบค่า NaN ใน attention weights"
    print("TransformerBlock test passed!")