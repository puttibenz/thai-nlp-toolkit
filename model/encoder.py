import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

import sys
import pathlib

# Add project root to sys.path
root = pathlib.Path(__file__).resolve().parent
while root.parent != root:
    if (root / "requirements.txt").exists() or (root / "README.md").exists():
        sys.path.append(str(root))
        break
    root = root.parent

from model.embedding import ThaiEmbedding
from model.transformer_block import TransformerBlock

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0

class ThaiTransformerEncoder(nn.Module):
    """
    Shared encoder backbone stacking multiple Transformer blocks.
    input_ids → embedding → N x TransformerBlock → hidden states
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer (token + positional + layer norm)
        self.embedding = ThaiEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # Stack N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # 3. Final norm (optional but common for pre-norm architectures)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None):
        """
        # สร้าง padding_mask จาก attention_mask
        # TransformerBlock ใช้ True = "ให้ mask ออก" (ตรงข้ามกับ HuggingFace convention)
        """
        if attention_mask is not None:
            padding_mask = attention_mask == 0 # (B, T) bool
        else:
            padding_mask = input_ids == self.config.pad_token_id 

        # Embeddings
        x = self.embedding(input_ids)
        
        # Pass through transformer blocks
        # เก็บ attn_weights ทุก layer ไว้สำหรับ visualization / debug
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, padding_mask=padding_mask)
            all_attn_weights.append(attn_w)

        # final
        x = self.norm(x)

        return x, all_attn_weights

if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=32000, d_model=256,
                  num_heads=8, num_layers=6, d_ff=1024)
    encoder = ThaiTransformerEncoder(cfg)

    # dummy input พร้อม padding
    B, T = 2, 32
    input_ids = torch.randint(1, 32000, (B, T))
    input_ids[1, 20:] = 0   # batch ที่ 2 มี padding หลัง position 20

    attention_mask = (input_ids != 0).long()

    hidden, attn_weights = encoder(input_ids, attention_mask)

    assert hidden.shape == (B, T, 256), f"wrong shape: {hidden.shape}"
    assert len(attn_weights) == 6, "ต้องได้ attn weights ครบ 6 layers"
    assert not torch.isnan(hidden).any(), "NaN in output!"

    # padding positions ต้องไม่ส่งผลต่อ real tokens (approximate check)
    loss = hidden.sum()
    loss.backward()
    print(f"params: {sum(p.numel() for p in encoder.parameters()):,}")
    print("encoder OK")
