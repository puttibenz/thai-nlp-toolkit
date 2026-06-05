import torch
import math
import torch.nn as nn
from torch import Tensor

class TokenEmbedding(nn.Module):
    """Token Embedding lookup layer."""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model
        # scale ตาม "Attention is All You Need" paper
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)
        # reset padding_idx เป็น zeros หลัง จาก init
        nn.init.constant_(self.embedding.weight.data[0], 0)

    def forward(self, input_ids: Tensor) -> Tensor:
        # scale embedding ด้วย √d_model ให้ magnitude ไม่จม PE
        return self.embedding(input_ids) * math.sqrt(self.d_model)  # scale embedding by sqrt(d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # สร้าง PE matrix ขนาด (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1).float()  # (max_seq_len, 1)

        # div_term: 1 / 10000^(2i/d_model) — คำนวณใน log-space เพื่อ numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # register_buffer: ไม่ใช่ parameter (ไม่ถูก update) แต่ติดไปกับ model.state_dict()
        # unsqueeze(0) → (1, T, d_model) เพื่อ broadcast กับ batch
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, T, d_model)
        # Adds PE to input tensor and applies dropout
        x = x+ self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ThaiEmbedding(nn.Module):
    """Token + positional embedding รวมกัน พร้อม layer norm."""
    def __init__(self, vocab_size: int, d_model: int,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.token_emb(input_ids)   # (B, T, d_model)
        x = self.pos_enc(x)             # บวก PE + dropout
        return self.norm(x)             # stabilize ก่อนเข้า transformer blocks

if __name__ == "__main__":
    emb = ThaiEmbedding(vocab_size=32000, d_model=256)
    emb.eval()
    ids = torch.randint(1, 32000, (2, 64))   # batch=2, seq_len=64
    out = emb(ids)

    assert out.shape == (2, 64, 256)
    assert not torch.isnan(out).any()
    # padding token (id=0) ต้องได้ PE อย่างเดียว ไม่มี token embedding
    pad_ids = torch.zeros(1, 10, dtype=torch.long)
    pad_out = emb.token_emb(pad_ids)
    assert pad_out.abs().sum() == 0, "padding token ต้องเป็น zero vector"
    print("embedding OK")