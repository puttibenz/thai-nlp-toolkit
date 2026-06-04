# Thai NLP Toolkit with Custom Transformer — Project Spec

## Overview

Build a **multi-task NLP framework for Thai language** using PyTorch from scratch.
The system shares a single Transformer backbone and branches into three task-specific heads:
Named Entity Recognition (NER), Sentiment Analysis, and Question Answering (QA).

Target environment: Python 3.10+, PyTorch 2.x, NVIDIA GPU (CUDA 12.x).

---

## Goals

- Custom Thai tokenizer (no reliance on HuggingFace tokenizers)
- Full Transformer implementation as `nn.Module` (attention, FFN, layer norm)
- Multi-task learning: one backbone, three heads, joint training
- Mixed-precision training (`torch.amp`) + gradient accumulation
- Evaluation with task-specific metrics (F1, Accuracy, Exact Match)
- Simple FastAPI inference endpoint

---

## Project Structure

```
thai-nlp-toolkit/
├── README.md
├── requirements.txt
├── configs/
│   └── base_config.yaml          # model hyperparams, training settings
├── tokenizer/
│   ├── __init__.py
│   ├── thai_tokenizer.py         # SentencePiece wrapper + Thai-specific rules
│   └── train_tokenizer.py        # script to train vocab from corpus
├── model/
│   ├── __init__.py
│   ├── embedding.py              # Token embedding + positional encoding
│   ├── attention.py              # Multi-head self-attention (nn.Module)
│   ├── transformer_block.py      # Single encoder block (attn + FFN + LayerNorm)
│   ├── encoder.py                # Stack of N transformer blocks
│   └── heads/
│       ├── ner_head.py           # Token-level classification head
│       ├── sentiment_head.py     # [CLS] pooling + linear classifier
│       └── qa_head.py            # Start/end span prediction head
├── data/
│   ├── __init__.py
│   ├── datasets.py               # torch.utils.data.Dataset for each task
│   ├── collator.py               # Custom collate_fn with dynamic padding
│   └── download.py               # Script to fetch public Thai datasets
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Main training loop (multi-task)
│   ├── optimizer.py              # AdamW + linear warmup scheduler
│   └── losses.py                 # Task-weighted combined loss
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # F1 (NER), Accuracy (Sentiment), EM/F1 (QA)
├── inference/
│   ├── __init__.py
│   ├── pipeline.py               # High-level inference class
│   └── api.py                    # FastAPI app with /predict endpoint
└── scripts/
    ├── train.py                  # Entry point for training
    └── evaluate.py               # Entry point for evaluation
```

---

## Module Specifications

### `tokenizer/thai_tokenizer.py`

```python
class ThaiTokenizer:
    """
    Wraps SentencePiece with Thai-specific preprocessing.
    Handles lack of whitespace word boundaries in Thai text.
    """
    def __init__(self, model_path: str, vocab_size: int = 32000): ...
    def train(self, corpus_path: str): ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def batch_encode(self, texts: list[str], max_length: int, padding: bool) -> dict: ...
```

Dependencies: `sentencepiece`, `pythainlp` (for pre-tokenization rules)

---

### `model/attention.py`

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product attention with multiple heads.
    Supports causal masking and padding mask.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1): ...
    def forward(
        self,
        x: Tensor,                    # (B, T, d_model)
        key_padding_mask: Tensor = None,   # (B, T) bool
        attn_mask: Tensor = None,          # (T, T) for causal
    ) -> tuple[Tensor, Tensor]: ...        # output, attn_weights
```

Implementation notes:
- Split Q, K, V via single linear projection then chunk
- Scale by `sqrt(d_k)` before softmax
- Apply dropout to attention weights
- Return attention weights for visualization

---

### `model/transformer_block.py`

```python
class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, d_model, num_heads, d_ff, dropout): ...
    def forward(self, x, padding_mask=None) -> Tensor: ...
    # LayerNorm → Attention → residual → LayerNorm → FFN → residual
```

---

### `model/encoder.py`

```python
class ThaiTransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig): ...
    def forward(self, input_ids, attention_mask) -> Tensor:
        # Returns (B, T, d_model) — full sequence hidden states
        ...
```

Config fields: `vocab_size`, `d_model`, `num_heads`, `num_layers`, `d_ff`, `max_seq_len`, `dropout`

---

### `model/heads/ner_head.py`

```python
class NERHead(nn.Module):
    """Token-level classification. Labels: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O"""
    def __init__(self, d_model: int, num_labels: int): ...
    def forward(self, hidden_states: Tensor) -> Tensor:
        # Returns (B, T, num_labels) logits
        ...
```

---

### `model/heads/sentiment_head.py`

```python
class SentimentHead(nn.Module):
    """Sentence-level classification using [CLS] token or mean pooling."""
    def __init__(self, d_model: int, num_classes: int = 3): ...
    # Classes: positive, neutral, negative
    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Returns (B, num_classes) logits
        ...
```

---

### `model/heads/qa_head.py`

```python
class QAHead(nn.Module):
    """Extractive QA — predict start and end token positions."""
    def __init__(self, d_model: int): ...
    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        # Returns start_logits (B, T), end_logits (B, T)
        ...
```

---

### `training/trainer.py`

```python
class MultiTaskTrainer:
    def __init__(self, model, tokenizer, config, train_loaders, val_loaders): ...

    def train_epoch(self) -> dict:
        # Uses torch.amp.autocast for mixed precision
        # Accumulates gradients every `config.grad_accum_steps` steps
        # Clips gradients to config.max_grad_norm
        ...

    def evaluate(self, task: str) -> dict: ...

    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...
```

Training loop pseudocode:
```
for batch in loader:
    with autocast():
        ner_loss, sent_loss, qa_loss = model(batch)
        loss = w_ner * ner_loss + w_sent * sent_loss + w_qa * qa_loss
        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()

    if step % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
```

---

## Datasets

| Task | Dataset | Source | Size |
|---|---|---|---|
| NER | BEST2020 | NECTEC | ~3k sentences |
| Sentiment | Wisesight Sentiment | Kaggle | ~26k samples |
| QA | iApp Thai QA | iApp Technology | ~5k pairs |
| Pre-train vocab | ThaiWiki dump | Wikipedia | ~1GB text |

Download script: `python data/download.py --task all`

---

## Config (`configs/base_config.yaml`)

```yaml
model:
  vocab_size: 32000
  d_model: 256
  num_heads: 8
  num_layers: 6
  d_ff: 1024
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 32
  grad_accum_steps: 4       # effective batch = 128
  learning_rate: 3e-4
  warmup_steps: 1000
  max_steps: 50000
  max_grad_norm: 1.0
  mixed_precision: true
  task_weights:
    ner: 1.0
    sentiment: 0.8
    qa: 1.2

evaluation:
  eval_every: 500           # steps
  save_every: 1000

paths:
  data_dir: ./data/raw
  output_dir: ./outputs
  tokenizer_model: ./tokenizer/thai_bpe.model
```

---

## Requirements (`requirements.txt`)

```
torch>=2.1.0
sentencepiece>=0.1.99
pythainlp>=4.0.0
datasets>=2.14.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
wandb>=0.15.0
fastapi>=0.103.0
uvicorn>=0.23.0
scikit-learn>=1.3.0
```

---

## Inference API (`inference/api.py`)

```
POST /predict
{
  "text": "สมเด็จพระราชินีเสด็จกรุงเทพฯ",
  "tasks": ["ner", "sentiment"]
}

Response:
{
  "ner": [{"token": "สมเด็จพระราชินี", "label": "B-PER"}, ...],
  "sentiment": {"label": "neutral", "confidence": 0.87}
}
```

Run: `uvicorn inference.api:app --host 0.0.0.0 --port 8000`

---

## Implementation Order (recommended)

1. `tokenizer/` — train vocab, verify encode/decode round-trip
2. `model/embedding.py` → `attention.py` → `transformer_block.py` → `encoder.py`
3. Unit test encoder with dummy input: `(B=2, T=64)` → output shape `(2, 64, 256)`
4. `model/heads/` — implement each head, verify output shapes
5. `data/` — write Dataset classes, verify batching
6. `training/trainer.py` — implement training loop on small subset first
7. Full training run with GPU
8. `evaluation/metrics.py` + `scripts/evaluate.py`
9. `inference/` — pipeline + FastAPI

---

## Notes for Coding Agent

- ใช้ `nn.Module` ทุก layer ห้าม hardcode weight
- ทุก forward pass ต้องรับ `attention_mask` และส่งต่อไปจนถึง attention layer
- ใช้ `@dataclass` สำหรับ config objects
- เขียน docstring และ type hints ทุก function
- Unit test ทุก module ก่อน integrate: ตรวจ output shape, gradient flow, และ loss ลดลง
- Device-agnostic: ใช้ `model.to(device)` ไม่ hardcode `cuda`
