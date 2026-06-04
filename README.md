# Thai NLP Toolkit with Custom Transformer

A multi-task NLP framework for the Thai language implemented in PyTorch from scratch. The system shares a single Transformer encoder backbone and branches into three task-specific heads:
1. **Named Entity Recognition (NER)** (Token-level classification)
2. **Sentiment Analysis** (Sentence-level classification)
3. **Question Answering (QA)** (Extractive span prediction)

## Key Features
- **Custom Thai Tokenizer**: Pre-tokenization using PyThaiNLP rules combined with a SentencePiece BPE tokenizer trained from scratch.
- **Custom Transformer Backbone**: Pure PyTorch implementation of Multi-Head Self-Attention, Feed Forward Networks, Layer Normalization, and Stacked Encoder Blocks.
- **Multi-task Learning**: Shared representation block optimized jointly with three distinct objective functions and dynamic weighting.
- **Optimized Training**: Built-in support for mixed-precision (`torch.amp`), gradient accumulation, and learning rate scheduling with warmup.
- **Inference API**: Lightweight FastAPI server for model predictions.

## Project Structure
Detailed structural components are defined in [thai_nlp_toolkit_spec.md](file:///c:/Users/jarun/OneDrive/Desktop/thai-nlp-toolkit/thai_nlp_toolkit_spec.md).

## Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Usage
- Refer to `scripts/train.py` for training commands.
- Refer to `scripts/evaluate.py` for evaluation commands.
