import sys
import pathlib

# Add project root to sys.path
root = pathlib.Path(__file__).resolve().parent
while root.parent != root:
    if (root / "requirements.txt").exists() or (root / "README.md").exists():
        sys.path.append(str(root))
        break
    root = root.parent

import torch
from model.heads.ner_head import NERHead
from model.heads.sentiment_head import SentimentHead
from model.heads.qa_head import QAHead

B, T, D = 2, 32, 256
hidden = torch.randn(B, T, D)
mask   = torch.ones(B, T).long()
mask[1, 20:] = 0  # batch 2 มี padding

# NER
ner  = NERHead(d_model=D, num_labels=7)
out  = ner(hidden)
assert out.shape == (B, T, 7)

# Sentiment
sent = SentimentHead(d_model=D, num_classes=3)
out  = sent(hidden, mask)
assert out.shape == (B, 3)
assert not torch.isnan(out).any()

# QA
qa   = QAHead(d_model=D)
s, e = qa(hidden)
assert s.shape == (B, T) and e.shape == (B, T)

print("all heads OK")
