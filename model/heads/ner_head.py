import torch.nn as nn
from torch import Tensor

# Label maps for NER token-level classification
NER_ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
}
NER_LABEL2ID = {v: k for k, v in NER_ID2LABEL.items()}


class NERHead(nn.Module):
    """
    Token-level classification head for Named Entity Recognition.
    Labels include: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O
    """
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        # Linear layer mapping hidden states to label logits
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Input shape: (B, T, d_model)
        # Returns shape: (B, T, num_labels)
        return self.classifier(hidden_states)
