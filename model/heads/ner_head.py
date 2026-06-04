import torch.nn as nn
from torch import Tensor

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
