import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any

class MultiTaskLoss(nn.Module):
    """
    Computes weighted sum of losses for NER, Sentiment, and QA tasks.
    """
    def __init__(self, task_weights: Dict[str, float]):
        super().__init__()
        self.task_weights = task_weights
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()
        self.qa_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        # TODO: Compute individual losses and return weighted sum
        return torch.tensor(0.0)
