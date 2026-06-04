from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Any

def get_optimizer(model: Any, learning_rate: float, weight_decay: float = 0.01) -> AdamW:
    """Configures AdamW optimizer with weight decay exclusion for bias/LayerNorm."""
    # TODO: Implement layer-wise weight decay configurations
    return AdamW(model.parameters(), lr=learning_rate)

def get_scheduler(optimizer: Any, warmup_steps: int, max_steps: int) -> LambdaLR:
    """Creates a linear warmup and linear decay learning rate scheduler."""
    # TODO: Implement scheduling logic
    return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
