"""A simple fully-connected network for demonstration."""

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput


class SimpleNet(nn.Module):
    """Simple feed-forward classifier."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        # Flatten if needed (e.g., [B, C, L] -> [B, C*L])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
