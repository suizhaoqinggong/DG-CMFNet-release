"""Loss function protocol."""

from typing import Protocol

from torch import Tensor


class ClassificationLoss(Protocol):
    """Protocol for classification loss functions."""

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute loss between logits and targets."""
        ...
