"""Metric protocol definition."""

from typing import Protocol

from .types import MetricResults, Predictions, Targets


class Metric(Protocol):
    """Protocol for evaluation metrics."""

    name: str
    higher_is_better: bool

    def update(self, preds: Predictions, targets: Targets) -> None:
        """Accumulate predictions and targets from a batch."""
        ...

    def compute(self) -> MetricResults:
        """Compute and return metric values."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...
