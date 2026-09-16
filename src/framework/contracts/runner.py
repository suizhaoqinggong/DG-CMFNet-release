"""Runner protocol definition."""

from __future__ import annotations

from typing import Protocol

from .types import MetricResults


class Runner(Protocol):
    """Protocol for training runners."""

    def train(self) -> MetricResults:
        """Run training and return best validation metrics."""
        ...

    def validate(self, checkpoint_path: str | None = None) -> MetricResults:
        """Run validation, optionally loading a checkpoint."""
        ...

    def test(self, checkpoint_path: str | None = None) -> MetricResults:
        """Run test evaluation, optionally loading a checkpoint."""
        ...

    def fit(self) -> MetricResults:
        """Train then test (fit = train + test)."""
        ...
