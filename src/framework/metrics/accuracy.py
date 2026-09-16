"""Accuracy metric."""

from typing import Any

import torch

from framework.contracts.metric import Metric
from framework.contracts.types import MetricResults, Predictions, Targets
from framework.metrics._helpers import prepare_metric_inputs, scalar_metric_value


class AccuracyMetric:
    """Multilabel accuracy metric."""

    name = "accuracy"
    higher_is_better = True

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        self.threshold = threshold
        self._correct = 0
        self._total = 0

    def update(self, preds: Predictions, targets: Targets) -> None:
        preds, targets = prepare_metric_inputs(preds, targets)
        valid_mask = ~torch.isnan(preds).any(dim=1)
        if not valid_mask.any():
            return
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        binary_preds = (preds > self.threshold).float()
        self._correct += (binary_preds == targets).sum().item()
        self._total += targets.numel()

    def compute(self) -> MetricResults:
        if self._total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
