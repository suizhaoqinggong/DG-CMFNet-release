"""IoU (Jaccard index) metric for semantic segmentation."""

from __future__ import annotations

from typing import Any

import torch

from framework.contracts.metric import Metric
from framework.contracts.types import MetricResults, Predictions, Targets
from framework.metrics._helpers import scalar_metric_value


class IoUMetric:
    """Multi-class IoU (Intersection over Union) for segmentation."""

    name = "iou"
    higher_is_better = True

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-6,
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self._intersection: torch.Tensor | None = None
        self._total: torch.Tensor | None = None

    def update(self, preds: Predictions, targets: Targets) -> None:
        preds = preds.detach()
        targets = targets.detach()
        # preds: [B, C, ...] softmax probabilities
        # targets: [B, ...] integer labels

        if self._intersection is None:
            self._intersection = torch.zeros(self.num_classes, device=preds.device)
            self._total = torch.zeros(self.num_classes, device=preds.device)

        assert self._total is not None

        pred_labels = torch.argmax(preds, dim=1)  # [B, ...]

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            pred_labels = pred_labels[mask]
            targets = targets[mask]

        for c in range(self.num_classes):
            pred_c = (pred_labels == c).float()
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = ((pred_c + target_c) > 0).float().sum()
            self._intersection[c] += intersection
            self._total[c] += union

    def compute(self) -> MetricResults:
        if self._intersection is None:
            return {"mean": 0.0}

        iou_per_class = (self._intersection + self.smooth) / (self._total + self.smooth)
        results: MetricResults = {f"class_{c}": scalar_metric_value(iou_per_class[c]) for c in range(self.num_classes)}
        results["mean"] = scalar_metric_value(iou_per_class.mean())
        return results

    def reset(self) -> None:
        self._intersection = None
        self._total = None
