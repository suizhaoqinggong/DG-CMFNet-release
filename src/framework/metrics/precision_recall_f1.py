"""Precision, Recall, F1 metrics."""

from typing import Any

import torch

from framework.contracts.types import MetricResults, Predictions, Targets
from framework.metrics._helpers import prepare_metric_inputs


class PrecisionRecallF1Metric:
    """Precision, Recall, and F1 score for multilabel classification."""

    name = "precision_recall_f1"
    higher_is_better = True

    def __init__(self, average: str = "macro", threshold: float = 0.5, **kwargs: Any) -> None:
        self.average = average
        self.threshold = threshold
        self._tp: list[int] = []
        self._fp: list[int] = []
        self._fn: list[int] = []

    def update(self, preds: Predictions, targets: Targets) -> None:
        preds, targets = prepare_metric_inputs(preds, targets)
        # Mask out NaN predictions
        valid_mask = ~torch.isnan(preds).any(dim=1)
        if not valid_mask.any():
            return
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        binary_preds = (preds > self.threshold).float()

        tp = (binary_preds * targets).sum(dim=0)
        fp = (binary_preds * (1 - targets)).sum(dim=0)
        fn = ((1 - binary_preds) * targets).sum(dim=0)

        if not self._tp:
            self._tp = [0] * len(tp)
            self._fp = [0] * len(tp)
            self._fn = [0] * len(tp)

        for i in range(len(tp)):
            self._tp[i] += int(tp[i].item())
            self._fp[i] += int(fp[i].item())
            self._fn[i] += int(fn[i].item())

    def compute(self) -> MetricResults:
        if not self._tp:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precisions = []
        recalls = []
        f1s = []
        for tp, fp, fn in zip(self._tp, self._fp, self._fn):
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        if self.average == "macro":
            return {
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
                "f1": sum(f1s) / len(f1s),
            }
        elif self.average == "micro":
            total_tp = sum(self._tp)
            total_fp = sum(self._fp)
            total_fn = sum(self._fn)
            precision = total_tp / (total_tp + total_fp + 1e-8)
            recall = total_tp / (total_tp + total_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return {"precision": precision, "recall": recall, "f1": f1}
        else:
            raise ValueError(f"Unknown average: {self.average}")

    def reset(self) -> None:
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()
