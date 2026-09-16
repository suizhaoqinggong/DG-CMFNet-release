"""AUC metrics: AUROC and AUPRC."""

from typing import Any

import torch

from framework.contracts.types import MetricResults, Predictions, Targets
from framework.metrics._helpers import prepare_metric_inputs


class AUCMetric:
    """Compute AUROC and AUPRC for multilabel classification."""

    name = "auc"
    higher_is_better = True

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self._probs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, preds: Predictions, targets: Targets) -> None:
        preds, targets = prepare_metric_inputs(preds, targets)
        valid_mask = ~torch.isnan(preds).any(dim=1)
        if not valid_mask.any():
            return
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        self._probs.append(torch.sigmoid(preds))
        self._targets.append(targets)

    def compute(self) -> MetricResults:
        if not self._probs:
            return {"auroc": 0.0, "auprc": 0.0}

        probs = torch.cat(self._probs, dim=0)
        targets = torch.cat(self._targets, dim=0)

        auroc_vals = []
        auprc_vals = []
        for i in range(self.num_classes):
            y_true = targets[:, i]
            y_score = probs[:, i]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue  # skip classes with all same label
            # manual AUROC using sklearn-like logic
            order = torch.argsort(y_score, descending=True)
            y_sorted = y_true[order]
            tps = torch.cumsum(y_sorted, dim=0)
            fps = torch.cumsum(1 - y_sorted, dim=0)
            tpr = tps / tps[-1]
            fpr = fps / fps[-1]
            auroc = torch.trapz(tpr, fpr).item()
            # AUPRC
            precision = tps / (tps + fps + 1e-8)
            auprc = torch.trapz(precision, tpr).item()
            auroc_vals.append(auroc)
            auprc_vals.append(auprc)

        return {
            "auroc": sum(auroc_vals) / len(auroc_vals) if auroc_vals else 0.0,
            "auprc": sum(auprc_vals) / len(auprc_vals) if auprc_vals else 0.0,
        }

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
