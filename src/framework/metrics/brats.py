"""BraTS composite-region metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from framework.contracts.types import MetricResults, Predictions, Targets
from framework.metrics._helpers import scalar_metric_value

BRATS_REGIONS: dict[str, tuple[int, ...]] = {
    "wt": (1, 2, 3),
    "tc": (1, 3),
    "et": (3,),
}


def _torch_region_mask(labels: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    mask = labels == classes[0]
    for class_id in classes[1:]:
        mask = mask | (labels == class_id)
    return mask


def _numpy_region_mask(labels: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    mask = labels == classes[0]
    for class_id in classes[1:]:
        mask = mask | (labels == class_id)
    return mask


class BraTSRegionDiceMetric:
    """Per-case mean Dice for BraTS WT, TC, and ET composite regions."""

    name = "brats_dice"
    higher_is_better = True

    def __init__(self, smooth: float = 1e-6, **kwargs: Any) -> None:
        self.smooth = smooth
        self._per_case_dice: dict[str, list[float]] = {name: [] for name in BRATS_REGIONS}

    def update(self, preds: Predictions, targets: Targets) -> None:
        pred_labels = torch.argmax(preds.detach(), dim=1)
        targets = targets.detach()

        for pred_case, target_case in zip(pred_labels, targets):
            for name, classes in BRATS_REGIONS.items():
                pred_mask = _torch_region_mask(pred_case, classes).float()
                target_mask = _torch_region_mask(target_case, classes).float()
                intersection = (pred_mask * target_mask).sum()
                union = (pred_mask + target_mask).sum()
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                self._per_case_dice[name].append(float(dice))

    def compute(self) -> MetricResults:
        results: MetricResults = {}
        region_means = []
        for name in BRATS_REGIONS:
            vals = self._per_case_dice[name]
            region_mean = sum(vals) / len(vals) if vals else 0.0
            results[name] = scalar_metric_value(region_mean)
            region_means.append(region_mean)
        results["mean"] = scalar_metric_value(sum(region_means) / len(region_means))
        return results

    def reset(self) -> None:
        self._per_case_dice = {name: [] for name in BRATS_REGIONS}


class BraTSRegionHausdorffMetric:
    """Symmetric Hausdorff distance for BraTS WT, TC, and ET regions."""

    name = "brats_hd"
    higher_is_better = False

    def __init__(self, spacing: tuple[float, float, float] | None = None, **kwargs: Any) -> None:
        self.spacing = spacing
        self._totals = {name: 0.0 for name in BRATS_REGIONS}
        self._counts = {name: 0 for name in BRATS_REGIONS}

    def update(self, preds: Predictions, targets: Targets) -> None:
        pred_labels = torch.argmax(preds.detach(), dim=1).cpu().numpy()
        target_labels = targets.detach().cpu().numpy()

        for pred_case, target_case in zip(pred_labels, target_labels):
            for name, classes in BRATS_REGIONS.items():
                pred_mask = _numpy_region_mask(pred_case, classes)
                target_mask = _numpy_region_mask(target_case, classes)
                self._totals[name] += _hausdorff_distance(pred_mask, target_mask, self.spacing)
                self._counts[name] += 1

    def compute(self) -> MetricResults:
        results: MetricResults = {}
        for name in BRATS_REGIONS:
            count = self._counts[name]
            results[name] = self._totals[name] / count if count else 0.0
        results["mean"] = sum(float(results[name]) for name in BRATS_REGIONS) / len(BRATS_REGIONS)
        return results

    def reset(self) -> None:
        self._totals = {name: 0.0 for name in BRATS_REGIONS}
        self._counts = {name: 0 for name in BRATS_REGIONS}


def _hausdorff_distance(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: tuple[float, float, float] | None,
) -> float:
    pred_any = bool(pred_mask.any())
    target_any = bool(target_mask.any())
    if not pred_any and not target_any:
        return 0.0
    if pred_any != target_any:
        sampling = spacing or (1.0, 1.0, 1.0)
        return math.sqrt(sum(((size - 1) * step) ** 2 for size, step in zip(pred_mask.shape, sampling)))

    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    sampling = spacing or (1.0, 1.0, 1.0)
    pred_to_target = distance_transform_edt(~target_surface, sampling=sampling)[pred_surface]
    target_to_pred = distance_transform_edt(~pred_surface, sampling=sampling)[target_surface]
    return float(max(pred_to_target.max(initial=0.0), target_to_pred.max(initial=0.0)))


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded
