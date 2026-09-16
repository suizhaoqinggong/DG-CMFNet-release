"""Metric helper utilities."""

from __future__ import annotations

from typing import Any

import torch


def prepare_metric_inputs(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach and move tensors to CPU for metric computation."""
    return preds.detach().cpu(), targets.detach().cpu()


def scalar_metric_value(value: Any) -> float | int:
    """Extract a scalar from a tensor or numeric value."""
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)
