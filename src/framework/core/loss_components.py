"""Helpers for logging optional loss component scalars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from framework.contracts.types import MetricResults

LOSS_COMPONENT_ORDER = ("loss_dice", "loss_focal", "loss_entropy")


def get_loss_components(task: Any) -> dict[str, torch.Tensor]:
    """Return the latest loss component tensors exposed by a task loss function."""
    loss_fn = getattr(task, "loss_fn", None)
    getter = getattr(loss_fn, "get_last_components", None)
    if not callable(getter):
        return {}

    raw_components = getter()
    if not isinstance(raw_components, Mapping):
        return {}

    components: dict[str, torch.Tensor] = {}
    for key, value in raw_components.items():
        if isinstance(value, torch.Tensor):
            components[str(key)] = value.detach()
        else:
            try:
                components[str(key)] = torch.tensor(float(value))
            except (TypeError, ValueError):
                continue
    return components


def update_loss_component_sums(task: Any, totals: dict[str, torch.Tensor], counts: dict[str, int]) -> None:
    """Add the latest batch loss components to running epoch sums (tensor accumulation)."""
    for key, value in get_loss_components(task).items():
        if key in totals:
            totals[key] = totals[key] + value
        else:
            totals[key] = value.clone()
        counts[key] = counts.get(key, 0) + 1


def add_averaged_loss_components(metrics: MetricResults, totals: dict[str, torch.Tensor], counts: dict[str, int]) -> None:
    """Append averaged loss components to a metrics dictionary (converts to float here)."""
    for key in ordered_loss_component_keys(totals):
        metrics[key] = (totals[key] / counts[key]).item()


def ordered_loss_component_keys(values: Mapping[str, Any]) -> list[str]:
    known = [key for key in LOSS_COMPONENT_ORDER if key in values]
    extra = sorted(key for key in values if key.startswith("loss_") and key not in LOSS_COMPONENT_ORDER)
    return known + extra
