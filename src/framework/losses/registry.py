"""Loss function factory."""

from __future__ import annotations

import torch.nn as nn

from framework.losses.focal import FocalLoss


def create_loss_fn(
    name: str,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> nn.Module:
    """Create a loss function by name."""
    loss_name = name.lower().strip()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss function: {name}")
