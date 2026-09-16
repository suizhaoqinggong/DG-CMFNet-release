"""Uncertainty-aware Hybrid Loss for segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwareHybridLoss(nn.Module):
    """
    UAHL = DiceLoss + lambda_alpha * FocalTerm + lambda_beta * EntropyTerm

    As described in DG-CMFNet paper (Eq. 15-16).
    """

    def __init__(
        self,
        num_classes: int = 4,
        lambda_alpha: float = 1.0,
        lambda_beta: float = 0.5,
        gamma: float = 2.0,
        epsilon: float = 1e-6,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self._last_components: dict[str, torch.Tensor] = {}

    def _dice_loss(
        self,
        probs: torch.Tensor,
        targets_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-class Dice loss (all classes combined)."""
        # probs: [B, C, ...], targets_onehot: [B, C, ...]
        dims = list(range(2, probs.ndim))
        intersection = (probs * targets_onehot).sum(dim=dims)
        union = (probs + targets_onehot).sum(dim=dims)
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice.mean()

    def _focal_term(
        self,
        probs: torch.Tensor,
        targets_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Focal term over target-class probabilities."""
        probs_clamped = torch.clamp(probs.float(), min=self.epsilon, max=1.0 - self.epsilon)
        valid_targets = targets_onehot.sum(dim=1) > 0
        target_probs = (probs_clamped * targets_onehot).sum(dim=1)
        target_probs = torch.clamp(target_probs, min=self.epsilon, max=1.0 - self.epsilon)
        focal = -((1 - target_probs) ** self.gamma) * torch.log(target_probs)
        if not torch.any(valid_targets):
            return probs.new_tensor(0.0)
        return focal[valid_targets].mean()

    def _entropy_term(self, probs: torch.Tensor) -> torch.Tensor:
        """Predictive uncertainty via entropy."""
        probs_clamped = torch.clamp(probs, min=self.epsilon, max=1.0 - self.epsilon)
        entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=1)
        return entropy.mean()

    def get_last_components(self) -> dict[str, torch.Tensor]:
        """Return the raw component tensors from the latest forward pass (detached)."""
        return dict(self._last_components)

    def _prepare_inputs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert logits and integer labels to masked probabilities and one-hot labels."""
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            scatter_targets = targets.masked_fill(~mask, 0)
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
            scatter_targets = targets

        targets_onehot = torch.zeros(
            logits.shape[0],
            self.num_classes,
            *logits.shape[2:],
            device=logits.device,
            dtype=logits.dtype,
        )
        targets_onehot = targets_onehot.scatter_(1, scatter_targets.unsqueeze(1), 1.0)

        if self.ignore_index is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(logits)
            logits = logits * mask_expanded
            targets_onehot = targets_onehot * mask_expanded

        return F.softmax(logits, dim=1), targets_onehot

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, D, H, W] raw model outputs
            targets: [B, D, H, W] integer class labels
        Returns:
            scalar loss tensor
        """
        probs, targets_onehot = self._prepare_inputs(logits, targets)

        # Compute components
        loss_dice = self._dice_loss(probs, targets_onehot)
        loss_focal = self._focal_term(probs, targets_onehot)
        loss_entropy = self._entropy_term(probs)

        self._last_components = {
            "loss_dice": loss_dice.detach(),
            "loss_focal": loss_focal.detach(),
            "loss_entropy": loss_entropy.detach(),
        }
        total = loss_dice + self.lambda_alpha * loss_focal + self.lambda_beta * loss_entropy
        return total


class SoftmaxFocalLoss(UncertaintyAwareHybridLoss):
    """Focal-only segmentation loss using the same term as U-AHL Eq. 16."""

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        epsilon: float = 1e-6,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            lambda_alpha=0.0,
            lambda_beta=0.0,
            gamma=gamma,
            epsilon=epsilon,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets_onehot = self._prepare_inputs(logits, targets)
        loss_focal = self._focal_term(probs, targets_onehot)
        self._last_components = {"loss_focal": loss_focal.detach()}
        return loss_focal
