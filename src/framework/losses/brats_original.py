"""Losses used to mirror original BraTS comparison-model training setups."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BRATS_REGION_CLASSES: dict[str, tuple[int, ...]] = {
    "wt": (1, 2, 3),
    "tc": (1, 3),
    "et": (3,),
}


def _targets_without_channel(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim >= 5 and targets.shape[1] == 1:
        targets = targets[:, 0]
    return targets.long()


def _region_targets(targets: torch.Tensor, order: Sequence[str]) -> torch.Tensor:
    targets = _targets_without_channel(targets)
    masks = []
    for region in order:
        classes = BRATS_REGION_CLASSES[region]
        mask = targets == classes[0]
        for class_id in classes[1:]:
            mask = mask | (targets == class_id)
        masks.append(mask.float())
    return torch.stack(masks, dim=1)


def _region_probs_from_softmax_logits(logits: torch.Tensor, order: Sequence[str]) -> torch.Tensor:
    if logits.shape[1] < 4:
        raise ValueError(
            "BraTS original region losses expect framework-native logits with "
            f"at least 4 channels [background, NCR/NET, ED, ET], got {tuple(logits.shape)}"
        )
    probs = F.softmax(logits, dim=1)
    region_probs = {
        "wt": probs[:, 1] + probs[:, 2] + probs[:, 3],
        "tc": probs[:, 1] + probs[:, 3],
        "et": probs[:, 3],
    }
    return torch.stack([region_probs[region] for region in order], dim=1)


class BraTSRegionDiceLoss(nn.Module):
    """SegFormer3D-style Dice loss over BraTS WT/TC/ET regions.

    The original SegFormer3D BraTS setup used sigmoid Dice over region masks.
    DG-CMFNet comparison models keep 4-class logits for shared metrics, so this
    loss derives WT/TC/ET probabilities from the 4-class softmax output.
    """

    def __init__(self, smooth: float = 1e-6, order: Sequence[str] = ("wt", "tc", "et")) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.order = tuple(order)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _region_probs_from_softmax_logits(logits, self.order)
        target_regions = _region_targets(targets, self.order).to(device=logits.device, dtype=probs.dtype)

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target_regions).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_regions.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice.mean()

        self._last_components = {"loss_dice": loss.detach()}
        return loss

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class BraTSEDiceLoss(nn.Module):
    """VT-UNet version_1 EDiceLoss adapted to 4-class DG-CMFNet logits.

    Original EDiceLoss averages ET/TC/WT region losses:
        (0.7 * soft Dice loss + 0.3 * BCE) / 3
    """

    def __init__(
        self,
        smooth: float = 1.0,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        order: Sequence[str] = ("et", "tc", "wt"),
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.order = tuple(order)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = _region_probs_from_softmax_logits(logits, self.order)
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        target_regions = _region_targets(targets, self.order).to(device=logits.device, dtype=probs.dtype)

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target_regions).sum(dim=dims)
        denominator = probs.pow(2).sum(dim=dims) + target_regions.pow(2).sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth))
        dice_loss = dice_loss.mean()

        bce_loss = F.binary_cross_entropy(probs, target_regions)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        self._last_components = {
            "loss_dice": dice_loss.detach(),
            "loss_bce": bce_loss.detach(),
        }
        return total

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class MultiLabelBCEFocalLoss(nn.Module):
    """Attention U-Net-style BCE/Focal loss for framework multi-class labels."""

    def __init__(self, num_classes: int, gamma: float = 0.0, balance_param: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.balance_param = float(balance_param)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _targets_without_channel(targets)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).movedim(-1, 1).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * one_hot + (1.0 - probs) * (1.0 - one_hot)
        focal = self.balance_param * ((1.0 - pt) ** self.gamma) * bce
        loss = focal.mean()

        self._last_components = {"loss_focal": loss.detach()}
        return loss

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class TransBTSSoftmaxDiceLoss(nn.Module):
    """TransBTS softmax Dice loss with explicit background supervision."""

    def __init__(self, smooth: float = 1e-5, classes: Sequence[int] = (0, 1, 2, 3)) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.classes = tuple(int(c) for c in classes)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _targets_without_channel(targets)
        probs = F.softmax(logits, dim=1)
        losses = []
        components: dict[str, torch.Tensor] = {}
        dims = tuple(range(1, targets.ndim))
        for class_id in self.classes:
            pred = probs[:, class_id]
            target = (targets == class_id).to(device=logits.device, dtype=pred.dtype)
            intersection = (pred * target).sum(dim=dims)
            denominator = pred.sum(dim=dims) + target.sum(dim=dims)
            loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            loss = loss.mean()
            losses.append(loss)
            components[f"loss_dice_class_{class_id}"] = loss.detach()
        total = torch.stack(losses).sum()
        self._last_components = components
        return total

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class BraTSSigmoidRegionDiceLoss(nn.Module):
    """NestedFormer-style sigmoid Dice over BraTS WT/TC/ET regions."""

    def __init__(self, smooth: float = 1e-5, order: Sequence[str] = ("tc", "wt", "et")) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.order = tuple(order)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_regions = _region_targets(targets, self.order).to(device=logits.device, dtype=logits.dtype)
        if logits.shape[1] == len(self.order):
            probs = torch.sigmoid(logits)
        else:
            probs = _region_probs_from_softmax_logits(logits, self.order)

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target_regions).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_regions.sum(dim=dims)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = dice_loss.mean()
        self._last_components = {"loss_dice": loss.detach()}
        return loss

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class SwinBTSDiceCELoss(nn.Module):
    """SwinBTS Dice plus cross-entropy loss over its direct TC/WT/ET outputs.

    Jiang et al. specify a three-map output and Dice + CE objective. Because
    TC, WT, and ET are nested targets, the CE term is implemented as binary
    cross entropy over the three sigmoid maps.
    """

    def __init__(self, smooth: float = 1e-5, order: Sequence[str] = ("tc", "wt", "et")) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.order = tuple(order)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != len(self.order):
            raise ValueError(
                "SwinBTSDiceCELoss expects direct region logits in TC/WT/ET order, "
                f"got shape {tuple(logits.shape)}"
            )
        target_regions = _region_targets(targets, self.order).to(device=logits.device, dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * target_regions).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_regions.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        ce_loss = F.binary_cross_entropy_with_logits(logits, target_regions)
        self._last_components = {
            "loss_dice": dice_loss.detach(),
            "loss_ce": ce_loss.detach(),
        }
        return dice_loss + ce_loss

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)


class SlimUNETRFocalDiceLoss(nn.Module):
    """Slim UNETR focal + Dice objective without requiring MONAI at runtime."""

    def __init__(self, num_classes: int, gamma: float = 2.0, dice_weight: float = 1.0, focal_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self._last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _targets_without_channel(targets)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).movedim(-1, 1).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        probs = F.softmax(logits, dim=1)
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()

        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = (log_probs * one_hot).sum(dim=1)
        target_probs = target_log_probs.exp()
        focal_loss = (-((1.0 - target_probs) ** self.gamma) * target_log_probs).mean()

        self._last_components = {
            "loss_dice": dice_loss.detach(),
            "loss_focal": focal_loss.detach(),
        }
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

    def get_last_components(self) -> dict[str, torch.Tensor]:
        return dict(self._last_components)
