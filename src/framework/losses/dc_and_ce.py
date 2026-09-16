"""Dice + Cross-Entropy loss from nnFormer (faithful copy).

Original source: nnFormer/nnformer/training/loss_functions/dice_loss.py
All building blocks are self-contained (no nnFormer pipeline dependencies).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities (from nnformer/utilities/nd_softmax.py and tensor_utilities.py)
# ---------------------------------------------------------------------------

softmax_helper = lambda x: F.softmax(x, 1)


def sum_tensor(
    inp: torch.Tensor, axes: list[int] | tuple[int, ...], keepdim: bool = False
) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


# ---------------------------------------------------------------------------
# Building blocks (from nnformer/training/loss_functions/)
# ---------------------------------------------------------------------------


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Compatibility layer: target may be float with an extra dimension."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


def get_tp_fp_fn_tn(
    net_output: torch.Tensor,
    gt: torch.Tensor,
    axes: list[int] | tuple[int, ...] | None = None,
    mask: torch.Tensor | None = None,
    square: bool = False,
):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all(i == j for i, j in zip(net_output.shape, gt.shape)):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    """Soft Dice loss with optional batch-level aggregation."""

    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None) -> torch.Tensor:
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


# ---------------------------------------------------------------------------
# Combined loss (from nnformer/training/loss_functions/dice_loss.py)
# ---------------------------------------------------------------------------


class DcAndCeLoss(nn.Module):
    """Dice + Cross-Entropy loss (nnFormer default).

    Matches the original nnFormer usage:
        DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    """

    def __init__(
        self,
        batch_dice: bool = True,
        smooth: float = 1e-5,
        do_bg: bool = False,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = RobustCrossEntropyLoss()
        self.dc = SoftDiceLoss(
            apply_nonlin=softmax_helper,
            batch_dice=batch_dice,
            do_bg=do_bg,
            smooth=smooth,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss
