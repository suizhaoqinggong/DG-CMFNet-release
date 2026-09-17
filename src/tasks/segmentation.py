"""3D semantic segmentation task implementation."""

from __future__ import annotations

import torch

from framework.contracts.task import ProblemType, Task
from framework.contracts.types import Batch, LossValue, ModelOutput, Predictions, Targets
from framework.losses.uahl import UncertaintyAwareHybridLoss


class SegmentationTask(Task):
    """Multi-class 3D semantic segmentation task."""

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        loss: str = "uahl",
        loss_lambda_alpha: float = 1.0,
        loss_lambda_beta: float = 0.05,
        loss_gamma: float = 2.0,
        loss_balance_param: float = 1.0,
        **kwargs: object,
    ) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.loss_name = loss

        if loss == "uahl":
            self.loss_fn = UncertaintyAwareHybridLoss(
                num_classes=num_classes,
                lambda_alpha=loss_lambda_alpha,
                lambda_beta=loss_lambda_beta,
                gamma=loss_gamma,
            )
        elif loss == "dice":
            # Simple Dice loss fallback
            from torch.nn import CrossEntropyLoss

            self.loss_fn = CrossEntropyLoss()
        elif loss == "ce":
            from torch.nn import CrossEntropyLoss

            self.loss_fn = CrossEntropyLoss()
        elif loss == "focal":
            from framework.losses.uahl import SoftmaxFocalLoss

            self.loss_fn = SoftmaxFocalLoss(num_classes=num_classes, gamma=loss_gamma)
        else:
            raise ValueError(f"Unknown segmentation loss: {loss}")

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue:
        targets = self.extract_targets(batch)
        if self.loss_name in ("ce", "dice"):
            # CrossEntropyLoss expects [B, C, ...] logits and [B, ...] targets
            return self.loss_fn(outputs, targets)
        return self.loss_fn(outputs, targets)

    def extract_targets(self, batch: Batch) -> Targets:
        return batch["label"]

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions:
        if self.num_classes == 4 and outputs.ndim >= 5 and outputs.shape[1] == 3:
            return self._postprocess_brats_region_outputs(outputs)
        return torch.softmax(outputs, dim=1)

    @staticmethod
    def _postprocess_brats_region_outputs(outputs: torch.Tensor) -> torch.Tensor:
        """Convert NestedFormer TC/WT/ET region logits to 4-class probabilities.

        NestedFormer's BraTS setup predicts three sigmoid region channels in
        TC, WT, ET order. DG-CMFNet metrics expect class probabilities ordered
        as background, NCR/NET, ED, ET.
        """
        region_probs = torch.sigmoid(outputs)
        tc = region_probs[:, 0]
        wt = region_probs[:, 1]
        et = region_probs[:, 2]

        background = (1.0 - wt).clamp(min=0.0, max=1.0)
        ncr_net = (tc - et).clamp(min=0.0, max=1.0)
        edema = (wt - tc).clamp(min=0.0, max=1.0)
        enhancing = et.clamp(min=0.0, max=1.0)

        probs = torch.stack([background, ncr_net, edema, enhancing], dim=1)
        normalizer = probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probs / normalizer

    def infer_problem_type(self) -> ProblemType:
        return "multiclass"
