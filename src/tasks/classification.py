"""Classification task implementation."""

from __future__ import annotations

import torch

from framework.contracts.task import ProblemType, Task
from framework.contracts.types import Batch, LossValue, ModelOutput, Predictions, Targets
from framework.losses.registry import create_loss_fn


class ClassificationTask(Task):
    """Multilabel classification task."""

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        loss: str = "bce",
        loss_gamma: float = 2.0,
        loss_alpha: float | None = None,
        **kwargs: object,
    ) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.loss_fn = create_loss_fn(
            loss,
            gamma=loss_gamma,
            alpha=loss_alpha,
        )

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue:
        targets = self.extract_targets(batch)
        return self.loss_fn(outputs, targets)

    def extract_targets(self, batch: Batch) -> Targets:
        return batch["label"]

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions:
        # Return probabilities for metrics
        return torch.sigmoid(outputs)

    def infer_problem_type(self) -> ProblemType:
        return "multilabel"
