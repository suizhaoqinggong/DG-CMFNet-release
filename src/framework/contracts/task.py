"""Task protocol definition."""

from typing import Literal, Protocol

from .types import Batch, LossValue, ModelOutput, Predictions, Targets

ProblemType = Literal["multilabel", "multiclass", "binary"]


class Task(Protocol):
    """Protocol for tasks that define loss computation and output processing."""

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> LossValue:
        """Compute loss from model outputs and batch."""
        ...

    def extract_targets(self, batch: Batch) -> Targets:
        """Extract ground-truth targets from batch."""
        ...

    def postprocess_outputs(self, outputs: ModelOutput) -> Predictions:
        """Post-process raw model outputs into predictions."""
        ...

    def infer_problem_type(self) -> ProblemType:
        """Return the problem type string."""
        ...
