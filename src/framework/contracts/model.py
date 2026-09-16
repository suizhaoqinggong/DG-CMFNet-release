"""ModelAdapter protocol definition."""

from typing import Protocol, Iterator, Mapping

from torch import Tensor, nn
from torch.nn import Module, Parameter

from .types import Batch, ModelOutput


class ModelAdapter(Protocol):
    """Protocol for models that can be used by the framework."""

    def forward(self, batch: Batch) -> ModelOutput:
        """Forward pass: receives a batch, returns logits."""
        ...

    def train(self, mode: bool = True) -> Module:
        ...

    def eval(self) -> Module:
        ...

    def to(self, device: object) -> Module:
        ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        ...

    def state_dict(self) -> Mapping[str, object]:
        ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        ...
