"""Core type definitions for the framework."""

from typing import Any, Dict, TypedDict, Union

import torch


class Sample(TypedDict):
    """A single data sample."""

    signal: torch.Tensor
    label: torch.Tensor
    id: str
    meta: dict[str, Any]


class Batch(TypedDict):
    """A batched collection of samples."""

    signal: torch.Tensor
    label: torch.Tensor
    id: list[str]
    meta: list[dict[str, Any]]


# Type aliases for model outputs and targets
ModelOutput = torch.Tensor
Predictions = torch.Tensor
Targets = torch.Tensor
LossValue = torch.Tensor
MetricResults = Dict[str, Union[float, int]]
