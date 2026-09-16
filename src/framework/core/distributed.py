"""Distributed training helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from framework.contracts.types import MetricResults
from framework.core.device import resolve_device


@dataclass(frozen=True)
class DistributedContext:
    """Runtime state for torch.distributed training."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def infer_distributed_context(environ: Mapping[str, str] | None = None) -> DistributedContext:
    """Infer DDP rank information from torchrun environment variables."""
    env = os.environ if environ is None else environ
    world_size = _env_int(env, "WORLD_SIZE", 1)
    rank = _env_int(env, "RANK", 0)
    local_rank = _env_int(env, "LOCAL_RANK", 0)
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def setup_distributed(device_spec: str | None = "auto") -> DistributedContext:
    """Initialize torch.distributed when launched with torchrun."""
    context = infer_distributed_context()
    if not context.enabled:
        return replace(context, device=resolve_device(device_spec))

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    if torch.cuda.is_available():
        device = torch.device("cuda", context.local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return replace(context, device=device)


def cleanup_distributed(context: DistributedContext) -> None:
    """Destroy the process group if this process initialized DDP."""
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(value: Any, context: DistributedContext, src: int = 0) -> Any:
    """Broadcast a Python object from src to every rank."""
    if not context.enabled:
        return value
    objects = [value if context.rank == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def barrier(context: DistributedContext) -> None:
    """Synchronize ranks when DDP is enabled."""
    if context.enabled:
        dist.barrier()


def average_metric_dict(metrics: MetricResults, context: DistributedContext) -> MetricResults:
    """Average scalar metrics across ranks."""
    if not context.enabled or not metrics:
        return metrics

    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=context.device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= context.world_size

    averaged: MetricResults = dict(metrics)
    for key, value in zip(keys, values.tolist()):
        averaged[key] = value
    return averaged
