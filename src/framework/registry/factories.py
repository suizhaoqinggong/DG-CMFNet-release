"""Config parsing and component bundle assembly."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli as tomllib

from framework.contracts.data import DataAdapter
from framework.contracts.model import ModelAdapter
from framework.contracts.task import Task
from framework.core.distributed import DistributedContext, broadcast_object
from framework.metrics.collection import (
    MetricCollection,
    build_split_metric_collections,
)

from .defaults import RegistryBundle, create_default_registries


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base. Mutates base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_merged_experiment_config(
    experiment_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    """Load experiment config and deep-merge model config into it."""
    config = load_toml(experiment_path)
    model_config = load_toml(model_path)
    return deep_merge(config, model_config)


def require_string(config: dict[str, Any], key: str, context: str) -> str:
    """Require a string value from config."""
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context} requires '{key}' to be a string, got {type(value)}")
    return value


def require_list(config: dict[str, Any], key: str, context: str) -> list[Any]:
    """Require a list value from config."""
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context} requires '{key}' to be a list, got {type(value)}")
    return value


def optional_int(value: Any) -> int | None:
    """Parse an optional integer config value."""
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class ComponentBundle:
    """Fully assembled experiment components."""

    data_adapter: DataAdapter
    model: ModelAdapter
    task: Task
    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection
    train_loader: Any
    val_loader: Any
    test_loader: Any
    optimizer: Any


def build_component_bundle(
    config: dict[str, Any],
    registries: RegistryBundle,
    distributed_context: DistributedContext | None = None,
) -> ComponentBundle:
    """Build a component bundle from merged config."""
    from torch.utils.data import DataLoader

    # --- Experiment section ---
    exp_section = config.get("experiment", {})
    class_names = require_list(exp_section, "class_names", "[experiment]")
    num_classes = len(class_names)
    threshold = float(exp_section.get("threshold", 0.5))
    seed = optional_int(exp_section.get("seed"))

    # --- Data adapter ---
    data_section = config.get("data", {})
    data_adapter_name = require_string(data_section, "adapter", "[data]")
    # Filter out generic dataloader params; only pass adapter-specific params
    _dl_keys = {
        "adapter",
        "batch_size",
        "num_workers",
        "shuffle",
        "pin_memory",
        "drop_last",
        "persistent_workers",
        "prefetch_factor",
    }
    data_kwargs = {k: v for k, v in data_section.items() if k not in _dl_keys}
    data_seed = int(data_section.get("seed", seed if seed is not None else 42))
    data_kwargs.setdefault("seed", data_seed)
    data_adapter = registries.data_adapters.create(data_adapter_name, **data_kwargs)
    data_adapter.prepare()

    train_ds, val_ds, test_ds = data_adapter.get_splits()
    batch_size = int(data_section.get("batch_size", 32))
    num_workers = int(data_section.get("num_workers", 0))
    pin_memory = bool(data_section.get("pin_memory", False))
    persistent_workers = bool(data_section.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(data_section.get("prefetch_factor", 2)) if num_workers > 0 else None
    drop_last = bool(data_section.get("drop_last", False))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": data_adapter.collate_fn,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_sampler = None
    if distributed_context is not None and distributed_context.enabled:
        from torch.utils.data.distributed import DistributedSampler

        sampler_seed = seed
        if sampler_seed is None:
            sampler_seed = random.randrange(2**32) if distributed_context.is_main_process else None
            sampler_seed = broadcast_object(sampler_seed, distributed_context)

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=True,
            seed=int(sampler_seed),
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # --- Model ---
    model_section = config.get("model", {})
    model_name = require_string(model_section, "name", "[model]")
    model_kwargs = {k: v for k, v in model_section.items() if k != "name"}
    model_kwargs.setdefault("num_classes", num_classes)
    model = registries.models.create(model_name, **model_kwargs)

    # --- Task ---
    task_section = config.get("task", {})
    task_name = require_string(task_section, "name", "[task]")
    task_kwargs = {k: v for k, v in task_section.items() if k != "name"}
    task_kwargs["num_classes"] = num_classes
    task_kwargs["threshold"] = threshold
    task = registries.tasks.create(task_name, **task_kwargs)

    # --- Metrics ---
    metrics_section = config.get("metrics", {})
    metric_params = metrics_section.get("params", {})
    metric_kwargs = {
        "num_classes": num_classes,
        "threshold": threshold,
        "class_names": class_names,
        "problem_type": task.infer_problem_type(),
    }
    split_metrics = build_split_metric_collections(
        metric_params,
        registries.metrics,
        metric_kwargs,
    )

    # --- Optimizer ---
    optim_section = config.get("optimizer", {})
    optim_name = optim_section.get("name", "adam")
    lr = float(optim_section.get("lr", 1e-3))
    weight_decay = float(optim_section.get("weight_decay", 0.0))
    import torch

    if optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "sgd":
        momentum = float(optim_section.get("momentum", 0.9))
        nesterov = bool(optim_section.get("nesterov", False))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    return ComponentBundle(
        data_adapter=data_adapter,
        model=model,
        task=task,
        train_metrics=split_metrics.train,
        val_metrics=split_metrics.val,
        test_metrics=split_metrics.test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
    )


def build_default_component_bundle(
    experiment_path: str | Path,
    model_path: str | Path,
) -> tuple[ComponentBundle, dict[str, Any], RegistryBundle]:
    """Convenience: load config, create registries, build bundle."""
    config = load_merged_experiment_config(experiment_path, model_path)
    registries = create_default_registries()
    bundle = build_component_bundle(config, registries)
    return bundle, config, registries
