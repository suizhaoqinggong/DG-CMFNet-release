"""Adapter for the original lhaof/nnMamba segmentation network."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Sequence, cast

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ()

_MISSING_SOURCE_MESSAGE = (
    "NnMambaAdapter requires the upstream lhaof/nnMamba source tree. "
    "Set source_root in the model config or export NNMAMBA_SOURCE_ROOT."
)

_MISSING_DEPENDENCY_MESSAGE = (
    "NnMambaAdapter requires mamba-ssm and causal-conv1d. "
    "Install them in this project with: uv sync --extra nnmamba"
)


class _SegmentationNetwork(nn.Module):
    """Minimal legacy nnU-Net base required by upstream nnMambaSeg."""

    def __init__(self) -> None:
        super().__init__()


def _install_nnunet_import_shim() -> None:
    """Avoid importing the entire legacy nnU-Net runtime for one base class."""
    module_name = "nnunet.network_architecture.neural_network"
    if module_name in sys.modules:
        return

    for package_name in ("nnunet", "nnunet.network_architecture"):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = package

    module = types.ModuleType(module_name)
    setattr(module, "SegmentationNetwork", _SegmentationNetwork)
    sys.modules[module_name] = module


def _as_spatial_shape(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {result}")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return cast(tuple[int, int, int], result)


def _validate_spatial_shape(values: Sequence[int]) -> tuple[int, int, int]:
    spatial_shape = _as_spatial_shape(values, "spatial shape")
    if any(size % 16 != 0 for size in spatial_shape):
        raise ValueError(f"Input spatial dimensions must be divisible by 16, got {spatial_shape}")
    return spatial_shape


def _resolve_source_file(source_root: str | os.PathLike[str] | None) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if source_root:
        candidates.append(source_root)
    env_root = os.environ.get("NNMAMBA_SOURCE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(_DEFAULT_SOURCE_ROOTS)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        source_candidates = (
            root / "nnunet" / "network_architecture" / "nnMamba.py",
            root / "nnMamba.py",
        )
        for source_file in source_candidates:
            if source_file.is_file():
                return source_file

    checked = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(f"{_MISSING_SOURCE_MESSAGE} Checked: {checked}")


def _load_upstream_network_class(source_file: Path) -> type[nn.Module]:
    _install_nnunet_import_shim()
    module_name = "_dgcmfnet_upstream_nnmamba"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load nnMamba module from {source_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in {"mamba_ssm", "causal_conv1d"}:
            raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc
        raise

    network_class = getattr(module, "nnMambaSeg", None)
    if network_class is None:
        raise ImportError(f"Upstream module {source_file} does not define nnMambaSeg")
    return cast(type[nn.Module], network_class)


class NnMambaAdapter(nn.Module):
    """Expose the original 3D nnMamba segmentation model to the framework."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        img_size: Sequence[int] = (128, 128, 128),
        channels: int = 32,
        blocks: int = 3,
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = _validate_spatial_shape(img_size)
        self.strict_img_size = bool(strict_img_size)

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if self.num_modalities < 1:
            raise ValueError(f"num_modalities must be positive, got {self.num_modalities}")
        if channels != 32:
            raise ValueError(
                "The upstream nnMamba auxiliary heads are fixed to channels=32; "
                f"got channels={channels}"
            )
        if blocks < 1:
            raise ValueError(f"blocks must be positive, got {blocks}")

        source_file = _resolve_source_file(source_root)
        network_class = _load_upstream_network_class(source_file)
        self.model = network_class(
            in_ch=self.num_modalities,
            channels=int(channels),
            blocks=int(blocks),
            number_classes=self.num_classes,
        )
        # DG-CMFNet losses consume only full-resolution logits.
        setattr(self.model, "do_ds", False)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"NnMambaAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        spatial_shape = _validate_spatial_shape(x.shape[2:])
        if self.strict_img_size and spatial_shape != self.img_size:
            raise ValueError(f"NnMambaAdapter expects spatial shape {self.img_size}, got {spatial_shape}")

        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"NnMambaAdapter expected tensor logits, got {type(logits)}")
        return logits
