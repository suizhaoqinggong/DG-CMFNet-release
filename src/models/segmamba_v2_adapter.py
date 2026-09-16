"""SegMamba-V2 adapter for the DG-CMFNet training framework.

The original network stays in the separately cloned upstream repository. This
module loads its BraTS implementation and translates the framework batch and
configuration interfaces to the original ``SegMamba`` constructor.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ()

_MISSING_SOURCE_MESSAGE = (
    "SegMambaV2Adapter requires the upstream SegMamba-V2 source tree. "
    "Set source_root in the model config or export SEGMAMBA_V2_SOURCE_ROOT."
)

_MISSING_DEPENDENCY_MESSAGE = (
    "SegMambaV2Adapter requires MONAI, einops, mamba-ssm, and causal-conv1d. "
    "Install them in this project with: uv sync --extra segmamba-v2"
)


def _as_int_tuple(values: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} integers, got {result}")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _resolve_source_file(source_root: str | os.PathLike[str] | None) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if source_root:
        candidates.append(source_root)
    env_root = os.environ.get("SEGMAMBA_V2_SOURCE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(_DEFAULT_SOURCE_ROOTS)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        source_candidates = (
            root / "brats23" / "models_segmamba" / "segmambav2.py",
            root / "models_segmamba" / "segmambav2.py",
        )
        for source_file in source_candidates:
            if source_file.is_file():
                return source_file

    checked = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(f"{_MISSING_SOURCE_MESSAGE} Checked: {checked}")


def _load_upstream_network_class(source_file: Path) -> type[nn.Module]:
    module_name = "_dgcmfnet_upstream_segmamba_v2"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SegMamba-V2 module from {source_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        dependency_roots = {"monai", "einops", "mamba_ssm", "causal_conv1d"}
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in dependency_roots:
            raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc
        raise

    network_class = getattr(module, "SegMamba", None)
    if network_class is None:
        raise ImportError(f"Upstream module {source_file} does not define SegMamba")
    return cast(type[nn.Module], network_class)


class SegMambaV2Adapter(nn.Module):
    """Wrap the original BraTS SegMamba-V2 network.

    Input batches must contain ``signal`` with shape ``[B, C, D, H, W]``.
    Spatial dimensions must be divisible by 16 because the original encoder
    performs four stride-two downsampling operations.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        img_size: Sequence[int] = (128, 128, 128),
        depths: Sequence[int] = (2, 2, 2, 2),
        feat_size: Sequence[int] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = _as_int_tuple(img_size, "img_size", 3)
        self.depths = _as_int_tuple(depths, "depths", 4)
        self.feat_size = _as_int_tuple(feat_size, "feat_size", 4)
        self.strict_img_size = bool(strict_img_size)

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if self.num_modalities < 1:
            raise ValueError(f"num_modalities must be positive, got {self.num_modalities}")
        if any(size % 16 != 0 for size in self.img_size):
            raise ValueError(f"img_size must be divisible by 16 in every dimension, got {self.img_size}")
        if self.feat_size[0] != 48:
            raise ValueError(
                "The upstream SegMamba-V2 output block is fixed to 48 input channels; "
                f"feat_size[0] must therefore be 48, got {self.feat_size[0]}"
            )
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        source_file = _resolve_source_file(source_root)
        network_class = _load_upstream_network_class(source_file)
        network_kwargs: dict[str, Any] = {
            "in_chans": self.num_modalities,
            "out_chans": self.num_classes,
            "depths": list(self.depths),
            "feat_size": list(self.feat_size),
            "drop_path_rate": float(drop_path_rate),
            "layer_scale_init_value": float(layer_scale_init_value),
            "hidden_size": int(hidden_size),
            "norm_name": norm_name,
            "conv_block": bool(conv_block),
            "res_block": bool(res_block),
            "spatial_dims": 3,
        }
        self.model = network_class(**network_kwargs)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SegMambaV2Adapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        spatial_shape = tuple(int(value) for value in x.shape[2:])
        if any(size % 16 != 0 for size in spatial_shape):
            raise ValueError(f"Input spatial dimensions must be divisible by 16, got {spatial_shape}")
        if self.strict_img_size and spatial_shape != self.img_size:
            raise ValueError(f"SegMambaV2Adapter expects spatial shape {self.img_size}, got {spatial_shape}")

        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"SegMambaV2Adapter expected tensor logits, got {type(logits)}")
        return logits
