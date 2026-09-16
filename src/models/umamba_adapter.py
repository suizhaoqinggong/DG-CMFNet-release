"""U-Mamba adapter for the DG-CMFNet training framework.

The network definition remains in the separately cloned upstream U-Mamba
repository. This module only translates framework configuration and batches to
the constructor and tensor interface used by the original 3D implementation.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ()

_MISSING_SOURCE_MESSAGE = (
    "UMambaAdapter requires the upstream U-Mamba source tree. "
    "Set source_root in the model config or export UMAMBA_SOURCE_ROOT."
)

_MISSING_DEPENDENCY_MESSAGE = (
    "UMambaAdapter requires mamba-ssm and causal-conv1d. "
    "Install them in this project with: uv sync --extra umamba"
)


def _as_3d_tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {result}")
    return result


def _as_3d_tuple_list(values: Sequence[int | Sequence[int]], name: str) -> list[tuple[int, int, int]]:
    return [_as_3d_tuple(value, f"{name}[{index}]") for index, value in enumerate(values)]


def _resolve_source_file(source_root: str | os.PathLike[str] | None, variant: str) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if source_root:
        candidates.append(source_root)
    env_root = os.environ.get("UMAMBA_SOURCE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(_DEFAULT_SOURCE_ROOTS)

    filename = "UMambaEnc_3d.py" if variant == "enc" else "UMambaBot_3d.py"
    for candidate in candidates:
        root = Path(candidate).expanduser()
        source_candidates = (
            root / "umamba" / "nnunetv2" / "nets" / filename,
            root / "nnunetv2" / "nets" / filename,
        )
        for source_file in source_candidates:
            if source_file.is_file():
                return source_file

    checked = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(f"{_MISSING_SOURCE_MESSAGE} Checked: {checked}")


class _InitWeightsHe:
    """Import shim matching the initializer used by upstream U-Mamba."""

    def __init__(self, neg_slope: float = 1e-2) -> None:
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def _install_nnunet_import_shims() -> None:
    """Provide only the nnU-Net symbols imported by the standalone network files."""
    try:
        importlib.import_module("nnunetv2.utilities.plans_handling.plans_handler")
        importlib.import_module("nnunetv2.utilities.network_initialization")
        return
    except ModuleNotFoundError:
        pass

    package_names = (
        "nnunetv2",
        "nnunetv2.utilities",
        "nnunetv2.utilities.plans_handling",
    )
    for package_name in package_names:
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = package

    plans_module_name = "nnunetv2.utilities.plans_handling.plans_handler"
    if plans_module_name not in sys.modules:
        plans_module = types.ModuleType(plans_module_name)
        setattr(plans_module, "ConfigurationManager", type("ConfigurationManager", (), {}))
        setattr(plans_module, "PlansManager", type("PlansManager", (), {}))
        sys.modules[plans_module_name] = plans_module

    initialization_module_name = "nnunetv2.utilities.network_initialization"
    if initialization_module_name not in sys.modules:
        initialization_module = types.ModuleType(initialization_module_name)
        setattr(initialization_module, "InitWeights_He", _InitWeightsHe)
        sys.modules[initialization_module_name] = initialization_module


def _load_upstream_network_class(source_file: Path, variant: str) -> type[nn.Module]:
    _install_nnunet_import_shims()
    module_name = f"_dgcmfnet_upstream_umamba_{variant}_3d"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load U-Mamba module from {source_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm" or (exc.name and exc.name.startswith("causal_conv1d")):
            raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc
        raise

    class_name = "UMambaEnc" if variant == "enc" else "UMambaBot"
    network_class = getattr(module, class_name, None)
    if network_class is None:
        raise ImportError(f"Upstream module {source_file} does not define {class_name}")
    return cast(type[nn.Module], network_class)


class UMambaAdapter(nn.Module):
    """Wrap the original U-Mamba 3D Bot or Enc network.

    Input batches must contain ``signal`` with shape ``[B, C, D, H, W]``.
    The adapter returns the full-resolution segmentation logits tensor expected
    by the framework.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        variant: str = "enc",
        source_root: str | None = None,
        img_size: int | Sequence[int] = (128, 128, 128),
        num_stages: int = 6,
        features_per_stage: Sequence[int] | None = None,
        kernel_sizes: Sequence[int | Sequence[int]] | None = None,
        strides: Sequence[int | Sequence[int]] | None = None,
        n_conv_per_stage: Sequence[int] | None = None,
        n_conv_per_stage_decoder: Sequence[int] | None = None,
        conv_bias: bool = True,
        deep_supervision: bool = False,
        strict_img_size: bool = True,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        normalized_variant = variant.strip().lower().replace("-", "_")
        variant_aliases = {
            "enc": "enc",
            "encoder": "enc",
            "umamba_enc": "enc",
            "bot": "bot",
            "bottleneck": "bot",
            "umamba_bot": "bot",
        }
        if normalized_variant not in variant_aliases:
            raise ValueError(f"variant must be 'enc' or 'bot', got {variant!r}")

        self.variant = variant_aliases[normalized_variant]
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = _as_3d_tuple(img_size, "img_size")
        self.strict_img_size = bool(strict_img_size)
        self.deep_supervision = bool(deep_supervision)

        if num_stages < 2:
            raise ValueError(f"num_stages must be at least 2, got {num_stages}")

        resolved_features = list(features_per_stage or [min(320, 32 * (2**stage)) for stage in range(num_stages)])
        resolved_kernel_sizes = _as_3d_tuple_list(kernel_sizes or [(3, 3, 3)] * num_stages, "kernel_sizes")
        resolved_strides = _as_3d_tuple_list(
            strides or [(1, 1, 1)] + [(2, 2, 2)] * (num_stages - 1),
            "strides",
        )
        resolved_encoder_blocks = list(n_conv_per_stage or [2] * num_stages)
        resolved_decoder_blocks = list(n_conv_per_stage_decoder or [2] * (num_stages - 1))

        expected_lengths = {
            "features_per_stage": (len(resolved_features), num_stages),
            "kernel_sizes": (len(resolved_kernel_sizes), num_stages),
            "strides": (len(resolved_strides), num_stages),
            "n_conv_per_stage": (len(resolved_encoder_blocks), num_stages),
            "n_conv_per_stage_decoder": (len(resolved_decoder_blocks), num_stages - 1),
        }
        for name, (actual, expected) in expected_lengths.items():
            if actual != expected:
                raise ValueError(f"{name} must contain {expected} values, got {actual}")
        if any(value < 1 for value in resolved_features):
            raise ValueError("features_per_stage values must be positive")
        if any(value < 1 for value in resolved_encoder_blocks + resolved_decoder_blocks):
            raise ValueError("encoder and decoder block counts must be positive")

        downsample_factor = tuple(math.prod(stride[axis] for stride in resolved_strides) for axis in range(3))
        if any(size % factor != 0 for size, factor in zip(self.img_size, downsample_factor)):
            raise ValueError(
                f"img_size={self.img_size} must be divisible by the total stride {downsample_factor}"
            )

        source_file = _resolve_source_file(source_root, self.variant)
        network_class = _load_upstream_network_class(source_file, self.variant)

        network_kwargs: dict[str, Any] = {
            "input_channels": self.num_modalities,
            "n_stages": int(num_stages),
            "features_per_stage": [int(value) for value in resolved_features],
            "conv_op": nn.Conv3d,
            "kernel_sizes": resolved_kernel_sizes,
            "strides": resolved_strides,
            "n_conv_per_stage": [int(value) for value in resolved_encoder_blocks],
            "num_classes": self.num_classes,
            "n_conv_per_stage_decoder": [int(value) for value in resolved_decoder_blocks],
            "conv_bias": bool(conv_bias),
            "norm_op": nn.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
            "deep_supervision": self.deep_supervision,
        }
        if self.variant == "enc":
            network_kwargs["input_size"] = self.img_size

        self.model = network_class(**network_kwargs)
        if initialize:
            self.model.apply(_InitWeightsHe(1e-2))

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"UMambaAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        spatial_shape = tuple(int(value) for value in x.shape[2:])
        if self.strict_img_size and spatial_shape != self.img_size:
            raise ValueError(f"UMambaAdapter expects spatial shape {self.img_size}, got {spatial_shape}")
        if self.variant == "enc" and spatial_shape != self.img_size:
            raise ValueError(
                "U-Mamba Enc derives channel-token dimensions from img_size at construction time; "
                f"expected {self.img_size}, got {spatial_shape}"
            )

        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"UMambaAdapter expected tensor logits, got {type(logits)}")
        return logits
