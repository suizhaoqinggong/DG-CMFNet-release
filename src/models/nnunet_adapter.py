"""nnU-Net architecture adapter for the DG-CMFNet framework.

This adapter uses the network implementations from the nnU-Net v2 dependency
stack, but keeps data loading, loss computation, metrics, checkpointing, and
training inside this repository's framework.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput


_MISSING_DEPENDENCY_MESSAGE = (
    "NnUNetAdapter requires nnU-Net v2's network dependency stack. "
    'Install it in this Python 3.9 project with: uv sync --extra nnunet'
)


def _resolve_external_class(qualified_name: str) -> type[Any]:
    module_name, _, class_name = qualified_name.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a fully qualified class name, got {qualified_name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_architecture(architecture: str) -> tuple[type[Any], bool]:
    """Resolve an architecture name to a dynamic-network-architectures class."""
    try:
        unet_module = importlib.import_module("dynamic_network_architectures.architectures.unet")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc

    aliases = {
        "plain_conv_unet": ("PlainConvUNet", False),
        "plainconvunet": ("PlainConvUNet", False),
        "plain": ("PlainConvUNet", False),
        "residual_encoder_unet": ("ResidualEncoderUNet", True),
        "residualencoderunet": ("ResidualEncoderUNet", True),
        "resenc_unet": ("ResidualEncoderUNet", True),
    }
    key = architecture.lower()
    if key in aliases:
        class_name, is_residual = aliases[key]
        return getattr(unet_module, class_name), is_residual

    architecture_class = _resolve_external_class(architecture)
    return architecture_class, "ResidualEncoderUNet" in architecture_class.__name__


def _as_3d_tuple(value: int | list[int] | tuple[int, ...], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {result}")
    return result


def _as_3d_tuple_list(
    values: list[int] | list[list[int]] | list[tuple[int, ...]] | tuple[Any, ...],
    name: str,
) -> list[tuple[int, int, int]]:
    return [_as_3d_tuple(value, f"{name}[{idx}]") for idx, value in enumerate(values)]


def _resolve_op(name: str | None, op_map: dict[str, type[Any]], default: type[Any] | None) -> type[Any] | None:
    if name is None or name.lower() in {"none", "null"}:
        return default
    key = name.lower()
    if key in op_map:
        return op_map[key]
    return _resolve_external_class(name)


class NnUNetAdapter(nn.Module):
    """Wrap a nnU-Net v2 network so it can be trained by this framework.

    The adapter expects batches with ``batch["signal"]`` shaped
    ``[B, num_modalities, D, H, W]`` and returns logits shaped
    ``[B, num_classes, D, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        architecture: str = "plain_conv_unet",
        num_stages: int = 6,
        base_num_features: int = 32,
        max_num_features: int = 320,
        features_per_stage: list[int] | None = None,
        kernel_sizes: list[list[int]] | list[tuple[int, ...]] | None = None,
        strides: list[list[int]] | list[tuple[int, ...]] | None = None,
        n_conv_per_stage: list[int] | None = None,
        n_blocks_per_stage: list[int] | None = None,
        n_conv_per_stage_decoder: list[int] | None = None,
        conv_bias: bool = True,
        norm_op: str = "instance_norm",
        norm_op_kwargs: dict[str, Any] | None = None,
        dropout_op: str | None = None,
        dropout_op_kwargs: dict[str, Any] | None = None,
        nonlin: str = "leaky_relu",
        nonlin_kwargs: dict[str, Any] | None = None,
        deep_supervision: bool = False,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.deep_supervision = deep_supervision

        architecture_class, is_residual = _resolve_architecture(architecture)

        resolved_features = (
            [int(v) for v in features_per_stage]
            if features_per_stage is not None
            else [min(max_num_features, base_num_features * (2**stage)) for stage in range(num_stages)]
        )
        if len(resolved_features) != num_stages:
            raise ValueError(
                f"features_per_stage must contain {num_stages} values, got {len(resolved_features)}"
            )

        resolved_kernel_sizes = (
            _as_3d_tuple_list(kernel_sizes, "kernel_sizes")
            if kernel_sizes is not None
            else [(3, 3, 3)] * num_stages
        )
        resolved_strides = (
            _as_3d_tuple_list(strides, "strides")
            if strides is not None
            else [(1, 1, 1)] + [(2, 2, 2)] * (num_stages - 1)
        )
        if len(resolved_kernel_sizes) != num_stages:
            raise ValueError(f"kernel_sizes must contain {num_stages} values, got {len(resolved_kernel_sizes)}")
        if len(resolved_strides) != num_stages:
            raise ValueError(f"strides must contain {num_stages} values, got {len(resolved_strides)}")

        resolved_n_conv_per_stage = n_conv_per_stage or [2] * num_stages
        resolved_n_blocks_per_stage = n_blocks_per_stage or resolved_n_conv_per_stage
        resolved_n_conv_per_stage_decoder = n_conv_per_stage_decoder or [2] * (num_stages - 1)
        if len(resolved_n_conv_per_stage_decoder) != num_stages - 1:
            raise ValueError(
                f"n_conv_per_stage_decoder must contain {num_stages - 1} values, "
                f"got {len(resolved_n_conv_per_stage_decoder)}"
            )

        norm_op_class = _resolve_op(
            norm_op,
            {
                "instance_norm": nn.InstanceNorm3d,
                "instancenorm": nn.InstanceNorm3d,
                "batch_norm": nn.BatchNorm3d,
                "batchnorm": nn.BatchNorm3d,
            },
            nn.InstanceNorm3d,
        )
        dropout_op_class = _resolve_op(
            dropout_op,
            {"dropout": nn.Dropout3d, "dropout3d": nn.Dropout3d},
            None,
        )
        nonlin_class = _resolve_op(
            nonlin,
            {"leaky_relu": nn.LeakyReLU, "leakyrelu": nn.LeakyReLU, "relu": nn.ReLU},
            nn.LeakyReLU,
        )

        network_kwargs: dict[str, Any] = {
            "input_channels": num_modalities,
            "n_stages": num_stages,
            "features_per_stage": resolved_features,
            "conv_op": nn.Conv3d,
            "kernel_sizes": resolved_kernel_sizes,
            "strides": resolved_strides,
            "num_classes": num_classes,
            "n_conv_per_stage_decoder": [int(v) for v in resolved_n_conv_per_stage_decoder],
            "conv_bias": conv_bias,
            "norm_op": norm_op_class,
            "norm_op_kwargs": norm_op_kwargs or {"eps": 1e-5, "affine": True},
            "dropout_op": dropout_op_class,
            "dropout_op_kwargs": dropout_op_kwargs,
            "nonlin": nonlin_class,
            "nonlin_kwargs": nonlin_kwargs or {"inplace": True},
            "deep_supervision": deep_supervision,
        }
        if is_residual:
            network_kwargs["n_blocks_per_stage"] = [int(v) for v in resolved_n_blocks_per_stage]
        else:
            network_kwargs["n_conv_per_stage"] = [int(v) for v in resolved_n_conv_per_stage]

        self.model = architecture_class(**network_kwargs)
        if initialize and hasattr(self.model, "initialize"):
            self.model.apply(self.model.initialize)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"NnUNetAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"NnUNetAdapter expected tensor logits, got {type(logits)}")
        return logits
