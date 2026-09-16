"""MONAI Swin UNETR adapter for the DG-CMFNet framework.

The adapter keeps data loading, losses, metrics, checkpoints, and logging in
this repository while using MONAI's SwinUNETR network implementation.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput

_MISSING_DEPENDENCY_MESSAGE = (
    "SwinUNETRAdapter requires MONAI. Install it in this Python 3.9 project with: "
    "uv sync --extra swinunetr"
)


def _as_3d_tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {result}")
    return result


def _as_4_list(values: Sequence[int] | None, default: Sequence[int], name: str) -> list[int]:
    result = list(default if values is None else values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain exactly four integers, got {len(result)}")
    return [int(v) for v in result]


def _load_swin_unetr_class() -> type[nn.Module]:
    try:
        monai_nets = importlib.import_module("monai.networks.nets")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc
    return getattr(monai_nets, "SwinUNETR")


def _filter_init_kwargs(model_cls: type[nn.Module], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(model_cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    changed = True
        state_dict[normalized_key] = value
    return state_dict


class SwinUNETRAdapter(nn.Module):
    """Wrap MONAI SwinUNETR so it can be trained by this framework.

    The adapter expects ``batch["signal"]`` shaped
    ``[B, num_modalities, D, H, W]`` and returns logits shaped
    ``[B, num_classes, D, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        img_size: int | Sequence[int] = (128, 128, 128),
        feature_size: int = 48,
        depths: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        patch_size: int | Sequence[int] = 2,
        window_size: int | Sequence[int] = 7,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        strict_img_size: bool = True,
        pretrained_path: str | None = None,
        strict_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = _as_3d_tuple(img_size, "img_size")
        self.patch_size = _as_3d_tuple(patch_size, "patch_size")
        self.window_size = _as_3d_tuple(window_size, "window_size")
        self.strict_img_size = bool(strict_img_size)

        model_cls = _load_swin_unetr_class()
        model_kwargs: dict[str, Any] = {
            "img_size": self.img_size,
            "in_channels": self.num_modalities,
            "out_channels": self.num_classes,
            "feature_size": int(feature_size),
            "depths": _as_4_list(depths, [2, 2, 2, 2], "depths"),
            "num_heads": _as_4_list(num_heads, [3, 6, 12, 24], "num_heads"),
            "patch_size": self.patch_size,
            "window_size": self.window_size,
            "norm_name": norm_name,
            "drop_rate": float(drop_rate),
            "attn_drop_rate": float(attn_drop_rate),
            "dropout_path_rate": float(dropout_path_rate),
            "normalize": bool(normalize),
            "use_checkpoint": bool(use_checkpoint),
            "spatial_dims": int(spatial_dims),
            "downsample": downsample,
            "use_v2": bool(use_v2),
            "qkv_bias": bool(qkv_bias),
            "mlp_ratio": float(mlp_ratio),
        }
        self.model = model_cls(**_filter_init_kwargs(model_cls, model_kwargs))

        if pretrained_path is not None and str(pretrained_path).strip().lower() not in {"", "none", "null"}:
            self._load_from_pretrain(str(pretrained_path), strict=bool(strict_pretrained))

    def _load_from_pretrain(self, pretrained_path: str, strict: bool) -> None:
        path = Path(pretrained_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SwinUNETR pretrain checkpoint does not exist: {path}")

        state_dict = _extract_state_dict(torch.load(path, map_location="cpu"))
        if strict:
            self.model.load_state_dict(state_dict, strict=True)
            return

        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        if not compatible_state:
            raise ValueError(f"No compatible SwinUNETR weights found in checkpoint: {path}")
        self.model.load_state_dict(compatible_state, strict=False)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SwinUNETRAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        input_spatial_shape = tuple(int(v) for v in x.shape[2:])
        if self.strict_img_size and input_spatial_shape != self.img_size:
            raise ValueError(f"SwinUNETRAdapter expects spatial shape {self.img_size}, got {input_spatial_shape}")

        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"SwinUNETRAdapter expected tensor logits, got {type(logits)}")
        if tuple(int(v) for v in logits.shape[2:]) != input_spatial_shape:
            logits = F.interpolate(logits, size=input_spatial_shape, mode="trilinear", align_corners=False)
        return logits
