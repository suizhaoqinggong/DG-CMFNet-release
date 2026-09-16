"""VT-UNet adapter for the DG-CMFNet framework.

This wraps the external VT-UNet network implementation so this repository can
run it as a fair comparison model while keeping data loading, losses, metrics,
checkpointing, and logging inside the DG-CMFNet framework.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ()

_MISSING_SOURCE_MESSAGE = (
    "VTUNetAdapter requires the external VT-UNet source tree. "
    "Set source_root in the model config or export VTUNET_SOURCE_ROOT."
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


def _minimal_rearrange(x: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
    """Small subset of einops.rearrange used by VT-UNet's forward graph."""
    normalized = " ".join(pattern.split())
    if normalized in {"b c d h w -> b d h w c", "n c d h w -> n d h w c"}:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    if normalized in {"b d h w c -> b c d h w", "n d h w c -> n c d h w"}:
        return x.permute(0, 4, 1, 2, 3).contiguous()
    if normalized == "b d h w (p1 p2 c)-> b d (h p1) (w p2) c":
        batch, depth, height, width, _ = x.shape
        p1, p2, channels = int(axes["p1"]), int(axes["p2"]), int(axes["c"])
        return (
            x.reshape(batch, depth, height, width, p1, p2, channels)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch, depth, height * p1, width * p2, channels)
            .contiguous()
        )
    if normalized == "b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c":
        batch, depth, height, width, _ = x.shape
        p1, p2, p3 = int(axes["p1"]), int(axes["p2"]), int(axes["p3"])
        channels = int(axes["c"])
        return (
            x.reshape(batch, depth, height, width, p1, p2, p3, channels)
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(batch, depth * p1, height * p2, width * p3, channels)
            .contiguous()
        )
    raise NotImplementedError(f"VTUNetAdapter does not provide einops pattern: {pattern!r}")


def _install_optional_dependency_shims() -> None:
    """Install narrow shims for optional VT-UNet training/inference dependencies."""
    try:
        importlib.import_module("einops")
    except ModuleNotFoundError:
        einops_module = types.ModuleType("einops")
        einops_module.rearrange = _minimal_rearrange
        sys.modules["einops"] = einops_module

    try:
        importlib.import_module("mmcv.runner")
    except ModuleNotFoundError:
        mmcv_module = sys.modules.get("mmcv") or types.ModuleType("mmcv")
        runner_module = types.ModuleType("mmcv.runner")

        def load_checkpoint(model: nn.Module, filename: str, strict: bool = False, **kwargs: Any) -> Any:
            checkpoint = torch.load(filename, map_location=kwargs.get("map_location", "cpu"))
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            return model.load_state_dict(state_dict, strict=strict)

        runner_module.load_checkpoint = load_checkpoint
        mmcv_module.runner = runner_module
        sys.modules["mmcv"] = mmcv_module
        sys.modules["mmcv.runner"] = runner_module

    neural_module_name = "vtunet.network_architecture.neural_network"
    if neural_module_name not in sys.modules:
        neural_module = types.ModuleType(neural_module_name)

        class SegmentationNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inference_apply_nonlin = lambda x: x

        neural_module.SegmentationNetwork = SegmentationNetwork
        sys.modules[neural_module_name] = neural_module


def _resolve_source_root(source_root: str | os.PathLike[str] | None) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if source_root:
        candidates.append(source_root)
    env_root = os.environ.get("VTUNET_SOURCE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(_DEFAULT_SOURCE_ROOTS)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        if (root / "vtunet" / "network_architecture" / "vtunet_tumor.py").exists():
            return root

    checked = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(f"{_MISSING_SOURCE_MESSAGE} Checked: {checked}")


def _load_vtunet_module(source_root: str | os.PathLike[str] | None) -> Any:
    root = _resolve_source_root(source_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _install_optional_dependency_shims()
    return importlib.import_module("vtunet.network_architecture.vtunet_tumor")


class VTUNetAdapter(nn.Module):
    """Wrap VT-UNet so it can be trained through the DG-CMFNet framework."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        img_size: Sequence[int] = (128, 128, 128),
        patch_size: Sequence[int] = (4, 4, 4),
        embed_dim: int = 48,
        depths: Sequence[int] | None = None,
        depths_decoder: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        window_size: int | Sequence[int] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrain_ckpt: str | None = None,
        load_pretrain: bool = False,
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = _as_3d_tuple(img_size, "img_size")
        self.patch_size = _as_3d_tuple(patch_size, "patch_size")
        self.strict_img_size = bool(strict_img_size)

        if self.img_size != (128, 128, 128):
            raise ValueError(
                "The available VT-UNet implementation hard-codes decoder depth for 128^3 patches; "
                f"got img_size={self.img_size}."
            )

        vtunet_module = _load_vtunet_module(source_root)
        swin_cls = vtunet_module.SwinTransformerSys3D

        self.model = swin_cls(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.num_modalities,
            num_classes=self.num_classes,
            embed_dim=int(embed_dim),
            depths=_as_4_list(depths, [2, 2, 2, 1], "depths"),
            depths_decoder=_as_4_list(depths_decoder, [1, 2, 2, 2], "depths_decoder"),
            num_heads=_as_4_list(num_heads, [3, 6, 12, 24], "num_heads"),
            window_size=_as_3d_tuple(window_size, "window_size"),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            norm_layer=nn.LayerNorm,
            patch_norm=bool(patch_norm),
            use_checkpoint=bool(use_checkpoint),
            frozen_stages=-1,
            final_upsample="expand_first",
        )

        if load_pretrain:
            if pretrain_ckpt is None or str(pretrain_ckpt).strip().lower() in {"", "none", "null"}:
                raise ValueError("load_pretrain=true requires a valid pretrain_ckpt path")
            self._load_from_pretrain(str(pretrain_ckpt))

    def _load_from_pretrain(self, pretrain_ckpt: str) -> None:
        pretrained_path = Path(pretrain_ckpt).expanduser()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"VT-UNet pretrain checkpoint does not exist: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "model" not in checkpoint:
            state_dict = {k[17:]: v for k, v in checkpoint.items()}
            for key in list(state_dict.keys()):
                if "output" in key:
                    del state_dict[key]
            self.model.load_state_dict(state_dict, strict=False)
            return

        pretrained_dict = checkpoint["model"]
        model_dict = self.model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for key, value in pretrained_dict.items():
            if "layers." in key:
                current_layer_num = 3 - int(key[7:8])
                current_key = "layers_up." + str(current_layer_num) + key[8:]
                full_dict[current_key] = value

        for key in list(full_dict.keys()):
            if key in model_dict and full_dict[key].shape != model_dict[key].shape:
                del full_dict[key]
        self.model.load_state_dict(full_dict, strict=False)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"VTUNetAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if self.strict_img_size and tuple(int(v) for v in x.shape[2:]) != self.img_size:
            raise ValueError(f"VTUNetAdapter expects spatial shape {self.img_size}, got {tuple(x.shape[2:])}")

        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"VTUNetAdapter expected tensor logits, got {type(logits)}")
        return logits
