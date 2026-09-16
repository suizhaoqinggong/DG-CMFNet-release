"""Adapter for the original SegMamba network.

The upstream source tree is kept outside this repository.  SegMamba ships a
small fork of ``mamba_ssm`` with the BiMamba arguments used by the paper, so
the adapter places that bundled package ahead of the regular installation
before importing the network.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ()

_MISSING_SOURCE_MESSAGE = (
    "SegMambaAdapter requires the upstream ge-xing/SegMamba source tree. "
    "Set source_root in the model config or export SEGMAMBA_SOURCE_ROOT."
)

_MISSING_DEPENDENCY_MESSAGE = (
    "SegMambaAdapter requires MONAI plus compiled mamba-ssm and causal-conv1d extensions. "
    "Install them in this project with: uv sync --extra segmamba"
)

_STAGE_SLICES = (64, 32, 16, 8)


class _CausalConv1dCompat:
    """Translate SegMamba's old extension calls to causal-conv1d 1.4+."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def causal_conv1d_fwd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: bool,
    ) -> torch.Tensor:
        try:
            return self.backend.causal_conv1d_fwd(x, weight, bias, None, None, None, activation)
        except TypeError:
            return self.backend.causal_conv1d_fwd(x, weight, bias, activation)

    def causal_conv1d_bwd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        dout: torch.Tensor,
        dx: torch.Tensor | None,
        activation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        try:
            result = self.backend.causal_conv1d_bwd(
                x,
                weight,
                bias,
                dout,
                None,
                None,
                None,
                dx,
                False,
                activation,
            )
            return result[0], result[1], result[2]
        except TypeError:
            return self.backend.causal_conv1d_bwd(x, weight, bias, dout, dx, activation)


def _as_int_tuple(values: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} integers, got {result}")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _validate_spatial_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    spatial_shape = cast(tuple[int, int, int], _as_int_tuple(shape, "spatial shape", 3))
    if any(size % 16 != 0 for size in spatial_shape):
        raise ValueError(f"Input spatial dimensions must be divisible by 16, got {spatial_shape}")

    for stage, num_slices in enumerate(_STAGE_SLICES, start=1):
        token_count = 1
        for size in spatial_shape:
            token_count *= size // (2**stage)
        if token_count % num_slices != 0:
            raise ValueError(
                f"Input shape {spatial_shape} produces {token_count} tokens at encoder stage {stage}; "
                f"the original SegMamba BiMamba block requires a multiple of {num_slices}"
            )
    return spatial_shape


def _resolve_source_file(source_root: str | os.PathLike[str] | None) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if source_root:
        candidates.append(source_root)
    env_root = os.environ.get("SEGMAMBA_SOURCE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(_DEFAULT_SOURCE_ROOTS)

    for candidate in candidates:
        root = Path(candidate).expanduser()
        source_candidates = (
            root / "model_segmamba" / "segmamba.py",
            root / "segmamba.py",
        )
        for source_file in source_candidates:
            if source_file.is_file():
                return source_file

    checked = ", ".join(str(Path(candidate).expanduser()) for candidate in candidates)
    raise FileNotFoundError(f"{_MISSING_SOURCE_MESSAGE} Checked: {checked}")


def _activate_upstream_mamba_fork(source_file: Path) -> None:
    source_root = source_file.parent.parent
    package_roots = (source_root / "mamba", source_root / "causal-conv1d")
    missing = [str(path) for path in package_roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"SegMamba upstream dependency directories are missing: {', '.join(missing)}")

    for package_root in reversed(package_roots):
        path = str(package_root)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    # The regular Mamba package does not accept SegMamba's bimamba_type and
    # nslices arguments.  Replace it with the fork bundled by upstream.
    expected_root = (source_root / "mamba").resolve()
    loaded_mamba = sys.modules.get("mamba_ssm")
    loaded_file = getattr(loaded_mamba, "__file__", None)
    if loaded_file is not None and not Path(loaded_file).resolve().is_relative_to(expected_root):
        for module_name in tuple(sys.modules):
            if module_name == "mamba_ssm" or module_name.startswith("mamba_ssm."):
                del sys.modules[module_name]
    importlib.invalidate_caches()


def _load_upstream_network_class(source_file: Path) -> type[nn.Module]:
    _activate_upstream_mamba_fork(source_file)
    module_name = "_dgcmfnet_upstream_segmamba"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SegMamba module from {source_file}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        missing_root = (getattr(exc, "name", None) or "").split(".", 1)[0]
        if missing_root in {"monai", "mamba_ssm", "causal_conv1d", "selective_scan_cuda", "causal_conv1d_cuda"}:
            raise ModuleNotFoundError(_MISSING_DEPENDENCY_MESSAGE) from exc
        raise

    selective_scan = sys.modules.get("mamba_ssm.ops.selective_scan_interface")
    causal_backend = getattr(selective_scan, "causal_conv1d_cuda", None)
    if selective_scan is not None and causal_backend is not None and not isinstance(causal_backend, _CausalConv1dCompat):
        selective_scan.causal_conv1d_cuda = _CausalConv1dCompat(causal_backend)

    network_class = getattr(module, "SegMamba", None)
    if network_class is None:
        raise ImportError(f"Upstream module {source_file} does not define SegMamba")
    return cast(type[nn.Module], network_class)


class SegMambaAdapter(nn.Module):
    """Expose the original SegMamba network through the framework contract."""

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
        self.img_size = _validate_spatial_shape(img_size)
        self.depths = _as_int_tuple(depths, "depths", 4)
        self.feat_size = _as_int_tuple(feat_size, "feat_size", 4)
        self.strict_img_size = bool(strict_img_size)

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if self.num_modalities < 1:
            raise ValueError(f"num_modalities must be positive, got {self.num_modalities}")
        if self.feat_size[0] != 48:
            raise ValueError(
                "The upstream SegMamba output block is fixed to 48 input channels; "
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
            raise ValueError(f"SegMambaAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        spatial_shape = _validate_spatial_shape(x.shape[2:])
        if self.strict_img_size and spatial_shape != self.img_size:
            raise ValueError(f"SegMambaAdapter expects spatial shape {self.img_size}, got {spatial_shape}")

        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"SegMambaAdapter expected tensor logits, got {type(logits)}")
        return logits
