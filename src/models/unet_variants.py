"""Framework-native comparison models for U-Net family papers.

The classes in this module keep DG-CMFNet's batch contract:
``batch["signal"]`` is ``[B, C, D, H, W]`` and the model returns dense logits.
They are intentionally compact so they can be trained by the existing framework
without depending on each paper's standalone trainer.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput


def _append_package_path(package_name: str, path: str | os.PathLike[str]) -> None:
    package = sys.modules.get(package_name)
    if package is None:
        package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ImportError(f"Package {package_name!r} has no __path__ and cannot be extended")
    resolved = str(Path(path).expanduser())
    if resolved not in list(package.__path__):
        package.__path__.append(resolved)


def _install_ml_collections_shim() -> None:
    """Provide the small ConfigDict subset used by NestedFormer's source."""
    try:
        importlib.import_module("ml_collections")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("ml_collections")

    class ConfigDict(dict):
        def __getattr__(self, key: str) -> object:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: object) -> None:
            self[key] = value

    module.ConfigDict = ConfigDict
    sys.modules["ml_collections"] = module


def _as_tuple(values: Sequence[int] | None, default: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(v) for v in (default if values is None else values))
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two levels, got {result}")
    return result


def _make_norm(norm: str, channels: int) -> nn.Module:
    key = norm.lower().strip()
    if key in {"instance", "instance_norm", "instancenorm"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if key in {"batch", "batch_norm", "batchnorm"}:
        return nn.BatchNorm3d(channels)
    if key in {"group", "group_norm", "groupnorm"}:
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if key in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown normalization: {norm}")


class BasicConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, dropout: float = 0.0) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, dropout: float = 0.0) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(norm, out_channels),
        )
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, bias=False), _make_norm(norm, out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + self.proj(x), inplace=True)


class DenseConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        dropout: float = 0.0,
        growth_rate: int | None = None,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        growth = int(growth_rate or max(4, out_channels // 4))
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(int(num_layers)):
            self.layers.append(
                nn.Sequential(
                    _make_norm(norm, current_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(current_channels, growth, 3, padding=1, bias=False),
                    nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            current_channels += growth
        self.compress = nn.Sequential(
            _make_norm(norm, current_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(current_channels, out_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.compress(torch.cat(features, dim=1))


class UNet3D(nn.Module):
    """3D U-Net with configurable convolution block type."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: int = 16,
        channel_multipliers: Sequence[int] | None = None,
        norm: str = "instance_norm",
        dropout: float = 0.0,
        block_type: str = "basic",
        dense_growth_rate: int | None = None,
        dense_layers: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.base_ch = int(base_ch)
        multipliers = _as_tuple(channel_multipliers, (1, 2, 4, 8, 16), "channel_multipliers")
        channels = [self.base_ch * value for value in multipliers]

        block_key = block_type.lower().strip()

        def make_block(in_ch: int, out_ch: int) -> nn.Module:
            if block_key == "basic":
                return BasicConvBlock3d(in_ch, out_ch, norm, dropout)
            if block_key == "residual":
                return ResidualConvBlock3d(in_ch, out_ch, norm, dropout)
            if block_key == "dense":
                return DenseConvBlock3d(in_ch, out_ch, norm, dropout, dense_growth_rate, dense_layers)
            raise ValueError(f"Unknown U-Net block_type: {block_type}")

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = self.num_modalities
        for out_channels in channels:
            self.encoder_blocks.append(make_block(in_channels, out_channels))
            in_channels = out_channels
        for _ in channels[:-1]:
            self.pools.append(nn.MaxPool3d(2, 2))

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for deep_channels, skip_channels in zip(reversed(channels[1:]), reversed(channels[:-1])):
            self.upconvs.append(nn.ConvTranspose3d(deep_channels, skip_channels, 2, stride=2))
            self.decoder_blocks.append(make_block(skip_channels * 2, skip_channels))

        self.out_conv = nn.Conv3d(channels[0], self.num_classes, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            if index < len(self.pools):
                x = self.pools[index](x)

        for index, (upconv, block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skips[-(index + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat([skip, x], dim=1))

        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode="trilinear", align_corners=False)
        return x

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"{self.__class__.__name__} expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        return self.out_conv(self.forward_features(x))


class ResUNet3D(UNet3D):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("block_type", "residual")
        super().__init__(*args, **kwargs)


class DenseUNet3D(UNet3D):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("block_type", "dense")
        super().__init__(*args, **kwargs)


class PaperUNetConvBlock3D(nn.Module):
    """The original U-Net convolutional block, lifted from 2D to 3D.

    Every convolution uses ``padding=1`` so a 128^3 BraTS patch can be trained
    end-to-end. Apart from this deliberate same-padding adaptation, the block
    follows Ronneberger et al.: two 3x3 convolutions, each followed by ReLU,
    without normalisation layers.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PaperUNet3D(nn.Module):
    """Paper-faithful U-Net architecture adapted only from 2D to 3D.

    The architecture preserves the paper's five feature levels
    (64, 128, 256, 512, 1024), four 2x down/up-sampling operations, two
    convolutions per level, concatenative skip connections, and final 1x1
    classifier. The sole structural departure is same padding on the 3x3x3
    convolutions; the paper uses valid convolutions, which cannot operate on
    the framework's fixed 128^3 crops without shrinking logits and targets.
    """

    encoder_channels = (64, 128, 256, 512, 1024)

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        bottleneck_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(bottleneck_dropout) < 1.0:
            raise ValueError(f"bottleneck_dropout must be in [0, 1), got {bottleneck_dropout}")

        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.bottleneck_dropout_rate = float(bottleneck_dropout)

        self.encoder_blocks = nn.ModuleList()
        in_channels = self.num_modalities
        for out_channels in self.encoder_channels:
            self.encoder_blocks.append(PaperUNetConvBlock3D(in_channels, out_channels))
            in_channels = out_channels
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=2, stride=2) for _ in range(4)])
        self.bottleneck_dropout = (
            nn.Dropout3d(p=self.bottleneck_dropout_rate)
            if self.bottleneck_dropout_rate > 0.0
            else nn.Identity()
        )

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for deep_channels, skip_channels in zip(
            reversed(self.encoder_channels[1:]),
            reversed(self.encoder_channels[:-1]),
        ):
            self.upconvs.append(
                nn.ConvTranspose3d(deep_channels, skip_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(PaperUNetConvBlock3D(skip_channels * 2, skip_channels))

        self.out_conv = nn.Conv3d(self.encoder_channels[0], self.num_classes, kernel_size=1)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"PaperUNet3D expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if any(int(size) % 16 != 0 for size in x.shape[2:]):
            raise ValueError(
                "PaperUNet3D requires each spatial dimension to be divisible by 16 "
                f"for its four 2x pooling stages, got {tuple(x.shape[2:])}"
            )

        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.pools):
                skips.append(x)
                x = self.pools[index](x)

        x = self.bottleneck_dropout(x)
        for upconv, block, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            # With same padding and a 16-divisible crop, the paper's crop step
            # becomes an exact shape match.
            if x.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    "Unexpected U-Net skip shape mismatch: "
                    f"decoder={tuple(x.shape[2:])}, encoder={tuple(skip.shape[2:])}"
                )
            x = block(torch.cat([skip, x], dim=1))

        return self.out_conv(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, num_heads: int, num_layers: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width).contiguous()


class TransBTSAdapter(nn.Module):
    """Adapter for the released TransBTS implementation."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        img_size: int = 128,
        patch_size: int = 8,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 4096,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
        conv_patch_representation: bool = True,
        positional_encoding_type: str = "learned",
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        if int(num_modalities) != 4 or int(num_classes) != 4:
            raise ValueError("The released TransBTS BraTS model hard-codes 4 input modalities and 4 output channels")
        if int(embedding_dim) != 512:
            raise ValueError("The released TransBTS decoder contains BatchNorm3d(128), so embedding_dim must be 512")
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.img_size = int(img_size)
        self.strict_img_size = bool(strict_img_size)

        root = Path(source_root or os.environ.get("TRANSBTS_SOURCE_ROOT", "")).expanduser()
        model_file = root / "models" / "TransBTS" / "TransBTS_downsample8x_skipconnection.py"
        if not model_file.exists():
            raise FileNotFoundError(f"TransBTS source tree not found: {root}")
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        _append_package_path("models", root / "models")
        module = importlib.import_module("models.TransBTS.TransBTS_downsample8x_skipconnection")
        bts_cls = getattr(module, "BTS")
        self.model = bts_cls(
            img_dim=self.img_size,
            patch_dim=int(patch_size),
            num_channels=self.num_modalities,
            num_classes=self.num_classes,
            embedding_dim=int(embedding_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            dropout_rate=float(dropout_rate),
            attn_dropout_rate=float(attn_dropout_rate),
            conv_patch_representation=bool(conv_patch_representation),
            positional_encoding_type=positional_encoding_type,
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"TransBTSAdapter expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if self.strict_img_size and any(int(v) != self.img_size for v in x.shape[2:]):
            raise ValueError(f"TransBTSAdapter expects cubic spatial shape {self.img_size}, got {tuple(x.shape[2:])}")
        probs = self.model(x)
        if not isinstance(probs, torch.Tensor):
            raise TypeError(f"TransBTSAdapter expected tensor output, got {type(probs)}")
        return torch.log(torch.clamp(probs, min=1e-6))


class NestedFormerAdapter(nn.Module):
    """Adapter for the released NestedFormer implementation."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        image_size: Sequence[int] = (128, 128, 128),
        out_channels: int = 3,
        fea: Sequence[int] = (16, 16, 32, 64, 128, 16),
        window_size: Sequence[int] = (4, 4, 4),
        pool_size: Sequence[Sequence[int]] = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        self_num_layer: int = 2,
        token_mixer_size: int = 32,
        token_learner: bool = True,
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        if int(num_modalities) != 4:
            raise ValueError("The released NestedFormer BraTS setup expects 4 modalities")
        if int(out_channels) != 3:
            raise ValueError("The released NestedFormer setup outputs 3 BraTS region channels")
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.image_size = tuple(int(v) for v in image_size)
        if len(self.image_size) != 3:
            raise ValueError(f"image_size must contain three integers, got {self.image_size}")
        self.strict_img_size = bool(strict_img_size)

        root = Path(source_root or os.environ.get("NESTEDFORMER_SOURCE_ROOT", "")).expanduser()
        model_file = root / "medical" / "model" / "nested_former.py"
        if not model_file.exists():
            raise FileNotFoundError(f"NestedFormer source tree not found: {root}")
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        _install_ml_collections_shim()
        module = importlib.import_module("medical.model.nested_former")
        nestedformer_cls = getattr(module, "NestedFormer")
        self.model = nestedformer_cls(
            model_num=self.num_modalities,
            out_channels=int(out_channels),
            image_size=list(self.image_size),
            fea=tuple(int(v) for v in fea),
            window_size=tuple(int(v) for v in window_size),
            pool_size=tuple(tuple(int(v) for v in item) for item in pool_size),
            self_num_layer=int(self_num_layer),
            token_mixer_size=int(token_mixer_size),
            token_learner=bool(token_learner),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"NestedFormerAdapter expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if self.strict_img_size and tuple(int(v) for v in x.shape[2:]) != self.image_size:
            raise ValueError(f"NestedFormerAdapter expects spatial shape {self.image_size}, got {tuple(x.shape[2:])}")
        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"NestedFormerAdapter expected tensor logits, got {type(logits)}")
        return logits


class SlimUNETRAdapter(nn.Module):
    """Adapter for the released Slim UNETR implementation."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        source_root: str | None = None,
        embed_dim: int = 96,
        embedding_dim: int = 64,
        channels: Sequence[int] = (24, 48, 60),
        blocks: Sequence[int] = (1, 2, 3, 2),
        heads: Sequence[int] = (1, 2, 4, 4),
        r: Sequence[int] = (4, 2, 2, 1),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        root = Path(source_root or os.environ.get("SLIM_UNETR_SOURCE_ROOT", "")).expanduser()
        if not (root / "src" / "SlimUNETR" / "SlimUNETR.py").exists():
            raise FileNotFoundError(f"SlimUNETR source tree not found: {root}")
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        external_src = str(root / "src")
        existing_src_package = sys.modules.get("src")
        if existing_src_package is None:
            existing_src_package = importlib.import_module("src")
        if existing_src_package is not None and hasattr(existing_src_package, "__path__"):
            src_paths = list(existing_src_package.__path__)
            if external_src not in src_paths:
                existing_src_package.__path__.append(external_src)
        slim_module = importlib.import_module("src.SlimUNETR.SlimUNETR")
        slim_cls = getattr(slim_module, "SlimUNETR")
        self.model = slim_cls(
            in_channels=self.num_modalities,
            out_channels=self.num_classes,
            embed_dim=int(embed_dim),
            embedding_dim=int(embedding_dim),
            channels=tuple(int(v) for v in channels),
            blocks=tuple(int(v) for v in blocks),
            heads=tuple(int(v) for v in heads),
            r=tuple(int(v) for v in r),
            dropout=float(dropout),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SlimUNETRAdapter expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"SlimUNETRAdapter expected tensor logits, got {type(logits)}")
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits
