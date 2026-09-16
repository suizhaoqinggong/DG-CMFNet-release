"""Paper-derived SwinBTS baseline.

Jiang et al. (2022) define SwinBTS with 4^3 patch partitioning, convolutional
down/up-sampling, NFCE blocks, and an ETrans bottleneck. The repository named
in the article is no longer publicly available, therefore this implementation
follows the architectural and training details stated in that paper rather
than relabelling a MONAI SwinUNETR model as SwinBTS.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput

from .nnformer_adapter import SwinTransformerBlock, window_partition


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain three integers, got {result}")
    return result


def _four(values: Sequence[int] | None, default: Sequence[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(item) for item in (default if values is None else values))
    if len(result) != 4:
        raise ValueError(f"{name} must contain four integers, got {result}")
    return result  # type: ignore[return-value]


def _tokens_to_map(tokens: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    batch_size, token_count, channels = tokens.shape
    expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    if token_count != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} tokens, got {token_count}")
    return tokens.transpose(1, 2).reshape(batch_size, channels, *spatial_shape).contiguous()


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


class NFCEBlock3D(nn.Module):
    """Neighbor-feature connection enhancement: residual depthwise convolution."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(self.norm(x))
        return residual + self.dropout(x)


class ConvDownsample3D(nn.Module):
    """The paper's stride-two convolution followed by LayerNorm and GELU."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels * 2, kernel_size=2, stride=2, bias=False)
        self.norm = nn.LayerNorm(in_channels * 2)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        spatial_shape = tuple(int(size) for size in x.shape[2:])
        return _tokens_to_map(self.activation(self.norm(_map_to_tokens(x))), spatial_shape)


class ETransBlock3D(nn.Module):
    """Enhanced Transformer block from SwinBTS Eq. (2)--(3)."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.attention(self.query(x) * self.key(x)), dim=1)
        x = x + attention * self.value(x)
        normalized = _tokens_to_map(self.norm(_map_to_tokens(x)), tuple(int(size) for size in x.shape[2:]))
        return x + self.mlp(normalized)


class SwinStage3D(nn.Module):
    """Alternating regular/shifted 3D Swin blocks at one fixed resolution."""

    def __init__(
        self,
        channels: int,
        spatial_shape: tuple[int, int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.spatial_shape = spatial_shape
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=channels,
                    input_resolution=spatial_shape,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=attention_dropout,
                    drop_path=drop_path,
                )
                for index in range(depth)
            ]
        )

    @staticmethod
    def _attention_mask(
        spatial_shape: tuple[int, int, int], window_size: int, shift_size: int, device: torch.device
    ) -> torch.Tensor | None:
        if shift_size == 0:
            return None
        depth, height, width = spatial_shape
        padded_depth = (depth + window_size - 1) // window_size * window_size
        padded_height = (height + window_size - 1) // window_size * window_size
        padded_width = (width + window_size - 1) // window_size * window_size
        mask = torch.zeros((1, padded_depth, padded_height, padded_width, 1), device=device)
        slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        region = 0
        for depth_slice in slices:
            for height_slice in slices:
                for width_slice in slices:
                    mask[:, depth_slice, height_slice, width_slice, :] = region
                    region += 1
        windows = window_partition(mask, window_size).view(-1, window_size**3)
        attention_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(attention_mask == 0, 0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(
                tokens,
                self._attention_mask(
                    self.spatial_shape, int(block.window_size), int(block.shift_size), tokens.device
                ),
            )
        return tokens


class SwinBTSAdapter(nn.Module):
    """SwinBTS emitting direct, nested BraTS region logits in TC/WT/ET order."""

    region_order = ("tc", "wt", "et")

    def __init__(
        self,
        num_modalities: int = 4,
        num_classes: int = 4,
        out_channels: int = 3,
        img_size: int | Sequence[int] = (128, 128, 128),
        embed_dim: int = 96,
        depths: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        etrans_depth: int = 2,
        strict_img_size: bool = True,
    ) -> None:
        super().__init__()
        if int(out_channels) != 3:
            raise ValueError("SwinBTS outputs exactly three nested BraTS region logits: TC, WT, and ET")
        self.num_modalities = int(num_modalities)
        self.num_classes = int(num_classes)
        self.out_channels = int(out_channels)
        self.img_size = _triple(img_size, "img_size")
        self.strict_img_size = bool(strict_img_size)
        if any(size % 32 != 0 for size in self.img_size):
            raise ValueError(f"SwinBTS img_size must be divisible by 32, got {self.img_size}")

        self.depths = _four(depths, (2, 2, 2, 2), "depths")
        self.num_heads = _four(num_heads, (3, 6, 12, 24), "num_heads")
        channels = tuple(int(embed_dim) * 2**stage for stage in range(4))
        stage_shapes = tuple(tuple(size // 4 // 2**stage for size in self.img_size) for stage in range(4))

        self.patch_embed = nn.Sequential(nn.Conv3d(self.num_modalities, channels[0], kernel_size=4, stride=4, bias=False), nn.GELU())
        self.encoder_stages = nn.ModuleList(
            [
                SwinStage3D(channels[index], stage_shapes[index], self.depths[index], self.num_heads[index], int(window_size), mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate)
                for index in range(4)
            ]
        )
        self.encoder_nfce = nn.ModuleList([NFCEBlock3D(channels[index], drop_rate) for index in range(3)])
        self.downsamples = nn.ModuleList([ConvDownsample3D(channels[index]) for index in range(3)])
        self.etrans = nn.Sequential(*[ETransBlock3D(channels[-1], mlp_ratio, drop_rate) for _ in range(int(etrans_depth))])
        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose3d(channels[index], channels[index - 1], kernel_size=2, stride=2, bias=False) for index in (3, 2, 1)]
        )
        self.decoder_stages = nn.ModuleList(
            [
                SwinStage3D(channels[index], stage_shapes[index], self.depths[index], self.num_heads[index], int(window_size), mlp_ratio, drop_rate, attn_drop_rate, drop_path_rate)
                for index in (2, 1, 0)
            ]
        )
        self.decoder_nfce = nn.ModuleList([NFCEBlock3D(channels[index], drop_rate) for index in (2, 1, 0)])
        self.final_expand = nn.ConvTranspose3d(channels[0], self.out_channels, kernel_size=4, stride=4)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SwinBTSAdapter expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if self.strict_img_size and tuple(int(size) for size in x.shape[2:]) != self.img_size:
            raise ValueError(f"SwinBTSAdapter expects spatial shape {self.img_size}, got {tuple(x.shape[2:])}")

        feature_map = self.patch_embed(x)
        encoder_skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.encoder_stages):
            feature_map = _tokens_to_map(stage(_map_to_tokens(feature_map)), stage.spatial_shape)
            if index < len(self.downsamples):
                feature_map = self.encoder_nfce[index](feature_map)
                encoder_skips.append(feature_map)
                feature_map = self.downsamples[index](feature_map)

        feature_map = self.etrans(feature_map)
        for upsample, stage, nfce, skip in zip(self.upsamples, self.decoder_stages, self.decoder_nfce, reversed(encoder_skips)):
            feature_map = upsample(feature_map)
            if feature_map.shape[2:] != skip.shape[2:]:
                feature_map = F.interpolate(feature_map, size=skip.shape[2:], mode="trilinear", align_corners=False)
            feature_map = feature_map + skip
            feature_map = nfce(_tokens_to_map(stage(_map_to_tokens(feature_map)), stage.spatial_shape))

        logits = self.final_expand(feature_map)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits
