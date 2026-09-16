"""3D Attention U-Net baseline for volumetric segmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput


def _normalize_channel_multipliers(
    channel_multipliers: Sequence[int] | None,
) -> tuple[int, ...]:
    values = (1, 2, 4, 8, 16) if channel_multipliers is None else channel_multipliers
    multipliers = tuple(int(v) for v in values)
    if len(multipliers) < 2:
        raise ValueError("channel_multipliers must contain at least two levels")
    if any(v <= 0 for v in multipliers):
        raise ValueError(f"channel_multipliers must be positive, got {multipliers}")
    return multipliers


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    key = norm.strip().lower()
    if key in {"batch", "batchnorm", "batch_norm"}:
        return nn.BatchNorm3d(num_channels)
    if key in {"instance", "instancenorm", "instance_norm"}:
        return nn.InstanceNorm3d(num_channels, affine=True)
    if key in {"group", "groupnorm", "group_norm"}:
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    if key in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(
        f"Unknown normalization '{norm}'. Expected batch_norm, instance_norm, group_norm, or none."
    )


class ConvBlock3d(nn.Module):
    """Two 3D convolutions with normalization and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate3d(nn.Module):
    """Attention gate that filters encoder skips using decoder context."""

    def __init__(
        self,
        gating_channels: int,
        skip_channels: int,
        intermediate_channels: int,
        norm: str,
    ) -> None:
        super().__init__()
        self.gating_proj = nn.Sequential(
            nn.Conv3d(gating_channels, intermediate_channels, kernel_size=1, bias=False),
            _make_norm(norm, intermediate_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv3d(skip_channels, intermediate_channels, kernel_size=1, bias=False),
            _make_norm(norm, intermediate_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gating.shape[2:] != skip.shape[2:]:
            gating = F.interpolate(
                gating,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        attention = self.psi(F.relu(self.gating_proj(gating) + self.skip_proj(skip), inplace=True))
        return skip * attention


class AttentionUNet(nn.Module):
    """3D Attention U-Net baseline using framework-native batch dictionaries.

    The model expects ``batch["signal"]`` shaped ``[B, num_modalities, D, H, W]``
    and returns dense logits shaped ``[B, num_classes, D, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: int = 16,
        channel_multipliers: Sequence[int] | None = None,
        norm: str = "instance_norm",
        dropout: float = 0.0,
        attention_intermediate_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if base_ch <= 0:
            raise ValueError(f"base_ch must be positive, got {base_ch}")
        if not 0 < attention_intermediate_ratio <= 1:
            raise ValueError(
                "attention_intermediate_ratio must be in the interval (0, 1], "
                f"got {attention_intermediate_ratio}"
            )

        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.base_ch = int(base_ch)
        self.norm = norm
        self.dropout = float(dropout)
        self.attention_intermediate_ratio = float(attention_intermediate_ratio)

        multipliers = _normalize_channel_multipliers(channel_multipliers)
        self.encoder_channels = [self.base_ch * multiplier for multiplier in multipliers]

        encoder_blocks = []
        in_channels = self.num_modalities
        for out_channels in self.encoder_channels:
            encoder_blocks.append(ConvBlock3d(in_channels, out_channels, norm=norm, dropout=dropout))
            in_channels = out_channels
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=2, stride=2) for _ in self.encoder_channels[:-1]])

        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for deep_channels, skip_channels in zip(
            reversed(self.encoder_channels[1:]),
            reversed(self.encoder_channels[:-1]),
        ):
            self.upconvs.append(
                nn.ConvTranspose3d(deep_channels, skip_channels, kernel_size=2, stride=2)
            )
            intermediate_channels = max(1, int(skip_channels * self.attention_intermediate_ratio))
            self.attention_gates.append(
                AttentionGate3d(
                    gating_channels=skip_channels,
                    skip_channels=skip_channels,
                    intermediate_channels=intermediate_channels,
                    norm=norm,
                )
            )
            self.decoder_blocks.append(
                ConvBlock3d(skip_channels * 2, skip_channels, norm=norm, dropout=dropout)
            )

        self.out_conv = nn.Conv3d(self.encoder_channels[0], self.num_classes, kernel_size=1)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"AttentionUNet expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        input_spatial_shape = x.shape[2:]
        skips: list[torch.Tensor] = []
        for idx, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            skips.append(x)
            if idx < len(self.pools):
                x = self.pools[idx](x)

        for idx, (upconv, gate, decoder_block) in enumerate(
            zip(self.upconvs, self.attention_gates, self.decoder_blocks)
        ):
            x = upconv(x)
            skip = skips[-(idx + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            gated_skip = gate(x, skip)
            x = decoder_block(torch.cat([gated_skip, x], dim=1))

        logits = self.out_conv(x)
        if logits.shape[2:] != input_spatial_shape:
            logits = F.interpolate(
                logits,
                size=input_spatial_shape,
                mode="trilinear",
                align_corners=False,
            )
        return logits
