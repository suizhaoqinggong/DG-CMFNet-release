"""Ronneberger et al. (2015) U-Net adapted for slice-wise BraTS use.

The network body preserves the original 2D valid-convolution topology.  A
thin framework adapter mirror-pads small BraTS slices before the network and
center-crops the logits back to the label size.  Validation volumes are
processed slice-wise and reassembled for the existing 3D BraTS metrics.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput


class PaperUNetConvBlock2D(nn.Module):
    """Two unpadded 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_crop_2d(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    height, width = x.shape[-2:]
    if target_h > height or target_w > width:
        raise ValueError(f"Cannot crop {tuple(x.shape[-2:])} to {size}")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


class PaperUNet2D(nn.Module):
    """Original 2D U-Net body with BraTS-compatible I/O.

    Slice batches use ``[N, C, H, W]``. Whole validation volumes use
    ``[B, C, D, H, W]`` and return ``[B, K, D, H, W]`` logits.
    """

    paper_channels = (64, 128, 256, 512, 1024)

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        channels: Sequence[int] = paper_channels,
        bottleneck_dropout: float = 0.5,
        eval_slice_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"channels must contain five levels, got {channels}")
        if not 0.0 <= float(bottleneck_dropout) < 1.0:
            raise ValueError("bottleneck_dropout must be in [0, 1)")
        if eval_slice_chunk_size < 1:
            raise ValueError("eval_slice_chunk_size must be positive")

        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.channels = tuple(int(value) for value in channels)
        self.eval_slice_chunk_size = int(eval_slice_chunk_size)

        self.encoder_blocks = nn.ModuleList()
        in_channels = self.num_modalities
        for out_channels in self.channels:
            self.encoder_blocks.append(PaperUNetConvBlock2D(in_channels, out_channels))
            in_channels = out_channels
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])
        self.bottleneck_dropout = (
            nn.Dropout2d(p=float(bottleneck_dropout)) if bottleneck_dropout > 0.0 else nn.Identity()
        )

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for deep_channels, skip_channels in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            self.upconvs.append(nn.ConvTranspose2d(deep_channels, skip_channels, kernel_size=2, stride=2))
            self.decoder_blocks.append(PaperUNetConvBlock2D(skip_channels * 2, skip_channels))
        self.out_conv = nn.Conv2d(self.channels[0], self.num_classes, kernel_size=1)

    @staticmethod
    def paper_output_size(input_size: int) -> int:
        """Return the raw output size of the original valid-convolution U-Net."""
        size = int(input_size)
        for _ in range(4):
            size -= 4
            if size < 2:
                return -1
            size //= 2
        size -= 4
        if size < 1:
            return -1
        for _ in range(4):
            size = size * 2 - 4
        return size

    @classmethod
    def padded_input_size(cls, target_size: int) -> int:
        """Find the smallest valid input whose raw output covers target_size."""
        for candidate in range(int(target_size), int(target_size) + 512):
            if cls.paper_output_size(candidate) >= target_size:
                return candidate
        raise ValueError(f"Could not find a valid U-Net input size for {target_size}")

    @classmethod
    def _pad_context(cls, x: torch.Tensor) -> torch.Tensor:
        target_h, target_w = x.shape[-2:]
        padded_h = cls.padded_input_size(target_h)
        padded_w = cls.padded_input_size(target_w)
        pad_h = padded_h - target_h
        pad_w = padded_w - target_w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        max_horizontal = max(padding[0], padding[1])
        max_vertical = max(padding[2], padding[3])
        mode = "reflect" if max_horizontal < target_w and max_vertical < target_h else "replicate"
        return F.pad(x, padding, mode=mode)

    def _forward_paper_body(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.pools):
                skips.append(x)
                x = self.pools[index](x)

        x = self.bottleneck_dropout(x)
        for upconv, block, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            skip = _center_crop_2d(skip, x.shape[-2:])
            x = block(torch.cat([skip, x], dim=1))
        return self.out_conv(x)

    def forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        output_size = (int(x.shape[-2]), int(x.shape[-1]))
        logits = self._forward_paper_body(self._pad_context(x))
        return _center_crop_2d(logits, output_size)

    def _forward_volume(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = volume.shape
        if channels != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {channels}")
        slices = volume.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        logits = [self.forward_slices(chunk) for chunk in slices.split(self.eval_slice_chunk_size, dim=0)]
        stacked = torch.cat(logits, dim=0)
        return stacked.reshape(batch_size, depth, self.num_classes, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, batch: Batch) -> ModelOutput:
        signal = batch["signal"]
        if signal.ndim == 4:
            return self.forward_slices(signal)
        if signal.ndim == 5:
            return self._forward_volume(signal)
        raise ValueError(f"Expected 2D slices or 3D volumes, got {tuple(signal.shape)}")
