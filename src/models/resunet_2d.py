"""Slice-wise 2D Deep ResUNet for volumetric BraTS segmentation.

The encoder/decoder topology and residual blocks follow the ResUnet repository
by rishikksh20.  At validation and test, a 3D volume is predicted slice by
slice then reassembled into dense 3D logits for whole-volume BraTS metrics.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from framework.contracts.types import Batch, ModelOutput


class ReferenceResidualConv2D(nn.Module):
    """Residual block used by the referenced 2D ResUnet implementation."""

    def __init__(self, input_dim: int, output_dim: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.conv_skip(x)


class SliceWiseResUNet2D(nn.Module):
    """2D Deep ResUNet with framework-native multi-class volume support."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        filters: Sequence[int] = (32, 64, 128, 256),
        eval_slice_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if len(filters) != 4:
            raise ValueError(f"filters must contain four levels, got {filters}")
        if eval_slice_chunk_size < 1:
            raise ValueError("eval_slice_chunk_size must be positive")

        f0, f1, f2, f3 = (int(value) for value in filters)
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.eval_slice_chunk_size = int(eval_slice_chunk_size)

        self.input_layer = nn.Sequential(
            nn.Conv2d(self.num_modalities, f0, kernel_size=3, padding=1),
            nn.BatchNorm2d(f0),
            nn.ReLU(inplace=True),
            nn.Conv2d(f0, f0, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(self.num_modalities, f0, kernel_size=3, padding=1)
        self.residual_conv_1 = ReferenceResidualConv2D(f0, f1, stride=2, padding=1)
        self.residual_conv_2 = ReferenceResidualConv2D(f1, f2, stride=2, padding=1)
        self.bridge = ReferenceResidualConv2D(f2, f3, stride=2, padding=1)

        self.upsample_1 = nn.ConvTranspose2d(f3, f3, kernel_size=2, stride=2)
        self.up_residual_conv_1 = ReferenceResidualConv2D(f3 + f2, f2, stride=1, padding=1)
        self.upsample_2 = nn.ConvTranspose2d(f2, f2, kernel_size=2, stride=2)
        self.up_residual_conv_2 = ReferenceResidualConv2D(f2 + f1, f1, stride=1, padding=1)
        self.upsample_3 = nn.ConvTranspose2d(f1, f1, kernel_size=2, stride=2)
        self.up_residual_conv_3 = ReferenceResidualConv2D(f1 + f0, f0, stride=1, padding=1)
        # The source model is binary and ends in sigmoid. BraTS needs raw 4-class logits.
        self.output_layer = nn.Conv2d(f0, self.num_classes, kernel_size=1)

    def forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            raise ValueError("Slice height and width must be divisible by 8")

        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.bridge(x3)
        x5 = self.up_residual_conv_1(torch.cat([self.upsample_1(x4), x3], dim=1))
        x6 = self.up_residual_conv_2(torch.cat([self.upsample_2(x5), x2], dim=1))
        x7 = self.up_residual_conv_3(torch.cat([self.upsample_3(x6), x1], dim=1))
        return self.output_layer(x7)

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
