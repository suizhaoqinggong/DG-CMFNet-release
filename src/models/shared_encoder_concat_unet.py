"""Shared-encoder concatenation U-Net for volumetric segmentation."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput

from .dgcmfnet import (
    CoarseGrainedGlobalFusionModule,
    FineGrainedGraphInteractionModule,
    FrequencyGuidance3d,
)


def _normalize_3d_tuple(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {values}")
    return values


def _resolve_stage_values(
    stages: list[int],
    default: Any,
    override: Optional[Union[Sequence[Any], Mapping[Any, Any]]],
    all_stages: list[int],
    name: str,
) -> dict[int, Any]:
    if override is None:
        return {stage: default for stage in stages}
    if isinstance(override, Mapping):
        values = {}
        for stage in stages:
            if stage in override:
                values[stage] = override[stage]
            elif str(stage) in override:
                values[stage] = override[str(stage)]
            else:
                raise ValueError(f"{name} is missing a value for stage {stage}")
        return values

    values_list = list(override)
    if len(values_list) == len(stages):
        return {stage: value for stage, value in zip(stages, values_list)}
    if len(values_list) == len(all_stages):
        return {stage: values_list[all_stages.index(stage)] for stage in stages}
    raise ValueError(
        f"{name} must have either {len(stages)} values for selected stages "
        f"or {len(all_stages)} values for all stages, got {len(values_list)}"
    )


def _make_frequency_guidance(
    semantic_ch: int,
    boundary_ch: int,
    frequency_kernel_size: int,
    gate_mode: str,
    gate_init_bias: float,
) -> nn.Module:
    try:
        return FrequencyGuidance3d(
            semantic_ch=semantic_ch,
            boundary_ch=boundary_ch,
            frequency_kernel_size=frequency_kernel_size,
            gate_mode=gate_mode,
            gate_init_bias=gate_init_bias,
        )
    except TypeError:
        return FrequencyGuidance3d(
            semantic_ch=semantic_ch,
            boundary_ch=boundary_ch,
            frequency_kernel_size=frequency_kernel_size,
        )


def _concat_modality_levels(modality_features: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    if not modality_features:
        raise ValueError("Expected at least one modality feature list")
    num_levels = len(modality_features[0])
    return list(
        reversed(
            [
                torch.cat([features[level_idx] for features in modality_features], dim=1)
                for level_idx in range(num_levels)
            ]
        )
    )


class ModalityLevelFusion3d(nn.Module):
    """Fuse same-level modality features before the decoder."""

    def __init__(
        self,
        channels: int,
        num_modalities: int,
        mode: str,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_modalities = int(num_modalities)
        self.mode = mode
        if mode == "concat":
            self.out_channels = self.channels * self.num_modalities
            self.proj = nn.Identity()
        elif mode == "concat_1x1":
            self.out_channels = self.channels
            self.proj = nn.Sequential(
                nn.Conv3d(self.channels * self.num_modalities, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown baseline_V2 skip_fusion '{mode}'. Expected 'concat' or 'concat_1x1'.")

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modality tensors, got {len(features)}")
        return self.proj(torch.cat(features, dim=1))


class AuxiliarySkipFusion3d(nn.Module):
    """Compress raw and refined skip features back to the original channel width."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.proj = nn.Sequential(
            nn.Conv3d(self.channels * 2, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, raw: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([raw, refined], dim=1))


class ConvolutionBlock3d(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=norm_affine)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.instance_norm(x)
        return self.activation(x)


class DownsampleConv3d(nn.Module):
    """Stride-2 Conv3d downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=norm_affine)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.instance_norm(x)
        return self.activation(x)


class SharedModalityEncoder(nn.Module):
    """Apply one shared 3D encoder independently to each modality."""

    def __init__(
        self,
        num_modalities: int = 4,
        input_channel: int = 1,
        base_ch: int = 8,
        norm_affine: bool = False,
    ) -> None:
        super().__init__()
        if input_channel != 1:
            raise ValueError("SharedModalityEncoder currently expects one channel per modality")

        self.num_modalities = int(num_modalities)
        self.input_channel = int(input_channel)
        self.base_ch = int(base_ch)

        self.encoder_level1 = ConvolutionBlock3d(input_channel, base_ch, norm_affine=norm_affine)
        self.encoder_down1 = DownsampleConv3d(base_ch, 2 * base_ch, norm_affine=norm_affine)
        self.encoder_level2 = ConvolutionBlock3d(2 * base_ch, 2 * base_ch, norm_affine=norm_affine)
        self.encoder_down2 = DownsampleConv3d(2 * base_ch, 4 * base_ch, norm_affine=norm_affine)
        self.encoder_level3 = ConvolutionBlock3d(4 * base_ch, 4 * base_ch, norm_affine=norm_affine)
        self.encoder_down3 = DownsampleConv3d(4 * base_ch, 8 * base_ch, norm_affine=norm_affine)
        self.encoder_level4 = ConvolutionBlock3d(8 * base_ch, 8 * base_ch, norm_affine=norm_affine)
        self.encoder_down4 = DownsampleConv3d(8 * base_ch, 16 * base_ch, norm_affine=norm_affine)
        self.encoder_level5 = ConvolutionBlock3d(16 * base_ch, 16 * base_ch, norm_affine=norm_affine)
        self.encoder_down5 = DownsampleConv3d(16 * base_ch, 32 * base_ch, norm_affine=norm_affine)
        self.encoder_level6 = ConvolutionBlock3d(32 * base_ch, 32 * base_ch, norm_affine=norm_affine)

    def _encode_modality(self, x: torch.Tensor) -> list[torch.Tensor]:
        level1 = self.encoder_level1(x)
        level2 = self.encoder_level2(self.encoder_down1(level1))
        level3 = self.encoder_level3(self.encoder_down2(level2))
        level4 = self.encoder_level4(self.encoder_down3(level3))
        level5 = self.encoder_level5(self.encoder_down4(level4))
        level6 = self.encoder_level6(self.encoder_down5(level5))
        return [level1, level2, level3, level4, level5, level6]

    def encode_modalities(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        if x.dim() != 5:
            raise ValueError(f"SharedModalityEncoder expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        return [
            self._encode_modality(x[:, modality_idx : modality_idx + 1, ...])
            for modality_idx in range(self.num_modalities)
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return _concat_modality_levels(self.encode_modalities(x))


class UpSamplingBlock3d(nn.Module):
    """Trilinear upsampling followed by a convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: tuple[int, int, int] = (2, 2, 2),
        align_corners: bool = True,
        norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="trilinear",
            align_corners=align_corners,
        )
        self.conv_block = ConvolutionBlock3d(in_channels, out_channels, norm_affine=norm_affine)

    def forward(
        self,
        x: torch.Tensor,
        target_spatial_shape: Optional[tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if target_spatial_shape is not None and x.shape[2:] != target_spatial_shape:
            x = F.interpolate(
                x,
                size=target_spatial_shape,
                mode="trilinear",
                align_corners=self.align_corners,
            )
        return self.conv_block(x)


class SharedConcatDecoder(nn.Module):
    """Decoder for modality-concatenated encoder features."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: int = 8,
        level_channels: Optional[Sequence[int]] = None,
        align_corners: bool = True,
        norm_affine: bool = False,
        decoder_type: str = "standard",
        frequency_kernel_size: int = 3,
        frequency_position: str = "decoder_feature",
        frequency_stage_mapping: Optional[Sequence[Sequence[int]]] = None,
        frequency_gate_mode: str = "residual",
        frequency_gate_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if level_channels is None:
            c1 = num_modalities * base_ch
            c2 = num_modalities * 2 * base_ch
            c3 = num_modalities * 4 * base_ch
            c4 = num_modalities * 8 * base_ch
            c5 = num_modalities * 16 * base_ch
            c6 = num_modalities * 32 * base_ch
        else:
            if len(level_channels) != 6:
                raise ValueError(f"level_channels must contain 6 values, got {len(level_channels)}")
            c1, c2, c3, c4, c5, c6 = [int(value) for value in level_channels]

        self.up_bottleneck = UpSamplingBlock3d(c6, c5, align_corners=align_corners, norm_affine=norm_affine)
        self.up4 = UpSamplingBlock3d(c5, c4, align_corners=align_corners, norm_affine=norm_affine)
        self.up3 = UpSamplingBlock3d(c4, c3, align_corners=align_corners, norm_affine=norm_affine)
        self.up2 = UpSamplingBlock3d(c3, c2, align_corners=align_corners, norm_affine=norm_affine)
        self.up1 = UpSamplingBlock3d(c2, c1, align_corners=align_corners, norm_affine=norm_affine)

        self.conv_5 = nn.Conv3d(c5 * 2, c5, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv3d(c4 * 2, c4, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv3d(c3 * 2, c3, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(c2 * 2, c2, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(c1 * 2, c1, kernel_size=3, padding=1)

        self.conv = nn.Conv3d(c1, base_ch, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(base_ch, num_classes, kernel_size=1, padding=0)
        self.decoder_feature_channels = [c6, c5, c4, c3, c2, c1]
        self.decoder_stage_input_channels = [c6, c5, c4, c3, c2]
        self.decoder_skip_channels = [c5, c4, c3, c2, c1]
        self.frequency_stage_mapping: tuple[tuple[int, int], ...] = ()
        self.frequency_guidance = nn.ModuleDict()
        self._boundary_guidance: dict[int, tuple[str, int]] = {}
        self.frequency_position = self._validate_frequency_position(frequency_position)

        decoder_type = decoder_type.strip().lower()
        if decoder_type == "standard":
            pass
        elif decoder_type == "frequency_guided":
            mapping_size = (
                len(self.decoder_feature_channels)
                if self.frequency_position == "decoder_feature"
                else len(self.decoder_skip_channels)
            )
            mapping = (
                self._default_frequency_stage_mapping(mapping_size)
                if frequency_stage_mapping is None
                else frequency_stage_mapping
            )
            self.frequency_stage_mapping = self._validate_frequency_stage_mapping(
                mapping,
                num_stages=mapping_size,
            )
            semantic_channels = (
                self.decoder_feature_channels
                if self.frequency_position == "decoder_feature"
                else self.decoder_stage_input_channels
            )
            boundary_channels = (
                self.decoder_feature_channels
                if self.frequency_position == "decoder_feature"
                else self.decoder_skip_channels
            )
            for idx, (semantic_idx, boundary_idx) in enumerate(self.frequency_stage_mapping):
                key = str(idx)
                self.frequency_guidance[key] = _make_frequency_guidance(
                    semantic_ch=semantic_channels[semantic_idx],
                    boundary_ch=boundary_channels[boundary_idx],
                    frequency_kernel_size=frequency_kernel_size,
                    gate_mode=frequency_gate_mode,
                    gate_init_bias=frequency_gate_init_bias,
                )
                self._boundary_guidance[boundary_idx] = (key, semantic_idx)
        else:
            raise ValueError(
                f"Unknown decoder_type '{decoder_type}'. Expected 'standard' or 'frequency_guided'."
            )

    @staticmethod
    def _validate_frequency_position(position: str) -> str:
        value = position.strip().lower()
        aliases = {"pre_concat_skip": "before_fusion"}
        value = aliases.get(value, value)
        if value not in {"decoder_feature", "before_fusion"}:
            raise ValueError(
                "SharedConcatDecoder frequency_position must be 'decoder_feature' or 'before_fusion'"
            )
        return value

    @staticmethod
    def _default_frequency_stage_mapping(num_stages: int) -> tuple[tuple[int, int], ...]:
        first_boundary_idx = num_stages // 2
        return tuple(
            (semantic_idx, boundary_idx)
            for semantic_idx, boundary_idx in enumerate(range(first_boundary_idx, num_stages))
        )

    @staticmethod
    def _validate_frequency_stage_mapping(
        mapping: Sequence[Sequence[int]],
        num_stages: int,
    ) -> tuple[tuple[int, int], ...]:
        normalized: list[tuple[int, int]] = []
        boundary_indices: set[int] = set()
        for pair in mapping:
            if len(pair) != 2:
                raise ValueError("Each frequency stage mapping item must contain two indices")
            semantic_idx, boundary_idx = int(pair[0]), int(pair[1])
            if not 0 <= semantic_idx < num_stages:
                raise ValueError(f"Invalid semantic decoder stage {semantic_idx}")
            if not 0 <= boundary_idx < num_stages:
                raise ValueError(f"Invalid boundary decoder stage {boundary_idx}")
            if semantic_idx >= boundary_idx:
                raise ValueError(
                    "Frequency guidance must map a deeper semantic stage to a shallower boundary stage"
                )
            if boundary_idx in boundary_indices:
                raise ValueError(f"Boundary decoder stage {boundary_idx} is mapped more than once")
            boundary_indices.add(boundary_idx)
            normalized.append((semantic_idx, boundary_idx))
        return tuple(normalized)

    def _apply_frequency_guidance(
        self,
        feature_idx: int,
        boundary: torch.Tensor,
        decoder_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        guidance = self._boundary_guidance.get(feature_idx)
        if guidance is None:
            return boundary
        key, semantic_idx = guidance
        return self.frequency_guidance[key](boundary, decoder_features[semantic_idx])

    def forward(self, decoder_input: list[torch.Tensor]) -> torch.Tensor:
        if len(decoder_input) != 6:
            raise ValueError(f"SharedConcatDecoder expects 6 feature levels, got {len(decoder_input)}")

        bottleneck_input, decoder_input5, decoder_input4, decoder_input3, decoder_input2, decoder_input1 = (
            decoder_input
        )
        up_blocks = (self.up_bottleneck, self.up4, self.up3, self.up2, self.up1)
        conv_blocks = (self.conv_5, self.conv_4, self.conv_3, self.conv_2, self.conv_1)
        skip_features = (decoder_input5, decoder_input4, decoder_input3, decoder_input2, decoder_input1)

        x = bottleneck_input
        if self.frequency_position == "before_fusion":
            decoder_features: dict[int, torch.Tensor] = {}
            for stage_idx, (up_block, conv_block, skip) in enumerate(
                zip(up_blocks, conv_blocks, skip_features)
            ):
                decoder_features[stage_idx] = x
                skip = self._apply_frequency_guidance(stage_idx, skip, decoder_features)
                upsampled = up_block(x, skip.shape[2:])
                x = conv_block(torch.cat([upsampled, skip], dim=1))
        else:
            decoder_features = {0: x}
            for stage_idx, (up_block, conv_block, skip) in enumerate(
                zip(up_blocks, conv_blocks, skip_features),
                start=1,
            ):
                upsampled = up_block(x, skip.shape[2:])
                x = conv_block(torch.cat([upsampled, skip], dim=1))
                x = self._apply_frequency_guidance(stage_idx, x, decoder_features)
                decoder_features[stage_idx] = x

        return self.final_conv(self.conv(x))


class SharedEncoderConcatUNet(nn.Module):
    """Framework adapter for the shared-encoder, per-level-concat 3D U-Net.

    The model expects ``batch["signal"]`` shaped ``[B, num_modalities, D, H, W]``
    and returns dense logits shaped ``[B, num_classes, D, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: Optional[int] = None,
        n_base_filters: Optional[int] = None,
        in_channels: Optional[int] = None,
        input_channel: int = 1,
        image_size: Optional[Sequence[int]] = None,
        norm_affine: bool = False,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        resolved_base_ch = int(base_ch if base_ch is not None else n_base_filters if n_base_filters is not None else 8)
        resolved_modalities = int(in_channels if in_channels is not None else num_modalities)
        if resolved_base_ch <= 0:
            raise ValueError(f"base_ch must be positive, got {resolved_base_ch}")
        if resolved_modalities <= 0:
            raise ValueError(f"num_modalities must be positive, got {resolved_modalities}")

        self.num_classes = int(num_classes)
        self.num_modalities = resolved_modalities
        self.base_ch = resolved_base_ch
        self.input_channel = int(input_channel)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.norm_affine = bool(norm_affine)
        self.align_corners = bool(align_corners)

        self.encoder = SharedModalityEncoder(
            num_modalities=self.num_modalities,
            input_channel=self.input_channel,
            base_ch=self.base_ch,
            norm_affine=self.norm_affine,
        )
        self.decoder = SharedConcatDecoder(
            num_classes=self.num_classes,
            num_modalities=self.num_modalities,
            base_ch=self.base_ch,
            align_corners=self.align_corners,
            norm_affine=self.norm_affine,
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SharedEncoderConcatUNet expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        input_spatial_shape = x.shape[2:]
        logits = self.decoder(self.encoder(x))
        if logits.shape[2:] != input_spatial_shape:
            logits = F.interpolate(
                logits,
                size=input_spatial_shape,
                mode="trilinear",
                align_corners=self.align_corners,
            )
        return logits


class BaselineV2(SharedEncoderConcatUNet):
    """Plain shared-encoder concat U-Net baseline."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: Optional[int] = None,
        n_base_filters: Optional[int] = None,
        in_channels: Optional[int] = None,
        input_channel: int = 1,
        image_size: Optional[Sequence[int]] = None,
        norm_affine: bool = False,
        align_corners: bool = True,
        fg_gim_stages: Optional[Sequence[int]] = None,
        fg_gim_embed_dim: int = 32,
        fg_gim_embed_dims: Optional[Union[Sequence[int], Mapping[Any, int]]] = None,
        fg_gim_num_heads: int = 4,
        fg_gim_pool_size: Sequence[int] = (4, 4, 4),
        fg_gim_pool_sizes: Optional[Union[Sequence[Sequence[int]], Mapping[Any, Sequence[int]]]] = None,
        fg_gim_window_size: Optional[Sequence[int]] = None,
        fg_gim_auxiliary_skip: bool = False,
        use_cg_gfm: bool = False,
        cg_gfm_embed_dim: int = 256,
        cg_gfm_num_heads: int = 8,
        cg_gfm_num_layers: int = 2,
        cg_gfm_patch_size: Sequence[int] = (1, 1, 1),
        cg_gfm_pos_grid_size: Sequence[int] = (4, 4, 4),
        dropout: float = 0.0,
        unet_bottleneck_dropout: float = 0.0,
        fg_gim_modality_aware: bool = False,
        fg_gim_relation_bias: bool = False,
        fg_gim_edge_mode: str = "all",
        skip_fusion: str = "concat",
        decoder_type: str = "standard",
        decoder_frequency_position: str = "decoder_feature",
        decoder_frequency_kernel_size: int = 3,
        decoder_frequency_stage_mapping: Optional[Sequence[Sequence[int]]] = None,
        decoder_frequency_gate_mode: str = "residual",
        decoder_frequency_gate_init_bias: float = 0.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_modalities=num_modalities,
            base_ch=base_ch,
            n_base_filters=n_base_filters,
            in_channels=in_channels,
            input_channel=input_channel,
            image_size=image_size,
            norm_affine=norm_affine,
            align_corners=align_corners,
        )

        self.encoder_channels = [self.base_ch * (2**stage) for stage in range(5)]
        self.bottleneck_stage = len(self.encoder_channels)
        self.bottleneck_channels = self.base_ch * 32
        self.unet_bottleneck_dropout = float(unet_bottleneck_dropout)
        self.bottleneck_dropout = (
            nn.Dropout3d(self.unet_bottleneck_dropout)
            if self.unet_bottleneck_dropout > 0
            else nn.Identity()
        )
        self.skip_fusion = self._validate_skip_fusion(skip_fusion)
        self.decoder_frequency_position = self._validate_decoder_frequency_position(decoder_frequency_position)
        channels_at_stage = {stage: channels for stage, channels in enumerate(self.encoder_channels)}
        channels_at_stage[self.bottleneck_stage] = self.bottleneck_channels
        valid_stages = list(range(self.bottleneck_stage + 1))

        self.fg_gim_stages = (
            [] if fg_gim_stages is None else [int(stage) for stage in fg_gim_stages]
        )
        self.fg_gim_auxiliary_skip = bool(fg_gim_auxiliary_skip)
        self.fg_gim_auxiliary_stages = {
            stage for stage in self.fg_gim_stages if stage < self.bottleneck_stage
        } if self.fg_gim_auxiliary_skip else set()
        stage_embed_dims = _resolve_stage_values(
            stages=self.fg_gim_stages,
            default=fg_gim_embed_dim,
            override=fg_gim_embed_dims,
            all_stages=valid_stages,
            name="fg_gim_embed_dims",
        )
        stage_pool_sizes = _resolve_stage_values(
            stages=self.fg_gim_stages,
            default=fg_gim_pool_size,
            override=fg_gim_pool_sizes,
            all_stages=valid_stages,
            name="fg_gim_pool_sizes",
        )
        normalized_window_size = (
            None
            if fg_gim_window_size is None
            else _normalize_3d_tuple(fg_gim_window_size, "fg_gim_window_size")
        )
        self.fg_gims = nn.ModuleDict()
        self.fg_gim_auxiliary_fusions = nn.ModuleDict()
        for stage in self.fg_gim_stages:
            if stage not in channels_at_stage:
                raise ValueError(f"Invalid FG-GIM stage {stage}; expected one of {valid_stages}")
            self.fg_gims[str(stage)] = FineGrainedGraphInteractionModule(
                in_channels=channels_at_stage[stage],
                embed_dim=int(stage_embed_dims[stage]),
                num_heads=int(fg_gim_num_heads),
                pool_size=_normalize_3d_tuple(stage_pool_sizes[stage], f"fg_gim_pool_sizes[{stage}]"),
                window_size=normalized_window_size,
                num_modalities=self.num_modalities,
                modality_aware=fg_gim_modality_aware,
                relation_bias=fg_gim_relation_bias,
                edge_mode=fg_gim_edge_mode,
                dropout=float(dropout),
            )
            if stage in self.fg_gim_auxiliary_stages:
                self.fg_gim_auxiliary_fusions[str(stage)] = AuxiliarySkipFusion3d(
                    channels=channels_at_stage[stage]
                )

        self.use_cg_gfm = bool(use_cg_gfm)
        if self.use_cg_gfm:
            self.cg_gfm = CoarseGrainedGlobalFusionModule(
                in_channels=self.bottleneck_channels,
                embed_dim=int(cg_gfm_embed_dim),
                patch_size=_normalize_3d_tuple(cg_gfm_patch_size, "cg_gfm_patch_size"),
                pos_grid_size=_normalize_3d_tuple(cg_gfm_pos_grid_size, "cg_gfm_pos_grid_size"),
                num_heads=int(cg_gfm_num_heads),
                num_layers=int(cg_gfm_num_layers),
                num_modalities=self.num_modalities,
                dropout=float(dropout),
            )
        else:
            self.cg_gfm = nn.Identity()

        self.level_fusions = nn.ModuleList(
            [
                ModalityLevelFusion3d(
                    channels=channels_at_stage[stage],
                    num_modalities=self.num_modalities,
                    mode=self.skip_fusion,
                )
                for stage in valid_stages
            ]
        )
        decoder_level_channels = [fusion.out_channels for fusion in self.level_fusions]
        self.decoder = SharedConcatDecoder(
            num_classes=self.num_classes,
            num_modalities=self.num_modalities,
            base_ch=self.base_ch,
            level_channels=decoder_level_channels,
            align_corners=self.align_corners,
            norm_affine=self.norm_affine,
            decoder_type=decoder_type,
            frequency_kernel_size=int(decoder_frequency_kernel_size),
            frequency_position=self.decoder_frequency_position,
            frequency_stage_mapping=decoder_frequency_stage_mapping,
            frequency_gate_mode=decoder_frequency_gate_mode,
            frequency_gate_init_bias=float(decoder_frequency_gate_init_bias),
        )

    @staticmethod
    def _validate_skip_fusion(skip_fusion: str) -> str:
        value = skip_fusion.strip().lower()
        aliases = {
            "raw_concat": "concat",
            "modality_concat": "concat",
        }
        value = aliases.get(value, value)
        if value not in {"concat", "concat_1x1"}:
            raise ValueError("baseline_V2 skip_fusion must be 'concat' or 'concat_1x1'")
        return value

    @staticmethod
    def _validate_decoder_frequency_position(position: str) -> str:
        value = position.strip().lower()
        aliases = {"pre_concat_skip": "before_fusion"}
        value = aliases.get(value, value)
        if value not in {"decoder_feature", "before_fusion"}:
            raise ValueError(
                "baseline_V2 decoder_frequency_position must be 'decoder_feature' or 'before_fusion'"
            )
        return value

    def _fuse_modality_levels(self, modality_features: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        fused_levels = [
            self.level_fusions[stage]([features[stage] for features in modality_features])
            for stage in range(len(self.level_fusions))
        ]
        return list(reversed(fused_levels))

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"BaselineV2 expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        input_spatial_shape = x.shape[2:]
        modality_features = self.encoder.encode_modalities(x)
        for modality_idx in range(self.num_modalities):
            modality_features[modality_idx][self.bottleneck_stage] = self.bottleneck_dropout(
                modality_features[modality_idx][self.bottleneck_stage]
            )

        for stage in self.fg_gim_stages:
            stage_features = [features[stage] for features in modality_features]
            refined_features = self.fg_gims[str(stage)](stage_features)
            for modality_idx, refined_feature in enumerate(refined_features):
                if stage in self.fg_gim_auxiliary_stages:
                    modality_features[modality_idx][stage] = self.fg_gim_auxiliary_fusions[str(stage)](
                        stage_features[modality_idx],
                        refined_feature,
                    )
                else:
                    modality_features[modality_idx][stage] = refined_feature

        if self.use_cg_gfm:
            bottleneck_features = [features[self.bottleneck_stage] for features in modality_features]
            refined_bottlenecks = self.cg_gfm(bottleneck_features)
            for modality_idx, refined_feature in enumerate(refined_bottlenecks):
                modality_features[modality_idx][self.bottleneck_stage] = refined_feature

        logits = self.decoder(self._fuse_modality_levels(modality_features))
        if logits.shape[2:] != input_spatial_shape:
            logits = F.interpolate(
                logits,
                size=input_spatial_shape,
                mode="trilinear",
                align_corners=self.align_corners,
            )
        return logits


class DGCMFNetV2(BaselineV2):
    """DG-CMFNet modules attached to the baseline_V2 backbone."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: Optional[int] = None,
        n_base_filters: Optional[int] = None,
        in_channels: Optional[int] = None,
        input_channel: int = 1,
        image_size: Optional[Sequence[int]] = None,
        norm_affine: bool = False,
        align_corners: bool = True,
        fg_gim_stages: Optional[Sequence[int]] = None,
        fg_gim_embed_dim: int = 32,
        fg_gim_embed_dims: Optional[Union[Sequence[int], Mapping[Any, int]]] = None,
        fg_gim_num_heads: int = 4,
        fg_gim_pool_size: Sequence[int] = (4, 4, 4),
        fg_gim_pool_sizes: Optional[Union[Sequence[Sequence[int]], Mapping[Any, Sequence[int]]]] = None,
        fg_gim_window_size: Optional[Sequence[int]] = None,
        fg_gim_auxiliary_skip: bool = False,
        use_cg_gfm: bool = True,
        cg_gfm_embed_dim: int = 256,
        cg_gfm_num_heads: int = 8,
        cg_gfm_num_layers: int = 2,
        cg_gfm_patch_size: Sequence[int] = (1, 1, 1),
        cg_gfm_pos_grid_size: Sequence[int] = (4, 4, 4),
        dropout: float = 0.0,
        unet_bottleneck_dropout: float = 0.0,
        fg_gim_modality_aware: bool = False,
        fg_gim_relation_bias: bool = False,
        fg_gim_edge_mode: str = "all",
        skip_fusion: str = "concat",
        decoder_type: str = "frequency_guided",
        decoder_frequency_position: str = "decoder_feature",
        decoder_frequency_kernel_size: int = 3,
        decoder_frequency_stage_mapping: Optional[Sequence[Sequence[int]]] = None,
        decoder_frequency_gate_mode: str = "residual",
        decoder_frequency_gate_init_bias: float = 0.0,
    ) -> None:
        resolved_fg_gim_stages = list(range(6)) if fg_gim_stages is None else fg_gim_stages
        super().__init__(
            num_classes=num_classes,
            num_modalities=num_modalities,
            base_ch=base_ch,
            n_base_filters=n_base_filters,
            in_channels=in_channels,
            input_channel=input_channel,
            image_size=image_size,
            norm_affine=norm_affine,
            align_corners=align_corners,
            fg_gim_stages=resolved_fg_gim_stages,
            fg_gim_embed_dim=fg_gim_embed_dim,
            fg_gim_embed_dims=fg_gim_embed_dims,
            fg_gim_num_heads=fg_gim_num_heads,
            fg_gim_pool_size=fg_gim_pool_size,
            fg_gim_pool_sizes=fg_gim_pool_sizes,
            fg_gim_window_size=fg_gim_window_size,
            fg_gim_auxiliary_skip=fg_gim_auxiliary_skip,
            use_cg_gfm=use_cg_gfm,
            cg_gfm_embed_dim=cg_gfm_embed_dim,
            cg_gfm_num_heads=cg_gfm_num_heads,
            cg_gfm_num_layers=cg_gfm_num_layers,
            cg_gfm_patch_size=cg_gfm_patch_size,
            cg_gfm_pos_grid_size=cg_gfm_pos_grid_size,
            dropout=dropout,
            unet_bottleneck_dropout=unet_bottleneck_dropout,
            fg_gim_modality_aware=fg_gim_modality_aware,
            fg_gim_relation_bias=fg_gim_relation_bias,
            fg_gim_edge_mode=fg_gim_edge_mode,
            skip_fusion=skip_fusion,
            decoder_type=decoder_type,
            decoder_frequency_position=decoder_frequency_position,
            decoder_frequency_kernel_size=decoder_frequency_kernel_size,
            decoder_frequency_stage_mapping=decoder_frequency_stage_mapping,
            decoder_frequency_gate_mode=decoder_frequency_gate_mode,
            decoder_frequency_gate_init_bias=decoder_frequency_gate_init_bias,
        )
