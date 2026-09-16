"""SegFormer3D adapter for the DG-CMFNet framework.

This module implements the SegFormer3D architecture as a framework-native model:
data loading, augmentation, loss computation, metrics, checkpointing, and training
all stay inside this repository.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput


def _sequence_or_default(
    values: Sequence[int] | None,
    default: Sequence[int],
    *,
    name: str,
    length: int = 4,
) -> list[int]:
    resolved = list(default if values is None else values)
    if len(resolved) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(resolved)}")
    return [int(value) for value in resolved]


def _check_spatial_tokens(tokens: torch.Tensor, spatial_shape: tuple[int, int, int], context: str) -> None:
    expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"{context} expected {expected_tokens} tokens for spatial shape {spatial_shape}, "
            f"got {tokens.shape[1]}"
        )


def _tokens_to_feature_map(tokens: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    _check_spatial_tokens(tokens, spatial_shape, "tokens_to_feature_map")
    batch_size, _, channels = tokens.shape
    depth, height, width = spatial_shape
    return tokens.reshape(batch_size, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


class PatchEmbedding3d(nn.Module):
    """3D convolutional patch embedding used at each SegFormer3D encoder stage."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        spatial_shape = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, spatial_shape


class EfficientSelfAttention3d(nn.Module):
    """Multi-head self-attention with SegFormer-style spatial reduction."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if sr_ratio < 1:
            raise ValueError(f"sr_ratio must be >= 1, got {sr_ratio}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.attn_dropout = float(attn_dropout)

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        if sr_ratio > 1:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)
        else:
            self.sr = None
            self.sr_norm = None

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        _check_spatial_tokens(x, spatial_shape, "EfficientSelfAttention3d")
        batch_size, num_tokens, channels = x.shape

        query = self.query(x)
        query = query.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3).contiguous()

        if self.sr is not None:
            depth, height, width = spatial_shape
            x_reduced = x.transpose(1, 2).reshape(batch_size, channels, depth, height, width).contiguous()
            x_reduced = self.sr(x_reduced).flatten(2).transpose(1, 2).contiguous()
            if self.sr_norm is None:
                raise RuntimeError("sr_norm is required when spatial reduction is enabled")
            x_reduced = self.sr_norm(x_reduced)
        else:
            x_reduced = x

        key_value = self.key_value(x_reduced)
        key_value = key_value.reshape(batch_size, -1, 2, self.num_heads, self.head_dim)
        key_value = key_value.permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attention = (query @ key.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)
            if self.training and self.attn_dropout > 0.0:
                attention = F.dropout(attention, p=self.attn_dropout, training=True)
            out = attention @ value

        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels).contiguous()
        out = self.proj(out)
        return self.proj_dropout(out)


class DepthwiseConv3d(nn.Module):
    """Depthwise 3D convolution inside SegFormer3D's Mix-FFN block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        _check_spatial_tokens(x, spatial_shape, "DepthwiseConv3d")
        batch_size, _, channels = x.shape
        depth, height, width = spatial_shape
        x = x.transpose(1, 2).reshape(batch_size, channels, depth, height, width).contiguous()
        x = self.dwconv(x)
        x = self.bn(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class MixFeedForward3d(nn.Module):
    """SegFormer3D Mix-FFN block."""

    def __init__(self, embed_dim: int, mlp_ratio: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = DepthwiseConv3d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, spatial_shape)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock3d(nn.Module):
    """SegFormer3D encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention3d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFeedForward3d(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), spatial_shape)
        return x + self.mlp(self.norm2(x), spatial_shape)


class MixVisionTransformer3d(nn.Module):
    """Hierarchical 3D MixVision Transformer encoder."""

    def __init__(
        self,
        in_channels: int,
        sr_ratios: Sequence[int],
        embed_dims: Sequence[int],
        patch_kernel_size: Sequence[int],
        patch_stride: Sequence[int],
        patch_padding: Sequence[int],
        mlp_ratios: Sequence[int],
        num_heads: Sequence[int],
        depths: Sequence[int],
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embeddings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        stage_in_channels = [in_channels] + list(embed_dims[:-1])
        for stage in range(4):
            self.patch_embeddings.append(
                PatchEmbedding3d(
                    in_channels=stage_in_channels[stage],
                    embed_dim=embed_dims[stage],
                    kernel_size=patch_kernel_size[stage],
                    stride=patch_stride[stage],
                    padding=patch_padding[stage],
                )
            )
            self.blocks.append(
                nn.ModuleList(
                    [
                        TransformerBlock3d(
                            embed_dim=embed_dims[stage],
                            num_heads=num_heads[stage],
                            sr_ratio=sr_ratios[stage],
                            mlp_ratio=mlp_ratios[stage],
                            qkv_bias=qkv_bias,
                            attn_dropout=attn_dropout,
                            proj_dropout=proj_dropout,
                            mlp_dropout=mlp_dropout,
                        )
                        for _ in range(depths[stage])
                    ]
                )
            )
            self.norms.append(nn.LayerNorm(embed_dims[stage]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for patch_embedding, blocks, norm in zip(self.patch_embeddings, self.blocks, self.norms):
            tokens, spatial_shape = patch_embedding(x)
            for block in blocks:
                tokens = block(tokens, spatial_shape)
            tokens = norm(tokens)
            x = _tokens_to_feature_map(tokens, spatial_shape)
            features.append(x)
        return features


class LinearEmbedding3d(nn.Module):
    """Linear projection used by the all-MLP decoder head."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return self.norm(x)


class SegFormerDecoderHead3d(nn.Module):
    """All-MLP SegFormer3D decoder head."""

    def __init__(
        self,
        input_feature_dims: Sequence[int],
        decoder_head_embedding_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(input_feature_dims) != 4:
            raise ValueError(f"input_feature_dims must contain four values, got {len(input_feature_dims)}")

        self.linear_c4 = LinearEmbedding3d(input_dim=input_feature_dims[0], embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = LinearEmbedding3d(input_dim=input_feature_dims[1], embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = LinearEmbedding3d(input_dim=input_feature_dims[2], embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = LinearEmbedding3d(input_dim=input_feature_dims[3], embed_dim=decoder_head_embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)

    @staticmethod
    def _project_feature(
        feature: torch.Tensor,
        projection: LinearEmbedding3d,
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, _, depth, height, width = feature.shape
        projected = projection(feature)
        projected = projected.permute(0, 2, 1).reshape(batch_size, -1, depth, height, width).contiguous()
        if (depth, height, width) != target_shape:
            projected = F.interpolate(projected, size=target_shape, mode="trilinear", align_corners=False)
        return projected

    def forward(self, features: list[torch.Tensor], output_shape: tuple[int, int, int]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"SegFormerDecoderHead3d expects four encoder feature maps, got {len(features)}")

        c1, c2, c3, c4 = features
        target_shape = (int(c1.shape[2]), int(c1.shape[3]), int(c1.shape[4]))

        c4 = self._project_feature(c4, self.linear_c4, target_shape)
        c3 = self._project_feature(c3, self.linear_c3, target_shape)
        c2 = self._project_feature(c2, self.linear_c2, target_shape)
        c1 = self._project_feature(c1, self.linear_c1, target_shape)

        x = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)
        if (int(x.shape[2]), int(x.shape[3]), int(x.shape[4])) != output_shape:
            x = F.interpolate(x, size=output_shape, mode="trilinear", align_corners=False)
        return x


class SegFormer3D(nn.Module):
    """SegFormer3D network returning dense per-voxel class logits."""

    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: Sequence[int] | None = None,
        embed_dims: Sequence[int] | None = None,
        patch_kernel_size: Sequence[int] | None = None,
        patch_stride: Sequence[int] | None = None,
        patch_padding: Sequence[int] | None = None,
        mlp_ratios: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        depths: Sequence[int] | None = None,
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 4,
        decoder_dropout: float = 0.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_sr_ratios = _sequence_or_default(sr_ratios, [4, 2, 1, 1], name="sr_ratios")
        resolved_embed_dims = _sequence_or_default(embed_dims, [32, 64, 160, 256], name="embed_dims")
        resolved_patch_kernel = _sequence_or_default(
            patch_kernel_size,
            [7, 3, 3, 3],
            name="patch_kernel_size",
        )
        resolved_patch_stride = _sequence_or_default(patch_stride, [4, 2, 2, 2], name="patch_stride")
        resolved_patch_padding = _sequence_or_default(patch_padding, [3, 1, 1, 1], name="patch_padding")
        resolved_mlp_ratios = _sequence_or_default(mlp_ratios, [4, 4, 4, 4], name="mlp_ratios")
        resolved_num_heads = _sequence_or_default(num_heads, [1, 2, 5, 8], name="num_heads")
        resolved_depths = _sequence_or_default(depths, [2, 2, 2, 2], name="depths")

        for stage, (embed_dim, heads) in enumerate(zip(resolved_embed_dims, resolved_num_heads)):
            if embed_dim % heads != 0:
                raise ValueError(
                    f"embed_dims[{stage}]={embed_dim} must be divisible by num_heads[{stage}]={heads}"
                )

        self.encoder = MixVisionTransformer3d(
            in_channels=in_channels,
            sr_ratios=resolved_sr_ratios,
            embed_dims=resolved_embed_dims,
            patch_kernel_size=resolved_patch_kernel,
            patch_stride=resolved_patch_stride,
            patch_padding=resolved_patch_padding,
            mlp_ratios=resolved_mlp_ratios,
            num_heads=resolved_num_heads,
            depths=resolved_depths,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            mlp_dropout=mlp_dropout,
        )
        self.decoder = SegFormerDecoderHead3d(
            input_feature_dims=list(reversed(resolved_embed_dims)),
            decoder_head_embedding_dim=int(decoder_head_embedding_dim),
            num_classes=int(num_classes),
            dropout=float(decoder_dropout),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv3d):
            kernel_volume = math.prod(module.kernel_size)
            fan_out = kernel_volume * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        features = self.encoder(x)
        return self.decoder(features, output_shape)


class SegFormer3DAdapter(nn.Module):
    """Wrap SegFormer3D so it conforms to the framework ModelAdapter protocol."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        sr_ratios: Sequence[int] | None = None,
        embed_dims: Sequence[int] | None = None,
        patch_kernel_size: Sequence[int] | None = None,
        patch_stride: Sequence[int] | None = None,
        patch_padding: Sequence[int] | None = None,
        mlp_ratios: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        depths: Sequence[int] | None = None,
        decoder_head_embedding_dim: int = 256,
        decoder_dropout: float = 0.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)
        self.model = SegFormer3D(
            in_channels=self.num_modalities,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=self.num_classes,
            decoder_dropout=decoder_dropout,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            mlp_dropout=mlp_dropout,
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]
        if x.dim() != 5:
            raise ValueError(f"SegFormer3DAdapter expects [B, C, D, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")
        return self.model(x)
