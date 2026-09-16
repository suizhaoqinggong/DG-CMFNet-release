"""DG-CMFNet: Dual-granularity cross-modal fusion network for brain tumor segmentation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.contracts.types import Batch, ModelOutput

# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------

class DoubleConv3d(nn.Module):
    """(Conv3d -> BN -> ReLU) x2"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down3d(nn.Module):
    """DoubleConv3d followed by max-pool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv3d(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        return x, self.pool(x)


class Up3d(nn.Module):
    """Upsample -> concat skip -> DoubleConv3d."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def upsample(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        return x

    def fuse(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x, skip)
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(x, skip)
        return self.conv(fused)


def _low_pass3d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Local average filter that preserves spatial size."""
    if kernel_size == 1:
        return x

    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="replicate")
    return F.avg_pool3d(x, kernel_size=kernel_size, stride=1)


class FrequencyGuidance3d(nn.Module):
    """Use deep low-frequency semantics to modulate shallow high-frequency boundaries.

    The paper-defined residual mode reconstructs the refined shallow feature as
    ``boundary_low + boundary_high * tanh(semantic_gate(semantic_low))``.
    """

    def __init__(
        self,
        semantic_ch: int,
        boundary_ch: int,
        frequency_kernel_size: int = 3,
        gate_mode: str = "residual",
        gate_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if frequency_kernel_size < 1 or frequency_kernel_size % 2 == 0:
            raise ValueError("frequency_kernel_size must be a positive odd integer")
        gate_mode = gate_mode.strip().lower()
        if gate_mode not in {"multiplicative", "residual"}:
            raise ValueError("gate_mode must be either 'multiplicative' or 'residual'")

        self.frequency_kernel_size = frequency_kernel_size
        self.gate_mode = gate_mode
        self.semantic_gate = nn.Conv3d(semantic_ch, boundary_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.constant_(self.semantic_gate.bias, float(gate_init_bias))

    def forward(self, boundary: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        semantic_low = _low_pass3d(semantic, self.frequency_kernel_size)
        boundary_low = _low_pass3d(boundary, self.frequency_kernel_size)
        boundary_high = boundary - boundary_low

        guide = self.semantic_gate(semantic_low)
        if guide.shape[2:] != boundary.shape[2:]:
            guide = F.interpolate(
                guide, size=boundary.shape[2:], mode="trilinear", align_corners=False
            )
        if self.gate_mode == "multiplicative":
            guide = torch.sigmoid(guide)
            return boundary_low + boundary_high * guide

        scale = torch.tanh(guide)
        return boundary_low + boundary_high * scale


# ---------------------------------------------------------------------------
# FG-GIM: Fine-grained Graph Interaction Module
# ---------------------------------------------------------------------------


class FineGrainedGraphInteractionModule(nn.Module):
    """Voxel-level cross-modal graph attention (GAT-style) with sparse spatial
    neighbourhoods, as described in DG-CMFNet Section 3.2 (Eqs 1–8).

    Key properties mandated by the paper:
    - Additive (concat) attention with LeakyReLU  (Eq 4), NOT scaled dot-product.
    - Sparse spatial neighbourhood  N(i)  to reduce computation and preserve
      anatomically meaningful interactions.
    - Multi-head: each head has its own attention vector  a_h ∈ R^{2d}.
    - Residual reconstruction via  1×1×1 conv after upsampling (Eq 8).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        pool_size: tuple[int, int, int] = (4, 4, 4),
        window_size: tuple[int, int, int] | None = None,
        num_modalities: int | None = None,
        modality_aware: bool = False,
        relation_bias: bool = False,
        edge_mode: str = "all",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pool_size = tuple(pool_size)
        self.num_nodes = math.prod(self.pool_size)
        self.edge_mode = self._validate_edge_mode(edge_mode)
        self.num_modalities = None if num_modalities is None else int(num_modalities)
        self.modality_aware = bool(modality_aware)
        self.use_relation_bias = bool(relation_bias)
        if (self.modality_aware or self.use_relation_bias) and self.num_modalities is None:
            raise ValueError("num_modalities is required for modality-aware FG-GIM")

        # Default window: if not given, use pool_size (i.e. dense spatial graph).
        # The paper calls for a *sparse* neighbourhood, so providing a value
        # smaller than pool_size is the intended usage.
        self.window_size = tuple(window_size) if window_size is not None else self.pool_size

        # ---- Shared projection  W : C → E  (Eq 3) ----
        self.W = nn.Linear(in_channels, embed_dim, bias=False)

        self.modality_embedding: nn.Parameter | None = None
        if self.modality_aware:
            self.modality_embedding = nn.Parameter(torch.zeros(self.num_modalities, embed_dim))

        self.relation_bias: nn.Parameter | None = None
        if self.use_relation_bias:
            self.relation_bias = nn.Parameter(
                torch.zeros(self.num_modalities, self.num_modalities, num_heads)
            )

        head_dim = embed_dim // num_heads

        # ---- Per-head additive attention vectors  a_h = [a_h_left ∥ a_h_right] ----
        # Decomposition:  a^⊤ [h_i ∥ h_j]  =  a_left^⊤ h_i  +  a_right^⊤ h_j
        self.a_left = nn.Parameter(torch.empty(num_heads, head_dim))
        self.a_right = nn.Parameter(torch.empty(num_heads, head_dim))
        self._reset_attention_parameters()

        self.leaky_relu = nn.LeakyReLU(0.2)

        # ---- Output  C' → C  (part of ψ in Eq 7–8) ----
        self.out_proj = nn.Linear(embed_dim, in_channels)

        # ---- Reconstruction:  1×1×1 conv after upsampling (Eq 8) ----
        self.reconstruct = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Lazy cache for the [M·N, M·N] boolean adjacency mask.
        self._graph_mask: torch.Tensor | None = None
        self._graph_mask_shape: tuple[int, int] | None = None

    @staticmethod
    def _validate_edge_mode(edge_mode: str) -> str:
        edge_mode = edge_mode.strip().lower()
        valid_modes = {"all", "no_self", "cross_modal", "aligned_cross"}
        if edge_mode not in valid_modes:
            expected = ", ".join(sorted(valid_modes))
            raise ValueError(f"Unknown FG-GIM edge_mode {edge_mode}. Expected one of {expected}.")
        return edge_mode

    @staticmethod
    def _modality_ids(M: int, N: int, device: torch.device) -> torch.Tensor:
        return torch.arange(M, device=device).repeat_interleave(N)

    # ------------------------------------------------------------------
    # Helper: sparse spatial mask
    # ------------------------------------------------------------------

    def _build_graph_mask(
        self,
        M: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a [M·N, M·N] bool mask where entry (s, t) is True iff
        the spatial positions of the two nodes fall within `window_size`.

        The mask is independent of the modality index — every cross-modal
        pair shares the same spatial adjacency pattern.
        """
        pd, ph, pw = self.pool_size
        N = self.num_nodes
        wd, wh, ww = self.window_size
        rd, rh, rw = wd // 2, wh // 2, ww // 2

        # Spatial positions on the pooled grid
        pos = torch.stack(
            torch.meshgrid(
                torch.arange(pd, device=device),
                torch.arange(ph, device=device),
                torch.arange(pw, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).float()  # [pd, ph, pw, 3]
        pos_flat = pos.view(N, 3)  # [N, 3]

        # Chebyshev (L∞) distance within radius → sparse neighbourhood
        dist = (pos_flat.unsqueeze(0) - pos_flat.unsqueeze(1)).abs()  # [N, N, 3]
        spatial_mask = (
            (dist[..., 0] <= rd) & (dist[..., 1] <= rh) & (dist[..., 2] <= rw)
        )  # [N, N]

        # Expand to the full graph. Node order is grouped by modality:
        # [(m=0, i=0..N-1), (m=1, i=0..N-1), ...].
        graph_mask = spatial_mask.repeat(M, M)  # [M·N, M·N]
        if self.edge_mode == "all":
            return graph_mask

        total = M * N
        modality_ids = self._modality_ids(M, N, device)
        same_modality = modality_ids.unsqueeze(1) == modality_ids.unsqueeze(0)
        if self.edge_mode == "no_self":
            return graph_mask & ~torch.eye(total, dtype=torch.bool, device=device)
        if self.edge_mode == "cross_modal":
            return graph_mask & ~same_modality

        node_ids = torch.arange(N, device=device).repeat(M)
        same_spatial_node = node_ids.unsqueeze(1) == node_ids.unsqueeze(0)
        return (~same_modality) & same_spatial_node

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: list of M tensors, each  [B, C, D, H, W]

        Returns:
            list of M refined tensors, each  [B, C, D, H, W]
        """
        M = len(features)
        B, C, D, H, W = features[0].shape
        device = features[0].device
        if self.num_modalities is not None and M != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {M}")
        pd, ph, pw = self.pool_size
        N = self.num_nodes
        total = M * N
        n_heads = self.num_heads
        head_dim = self.embed_dim // n_heads

        # ----------------------------------------------------------------
        # 1. Adaptive pool → compact node representations (Eq 1)
        # ----------------------------------------------------------------
        z_parts: list[torch.Tensor] = []
        for feat in features:
            p = F.adaptive_avg_pool3d(feat, self.pool_size)       # [B, C, pd, ph, pw]
            p = p.view(B, C, -1).permute(0, 2, 1).contiguous()    # [B, N, C]
            z_parts.append(p)

        # Concatenate all modality nodes → graph node set  V (Eq 2)
        z = torch.cat(z_parts, dim=1)  # [B, M·N, C]

        # ----------------------------------------------------------------
        # 2. Build / cache sparse spatial adjacency mask
        # ----------------------------------------------------------------
        mask_key = (total, total)
        if self._graph_mask is None or self._graph_mask_shape != mask_key:
            self._graph_mask = self._build_graph_mask(M, device)
            self._graph_mask_shape = mask_key

        graph_mask = self._graph_mask.to(device=device)  # [total, total]

        # ----------------------------------------------------------------
        # 3. Project to shared embedding (Eq 3):  h = W z
        # ----------------------------------------------------------------
        h = self.W(z)  # [B, total, E]
        modality_ids = self._modality_ids(M, N, device)
        if self.modality_embedding is not None:
            h = h + self.modality_embedding[modality_ids].unsqueeze(0).to(dtype=h.dtype)
        h = h.view(B, total, n_heads, head_dim)  # [B, total, H, d]

        # ----------------------------------------------------------------
        # 4. GAT additive attention (Eq 4)
        #    e_{ij}^h = LeakyReLU( a_h^⊤ [h_i^h ∥ h_j^h] )
        #             = LeakyReLU( a_left_h^⊤ h_i^h  +  a_right_h^⊤ h_j^h )
        # ----------------------------------------------------------------
        score_src = (h * self.a_left.view(1, 1, n_heads, head_dim)).sum(dim=-1)   # [B, total, H]
        score_dst = (h * self.a_right.view(1, 1, n_heads, head_dim)).sum(dim=-1)  # [B, total, H]

        # Pairwise scores (dense, decomposed): [B, total, total, H].
        # Optional relation bias lets the module distinguish source/target
        # modality pairs, e.g. FLAIR->T2 versus T1ce->T1.
        e = score_src.unsqueeze(2) + score_dst.unsqueeze(1)
        if self.relation_bias is not None:
            relation_bias = self.relation_bias[
                modality_ids.unsqueeze(1), modality_ids.unsqueeze(0)
            ]
            e = e + relation_bias.unsqueeze(0).to(dtype=e.dtype)
        e = self.leaky_relu(e)

        # ----------------------------------------------------------------
        # 5. Softmax normalisation over neighbourhood (Eq 5)
        # ----------------------------------------------------------------
        e = e.masked_fill(~graph_mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
        alpha = F.softmax(e, dim=2)  # [B, total, total, H] — dim 2 = dst
        alpha = torch.nan_to_num(alpha, nan=0.0)  # safeguard isolated nodes
        alpha = self.dropout(alpha)

        # ----------------------------------------------------------------
        # 6. Weighted aggregation (Eq 6):  ẑ_i = Σ_j α_{ij} h_j
        # ----------------------------------------------------------------
        out = torch.einsum("bijh,bjhd->bihd", alpha, h)  # [B, total, H, d]
        out = out.reshape(B, total, self.embed_dim)      # [B, total, E]

        # ----------------------------------------------------------------
        # 7. Project  C' → C  (part of ψ in Eqs 7–8)
        # ----------------------------------------------------------------
        out = self.out_proj(out)  # [B, total, C]

        # ----------------------------------------------------------------
        # 8. Reconstruction (Eq 8):  X̃ = X + ψ(Ẑ)
        #    ψ:  reshape → upsample → 1×1×1 conv
        # ----------------------------------------------------------------
        refined: list[torch.Tensor] = []
        for m in range(M):
            m_start = m * N
            m_end = (m + 1) * N
            node_feat = out[:, m_start:m_end, :]  # [B, N, C]

            # Reshape to spatial volume
            node_feat = node_feat.permute(0, 2, 1).contiguous().view(B, C, pd, ph, pw)

            # Upsample to original resolution
            node_feat = F.interpolate(
                node_feat, size=(D, H, W), mode="trilinear", align_corners=False
            )

            # 1×1×1 conv refinement + residual
            node_feat = self.reconstruct(node_feat)
            refined.append(features[m] + node_feat)

        return refined

    def _reset_attention_parameters(self) -> None:
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)


# ---------------------------------------------------------------------------
# CG-GFM: Coarse-grained Global Fusion Module
# ---------------------------------------------------------------------------

class PatchEmbed3d(nn.Module):
    """3D patch embedding."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, E, D//p, H//p, W//p]
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, E]
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm MHSA
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class CoarseGrainedGlobalFusionModule(nn.Module):
    """Patch-level Transformer-based global fusion at bottleneck."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        patch_size: tuple[int, int, int] = (4, 4, 4),
        pos_grid_size: tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 8,
        num_layers: int = 2,
        num_modalities: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_grid_size = pos_grid_size
        self.num_modalities = num_modalities

        self.patch_embed = nn.ModuleList(
            [PatchEmbed3d(in_channels, embed_dim, patch_size) for _ in range(num_modalities)]
        )

        pos_tokens = num_modalities * math.prod(pos_grid_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Project back to spatial feature maps
        self.unpatch_proj = nn.Sequential(
            nn.Linear(embed_dim, in_channels * math.prod(patch_size)),
        )

    def _position_embedding(
        self,
        grid_size: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        gd, gh, gw = self.pos_grid_size
        td, th, tw = grid_size
        pos = self.pos_embed.view(1, self.num_modalities, gd, gh, gw, self.embed_dim)

        if (gd, gh, gw) != (td, th, tw):
            pos = pos.squeeze(0).permute(0, 4, 1, 2, 3).contiguous()
            pos = F.interpolate(pos, size=(td, th, tw), mode="trilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 4, 1).contiguous().unsqueeze(0)

        return pos.view(1, self.num_modalities * td * th * tw, self.embed_dim).to(device=device, dtype=dtype)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: list of M tensors, each [B, C, D, H, W]
        Returns:
            list of M tensors, each [B, C, D, H, W]
        """
        M = len(features)
        B, C, D, H, W = features[0].shape
        device = features[0].device
        p_d, p_h, p_w = self.patch_size

        # Pad to multiples of patch size
        pad_d = (p_d - D % p_d) % p_d
        pad_h = (p_h - H % p_h) % p_h
        pad_w = (p_w - W % p_w) % p_w
        padded_features = []
        for feat in features:
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                feat = F.pad(feat, (0, pad_w, 0, pad_h, 0, pad_d))
            padded_features.append(feat)

        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        # Patch embed each modality
        tokens_list = []
        for m in range(M):
            tok = self.patch_embed[m](padded_features[m])  # [B, N_m, E]
            tokens_list.append(tok)

        # Concatenate across modalities: [B, sum(N_m), E]
        tokens = torch.cat(tokens_list, dim=1)  # [B, M*N, E]

        token_grid = (Dp // p_d, Hp // p_h, Wp // p_w)
        tokens = tokens + self._position_embedding(token_grid, device=device, dtype=tokens.dtype)

        # Transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Split back to modalities
        N_per_mod = tokens_list[0].shape[1]
        out_features = []
        for m in range(M):
            tok = tokens[:, m * N_per_mod : (m + 1) * N_per_mod, :]  # [B, N, E]
            tok = self.unpatch_proj(tok)  # [B, N, C * p_d * p_h * p_w]

            # Reshape to patches
            tok = tok.view(B, N_per_mod, C, p_d, p_h, p_w)

            # Fold patches back to volume
            n_d, n_h, n_w = Dp // p_d, Hp // p_h, Wp // p_w
            tok = tok.view(B, n_d, n_h, n_w, C, p_d, p_h, p_w)
            tok = tok.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            tok = tok.view(B, C, Dp, Hp, Wp)

            # Crop back to original size and residual
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                tok = tok[:, :, :D, :H, :W]
            out_features.append(features[m] + tok)

        return out_features


# ---------------------------------------------------------------------------
# Modality-specific Encoder
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """U-Net style encoder for a single modality."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        bottleneck_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_channels = [base_ch * (2**i) for i in range(5)]
        down_blocks = []
        current_ch = in_ch
        for out_ch in self.encoder_channels:
            down_blocks.append(Down3d(current_ch, out_ch))
            current_ch = out_ch
        self.downs = nn.ModuleList(down_blocks)
        self.bottleneck_channels = base_ch * 32
        self.bottleneck = DoubleConv3d(current_ch, self.bottleneck_channels)
        self.bottleneck_dropout = nn.Dropout3d(bottleneck_dropout) if bottleneck_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        return skips, self.bottleneck_dropout(x)


# ---------------------------------------------------------------------------
# Skip Fusion
# ---------------------------------------------------------------------------

class SkipFusion3d(nn.Module):
    """Fuse same-stage modality skip tensors before the shared decoder."""

    def __init__(
        self,
        channels: int,
        num_modalities: int,
        mode: str = "mean",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_modalities = num_modalities
        self.mode = self._validate_mode(mode)
        self.out_channels = channels * num_modalities if self.mode == "concat" else channels

        if self.mode in {"mean", "concat"}:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv3d(channels * num_modalities, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.out_channels),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _validate_mode(mode: str) -> str:
        mode = mode.strip().lower()
        aliases = {
            "raw_concat": "concat",
            "modality_concat": "concat",
        }
        mode = aliases.get(mode, mode)
        valid_modes = {"mean", "concat", "concat_1x1"}
        if mode not in valid_modes:
            expected = "', '".join(sorted(valid_modes))
            raise ValueError(f"Unknown skip_fusion '{mode}'. Expected one of '{expected}'.")
        return mode

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality skip tensors, got {len(features)}"
            )
        if self.mode == "mean":
            return torch.stack(features, dim=0).mean(dim=0)
        if self.mode == "concat":
            return torch.cat(features, dim=1)
        return self.proj(torch.cat(features, dim=1))


# ---------------------------------------------------------------------------
# Shared Decoder
# ---------------------------------------------------------------------------

class SharedDecoder(nn.Module):
    """U-Net style decoder with skip connections."""

    def __init__(
        self,
        base_ch: int = 32,
        out_ch: int = 4,
        decoder_type: str = "standard",
        frequency_kernel_size: int = 3,
        frequency_position: str = "decoder_feature",
        frequency_stage_mapping: list[list[int]] | None = None,
        frequency_gate_mode: str = "residual",
        frequency_gate_init_bias: float = 0.0,
        level_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        decoder_type = decoder_type.strip().lower()
        if level_channels is None:
            encoder_channels = [base_ch * (2**i) for i in range(5)]
            bottleneck_channels = base_ch * 32
        else:
            if len(level_channels) != 6:
                raise ValueError(f"level_channels must contain 6 values, got {len(level_channels)}")
            encoder_channels = [int(ch) for ch in level_channels[:5]]
            bottleneck_channels = int(level_channels[5])
        up_in_channels = [bottleneck_channels] + list(reversed(encoder_channels[1:]))
        up_out_channels = list(reversed(encoder_channels))
        decoder_feature_channels = [bottleneck_channels] + up_out_channels
        self.decoder_feature_channels = decoder_feature_channels
        self.ups = nn.ModuleList(
            [Up3d(in_ch, out_ch) for in_ch, out_ch in zip(up_in_channels, up_out_channels)]
        )
        self.frequency_stage_mapping: tuple[tuple[int, int], ...] = ()
        self.frequency_guidance = nn.ModuleDict()
        self._boundary_guidance: dict[int, tuple[str, int]] = {}
        self.frequency_position = self._validate_frequency_position(frequency_position)

        if decoder_type == "standard":
            pass
        elif decoder_type == "frequency_guided":
            mapping_size = (
                len(decoder_feature_channels)
                if self.frequency_position == "decoder_feature"
                else len(self.ups)
            )
            if frequency_stage_mapping is None:
                mapping = self._default_frequency_stage_mapping(mapping_size)
            else:
                mapping = frequency_stage_mapping
            self.frequency_stage_mapping = self._validate_frequency_stage_mapping(
                mapping,
                num_stages=mapping_size,
            )
            if self.frequency_position == "decoder_feature":
                semantic_channels = decoder_feature_channels
                boundary_channels = decoder_feature_channels
            else:
                semantic_channels = up_in_channels
                boundary_channels = self._frequency_boundary_channels(
                    self.frequency_position,
                    up_in_channels,
                    up_out_channels,
                )
            for idx, (semantic_idx, boundary_idx) in enumerate(self.frequency_stage_mapping):
                key = str(idx)
                self.frequency_guidance[key] = FrequencyGuidance3d(
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
        self.out_conv = nn.Conv3d(encoder_channels[0], out_ch, kernel_size=1)

    @staticmethod
    def _validate_frequency_position(position: str) -> str:
        position = position.strip().lower()
        valid_positions = {
            "decoder_feature",
            "pre_concat_skip",
            "post_concat_pre_conv",
            "post_conv",
        }
        if position not in valid_positions:
            expected = "', '".join(sorted(valid_positions))
            raise ValueError(
                f"Unknown frequency_position '{position}'. Expected one of '{expected}'."
            )
        return position

    @staticmethod
    def _frequency_boundary_channels(
        frequency_position: str,
        up_in_channels: list[int],
        up_out_channels: list[int],
    ) -> list[int]:
        if frequency_position == "pre_concat_skip":
            return [in_ch // 2 for in_ch in up_in_channels]
        if frequency_position == "post_concat_pre_conv":
            return list(up_in_channels)
        if frequency_position == "post_conv":
            return list(up_out_channels)
        raise ValueError(f"Unknown frequency_position '{frequency_position}'")

    @staticmethod
    def _default_frequency_stage_mapping(num_stages: int) -> tuple[tuple[int, int], ...]:
        """Map deep decoder-side features to the shallow half of the hierarchy."""
        first_boundary_idx = num_stages // 2
        return tuple(
            (semantic_idx, boundary_idx)
            for semantic_idx, boundary_idx in enumerate(range(first_boundary_idx, num_stages))
        )

    @staticmethod
    def _validate_frequency_stage_mapping(
        mapping: list[list[int]] | tuple[tuple[int, int], ...],
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

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(skips) != len(self.ups):
            raise ValueError(f"Expected {len(self.ups)} skip tensors, got {len(skips)}")
        if self.frequency_position == "decoder_feature":
            decoder_features: dict[int, torch.Tensor] = {0: x}
            for idx, (up, skip) in enumerate(zip(self.ups, reversed(skips))):
                x = up(x, skip)
                feature_idx = idx + 1
                guidance = self._boundary_guidance.get(feature_idx)
                if guidance is not None:
                    key, semantic_idx = guidance
                    x = self.frequency_guidance[key](x, decoder_features[semantic_idx])
                decoder_features[feature_idx] = x
            return self.out_conv(x)

        decoder_features: dict[int, torch.Tensor] = {}
        for idx, (up, skip) in enumerate(zip(self.ups, reversed(skips))):
            decoder_features[idx] = x
            guidance = self._boundary_guidance.get(idx)
            if guidance is not None:
                key, semantic_idx = guidance
                semantic = decoder_features[semantic_idx]
                if self.frequency_position == "pre_concat_skip":
                    skip = self.frequency_guidance[key](skip, semantic)

            fused = up.fuse(x, skip)
            if guidance is not None and self.frequency_position == "post_concat_pre_conv":
                key, semantic_idx = guidance
                fused = self.frequency_guidance[key](fused, decoder_features[semantic_idx])

            x = up.conv(fused)
            if guidance is not None and self.frequency_position == "post_conv":
                key, semantic_idx = guidance
                x = self.frequency_guidance[key](x, decoder_features[semantic_idx])
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# DG-CMFNet
# ---------------------------------------------------------------------------

class DGCMFNet(nn.Module):
    """Dual-granularity Cross-Modal Fusion Network for brain tumor segmentation."""

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_ch: int = 32,
        fg_gim_stages: list[int] | None = None,
        fg_gim_embed_dim: int = 64,
        fg_gim_embed_dims: list[int] | dict[int, int] | None = None,
        fg_gim_num_heads: int = 4,
        fg_gim_pool_size: tuple[int, int, int] = (4, 4, 4),
        fg_gim_pool_sizes: (
            list[tuple[int, int, int]] | dict[int, tuple[int, int, int]] | None
        ) = None,
        fg_gim_window_size: tuple[int, int, int] | None = None,
        cg_gfm_embed_dim: int = 256,
        cg_gfm_num_heads: int = 8,
        cg_gfm_num_layers: int = 2,
        cg_gfm_patch_size: tuple[int, int, int] = (1, 1, 1),
        cg_gfm_pos_grid_size: tuple[int, int, int] = (4, 4, 4),
        use_cg_gfm: bool = True,
        dropout: float = 0.0,
        fg_gim_modality_aware: bool = False,
        fg_gim_relation_bias: bool = False,
        fg_gim_edge_mode: str = "all",
        unet_bottleneck_dropout: float = 0.0,
        decoder_type: str = "standard",
        decoder_frequency_position: str = "decoder_feature",
        decoder_frequency_kernel_size: int = 3,
        decoder_frequency_stage_mapping: list[list[int]] | None = None,
        decoder_frequency_gate_mode: str = "residual",
        decoder_frequency_gate_init_bias: float = 0.0,
        skip_fusion: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.base_ch = base_ch
        self.encoder_channels = [base_ch * (2**i) for i in range(5)]
        self.bottleneck_channels = base_ch * 32
        self.bottleneck_stage = len(self.encoder_channels)
        self.use_cg_gfm = use_cg_gfm
        self.unet_bottleneck_dropout = unet_bottleneck_dropout
        self.skip_fusion_mode = SkipFusion3d._validate_mode(skip_fusion)

        # Modality-independent encoders
        self.encoders = nn.ModuleList(
            [
                ModalityEncoder(
                    in_ch=1,
                    base_ch=base_ch,
                    bottleneck_dropout=unet_bottleneck_dropout,
                )
                for _ in range(num_modalities)
            ]
        )

        # FG-GIM is applied to encoder skips by default; stage 5 denotes bottleneck.
        stages = list(range(len(self.encoder_channels))) if fg_gim_stages is None else list(fg_gim_stages)
        self.fg_gim_stages = stages
        channels_at_stage = {stage: ch for stage, ch in enumerate(self.encoder_channels)}
        channels_at_stage[self.bottleneck_stage] = self.bottleneck_channels
        valid_stages = list(range(self.bottleneck_stage + 1))
        stage_embed_dims = self._resolve_stage_values(
            stages=stages,
            default=fg_gim_embed_dim,
            override=fg_gim_embed_dims,
            all_stages=valid_stages,
            name="fg_gim_embed_dims",
        )
        stage_pool_sizes = self._resolve_stage_values(
            stages=stages,
            default=fg_gim_pool_size,
            override=fg_gim_pool_sizes,
            all_stages=valid_stages,
            name="fg_gim_pool_sizes",
        )
        self.fg_gims = nn.ModuleDict()
        for stage in stages:
            if stage not in channels_at_stage:
                raise ValueError(f"Invalid FG-GIM stage {stage}; expected one of {valid_stages}")
            ch = channels_at_stage[stage]
            self.fg_gims[str(stage)] = FineGrainedGraphInteractionModule(
                in_channels=ch,
                embed_dim=int(stage_embed_dims[stage]),
                num_heads=fg_gim_num_heads,
                pool_size=self._normalize_3d_tuple(
                    stage_pool_sizes[stage],
                    f"fg_gim_pool_sizes[{stage}]",
                ),
                window_size=fg_gim_window_size,
                num_modalities=num_modalities,
                modality_aware=fg_gim_modality_aware,
                relation_bias=fg_gim_relation_bias,
                edge_mode=fg_gim_edge_mode,
                dropout=dropout,
            )

        if use_cg_gfm:
            # CG-GFM at bottleneck
            self.cg_gfm = CoarseGrainedGlobalFusionModule(
                in_channels=self.bottleneck_channels,
                embed_dim=cg_gfm_embed_dim,
                patch_size=cg_gfm_patch_size,
                pos_grid_size=cg_gfm_pos_grid_size,
                num_heads=cg_gfm_num_heads,
                num_layers=cg_gfm_num_layers,
                num_modalities=num_modalities,
                dropout=dropout,
            )
        else:
            self.cg_gfm = None

        # Fusion: combine modality features before decoder
        if self.skip_fusion_mode == "concat":
            self.bottleneck_out_channels = self.bottleneck_channels * num_modalities
            self.bottleneck_fusion = nn.Identity()
        else:
            self.bottleneck_out_channels = self.bottleneck_channels
            self.bottleneck_fusion = nn.Sequential(
                nn.Conv3d(self.bottleneck_channels * num_modalities, self.bottleneck_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.bottleneck_channels),
                nn.ReLU(inplace=True),
            )
        self.skip_fusions = nn.ModuleList(
            [
                SkipFusion3d(
                    channels=ch,
                    num_modalities=num_modalities,
                    mode=self.skip_fusion_mode,
                )
                for ch in self.encoder_channels
            ]
        )
        decoder_level_channels = [fusion.out_channels for fusion in self.skip_fusions]
        decoder_level_channels.append(self.bottleneck_out_channels)

        # Shared decoder
        self.decoder = SharedDecoder(
            base_ch=base_ch,
            out_ch=num_classes,
            decoder_type=decoder_type,
            frequency_kernel_size=decoder_frequency_kernel_size,
            frequency_position=decoder_frequency_position,
            frequency_stage_mapping=decoder_frequency_stage_mapping,
            frequency_gate_mode=decoder_frequency_gate_mode,
            frequency_gate_init_bias=decoder_frequency_gate_init_bias,
            level_channels=decoder_level_channels,
        )

    @staticmethod
    def _resolve_stage_values(
        stages: list[int],
        default: object,
        override: list[object] | dict[int, object] | None,
        all_stages: list[int],
        name: str,
    ) -> dict[int, object]:
        if override is None:
            return {stage: default for stage in stages}
        if isinstance(override, dict):
            values: dict[int, object] = {}
            for stage in stages:
                key = stage if stage in override else str(stage)
                if key not in override:
                    raise ValueError(
                        f"{name} is missing a value for FG-GIM stage {stage}"
                    )
                values[stage] = override[key]
            return values

        values_list = list(override)
        if len(values_list) == len(stages):
            return {stage: value for stage, value in zip(stages, values_list)}
        if len(values_list) == len(all_stages):
            return {stage: values_list[all_stages.index(stage)] for stage in stages}
        raise ValueError(
            f"{name} must have either {len(stages)} values for selected FG-GIM stages "
            f"or {len(all_stages)} values for all stages, got {len(values_list)}"
        )

    @staticmethod
    def _normalize_3d_tuple(value: object, name: str) -> tuple[int, int, int]:
        values = tuple(int(v) for v in value)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly three integers, got {values}")
        return values

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["signal"]  # [B, M, D, H, W]
        B, M, D, H, W = x.shape

        # Split modalities
        modality_inputs = [x[:, m : m + 1, ...] for m in range(M)]

        # Encode each modality independently
        all_skips: list[list[torch.Tensor]] = []
        all_bottlenecks: list[torch.Tensor] = []
        for m in range(M):
            skips, bottleneck = self.encoders[m](modality_inputs[m])
            all_skips.append(skips)
            all_bottlenecks.append(bottleneck)

        # Apply FG-GIM at selected stages
        for stage in self.fg_gim_stages:
            if stage == self.bottleneck_stage:
                all_bottlenecks = self.fg_gims[str(stage)](all_bottlenecks)
                continue
            stage_feats = [all_skips[m][stage] for m in range(M)]
            refined = self.fg_gims[str(stage)](stage_feats)
            for m in range(M):
                all_skips[m][stage] = refined[m]

        if self.use_cg_gfm:
            # Apply CG-GFM at bottleneck
            refined_bottlenecks = self.cg_gfm(all_bottlenecks)
        else:
            refined_bottlenecks = all_bottlenecks

        # Fuse modality bottleneck features
        fused_bottleneck = torch.cat(refined_bottlenecks, dim=1)  # [B, M*C, D', H', W']
        fused_bottleneck = self.bottleneck_fusion(fused_bottleneck)

        # Aggregate same-stage modality skip connections before the shared decoder.
        fused_skips: list[torch.Tensor] = []
        for stage, fusion in enumerate(self.skip_fusions):
            stage_feats = [all_skips[m][stage] for m in range(M)]
            fused_skips.append(fusion(stage_feats))

        # Decode
        logits = self.decoder(fused_bottleneck, fused_skips)
        return logits
