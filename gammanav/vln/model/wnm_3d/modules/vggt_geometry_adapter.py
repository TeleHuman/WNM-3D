from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _fourier_encode(coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    freqs = (
        2.0 ** torch.arange(num_bands, device=coords.device, dtype=torch.float32)
    ) * math.pi
    x = coords.float().unsqueeze(-1) * freqs
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    return x.flatten(start_dim=-2)


def _make_normalized_coords(t: int, h: int, w: int) -> torch.Tensor:
    tt = torch.linspace(-1.0, 1.0, t, dtype=torch.float32)
    hh = torch.linspace(-1.0, 1.0, h, dtype=torch.float32)
    ww = torch.linspace(-1.0, 1.0, w, dtype=torch.float32)
    grid_t, grid_h, grid_w = torch.meshgrid(tt, hh, ww, indexing="ij")
    return torch.stack([grid_t, grid_h, grid_w], dim=-1)


def _make_anchor_indices(
    target_t: int,
    target_h: int,
    target_w: int,
    source_t: int,
    source_h: int,
    source_w: int,
) -> torch.Tensor:
    tt = (torch.arange(target_t, dtype=torch.float32) + 0.5) * source_t / target_t - 0.5
    hh = (torch.arange(target_h, dtype=torch.float32) + 0.5) * source_h / target_h - 0.5
    ww = (torch.arange(target_w, dtype=torch.float32) + 0.5) * source_w / target_w - 0.5
    grid_t, grid_h, grid_w = torch.meshgrid(tt, hh, ww, indexing="ij")
    return torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(
        1, target_t * target_h * target_w, 3
    )


def _make_position_ids(target_t: int, target_h: int, target_w: int) -> torch.Tensor:
    tt = torch.arange(target_t, dtype=torch.long)
    hh = torch.arange(target_h, dtype=torch.long)
    ww = torch.arange(target_w, dtype=torch.long)
    grid_t, grid_h, grid_w = torch.meshgrid(tt, hh, ww, indexing="ij")
    return torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(
        target_t * target_h * target_w, 3
    )


def _normalize_index_coords_t_h_w(
    coords: torch.Tensor,
    source_t: int,
    source_h: int,
    source_w: int,
) -> torch.Tensor:
    t = 2.0 * coords[..., 0] / max(source_t - 1, 1) - 1.0
    h = 2.0 * coords[..., 1] / max(source_h - 1, 1) - 1.0
    w = 2.0 * coords[..., 2] / max(source_w - 1, 1) - 1.0
    return torch.stack([t, h, w], dim=-1)


class SharedCoordEncoder(nn.Module):
    def __init__(
        self, num_bands: int = 6, hidden_dim: int = 256, out_dim: int = 512
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_bands * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(_fourier_encode(coords, self.num_bands))


class MultiLayerVGGTFusion(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        vggt_dim: int = 2048,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(vggt_dim) for _ in range(num_layers)]
        )
        self.layer_projs = nn.ModuleList(
            [nn.Linear(vggt_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.layer_gate_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.layer_gates = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_layers)]
        )
        self.fused_norm = nn.LayerNorm(hidden_dim)
        self.last_layer_weight_means: torch.Tensor | None = None
        self.record_metrics = True

        for gate in self.layer_gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.layer_projs):
            raise ValueError(
                f"Expected {len(self.layer_projs)} VGGT tap features, got {len(feats)}."
            )

        projected: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for feat, ln, proj, gate_ln, gate in zip(
            feats,
            self.layer_norms,
            self.layer_projs,
            self.layer_gate_norms,
            self.layer_gates,
            strict=True,
        ):
            x = proj(ln(feat))
            projected.append(x)
            gate_logits.append(gate(gate_ln(x)))

        logits = torch.cat(gate_logits, dim=-1)
        weights = logits.softmax(dim=-1)
        source = torch.zeros_like(projected[0])
        for layer_idx, x in enumerate(projected):
            source = source + weights[..., layer_idx : layer_idx + 1] * x

        if self.record_metrics:
            self.last_layer_weight_means = (
                weights.detach().float().mean(dim=(0, 1, 2, 3))
            )
        else:
            self.last_layer_weight_means = None
        return self.fused_norm(source)


class CoarseBasePool(nn.Module):
    def __init__(
        self, target_t: int = 9, target_h: int = 5, target_w: int = 10
    ) -> None:
        super().__init__()
        self.output_size = (target_t, target_h, target_w)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        source_5d = source.permute(0, 4, 1, 2, 3).float()
        base = F.adaptive_avg_pool3d(source_5d, output_size=self.output_size)
        return base.permute(0, 2, 3, 4, 1).to(dtype=source.dtype)


class TargetQueryInitializer(nn.Module):
    def __init__(
        self,
        target_t: int = 9,
        target_h: int = 5,
        target_w: int = 10,
        dim: int = 512,
    ) -> None:
        super().__init__()
        self.q_global = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.q_time = nn.Parameter(torch.zeros(1, target_t, 1, 1, dim))
        self.q_row = nn.Parameter(torch.zeros(1, 1, target_h, 1, dim))
        self.q_col = nn.Parameter(torch.zeros(1, 1, 1, target_w, dim))
        self.base_norm = nn.LayerNorm(dim)
        self.base_proj = nn.Linear(dim, dim)
        self.query_type = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

        for param in (self.q_global, self.q_time, self.q_row, self.q_col):
            nn.init.trunc_normal_(param, std=0.02)

    def forward(
        self,
        base: torch.Tensor,
        target_coord_embed: torch.Tensor,
    ) -> torch.Tensor:
        query_seed = self.q_global + self.q_time + self.q_row + self.q_col
        return (
            query_seed
            + self.base_proj(self.base_norm(base))
            + target_coord_embed
            + self.query_type
        )


class RelativeOffsetBias(nn.Module):
    def __init__(
        self, num_heads: int = 8, num_bands: int = 6, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_bands * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, relative_offsets: torch.Tensor) -> torch.Tensor:
        bias_all = self.mlp(_fourier_encode(relative_offsets, self.num_bands))
        per_head = [
            bias_all[:, :, head_idx, :, head_idx] for head_idx in range(self.num_heads)
        ]
        return torch.stack(per_head, dim=2)


class AnchoredDeformableResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        samples_per_head: int = 8,
        mlp_ratio: float = 4.0,
        source_t: int = 33,
        source_h: int = 32,
        source_w: int = 32,
        target_t: int = 9,
        target_h: int = 5,
        target_w: int = 10,
        coord_num_bands: int = 6,
        extra_offset_radius: tuple[float, float, float] = (1.5, 2.5, 1.5),
        seed: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.samples_per_head = samples_per_head
        self.head_dim = dim // num_heads
        self.source_t = source_t
        self.source_h = source_h
        self.source_w = source_w

        self.norm_q = nn.LayerNorm(dim)
        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, num_heads * samples_per_head * 3),
        )
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.rel_bias = RelativeOffsetBias(
            num_heads=num_heads, num_bands=coord_num_bands
        )
        self.gamma_attn = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma_mlp = nn.Parameter(1e-4 * torch.ones(dim))
        self.source_coord_scale = nn.Parameter(torch.tensor(0.1))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

        bin_radius = (
            source_t / target_t / 2.0,
            source_h / target_h / 2.0,
            source_w / target_w / 2.0,
        )
        self.register_buffer(
            "base_offsets", self._make_base_offsets(bin_radius, seed), persistent=False
        )
        self.register_buffer(
            "extra_radius",
            torch.tensor(extra_offset_radius, dtype=torch.float32).view(1, 1, 1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "bin_radius",
            torch.tensor(bin_radius, dtype=torch.float32).view(1, 1, 1, 1, 3),
            persistent=False,
        )

    def _make_base_offsets(
        self, bin_radius: tuple[float, float, float], seed: int
    ) -> torch.Tensor:
        offsets = torch.zeros(
            self.num_heads, self.samples_per_head, 3, dtype=torch.float32
        )
        if self.samples_per_head <= 1:
            return offsets
        engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
        points = engine.draw(self.num_heads * (self.samples_per_head - 1))
        points = points.mul(2.0).sub(1.0)
        # SobolEngine draws on CPU even when model parameters are being created on
        # the meta device. Keep this scalar tensor colocated with the sampled points
        # so empty-weight model construction can validate full checkpoints.
        points = points * points.new_tensor(bin_radius, dtype=torch.float32)
        offsets[:, 1:, :] = points.view(self.num_heads, self.samples_per_head - 1, 3)
        return offsets

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        anchors: torch.Tensor,
        coord_encoder: SharedCoordEncoder,
    ) -> torch.Tensor:
        batch_size, target_t, target_h, target_w, dim = query.shape
        num_queries = target_t * target_h * target_w
        q = query.reshape(batch_size, num_queries, dim)
        q_norm = self.norm_q(q)

        learned_offsets = self.offset_mlp(q_norm).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.samples_per_head,
            3,
        )
        offsets = self.base_offsets.view(1, 1, self.num_heads, self.samples_per_head, 3)
        offsets = (
            offsets
            + self.extra_radius.to(device=q.device) * learned_offsets.float().tanh()
        )
        coords_idx = (
            anchors.to(device=q.device, dtype=torch.float32).view(
                1, num_queries, 1, 1, 3
            )
            + offsets
        )
        coords_idx = torch.stack(
            [
                coords_idx[..., 0].clamp(0.0, float(self.source_t - 1)),
                coords_idx[..., 1].clamp(0.0, float(self.source_h - 1)),
                coords_idx[..., 2].clamp(0.0, float(self.source_w - 1)),
            ],
            dim=-1,
        )

        source_5d = source.permute(0, 4, 1, 2, 3)
        grid = coords_idx[..., [2, 1, 0]].clone()
        grid[..., 0] = 2.0 * grid[..., 0] / max(self.source_w - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(self.source_h - 1, 1) - 1.0
        grid[..., 2] = 2.0 * grid[..., 2] / max(self.source_t - 1, 1) - 1.0
        sampled = F.grid_sample(
            source_5d.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 3, 4, 1).to(dtype=query.dtype)

        sample_coords = _normalize_index_coords_t_h_w(
            coords_idx, self.source_t, self.source_h, self.source_w
        )
        sampled = (
            sampled
            + self.source_coord_scale.to(dtype=query.dtype)
            * coord_encoder(sample_coords).to(dtype=query.dtype)
            + self.visual_type.to(device=query.device, dtype=query.dtype)
        )

        q_head = self.q_proj(q_norm).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        )
        k_all = self.k_proj(sampled).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.samples_per_head,
            self.num_heads,
            self.head_dim,
        )
        v_all = self.v_proj(sampled).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.samples_per_head,
            self.num_heads,
            self.head_dim,
        )
        k_head = torch.stack(
            [k_all[:, :, h, :, h, :] for h in range(self.num_heads)], dim=2
        )
        v_head = torch.stack(
            [v_all[:, :, h, :, h, :] for h in range(self.num_heads)], dim=2
        )

        attn_logits = (q_head.unsqueeze(-2) * k_head).sum(dim=-1) / math.sqrt(
            self.head_dim
        )
        relative_offsets = offsets / self.bin_radius.to(device=q.device).clamp_min(1e-6)
        attn_logits = attn_logits + self.rel_bias(relative_offsets).to(
            dtype=attn_logits.dtype
        )
        attn = attn_logits.softmax(dim=-1)

        out = (
            (attn.unsqueeze(-1) * v_head)
            .sum(dim=-2)
            .reshape(batch_size, num_queries, dim)
        )
        out = self.out_proj(out)
        q = q + self.gamma_attn.to(dtype=q.dtype) * out
        q = q + self.gamma_mlp.to(dtype=q.dtype) * self.mlp(self.norm_mlp(q))
        return q.reshape(batch_size, target_t, target_h, target_w, dim)


class SelfAttentionWithRelativeBias(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits + rel_bias.to(
            device=x.device, dtype=attn_logits.dtype
        ).unsqueeze(0)
        attn = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.out_proj(out)


class GeometryTemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        target_t: int = 9,
        target_h: int = 5,
        target_w: int = 10,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.target_t = target_t
        self.target_h = target_h
        self.target_w = target_w
        self.num_heads = num_heads

        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = SelfAttentionWithRelativeBias(dim, num_heads)
        self.spatial_rel_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * target_h - 1, 2 * target_w - 1)
        )
        self.register_buffer(
            "spatial_rel_h",
            self._make_spatial_indices(target_h, target_w)[0],
            persistent=False,
        )
        self.register_buffer(
            "spatial_rel_w",
            self._make_spatial_indices(target_h, target_w)[1],
            persistent=False,
        )

        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = SelfAttentionWithRelativeBias(dim, num_heads)
        self.temporal_rel_bias = nn.Parameter(torch.zeros(num_heads, 2 * target_t - 1))
        self.register_buffer(
            "temporal_rel_t", self._make_temporal_indices(target_t), persistent=False
        )

        self.norm_ffn = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gamma_spatial = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma_temporal = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(1e-4 * torch.ones(dim))

    @staticmethod
    def _make_spatial_indices(
        target_h: int, target_w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hh = torch.arange(target_h)
        ww = torch.arange(target_w)
        grid_h, grid_w = torch.meshgrid(hh, ww, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).reshape(target_h * target_w, 2)
        rel_h = coords[:, 0:1] - coords[:, 0].view(1, -1) + (target_h - 1)
        rel_w = coords[:, 1:2] - coords[:, 1].view(1, -1) + (target_w - 1)
        return rel_h.long(), rel_w.long()

    @staticmethod
    def _make_temporal_indices(target_t: int) -> torch.Tensor:
        idx = torch.arange(target_t)
        return (idx[:, None] - idx[None, :] + (target_t - 1)).long()

    def _spatial_bias(self) -> torch.Tensor:
        return self.spatial_rel_bias[:, self.spatial_rel_h, self.spatial_rel_w]

    def _temporal_bias(self) -> torch.Tensor:
        return self.temporal_rel_bias[:, self.temporal_rel_t]

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, target_t, target_h, target_w, dim = query.shape
        if (target_t, target_h, target_w) != (
            self.target_t,
            self.target_h,
            self.target_w,
        ):
            raise ValueError(
                f"Expected query grid {(self.target_t, self.target_h, self.target_w)}, "
                f"got {(target_t, target_h, target_w)}."
            )

        x = query.reshape(batch_size * target_t, target_h * target_w, dim)
        x = x + self.gamma_spatial.to(dtype=x.dtype) * self.spatial_attn(
            self.norm_spatial(x),
            self._spatial_bias(),
        )
        query = x.reshape(batch_size, target_t, target_h, target_w, dim)

        x = query.permute(0, 2, 3, 1, 4).reshape(
            batch_size * target_h * target_w, target_t, dim
        )
        x = x + self.gamma_temporal.to(dtype=x.dtype) * self.temporal_attn(
            self.norm_temporal(x),
            self._temporal_bias(),
        )
        query = x.reshape(batch_size, target_h, target_w, target_t, dim).permute(
            0, 3, 1, 2, 4
        )

        query = query + self.gamma_ffn.to(dtype=query.dtype) * self.ffn(
            self.norm_ffn(query)
        )
        return query


class WanTokenHead(nn.Module):
    def __init__(
        self, dim: int = 512, out_dim: int = 3072, zero_init_output: bool = False
    ) -> None:
        super().__init__()
        self.detail_norm = nn.LayerNorm(dim)
        self.detail_proj = nn.Linear(dim, out_dim)
        self.coarse_norm = nn.LayerNorm(dim)
        self.coarse_proj = nn.Linear(dim, out_dim)
        if zero_init_output:
            nn.init.zeros_(self.detail_proj.weight)
            nn.init.zeros_(self.detail_proj.bias)
            nn.init.zeros_(self.coarse_proj.weight)
            nn.init.zeros_(self.coarse_proj.bias)

    def forward(self, query: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        detail = self.detail_proj(self.detail_norm(query))
        coarse = self.coarse_proj(self.coarse_norm(base))
        tokens = detail + coarse
        return tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])


class VGGTOmegaGeometryAdapter(nn.Module):
    """Temporal geometry encoder over frozen VGGT-Omega patch taps.

    The class name is intentionally kept stable for the existing WNM3D wiring.
    Internally this is the TGE design: high-resolution VGGT memory, anchored
    deformable reads, target-grid temporal/spatial reasoning, then Wan-width tokens.
    """

    def __init__(
        self,
        output_dim: int,
        vggt_token_dim: int = 2048,
        adapter_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_blocks: int = 2,
        max_temporal_frames: int = 128,
        max_spatial_tokens: int = 4096,
        zero_init_output: bool = False,
        source_t: int = 33,
        source_h: int = 32,
        source_w: int = 32,
        target_t: int = 9,
        target_h: int = 5,
        target_w: int = 10,
        samples_per_head: int = 8,
        num_resampler_blocks: int | None = None,
        num_geometry_blocks: int | None = None,
        mlp_ratio: float = 4.0,
        coord_num_bands: int = 6,
        extra_offset_radius: tuple[float, float, float] = (1.5, 2.5, 1.5),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del max_temporal_frames, max_spatial_tokens, dropout
        self.output_dim = output_dim
        self.vggt_token_dim = vggt_token_dim
        self.adapter_dim = adapter_dim
        self.num_vggt_taps = num_layers
        self.source_t = source_t
        self.source_h = source_h
        self.source_w = source_w
        self.target_t = target_t
        self.target_h = target_h
        self.target_w = target_w
        self.source_patch_tokens = source_h * source_w
        self.target_tokens = target_t * target_h * target_w
        self.last_layer_weight_means: torch.Tensor | None = None
        self.record_metrics = True

        self.layer_fusion = MultiLayerVGGTFusion(
            num_layers, vggt_token_dim, adapter_dim
        )
        self.coord_encoder = SharedCoordEncoder(
            coord_num_bands, hidden_dim=256, out_dim=adapter_dim
        )
        self.coarse_pool = CoarseBasePool(target_t, target_h, target_w)
        self.query_initializer = TargetQueryInitializer(
            target_t, target_h, target_w, adapter_dim
        )
        resampler_blocks = (
            num_blocks if num_resampler_blocks is None else num_resampler_blocks
        )
        geometry_blocks = (
            num_blocks if num_geometry_blocks is None else num_geometry_blocks
        )
        self.resampler_blocks = nn.ModuleList(
            [
                AnchoredDeformableResamplerBlock(
                    dim=adapter_dim,
                    num_heads=num_heads,
                    samples_per_head=samples_per_head,
                    mlp_ratio=mlp_ratio,
                    source_t=source_t,
                    source_h=source_h,
                    source_w=source_w,
                    target_t=target_t,
                    target_h=target_h,
                    target_w=target_w,
                    coord_num_bands=coord_num_bands,
                    extra_offset_radius=extra_offset_radius,
                    seed=block_idx + 1,
                )
                for block_idx in range(resampler_blocks)
            ]
        )
        self.geometry_blocks = nn.ModuleList(
            [
                GeometryTemporalBlock(
                    adapter_dim,
                    num_heads,
                    target_t=target_t,
                    target_h=target_h,
                    target_w=target_w,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(geometry_blocks)
            ]
        )
        self.token_head = WanTokenHead(
            adapter_dim, output_dim, zero_init_output=zero_init_output
        )

        self.register_buffer(
            "source_coords",
            _make_normalized_coords(source_t, source_h, source_w).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "target_coords",
            _make_normalized_coords(target_t, target_h, target_w).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "anchor_indices",
            _make_anchor_indices(
                target_t, target_h, target_w, source_t, source_h, source_w
            ),
            persistent=False,
        )
        self.register_buffer(
            "past_position_ids",
            _make_position_ids(target_t, target_h, target_w),
            persistent=False,
        )

    def forward(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor | None],
        patch_token_start: int,
        target_frames: int,
        target_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        target_h, target_w = target_grid_size
        if (target_frames, target_h, target_w) != (
            self.target_t,
            self.target_h,
            self.target_w,
        ):
            raise ValueError(
                f"TGE target grid is fixed to {(self.target_t, self.target_h, self.target_w)}, "
                f"got {(target_frames, target_h, target_w)}."
            )

        cached_layers = [
            tokens for tokens in aggregated_tokens_list if tokens is not None
        ]
        if len(cached_layers) < self.num_vggt_taps:
            raise ValueError(
                f"Expected at least {self.num_vggt_taps} VGGT cached layers, got {len(cached_layers)}."
            )
        cached_layers = cached_layers[-self.num_vggt_taps :]

        feats: list[torch.Tensor] = []
        for tokens in cached_layers:
            if tokens.shape[1] != self.source_t:
                raise ValueError(
                    f"Expected VGGT source_t={self.source_t}, got tokens shape {tuple(tokens.shape)}."
                )
            patch_tokens = tokens[
                :, :, patch_token_start : patch_token_start + self.source_patch_tokens
            ]
            if patch_tokens.shape[2] != self.source_patch_tokens:
                raise ValueError(
                    f"Expected {self.source_patch_tokens} VGGT patch tokens after patch_token_start={patch_token_start}, "
                    f"got {patch_tokens.shape[2]} from {tuple(tokens.shape)}."
                )
            feats.append(
                patch_tokens.reshape(
                    tokens.shape[0],
                    self.source_t,
                    self.source_h,
                    self.source_w,
                    self.vggt_token_dim,
                )
            )

        self.layer_fusion.record_metrics = self.record_metrics
        source = self.layer_fusion(feats)
        self.last_layer_weight_means = (
            self.layer_fusion.last_layer_weight_means if self.record_metrics else None
        )
        base = self.coarse_pool(source)
        target_coord_embed = self.coord_encoder(
            self.target_coords.to(device=source.device)
        ).to(dtype=source.dtype)
        query = self.query_initializer(base, target_coord_embed)

        for block in self.resampler_blocks:
            query = block(
                query=query,
                source=source,
                anchors=self.anchor_indices,
                coord_encoder=self.coord_encoder,
            )
        for block in self.geometry_blocks:
            query = block(query)

        return self.token_head(query, base)
