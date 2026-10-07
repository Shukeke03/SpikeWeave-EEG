"""A small original axial WaveDiT over every signed DWT coefficient."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triwave.config import ModelConfig
from triwave.types import FusedCondition

from .attention import MaskedAttention, sinusoidal_embedding
from .encoders import check_geometry


class _WaveDiTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.denoiser_dim
        self.norms = nn.ModuleList(nn.LayerNorm(dim, elementwise_affine=False) for _ in range(4))
        self.time_attention = MaskedAttention(dim, config.denoiser_heads, config.dropout)
        self.lead_attention = MaskedAttention(dim, config.denoiser_heads, config.dropout)
        self.cross_attention = MaskedAttention(dim, config.denoiser_heads, config.dropout,
                                               context_dim=config.dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.physical_bias_strength = nn.Parameter(torch.ones(config.denoiser_heads))

    def forward(self, x: Tensor, global_condition: Tensor, context: Tensor,
                context_valid: Tensor, physical_bias: Tensor) -> Tensor:
        batch, channels, count, dim = x.shape
        modulation = self.modulation(global_condition).chunk(12, dim=-1)

        def normalized(index: int) -> tuple[Tensor, Tensor]:
            shift, scale, gate = modulation[3 * index:3 * index + 3]
            value = self.norms[index](x) * (1 + scale[:, None, None]) + shift[:, None, None]
            return value, gate[:, None, None]

        valid = torch.ones((batch * channels, count), dtype=torch.bool, device=x.device)
        value, gate = normalized(0)
        bias = physical_bias[None] * self.physical_bias_strength[:, None, None]
        update = self.time_attention(value.reshape(batch * channels, count, dim), valid, bias=bias)
        x = x + gate * update.reshape(batch, channels, count, dim)
        value, gate = normalized(1)
        value = value.transpose(1, 2).reshape(batch * count, channels, dim)
        lead_valid = torch.ones((batch * count, channels), dtype=torch.bool, device=x.device)
        update = self.lead_attention(value, lead_valid)
        update = update.reshape(batch, count, channels, dim).transpose(1, 2)
        x = x + gate * update
        value, gate = normalized(2)
        query_valid = torch.ones((batch, channels * count), dtype=torch.bool, device=x.device)
        update = self.cross_attention(value.reshape(batch, channels * count, dim), query_valid,
                                      context=context, context_valid=context_valid)
        x = x + gate * update.reshape(batch, channels, count, dim)
        value, gate = normalized(3)
        return x + gate * self.mlp(value)


class WaveDiT(nn.Module):
    """Predicts DDPM-v or a CFM field; the caller defines the probability target.

    ``time`` is normalized transport/noise time [B]. Token coordinates are
    independently defined in physical seconds using the configured sample rate.
    All bands are patchified and unpatchified without dropping any scalar.
    """

    def __init__(self, config: ModelConfig, band_lengths: tuple[int, ...] | list[int]):
        super().__init__()
        self.config = config
        self.band_lengths = tuple(band_lengths)
        if sum(self.band_lengths) != config.samples:
            raise ValueError("complete periodized DWT lengths must sum to samples")
        if any(length <= 0 or length % config.coefficient_patch for length in self.band_lengths):
            raise ValueError("each coefficient band must contain complete patches")
        dim, patch = config.denoiser_dim, config.coefficient_patch
        self.patch_embeddings = nn.ModuleList(nn.Linear(patch, dim) for _ in self.band_lengths)
        self.output_heads = nn.ModuleList(nn.Linear(dim, patch) for _ in self.band_lengths)
        self.band_embedding = nn.Embedding(len(self.band_lengths), dim)
        self.geometry = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.support_embedding = nn.Linear(2, dim)
        self.time_embedding = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.global_embedding = nn.Linear(config.dim, dim)
        self.blocks = nn.ModuleList(_WaveDiTBlock(config) for _ in range(config.denoiser_depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        centers, supports = [], []
        for length in self.band_lengths:
            count = length // patch
            decimation = config.samples / length
            span = patch * decimation / config.sample_rate
            # Approximate filter support, distinct from diffusion time. db4 has
            # eight taps; exact boundary validity is computed by data transforms.
            support = span + (decimation - 1) * 7 / config.sample_rate
            centers.append((torch.arange(count, dtype=torch.float32) + 0.5) * span)
            supports.append(torch.full((count,), support))
        centers, supports = torch.cat(centers), torch.cat(supports)
        self.register_buffer("physical_centers", centers, persistent=False)
        self.register_buffer("physical_supports", supports, persistent=False)
        distance = (centers[:, None] - centers[None, :]).abs()
        physical_bias = -distance / (supports[:, None] + supports[None, :]).clamp_min(1e-8)
        self.register_buffer("physical_bias", physical_bias, persistent=False)

    def forward(self, noisy_coefficients: Tensor, time: Tensor, condition: FusedCondition,
                lead_geometry: Tensor) -> Tensor:
        cfg = self.config
        if noisy_coefficients.ndim != 3 or noisy_coefficients.shape[1:] != (cfg.channels, cfg.samples):
            raise ValueError("noisy coefficients must match configured [B,C,L]")
        batch, channels, _ = noisy_coefficients.shape
        check_geometry(lead_geometry, batch, channels)
        if time.shape != (batch,):
            raise ValueError("noise/transport time must be [B]")
        if condition.tokens.shape[0] != batch or condition.global_vector.shape != (batch, cfg.dim):
            raise ValueError("fused condition must match batch and condition dimension")
        if len(condition.structure_scales) != len(condition.structure_scale_valid):
            raise ValueError("structure scales need explicit masks")
        tokens = []
        offset = 0
        for index, (length, embedding) in enumerate(zip(self.band_lengths, self.patch_embeddings)):
            values = noisy_coefficients[:, :, offset:offset + length]
            values = values.reshape(batch, channels, length // cfg.coefficient_patch, cfg.coefficient_patch)
            token = embedding(values) + self.band_embedding.weight[index]
            tokens.append(token)
            offset += length
        x = torch.cat(tokens, dim=2)
        physical_centers = self.physical_centers.to(x.dtype)
        physical_supports = self.physical_supports.to(x.dtype)
        x = x + sinusoidal_embedding(physical_centers, cfg.denoiser_dim)[None, None]
        coordinates = torch.stack((physical_centers, physical_supports), dim=-1)
        x = x + self.support_embedding(coordinates)[None, None]
        x = x + self.geometry(lead_geometry.to(x.dtype))[:, :, None]
        global_condition = self.global_embedding(condition.global_vector)
        global_condition = global_condition + self.time_embedding(
            sinusoidal_embedding(time.to(x.dtype) * 1000, cfg.denoiser_dim))
        for index, block in enumerate(self.blocks):
            context, valid = condition.tokens, condition.valid
            if condition.structure_scales:
                scale = min(len(condition.structure_scales) - 1,
                            index * len(condition.structure_scales) // max(len(self.blocks), 1))
                context = torch.cat((context, condition.structure_scales[scale]), dim=1)
                valid = torch.cat((valid, condition.structure_scale_valid[scale]), dim=1)
            x = block(x, global_condition, context, valid, self.physical_bias)
        shift, scale = self.final_modulation(global_condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        output, offset = [], 0
        for length, head in zip(self.band_lengths, self.output_heads):
            count = length // cfg.coefficient_patch
            output.append(head(x[:, :, offset:offset + count]).reshape(batch, channels, length))
            offset += count
        return torch.cat(output, dim=-1)
