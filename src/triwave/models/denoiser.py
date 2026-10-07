"""Continuous MCWave clean prediction on lifted physical-lead DWT coefficients."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triwave.config import ModelConfig
from triwave.types import FusedCondition

from .attention import MaskedAttention, sinusoidal_embedding


class _MCWaveBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.denoiser_dim
        self.norms = nn.ModuleList(nn.LayerNorm(dim, elementwise_affine=False) for _ in range(4))
        self.time_attention = MaskedAttention(dim, config.denoiser_heads, 0.0)
        self.lead_attention = MaskedAttention(dim, config.denoiser_heads, 0.0)
        self.cross_attention = MaskedAttention(dim, config.denoiser_heads, 0.0,
                                               context_dim=config.condition_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.physical_bias_strength = nn.Parameter(torch.ones(config.denoiser_heads))
        self.geometry_bias_strength = nn.Parameter(torch.zeros(config.denoiser_heads))

    def forward(self, x: Tensor, global_condition: Tensor, context: Tensor,
                context_valid: Tensor, physical_bias: Tensor, token_valid: Tensor,
                lead_bias: Tensor) -> Tensor:
        batch, channels, count, dim = x.shape
        modulation = self.modulation(global_condition).chunk(12, dim=-1)

        def normalized(index: int) -> tuple[Tensor, Tensor]:
            shift, scale, gate = modulation[3 * index:3 * index + 3]
            value = self.norms[index](x) * (1 + scale[:, None, None]) + shift[:, None, None]
            return value, gate[:, None, None]

        valid = token_valid.reshape(batch * channels, count)
        value, gate = normalized(0)
        bias = physical_bias[None] * self.physical_bias_strength[:, None, None]
        update = self.time_attention(value.reshape(batch * channels, count, dim), valid, bias=bias)
        x = x + gate * update.reshape(batch, channels, count, dim)
        value, gate = normalized(1)
        value = value.transpose(1, 2).reshape(batch * count, channels, dim)
        lead_valid = token_valid.transpose(1, 2).reshape(batch * count, channels)
        bias = (lead_bias[:, None, None] * self.geometry_bias_strength[None, None, :, None, None])
        bias = bias.expand(batch, count, -1, channels, channels).reshape(
            batch * count, -1, channels, channels)
        update = self.lead_attention(value, lead_valid, bias=bias)
        update = update.reshape(batch, count, channels, dim).transpose(1, 2)
        x = x + gate * update
        value, gate = normalized(2)
        query_valid = token_valid.reshape(batch, channels * count)
        update = self.cross_attention(value.reshape(batch, channels * count, dim), query_valid,
                                      context=context, context_valid=context_valid)
        x = x + gate * update.reshape(batch, channels, count, dim)
        value, gate = normalized(3)
        return torch.where(token_valid[..., None], x + gate * self.mlp(value), 0)


class MCWave(nn.Module):
    """Predict clean signed coefficients with physical lead/time/scale tokens.

    The caller lifts intrinsic montage state [B,r,L] to [B,C,L] before
    tokenization and reduces this head's output back into the subspace. Its Q
    coordinates must never be assigned electrode identities. ``time`` is
    normalized VP noise time [B]. Token coordinates are
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
        self.coverage_embedding = nn.Linear(1, dim, bias=False)
        self.geometry = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.support_embedding = nn.Linear(2, dim)
        self.time_embedding = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.global_embedding = nn.Linear(config.condition_dim, dim)
        self.blocks = nn.ModuleList(_MCWaveBlock(config) for _ in range(config.denoiser_depth))
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
                lead_geometry: Tensor, lead_valid: Tensor | None = None,
                coefficient_valid: Tensor | None = None) -> Tensor:
        cfg = self.config
        if noisy_coefficients.ndim != 3 or noisy_coefficients.shape[1:] != (cfg.channels, cfg.samples):
            raise ValueError("noisy coefficients must match configured [B,C,L]")
        batch, channels, _ = noisy_coefficients.shape
        if lead_geometry.shape != (batch, channels, 6):
            raise ValueError("lead_geometry must contain positive/negative endpoints [B,C,6]")
        if lead_valid is None:
            lead_valid = torch.ones((batch, channels), dtype=torch.bool,
                                    device=noisy_coefficients.device)
        if lead_valid.dtype != torch.bool or lead_valid.shape != (batch, channels):
            raise ValueError("lead_valid must be bool [B,C]")
        if not lead_valid.any(-1).all():
            raise ValueError("each generated sample requires at least one physical lead")
        geometry = torch.where(lead_valid[..., None], lead_geometry, 0)
        if not torch.isfinite(geometry).all():
            raise ValueError("observed lead endpoints must be finite")
        if coefficient_valid is None:
            coefficient_valid = lead_valid[..., None].expand_as(noisy_coefficients)
        elif coefficient_valid.dtype != torch.bool or coefficient_valid.shape != noisy_coefficients.shape:
            raise ValueError("coefficient_valid must be bool [B,C,L]")
        coefficient_valid = coefficient_valid & lead_valid[..., None]
        observed = torch.where(coefficient_valid, noisy_coefficients, 0)
        if not torch.isfinite(observed).all():
            raise ValueError("observed noisy coefficients must be finite")
        if time.shape != (batch,):
            raise ValueError("noise/transport time must be [B]")
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("VP noise time must be finite in [0,1]")
        if condition.tokens.shape[0] != batch or condition.global_vector.shape != (batch, cfg.condition_dim):
            raise ValueError("fused condition must match batch and condition dimension")
        if len(condition.structure_scales) != len(condition.structure_scale_valid):
            raise ValueError("structure scales need explicit masks")
        if (condition.valid.dtype != torch.bool
                or condition.valid.shape != condition.tokens.shape[:2]):
            raise ValueError("condition validity must be bool [batch,tokens]")
        if not torch.isfinite(condition.global_vector).all():
            raise ValueError("condition global vector must be finite")
        tokens, token_masks = [], []
        offset = 0
        for index, (length, embedding) in enumerate(zip(self.band_lengths, self.patch_embeddings)):
            values = observed[:, :, offset:offset + length]
            values = values.reshape(batch, channels, length // cfg.coefficient_patch, cfg.coefficient_patch)
            masks = coefficient_valid[:, :, offset:offset + length].reshape_as(values)
            token = (embedding(values) + self.band_embedding.weight[index]
                     + self.coverage_embedding(masks.float().mean(-1, keepdim=True).to(values.dtype)))
            tokens.append(token)
            token_masks.append(masks.any(-1))
            offset += length
        x = torch.cat(tokens, dim=2)
        token_valid = torch.cat(token_masks, dim=2)
        physical_centers = self.physical_centers.to(x.dtype)
        physical_supports = self.physical_supports.to(x.dtype)
        x = x + sinusoidal_embedding(physical_centers, cfg.denoiser_dim)[None, None]
        coordinates = torch.stack((physical_centers, physical_supports), dim=-1)
        x = x + self.support_embedding(coordinates)[None, None]
        x = x + self.geometry(geometry.to(x.dtype))[:, :, None]
        x = torch.where(token_valid[..., None], x, 0)
        # Endpoint geometry is an attention feature, not a physiological graph
        # smoothness assumption. Preserve positive and negative endpoint roles.
        endpoints = geometry.float()
        lead_distance = (torch.cdist(endpoints[..., :3], endpoints[..., :3])
                         + torch.cdist(endpoints[..., 3:], endpoints[..., 3:])) / 2
        lead_bias = -lead_distance
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
            x = block(x, global_condition, context, valid, self.physical_bias,
                      token_valid, lead_bias)
        shift, scale = self.final_modulation(global_condition).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        output, offset = [], 0
        for length, head in zip(self.band_lengths, self.output_heads):
            count = length // cfg.coefficient_patch
            output.append(head(x[:, :, offset:offset + count]).reshape(batch, channels, length))
            offset += count
        return torch.where(coefficient_valid, torch.cat(output, dim=-1), 0)


# Import compatibility only; new training uses VP clean prediction explicitly.
WaveDiT = MCWave
