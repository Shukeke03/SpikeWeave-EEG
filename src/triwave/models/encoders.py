"""Independent waveform, numerical wavelet structure, and text encoders.

The offline text backend is deliberately an untrained small Transformer. It is
not a clinical pretrained language model and never downloads model weights.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from triwave.config import ModelConfig
from triwave.types import BranchOutput

from .attention import (
    AxialLeadBlock,
    MaskedPatchMerging,
    ShiftedWindowBlock,
    TransformerBlock,
    masked_mean,
    sinusoidal_embedding,
    visible_null,
)


def check_geometry(geometry: Tensor, batch: int, channels: int) -> None:
    if geometry.shape != (batch, channels, 6):
        raise ValueError("lead_geometry must be [B,C,6] positive/negative endpoint coordinates")
    if not torch.isfinite(geometry).all():
        raise ValueError("lead_geometry must contain finite measured endpoint coordinates")


class _BranchEncoder(nn.Module):
    def __init__(self, dim: int, projection_dim: int):
        super().__init__()
        self.null_token = nn.Parameter(torch.zeros(1, dim))
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, projection_dim))
        nn.init.normal_(self.null_token, std=0.02)

    def project(self, pooled: Tensor) -> Tensor:
        return F.normalize(self.projection(pooled), dim=-1, eps=1e-8)

    def null_output(self, batch: int) -> BranchOutput:
        tokens = self.null_token[None].expand(batch, 1, -1)
        valid = torch.ones((batch, 1), dtype=torch.bool, device=tokens.device)
        return BranchOutput(tokens=tokens, valid=valid, pooled=self.project(tokens[:, 0]))

    def _output(self, tokens: Tensor, valid: Tensor, pooled: Tensor,
                presence: Tensor, scales: tuple[Tensor, ...] = (),
                scale_valid: tuple[Tensor, ...] = ()) -> BranchOutput:
        pooled = torch.where(presence[:, None], pooled, self.null_token.to(pooled.dtype))
        tokens, valid = visible_null(tokens, valid, self.null_token)
        return BranchOutput(tokens=tokens, valid=valid, pooled=self.project(pooled),
                            scales=scales, scale_valid=scale_valid)


class TemporalLeadEncoder(_BranchEncoder):
    def __init__(self, config: ModelConfig):
        super().__init__(config.dim, config.projection_dim)
        self.config = config
        self.patch = nn.Linear(config.waveform_patch, config.dim)
        self.coverage = nn.Linear(1, config.dim, bias=False)
        self.geometry = nn.Sequential(nn.Linear(6, config.dim), nn.GELU(),
                                      nn.Linear(config.dim, config.dim))
        self.blocks = nn.ModuleList(AxialLeadBlock(config.dim, config.heads, config.dropout)
                                    for _ in range(config.encoder_depth))
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, waveform: Tensor, lead_geometry: Tensor,
                valid: Tensor | None = None) -> BranchOutput:
        cfg = self.config
        if waveform.ndim != 3 or waveform.shape[1:] != (cfg.channels, cfg.samples):
            raise ValueError("waveform must match configured [B,C,L]")
        batch, channels, length = waveform.shape
        check_geometry(lead_geometry, batch, channels)
        if valid is None:
            valid = torch.ones_like(waveform, dtype=torch.bool)
        elif valid.shape == (batch, channels):
            valid = valid[..., None].expand_as(waveform)
        elif valid.shape != waveform.shape:
            raise ValueError("waveform valid must be [B,C] or [B,C,L]")
        valid = valid.bool()
        waveform = torch.where(valid, waveform, 0)
        patch_count = length // cfg.waveform_patch
        patch_valid = valid.reshape(batch, channels, patch_count, cfg.waveform_patch)
        weight = patch_valid.float().mean(-1)
        tokens = self.patch(waveform.reshape(batch, channels, patch_count, cfg.waveform_patch))
        tokens = tokens + self.coverage(weight[..., None].to(tokens.dtype))
        time = (torch.arange(patch_count, device=waveform.device, dtype=tokens.dtype) + 0.5)
        time = time * cfg.waveform_patch / cfg.sample_rate
        tokens = tokens + sinusoidal_embedding(time, cfg.dim)[None, None]
        tokens = tokens + self.geometry(lead_geometry.to(tokens.dtype))[:, :, None]
        token_valid = weight > 0
        tokens = torch.where(token_valid[..., None], tokens, 0)
        for block in self.blocks:
            tokens = block(tokens, token_valid)
        tokens = torch.where(token_valid[..., None], self.norm(tokens), 0)
        pooled = masked_mean(tokens, weight, (1, 2))
        # Mask-weighted time readout keeps exactly the reference token budget.
        values = tokens.reshape(batch * channels, patch_count, cfg.dim).transpose(1, 2)
        weights = weight.reshape(batch * channels, 1, patch_count).to(tokens.dtype)
        numerator = F.adaptive_avg_pool1d(values * weights, cfg.condition_time_bins)
        denominator = F.adaptive_avg_pool1d(weights, cfg.condition_time_bins)
        summaries = (numerator / denominator.clamp_min(1e-8)).transpose(1, 2)
        summary_valid = denominator[:, 0] > 0
        summaries = torch.where(summary_valid[..., None], summaries, 0)
        summaries = summaries.reshape(batch, channels * cfg.condition_time_bins, cfg.dim)
        summary_valid = summary_valid.reshape(batch, -1)
        return self._output(summaries, summary_valid, pooled, valid.flatten(1).any(-1))


class StructureEncoder(_BranchEncoder):
    """Shared per-lead, hierarchical masked Swin over actual frequency and time."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.dim, config.projection_dim)
        self.config = config
        area = config.structure_frequency_patch * config.structure_time_patch
        self.patch = nn.Linear(area, config.structure_base_dim)
        self.coverage = nn.Linear(1, config.structure_base_dim, bias=False)
        self.geometry = nn.Linear(6, config.structure_base_dim)
        self.stages, self.mergers = nn.ModuleList(), nn.ModuleList()
        self.readouts = nn.ModuleList()
        block_index = 0
        for stage, depth in enumerate(config.structure_depths):
            dim = config.structure_base_dim * 2 ** stage
            heads = min(config.heads, dim)
            while dim % heads:
                heads -= 1
            blocks = nn.ModuleList()
            for _ in range(depth):
                shift = config.structure_window // 2 if block_index % 2 else 0
                blocks.append(ShiftedWindowBlock(dim, heads, config.structure_window,
                                                 shift, config.dropout))
                block_index += 1
            self.stages.append(blocks)
            self.readouts.append(nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.dim)))
            if stage + 1 < len(config.structure_depths):
                self.mergers.append(MaskedPatchMerging(dim))
        self.lead_attention = TransformerBlock(config.dim, config.heads, config.dropout)

    def _readout(self, x: Tensor, weight: Tensor, projection: nn.Module,
                 batch: int, channels: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        size = (cfg.structure_frequency_bins, cfg.condition_time_bins)
        values = x.permute(0, 3, 1, 2)
        weights = weight[:, None].to(x.dtype)
        numerator = F.adaptive_avg_pool2d(values * weights, size)
        denominator = F.adaptive_avg_pool2d(weights, size)
        means = (numerator / denominator.clamp_min(1e-8)).permute(0, 2, 3, 1)
        valid = denominator[:, 0] > 0
        summaries = projection(means)
        summaries = torch.where(valid[..., None], summaries, 0)
        summaries = summaries.reshape(batch, channels, -1, cfg.dim)
        valid = valid.reshape(batch, channels, -1)
        # At fixed physical frequency/time summaries, mix leads separately.
        count = summaries.shape[2]
        summaries = self.lead_attention(
            summaries.transpose(1, 2).reshape(batch * count, channels, cfg.dim),
            valid.transpose(1, 2).reshape(batch * count, channels))
        summaries = summaries.reshape(batch, count, channels, cfg.dim).transpose(1, 2)
        global_values = masked_mean(x, weight, (1, 2))
        global_values = projection(global_values).reshape(batch, channels, cfg.dim)
        global_valid = weight.flatten(1).any(-1).reshape(batch, channels)
        global_values = torch.where(global_valid[..., None], global_values, 0)
        return summaries.reshape(batch, -1, cfg.dim), valid.reshape(batch, -1), global_values, global_valid

    def forward(self, structure: Tensor, valid: Tensor | None,
                lead_geometry: Tensor) -> BranchOutput:
        cfg = self.config
        if structure.ndim != 4 or structure.shape[1:] != (
                cfg.channels, cfg.cwt_frequencies, cfg.cwt_times):
            raise ValueError("structure must match configured [B,C,F,T]")
        batch, channels, frequencies, times = structure.shape
        check_geometry(lead_geometry, batch, channels)
        if valid is None:
            valid = torch.ones_like(structure, dtype=torch.bool)
        if valid.shape != structure.shape:
            raise ValueError("structure valid must have the same [B,C,F,T] shape")
        valid = valid.bool()
        presence = valid.flatten(1).any(-1)
        structure = torch.where(valid, structure, 0)
        pf, pt = cfg.structure_frequency_patch, cfg.structure_time_patch
        fh, tw = (-frequencies) % pf, (-times) % pt
        structure = F.pad(structure, (0, tw, 0, fh))
        valid = F.pad(valid, (0, tw, 0, fh))
        height, width = structure.shape[-2] // pf, structure.shape[-1] // pt
        patches = structure.reshape(batch * channels, height, pf, width, pt).permute(
            0, 1, 3, 2, 4).reshape(batch * channels, height, width, pf * pt)
        weights = valid.reshape(batch * channels, height, pf, width, pt).permute(
            0, 1, 3, 2, 4).reshape(batch * channels, height, width, pf * pt).float().mean(-1)
        x = self.patch(patches) + self.coverage(weights[..., None].to(patches.dtype))
        log_frequency = torch.linspace(math.log(cfg.cwt_min_frequency), math.log(cfg.cwt_max_frequency),
                                       frequencies, device=x.device, dtype=x.dtype)
        log_frequency = F.pad(log_frequency[None], (0, fh), mode="replicate")[0].reshape(height, pf).mean(-1)
        physical_time = (torch.arange(width, device=x.device, dtype=x.dtype) + 0.5)
        physical_time = physical_time * pt / times * cfg.samples / cfg.sample_rate
        x = x + sinusoidal_embedding(log_frequency, x.shape[-1])[None, :, None]
        x = x + sinusoidal_embedding(physical_time, x.shape[-1])[None, None]
        x = x + self.geometry(lead_geometry.to(x.dtype)).reshape(batch * channels, 1, 1, -1)
        x = torch.where((weights > 0)[..., None], x, 0)
        scales, scale_valid = [], []
        for stage, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, weights)
            summaries, summary_valid, lead_global, lead_valid = self._readout(
                x, weights, self.readouts[stage], batch, channels)
            scales.append(summaries)
            scale_valid.append(summary_valid)
            if stage < len(self.mergers):
                x, weights = self.mergers[stage](x, weights)
        tokens = torch.cat((scales[-1], lead_global), dim=1)
        token_valid = torch.cat((scale_valid[-1], lead_valid), dim=1)
        pooled = masked_mean(lead_global, lead_valid, 1)
        return self._output(tokens, token_valid, pooled, presence, tuple(scales), tuple(scale_valid))


class TextEncoder(_BranchEncoder):
    def __init__(self, config: ModelConfig):
        super().__init__(config.dim, config.projection_dim)
        self.config = config
        self.is_medically_pretrained = config.text_backend == "clinicalbert"
        self.hf_model = None
        if config.text_backend == "clinicalbert":
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("clinicalbert backend requires transformers; offline backend needs no HF model") from exc
            self.hf_model = AutoModel.from_pretrained(
                config.text_model_name, local_files_only=config.text_local_files_only)
            self.input_projection = nn.Linear(self.hf_model.config.hidden_size, config.dim)
            if config.freeze_text:
                self.hf_model.requires_grad_(False)
                self.hf_model.eval()
        else:
            self.embedding = nn.Embedding(config.text_vocab_size, config.dim)
            self.position = nn.Embedding(config.text_max_length, config.dim)
            self.blocks = nn.ModuleList(TransformerBlock(config.dim, config.heads, config.dropout)
                                        for _ in range(config.text_depth))
        self.norm = nn.LayerNorm(config.dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.hf_model is not None and self.config.freeze_text:
            self.hf_model.eval()
        return self

    def forward(self, input_ids: Tensor, text_valid: Tensor | None) -> BranchOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.text_max_length or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be [B,N] within configured nonempty text length")
        if text_valid is None:
            text_valid = torch.ones_like(input_ids, dtype=torch.bool)
        if text_valid.shape != input_ids.shape:
            raise ValueError("text_valid must match input_ids")
        text_valid = text_valid.bool()
        presence = text_valid.any(-1)
        # Masked sentinels need not be legal vocabulary IDs.
        safe_ids = torch.where(text_valid, input_ids, 0).long()
        if self.hf_model is not None:
            safe_mask = text_valid.clone()
            safe_mask[~presence, 0] = True
            with torch.no_grad() if self.config.freeze_text else nullcontext():
                output = self.hf_model(input_ids=safe_ids, attention_mask=safe_mask.long()).last_hidden_state
            tokens = self.input_projection(output)
        else:
            position = torch.arange(input_ids.shape[1], device=input_ids.device)
            tokens = self.embedding(safe_ids) + self.position(position)[None]
            tokens = torch.where(text_valid[..., None], tokens, 0)
            for block in self.blocks:
                tokens = block(tokens, text_valid)
        tokens = torch.where(text_valid[..., None], self.norm(tokens), 0)
        pooled = masked_mean(tokens, text_valid, 1)
        return self._output(tokens, text_valid, pooled, presence)
