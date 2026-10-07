"""Original hybrid spiking EEG encoder, adapted from the STAtten design.

This is an EEG adaptation, not a reproduction of a published visual checkpoint.
The sequence dimension is *physical patch time*. Every PLIF state advances once
per patch, rather than repeating a static EEG window for simulation steps.
Q/K/V and both FFN spike locations use the same parametric LIF mechanism. The
continuous membrane residual and joint spike/membrane readout preserve information
that a binary spike-only readout would discard; no energy advantage is claimed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from triwave.config import ModelConfig
from triwave.types import BranchOutput

from .attention import masked_mean, sinusoidal_embedding, visible_null
from .encoders import check_geometry


class _ATanSpike(torch.autograd.Function):
    """Hard threshold forward, derivative of atan(pi*x)/pi backward."""

    @staticmethod
    def forward(ctx, voltage: Tensor) -> Tensor:
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor]:
        (voltage,) = ctx.saved_tensors
        return (gradient / (1 + (math.pi * voltage).square()),)


class ParametricLIF(nn.Module):
    """Per-feature PLIF with FP32, call-local state and detached hard reset.

    Inputs and outputs are [B,K,C,D]. K is the real physical patch sequence.
    Missing time points advance the decay clock with zero current but cannot
    spike or appear in the returned readout. Returned membrane values precede
    reset, so suprathreshold amplitudes remain available to continuous readouts.
    """

    def __init__(self, dim: int, initial_tau: tuple[float, ...] = (2, 4, 8, 16),
                 threshold: float = 1.0, tau_min: float = 1.1,
                 tau_max: float = 64.0):
        super().__init__()
        if dim < 1 or not initial_tau or not 1 < tau_min < tau_max:
            raise ValueError("PLIF requires positive width and 1 < tau_min < tau_max")
        if threshold <= 0 or any(not tau_min < tau < tau_max for tau in initial_tau):
            raise ValueError("PLIF threshold/tau initialization is outside its valid range")
        self.dim, self.threshold = dim, threshold
        self.beta_min, self.beta_max = 1 - 1 / tau_min, 1 - 1 / tau_max
        # Contiguous groups reproduce 128 features per initial tau at D=512.
        group = torch.div(torch.arange(dim) * len(initial_tau), dim, rounding_mode="floor")
        tau = torch.tensor(initial_tau, dtype=torch.float32)[group]
        beta = 1 - tau.reciprocal()
        fraction = (beta - self.beta_min) / (self.beta_max - self.beta_min)
        self.decay_logit = nn.Parameter(torch.logit(fraction))

    @property
    def decay(self) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * self.decay_logit.sigmoid()

    @property
    def tau(self) -> Tensor:
        return (1 - self.decay).reciprocal()

    def forward(self, current: Tensor, valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if current.ndim != 4 or current.shape[-1] != self.dim or current.shape[1] == 0:
            raise ValueError("PLIF current must be nonempty [B,K,C,D]")
        if valid is None:
            valid = torch.ones(current.shape[:-1], dtype=torch.bool, device=current.device)
        if valid.shape != current.shape[:-1]:
            raise ValueError("PLIF valid must match [B,K,C]")
        valid = valid.bool()
        # State is a local tensor: independent records, batch members and calls
        # never share hidden history. AMP cannot quantize threshold comparison.
        with torch.autocast(device_type=current.device.type, enabled=False):
            safe_current = torch.where(valid[..., None], current.float(), 0)
            beta = self.decay.float()
            state = torch.zeros_like(safe_current[:, 0])
            spikes, membranes = [], []
            for step in range(current.shape[1]):
                voltage = beta * state + (1 - beta) * safe_current[:, step]
                spike = _ATanSpike.apply(voltage - self.threshold)
                spike = torch.where(valid[:, step, :, None], spike, 0)
                # The surrogate derivative does not pass through the reset gate.
                state = voltage * (1 - spike.detach())
                spikes.append(spike)
                membranes.append(torch.where(valid[:, step, :, None], voltage, 0))
            return torch.stack(spikes, dim=1), torch.stack(membranes, dim=1)


class TimeSharedDropPath(nn.Module):
    """Drop a continuous residual per record, with one mask for all times/leads."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("drop-path probability must lie in [0,1)")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class SpikingChunkAttention(nn.Module):
    """Softmax-free spike attention over leads and a short physical-time chunk.

    Q/K/V neurons scan the entire physical sequence without resetting at chunk
    boundaries. Chunking only limits attention mixing. LayerNorm replaces visual
    BatchNorm so neither recordings nor future batches alter another sample's
    membrane input. The key-count scale is an EEG adaptation, not a published
    STAtten setting. No ordinary dropout is applied to binary spikes.
    """

    def __init__(self, dim: int, heads: int, chunk_size: int):
        super().__init__()
        if dim % heads or heads <= 0 or chunk_size <= 0:
            raise ValueError("spiking attention requires divisible width and a positive chunk")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.chunk_size = chunk_size
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.query_norm, self.key_norm, self.value_norm = (nn.LayerNorm(dim) for _ in range(3))
        self.query_lif, self.key_lif, self.value_lif = (ParametricLIF(dim) for _ in range(3))
        self.output_norm = nn.LayerNorm(dim)
        self.output_lif = ParametricLIF(dim)
        self.output = nn.Linear(2 * dim, dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, steps, leads, dim = x.shape
        query, _ = self.query_lif(self.query_norm(self.query(x)), valid)
        key, _ = self.key_lif(self.key_norm(self.key(x)), valid)
        value, _ = self.value_lif(self.value_norm(self.value(x)), valid)
        padding = (-steps) % self.chunk_size
        chunks = (steps + padding) // self.chunk_size

        def pack(spikes: Tensor) -> Tensor:
            spikes = F.pad(spikes, (0, 0, 0, 0, 0, padding))
            return spikes.reshape(batch, chunks, self.chunk_size, leads,
                                  self.heads, self.head_dim).permute(
                                      0, 1, 4, 2, 3, 5).reshape(
                                          batch, chunks, self.heads,
                                          self.chunk_size * leads, self.head_dim)

        query, key, value = pack(query), pack(key), pack(value)
        # Associative Q(K^T V) avoids materializing a quadratic attention matrix.
        mixed = query @ (key.transpose(-1, -2) @ value)
        chunk_valid = F.pad(valid, (0, 0, 0, padding)).reshape(batch, chunks, -1)
        key_count = chunk_valid.sum(-1).clamp_min(1).to(mixed.dtype)
        mixed = mixed / (key_count.sqrt()[:, :, None, None, None] * math.sqrt(self.head_dim))
        mixed = mixed.reshape(batch, chunks, self.heads, self.chunk_size, leads,
                              self.head_dim).permute(0, 1, 3, 4, 2, 5).reshape(
                                  batch, steps + padding, leads, dim)[:, :steps]
        spike, membrane = self.output_lif(self.output_norm(mixed.to(x.dtype)), valid)
        output = self.output(torch.cat((spike, membrane), dim=-1).to(x.dtype))
        return torch.where(valid[..., None], output, 0)


class SpikingEEGBlock(nn.Module):
    """Pre-norm hybrid membrane residual with actual QKV/FFN PLIF neurons."""

    def __init__(self, dim: int, heads: int, chunk_size: int, drop_path: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = SpikingChunkAttention(dim, heads, chunk_size)
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.hidden_norm, self.output_norm = nn.LayerNorm(4 * dim), nn.LayerNorm(dim)
        self.hidden_lif, self.output_lif = ParametricLIF(4 * dim), ParametricLIF(dim)
        self.drop_path = TimeSharedDropPath(drop_path)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        x = torch.where(valid[..., None], x, 0)
        x = x + self.drop_path(self.attention(self.norm1(x), valid))
        hidden, _ = self.hidden_lif(self.hidden_norm(self.fc1(self.norm2(x))), valid)
        spike, membrane = self.output_lif(
            self.output_norm(self.fc2(hidden.to(x.dtype))), valid)
        # This is a continuous membrane residual, not inverted spike dropout.
        x = x + self.drop_path((spike + membrane).to(x.dtype))
        return torch.where(valid[..., None], x, 0)


class SpikingEEGEncoder(nn.Module):
    """E branch: physical-time SNN with separate semantic/structural readouts.

    ``patch_mask`` [B,C,K] or [B,K] is True for reconstruction targets. Those
    samples are removed *before* the waveform stem; only a learned token and
    measured time/lead metadata are supplied at their locations. Missing samples
    and reconstruction masks are separate concepts. Reconstruction has shape
    [B,C,K,P]; loss selection and clean target ownership belong to the trainer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.condition_dim = getattr(config, "condition_dim", dim)
        hidden_dim = getattr(config, "projection_hidden_dim", 2 * dim)
        dropout = getattr(config, "representation_dropout", 0.1)
        chunks = tuple(getattr(config, "snn_chunk_sizes", (4, 4, 4, 4, 8, 8, 8, 8)))
        drop_path = getattr(config, "snn_drop_path", 0.1)
        if config.samples % config.waveform_patch or config.encoder_depth <= 0:
            raise ValueError("SNN requires a divisible waveform and at least one block")
        if not chunks or any(chunk <= 0 for chunk in chunks):
            raise ValueError("SNN chunk sizes must be nonempty and positive")
        if not 0 <= dropout < 1 or not 0 <= drop_path < 1:
            raise ValueError("SNN representation dropout and drop path must lie in [0,1)")
        self.patch = nn.Linear(config.waveform_patch, dim)
        self.coverage = nn.Linear(1, dim, bias=False)
        self.geometry = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, dim))
        self.null_embedding = nn.Parameter(torch.empty(1, dim))
        self.null_token = nn.Parameter(torch.empty(1, self.condition_dim))
        self.blocks = nn.ModuleList(
            SpikingEEGBlock(dim, config.heads, chunks[min(index, len(chunks) - 1)],
                           drop_path * index / max(config.encoder_depth - 1, 1))
            for index in range(config.encoder_depth))
        self.readout_lif = ParametricLIF(dim)
        self.readout = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.token_projection = nn.Sequential(nn.Linear(dim, self.condition_dim),
                                              nn.Dropout(dropout))

        def projection() -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, config.projection_dim))

        self.semantic_head, self.structural_head = projection(), projection()
        # No bounded activation: reconstruction must retain signed amplitude.
        self.reconstruction_head = nn.Linear(dim, config.waveform_patch)
        for parameter in (self.mask_token, self.null_embedding, self.null_token):
            nn.init.normal_(parameter, std=0.02)

    def _project(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        semantic = F.normalize(self.semantic_head(pooled), dim=-1, eps=1e-8)
        structural = F.normalize(self.structural_head(pooled), dim=-1, eps=1e-8)
        return semantic, structural

    def null_output(self, batch: int) -> BranchOutput:
        """Visible learned null for fusion, explicitly excluded from alignment."""
        tokens = self.null_token[None].expand(batch, 1, -1)
        valid = torch.ones((batch, 1), dtype=torch.bool, device=tokens.device)
        present = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
        semantic, structural = self._project(self.null_embedding.expand(batch, -1))
        return BranchOutput(tokens=tokens, valid=valid, pooled=semantic,
                            semantic=semantic, structural=structural, present=present,
                            semantic_valid=present, structural_valid=present)

    def sample_patch_mask(self, valid: Tensor, ratio: float = 0.4,
                          generator: torch.Generator | None = None) -> Tensor:
        """Mask random physical time patches, shared across observed leads.

        ``valid`` is [B,C,L] or [B,C]. Each record masks round(ratio*K_observed)
        distinct observed times. Sharing the time mask avoids exposing the same
        target transient through an unmasked parallel lead.
        """
        if not 0 <= ratio <= 1:
            raise ValueError("patch mask ratio must lie in [0,1]")
        if valid.ndim == 2 and valid.shape[1] == self.config.channels:
            valid = valid[..., None].expand(-1, -1, self.config.samples)
        if valid.ndim != 3 or valid.shape[1:] != (self.config.channels, self.config.samples):
            raise ValueError("patch masking valid must be [B,C] or configured [B,C,L]")
        batch, channels, _ = valid.shape
        observed = valid.bool().reshape(batch, channels, -1, self.config.waveform_patch).any(-1)
        mask = torch.zeros_like(observed)
        for record in range(batch):
            times = observed[record].any(0).nonzero(as_tuple=False).flatten()
            count = int(round(ratio * times.numel()))
            if count:
                order = torch.randperm(times.numel(), device=valid.device, generator=generator)
                selected = times[order[:count]]
                mask[record, :, selected] = observed[record, :, selected]
        return mask

    def forward(self, waveform: Tensor, lead_geometry: Tensor, valid: Tensor | None = None,
                patch_mask: Tensor | None = None) -> BranchOutput:
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
        steps, patch_size = length // cfg.waveform_patch, cfg.waveform_patch
        sample_valid = valid.reshape(batch, channels, steps, patch_size)
        weight = sample_valid.float().mean(-1)
        patch_valid = weight > 0
        if patch_mask is None:
            patch_mask = torch.zeros_like(patch_valid)
        elif patch_mask.shape == (batch, steps):
            patch_mask = patch_mask[:, None].expand(batch, channels, steps)
        elif patch_mask.shape != patch_valid.shape:
            raise ValueError("patch_mask must be [B,K] or [B,C,K]")
        patch_mask = patch_mask.bool() & patch_valid
        observed = sample_valid & ~patch_mask[..., None]
        patches = waveform.reshape(batch, channels, steps, patch_size)
        # torch.where happens before any numerical operation, including Linear;
        # masked NaNs and arbitrary target values cannot enter the teacher input.
        patches = torch.where(observed, patches, 0)
        if not torch.isfinite(patches).all():
            raise ValueError("observed waveform values must be finite")
        tokens = self.patch(patches)
        tokens = torch.where(patch_mask[..., None], self.mask_token.to(tokens.dtype), tokens)
        tokens = tokens + self.coverage(weight[..., None].to(tokens.dtype))
        time = (torch.arange(steps, device=waveform.device, dtype=tokens.dtype) + 0.5)
        time = time * patch_size / cfg.sample_rate
        tokens = tokens + sinusoidal_embedding(time, cfg.dim)[None, None]
        tokens = tokens + self.geometry(lead_geometry.to(tokens.dtype))[:, :, None]
        tokens = torch.where(patch_valid[..., None], tokens, 0).transpose(1, 2)
        sequence_valid = patch_valid.transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, sequence_valid)
        spike, membrane = self.readout_lif(tokens, sequence_valid)
        readout = tokens + self.readout(torch.cat((spike, membrane), -1).to(tokens.dtype))
        readout = torch.where(sequence_valid[..., None], readout, 0).transpose(1, 2)
        reconstruction = self.reconstruction_head(readout)
        reconstruction = torch.where(patch_valid[..., None], reconstruction, 0)
        features = torch.where(patch_valid[..., None], self.norm(readout), 0)
        pooled = masked_mean(features, weight, (1, 2))
        present = valid.flatten(1).any(-1)
        pooled = torch.where(present[:, None], pooled, self.null_embedding.to(pooled.dtype))
        semantic, structural = self._project(pooled)
        values = features.reshape(batch * channels, steps, cfg.dim).transpose(1, 2)
        weights = weight.reshape(batch * channels, 1, steps).to(features.dtype)
        numerator = F.adaptive_avg_pool1d(values * weights, cfg.condition_time_bins)
        denominator = F.adaptive_avg_pool1d(weights, cfg.condition_time_bins)
        summaries = (numerator / denominator.clamp_min(1e-8)).transpose(1, 2)
        summary_valid = (denominator[:, 0] > 0).reshape(batch, -1)
        summaries = summaries.reshape(batch, channels * cfg.condition_time_bins, cfg.dim)
        summaries = self.token_projection(summaries)
        summaries, summary_valid = visible_null(summaries, summary_valid, self.null_token)
        return BranchOutput(tokens=summaries, valid=summary_valid, pooled=semantic,
                            semantic=semantic, structural=structural, present=present,
                            semantic_valid=present, structural_valid=present,
                            reconstruction=reconstruction, patch_mask=patch_mask)
