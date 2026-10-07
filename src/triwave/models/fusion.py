"""ViLT-inspired condition-token interaction, after independent alignment."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triwave.config import ModelConfig
from triwave.types import BranchOutput, FusedCondition

from .attention import TransformerBlock, masked_mean, visible_null


class ConditionFusion(nn.Module):
    """Bounded text FiLM and one shared, light condition-token Transformer.

    Modality dropout is performed by TriWaveModel before this method. Replacing
    the complete text BranchOutput replaces its tokens, pool, FiLM, and global
    contribution together. Alignment never calls this module.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modality = nn.Embedding(5, config.dim)
        self.text_film = nn.Linear(config.projection_dim, 2 * config.dim)
        nn.init.zeros_(self.text_film.weight)
        nn.init.zeros_(self.text_film.bias)
        self.attribute_value = nn.Linear(1, config.dim)
        self.attribute_identity = nn.Embedding(config.attribute_dim, config.dim)
        self.attribute_null = nn.Parameter(torch.zeros(1, config.dim))
        self.geometry = nn.Sequential(nn.Linear(6, config.dim), nn.GELU(),
                                      nn.Linear(config.dim, config.dim))
        self.blocks = nn.ModuleList(TransformerBlock(config.dim, config.heads, config.dropout)
                                    for _ in range(config.fusion_depth))
        self.norm = nn.LayerNorm(config.dim)
        nn.init.normal_(self.attribute_null, std=0.02)

    def _attributes(self, values: Tensor | None, valid: Tensor | None,
                    batch: int) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if values is None:
            return self.attribute_null[None].expand(batch, 1, -1), torch.ones(
                (batch, 1), dtype=torch.bool, device=self.attribute_null.device)
        if values.shape != (batch, cfg.attribute_dim):
            raise ValueError("attribute_values must be [B,attribute_dim]")
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        if valid.shape != values.shape:
            raise ValueError("attribute_valid must match attribute_values")
        valid = valid.bool()
        safe_values = torch.where(valid, values, 0)
        tokens = self.attribute_value(safe_values[..., None])
        identity = torch.arange(cfg.attribute_dim, device=tokens.device)
        tokens = tokens + self.attribute_identity(identity)[None]
        return visible_null(tokens, valid, self.attribute_null)

    def forward(self, reference: BranchOutput, structure: BranchOutput,
                text: BranchOutput, attributes: Tensor | None = None,
                attribute_valid: Tensor | None = None,
                lead_geometry: Tensor | None = None) -> FusedCondition:
        batch = reference.tokens.shape[0]
        if structure.tokens.shape[0] != batch or text.tokens.shape[0] != batch:
            raise ValueError("condition branches must have identical batch sizes")
        scale, shift = self.text_film(text.pooled).chunk(2, dim=-1)
        scale, shift = 0.1 * torch.tanh(scale), 0.1 * torch.tanh(shift)

        def modulate(tokens: Tensor, valid: Tensor) -> Tensor:
            tokens = tokens * (1 + scale[:, None]) + shift[:, None]
            return torch.where(valid[..., None], tokens, 0)

        structure_tokens = modulate(structure.tokens, structure.valid)
        structure_scales = tuple(modulate(tokens, valid) for tokens, valid in
                                 zip(structure.scales, structure.scale_valid))
        if len(structure.scales) != len(structure.scale_valid):
            raise ValueError("structure scale banks require explicit validity masks")
        attr_tokens, attr_valid = self._attributes(attributes, attribute_valid, batch)
        tokens = [reference.tokens + self.modality.weight[0],
                  structure_tokens + self.modality.weight[1],
                  text.tokens + self.modality.weight[2],
                  attr_tokens + self.modality.weight[3]]
        masks = [reference.valid, structure.valid, text.valid, attr_valid]
        if lead_geometry is not None:
            if lead_geometry.shape != (batch, self.config.channels, 6):
                raise ValueError("fusion geometry must be [B,C,6]")
            tokens.append(self.geometry(lead_geometry.to(reference.tokens.dtype)) + self.modality.weight[4])
            masks.append(torch.ones(lead_geometry.shape[:2], device=lead_geometry.device, dtype=torch.bool))
        tokens = torch.cat(tokens, dim=1)
        valid = torch.cat(masks, dim=1)
        tokens = torch.where(valid[..., None], tokens, 0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = torch.where(valid[..., None], self.norm(tokens), 0)
        return FusedCondition(tokens=tokens, valid=valid,
                              global_vector=masked_mean(tokens, valid, 1),
                              structure_scales=structure_scales,
                              structure_scale_valid=structure.scale_valid)
