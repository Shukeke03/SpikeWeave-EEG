"""Explicit condition-only interfaces prevent accidental clean-target conditioning."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from torch import Tensor


@dataclass
class BranchOutput:
    tokens: Tensor
    valid: Tensor
    pooled: Tensor
    scales: tuple[Tensor, ...] = ()
    scale_valid: tuple[Tensor, ...] = ()


@dataclass
class FusedCondition:
    tokens: Tensor
    valid: Tensor
    global_vector: Tensor
    structure_scales: tuple[Tensor, ...] = ()
    structure_scale_valid: tuple[Tensor, ...] = ()


@dataclass
class ConditionBatch:
    """Contains allowed inference inputs only, never the target waveform/view."""

    batch_size: int
    lead_geometry: Tensor  # [B,C,6], positive and negative endpoint coordinates
    reference_waveform: Tensor | None = None
    reference_valid: Tensor | None = None  # [B,C] bool
    structure: Tensor | None = None  # [B,C,F,T]
    structure_valid: Tensor | None = None  # same shape, bool
    input_ids: Tensor | None = None
    text_valid: Tensor | None = None
    attribute_values: Tensor | None = None
    attribute_valid: Tensor | None = None

    def to(self, device: str) -> ConditionBatch:
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), Tensor)})


@dataclass
class AlignmentOutput:
    eeg: Tensor
    structure: Tensor
    text: Tensor
    eeg_valid: Tensor
    structure_valid: Tensor
    text_valid: Tensor
