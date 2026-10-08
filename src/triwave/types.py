"""Explicit condition-only interfaces prevent accidental clean-target conditioning."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from triwave.signal import CWT_MASK_PROTOCOL


def condition_rows(value, batch_size: int, name: str):
    """Normalize scalar/per-record metadata without silently broadcasting a bad list."""
    if value is None:
        return (None,) * batch_size
    if isinstance(value, (str, dict)):
        return (value,) * batch_size
    if not isinstance(value, (tuple, list)) or len(value) != batch_size:
        raise ValueError(f"{name} must be one value or one value per batch record")
    return tuple(value)


@dataclass
class BranchOutput:
    tokens: Tensor
    valid: Tensor
    pooled: Tensor
    scales: tuple[Tensor, ...] = ()
    scale_valid: tuple[Tensor, ...] = ()
    semantic: Tensor | None = None
    structural: Tensor | None = None
    present: Tensor | None = None
    semantic_valid: Tensor | None = None
    structural_valid: Tensor | None = None
    reconstruction: Tensor | None = None
    patch_mask: Tensor | None = None


@dataclass
class FusedCondition:
    tokens: Tensor
    valid: Tensor
    global_vector: Tensor
    structure_scales: tuple[Tensor, ...] = ()
    structure_scale_valid: tuple[Tensor, ...] = ()


@dataclass
class ConditionBatch:
    """Explicit allowed conditions; target-derived attributes are source-training only.

    Full target images are never conditions. Partial observations retain a
    separately declared available/visible region; only visible pixels reach C.
    ``validate_for_inference`` also excludes source target-derived attributes.
    """

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
    structure_role: str | tuple[str, ...] | None = None
    structure_visible: Tensor | None = None
    structure_provenance: dict | tuple[dict, ...] | None = None
    cwt_mask_protocol: str | tuple[str, ...] | None = None
    attribute_origin: dict | tuple[dict, ...] | None = None
    # Retained across dataclasses.replace/dropout; these are masks, not hidden values.
    structure_available_valid: Tensor | None = None
    reference_provenance: dict | tuple[dict, ...] | None = None

    def __post_init__(self):
        if self.batch_size < 1 or self.lead_geometry.shape[0] != self.batch_size:
            raise ValueError("conditions require a positive matching batch size")
        self._validate_reference()
        roles = condition_rows(self.structure_role, self.batch_size, "structure_role")
        provenances = condition_rows(self.structure_provenance, self.batch_size, "structure_provenance")
        protocols = condition_rows(self.cwt_mask_protocol, self.batch_size, "cwt_mask_protocol")
        if self.structure is not None:
            if self.structure.ndim != 4 or self.structure.shape[0] != self.batch_size:
                raise ValueError("structure must be [B,C,F,T]")
            if self.structure_valid is None or self.structure_valid.shape != self.structure.shape:
                raise ValueError("structure requires an explicit same-shape validity mask")
            if self.structure_valid.dtype != torch.bool:
                raise ValueError("structure validity must be boolean")
            available = self.structure_available_valid
            if available is None:
                available = self.structure_valid.clone()
            if available.dtype != torch.bool or available.shape != self.structure.shape:
                raise ValueError("structure available validity must match structure")
            visible = self.structure_visible
            if visible is None:
                visible = self.structure_valid.clone()
            if visible.dtype != torch.bool or visible.shape != self.structure.shape:
                raise ValueError("structure visible mask must match structure")
            if (visible & ~available).any():
                raise ValueError("visible structure must be a subset of valid available bins")
            if (self.structure_valid & ~available).any():
                raise ValueError("structure validity exceeds the declared available region")
            observed = self.structure_valid & visible
            for i, (role, provenance, protocol) in enumerate(zip(roles, provenances, protocols)):
                if role not in {"none", "reference_condition", "partial_observation"}:
                    raise ValueError("external structure role must be declared; full target teachers are forbidden")
                if protocol != CWT_MASK_PROTOCOL:
                    raise ValueError("external CWT needs the cwt-all-contribution-v2 mask protocol")
                if role == "none":
                    if observed[i].any():
                        raise ValueError("observed structure needs an explicit condition role/provenance")
                    continue
                if not observed[i].any():
                    # Declared empty/missing modalities and CFG drops carry no
                    # value information. Preserve their audit metadata safely.
                    continue
                if not isinstance(provenance, dict) or not provenance.get("origin_id"):
                    raise ValueError("structure conditions need declared origin_id provenance")
                if provenance.get("inference_available") is not True:
                    raise ValueError("structure is not available under the declared inference protocol")
                if role == "reference_condition" and provenance.get("target_derived") is not False:
                    raise ValueError("reference structure must be independent of the target")
                if role == "partial_observation":
                    if (provenance.get("target_derived") is not True
                            or provenance.get("protocol") != "observation-conditioned"):
                        raise ValueError("partial target observations need an observation-conditioned declaration")
                    if observed[i].any() and observed[i].sum() >= available[i].sum():
                        raise ValueError("full target structure is forbidden; partial observations must hide valid bins")
            self.structure_available_valid = available
            self.structure_visible = visible
            self.structure_valid = observed
            self.structure = torch.where(observed, self.structure, 0)
        elif any(role not in {None, "none"} for role in roles):
            raise ValueError("structure role requires an actual structure tensor")
        self.structure_role = tuple("none" if role is None else role for role in roles)
        self.structure_provenance = provenances
        origins = condition_rows(self.attribute_origin, self.batch_size, "attribute_origin")
        if self.attribute_values is not None:
            if (self.attribute_values.ndim != 2 or self.attribute_values.shape[0] != self.batch_size
                    or self.attribute_valid is None or self.attribute_valid.shape != self.attribute_values.shape
                    or self.attribute_valid.dtype != torch.bool):
                raise ValueError("numeric conditions require [B,A] values and boolean masks")
            for i, origin in enumerate(origins):
                if not self.attribute_valid[i].any():
                    continue
                if not isinstance(origin, dict) or not origin.get("origin_id"):
                    raise ValueError("supplied numeric attributes need explicit origin provenance")
                if origin.get("target_derived") is True:
                    if origin.get("source_role") != "train":
                        raise ValueError("target-derived attributes are source/train teachers only")
                elif (origin.get("target_derived") is not False
                      or origin.get("inference_available") is not True):
                    raise ValueError("numeric conditions must be available user/reference/support inputs")
            self.attribute_values = torch.where(self.attribute_valid, self.attribute_values, 0)
        elif self.attribute_valid is not None and self.attribute_valid.any():
            raise ValueError("numeric validity requires actual attribute values")
        self.attribute_origin = origins

    def _validate_reference(self):
        """Reference EEG provenance is independent of all image/attribute branches."""
        origins = condition_rows(self.reference_provenance, self.batch_size, "reference_provenance")
        reference = self.reference_waveform
        if reference is None:
            if self.reference_valid is not None and self.reference_valid.any():
                raise ValueError("valid reference mask requires an actual reference waveform")
            self.reference_provenance = origins
            return
        if (reference.ndim != 3 or reference.shape[0] != self.batch_size
                or reference.shape[1] != self.lead_geometry.shape[1]):
            raise ValueError("reference waveform must match [B,C,L] measurement dimensions")
        mask = self.reference_valid
        if mask is None:
            mask = torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
            self.reference_valid = mask
        if mask.dtype != torch.bool or mask.shape not in {reference.shape[:2], reference.shape}:
            raise ValueError("reference validity must be boolean [B,C] or [B,C,L]")
        observed = mask[..., None].expand_as(reference) if mask.ndim == 2 else mask
        for i, origin in enumerate(origins):
            if not observed[i].any():
                continue
            if (not isinstance(origin, dict) or not origin.get("origin_id")
                    or origin.get("target_derived") is not False
                    or origin.get("inference_available") is not True):
                raise ValueError("valid reference EEG requires independent inference-available reference provenance")
            protocol = origin.get("protocol", "source-only")
            if protocol not in {"source-only", "few-shot-support"}:
                raise ValueError("reference EEG needs a declared source-only or few-shot-support protocol")
            if protocol == "few-shot-support":
                budget = origin.get("support_budget")
                ids = budget.get("support_ids") if isinstance(budget, dict) else None
                if (not isinstance(ids, (list, tuple)) or origin["origin_id"] not in ids
                        or origin.get("context_overlap") is not False):
                    raise ValueError("few-shot reference needs an audited support budget and isolated context")
        self.reference_provenance = origins
        self.reference_waveform = torch.where(observed, reference, 0)

    def validate_for_inference(self):
        self._validate_reference()
        for i, origin in enumerate(self.attribute_origin):
            if (self.attribute_valid is not None and self.attribute_valid[i].any()
                    and origin.get("target_derived") is True):
                raise ValueError("source target-derived teacher attributes cannot be deployed as query conditions")
        return self

    def to(self, device: str) -> ConditionBatch:
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )


@dataclass
class AlignmentOutput:
    eeg: Tensor
    eeg_structure: Tensor
    joint: Tensor
    text: Tensor
    image: Tensor
    eeg_valid: Tensor
    joint_valid: Tensor
    text_valid: Tensor
    image_valid: Tensor

    @property
    def structure(self) -> Tensor:
        return self.image

    @property
    def structure_valid(self) -> Tensor:
        return self.image_valid
