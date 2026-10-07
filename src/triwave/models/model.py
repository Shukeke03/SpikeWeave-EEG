"""Explicitly separate target-only alignment and inference-only generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triwave.config import ModelConfig
from triwave.types import AlignmentOutput, ConditionBatch, FusedCondition

from .denoiser import WaveDiT
from .encoders import StructureEncoder, TemporalLeadEncoder, TextEncoder, check_geometry
from .fusion import ConditionFusion


class TriWaveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.band_lengths = (config.samples // 2 ** config.wavelet_levels,
                             *(config.samples // 2 ** level for level in range(config.wavelet_levels, 0, -1)))
        self.waveform_encoder = TemporalLeadEncoder(config)
        self.structure_encoder = StructureEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.fusion = ConditionFusion(config)
        self.denoiser = WaveDiT(config, self.band_lengths)

    def encode_conditions(self, conditions: ConditionBatch, drop_text: bool = False,
                          drop_structure: bool = False, drop_reference: bool = False,
                          drop_attributes: bool = False) -> FusedCondition:
        """Accept only legal condition fields; never accepts a clean generation target.

        Drop flags replace complete branch outputs before all fusion/FiLM paths.
        Mandatory measurement geometry remains available in the unconditional arm.
        """
        batch = conditions.batch_size
        check_geometry(conditions.lead_geometry, batch, self.config.channels)
        if conditions.reference_waveform is None or drop_reference:
            reference = self.waveform_encoder.null_output(batch)
        else:
            reference = self.waveform_encoder(conditions.reference_waveform,
                                              conditions.lead_geometry, conditions.reference_valid)
        if conditions.structure is None or drop_structure:
            structure = self.structure_encoder.null_output(batch)
        else:
            structure = self.structure_encoder(conditions.structure, conditions.structure_valid,
                                                conditions.lead_geometry)
        if conditions.input_ids is None or drop_text:
            text = self.text_encoder.null_output(batch)
        else:
            text = self.text_encoder(conditions.input_ids, conditions.text_valid)
        values = None if drop_attributes else conditions.attribute_values
        valid = None if drop_attributes else conditions.attribute_valid
        return self.fusion(reference, structure, text, values, valid, conditions.lead_geometry)

    def alignment(self, target_waveform: Tensor, target_structure: Tensor,
                  target_structure_valid: Tensor | None, input_ids: Tensor | None,
                  text_valid: Tensor | None, lead_geometry: Tensor,
                  lead_valid: Tensor | None = None) -> AlignmentOutput:
        """Three independent projections before text FiLM or multimodal fusion."""
        eeg = self.waveform_encoder(target_waveform, lead_geometry, lead_valid)
        if target_structure_valid is None:
            target_structure_valid = torch.ones_like(target_structure, dtype=torch.bool)
        if lead_valid is not None:
            if lead_valid.ndim == 3:
                lead_presence = lead_valid.any(-1)
            else:
                lead_presence = lead_valid
            target_structure_valid = target_structure_valid & lead_presence[..., None, None]
        structure = self.structure_encoder(target_structure, target_structure_valid, lead_geometry)
        text = (self.text_encoder.null_output(target_waveform.shape[0]) if input_ids is None
                else self.text_encoder(input_ids, text_valid))
        eeg_valid = (torch.ones(target_waveform.shape[0], dtype=torch.bool, device=target_waveform.device)
                     if lead_valid is None else lead_valid.flatten(1).any(-1))
        if input_ids is None:
            text_presence = torch.zeros(target_waveform.shape[0], dtype=torch.bool,
                                        device=target_waveform.device)
        else:
            text_presence = (torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                             if text_valid is None else text_valid.any(-1))
        return AlignmentOutput(eeg=eeg.pooled, structure=structure.pooled, text=text.pooled,
                               eeg_valid=eeg_valid,
                               structure_valid=target_structure_valid.flatten(1).any(-1),
                               text_valid=text_presence)

    def forward(self, noisy_coefficients: Tensor, time: Tensor,
                conditions: ConditionBatch | FusedCondition,
                lead_geometry: Tensor | None = None, **drop_flags: bool) -> Tensor:
        if isinstance(conditions, ConditionBatch):
            condition = self.encode_conditions(conditions, **drop_flags)
            lead_geometry = conditions.lead_geometry
        elif isinstance(conditions, FusedCondition):
            if drop_flags:
                raise ValueError("apply modality dropout before fusion")
            condition = conditions
        else:
            raise TypeError("generation accepts a ConditionBatch or FusedCondition, not target tensors")
        if lead_geometry is None:
            raise ValueError("generation requires explicit lead geometry")
        return self.denoiser(noisy_coefficients, time, condition, lead_geometry)
