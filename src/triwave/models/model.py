"""SpikeWeave: exactly two representation branches, plus continuous MCWave."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triwave.config import ModelConfig
from triwave.montage import MontageSubspace
from triwave.types import AlignmentOutput, ConditionBatch, FusedCondition

from .attention import masked_mean
from .denoiser import MCWave
from .encoders import check_geometry
from .snn import SpikingEEGEncoder
from .vision_language import SharedVisionLanguageEncoder


class SpikeWeaveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.band_lengths = (
            config.samples // 2**config.wavelet_levels,
            *(config.samples // 2**level for level in range(config.wavelet_levels, 0, -1)),
        )
        self.eeg_encoder = SpikingEEGEncoder(config)
        self.condition_encoder = SharedVisionLanguageEncoder(config)
        self.geometry_projection = nn.Linear(6, config.condition_dim)
        self.denoiser = MCWave(config, self.band_lengths)

    @property
    def waveform_encoder(self):
        """Legacy access to E; does not register another encoder."""
        return self.eeg_encoder

    def encode_conditions(
        self,
        conditions: ConditionBatch,
        drop_text: bool = False,
        drop_structure: bool = False,
        drop_reference: bool = False,
        drop_attributes: bool = False,
    ) -> FusedCondition:
        check_geometry(conditions.lead_geometry, conditions.batch_size, self.config.channels)
        c = self.condition_encoder(
            None if drop_structure else conditions.structure,
            None if drop_structure else conditions.structure_valid,
            None if drop_text else conditions.input_ids,
            None if drop_text else conditions.text_valid,
            lead_geometry=conditions.lead_geometry,
            attribute_values=None if drop_attributes else conditions.attribute_values,
            attribute_valid=None if drop_attributes else conditions.attribute_valid,
        )
        reference = (
            self.eeg_encoder.null_output(conditions.batch_size)
            if conditions.reference_waveform is None or drop_reference
            else self.eeg_encoder(
                conditions.reference_waveform, conditions.lead_geometry, conditions.reference_valid
            )
        )
        geometry = self.geometry_projection(conditions.lead_geometry.to(c.tokens.dtype))
        tokens = torch.cat((c.tokens, reference.tokens, geometry), 1)
        valid = torch.cat(
            (
                c.valid,
                reference.valid,
                torch.ones(geometry.shape[:2], device=tokens.device, dtype=torch.bool),
            ),
            1,
        )
        return FusedCondition(tokens, valid, masked_mean(tokens, valid, 1))

    def alignment(
        self,
        target_waveform: Tensor,
        target_structure: Tensor,
        target_structure_valid: Tensor | None,
        input_ids: Tensor | None,
        text_valid: Tensor | None,
        lead_geometry: Tensor,
        lead_valid: Tensor | None = None,
    ) -> AlignmentOutput:
        e = self.eeg_encoder(target_waveform, lead_geometry, lead_valid)
        sv = (
            torch.ones_like(target_structure, dtype=torch.bool)
            if target_structure_valid is None
            else target_structure_valid
        )
        if lead_valid is not None:
            lv = lead_valid.any(-1) if lead_valid.ndim == 3 else lead_valid
            sv = sv & lv[..., None, None]
        joint = self.condition_encoder(
            target_structure, sv, input_ids, text_valid, lead_geometry=lead_geometry
        )
        text = self.condition_encoder(
            input_ids=input_ids, text_valid=text_valid, lead_geometry=lead_geometry
        )
        image = self.condition_encoder(target_structure, sv, lead_geometry=lead_geometry)
        return AlignmentOutput(
            e.semantic,
            e.structural,
            joint.semantic,
            text.semantic,
            image.structural,
            e.present,
            joint.semantic_valid & joint.structural_valid,
            text.semantic_valid,
            image.structural_valid,
        )

    def montage(self, device=None) -> MontageSubspace | None:
        if self.config.montage_mode == "off":
            return None
        names = self.config.lead_names
        if len(names) != self.config.channels:
            raise ValueError(
                "montage audit requires configured explicit lead_names matching channel order"
            )
        electrodes = sorted({name for lead in names for name in lead.split("-")})
        if any(len(lead.split("-")) != 2 for lead in names):
            raise ValueError(
                "lead_names must explicitly identify positive-negative electrode pairs"
            )
        endpoints = torch.tensor(
            [[electrodes.index(name) for name in lead.split("-")] for lead in names],
            device=device,
            dtype=torch.long,
        )
        return MontageSubspace.from_endpoints(
            endpoints, max_relative_residual=self.config.montage_residual_tolerance
        ).to(device)

    def predict_clean(
        self,
        state: Tensor,
        time: Tensor,
        condition: FusedCondition,
        lead_geometry: Tensor,
        basis: MontageSubspace | None = None,
    ) -> Tensor:
        physical = state if basis is None else basis.lift(state)
        prediction = self.denoiser(physical, time, condition, lead_geometry)
        return prediction if basis is None else basis.reduce(prediction, require_audit=False)

    def forward(
        self,
        noisy_coefficients: Tensor,
        time: Tensor,
        conditions: ConditionBatch | FusedCondition,
        lead_geometry: Tensor | None = None,
        **drop_flags: bool,
    ) -> Tensor:
        if isinstance(conditions, ConditionBatch):
            fused = self.encode_conditions(conditions, **drop_flags)
            lead_geometry = conditions.lead_geometry
        elif isinstance(conditions, FusedCondition):
            if drop_flags:
                raise ValueError("apply modality dropout before fusion")
            fused = conditions
        else:
            raise TypeError(
                "generation accepts explicit legal conditions, never a clean target tensor"
            )
        if lead_geometry is None:
            raise ValueError("generation requires explicit lead geometry")
        return self.denoiser(noisy_coefficients, time, fused, lead_geometry)


TriWaveModel = SpikeWeaveModel  # import compatibility; old checkpoints are rejected
