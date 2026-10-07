"""Validated, serializable architecture configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 16
    samples: int = 1024
    sample_rate: float = 256.0
    wavelet: str = "db4"
    wavelet_levels: int = 3
    waveform_patch: int = 16
    coefficient_patch: int = 16
    dim: int = 512
    encoder_depth: int = 8
    heads: int = 8
    structure_base_dim: int = 32
    structure_depths: tuple[int, ...] = (2, 2, 2)
    structure_window: int = 4
    structure_frequency_patch: int = 4
    structure_time_patch: int = 8
    structure_frequency_bins: int = 2
    condition_time_bins: int = 4
    cwt_frequencies: int = 64
    cwt_times: int = 128
    cwt_min_frequency: float = 2.0
    cwt_max_frequency: float = 80.0
    text_backend: str = "offline"
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_local_files_only: bool = True
    freeze_text: bool = True
    text_vocab_size: int = 4096
    text_max_length: int = 256
    text_depth: int = 2
    fusion_depth: int = 6
    attribute_dim: int = 8
    attribute_schema: tuple[str, ...] = (
        "spectral_centroid_hz",
        "rms_uV",
        "std_uV",
        "mean_uV",
        "delta_fraction",
        "theta_fraction",
        "alpha_fraction",
        "beta_fraction",
        "gamma_fraction",
        "line_length_uV",
        "delta_log_power_uV2",
        "theta_log_power_uV2",
        "alpha_log_power_uV2",
        "beta_log_power_uV2",
        "gamma_log_power_uV2",
        "peak_abs_uV",
    )
    denoiser_dim: int = 1024
    denoiser_heads: int = 16
    denoiser_depth: int = 24
    dropout: float = 0.0
    projection_dim: int = 512

    lead_names: tuple[str, ...] = (
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
    )
    condition_dim: int = 768
    condition_heads: int = 12
    projection_hidden_dim: int = 1024
    snn_chunk_sizes: tuple[int, ...] = (4, 4, 4, 4, 8, 8, 8, 8)
    snn_drop_path: float = 0.1
    representation_dropout: float = 0.1
    frontend_backend: str = "offline"
    frontend_test_fixture: bool = False
    biomedclip_checkpoint: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomedclip_local_path: str | None = None
    frontend_text_config_path: str | None = None
    visual_depth: int = 12
    visual_heads: int = 12
    visual_patch: int = 16
    image_summary_tokens: int = 8
    montage_mode: str = "audit"
    montage_residual_tolerance: float = 0.01

    def validate(self) -> None:
        for name in (
            "channels",
            "samples",
            "wavelet_levels",
            "waveform_patch",
            "coefficient_patch",
            "dim",
            "heads",
            "denoiser_dim",
            "denoiser_heads",
            "cwt_frequencies",
            "cwt_times",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.samples % (2**self.wavelet_levels):
            raise ValueError("samples must be divisible by 2**wavelet_levels")
        if self.samples % self.waveform_patch:
            raise ValueError("samples must be divisible by waveform_patch")
        if (self.samples // (2**self.wavelet_levels)) % self.coefficient_patch:
            raise ValueError("coarsest DWT band must be divisible by coefficient_patch")
        if self.dim % self.heads or self.denoiser_dim % self.denoiser_heads:
            raise ValueError("attention width must be divisible by number of heads")
        if not 0 < self.cwt_min_frequency < self.cwt_max_frequency < self.sample_rate / 2:
            raise ValueError("CWT frequencies must lie strictly between zero and Nyquist")
        if self.condition_dim % self.condition_heads:
            raise ValueError("condition_dim must be divisible by condition_heads")
        if self.frontend_backend not in {"offline", "biomedclip"}:
            raise ValueError("frontend_backend must be offline or biomedclip")
        if self.montage_mode not in {"audit", "hard", "off"}:
            raise ValueError("montage_mode must be audit, hard or off")
        if self.montage_residual_tolerance <= 0:
            raise ValueError("montage residual tolerance must be positive")
        if self.cwt_frequencies % self.visual_patch or self.cwt_times % self.visual_patch:
            raise ValueError("CWT dimensions must be divisible by visual_patch")
        if any(size <= 0 for size in self.snn_chunk_sizes):
            raise ValueError("SNN chunks must be positive")
        if self.text_backend not in {"offline", "clinicalbert"}:
            raise ValueError("text_backend must be offline or clinicalbert")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open() as handle:
            content = yaml.safe_load(handle) or {}
        if "model" in content:
            content = content["model"]
        for field in ("structure_depths", "snn_chunk_sizes", "lead_names", "attribute_schema"):
            if field in content:
                content[field] = tuple(content[field])
        obj = cls(**content)
        obj.validate()
        return obj

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls) -> ModelConfig:
        return cls(
            channels=4,
            samples=128,
            sample_rate=128.0,
            waveform_patch=8,
            coefficient_patch=8,
            dim=32,
            encoder_depth=1,
            heads=4,
            structure_base_dim=8,
            structure_depths=(1, 1, 1),
            structure_frequency_patch=2,
            structure_time_patch=2,
            cwt_frequencies=16,
            cwt_times=32,
            cwt_max_frequency=48.0,
            text_max_length=16,
            text_depth=1,
            fusion_depth=1,
            denoiser_dim=32,
            denoiser_heads=4,
            denoiser_depth=2,
            projection_dim=32,
            condition_dim=32,
            condition_heads=4,
            projection_hidden_dim=64,
            snn_chunk_sizes=(4,),
            snn_drop_path=0.0,
            representation_dropout=0.0,
            visual_depth=1,
            visual_heads=4,
            visual_patch=4,
            image_summary_tokens=8,
            montage_mode="off",
        )
