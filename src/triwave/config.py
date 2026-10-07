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
    waveform_patch: int = 32
    coefficient_patch: int = 32
    dim: int = 128
    encoder_depth: int = 4
    heads: int = 4
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
    text_max_length: int = 64
    text_depth: int = 2
    fusion_depth: int = 3
    attribute_dim: int = 8
    denoiser_dim: int = 256
    denoiser_heads: int = 8
    denoiser_depth: int = 8
    dropout: float = 0.0
    projection_dim: int = 128

    def validate(self) -> None:
        for name in ("channels", "samples", "wavelet_levels", "waveform_patch",
                     "coefficient_patch", "dim", "heads", "denoiser_dim",
                     "denoiser_heads", "cwt_frequencies", "cwt_times"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.samples % (2 ** self.wavelet_levels):
            raise ValueError("samples must be divisible by 2**wavelet_levels")
        if self.samples % self.waveform_patch:
            raise ValueError("samples must be divisible by waveform_patch")
        if (self.samples // (2 ** self.wavelet_levels)) % self.coefficient_patch:
            raise ValueError("coarsest DWT band must be divisible by coefficient_patch")
        if self.dim % self.heads or self.denoiser_dim % self.denoiser_heads:
            raise ValueError("attention width must be divisible by number of heads")
        if not 0 < self.cwt_min_frequency < self.cwt_max_frequency < self.sample_rate / 2:
            raise ValueError("CWT frequencies must lie strictly between zero and Nyquist")
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
        if "structure_depths" in content:
            content["structure_depths"] = tuple(content["structure_depths"])
        obj = cls(**content)
        obj.validate()
        return obj

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls) -> ModelConfig:
        return cls(channels=4, samples=128, sample_rate=128.0, waveform_patch=8,
                   coefficient_patch=8, dim=32, encoder_depth=1, heads=4,
                   structure_base_dim=8, structure_depths=(1, 1, 1),
                   structure_frequency_patch=2, structure_time_patch=2,
                   cwt_frequencies=16, cwt_times=32, cwt_max_frequency=48.0,
                   text_max_length=16, text_depth=1, fusion_depth=1,
                   denoiser_dim=32, denoiser_heads=4, denoiser_depth=2,
                   projection_dim=32)
