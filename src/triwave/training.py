"""Executable alignment + conditioned generation training, without target conditioning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from triwave.config import ModelConfig
from triwave.diffusion import GaussianDiffusion, flow_pair, sample_flow
from triwave.losses import alignment_loss, signal_regularizers
from triwave.models import TriWaveModel
from triwave.signal import MorletCWT, WaveletTransform
from triwave.types import ConditionBatch


@dataclass
class TrainingConfig:
    objective: str = "diffusion"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    alignment_weight: float = 0.1
    psd_weight: float = 0.01
    topology_weight: float = 0.01
    wavelet_weight: float = 0.01
    condition_dropout: float = 0.1
    gradient_clip: float = 1.0
    diffusion_timesteps: int = 1000
    mixed_precision: bool = False

    def validate(self) -> None:
        if self.objective not in {"diffusion", "flow"}:
            raise ValueError("objective must be diffusion or flow")
        if not 0 <= self.condition_dropout <= 1:
            raise ValueError("condition_dropout must lie in [0,1]")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")


class Trainer:
    def __init__(self, config: ModelConfig, training: TrainingConfig,
                 device: torch.device | str = "cpu"):
        config.validate()
        training.validate()
        self.config, self.training = config, training
        self.device = torch.device(device)
        self.model = TriWaveModel(config).to(self.device)
        self.wavelet = WaveletTransform(config.samples, config.wavelet_levels,
                                        config.wavelet).to(self.device)
        self.cwt = MorletCWT(config.sample_rate, config.cwt_frequencies,
                            config.cwt_min_frequency, config.cwt_max_frequency,
                            config.cwt_times).to(self.device)
        self.diffusion = GaussianDiffusion(training.diffusion_timesteps).to(self.device)
        self.optimizer = torch.optim.AdamW((p for p in self.model.parameters()
                                           if p.requires_grad), lr=training.learning_rate,
                                          weight_decay=training.weight_decay)
        self.step = 0
        self.normalization: dict | None = None

    def move_batch(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def conditions(self, batch: dict) -> ConditionBatch:
        """Use independently selected support EEG, or explicitly supplied structure only."""
        reference = batch.get("reference_waveform")
        structure, structure_valid = batch.get("structure"), batch.get("structure_valid")
        if structure is None and reference is not None:
            structure, structure_valid = self.cwt(reference, batch.get("reference_valid"))
        return ConditionBatch(batch_size=len(batch["lead_geometry"]),
                              lead_geometry=batch["lead_geometry"],
                              reference_waveform=reference,
                              reference_valid=batch.get("reference_valid"),
                              structure=structure, structure_valid=structure_valid,
                              input_ids=batch.get("input_ids"), text_valid=batch.get("text_valid"),
                              attribute_values=batch.get("attribute_values"),
                              attribute_valid=batch.get("attribute_valid"))

    def train_step(self, batch: dict, *, alignment_only: bool = False) -> dict[str, float]:
        self.model.train()
        batch = self.move_batch(batch)
        target = batch["target_waveform"].float()
        valid = batch.get("target_valid")
        if valid is not None:
            if not alignment_only and not bool(valid.all()):
                raise ValueError("generation training currently requires complete target leads/time; "
                                 "missing reference conditions remain supported")
            mask = valid[..., None] if valid.ndim == 2 else valid
            target = torch.where(mask, target, 0)
        coefficients = self.wavelet.encode(target)
        self.optimizer.zero_grad(set_to_none=True)
        # CWT and signed DWT stay float32; Transformer blocks may use CUDA bf16.
        condition_batch = self.conditions(batch)
        target_structure, target_structure_valid = self.cwt(target, valid)
        enabled_amp = self.training.mixed_precision and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=enabled_amp):
            aligned = self.model.alignment(target, target_structure, target_structure_valid,
                                           batch.get("input_ids"), batch.get("text_valid"),
                                           batch["lead_geometry"], valid)
            text_groups = None
            if batch.get("input_ids") is not None:
                _, text_groups = torch.unique(batch["input_ids"], dim=0, return_inverse=True)
            representation_loss = alignment_loss(aligned, text_groups=text_groups)
            if alignment_only:
                loss = representation_loss
                generation_loss = loss * 0
                physics = {"psd": loss * 0, "topology": loss * 0, "wavelet": loss * 0}
            else:
                drop_all = torch.rand((), device=self.device).item() < self.training.condition_dropout
                condition = self.model.encode_conditions(
                    condition_batch, drop_text=drop_all, drop_structure=drop_all,
                    drop_reference=drop_all, drop_attributes=drop_all)
                epsilon = torch.randn_like(coefficients)
                if self.training.objective == "diffusion":
                    indices = torch.randint(self.diffusion.timesteps, (len(target),),
                                            device=self.device)
                    noisy, desired = self.diffusion.noise(coefficients, epsilon, indices)
                    time = self.diffusion.normalized_time(indices)
                else:
                    time = torch.rand(len(target), device=self.device)
                    noisy, desired = flow_pair(coefficients, epsilon, time)
                predicted = self.model.denoiser(noisy, time, condition, batch["lead_geometry"])
                if valid is None:
                    generation_loss = F.mse_loss(predicted.float(), desired)
                else:
                    # Missing sample intervals cannot be localized after DWT. Train complete leads.
                    lead_valid = valid if valid.ndim == 2 else valid.all(dim=-1)
                    squared = (predicted.float() - desired).square()
                    generation_loss = (squared * lead_valid[..., None]).sum() / (
                        lead_valid.sum() * coefficients.shape[-1]).clamp_min(1)
                if self.training.objective == "diffusion":
                    clean_prediction = self.diffusion.clean_from_v(noisy, predicted.float(), indices)
                    alpha, _ = self.diffusion.factors(indices, noisy)
                    reliability = alpha.square().flatten(1).mean(-1)
                else:
                    clean_prediction = noisy - time[:, None, None] * predicted.float()
                    reliability = (1 - time).square()
                waveform_prediction = self.wavelet.decode(clean_prediction)
                physics = signal_regularizers(waveform_prediction, target, valid, reduction="none")
                band_losses = []
                for estimate, truth in zip(clean_prediction.split(self.wavelet.band_lengths, -1),
                                            coefficients.split(self.wavelet.band_lengths, -1)):
                    error = (estimate - truth).square()
                    if valid is not None:
                        error = error * lead_valid[..., None]
                        error_mean = error.flatten(1).sum(-1) / (
                            lead_valid.sum(-1) * truth.shape[-1]).clamp_min(1)
                    else:
                        error_mean = error.flatten(1).mean(-1)
                    band_losses.append(error_mean / truth.square().flatten(1).mean(-1)
                                       .detach().clamp_min(1))
                physics["wavelet"] = torch.stack(band_losses).mean(0)
                loss = (generation_loss + self.training.alignment_weight * representation_loss
                        + (reliability * (self.training.psd_weight * physics["psd"]
                                         + self.training.topology_weight * physics["topology"]
                                         + self.training.wavelet_weight * physics["wavelet"])).mean())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss; update was not applied")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.gradient_clip,
                                              error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        return {"step": self.step, "loss": float(loss.detach()),
                "generation": float(generation_loss.detach()),
                "alignment": float(representation_loss.detach()),
                "psd": float(physics["psd"].detach().mean()),
                "topology": float(physics["topology"].detach().mean()),
                "wavelet": float(physics["wavelet"].detach().mean()),
                "gradient_norm": float(norm)}

    @torch.no_grad()
    def sample(self, conditions: ConditionBatch, steps: int = 30,
               guidance: float = 1.0, seed: int = 0) -> Tensor:
        if guidance < 0:
            raise ValueError("guidance must be non-negative")
        self.model.eval()
        conditions = conditions.to(str(self.device))
        conditional = self.model.encode_conditions(conditions)
        unconditional = None
        if guidance != 1:
            unconditional = self.model.encode_conditions(
                conditions, drop_text=True, drop_structure=True,
                drop_reference=True, drop_attributes=True)

        def predict(noisy: Tensor, time: Tensor) -> Tensor:
            result = self.model.denoiser(noisy, time, conditional, conditions.lead_geometry)
            if unconditional is not None:
                baseline = self.model.denoiser(noisy, time, unconditional, conditions.lead_geometry)
                result = baseline + guidance * (result - baseline)
            return result

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(conditions.batch_size, self.config.channels, self.config.samples,
                            generator=generator, device=self.device)
        if self.training.objective == "diffusion":
            coefficients = self.diffusion.sample(predict, noise, steps)
        else:
            coefficients = sample_flow(predict, noise, steps)
        generated = self.wavelet.decode(coefficients)
        if not torch.isfinite(generated).all():
            raise FloatingPointError("non-finite generated EEG")
        return generated

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format_version": 1, "model_config": self.config.to_dict(),
                    "training_config": asdict(self.training), "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "step": self.step,
                    "normalization": self.normalization,
                    "torch_rng": torch.get_rng_state()}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu",
             resume_optimizer: bool = False) -> Trainer:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        trainer = cls(ModelConfig(**checkpoint["model_config"]),
                      TrainingConfig(**checkpoint["training_config"]), device)
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.step = checkpoint["step"]
        trainer.normalization = checkpoint.get("normalization")
        if resume_optimizer:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
        return trainer


def append_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")
