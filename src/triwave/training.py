"""Staged two-branch representation and VP clean-prediction training."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from triwave.alignment import fixed_alignment_loss
from triwave.config import ModelConfig
from triwave.diffusion import VPCleanDiffusion
from triwave.losses import CONSTRAINT_WEIGHTS, attribute_control_loss, clean_constraints
from triwave.models import SpikeWeaveModel
from triwave.signal import MorletCWT, WaveletTransform
from triwave.types import ConditionBatch


@dataclass
class TrainingConfig:
    objective: str = "diffusion"
    stage: str = "diffusion"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    mixed_precision: bool = False
    gradient_accumulation: int = 1
    effective_batch_size: int = 256
    warmup_steps: int = 5000
    total_steps: int = 100000
    ema_decay: float = 0.9999
    temperature: float = 0.07
    mask_ratio: float = 0.4
    condition_dropout: float = 0.1
    reference_probability: float = 0.25
    # Accepted only for historical CLI/config compatibility, not v0.2 losses.
    diffusion_timesteps: int = 1000

    def validate(self):
        if self.objective != "diffusion":
            raise ValueError(
                "v0.2 uses VP clean diffusion; legacy CFM remains a separate comparison"
            )
        if self.stage not in {"pretrain", "align", "diffusion", "joint_smoke"}:
            raise ValueError("stage must be pretrain, align, diffusion or joint_smoke")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.gradient_accumulation < 1:
            raise ValueError("invalid optimizer/accumulation settings")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.reference_probability <= 1:
            raise ValueError("invalid EMA/reference settings")
        if self.temperature != 0.07 or self.mask_ratio != 0.4 or self.condition_dropout != 0.1:
            raise ValueError(
                "main protocol fixes temperature=.07, mask_ratio=.4 and all-null probability=.1"
            )


class Trainer:
    def __init__(self, config: ModelConfig, training: TrainingConfig, device="cpu"):
        config.validate()
        training.validate()
        self.config, self.training, self.device = config, training, torch.device(device)
        self.model = SpikeWeaveModel(config).to(self.device)
        self.wavelet = WaveletTransform(config.samples, config.wavelet_levels, config.wavelet).to(
            self.device
        )
        self.cwt = MorletCWT(
            config.sample_rate,
            config.cwt_frequencies,
            config.cwt_min_frequency,
            config.cwt_max_frequency,
            config.cwt_times,
        ).to(self.device)
        self.diffusion = VPCleanDiffusion().to(self.device)
        self.step, self.microstep = 0, 0
        self.normalization = None
        self.hard_audit_approved = False
        self.representation_initialized = False
        self.last_sampler_stats = {}
        self._configure_stage(training.stage)
        self.ema_denoiser = copy.deepcopy(self.model.denoiser).eval().requires_grad_(False)

    def _configure_stage(self, stage):
        self.active_stage = stage
        for name, p in self.model.named_parameters():
            medical_frozen = ".medical." in name
            if stage == "align" and medical_frozen:
                medical_frozen = not any(
                    key in name
                    for key in (
                        "medical.visual.trunk.blocks.10.",
                        "medical.visual.trunk.blocks.11.",
                        "medical.text.transformer.encoder.layer.10.",
                        "medical.text.transformer.encoder.layer.11.",
                    )
                )
            p.requires_grad_(
                not medical_frozen
                and (
                    stage == "joint_smoke"
                    or (
                        stage == "diffusion"
                        and (
                            name.startswith("denoiser.") or name.startswith("geometry_projection.")
                        )
                    )
                    or (stage in {"pretrain", "align"} and not name.startswith("denoiser."))
                )
            )
        decay, no_decay, pretrained = [], [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if ".medical." in name:
                    pretrained.append(p)
                else:
                    (no_decay if p.ndim < 2 or "tau" in name or "norm" in name else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": pretrained, "weight_decay": 0.0, "lr_scale": 0.1},
            ],
            lr=self.training.learning_rate,
            betas=(0.9, 0.95),
        )
        self.optimizer.zero_grad(set_to_none=True)

    def move_batch(self, batch):
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def conditions(self, batch):
        reference = batch.get("reference_waveform")
        structure, sv = batch.get("structure"), batch.get("structure_valid")
        if structure is None and reference is not None:
            structure, sv = self.cwt(reference, batch.get("reference_valid"))
        return ConditionBatch(
            len(batch["lead_geometry"]),
            batch["lead_geometry"],
            reference,
            batch.get("reference_valid"),
            structure,
            sv,
            batch.get("input_ids"),
            batch.get("text_valid"),
            batch.get("attribute_values"),
            batch.get("attribute_valid"),
        )

    def _drop_conditions(self, condition):
        b = condition.batch_size
        u = torch.rand(b, device=self.device)
        # joint40%, text40%, image10%, all-null10%. Null drops attributes/ref too.
        keep_text, keep_image, nonnull = u < 0.8, (u < 0.4) | ((u >= 0.8) & (u < 0.9)), u < 0.9
        keep_reference = nonnull & (torch.rand(b, device=self.device) < 0.25)

        def mask(value, valid, keep):
            if value is None:
                return None, None
            observed = torch.ones_like(value, dtype=torch.bool) if valid is None else valid
            shape = (b,) + (1,) * (observed.ndim - 1)
            observed = observed.bool() & keep.reshape(shape)
            vshape = observed if observed.ndim == value.ndim else observed[..., None]
            return torch.where(vshape, value, 0), observed

        text, tv = mask(condition.input_ids, condition.text_valid, keep_text & nonnull)
        image, iv = mask(condition.structure, condition.structure_valid, keep_image & nonnull)
        ref, rv = mask(condition.reference_waveform, condition.reference_valid, keep_reference)
        attrs, av = mask(condition.attribute_values, condition.attribute_valid, nonnull)
        return replace(
            condition,
            input_ids=text,
            text_valid=tv,
            structure=image,
            structure_valid=iv,
            reference_waveform=ref,
            reference_valid=rv,
            attribute_values=attrs,
            attribute_valid=av,
        )

    def _representation(self, target, valid, batch):
        ts, sv = self.cwt(target, valid)
        if self.active_stage == "pretrain":
            aligned, metrics = target.sum() * 0, {"alignment_valid_items": 0}
        else:
            align = self.model.alignment(
                target,
                ts,
                sv,
                batch.get("input_ids"),
                batch.get("text_valid"),
                batch["lead_geometry"],
                valid,
            )
            aligned, metrics = fixed_alignment_loss(
                align, batch.get("alignment_metadata"), self.training.temperature
            )
        mask = self.model.eeg_encoder.sample_patch_mask(valid, self.training.mask_ratio)
        e = self.model.eeg_encoder(target, batch["lead_geometry"], valid, mask)
        truth = target.reshape_as(e.reconstruction)
        observed = valid.reshape_as(truth) & mask[..., None]
        reconstruction = torch.where(
            observed, e.reconstruction.float() - truth, 0
        ).square().sum() / observed.sum().clamp_min(1)
        estimate = torch.where(mask[..., None], e.reconstruction.float(), truth).reshape_as(target)
        # Differentiable signed/full-window spectral auxiliary, source scalar units.
        spectral = (
            torch.log1p(torch.fft.rfft(estimate).abs().square())
            - torch.log1p(torch.fft.rfft(target).abs().square())
        ).square()
        complete = valid.all(-1)
        spectral_loss = torch.where(complete[..., None], spectral, 0).sum() / (
            complete.sum() * spectral.shape[-1]
        ).clamp_min(1)
        metrics.update(
            masked_patch=float(reconstruction.detach()),
            spectral=float(spectral_loss.detach()),
            alignment=float(aligned.detach()),
        )
        visual_loss = target.sum() * 0
        if self.active_stage == "pretrain":
            p = self.config.visual_patch
            shape = (
                len(target),
                self.config.channels,
                self.config.cwt_frequencies // p,
                self.config.cwt_times // p,
            )
            visual_mask = torch.rand(shape, device=self.device) < 0.4
            estimate = self.model.condition_encoder.numeric_reconstruction(ts, sv, visual_mask)
            selected = sv & visual_mask.repeat_interleave(p, -2).repeat_interleave(p, -1)
            visual_loss = torch.where(
                selected, estimate.float() - ts, 0
            ).square().sum() / selected.sum().clamp_min(1)
        metrics["numeric_visual_reconstruction"] = float(visual_loss.detach())
        return aligned + reconstruction + 0.1 * spectral_loss + visual_loss, metrics

    def train_step(self, batch, *, alignment_only=False):
        if alignment_only and self.active_stage == "diffusion":
            self._configure_stage("align")
        self.model.train()
        if self.active_stage == "diffusion":
            self.model.eeg_encoder.eval()
            self.model.condition_encoder.eval()
        batch = self.move_batch(batch)
        target = batch["target_waveform"].float()
        valid = batch.get("target_valid", torch.ones_like(target, dtype=torch.bool))
        if valid.ndim == 2:
            valid = valid[..., None].expand_as(target)
        if self.active_stage in {"diffusion", "joint_smoke"} and not valid.all():
            raise ValueError(
                "generation requires complete target leads/time; masked reference remains legal"
            )
        target = torch.where(valid, target, 0)
        if not torch.isfinite(target).all():
            raise ValueError("observed target must be finite")
        amp = self.training.mixed_precision and self.device.type == "cuda"
        metrics = {"stage": self.active_stage, "alignment": 0.0, "generation": 0.0}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp):
            loss = target.sum() * 0
            if self.active_stage in {"pretrain", "align", "joint_smoke"}:
                loss, rep_metrics = self._representation(target, valid, batch)
                metrics.update(rep_metrics)
            if self.active_stage in {"diffusion", "joint_smoke"}:
                if (
                    self.normalization
                    and self.normalization.get("scale_protocol") != "source_rms_v2"
                ):
                    raise ValueError(
                        "v0.2 clinical diffusion needs source_rms_v2; refit source scalar first"
                    )
                coefficients = self.wavelet.encode(target)
                montage = self.model.montage(self.device)
                basis = montage if self.config.montage_mode == "hard" else None
                if montage is not None:
                    audit = montage.audit(target)
                    metrics["montage_max_relative_residual"] = max(audit["relative_residual"])
                    metrics["montage_audit_passed"] = int(audit["passed"])
                    if basis is not None and audit["passed"]:
                        self.hard_audit_approved = True
                clean = coefficients if basis is None else basis.reduce(coefficients)
                time = torch.rand(len(target), device=self.device) * (0.999) + 0.001
                noisy = self.diffusion.noise(clean, torch.randn_like(clean), time)
                condition = self.model.encode_conditions(
                    self._drop_conditions(self.conditions(batch))
                )
                predicted = self.model.predict_clean(
                    noisy, time, condition, batch["lead_geometry"], basis
                ).float()
                generation = self.diffusion.clean_loss(predicted, clean, time)
                physical = predicted if basis is None else basis.lift(predicted)
                waveform = self.wavelet.decode(physical)
                control, cv = batch.get("structure"), batch.get("structure_valid")
                control_prediction = None
                if control is not None:
                    control_prediction, cwt_valid = self.cwt(waveform, valid)
                    cv = cwt_valid if cv is None else cv & cwt_valid
                constraints = clean_constraints(
                    waveform,
                    target,
                    physical,
                    coefficients,
                    self.wavelet.band_lengths,
                    valid,
                    int(self.config.sample_rate / 2),
                    control_prediction,
                    control,
                    cv,
                )
                constraints["control"] = constraints["control"] + attribute_control_loss(
                    waveform,
                    batch.get("attribute_values"),
                    batch.get("attribute_valid"),
                    self.config.attribute_schema,
                    self.config.sample_rate,
                    self.normalization["std"] if self.normalization else 1.0,
                )
                penalty = sum(CONSTRAINT_WEIGHTS[k] * v for k, v in constraints.items())
                loss = loss + generation + (self.diffusion.auxiliary_weight(time) * penalty).mean()
                metrics["generation"] = float(generation.detach())
                metrics.update(
                    {f"constraint_{k}": float(v.detach().mean()) for k, v in constraints.items()}
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss; optimizer update not applied")
        (loss / self.training.gradient_accumulation).backward()
        self.microstep += 1
        updated = self.microstep % self.training.gradient_accumulation == 0
        norm = loss.detach() * 0
        if updated:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training.gradient_clip, error_if_nonfinite=True
            )
            warm = min((self.step + 1) / max(self.training.warmup_steps, 1), 1.0)
            progress = max(
                0.0,
                (self.step - self.training.warmup_steps)
                / max(self.training.total_steps - self.training.warmup_steps, 1),
            )
            lr = (
                self.training.learning_rate
                * warm
                * 0.5
                * (1 + math.cos(math.pi * min(progress, 1.0)))
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            with torch.no_grad():
                for ema, p in zip(self.ema_denoiser.parameters(), self.model.denoiser.parameters()):
                    ema.lerp_(p.detach(), 1 - self.training.ema_decay)
        metrics.update(
            step=self.step,
            microstep=self.microstep,
            loss=float(loss.detach()),
            gradient_norm=float(norm),
            optimizer_updated=int(updated),
        )
        return metrics

    @torch.no_grad()
    def sample(self, conditions, steps=50, guidance=2.0, seed=0, use_ema=True):
        if guidance < 0:
            raise ValueError("guidance must be nonnegative")
        self.model.eval()
        conditions = conditions.to(str(self.device))
        conditional = self.model.encode_conditions(conditions)
        null = (
            self.model.encode_conditions(
                conditions,
                drop_text=True,
                drop_structure=True,
                drop_reference=True,
                drop_attributes=True,
            )
            if guidance != 1
            else None
        )
        basis = self.model.montage(self.device) if self.config.montage_mode == "hard" else None
        if basis is not None and not self.hard_audit_approved:
            raise ValueError(
                "hard montage sampling requires a passed real/source training residual audit"
            )
        denoiser = self.ema_denoiser if use_ema else self.model.denoiser

        def predict(state, time):
            physical = state if basis is None else basis.lift(state)
            result = denoiser(physical, time, conditional, conditions.lead_geometry).float()
            if null is not None:
                baseline = denoiser(physical, time, null, conditions.lead_geometry).float()
                result = baseline + guidance * (result - baseline)
            return result if basis is None else basis.reduce(result, require_audit=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            conditions.batch_size,
            self.config.channels if basis is None else basis.rank,
            self.config.samples,
            device=self.device,
            generator=generator,
        )
        coeffs, stats = self.diffusion.sample_heun(predict, noise, steps, return_stats=True)
        stats["network_calls"] = stats["predictor_calls"] * (2 if null is not None else 1)
        self.last_sampler_stats = stats
        generated = self.wavelet.decode(coeffs if basis is None else basis.lift(coeffs))
        if not torch.isfinite(generated).all():
            raise FloatingPointError("nonfinite generated EEG")
        return generated

    def initialize_representation(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2 or checkpoint.get("active_stage") not in {
            "pretrain",
            "align",
        }:
            raise ValueError(
                "representation initialization requires a v0.2 pretrain/align checkpoint"
            )
        for name in ("eeg_encoder", "condition_encoder"):
            prefix = name + "."
            state = {
                key[len(prefix) :]: value
                for key, value in checkpoint["model"].items()
                if key.startswith(prefix)
            }
            getattr(self.model, name).load_state_dict(state, strict=True)
        self.representation_initialized = True

    def save(self, path):
        if self.microstep % self.training.gradient_accumulation:
            raise ValueError("save only at accumulation boundaries to preserve optimizer resume")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format_version": 2,
                "model_config": self.config.to_dict(),
                "training_config": asdict(self.training),
                "active_stage": self.active_stage,
                "hard_audit_approved": self.hard_audit_approved,
                "representation_initialized": self.representation_initialized,
                "model": self.model.state_dict(),
                "ema": self.ema_denoiser.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "microstep": self.microstep,
                "normalization": self.normalization,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            temporary,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu", resume_optimizer=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "v0.1 checkpoints cannot initialize v0.2; explicit migration/retraining required"
            )
        trainer = cls(
            ModelConfig(**checkpoint["model_config"]),
            TrainingConfig(**checkpoint["training_config"]),
            device,
        )
        if trainer.active_stage != checkpoint["active_stage"]:
            trainer._configure_stage(checkpoint["active_stage"])
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.ema_denoiser.load_state_dict(checkpoint["ema"])
        trainer.step, trainer.microstep = checkpoint["step"], checkpoint["microstep"]
        trainer.normalization = checkpoint.get("normalization")
        trainer.hard_audit_approved = checkpoint.get("hard_audit_approved", False)
        trainer.representation_initialized = checkpoint.get("representation_initialized", False)
        if resume_optimizer:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if trainer.device.type == "cuda" and checkpoint["cuda_rng"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        return trainer


def append_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")
