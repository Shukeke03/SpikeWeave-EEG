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
from triwave.attributes import ATTRIBUTE_PROTOCOL, ATTRIBUTE_SCHEMA
from triwave.config import ModelConfig
from triwave.diffusion import VPCleanDiffusion
from triwave.losses import CONSTRAINT_WEIGHTS, attribute_control_loss, clean_constraints
from triwave.models import SpikeWeaveModel
from triwave.signal import (
    CWT_MASK_PROTOCOL,
    CWT_NUMERIC_PROTOCOL,
    MorletCWT,
    WaveletTransform,
    strict_patch_valid,
)
from triwave.types import ConditionBatch, condition_rows


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
        self.protocol_migration = None
        self.run_provenance = {}
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

    @staticmethod
    def condition_protocols():
        return {"cwt_numeric": CWT_NUMERIC_PROTOCOL, "cwt_mask": CWT_MASK_PROTOCOL,
                "attributes": ATTRIBUTE_PROTOCOL, "structure": "role-visible-v2",
                "attribute_conditioning": "source-teacher-explicit-v2",
                "reference": "independent-reference-provenance-v2"}

    @staticmethod
    def _assert_source_training_targets(batch):
        b = len(batch["lead_geometry"])
        rows = condition_rows(batch.get("metadata"), b, "metadata")
        splits = condition_rows(batch.get("split"), b, "split")
        roles = condition_rows(batch.get("role"), b, "role")
        for i, row in enumerate(rows):
            declarations = [(splits[i], roles[i])]
            if isinstance(row, dict):
                declarations.append((row.get("split", row.get("target_split")),
                                     row.get("role", row.get("target_role"))))
            if any((split is not None and split != "train") or (role is not None and role != "source")
                   for split, role in declarations):
                raise ValueError("train_step only accepts source/train targets; query/support need a separate adaptation protocol")

    @staticmethod
    def _reference_provenance(batch):
        """Resolve each EEG reference origin even when an unrelated image is supplied."""
        b = len(batch["lead_geometry"])
        explicit = condition_rows(batch.get("reference_provenance", batch.get("reference_origin")),
                                  b, "reference_provenance")
        reference = batch.get("reference_waveform")
        if reference is None:
            return explicit
        valid = batch.get("reference_valid")
        if valid is not None and (valid.dtype != torch.bool or valid.shape not in {reference.shape[:2], reference.shape}):
            raise ValueError("reference validity must be boolean [B,C] or [B,C,L]")
        active = (torch.ones(b, dtype=torch.bool, device=reference.device) if valid is None
                  else valid.bool().flatten(1).any(-1))
        rows = condition_rows(batch.get("metadata"), b, "metadata")
        origins = []
        for i, (origin, row) in enumerate(zip(explicit, rows)):
            if origin is not None or not active[i]:
                origins.append(origin)
                continue
            audit = row.get("reference_audit", {}) if isinstance(row, dict) else {}
            origin_id = audit.get("reference_id") or audit.get("reference_record")
            source_ok = (audit.get("reference_split") == "train"
                         and audit.get("reference_role", "source" if audit.get("reference_record") else None) == "source")
            support_ok = audit.get("protocol") == "few-shot-support" and audit.get("reference_role") == "support"
            if not origin_id or audit.get("context_overlap") is not False or not (source_ok or support_ok):
                raise ValueError("reference conditions need explicit independent source provenance")
            if not audit.get("reference_id"):
                origin_id = f"{origin_id}:{audit.get('reference_index', -1)}"
            origin = {"origin_id": origin_id, "target_derived": False, "inference_available": True,
                      "protocol": "few-shot-support" if support_ok else "source-only",
                      "context_overlap": False, "reference_role": "support" if support_ok else "source"}
            if support_ok:
                origin["support_budget"] = audit.get("support_budget", row.get("support_budget"))
            origins.append(origin)
        return tuple(origins)

    @staticmethod
    def _source_teacher_origin(batch, index, origin):
        if (not isinstance(origin, dict) or origin.get("target_derived") is not True
                or origin.get("source_role") != "train"):
            raise ValueError("teacher attributes require explicit source/train target provenance")
        metadata = condition_rows(batch.get("metadata"), len(batch["lead_geometry"]), "metadata")[index]
        if isinstance(metadata, dict) and (metadata.get("split", "train") != "train"
                or metadata.get("role", "source") != "source"):
            raise ValueError("query/support targets cannot supply source/train teacher attributes")
        return origin

    def conditions(self, batch, *, allow_teacher_attributes=False):
        """Build allowed conditions; source target-derived attributes need an explicit training call."""
        if allow_teacher_attributes:
            self._assert_source_training_targets(batch)
        b = len(batch["lead_geometry"])
        reference = batch.get("reference_waveform")
        reference_provenance = self._reference_provenance(batch)
        structure, sv = batch.get("structure"), batch.get("structure_valid")
        role, visible = batch.get("structure_role"), batch.get("structure_visible")
        provenance, protocol = batch.get("structure_provenance"), batch.get("cwt_mask_protocol")
        if structure is None and reference is not None:
            structure, sv = self.cwt(reference, batch.get("reference_valid"))
            role, visible, protocol = "reference_condition", sv.clone(), CWT_MASK_PROTOCOL
            provenance = reference_provenance
        values = batch.get("condition_attributes", batch.get("attribute_values"))
        av = batch.get("condition_attribute_valid", batch.get("attribute_valid"))
        origins = condition_rows(batch.get("attribute_origin"), b, "attribute_origin")
        if values is not None:
            if av is None or av.shape != values.shape:
                raise ValueError("condition attributes require explicit matching validity")
            values, av = values.clone(), av.clone().bool()
            if values.shape[1] != self.config.attribute_dim:
                raise ValueError("condition attribute width must match the model schema")
            for i, origin in enumerate(origins):
                if av[i].any() and isinstance(origin, dict) and origin.get("target_derived") is True:
                    if not allow_teacher_attributes:
                        raise ValueError("source teacher attributes are training-only, not external query conditions")
                    self._source_teacher_origin(batch, i, origin)
        teacher = batch.get("teacher_attributes")
        teacher_valid = batch.get("teacher_attribute_valid")
        if allow_teacher_attributes and teacher is not None:
            if teacher_valid is None or teacher.shape != teacher_valid.shape:
                raise ValueError("teacher attributes require explicit matching validity")
            if values is None:
                values = teacher.new_zeros(b, self.config.attribute_dim)
                av = torch.zeros_like(values, dtype=torch.bool)
            teacher_origins = condition_rows(batch.get("teacher_attribute_origin", batch.get("attribute_origin")),
                                            b, "teacher_attribute_origin")
            origins = list(origins)
            for i, origin in enumerate(teacher_origins):
                if av[i].any() or not teacher_valid[i].any():
                    continue
                self._source_teacher_origin(batch, i, origin)
                for j, name in enumerate(self.config.attribute_schema[:self.config.attribute_dim]):
                    k = ATTRIBUTE_SCHEMA.index(name)
                    if k < teacher.shape[1]:
                        values[i, j], av[i, j] = teacher[i, k], teacher_valid[i, k]
                origins[i] = origin
        return ConditionBatch(
            batch_size=b, lead_geometry=batch["lead_geometry"], reference_waveform=reference,
            reference_valid=batch.get("reference_valid"), structure=structure, structure_valid=sv,
            input_ids=batch.get("input_ids"), text_valid=batch.get("text_valid"),
            attribute_values=values, attribute_valid=av, structure_role=role, structure_visible=visible,
            structure_provenance=provenance, cwt_mask_protocol=protocol, attribute_origin=origins,
            reference_provenance=reference_provenance,
        )

    def _drop_conditions(self, condition):
        b = condition.batch_size
        u = torch.rand(b, device=self.device)
        # joint40%, text40%, image10%, all-null10%. Null drops attributes/ref too.
        keep_text, keep_image, nonnull = u < 0.8, (u < 0.4) | ((u >= 0.8) & (u < 0.9)), u < 0.9
        keep_reference = nonnull & (torch.rand(b, device=self.device) < self.training.reference_probability)

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
        attrs, av = condition.attribute_values, condition.attribute_valid
        if attrs is not None:
            available = av.bool()
            keep_attributes = nonnull & (torch.rand(b, device=self.device) < 0.5)
            width = available.sum(-1).clamp(max=4)
            requested = (torch.rand(b, device=self.device) * width).long() + 1
            scores = torch.rand(attrs.shape, device=self.device).masked_fill(~available, torch.inf)
            ranks = scores.argsort(-1).argsort(-1)
            av = available & (ranks < requested[:, None]) & keep_attributes[:, None]
            attrs = torch.where(av, attrs, 0)
        visible = None if condition.structure_visible is None else condition.structure_visible & iv
        return replace(
            condition,
            input_ids=text,
            text_valid=tv,
            structure=image,
            structure_valid=iv,
            structure_visible=visible,
            reference_waveform=ref,
            reference_valid=rv,
            attribute_values=attrs,
            attribute_valid=av,
        )

    def structure_control(self, condition):
        """Only visible target partial observations are pointwise control truth."""
        if condition.structure is None:
            return None, None
        partial = torch.tensor([role == "partial_observation" for role in condition.structure_role],
                               device=condition.structure.device, dtype=torch.bool)
        valid = condition.structure_valid & condition.structure_visible & partial[:, None, None, None]
        # Incomplete patches do not enter C, so their pixels cannot impose a
        # control that the generator never received through the image tokens.
        p = self.config.visual_patch
        valid = strict_patch_valid(valid, p).repeat_interleave(p, -2).repeat_interleave(p, -1)
        if not valid.any():
            return None, None
        return condition.structure, valid

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
            full_patch = strict_patch_valid(sv, p).repeat_interleave(p, -2).repeat_interleave(p, -1)
            selected = full_patch & visual_mask.repeat_interleave(p, -2).repeat_interleave(p, -1)
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
        self._assert_source_training_targets(batch)
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
                if basis is not None and not self.hard_audit_approved:
                    raise ValueError("hard montage training requires a passed complete source-dataset audit")
                if montage is not None:
                    audit = montage.audit(target)
                    metrics["montage_max_relative_residual"] = max(audit["relative_residual"])
                    metrics["montage_audit_passed"] = int(audit["passed"])
                clean = coefficients if basis is None else basis.reduce(coefficients)
                time = torch.rand(len(target), device=self.device) * (0.999) + 0.001
                noisy = self.diffusion.noise(clean, torch.randn_like(clean), time)
                given = self._drop_conditions(self.conditions(batch, allow_teacher_attributes=True))
                condition = self.model.encode_conditions(given)
                predicted = self.model.predict_clean(
                    noisy, time, condition, batch["lead_geometry"], basis
                ).float()
                generation = self.diffusion.clean_loss(predicted, clean, time)
                physical = predicted if basis is None else basis.lift(predicted)
                waveform = self.wavelet.decode(physical)
                control, cv = self.structure_control(given)
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
                    given.attribute_values,
                    given.attribute_valid,
                    self.config.attribute_schema,
                    self.config.sample_rate,
                    self.normalization["std"] if self.normalization else 1.0,
                    valid,
                    batch.get("source_max_frequency", 80.0),
                )
                penalty = sum(CONSTRAINT_WEIGHTS[k] * v for k, v in constraints.items())
                loss = loss + generation + (self.diffusion.auxiliary_weight(time) * penalty).mean()
                metrics["generation"] = float(generation.detach())
                metrics["provided_attribute_count"] = (
                    float(given.attribute_valid.sum(-1).float().mean()) if given.attribute_valid is not None else 0.0
                )
                metrics["provided_structure_control_bins"] = int(cv.sum()) if cv is not None else 0
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
        conditions = conditions.to(str(self.device)).validate_for_inference()
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

    @classmethod
    def _checkpoint_protocol_migration(cls, checkpoint, allow_protocol_migration):
        protocols = checkpoint.get("condition_protocols")
        if protocols == cls.condition_protocols():
            return None
        if not allow_protocol_migration:
            raise ValueError("checkpoint CWT mask/attribute protocols differ; explicit protocol migration is required")
        if isinstance(protocols, dict) and protocols.get("cwt_numeric", CWT_NUMERIC_PROTOCOL) != CWT_NUMERIC_PROTOCOL:
            raise ValueError("numeric CWT changes require a separately defined weight migration")
        return {"source": protocols, "destination": cls.condition_protocols(), "weights_only": True}

    def initialize_representation(self, path, *, allow_protocol_migration=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2 or checkpoint.get("active_stage") not in {
            "pretrain",
            "align",
        }:
            raise ValueError(
                "representation initialization requires a v0.2 pretrain/align checkpoint"
            )
        migration = self._checkpoint_protocol_migration(checkpoint, allow_protocol_migration)
        for name in ("eeg_encoder", "condition_encoder"):
            prefix = name + "."
            state = {
                key[len(prefix) :]: value
                for key, value in checkpoint["model"].items()
                if key.startswith(prefix)
            }
            getattr(self.model, name).load_state_dict(state, strict=True)
        self.representation_initialized = True
        if migration is not None:
            self.protocol_migration = migration

    def save(self, path):
        if self.microstep % self.training.gradient_accumulation:
            raise ValueError("save only at accumulation boundaries to preserve optimizer resume")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format_version": 2,
                "condition_protocols": self.condition_protocols(),
                "protocol_migration": self.protocol_migration,
                "run_provenance": copy.deepcopy(self.run_provenance),
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
    def load(cls, path, device="cpu", resume_optimizer=False, *, allow_protocol_migration=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "v0.1 checkpoints cannot initialize v0.2; explicit migration/retraining required"
            )
        migration = cls._checkpoint_protocol_migration(checkpoint, allow_protocol_migration)
        if migration is not None and resume_optimizer:
            raise ValueError("protocol migration loads numeric weights only; optimizer resume is forbidden")
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
        trainer.protocol_migration = migration or checkpoint.get("protocol_migration")
        trainer.run_provenance = copy.deepcopy(checkpoint.get("run_provenance", {}))
        if migration is not None:
            trainer.step = trainer.microstep = 0
            trainer.normalization = None
            trainer.hard_audit_approved = False
            trainer.run_provenance = {"weights_only_migration": True,
                                      "parent_run_provenance": trainer.run_provenance}
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
