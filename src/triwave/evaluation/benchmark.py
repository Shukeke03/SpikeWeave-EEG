"""Runnable G0/G1 development diagnostics and a budget-matched small CNN.

Imported generators are explicitly identified by provenance, not silently
implemented under familiar baseline names. This module never trains JET or a
diffusion model, accesses query conditions, or fits statistics to validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from triwave.data.manifest import canonical_patient
from triwave.montage import MontageSubspace
from triwave.signal import WaveletTransform

from .data import EvaluationData, json_digest
from .gates import GateLedger, write_report
from .metrics import (
    frechet_distance,
    nearest_copy_audit,
    patient_average_precision,
    physical_features,
    source_scaled_mmd,
    spectral_ts_features,
)


def _signature(train: EvaluationData, validation: EvaluationData) -> str:
    return json_digest({"train": train.digest, "validation": validation.digest,
                        "protocol": "strict_source_only_macro_patient_AP_v1"})


def _identities(data: EvaluationData, key: str, fallback: np.ndarray) -> set[str]:
    values = data.metadata.get(key) or fallback.tolist()
    return {canonical_patient(p) for p in values} if "patient" in key else set(values)


def _montage_audit(values: np.ndarray, lead_names: list[str]) -> dict:
    if any(len(name.split("-")) != 2 for name in lead_names):
        return {"status": "unavailable", "reason": "lead names are not explicit bipolar endpoints"}
    electrodes = sorted({name for lead in lead_names for name in lead.split("-")})
    endpoints = torch.tensor([[electrodes.index(name) for name in lead.split("-")]
                              for lead in lead_names])
    subspace = MontageSubspace.from_endpoints(endpoints, max_relative_residual=0.01)
    result = subspace.audit(torch.from_numpy(values))
    return result | {"status": "complete", "unit": "uV",
                     "threshold_interpretation": "engineering_screen_not_clinical_tolerance",
                     "hard_projection_enabled": False}


def _real_inputs(train: EvaluationData, validation: EvaluationData) -> list[str]:
    errors = []
    if train.split != "train" or train.role != "source" or validation.split != "validation":
        errors.append("G0/G1 require source/train and real validation only; locked test is prohibited")
    if validation.role not in {"source", "query"}:
        errors.append("validation must be real source/query, never generated or support")
    if train.waveforms.shape[1:] != validation.waveforms.shape[1:]:
        errors.append("train and validation measurement shapes differ")
    if (train.sample_rate != validation.sample_rate
            or train.metadata["lead_names"] != validation.metadata["lead_names"]):
        errors.append("sample rate or explicit lead order differs")
    for key, fallback in (("population_patient_ids", "patient_ids"),
                          ("population_record_ids", "record_ids"),
                          ("population_sample_ids", "sample_ids")):
        left = _identities(train, key, getattr(train, fallback))
        right = _identities(validation, key, getattr(validation, fallback))
        if left & right:
            errors.append(f"train/validation overlap in {key}")
    source_hashes = set(train.metadata.get("population_source_hashes", []))
    if source_hashes & set(validation.metadata.get("population_source_hashes", [])):
        errors.append("train/validation reuse a source file hash")
    for field in ("label_namespace", "label_scope"):
        if not train.metadata.get(field) or train.metadata.get(field) != validation.metadata.get(field):
            errors.append(f"label ontology differs in {field}")
    if validation.metadata.get("natural_distribution") is not True:
        errors.append("validation must explicitly declare unbalanced natural-distribution selection")
    for data in (train, validation):
        if data.metadata.get("query_used_for_fit", False):
            errors.append("query was used to fit a processing/evaluation component")
        if data.metadata.get("contains_target_conditions", False):
            errors.append("blind source-only input contains target-derived conditions")
    return errors


def run_g0(train: EvaluationData, validation: EvaluationData, *,
           output: str | Path | None = None, ledger: GateLedger | None = None,
           numerical_windows: int = 2) -> dict:
    """Audit all loaded identities, with bounded source-only numerical checks."""
    if numerical_windows < 1:
        raise ValueError("G0 requires at least one numerical source window")
    errors = _real_inputs(train, validation)
    numerical = {}
    try:
        transform = WaveletTransform(train.waveforms.shape[-1], 3, "db4")
        values = torch.from_numpy(train.waveforms[:numerical_windows])
        with torch.no_grad():
            residual = transform.decode(transform.encode(values)) - values
        relative = float(residual.norm() / values.norm().clamp_min(1e-8))
        numerical = {"windows": len(values), "DWT_relative_l2": relative,
                     "max_abs_error_uV": float(residual.abs().max()),
                     "numerical_split": "train", "numerical_source_only": True,
                     "montage": _montage_audit(train.waveforms[:numerical_windows],
                                                train.metadata["lead_names"])}
        if relative > 1e-5:
            errors.append("DWT reconstruction error exceeds engineering tolerance")
    except (ValueError, RuntimeError) as exception:
        errors.append(str(exception))
    report = {"stage": "G0", "status": "complete", "data_signature": _signature(train, validation),
              "train_digest": train.digest, "validation_digest": validation.digest,
              "train_counts": train.counts(), "validation_counts": validation.counts(),
              "identity_audit": "complete_loaded_inputs_plus_declared_population_identities",
              "identities": {
                  "train_patients": sorted(_identities(train, "population_patient_ids", train.patient_ids)),
                  "validation_patients": sorted(_identities(validation, "population_patient_ids", validation.patient_ids)),
                  "train_records": sorted(_identities(train, "population_record_ids", train.record_ids)),
                  "validation_records": sorted(_identities(validation, "population_record_ids", validation.record_ids)),
                  "train_source_hashes": train.metadata.get("population_source_hashes", []),
                  "validation_source_hashes": validation.metadata.get("population_source_hashes", [])},
              "population_identity_available": {
                  "train": bool(train.metadata.get("full_identity_available", False)),
                  "validation": bool(validation.metadata.get("full_identity_available", False))},
              "physical_unit": "uV", "numerical": numerical, "errors": errors,
              "gate_passed": not errors, "test_used": False,
              "limitations": ["CWT/montage/full-raw processing audit remains dataset-audit responsibility",
                              "loaded-array audit does not attest original-file completeness"],
              "clinical_validity": "not_established"}
    write_report(output, report)
    if ledger is not None:
        ledger.record(report)
    return report


@dataclass
class G1Config:
    updates: int = 100
    batch_size: int = 16
    seed: int = 0
    learning_rate: float = 1e-3
    synthetic_ratio: float = 0.5
    device: str = "cpu"
    review_windows_per_class: int = 32
    source_quality_windows: int = 256

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 2 or self.review_windows_per_class < 1:
            raise ValueError("updates/review budget must be positive and batch_size >=2")
        if self.source_quality_windows < 2:
            raise ValueError("source feature fitting requires at least two windows")
        if self.synthetic_ratio <= 0 or not np.isfinite(self.synthetic_ratio):
            raise ValueError("synthetic ratio must be finite and positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")


class SmallEEGCNN(nn.Module):
    """Independent development evaluator, deliberately not named EEGNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, 16, 9, padding=4), nn.GELU(),
                                      nn.AvgPool1d(2), nn.Conv1d(16, 32, 7, padding=3),
                                      nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(32, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values).squeeze(-1)).squeeze(-1)


def _fit_predict(train: EvaluationData, validation: EvaluationData, config: G1Config,
                 *, mode: str, generated: EvaluationData | None,
                 generated_positions: np.ndarray | None, source_scale: float) -> tuple[np.ndarray, dict]:
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    # Isolate evaluator RNG from the generator/trainer's persistent random state.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model = SmallEEGCNN(train.waveforms.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        rng = np.random.default_rng(config.seed)
        exposure = {"real": 0, "synthetic": 0, "transformed_real": 0}
        requested = max(2, 2 * int(np.ceil(len(train.labels) * config.synthetic_ratio / 2)))
        auxiliary = 0 if mode == "real_only" else requested
        if generated_positions is not None:
            auxiliary = len(generated_positions)
        classes = [np.flatnonzero(train.labels == label) for label in (0, 1)]
        if any(not len(items) for items in classes):
            raise ValueError("small binary CNN needs both source training classes")
        model.train()
        for _ in range(config.updates):
            # Each method sees the same draws of real/auxiliary slots. Repeated
            # and conventional controls use the generated method's class quota.
            slots = rng.integers(len(train.labels) + auxiliary, size=config.batch_size)
            real_slots = slots < len(train.labels)
            indices = slots % len(train.labels)
            x = np.array(train.waveforms[indices], copy=True)
            y = np.array(train.labels[indices], copy=True)
            exposure["real"] += int(real_slots.sum())
            count = int((~real_slots).sum())
            if count:
                aux_indices = slots[~real_slots] - len(train.labels)
                if generated is not None:
                    chosen = generated_positions[aux_indices]
                    x[~real_slots], y[~real_slots] = generated.waveforms[chosen], generated.labels[chosen]
                    exposure["synthetic"] += count
                else:
                    desired = aux_indices % 2
                    selected = np.asarray([classes[label][rng.integers(len(classes[label]))]
                                           for label in desired])
                    x[~real_slots], y[~real_slots] = train.waveforms[selected], train.labels[selected]
                    exposure["real"] += count
                    if mode == "normalaug":
                        # Mild global amplitude/noise, preserving time/frequency
                        # axes and all lead signs; its recipe is reported.
                        scales = rng.uniform(0.9, 1.1, size=(count, 1, 1))
                        noise = rng.normal(0, 0.01 * source_scale, size=x[~real_slots].shape)
                        x[~real_slots] = x[~real_slots] * scales + noise
                        exposure["transformed_real"] += count
            values = torch.from_numpy(x / source_scale).to(device)
            targets = torch.from_numpy(y.astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(values), targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite downstream evaluator loss")
            loss.backward()
            optimizer.step()
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(validation.labels), config.batch_size):
                batch = torch.from_numpy(validation.waveforms[start:start + config.batch_size]
                                         / source_scale).to(device)
                predictions.extend(torch.sigmoid(model(batch)).cpu().numpy().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.asarray(predictions), {"optimizer_updates": config.updates,
                                     "batch_size": config.batch_size, "seed": config.seed,
                                     "exposure": exposure, "source_scale_uV": source_scale,
                                     "auxiliary_pool_windows": auxiliary,
                                     "wall_seconds": perf_counter() - started,
                                     "device": str(device),
                                     "model": "SmallEEGCNN_v1", "conditions_to_detector": False}


def _generated_import(data: EvaluationData, train: EvaluationData) -> dict:
    required = ("algorithm", "code_commit", "budget", "source_train_digest", "condition_origin")
    if data.split != "generated" or data.role != "generated":
        raise ValueError("generator import must explicitly declare generated data")
    if any(key not in data.metadata for key in required):
        raise ValueError("generator imports require algorithm/code_commit/budget/source/condition provenance")
    if (any(not isinstance(data.metadata[key], str) or not data.metadata[key].strip()
            for key in ("algorithm", "code_commit"))
            or not isinstance(data.metadata["budget"], dict) or not data.metadata["budget"]):
        raise ValueError("generator algorithm/code version and nonempty budget must be explicit")
    if (data.metadata["source_train_digest"] != train.digest
            or data.metadata["condition_origin"] != "source_request"):
        raise ValueError("generator import was not trained/conditioned on this allowed source population")
    if (data.waveforms.shape[1:] != train.waveforms.shape[1:]
            or data.sample_rate != train.sample_rate
            or data.metadata["lead_names"] != train.metadata["lead_names"]):
        raise ValueError("generator import measurement definition differs")
    for key in ("label_namespace", "label_scope"):
        if data.metadata.get(key) != train.metadata.get(key):
            raise ValueError("generated requested-label ontology differs from real annotations")
    if data.metadata.get("target_query_used", False):
        raise ValueError("generator import discloses target query use")
    return {key: data.metadata[key] for key in required} | {
        "import_digest": data.digest, "status": "explicit_precomputed_import",
        "generator_training_seed": data.metadata.get("generator_training_seed"),
        "generator_checkpoint_digest": data.metadata.get(
            "generator_checkpoint_digest", data.metadata.get("checkpoint_sha256")),
        "implementation_sha256": data.metadata.get("implementation_sha256"),
        "generator_recipe_digest": data.metadata.get("generator_recipe_digest"),
        "generator_condition_budget": data.metadata.get("generator_condition_budget"),
        "code_dirty": data.metadata.get("code_dirty"),
        "faithful_author_reproduction_verified": False,
        "label_interpretation": "requested_condition_not_real_patient_diagnosis"}


def _quality(generated: EvaluationData, train: EvaluationData,
              validation: EvaluationData, config: G1Config) -> dict:
    rows = {}
    source_positions = np.linspace(0, len(train.labels) - 1,
                                   min(config.source_quality_windows, len(train.labels)), dtype=int)
    source_features = physical_features(train.waveforms[source_positions], train.sample_rate)
    passed = True
    rng = np.random.default_rng(config.seed)
    for label in (0, 1):
        real = np.flatnonzero(validation.labels == label)
        fake = np.flatnonzero(generated.labels == label)
        count = min(32, config.review_windows_per_class, len(real))
        if count < 2 or len(fake) < count:
            rows[str(label)] = {"status": "insufficient_class_samples", "matched_windows": 0,
                                "required_generated_windows": count,
                                "available_generated_windows": len(fake)}
            passed = False
            continue
        real_positions = np.sort(rng.choice(real, count, replace=False))
        fake_positions = np.sort(rng.choice(fake, count, replace=False))
        x, y = validation.waveforms[real_positions], generated.waveforms[fake_positions]
        f_real, f_fake = spectral_ts_features(x, train.sample_rate), spectral_ts_features(y, train.sample_rate)
        copy = nearest_copy_audit(y, train.waveforms, max_source=config.source_quality_windows)
        # Small-sample screens detect exact copying/collapse, not a calibrated
        # physiological acceptance threshold. All physical summaries remain.
        collapsed = bool(np.max(np.std(y.astype(float), axis=0)) < 1e-8)
        passed = passed and not collapsed and copy["near_exact_count"] == 0
        candidates = np.flatnonzero(validation.labels == label)
        real_reference = {"status": "insufficient_disjoint_real_windows"}
        if len(candidates) >= 2 * count:
            other = np.setdiff1d(candidates, real_positions)
            other = np.sort(rng.choice(other, count, replace=False))
            real_reference = {"status": "complete", "matched_windows": count,
                              "ts_fid": frechet_distance(f_real, spectral_ts_features(
                                  validation.waveforms[other], train.sample_rate)),
                              "sample_ids": validation.sample_ids[other].tolist()}
        rows[str(label)] = {"status": "complete", "matched_windows": count,
                            "real_sample_ids": validation.sample_ids[real_positions].tolist(),
                            "generated_sample_ids": generated.sample_ids[fake_positions].tolist(),
                            "ts_fid": frechet_distance(f_real, f_fake),
                            "physical_mmd": source_scaled_mmd(
                                physical_features(x, train.sample_rate),
                                physical_features(y, train.sample_rate), source_features),
                            "physical_feature_mean_real": physical_features(x, train.sample_rate).mean(0).tolist(),
                            "physical_feature_mean_generated": physical_features(y, train.sample_rate).mean(0).tolist(),
                            "nearest_copy": copy, "exact_collapse": collapsed,
                            "montage_real": _montage_audit(x, train.metadata["lead_names"]),
                            "montage_generated": _montage_audit(y, train.metadata["lead_names"]),
                            "real_real_reference": real_reference}
    return {"classes": rows, "minimum_screen_passed": bool(passed),
            "metric_protocol": "JET_equations_5_6_rfft_adaptive_mean_log1p_ddof0_v1",
            "spectral_bins": 256, "spatial_bins": 4, "cutoff_hz": 80,
            "FID_covariance": "sample_covariance_ddof1_low_rank_exact",
            "statistical_significance_claim": False,
            "clinical_validity_claim": False}


def run_g1(train: EvaluationData, validation: EvaluationData,
           generated: dict[str, EvaluationData], *, candidate: str = "spikeweave",
           required_baselines: tuple[str, ...] = ("jet", "conditional_vp"),
           config: G1Config | None = None, ledger: GateLedger | None = None,
           output: str | Path | None = None) -> dict:
    """Run a small real-query enhancement comparison; never read test inputs."""
    config = config or G1Config()
    signature = _signature(train, validation)
    if ledger is not None:
        ledger.require("G0", signature)
    errors = _real_inputs(train, validation)
    if errors:
        if ledger is not None:
            ledger.record_failure("G1", "; ".join(errors), seed=config.seed)
        raise ValueError("; ".join(errors))
    if candidate in {"real_only", "repeat", "normalaug"}:
        raise ValueError("candidate name collides with a fixed control")
    if not len(required_baselines) or len(set(required_baselines)) != len(required_baselines):
        raise ValueError("required strong baseline names must be a nonempty unique set")
    if any(name in {candidate, "real_only", "repeat", "normalaug"} for name in required_baselines):
        raise ValueError("required strong baselines cannot be candidate or conventional controls")
    imports, quality, downstream, failures = {}, {}, {}, []
    selected = {}
    budget = asdict(config)
    budget.pop("seed")
    source_scale = float(np.sqrt(np.square(train.waveforms.astype(np.float64)).mean()))
    if source_scale <= 1e-8:
        raise ValueError("source scale is degenerate")
    requested = max(2, 2 * int(np.ceil(len(train.labels) * config.synthetic_ratio / 2)))
    quotas = {0: requested // 2, 1: requested - requested // 2}
    for name, data in generated.items():
        try:
            if name in {"real_only", "repeat", "normalaug"}:
                raise ValueError("import name collides with fixed conventional control")
            imports[name] = _generated_import(data, train)
            expected_names = {"jet": {"jet", "just eeg transformer"},
                              "conditional_vp": {"conditional_vp", "conditional vp", "vp", "dit"}}
            if (name in expected_names
                    and data.metadata["algorithm"].strip().lower() not in expected_names[name]):
                raise ValueError("baseline alias disagrees with its explicitly declared algorithm")
            choices = []
            for label in (0, 1):
                rows = np.flatnonzero(data.labels == label)
                if len(rows) < quotas[label]:
                    raise ValueError("generated import has insufficient fixed class quota")
                choices.extend(np.random.default_rng(config.seed).choice(
                    rows, quotas[label], replace=False).tolist())
            # Interleave the class quotas so repeated-control auxiliary slots
            # use the same requested labels rather than a different imbalance.
            by_class = [np.asarray(choices[:quotas[0]]), np.asarray(choices[quotas[0]:])]
            interleaved = [int(rows[i]) for i in range(max(quotas.values()))
                           for rows in by_class if i < len(rows)]
            selected[name] = np.asarray(interleaved)
            quality[name] = _quality(data, train, validation, config)
        except (ValueError, KeyError) as exception:
            failures.append({"method": name, "phase": "import_or_quality", "error": str(exception)})
    for name in ["real_only", "repeat", "normalaug", *selected]:
        try:
            data = generated.get(name) if name in selected else None
            scores, audit = _fit_predict(train, validation, config, mode=name, generated=data,
                                         generated_positions=selected.get(name), source_scale=source_scale)
            downstream[name] = patient_average_precision(validation.labels, scores, validation.patient_ids)
            downstream[name]["training_audit"] = audit
            downstream[name]["predictions"] = scores.tolist()
        except (ValueError, RuntimeError) as exception:
            failures.append({"method": name, "phase": "downstream", "error": str(exception)})
    required = [candidate, *required_baselines, "real_only", "repeat", "normalaug"]
    missing = [name for name in required if name not in downstream]
    minimum = (not missing and all(quality.get(name, {}).get("minimum_screen_passed", False)
                                  for name in [candidate, *required_baselines]))
    values = {name: downstream[name]["macro_patient_ap"] for name in required if name in downstream}
    computable = not missing and all(value is not None for value in values.values())
    strongest = (max([name for name in required if name != candidate], key=lambda n: values[n])
                 if computable else None)
    passed = bool(minimum and computable and values[candidate] >= values[strongest])
    report = {"stage": "G1", "status": "complete" if not missing and not failures else "incomplete",
              "candidate": candidate, "seed": config.seed, "data_signature": signature,
              "train_digest": train.digest, "validation_digest": validation.digest,
              "train_counts": train.counts(), "validation_counts": validation.counts(),
              "required_baselines": list(required_baselines), "budget": budget,
              "budget_digest": json_digest(budget), "imports": imports,
              "quality": quality, "quality_minimum_passed": bool(minimum),
              "downstream": downstream, "missing_methods": missing, "failures": failures,
              "strongest_baseline": strongest, "gate_passed": passed and not failures,
              "test_used": False, "primary": "strict_source_only_macro_patient_AP",
              "SOTA_claim": False,
              "conclusion": "small_development_diagnostic_not_formal_clinical_evaluation"}
    write_report(output, report)
    if ledger is not None:
        ledger.record(report)
    return report
