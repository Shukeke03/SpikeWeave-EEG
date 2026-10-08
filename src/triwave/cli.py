"""Local/server command-line workflow. Synthetic checks carry explicit provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from triwave.config import ModelConfig
from triwave.runtime import seed_everything, select_device
from triwave.training import Trainer, TrainingConfig, append_metrics


def _config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig.from_yaml(args.config) if args.config else ModelConfig.tiny()


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    import yaml
    fields = {}
    if args.config:
        fields = (yaml.safe_load(Path(args.config).read_text()) or {}).get("training", {})
    for key in ("learning_rate", "stage", "gradient_accumulation"):
        if getattr(args, key, None) is not None:
            fields[key] = getattr(args, key)
    if args.alignment_only:
        fields["stage"] = "align"
    fields["objective"] = args.objective
    fields["mixed_precision"] = args.amp or fields.get("mixed_precision", False)
    return TrainingConfig(**fields)


def _emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def doctor(args: argparse.Namespace) -> None:
    import mne
    import pywt
    import transformers

    from triwave.models import TriWaveModel

    device = select_device(args.device)
    config = _config(args)
    if args.config_only:
        _emit({"status": "config_valid", "representation_branches": 2,
               "eeg": [config.encoder_depth, config.dim, config.heads],
               "shared_condition": [config.fusion_depth, config.condition_dim, config.condition_heads],
               "mcwave": [config.denoiser_depth, config.denoiser_dim, config.denoiser_heads],
               "frontend_backend": config.frontend_backend, "montage_mode": config.montage_mode})
        return
    model = TriWaveModel(config)
    _emit({"python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": torch.cuda.is_available(), "selected_device": str(device),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           "mne": mne.__version__, "pywavelets": pywt.__version__,
           "transformers": transformers.__version__, "text_backend": config.text_backend,
           "parameters": sum(p.numel() for p in model.parameters()),
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "eeg_shape": [config.channels, config.samples],
           "clinical_pretraining": model.condition_encoder.is_medically_pretrained,
           "representation_branches": 2, "frontend_backend": config.frontend_backend})


def train_synthetic(args: argparse.Namespace) -> None:
    from triwave.data.synthetic import make_synthetic_batch

    device = select_device(args.device)
    trainer = (Trainer.load(args.resume, device, resume_optimizer=True) if args.resume
               else Trainer(_config(args), _training_config(args), device))
    if getattr(args, "init_representation", None) and not args.resume:
        trainer.initialize_representation(args.init_representation)
    if not args.resume:
        trainer.run_provenance = {"generator_training_seed": args.seed,
                                  "data_kind": "synthetic_engineering_check"}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps * trainer.training.gradient_accumulation):
        batch = make_synthetic_batch(trainer.config, args.batch_size, seed=args.seed + trainer.step)
        metrics = trainer.train_step(batch, alignment_only=args.alignment_only)
        metrics["data_kind"] = "synthetic_engineering_check"
        append_metrics(output / "metrics.jsonl", metrics)
        _emit(metrics)
    trainer.save(output / "last.pt")
    batch = trainer.move_batch(make_synthetic_batch(trainer.config, args.batch_size, args.seed + 999))
    condition = trainer.conditions(batch)
    generated = trainer.sample(condition, steps=args.sampling_steps, seed=args.seed)
    np.savez_compressed(output / "generated.npz", waveform=generated.cpu().numpy(),
                        sample_rate=trainer.config.sample_rate, units="synthetic_arbitrary_units",
                        provenance="synthetic_engineering_check; no clinical efficacy measured")
    report = {"status": "ok", "data_kind": "synthetic_engineering_check", "step": trainer.step,
              "objective": trainer.training.objective, "device": str(device),
              "text_backend": trainer.config.text_backend,
              "generated_shape": list(generated.shape), "checkpoint": str(output / "last.pt"),
              "sampler": trainer.last_sampler_stats,
              "peak_cuda_mib": (torch.cuda.max_memory_allocated() / 1024**2
                                if device.type == "cuda" else None)}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _emit(report)


def train_manifest(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from triwave.data.dataset import EEGManifestDataset, TrainNormalizer
    from triwave.data.dualbranch import DualBranchEEGDataset, PatientEventBalancedSampler
    from triwave.data.sharded import ShardedEEGDataset, fit_sharded_statistics

    device = select_device(args.device)
    data_path = Path(args.manifest)
    shard_root = data_path if data_path.is_dir() else data_path.parent
    dualbranch = (shard_root / "dualbranch_manifest.json").is_file()
    sharded = (shard_root / "window_index.npy").is_file() and (
        data_path.is_dir() or data_path.name == "window_index.npy")
    if args.resume:
        trainer = Trainer.load(args.resume, device, resume_optimizer=True)
        normalizer = TrainNormalizer.from_dict(trainer.normalization) if trainer.normalization else None
        if normalizer is None:
            raise ValueError("clinical manifest training requires a checkpoint with train normalization")
    else:
        config = _config(args)
        if dualbranch:
            raw_train = DualBranchEEGDataset(shard_root, config=config, split="train")
            normalizer = raw_train.normalizer
        elif sharded:
            raw_train = ShardedEEGDataset(shard_root, config=config, split="train")
            fitted_path = shard_root / "train_scale_v2.json"
            normalizer = (TrainNormalizer.load(fitted_path) if fitted_path.is_file()
                          else fit_sharded_statistics(raw_train))
        else:
            raw_train = EEGManifestDataset(args.manifest, config=config, split="train", role="source")
            normalizer = TrainNormalizer().fit(raw_train)
        trainer = Trainer(config, _training_config(args), device)
        if normalizer.scale_protocol != "source_rms_v2":
            raise ValueError("refit source_rms_v2; legacy affine normalizers cannot train v0.3")
        trainer.normalization = normalizer.to_dict()
        trainer.run_provenance = {"generator_training_seed": args.seed,
                                  "source_scale_fit_digest": normalizer.fit_manifest_digest,
                                  "stage": trainer.training.stage}
        if sharded and not dualbranch:
            normalizer.save(shard_root / "train_scale_v2.json")
    expected_batch = trainer.training.effective_batch_size
    if args.config and expected_batch and args.batch_size * trainer.training.gradient_accumulation != expected_batch:
        raise ValueError(f"configured effective batch is {expected_batch}; set batch-size × accumulation accordingly")
    if dualbranch:
        purpose = args.data_purpose or (
            "representation" if trainer.training.stage == "pretrain" else "training")
        dataset = DualBranchEEGDataset(shard_root, config=trainer.config, split="train",
                                      purpose=purpose, seed=args.seed)
        if dataset.normalizer.to_dict() != normalizer.to_dict():
            raise ValueError("dataset source scale differs from the resumed checkpoint")
        if trainer.config.montage_mode == "hard":
            from triwave.data.dualbranch import approve_hard_montage

            trainer.hard_audit_approved = approve_hard_montage(shard_root, trainer.config)
    else:
        if args.data_purpose:
            raise ValueError("explicit condition roles require a dualbranch-v2 dataset")
        dataset = (ShardedEEGDataset(shard_root, config=trainer.config, split="train",
                                    normalizer=normalizer) if sharded else
                   EEGManifestDataset(args.manifest, config=trainer.config, split="train",
                                      role="source", normalizer=normalizer))
    if not len(dataset):
        raise ValueError("manifest has no train/source windows")
    sampler = None
    if args.balanced and dualbranch and purpose != "representation":
        sampler = PatientEventBalancedSampler(dataset, seed=args.seed)
    elif args.balanced and not dualbranch:
        labels = (torch.as_tensor(np.array(dataset.window_index["label"][dataset.indices], copy=True),
                                  dtype=torch.long) if sharded else torch.tensor([
                                      record.conditions.get("class", {}).get("id", -1)
                                      for record in dataset.records], dtype=torch.long))
        if (labels < 0).any():
            raise ValueError("balanced sampling requires known nonnegative class IDs")
        counts = torch.bincount(labels).clamp_min(1)
        weights = 1 / counts[labels].double()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=0, collate_fn=_collate,
                        drop_last=len(dataset) >= args.batch_size)
    if getattr(args, "init_representation", None) and not args.resume:
        trainer.initialize_representation(args.init_representation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    sampling_epoch = 0
    for _ in range(args.steps * trainer.training.gradient_accumulation):
        try:
            batch = next(iterator)
        except StopIteration:
            sampling_epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(sampling_epoch)
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch, alignment_only=args.alignment_only)
        append_metrics(output / "metrics.jsonl", metrics)
        _emit(metrics)
    trainer.save(output / "last.pt")
    _emit({"status": "ok", "step": trainer.step, "train_windows": len(dataset),
           "checkpoint": str(output / "last.pt"), "clinical_efficacy_evaluated": False})


def build_dualbranch(args: argparse.Namespace) -> None:
    from triwave.data.dualbranch import build_dualbranch_dataset

    path = build_dualbranch_dataset(
        args.root, args.output, _config(args), batch_size=args.batch_size,
        include_query_attributes=args.include_query_attributes,
        include_unlabelled_source=args.include_unlabelled_source, seed=args.seed)
    _emit({"status": "ok", "dataset": str(path), "clinical_efficacy_evaluated": False})


def _collate(items: list[dict]) -> dict:
    """Tensor fields are stackable; heterogeneous provenance stays a list."""
    keys = set.intersection(*(set(item) for item in items))
    return {key: torch.stack([item[key] for item in items]) if isinstance(items[0][key], torch.Tensor)
            else [item[key] for item in items] for key in keys}


def _condition_slice(batch: dict, start: int, stop: int, size: int) -> dict:
    """Slice tensors and per-record provenance together before device/CWT work."""
    def select(value):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            return value[start:stop]
        if isinstance(value, (list, tuple)) and len(value) == size:
            return value[start:stop]
        return value

    return {key: select(value) for key, value in batch.items()}


def sample(args: argparse.Namespace) -> None:
    from triwave.attributes import ATTRIBUTE_PROTOCOL
    from triwave.data.dataset import tokenize_texts
    from triwave.signal import CWT_MASK_PROTOCOL

    trainer = Trainer.load(args.checkpoint, select_device(args.device))
    source = None
    if args.evaluation_source:
        from triwave.evaluation import load_evaluation_data

        source = load_evaluation_data(args.evaluation_source, stage="G1")
        if source.split != "train" or source.role != "source":
            raise ValueError("evaluation generation requires the declared train/source population")
    metadata = {}
    requested_classes, prompts = None, None
    # Only allowlisted conditions enter the model; teacher/target fields stay excluded.
    with np.load(args.conditions, allow_pickle=False) as archive:
        if "structure" in archive:
            expected_scale = "normalized_uV" if trainer.normalization else "synthetic_arbitrary_units"
            if (str(archive.get("cwt_protocol", "")) != "triwave-morlet-v1"
                    or str(archive.get("structure_scale", "")) != expected_scale
                    or str(archive.get("cwt_mask_protocol", "")) != CWT_MASK_PROTOCOL):
                raise ValueError("external structure requires cwt_protocol=triwave-morlet-v1 and "
                                 f"structure_scale={expected_scale} and mask={CWT_MASK_PROTOCOL}")
        if trainer.config.montage_mode != "off":
            if "lead_names" not in archive or tuple(archive["lead_names"].tolist()) != tuple(trainer.config.lead_names):
                raise ValueError("condition NPZ must declare lead_names in the configured montage order")
        if "structure" in archive and trainer.normalization:
            norm = trainer.normalization
            if (str(archive.get("structure_normalizer_protocol", "")) != "source_rms_v2"
                or str(archive.get("structure_fit_digest", "")) != norm["fit_manifest_digest"]
                or not np.isclose(float(archive.get("structure_source_scale", np.nan)), norm["std"])):
                raise ValueError("external CWT must declare the checkpoint's source scalar protocol/digest/scale")
        tensors = {k: torch.from_numpy(archive[k].copy()) for k in archive.files
                   if k in {"lead_geometry", "reference_waveform", "reference_valid", "structure",
                            "structure_valid", "structure_visible", "attribute_values", "attribute_valid"}}
        for key in ("reference_provenance", "structure_provenance", "attribute_origin"):
            if key in archive:
                metadata[key] = json.loads(str(archive[key].item()))
        for key in ("structure_role", "cwt_mask_protocol"):
            if key in archive:
                value = archive[key]
                metadata[key] = str(value.item()) if value.ndim == 0 else value.tolist()
        if "attribute_valid" in tensors and tensors["attribute_valid"].any():
            if str(archive.get("attribute_protocol", "")) != ATTRIBUTE_PROTOCOL:
                raise ValueError(f"numeric conditions require attribute_protocol={ATTRIBUTE_PROTOCOL}")
        if "requested_classes" in archive:
            requested_classes = np.array(archive["requested_classes"], copy=True)
        if "prompts" in archive:
            prompts = archive["prompts"].tolist()
    if "lead_geometry" not in tensors:
        raise ValueError("condition NPZ requires lead_geometry [B,C,6]")
    if "reference_waveform" in tensors and "reference_valid" not in tensors:
        raise ValueError("reference waveform requires an explicit reference_valid mask")
    if ("reference_waveform" in tensors and tensors["reference_valid"].any()
            and "reference_provenance" not in metadata):
        raise ValueError("observed reference requires independent reference_provenance, even with external structure")
    batch_size = len(tensors["lead_geometry"])
    if batch_size < 1:
        raise ValueError("generation requires at least one condition record")
    if args.requested_class is not None:
        requested_classes = np.full(batch_size, args.requested_class, dtype=np.int64)
    if requested_classes is not None:
        if requested_classes.shape != (batch_size,) or not np.isin(requested_classes, [0, 1]).all():
            raise ValueError("requested_classes must be one explicit binary request per generated window")
    if args.prompt:
        prompts = [args.prompt] * batch_size
    if requested_classes is not None:
        class_prompts = {
            "0": "No seizure interval is annotated within this window.",
            "1": "This window overlaps an annotated seizure."}
        if source is not None:
            declared = source.metadata.get("class_prompts")
            if declared is None and source.metadata.get("label_namespace") != "chbmit.window_event":
                raise ValueError("evaluation source ontology must declare class_prompts for binary requests")
            class_prompts = declared or class_prompts
            if (not isinstance(class_prompts, dict) or set(class_prompts) != {"0", "1"}
                    or any(not isinstance(value, str) or not value.strip()
                           for value in class_prompts.values())
                    or class_prompts["0"] == class_prompts["1"]):
                raise ValueError("class_prompts must map binary request IDs to distinct nonempty texts")
        expected_prompts = [class_prompts[str(int(label))] for label in requested_classes]
        if prompts is not None and prompts != expected_prompts:
            raise ValueError("request classes and supplied prompts disagree with the source ontology")
        prompts = expected_prompts
    if prompts is not None:
        if not isinstance(prompts, list) or len(prompts) != batch_size or any(
                not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must contain one string per generated window")
        tensors["input_ids"], tensors["text_valid"] = tokenize_texts(
            prompts, trainer.config)
    if trainer.normalization and "reference_waveform" in tensors:
        from triwave.data.dataset import TrainNormalizer

        normalizer = TrainNormalizer.from_dict(trainer.normalization)
        tensors["reference_waveform"] = normalizer.transform(tensors["reference_waveform"])
    if source is not None:
        if requested_classes is None or not trainer.normalization:
            raise ValueError("evaluation export needs physical source/train inputs and explicit request classes")
        if (source.metadata.get("scale_protocol") != "source_rms_v2"
                or source.metadata.get("scale_fit_digest") != trainer.normalization["fit_manifest_digest"]):
            raise ValueError("evaluation source must declare the checkpoint's fitted source-scale digest")
        roles = metadata.get("structure_role", "none")
        if "partial_observation" in ([roles] if isinstance(roles, str) else roles):
            raise ValueError("observation-conditioned generation is a separate evaluation protocol")
        for key in ("reference_provenance", "structure_provenance", "attribute_origin"):
            origins = metadata.get(key)
            origins = [origins] if isinstance(origins, dict) else origins or []
            if any(not isinstance(origin, dict) or origin.get("target_derived") is not False
                   or origin.get("source_role") != "train" for origin in origins):
                raise ValueError("source-only evaluation requires source/train provenance for each condition")
        if (source.waveforms.shape[1:] != (trainer.config.channels, trainer.config.samples)
                or source.sample_rate != trainer.config.sample_rate
                or source.metadata["lead_names"] != list(trainer.config.lead_names[:trainer.config.channels])):
            raise ValueError("evaluation source measurement definition differs from the generator")
    generated_chunks, network_calls, predictor_calls = [], 0, 0
    for start in range(0, batch_size, args.batch_size):
        selected = _condition_slice(tensors | metadata, start, min(start + args.batch_size, batch_size), batch_size)
        condition = trainer.conditions(trainer.move_batch(selected))
        generated_chunks.append(trainer.sample(condition, args.steps, args.guidance, args.seed + start).cpu())
        network_calls += trainer.last_sampler_stats["network_calls"]
        predictor_calls += trainer.last_sampler_stats["predictor_calls"]
    generated = torch.cat(generated_chunks)
    trainer.last_sampler_stats = trainer.last_sampler_stats | {
        "microbatches": len(generated_chunks), "microbatch_size": args.batch_size,
        "total_network_calls": network_calls, "total_predictor_calls": predictor_calls}
    units = "synthetic_arbitrary_units"
    if trainer.normalization:
        from triwave.data.dataset import TrainNormalizer

        generated = TrainNormalizer.from_dict(trainer.normalization).inverse_transform(generated)
        units = "uV"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"waveform": generated.numpy(), "units": units,
              "sample_rate": trainer.config.sample_rate,
              "lead_geometry": tensors["lead_geometry"].cpu().numpy(),
              "lead_names": np.asarray(trainer.config.lead_names[:trainer.config.channels]),
              "provenance": "model generated; research output; clinical efficacy unmeasured"}
    if source is not None:
        code = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=False)
        code_commit = code.stdout.strip() if code.returncode == 0 else "unavailable"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                               cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, check=False)
        implementation_digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            implementation_digest.update(str(path.relative_to(Path(__file__).parent)).encode())
            implementation_digest.update(path.read_bytes())
        budget = {"training_optimizer_steps": trainer.step, "sampling_steps": args.steps,
                  "guidance": args.guidance, "generated_windows": batch_size,
                  "sampling_microbatch": args.batch_size, "chunk_seed_rule": "request_seed+start_index"}
        if args.generation_budget:
            declared = json.loads(Path(args.generation_budget).read_text())
            if not isinstance(declared, dict):
                raise ValueError("generation-budget must contain a JSON object")
            budget["declared_training_budget"] = declared
        checkpoint_digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        recipe = {"model": trainer.config.to_dict(), "training": asdict(trainer.training),
                  "condition_protocols": trainer.condition_protocols()}
        recipe_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
        condition_budget = {"requested_class_counts": np.bincount(requested_classes, minlength=2).tolist(),
                            "class_prompts": class_prompts,
                            "reference_records": int(tensors.get("reference_valid", torch.zeros(batch_size, 1))
                                                     .bool().reshape(batch_size, -1).any(-1).sum()),
                            "attribute_field_counts": tensors.get("attribute_valid", torch.zeros(batch_size, 0))
                                                      .bool().sum(0).tolist(),
                            "structure_valid_cells": int(tensors.get("structure_valid", torch.zeros(0))
                                                         .bool().sum()),
                            "structure_visible_cells": int(tensors.get("structure_visible", torch.zeros(0))
                                                           .bool().sum())}
        evaluation_metadata = {
            "split": "generated", "role": "generated", "unit": "uV",
            "sample_rate": trainer.config.sample_rate,
            "lead_names": list(trainer.config.lead_names[:trainer.config.channels]),
            "algorithm": "SpikeWeave-EEG-v0.3", "code_commit": code_commit, "budget": budget,
            "code_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "implementation_sha256": implementation_digest.hexdigest(),
            "generator_recipe_digest": recipe_digest, "generator_condition_budget": condition_budget,
            "checkpoint_sha256": checkpoint_digest, "source_train_digest": source.digest,
            "generator_checkpoint_digest": checkpoint_digest,
            "generator_training_seed": trainer.run_provenance.get("generator_training_seed"),
            "condition_origin": "source_request", "evaluation_protocol": "source-only",
            "target_query_used": False, "request_seed": args.seed,
            "label_interpretation": "requested_condition_not_real_patient_diagnosis",
            **{key: source.metadata.get(key) for key in ("label_namespace", "label_scope")}}
        arrays.update(waveforms=generated.numpy(), labels=requested_classes.astype(np.int64),
                      patient_ids=np.full(batch_size, "generated-source-request"),
                      record_ids=np.full(batch_size, f"generated:{checkpoint_digest}:{args.seed}"),
                      sample_ids=np.asarray([f"generated:{checkpoint_digest}:{args.seed}:{i}"
                                             for i in range(batch_size)]),
                      metadata=json.dumps(evaluation_metadata, allow_nan=False))
    elif requested_classes is not None:
        arrays["requested_classes"] = requested_classes
    np.savez_compressed(output, **arrays)
    _emit({"status": "ok", "sampler": trainer.last_sampler_stats, "output": str(output), "shape": list(generated.shape), "units": units})


def prepare_chbmit(args: argparse.Namespace) -> None:
    from triwave.data.chbmit import prepare_chbmit as prepare

    split_map = json.loads(Path(args.split_map).read_text())
    result = prepare(args.root, args.output, split_map, window_seconds=args.window_seconds,
                     target_sfreq=args.sample_rate)
    _emit({"status": "ok", "result": str(result)})


def attach_references(args: argparse.Namespace) -> None:
    from triwave.data.manifest import attach_source_references

    path = attach_source_references(args.manifest, args.output, seed=args.seed)
    _emit({"status": "ok", "manifest": str(path)})


def build_chbmit(args: argparse.Namespace) -> None:
    from triwave.data.build import build_chbmit as build

    split_map = json.loads(Path(args.split_map).read_text()) if args.split_map else None
    path = build(args.root, args.output, split_map, workers=args.workers,
                 samples=args.window_samples, sample_rate=args.sample_rate, seed=args.seed)
    _emit({"status": "ok", "index": str(path)})


def audit_dataset(args: argparse.Namespace) -> None:
    from triwave.data.validation import validate_sharded_dataset

    if (Path(args.root) / "dualbranch_manifest.json").is_file():
        from triwave.data.dualbranch import validate_dualbranch_dataset

        result = validate_dualbranch_dataset(args.root, config=_config(args),
                                            numerical_windows=args.numerical_windows)
    else:
        result = validate_sharded_dataset(args.root, config=_config(args),
                                         numerical_windows=args.numerical_windows, device=args.device)
    _emit(result)


def evaluate(args: argparse.Namespace) -> None:
    """Bounded development comparisons; the final-test suite remains gated."""
    from triwave.evaluation import (
        G1Config,
        GateLedger,
        authorize_g3,
        evaluation_data_from_dataset,
        load_evaluation_data,
        record_g2,
        run_g0,
        run_g1,
    )

    ledger = GateLedger(args.ledger)

    def load(path, split):
        if path is None:
            raise ValueError(f"{args.stage} needs --{split}")
        if Path(path).is_dir():
            from triwave.data.dualbranch import DualBranchEEGDataset

            dataset = DualBranchEEGDataset(path, _config(args), split=split,
                                          purpose="source_only", seed=args.selection_seed)
            return evaluation_data_from_dataset(dataset, max_windows=args.max_windows,
                                                stage=args.stage, ledger=ledger,
                                                seed=args.selection_seed)
        return load_evaluation_data(path, stage=args.stage, ledger=ledger)

    try:
        if args.stage == "G2":
            report = record_g2(ledger, args.evidence, candidate=args.candidate, output=args.output)
        elif args.stage == "G3":
            if not args.operation:
                raise ValueError("G3 requires --operation for a predeclared final-test operation")
            ledger.require_g3()
            test = load(args.test, "test") if args.test else None
            report = authorize_g3(ledger, args.operation, test=test, output=args.output)
        else:
            train, validation = load(args.train, "train"), load(args.validation, "validation")
            if args.stage == "G0":
                report = run_g0(train, validation, ledger=ledger, output=args.output)
            else:
                generated = {}
                for specification in args.generated:
                    name, separator, path = specification.partition("=")
                    if not separator or not name or not path or name in generated:
                        raise ValueError("--generated must be a unique method=archive.npz")
                    generated[name] = load_evaluation_data(path, stage="G1", ledger=ledger)
                report = run_g1(
                    train, validation, generated, candidate=args.candidate,
                    config=G1Config(updates=args.updates, batch_size=args.batch_size,
                                    seed=args.seed), ledger=ledger, output=args.output)
    except Exception as error:
        # Runtime/import failures also invalidate prior evidence; re-raise so
        # they cannot be mistaken for a completed evaluation.
        ledger.record_failure(args.stage, error, seed=args.seed)
        raise
    _emit(report)


def export_evaluation(args: argparse.Namespace) -> None:
    """Physical source/validation interchange, without teacher or condition fields."""
    from triwave.data.dualbranch import DualBranchEEGDataset
    from triwave.evaluation import evaluation_data_from_dataset

    dataset = DualBranchEEGDataset(args.root, _config(args), split=args.split,
                                  purpose="source_only", seed=args.selection_seed)
    data = evaluation_data_from_dataset(dataset, max_windows=args.max_windows,
                                        seed=args.selection_seed)
    metadata = data.metadata | {"split": data.split, "role": data.role, "unit": data.unit,
                                "sample_rate": data.sample_rate}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, waveforms=data.waveforms, labels=data.labels,
                        patient_ids=data.patient_ids, record_ids=data.record_ids,
                        sample_ids=data.sample_ids, metadata=json.dumps(metadata, allow_nan=False))
    _emit({"status": "ok", "output": str(target), "split": data.split,
           "counts": data.counts(), "clinical_efficacy_evaluated": False})


def prepare_biomedclip(args):
    """Explicit free checkpoint preparation; generation constructors are offline."""
    from huggingface_hub import snapshot_download
    root = Path(args.output).resolve()
    snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                      local_dir=str(root), max_workers=8,
                      allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin", "README.md"])
    snapshot_download("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                      local_dir=str(root / "text_config"), max_workers=8,
                      allow_patterns=["config.json", "tokenizer*", "vocab.txt", "special_tokens_map.json"])
    _emit({"status": "prepared", "path": str(root), "weights_loaded_and_tested": False})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpikeWeave-EEG v0.3 research implementation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--config")
    doctor_parser.add_argument("--config-only", action="store_true")
    doctor_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    doctor_parser.set_defaults(function=doctor)
    for name in ["smoke", "train-synthetic", "train"]:
        command = commands.add_parser(name)
        command.add_argument("--config")
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--steps", type=int, default=3 if name == "smoke" else 100)
        command.add_argument("--batch-size", type=int, default=2)
        command.add_argument("--objective", choices=["diffusion"], default="diffusion")
        command.add_argument("--learning-rate", type=float, default=None)
        command.add_argument("--stage", choices=["pretrain", "align", "diffusion", "joint_smoke"])
        command.add_argument("--gradient-accumulation", type=int)
        command.add_argument("--init-representation")
        command.add_argument("--amp", action="store_true", help="CUDA bf16 Transformer autocast")
        command.add_argument("--alignment-only", action="store_true")
        command.add_argument("--resume")
        command.add_argument("--output", default=f"runs/{name}")
        command.add_argument("--sampling-steps", type=int, default=4)
        if name == "train":
            command.add_argument("--manifest", required=True)
            command.add_argument("--data-purpose", choices=["training", "representation",
                                                            "partial_observation"])
            command.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                                 help="v2 patient/event/class sampling; legacy class-frequency sampling")
        command.set_defaults(function=train_manifest if name == "train" else train_synthetic)
    prepare = commands.add_parser("prepare-chbmit")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split-map", required=True)
    prepare.add_argument("--window-seconds", type=float, default=4)
    prepare.add_argument("--sample-rate", type=float, default=256)
    prepare.set_defaults(function=prepare_chbmit)
    references = commands.add_parser("attach-references")
    references.add_argument("--manifest", required=True)
    references.add_argument("--output")
    references.set_defaults(function=attach_references)
    build = commands.add_parser("build-chbmit")
    build.add_argument("--root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--split-map")
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--window-samples", type=int, default=1024)
    build.add_argument("--sample-rate", type=float, default=256)
    build.set_defaults(function=build_chbmit)
    dualbranch = commands.add_parser("build-dualbranch")
    dualbranch.add_argument("--root", required=True, help="completed v1 continuous-shard root")
    dualbranch.add_argument("--output", required=True)
    dualbranch.add_argument("--config", default="configs/legion.yaml")
    dualbranch.add_argument("--batch-size", type=int, default=128)
    dualbranch.add_argument("--include-query-attributes", action="store_true",
                            help="offline evaluation teachers only; never query conditions")
    dualbranch.add_argument("--include-unlabelled-source", action="store_true")
    dualbranch.set_defaults(function=build_dualbranch)
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--root", required=True)
    audit.add_argument("--config", default="configs/legion.yaml")
    audit.add_argument("--numerical-windows", type=int, default=16)
    audit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    audit.set_defaults(function=audit_dataset)
    medical = commands.add_parser("prepare-biomedclip")
    medical.add_argument("--output", default="weights/BiomedCLIP")
    medical.set_defaults(function=prepare_biomedclip)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--stage", choices=["G0", "G1", "G2", "G3"], required=True)
    evaluation.add_argument("--config", default="configs/legion.yaml")
    evaluation.add_argument("--train", help="physical NPZ or dualbranch-v2 directory")
    evaluation.add_argument("--validation", help="natural validation query split")
    evaluation.add_argument("--test", help="G3 only; locked final-test input")
    evaluation.add_argument("--generated", action="append", default=[], metavar="METHOD=NPZ")
    evaluation.add_argument("--candidate", default="spikeweave")
    evaluation.add_argument("--ledger", default="runs/evaluation/gates.json")
    evaluation.add_argument("--output", default="runs/evaluation/report.json")
    evaluation.add_argument("--updates", type=int, default=200)
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--max-windows", type=int, default=4096,
                            help="per-split uniform numerical cap for directory inputs (default 4096)")
    evaluation.add_argument("--selection-seed", type=int, default=0,
                            help="fixed data subset seed, separate from model/evaluator seed")
    evaluation.add_argument("--evidence", action="append", default=[], help="G2 paired seed report")
    evaluation.add_argument("--operation", help="G3 predeclared operation, without launching full suite")
    evaluation.set_defaults(function=evaluate)
    export = commands.add_parser("export-evaluation")
    export.add_argument("--root", required=True)
    export.add_argument("--config", default="configs/legion.yaml")
    export.add_argument("--split", choices=["train", "validation"], required=True)
    export.add_argument("--max-windows", type=int, default=4096)
    export.add_argument("--selection-seed", type=int, default=0)
    export.add_argument("--output", required=True)
    export.set_defaults(function=export_evaluation)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--conditions", required=True)
    sampling.add_argument("--prompt")
    sampling.add_argument("--requested-class", type=int, choices=[0, 1],
                          help="binary request enters C as canonical text; default ontology is CHB event overlap")
    sampling.add_argument("--generation-budget", help="JSON generation-budget declaration")
    sampling.add_argument("--evaluation-source", help="physical train/source evaluation NPZ; exports auditable requests")
    sampling.add_argument("--output", default="outputs/generated.npz")
    sampling.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    sampling.add_argument("--batch-size", type=int, default=4, help="generation microbatch, before CWT/device loading")
    sampling.add_argument("--steps", type=int, default=50)
    sampling.add_argument("--guidance", type=float, default=2)
    sampling.set_defaults(function=sample)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "steps", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        parser.error("steps and batch-size must be positive")
    seed_everything(args.seed, args.threads)
    args.function(args)


if __name__ == "__main__":
    main()
