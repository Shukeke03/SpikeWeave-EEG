"""Local/server command-line workflow. Synthetic checks carry explicit provenance."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from triwave.config import ModelConfig
from triwave.runtime import seed_everything, select_device
from triwave.training import Trainer, TrainingConfig, append_metrics
from triwave.types import ConditionBatch


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
    from triwave.data.sharded import ShardedEEGDataset, fit_sharded_statistics

    device = select_device(args.device)
    data_path = Path(args.manifest)
    shard_root = data_path if data_path.is_dir() else data_path.parent
    sharded = (shard_root / "window_index.npy").is_file() and (
        data_path.is_dir() or data_path.name == "window_index.npy")
    if args.resume:
        trainer = Trainer.load(args.resume, device, resume_optimizer=True)
        normalizer = TrainNormalizer.from_dict(trainer.normalization) if trainer.normalization else None
        if normalizer is None:
            raise ValueError("clinical manifest training requires a checkpoint with train normalization")
    else:
        config = _config(args)
        if sharded:
            raw_train = ShardedEEGDataset(shard_root, config=config, split="train")
            fitted_path = shard_root / "train_scale_v2.json"
            normalizer = (TrainNormalizer.load(fitted_path) if fitted_path.is_file()
                          else fit_sharded_statistics(raw_train))
        else:
            raw_train = EEGManifestDataset(args.manifest, config=config, split="train", role="source")
            normalizer = TrainNormalizer().fit(raw_train)
        trainer = Trainer(config, _training_config(args), device)
        if normalizer.scale_protocol != "source_rms_v2":
            raise ValueError("refit source_rms_v2; legacy affine normalizers cannot train v0.2")
        trainer.normalization = normalizer.to_dict()
        if sharded:
            normalizer.save(shard_root / "train_scale_v2.json")
    expected_batch = trainer.training.effective_batch_size
    if args.config and expected_batch and args.batch_size * trainer.training.gradient_accumulation != expected_batch:
        raise ValueError(f"configured effective batch is {expected_batch}; set batch-size × accumulation accordingly")
    dataset = (ShardedEEGDataset(shard_root, config=trainer.config, split="train", normalizer=normalizer)
               if sharded else EEGManifestDataset(args.manifest, config=trainer.config, split="train",
                                                   role="source", normalizer=normalizer))
    if not len(dataset):
        raise ValueError("manifest has no train/source windows")
    sampler = None
    if args.balanced:
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
    for _ in range(args.steps * trainer.training.gradient_accumulation):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch, alignment_only=args.alignment_only)
        append_metrics(output / "metrics.jsonl", metrics)
        _emit(metrics)
    trainer.save(output / "last.pt")
    _emit({"status": "ok", "step": trainer.step, "train_windows": len(dataset),
           "checkpoint": str(output / "last.pt"), "clinical_efficacy_evaluated": False})


def _collate(items: list[dict]) -> dict:
    """Tensor fields are stackable; heterogeneous provenance stays a list."""
    keys = set.intersection(*(set(item) for item in items))
    return {key: torch.stack([item[key] for item in items]) if isinstance(items[0][key], torch.Tensor)
            else [item[key] for item in items] for key in keys}


def sample(args: argparse.Namespace) -> None:
    from triwave.data.dataset import tokenize_texts

    trainer = Trainer.load(args.checkpoint, select_device(args.device))
    # Geometry and independently obtained references/structures must be supplied explicitly.
    with np.load(args.conditions, allow_pickle=False) as archive:
        if "structure" in archive:
            expected_scale = "normalized_uV" if trainer.normalization else "synthetic_arbitrary_units"
            if (str(archive.get("cwt_protocol", "")) != "triwave-morlet-v1"
                    or str(archive.get("structure_scale", "")) != expected_scale):
                raise ValueError("external structure requires cwt_protocol=triwave-morlet-v1 and "
                                 f"structure_scale={expected_scale}")
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
                            "structure_valid", "attribute_values", "attribute_valid"}}
    if "lead_geometry" not in tensors:
        raise ValueError("condition NPZ requires lead_geometry [B,C,6]")
    batch_size = len(tensors["lead_geometry"])
    if args.prompt:
        tensors["input_ids"], tensors["text_valid"] = tokenize_texts(
            [args.prompt] * batch_size, trainer.config)
    if trainer.normalization and "reference_waveform" in tensors:
        from triwave.data.dataset import TrainNormalizer

        normalizer = TrainNormalizer.from_dict(trainer.normalization)
        tensors["reference_waveform"] = normalizer.transform(tensors["reference_waveform"])
    condition = ConditionBatch(batch_size=batch_size, **tensors)
    if condition.structure is None and condition.reference_waveform is not None:
        condition = condition.to(str(trainer.device))
        condition.structure, condition.structure_valid = trainer.cwt(
            condition.reference_waveform, condition.reference_valid)
    generated = trainer.sample(condition, args.steps, args.guidance, args.seed).cpu()
    units = "synthetic_arbitrary_units"
    if trainer.normalization:
        from triwave.data.dataset import TrainNormalizer

        generated = TrainNormalizer.from_dict(trainer.normalization).inverse_transform(generated)
        units = "uV"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, waveform=generated.numpy(), units=units,
                        sample_rate=trainer.config.sample_rate,
                        lead_geometry=condition.lead_geometry.cpu().numpy(),
                        provenance="model generated; research output; clinical efficacy unmeasured")
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

    result = validate_sharded_dataset(args.root, config=_config(args),
                                      numerical_windows=args.numerical_windows, device=args.device)
    _emit(result)


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
    parser = argparse.ArgumentParser(description="SpikeWeave-EEG v0.2 research implementation")
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
            command.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                                 help="inverse-class-frequency sampling on source train only")
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
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--root", required=True)
    audit.add_argument("--config", default="configs/legion.yaml")
    audit.add_argument("--numerical-windows", type=int, default=16)
    audit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    audit.set_defaults(function=audit_dataset)
    medical = commands.add_parser("prepare-biomedclip")
    medical.add_argument("--output", default="weights/BiomedCLIP")
    medical.set_defaults(function=prepare_biomedclip)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--conditions", required=True)
    sampling.add_argument("--prompt")
    sampling.add_argument("--output", default="outputs/generated.npz")
    sampling.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
