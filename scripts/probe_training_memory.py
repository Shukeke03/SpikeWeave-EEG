"""Bounded real-source Stage A updates; no clinical efficacy claim."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from triwave.cli import _collate
from triwave.config import ModelConfig
from triwave.data.dataset import TrainNormalizer
from triwave.data.dualbranch import DualBranchEEGDataset
from triwave.data.sharded import ShardedEEGDataset
from triwave.runtime import seed_everything
from triwave.training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.43)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("invalid batch size or CUDA memory fraction")
    seed_everything(17, 4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    config = ModelConfig.from_yaml(args.config)
    fields = yaml.safe_load(Path(args.config).read_text())["training"]
    training = replace(TrainingConfig(**fields), stage="pretrain", gradient_accumulation=1,
                       effective_batch_size=args.batch_size)
    if (args.data / "dualbranch_manifest.json").is_file():
        dataset = DualBranchEEGDataset(args.data, config, split="train", purpose="representation")
        normalizer = dataset.normalizer
    else:
        normalizer = TrainNormalizer.load(args.data / "train_normalizer.json")
        dataset = ShardedEEGDataset(args.data, config, split="train", normalizer=normalizer)
    trainer = Trainer(config, training, "cuda")
    trainer.normalization = normalizer.to_dict()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for update in range(2):
        # Spaced source windows, avoiding a probe dominated by one recording.
        positions = [(update * args.batch_size + j) * len(dataset) // (2 * args.batch_size)
                     for j in range(args.batch_size)]
        results.append(trainer.train_step(_collate([dataset[i] for i in positions])))
    torch.cuda.synchronize()
    gpu = subprocess.run(
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(json.dumps({"batch_size": args.batch_size, "updates": trainer.step,
                      "actual_stage": trainer.active_stage, "source_windows": len(dataset),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                      "gpu_used_total_mib": gpu, "seconds": time.monotonic() - started,
                      "parameters": sum(p.numel() for p in trainer.model.parameters()),
                      "medically_pretrained": trainer.model.condition_encoder.is_medically_pretrained,
                      "metrics": results, "clinical_efficacy_evaluated": False}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
