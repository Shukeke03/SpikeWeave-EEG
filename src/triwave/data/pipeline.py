"""Persistent incremental processing while the official mirror download runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from triwave.config import ModelConfig

from .build import _atomic_json, build_chbmit
from .sharded import fit_sharded_statistics
from .validation import validate_sharded_dataset


def run_pipeline(base: str | Path, config: ModelConfig, *, split_map: str | Path | None = None,
                 workers: int = 2, interval: float = 120.0, device: str = "cpu",
                 max_passes: int | None = None) -> dict:
    """Publish final statistics only after complete source verification/processing.

    A partial index is explicitly a preview; the normal training loader rejects it.
    The advisory lock prevents two writers from publishing mixed index generations.
    """
    if interval <= 0 or workers <= 0 or (max_passes is not None and max_passes <= 0):
        raise ValueError("positive interval, workers and max_passes are required")
    config.validate()
    base = Path(base)
    output = base / "processed" / "v1"
    metadata = base / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    status_path = metadata / "dataset-status.json"
    started = time.time()
    with (metadata / "preprocess.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for iteration in range(1, (max_passes or 1000000) + 1):
            state = {"state": "processing", "pid": os.getpid(), "started_unix": started,
                     "updated_unix": time.time(), "pass": iteration, "output": str(output)}
            _atomic_json(status_path, state)
            try:
                build_chbmit(base / "raw", output, split_map, samples=config.samples,
                             sample_rate=config.sample_rate, workers=workers)
                build = json.loads((output / "build_report.json").read_text())
                state.update({"updated_unix": time.time(), "records": build["records"],
                              "ready_windows": build["ready_windows"],
                              "isolated_windows": build["isolated_windows"],
                              "source_download_complete": build["source_download_complete"],
                              "source_fully_processed": build["source_fully_processed"]})
                print(json.dumps(state, allow_nan=False), flush=True)
                if build["source_download_complete"] and build["source_fully_processed"]:
                    state["state"] = "fitting_training_statistics"
                    _atomic_json(status_path, state)
                    normalizer = fit_sharded_statistics(output, config)
                    temporary = output / "train_normalizer.json.partial"
                    normalizer.save(temporary)
                    os.replace(temporary, output / "train_normalizer.json")
                    state["state"] = "auditing"
                    _atomic_json(status_path, state)
                    audit = validate_sharded_dataset(output, config=config,
                                                     numerical_windows=16, device=device)
                    accepted = audit["status"] == "complete" and audit["completeness"]["complete"]
                    state.update({"state": "complete" if accepted else "needs_attention",
                                  "updated_unix": time.time(), "audit_status": audit["status"],
                                  "audit_errors": audit["errors"], "storage": audit["storage"]})
                    _atomic_json(status_path, state)
                    print(json.dumps(state, allow_nan=False), flush=True)
                    return state
                if build["source_download_complete"] and build["records"]["failed"]:
                    state.update({"state": "needs_attention", "failed_records": build["failed_records"]})
                    _atomic_json(status_path, state)
                    return state
                state["state"] = "waiting_for_download"
                _atomic_json(status_path, state)
            except Exception as error:
                state.update({"state": "failed", "updated_unix": time.time(),
                              "error": f"{type(error).__name__}: {error}"})
                _atomic_json(status_path, state)
                raise
            if max_passes is not None and iteration == max_passes:
                return state
            time.sleep(interval)
    raise RuntimeError("pipeline iteration limit reached")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-map")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    state = run_pipeline(args.base, ModelConfig.from_yaml(args.config), split_map=args.split_map,
                         workers=args.workers, interval=args.interval, device=args.device)
    if state["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
