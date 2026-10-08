"""One read-only training snapshot, intended for watch over SSH."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(f"SpikeWeave-EEG | {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"Run: {args.run}\n")
    gpu = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        state = subprocess.run([gpu, "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
        print(f"GPU: {state.stdout.strip() or 'unavailable'}")
    except (OSError, subprocess.TimeoutExpired):
        print("GPU: unavailable")
    launch = read_json(args.run / "launcher.json")
    pid = launch.get("pid")
    try:
        active = b"spikeweave" in Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError, TypeError):
        active = False
    print(f"Process: {'RUNNING' if active else 'STOPPED'} | PID {pid}")
    config = read_json(args.run / "run.json")
    print(f"Stage: {config.get('stage', launch.get('stage', '--'))}")
    if config:
        print(f"Batch: {config.get('batch_size')} x {config.get('gradient_accumulation')} = {config.get('effective_batch_size')}")
        print(f"Target optimizer updates: {config.get('target_step')}")
    latest = {}
    try:
        with (args.run / "metrics.jsonl").open("rb") as stream:
            stream.seek(max(0, (args.run / "metrics.jsonl").stat().st_size - 32000))
            for line in stream.read().decode(errors="replace").splitlines():
                try:
                    latest = json.loads(line)
                except ValueError:
                    pass
    except OSError:
        pass
    for key in ("step", "microstep", "loss", "masked_patch", "spectral", "numeric_visual_reconstruction",
                "learning_rate", "elapsed_seconds", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib"):
        if key in latest:
            value = latest[key]
            print(f"{key}: {value:.6g}" if isinstance(value, float) else f"{key}: {value}")
    checkpoint = args.run / "last.pt"
    if checkpoint.exists():
        print(f"Checkpoint: {datetime.fromtimestamp(checkpoint.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print("\nRecent log:")
    try:
        with (args.run / "train.log").open("rb") as stream:
            stream.seek(max(0, (args.run / "train.log").stat().st_size - 12000))
            print("\n".join(stream.read().decode(errors="replace").splitlines()[-4:]))
    except OSError:
        print("Awaiting startup log.")
    print("\nCtrl+C stops this monitor; background training continues.")


if __name__ == "__main__":
    main()
