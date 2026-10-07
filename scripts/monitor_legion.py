"""Read-only live status: run with system Python through watch over SSH."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def process_alive(info: dict | None, marker: bytes) -> bool:
    if not info:
        return False
    try:
        return marker in Path(f"/proc/{int(info['pid'])}/cmdline").read_bytes()
    except (OSError, KeyError, ValueError):
        return False


def show(base: Path) -> None:
    metadata, processed = base / "metadata", base / "processed/v1"
    print(f"TriWave-EEG / legion    {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"Data: {base}\n")
    inventory = read_json(metadata / "official-inventory.json", {"items": []})["items"]
    edfs = [entry for entry in inventory if entry["path"].endswith(".edf")]
    completed = []
    for entry in edfs:
        path = base / "raw" / entry["path"]
        try:
            if path.stat().st_size == entry["size"] and not Path(str(path) + ".aria2").exists():
                completed.append(entry)
        except OSError:
            pass
    expected = sum(entry["size"] for entry in edfs)
    received = sum(entry["size"] for entry in completed)
    download = read_json(metadata / "download-process.json")
    downloading = process_alive(download, b"aria2c")
    speed = "--"
    if downloading:
        try:
            log = metadata / "aria2-download.log"
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 16000))
                lines = handle.read().decode(errors="replace").splitlines()
            aggregate = [line for line in lines if line.startswith("[DL:")]
            if aggregate:
                speed = re.search(r"DL:([^\]]+)", aggregate[-1]).group(1) + "/s (latest log)"
            else:
                transfers = [line for line in lines if " DL:" in line]
                if transfers:
                    speed = re.search(r"DL:(\S+)", transfers[-1]).group(1) + "/s (single transfer)"
        except (OSError, AttributeError):
            pass
    complete = bool(edfs) and len(completed) == len(edfs)
    download_state = "FINISHED (await processor SHA ledger)" if complete else "RUNNING" if downloading else "STOPPED / incomplete"
    percent = 100 * received / expected if expected else 0
    print(f"Download: {download_state}")
    print(f"  EDF {len(completed)}/{len(edfs)} | completed files {received / 1024**3:.2f}/{expected / 1024**3:.2f} GiB ({percent:.1f}%)")
    print(f"  Speed: {speed} | PID: {(download or {}).get('pid', '--')}")
    print("  Progress counts full files; incomplete segments are additional resumable bytes.")
    statuses = Counter()
    done_count = 0
    for path in (processed / "records").glob("*/done.json"):
        done = read_json(path)
        if done:
            statuses[done["record"]["status"]] += 1
            done_count += int(done.get("complete", False))
    job = read_json(metadata / "preprocess-process.json")
    state = read_json(metadata / "dataset-status.json", {})
    print(f"\nProcessing: {state.get('state', 'not started')} | process {'RUNNING' if process_alive(job, b'triwave.data.pipeline') else 'STOPPED'}")
    print(f"  Completed record ledger: {done_count}/{len(edfs)} | {dict(statuses)}")
    report = read_json(processed / "build_report.json", {})
    if report:
        print(f"  Published windows: ready={report.get('ready_windows', '--')}, isolated={report.get('isolated_windows', '--')}")
    if state.get("error"):
        print(f"  Error: {state['error']}")
    audit = read_json(processed / "dataset_audit.json", {})
    if audit:
        print(f"  Final audit: {audit.get('status', '--')}")
    print(f"\nCPU: {os.cpu_count()} logical cores | load 1/5/15 min: {' / '.join(f'{n:.2f}' for n in os.getloadavg())}")
    try:
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        used = memory["MemTotal"] - memory["MemAvailable"]
        print(f"RAM: {used / 1024**2:.2f}/{memory['MemTotal'] / 1024**2:.2f} GiB | available {memory['MemAvailable'] / 1024**2:.2f} GiB")
    except (OSError, ValueError, KeyError):
        pass
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=2)
        print(f"GPU: {gpu.stdout.strip() or 'unavailable'}")
    except (OSError, subprocess.TimeoutExpired):
        print("GPU: unavailable")
    disk = os.statvfs(base)
    print(f"Disk: available {disk.f_bavail * disk.f_frsize / 1024**3:.1f} GiB")
    print("\nCtrl+C closes the monitor; background download and processing continue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/home/shuxing/workspace/datasets/TriWave-EEG/chbmit-1.0.0"))
    show(parser.parse_args().base)
