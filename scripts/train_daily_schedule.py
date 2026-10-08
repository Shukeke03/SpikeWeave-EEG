"""Run one Stage A job at a time with Shanghai daytime/nighttime VRAM recipes.

This controller reads the clock and child-process state, not training telemetry.
It gracefully checkpoints before each boundary and never changes an in-flight
CUDA allocator or gradient-accumulation recipe.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as wall_time
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import yaml

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Profile:
    name: str
    config: Path
    batch_size: int
    accumulation: int
    effective_batch: int
    cuda_fraction: float

    def to_dict(self) -> dict:
        return {"name": self.name, "config": str(self.config),
                "config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(),
                "batch_size": self.batch_size, "gradient_accumulation": self.accumulation,
                "effective_batch_size": self.effective_batch,
                "cuda_memory_fraction": self.cuda_fraction}


def local_time(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("schedule time must include a timezone")
    return now.astimezone(SHANGHAI)


def profile_name(now: datetime) -> str:
    local = local_time(now)
    return "day80" if 9 <= local.hour < 23 else "night50"


def next_boundary(now: datetime) -> datetime:
    """Return the next strictly future 09:00 or 23:00 Shanghai boundary."""
    local = local_time(now)
    morning = datetime.combine(local.date(), wall_time(9), SHANGHAI)
    evening = datetime.combine(local.date(), wall_time(23), SHANGHAI)
    if local < morning:
        return morning
    if local < evening:
        return evening
    return morning + timedelta(days=1)


def within_stop_window(now: datetime, lead_seconds: float) -> bool:
    return (next_boundary(now) - local_time(now)).total_seconds() <= lead_seconds


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def checkpoint_info(checkpoint: Path, *, require_stopped: bool = False) -> dict:
    """Check CLI metadata without loading EEG tensors, model weights or CUDA."""
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"checkpoint is missing or empty: {checkpoint}")
    metadata = read_json(checkpoint.parent / "run.json")
    if metadata.get("stage") != "pretrain":
        raise ValueError("daily scheduling only accepts Stage A/pretrain checkpoints")
    if require_stopped and metadata.get("status") not in {"stopped", "complete"}:
        raise ValueError("initial checkpoint owner must stop and save before scheduling")
    step = metadata.get("latest_checkpoint_step")
    if type(step) is not int or step < 0 or metadata.get("completed_step") != step:
        raise ValueError("checkpoint metadata must record a complete saved optimizer update")
    recorded_path = metadata.get("checkpoint")
    if recorded_path and Path(recorded_path).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint path differs from its run metadata")
    return {"checkpoint": str(checkpoint.resolve()), "completed_step": step}


def child_command(args: argparse.Namespace, profile: Profile, checkpoint: Path,
                  output: Path, completed_step: int) -> list[str]:
    remaining = args.target_steps - completed_step
    if remaining < 1:
        raise ValueError("there are no remaining optimizer updates")
    return [str(args.cli), "--seed", str(args.seed), "--threads", str(args.threads), "train",
            "--manifest", str(args.manifest), "--config", str(profile.config),
            "--fork-resume", str(checkpoint), "--device", "cuda", "--stage", "pretrain",
            "--data-purpose", "representation", "--no-balanced", "--amp",
            "--batch-size", str(profile.batch_size), "--steps", str(remaining),
            "--cuda-memory-fraction", str(profile.cuda_fraction),
            "--checkpoint-every", "100", "--checkpoint-seconds", "600",
            "--output", str(output)]


def profiles_for(args: argparse.Namespace) -> dict[str, Profile]:
    profiles = {"day80": Profile("day80", args.day_config, 20, 13, 260, 0.77),
                "night50": Profile("night50", args.night_config, 8, 32, 256, 0.47)}
    recipes = []
    for profile in profiles.values():
        recipe = yaml.safe_load(profile.config.read_text())
        model, training = recipe["model"], recipe["training"]
        if training.get("stage") != "pretrain":
            raise ValueError("daily profiles must use Stage A/pretrain")
        if (model.get("denoiser_depth"), model.get("denoiser_dim"),
                model.get("denoiser_heads")) != (12, 512, 8):
            raise ValueError("daily profiles must retain MCWave-M metadata (12 x 512, 8 heads)")
        if (training.get("gradient_accumulation"), training.get("effective_batch_size")) != (
                profile.accumulation, profile.effective_batch):
            raise ValueError(f"{profile.name} config differs from its measured batch recipe")
        fixed_training = {key: value for key, value in training.items()
                          if key not in {"gradient_accumulation", "effective_batch_size"}}
        recipes.append((model, fixed_training))
    if recipes[0] != recipes[1]:
        raise ValueError("daily profiles must share all model and fixed training fields")
    if recipes[0][1].get("total_steps") != args.target_steps:
        raise ValueError("controller target must equal the shared learning-rate schedule budget")
    return profiles


def process_start_token(pid: int) -> str | None:
    try:
        # Field 22 follows the final ')' even when a process name contains spaces.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def process_is_alive(pid: int, token: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return token is None or token == process_start_token(pid)


@contextlib.contextmanager
def singleton_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another daily controller owns this output directory") from error
        yield


class Controller:
    def __init__(self, args: argparse.Namespace, *,
                 now: Callable[[], datetime] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 popen: Callable = subprocess.Popen):
        self.args = args
        self.profiles = profiles_for(args)
        self.now = now or (lambda: datetime.now(SHANGHAI))
        self.sleep = sleep
        self.popen = popen
        self.stop_requested = False
        self.state: dict = {}

    def request_stop(self, number=None, frame=None) -> None:
        self.stop_requested = True

    def plan(self) -> dict:
        return {"schema": "spikeweave-daily-vram-v1", "timezone": "Asia/Shanghai",
                "stage": "pretrain", "target_optimizer_updates": self.args.target_steps,
                "seed": self.args.seed, "threads": self.args.threads,
                "day_interval": "09:00 inclusive - 23:00 exclusive",
                "night_interval": "23:00 inclusive - 09:00 exclusive",
                "stop_lead_seconds": self.args.stop_lead_seconds,
                "poll_seconds": self.args.poll_seconds,
                "cuda_allocator_config": "expandable_segments:True",
                "profiles": {key: profile.to_dict() for key, profile in self.profiles.items()},
                "repo": str(self.args.repo), "cli": str(self.args.cli),
                "manifest": str(self.args.manifest), "output": str(self.args.output),
                "initial_checkpoint": str(self.args.checkpoint),
                "switch_method": "graceful_checkpoint_exit_then_explicit_stage_a_recipe_fork",
                "data_iterator_on_switch": "restart; not sample-by-sample equivalent"}

    def update(self, **changes) -> None:
        if all(self.state.get(key) == value for key, value in changes.items()):
            return
        self.state.update(changes, updated_at=local_time(self.now()).isoformat())
        write_json(self.args.output / "controller.json", self.state)
        print(json.dumps({"time": self.state["updated_at"], **changes},
                         ensure_ascii=False, allow_nan=False), flush=True)

    def _run_segment(self, profile: Profile, checkpoint: Path, step: int) -> tuple[Path, int]:
        number = self.state.get("segment_count", 0) + 1
        output = self.args.output / (
            f"{number:04d}-{profile.name}-{local_time(self.now()):%Y%m%dT%H%M%S}")
        output.mkdir()
        command = child_command(self.args, profile, checkpoint, output, step)
        environment = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        self.update(status="starting", profile=profile.name, active_run=str(output),
                    segment_count=number, next_boundary=next_boundary(self.now()).isoformat())
        with (output / "train.log").open("a", buffering=1) as log:
            child = self.popen(command, cwd=self.args.repo, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True)
            try:
                self.update(status="running", child_pid=child.pid,
                            child_start_token=process_start_token(child.pid))
                write_json(output / "launcher.json", {
                    "pid": child.pid, "stage": "pretrain", "command": command,
                    "profile": profile.to_dict(), "schedule": str(self.args.output / "schedule.json"),
                    "controller_pid": os.getpid(), "initial_checkpoint": str(checkpoint),
                    "cuda_allocator_config": "expandable_segments:True"})
                current = self.args.output / "current"
                if current.exists() and not current.is_symlink():
                    raise ValueError("current must be a controller-owned symlink")
                temporary_link = self.args.output / ".current.tmp"
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(output.name, target_is_directory=True)
                temporary_link.replace(current)
                signal_sent = False
                handoff_requested = False
                while child.poll() is None:
                    now = self.now()
                    changing = profile_name(now) != profile.name
                    near_boundary = within_stop_window(now, self.args.stop_lead_seconds)
                    handoff_requested |= changing or near_boundary
                    if (self.stop_requested or handoff_requested) and not signal_sent:
                        # First metrics output happens after CLI installs graceful signal handling.
                        # An initialization failure is caught by child.poll(), without retrying.
                        if (output / "metrics.jsonl").is_file():
                            child.send_signal(signal.SIGTERM)
                            signal_sent = True
                            self.update(status="stopping", stop_reason=("controller_signal"
                                        if self.stop_requested else "clock_profile_boundary"))
                    self.sleep(self.args.poll_seconds)
                code = child.wait()
                if code != 0:
                    saved = output / "last.pt"
                    try:
                        self.state.update(checkpoint_info(saved))
                    except (OSError, ValueError, KeyError):
                        pass
                    raise RuntimeError(f"training child exited {code}; automatic restart is disabled")
                saved = output / "last.pt"
                info = checkpoint_info(saved, require_stopped=True)
                if info["completed_step"] <= step:
                    raise RuntimeError("training child saved no new complete optimizer update")
                if info["completed_step"] < self.args.target_steps and not (
                        signal_sent or self.stop_requested):
                    raise RuntimeError("training child ended before the target without a stop request")
                self.update(**info, child_pid=None, child_start_token=None, status="saved")
                return saved, info["completed_step"]
            finally:
                if child.poll() is None:
                    # An unexpected controller exception still releases the GPU before exit.
                    child.send_signal(signal.SIGTERM)
                    child.wait()

    def run(self) -> None:
        args = self.args
        if not args.cli.is_file() or not os.access(args.cli, os.X_OK):
            raise ValueError(f"training CLI is not executable: {args.cli}")
        if not args.manifest.is_dir():
            raise ValueError(f"dualbranch-v2 dataset directory is missing: {args.manifest}")
        with singleton_lock(args.output):
            plan = self.plan()
            schedule = args.output / "schedule.json"
            if schedule.exists() and read_json(schedule) != plan:
                raise ValueError("existing schedule differs; choose a new output directory")
            state_path = args.output / "controller.json"
            if state_path.exists():
                self.state = read_json(state_path)
                pid = self.state.get("child_pid")
                if pid and process_is_alive(pid, self.state.get("child_start_token")):
                    raise RuntimeError("previous training child is alive; refusing to overlap jobs")
            checkpoint = Path(self.state.get("checkpoint", str(args.checkpoint)))
            initial = checkpoint_info(checkpoint, require_stopped=True)
            write_json(schedule, plan)
            self.update(**initial, controller_pid=os.getpid(), child_pid=None,
                        child_start_token=None, status="starting", error=None)
            previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
            for number in previous:
                signal.signal(number, self.request_stop)
            try:
                step = initial["completed_step"]
                while not self.stop_requested and step < args.target_steps:
                    now = self.now()
                    if within_stop_window(now, args.stop_lead_seconds):
                        self.update(status="waiting", profile=profile_name(now),
                                    next_boundary=next_boundary(now).isoformat(), active_run=None)
                        self.sleep(args.poll_seconds)
                        continue
                    profile = self.profiles[profile_name(now)]
                    checkpoint, step = self._run_segment(profile, checkpoint, step)
                self.update(status="complete" if step >= args.target_steps else "stopped",
                            controller_pid=None, child_pid=None, child_start_token=None)
            except BaseException as error:
                self.update(status="failed", controller_pid=None, error=f"{type(error).__name__}: {error}")
                raise
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cli", type=Path, default=Path(sys.executable).with_name("spikeweave"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day-config", type=Path, default=Path("configs/legion-m-pretrain-80.yaml"))
    parser.add_argument("--night-config", type=Path, default=Path("configs/legion-m-pretrain-50.yaml"))
    parser.add_argument("--target-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stop-lead-seconds", type=float, default=180.0)
    parser.add_argument("--plan-only", action="store_true",
                        help="print the schedule without opening checkpoints or starting CUDA")
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 5 or not 0 <= args.stop_lead_seconds <= 3600:
        parser.error("poll-seconds must be 1..5 and stop-lead-seconds 0..3600")
    if args.target_steps < 1 or args.threads < 1:
        parser.error("target-steps and threads must be positive")
    args.repo = args.repo.resolve()
    for key in ("cli", "manifest", "checkpoint", "output", "day_config", "night_config"):
        path = getattr(args, key)
        setattr(args, key, (args.repo / path).resolve() if not path.is_absolute() else path.resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        controller = Controller(args)
        if args.plan_only:
            print(json.dumps({**controller.plan(), "profile_now": profile_name(controller.now()),
                              "next_boundary": next_boundary(controller.now()).isoformat()},
                             indent=2, ensure_ascii=False, allow_nan=False))
        else:
            controller.run()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Daily controller failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
