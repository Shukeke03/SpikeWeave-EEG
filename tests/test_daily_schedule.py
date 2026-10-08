"""Daily VRAM handoffs preserve checkpoint boundaries and never overlap jobs."""

import importlib.util
import json
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_daily_schedule.py"
SPEC = importlib.util.spec_from_file_location("spikeweave_daily_schedule", SCRIPT)
daily = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daily
SPEC.loader.exec_module(daily)


def shanghai(text):
    return datetime.fromisoformat(text).replace(tzinfo=daily.SHANGHAI)


@pytest.mark.parametrize("now,profile,boundary", [
    ("2026-10-08T08:59:59", "night50", "2026-10-08T09:00:00"),
    ("2026-10-08T09:00:00", "day80", "2026-10-08T23:00:00"),
    ("2026-10-08T22:59:59", "day80", "2026-10-08T23:00:00"),
    ("2026-10-08T23:00:00", "night50", "2026-10-09T09:00:00"),
    ("2026-10-09T00:00:00", "night50", "2026-10-09T09:00:00"),
])
def test_clock_boundaries_and_next_day(now, profile, boundary):
    assert daily.profile_name(shanghai(now)) == profile
    assert daily.next_boundary(shanghai(now)) == shanghai(boundary)


def test_schedule_uses_shanghai_not_server_timezone():
    now = datetime(2026, 10, 8, 1, tzinfo=timezone.utc)
    assert daily.profile_name(now) == "day80"
    assert daily.next_boundary(now) == shanghai("2026-10-08T23:00:00")
    with pytest.raises(ValueError, match="timezone"):
        daily.profile_name(datetime(2026, 10, 8, 9))


def test_advance_stop_window_exact_boundary():
    assert not daily.within_stop_window(shanghai("2026-10-08T22:56:59"), 180)
    assert daily.within_stop_window(shanghai("2026-10-08T22:57:00"), 180)
    assert not daily.within_stop_window(shanghai("2026-10-08T23:00:00"), 180)


def saved_checkpoint(output, step, status="stopped", stage="pretrain"):
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"fake checkpoint; no EEG or torch allocation")
    daily.write_json(output / "run.json", {
        "stage": stage, "status": status, "completed_step": step,
        "latest_checkpoint_step": step, "checkpoint": str(checkpoint.resolve())})
    return checkpoint


def arguments(tmp_path, *, target=100000, checkpoint=True):
    base = {"model": {"denoiser_depth": 12, "denoiser_dim": 512, "denoiser_heads": 8},
            "training": {"stage": "pretrain", "learning_rate": 0.0001, "total_steps": target}}
    for name, accumulation, effective in [("day", 13, 260), ("night", 32, 256)]:
        recipe = {**base, "training": {**base["training"],
                  "gradient_accumulation": accumulation, "effective_batch_size": effective}}
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(recipe))
    cli = tmp_path / "spikeweave"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    manifest = tmp_path / "data"
    manifest.mkdir()
    source = (saved_checkpoint(tmp_path / "source", 1) if checkpoint
              else tmp_path / "absent" / "last.pt")
    return daily.parse_args([
        "--repo", str(tmp_path), "--cli", str(cli), "--manifest", str(manifest),
        "--checkpoint", str(source), "--output", str(tmp_path / "schedule"),
        "--day-config", "day.yaml", "--night-config", "night.yaml",
        "--target-steps", str(target)])


def test_commands_use_remaining_global_updates_and_measured_profiles(tmp_path):
    args = arguments(tmp_path)
    profiles = daily.profiles_for(args)
    for name, batch, fraction in [("day80", "20", "0.77"), ("night50", "8", "0.47")]:
        command = daily.child_command(args, profiles[name], args.checkpoint, args.output / name, 137)
        assert command[command.index("--steps") + 1] == "99863"
        assert command[command.index("--batch-size") + 1] == batch
        assert command[command.index("--cuda-memory-fraction") + 1] == fraction
        assert command[command.index("--stage") + 1] == "pretrain"
        assert "--fork-resume" in command and "--resume" not in command
        assert "--no-balanced" in command


def test_plan_only_does_not_read_checkpoint_start_process_or_allocate_cuda(tmp_path, capsys):
    args = arguments(tmp_path, checkpoint=False)
    result = daily.main([
        "--repo", str(tmp_path), "--manifest", "data", "--checkpoint", str(args.checkpoint),
        "--output", "schedule", "--day-config", "day.yaml", "--night-config", "night.yaml",
        "--plan-only"])
    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["timezone"] == "Asia/Shanghai"
    assert not args.output.exists()


@pytest.mark.parametrize("change", ["stage", "model", "schedule", "batch"])
def test_profile_validation_rejects_incompatible_recipes(tmp_path, change):
    args = arguments(tmp_path)
    recipe = yaml.safe_load(args.night_config.read_text())
    if change == "stage":
        recipe["training"]["stage"] = "align"
    elif change == "model":
        recipe["model"]["condition_dim"] = 32
    elif change == "schedule":
        recipe["training"]["learning_rate"] = 0.01
    else:
        recipe["training"]["effective_batch_size"] = 255
    args.night_config.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        daily.Controller(args)


class Clock:
    def __init__(self, now):
        self.value = now

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=60)


class FakeProcess:
    def __init__(self, command, clock, events, *, number, target, failure=False):
        self.pid = 9000000 + number
        self.command, self.clock, self.events = command, clock, events
        self.output = Path(command[command.index("--output") + 1])
        self.number, self.target, self.failure = number, target, failure
        self.code = None
        self.signaled = False
        (self.output / "metrics.jsonl").touch()

    def poll(self):
        if self.code is None and self.failure:
            self.code = 2
            self.events.append(("exit", self.number, self.clock.now()))
        elif self.code is None and (self.signaled or self.number == 2):
            step = self.target if self.number == 2 else 2
            saved_checkpoint(self.output, step, "complete" if self.number == 2 else "stopped")
            self.code = 0
            self.events.append(("exit", self.number, self.clock.now()))
        return self.code

    def send_signal(self, number):
        assert number == signal.SIGTERM
        self.signaled = True
        self.events.append(("stop", self.number, self.clock.now()))

    def wait(self):
        return self.poll()


def test_handoff_stops_early_saves_waits_for_boundary_and_never_overlaps(tmp_path):
    args = arguments(tmp_path, target=3)
    clock = Clock(shanghai("2026-10-08T22:56:00"))
    events, processes = [], []

    def popen(command, **kwargs):
        assert all(process.poll() is not None for process in processes)
        assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        number = len(processes) + 1
        events.append(("launch", number, clock.now()))
        process = FakeProcess(command, clock, events, number=number, target=3)
        processes.append(process)
        return process

    controller = daily.Controller(args, now=clock.now, sleep=clock.sleep, popen=popen)
    controller.run()
    assert events[1] == ("stop", 1, shanghai("2026-10-08T22:57:00"))
    second_launch = next(event for event in events if event[:2] == ("launch", 2))
    assert second_launch[2] == shanghai("2026-10-08T23:00:00")
    assert [event[:2] for event in events] == [
        ("launch", 1), ("stop", 1), ("exit", 1), ("launch", 2), ("exit", 2)]
    state = daily.read_json(args.output / "controller.json")
    assert state["status"] == "complete" and state["completed_step"] == 3
    assert Path(processes[1].command[processes[1].command.index("--fork-resume") + 1]) == (
        processes[0].output / "last.pt")
    assert args.output.joinpath("current").resolve() == processes[1].output
    assert daily.read_json(args.output / "current" / "launcher.json")["pid"] == processes[1].pid


def test_waiting_can_be_cancelled_without_launching_a_child(tmp_path):
    args = arguments(tmp_path)
    clock = Clock(shanghai("2026-10-08T08:57:00"))
    controller = daily.Controller(args, now=clock.now,
                                  popen=lambda *a, **kw: pytest.fail("child must not start"))
    controller.sleep = lambda seconds: controller.request_stop()
    controller.run()
    state = daily.read_json(args.output / "controller.json")
    assert state["status"] == "stopped" and state["completed_step"] == 1


def test_clock_jump_to_other_profile_stops_child_immediately(tmp_path):
    args = arguments(tmp_path, target=3)
    clock = Clock(shanghai("2026-10-08T12:00:00"))
    events, processes = [], []

    def jump(seconds):
        clock.value = shanghai("2026-10-09T00:00:00")

    def popen(command, **kwargs):
        assert all(process.poll() is not None for process in processes)
        number = len(processes) + 1
        process = FakeProcess(command, clock, events, number=number, target=3)
        processes.append(process)
        return process

    daily.Controller(args, now=clock.now, sleep=jump, popen=popen).run()
    assert next(event for event in events if event[0] == "stop") == (
        "stop", 1, shanghai("2026-10-09T00:00:00"))
    assert len(processes) == 2
    assert processes[1].command[processes[1].command.index("--batch-size") + 1] == "8"


def test_controller_signal_finishes_child_accumulation_then_stops(tmp_path):
    args = arguments(tmp_path)
    clock = Clock(shanghai("2026-10-08T12:00:00"))
    events, processes = [], []

    def popen(command, **kwargs):
        process = FakeProcess(command, clock, events, number=1, target=100000)
        processes.append(process)
        return process

    controller = daily.Controller(args, now=clock.now, popen=popen)
    controller.sleep = lambda seconds: controller.request_stop()
    controller.run()
    state = daily.read_json(args.output / "controller.json")
    assert state["status"] == "stopped" and state["completed_step"] == 2
    assert [event[0] for event in events] == ["stop", "exit"]
    assert len(processes) == 1


def test_child_failure_halts_without_restart_and_preserves_source(tmp_path):
    args = arguments(tmp_path)
    clock = Clock(shanghai("2026-10-08T12:00:00"))
    events, calls = [], []

    def popen(command, **kwargs):
        calls.append(command)
        process = FakeProcess(command, clock, events, number=1, target=100000, failure=True)
        # Fail during startup before the CLI signal handler can become ready.
        (process.output / "metrics.jsonl").unlink()
        return process

    controller = daily.Controller(args, now=clock.now, sleep=clock.sleep, popen=popen)
    with pytest.raises(RuntimeError, match="automatic restart is disabled"):
        controller.run()
    state = daily.read_json(args.output / "controller.json")
    assert state["status"] == "failed" and len(calls) == 1
    assert state["checkpoint"] == str(args.checkpoint)
    assert args.checkpoint.is_file()


def test_singleton_lock_and_surviving_child_prevent_duplicates(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    with daily.singleton_lock(args.output):
        with pytest.raises(RuntimeError, match="another daily controller"):
            daily.Controller(args).run()
    daily.write_json(args.output / "controller.json", {"child_pid": 123, "child_start_token": "same"})
    monkeypatch.setattr(daily, "process_is_alive", lambda pid, token: True)
    with pytest.raises(RuntimeError, match="refusing to overlap"):
        daily.Controller(args).run()


def test_checkpoint_stage_and_complete_update_are_required(tmp_path):
    checkpoint = saved_checkpoint(tmp_path / "run", 3, stage="align")
    with pytest.raises(ValueError, match="Stage A"):
        daily.checkpoint_info(checkpoint)
    saved_checkpoint(checkpoint.parent, 3, status="running")
    with pytest.raises(ValueError, match="must stop"):
        daily.checkpoint_info(checkpoint, require_stopped=True)
    metadata = daily.read_json(checkpoint.parent / "run.json")
    metadata["completed_step"] = 4
    daily.write_json(checkpoint.parent / "run.json", metadata)
    with pytest.raises(ValueError, match="complete saved"):
        daily.checkpoint_info(checkpoint)
