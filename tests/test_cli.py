"""Complete CLI wiring using artificial physical-unit files, never clinical data."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from triwave import cli
from triwave.config import ModelConfig
from triwave.data.dataset import EEGManifestDataset, TrainNormalizer, tokenize_texts
from triwave.data.manifest import SampleRecord, write_manifest
from triwave.training import Trainer


def run_cli(monkeypatch, capsys, *arguments):
    """Use the real argparse/main path while keeping the fixture in one process."""
    monkeypatch.setattr(sys, "argv", ["triwave", "--seed", "41", "--threads", "2", *map(str, arguments)])
    cli.main()
    output = capsys.readouterr().out
    reports = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert reports, output
    return reports


@pytest.fixture
def physical_manifest(tmp_path):
    """Two source files and an extreme-amplitude query from a different patient."""
    config = ModelConfig.tiny()
    generator = np.random.default_rng(91)
    geometry = generator.normal(size=(config.channels, 6)).astype(np.float32)
    lead_valid = np.ones(config.channels, dtype=bool)
    names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
    source_waveforms = [generator.normal(3, 20, (config.channels, config.samples)).astype(np.float32),
                        generator.normal(-2, 35, (config.channels, config.samples)).astype(np.float32)]
    records = []
    for index, waveform in enumerate(source_waveforms):
        sample_id = f"source-{index}"
        filename = f"{sample_id}.npz"
        np.savez_compressed(tmp_path / filename, waveform=waveform,
                            lead_geometry=geometry, lead_valid=lead_valid,
                            units="uV", provenance="synthetic engineering fixture")
        records.append(SampleRecord(
            sample_id=sample_id, dataset="synthetic-cli-fixture", subject_id="source-patient",
            record_id=f"source-record-{index}", waveform_path=filename,
            start_seconds=0, end_seconds=config.samples / config.sample_rate,
            sample_rate=config.sample_rate, lead_names=names, split="train", role="source",
            conditions={"reference": {"origin_id": f"source-{1 - index}",
                                      "target_derived": False, "inference_available": True},
                        "text": {"content": f"synthetic engineering event {index}",
                                 "inference_available": True},
                        "class": {"id": index, "label": "synthetic_engineering_event"}},
            provenance={"synthetic": True, "purpose": "CLI engineering fixture"}))
    query_path = tmp_path / "query.npz"
    np.savez_compressed(query_path,
                        waveform=np.full((config.channels, config.samples), 1e8, dtype=np.float32),
                        lead_geometry=geometry, lead_valid=lead_valid, units="uV",
                        provenance="synthetic query; must never fit normalization")
    records.append(SampleRecord(
        sample_id="query-extreme", dataset="synthetic-cli-fixture", subject_id="query-patient",
        record_id="query-record", waveform_path=query_path.name,
        start_seconds=0, end_seconds=config.samples / config.sample_rate,
        sample_rate=config.sample_rate, lead_names=names, split="test", role="query",
        provenance={"synthetic": True, "purpose": "normalization leakage sentinel"}))
    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    return config, manifest, source_waveforms, geometry, query_path


def test_manifest_train_checkpoint_reference_sample_and_source_only_statistics(
        tmp_path, monkeypatch, capsys, physical_manifest):
    config, manifest, source_waveforms, geometry, query_path = physical_manifest
    output = tmp_path / "manifest-training"
    reports = run_cli(monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
                      "--steps", "1", "--batch-size", "2", "--output", output)
    assert reports[-1]["status"] == "ok" and reports[-1]["step"] == 1
    assert reports[-1]["train_windows"] == 2
    assert reports[-1]["clinical_efficacy_evaluated"] is False
    checkpoint = output / "last.pt"
    restored = Trainer.load(checkpoint, resume_optimizer=True)
    assert restored.step == 1 and restored.optimizer.state
    saved = TrainNormalizer.from_dict(restored.normalization)
    assert saved.fit_sample_ids == ("source-0", "source-1")
    assert "query-extreme" not in saved.fit_sample_ids
    assert saved.count == 2 * config.channels * config.samples
    source_values = np.concatenate([waveform.astype(np.float64).ravel() for waveform in source_waveforms])
    np.testing.assert_allclose(saved.mean, 0.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(saved.std, np.sqrt(np.mean(source_values**2)), rtol=1e-12)

    # Change the held-out patient's amplitude by four orders of magnitude. Both
    # the stored and freshly fitted statistics must still equal source-only values.
    np.savez_compressed(query_path,
                        waveform=np.full((config.channels, config.samples), 1e12, dtype=np.float32),
                        lead_geometry=geometry, lead_valid=np.ones(config.channels, dtype=bool))
    raw_source = EEGManifestDataset(manifest, config, split="train", role="source")
    fresh = TrainNormalizer().fit(raw_source)
    assert fresh.to_dict() == saved.to_dict()

    condition_path = tmp_path / "conditions.npz"
    physical_reference = source_waveforms[0][None]
    np.savez_compressed(condition_path, lead_geometry=geometry[None],
                        reference_waveform=physical_reference,
                        reference_valid=np.ones((1, config.channels), dtype=bool),
                        reference_provenance=json.dumps({
                            "origin_id": "source-0", "target_derived": False,
                            "inference_available": True, "source_role": "train"}),
                        # Unknown teacher/target fields must not enter CLI conditioning.
                        target_waveform=np.full_like(physical_reference, 1e12),
                        teacher_cwt=np.full((1, config.channels, config.cwt_frequencies,
                                             config.cwt_times), 1e12, dtype=np.float32))
    prompt = "synthetic engineering event"
    generated_path = tmp_path / "sampled-physical.npz"
    sampled = run_cli(monkeypatch, capsys, "sample", "--checkpoint", checkpoint,
                      "--conditions", condition_path, "--prompt", prompt, "--device", "cpu",
                      "--steps", "2", "--output", generated_path)
    assert sampled[-1]["status"] == "ok" and sampled[-1]["units"] == "uV"
    ids, text_valid = tokenize_texts([prompt], config)
    conditions = restored.conditions({
        "lead_geometry": torch.from_numpy(geometry[None]),
        "reference_waveform": saved.transform(torch.from_numpy(physical_reference)),
        "reference_valid": torch.ones(1, config.channels, dtype=torch.bool),
        "reference_provenance": {"origin_id": "source-0", "target_derived": False,
                                 "inference_available": True, "source_role": "train"},
        "input_ids": ids, "text_valid": text_valid})
    expected = saved.inverse_transform(restored.sample(conditions, steps=2, seed=41))
    with np.load(generated_path, allow_pickle=False) as archive:
        assert archive["waveform"].shape == (1, config.channels, config.samples)
        assert str(archive["units"]) == "uV"
        assert archive["sample_rate"] == config.sample_rate
        assert np.isfinite(archive["waveform"]).all()
        np.testing.assert_array_equal(archive["lead_geometry"], geometry[None])
        # This also proves reference normalization and output inverse normalization
        # happen exactly once, and unknown target fields do not affect generation.
        np.testing.assert_allclose(archive["waveform"], expected.numpy(), rtol=1e-6, atol=1e-5)

    # Export explicit user requests with the same source scale and ontology.
    source_archive = tmp_path / "source-evaluation.npz"
    metadata = {"split": "train", "role": "source", "unit": "uV",
                "sample_rate": config.sample_rate, "lead_names": config.lead_names[:config.channels],
                "natural_distribution": True, "label_namespace": "engineering.binary",
                "label_scope": "synthetic_window", "scale_protocol": "source_rms_v2",
                "scale_fit_digest": saved.fit_manifest_digest,
                "class_prompts": {"0": "engineering no event", "1": "engineering event"}}
    np.savez_compressed(source_archive, waveforms=np.stack(source_waveforms), labels=[0, 1],
                        patient_ids=["source-patient", "source-patient"],
                        record_ids=["source-record-0", "source-record-1"],
                        sample_ids=["source-0", "source-1"], metadata=json.dumps(metadata))
    requests = tmp_path / "explicit-requests.npz"
    np.savez_compressed(requests, lead_geometry=np.stack([geometry, geometry]),
                        requested_classes=[0, 1])
    evaluation_output = tmp_path / "generated-evaluation.npz"
    run_cli(monkeypatch, capsys, "sample", "--checkpoint", checkpoint, "--conditions", requests,
            "--evaluation-source", source_archive, "--steps", "1", "--device", "cpu", "--batch-size", "1",
            "--output", evaluation_output)
    from triwave.evaluation import load_evaluation_data

    exported = load_evaluation_data(evaluation_output)
    source = load_evaluation_data(source_archive)
    assert exported.metadata["source_train_digest"] == source.digest
    assert exported.metadata["algorithm"] == "SpikeWeave-EEG-v0.3"
    assert exported.metadata["label_interpretation"] == "requested_condition_not_real_patient_diagnosis"
    np.testing.assert_array_equal(exported.labels, [0, 1])
    request_ids, request_valid = tokenize_texts(["engineering no event", "engineering event"], config)
    request_chunks = []
    for index in range(2):
        expected_requests = restored.conditions({"lead_geometry": torch.from_numpy(geometry[None]),
                                                 "input_ids": request_ids[index:index + 1],
                                                 "text_valid": request_valid[index:index + 1]})
        request_chunks.append(restored.sample(expected_requests, steps=1, seed=41 + index))
    request_waveforms = saved.inverse_transform(torch.cat(request_chunks))
    np.testing.assert_allclose(exported.waveforms, request_waveforms.numpy(), rtol=1e-6, atol=1e-5)
    assert exported.metadata["budget"]["sampling_microbatch"] == 1
    with pytest.raises(ValueError, match="disagree"):
        run_cli(monkeypatch, capsys, "sample", "--checkpoint", checkpoint, "--conditions", requests,
                "--prompt", "engineering no event", "--evaluation-source", source_archive,
                "--steps", "1", "--device", "cpu", "--output", tmp_path / "conflicting.npz")

    resumed_output = tmp_path / "manifest-resumed"
    resumed = run_cli(monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
                      "--resume", checkpoint, "--steps", "1", "--batch-size", "2",
                      "--output", resumed_output)
    assert resumed[-1]["step"] == 2
    continued = Trainer.load(resumed_output / "last.pt", resume_optimizer=True)
    assert continued.step == 2 and continued.normalization == restored.normalization


def test_doctor_and_smoke_resume_use_public_cli_api(tmp_path, monkeypatch, capsys):
    doctor = run_cli(monkeypatch, capsys, "doctor", "--device", "cpu")[-1]
    assert doctor["selected_device"] == "cpu"
    assert doctor["text_backend"] == "offline"
    assert doctor["clinical_pretraining"] is False
    assert doctor["parameters"] > 0
    output = tmp_path / "smoke"
    first = run_cli(monkeypatch, capsys, "smoke", "--device", "cpu", "--objective", "diffusion",
                    "--steps", "1", "--sampling-steps", "2", "--output", output)[-1]
    assert first["status"] == "ok" and first["step"] == 1
    assert first["data_kind"] == "synthetic_engineering_check"
    checkpoint = output / "last.pt"
    restored = Trainer.load(checkpoint, resume_optimizer=True)
    assert restored.normalization is None and restored.training.objective == "diffusion"
    resumed = tmp_path / "smoke-resumed"
    second = run_cli(monkeypatch, capsys, "smoke", "--device", "cpu", "--resume", checkpoint,
                     "--steps", "1", "--sampling-steps", "2", "--output", resumed)[-1]
    assert second["status"] == "ok" and second["step"] == 2
    assert second["objective"] == "diffusion"  # Resume keeps the saved probability path.
    with np.load(resumed / "generated.npz", allow_pickle=False) as archive:
        assert str(archive["units"]) == "synthetic_arbitrary_units"
        assert "no clinical efficacy measured" in str(archive["provenance"])
        assert np.isfinite(archive["waveform"]).all()


def test_cli_development_evaluation_missing_baselines_stays_closed(tmp_path, monkeypatch, capsys):
    config = ModelConfig.tiny()
    paths = {}
    rng = np.random.default_rng(101)
    for split, role in (("train", "source"), ("validation", "query")):
        path = tmp_path / f"{split}.npz"
        metadata = {"split": split, "role": role, "unit": "uV", "sample_rate": config.sample_rate,
                    "lead_names": config.lead_names[:config.channels], "natural_distribution": True,
                    "label_namespace": "engineering.binary", "label_scope": "synthetic_window"}
        np.savez_compressed(path, waveforms=rng.normal(size=(8, config.channels, config.samples)),
                            labels=[0, 1] * 4, patient_ids=[f"{split}-patient-{i // 4}" for i in range(8)],
                            record_ids=[f"{split}-record-{i}" for i in range(8)],
                            sample_ids=[f"{split}-window-{i}" for i in range(8)],
                            metadata=json.dumps(metadata))
        paths[split] = path
    ledger = tmp_path / "gates.json"
    shared = ("--train", paths["train"], "--validation", paths["validation"], "--ledger", ledger)
    audit = run_cli(monkeypatch, capsys, "evaluate", "--stage", "G0", *shared,
                    "--output", tmp_path / "g0.json")[-1]
    assert audit["gate_passed"] and not audit["test_used"]
    screening = run_cli(monkeypatch, capsys, "evaluate", "--stage", "G1", *shared,
                        "--updates", "2", "--batch-size", "4", "--output", tmp_path / "g1.json")[-1]
    assert not screening["gate_passed"] and not screening["test_used"]
    assert set(screening["downstream"]) == {"real_only", "repeat", "normalaug"}
    assert not json.loads(ledger.read_text())["gates"]["G1"]["passed"]
    with pytest.raises(ValueError, match="G2"):
        run_cli(monkeypatch, capsys, "evaluate", "--stage", "G3", "--ledger", ledger,
                "--operation", "engineering-final-test", "--output", tmp_path / "g3.json")


def test_condition_microbatch_preserves_per_record_provenance():
    values = {"lead_geometry": torch.zeros(3, 4, 6), "input_ids": torch.arange(12).reshape(3, 4),
              "structure_role": ("none", "reference_condition", "partial_observation"),
              "structure_provenance": [{"origin_id": f"sample-{i}"} for i in range(3)],
              "cwt_mask_protocol": "cwt-all-contribution-v2", "attribute_origin": {"origin_id": "user"}}
    selected = cli._condition_slice(values, 1, 3, 3)
    assert selected["lead_geometry"].shape == (2, 4, 6)
    assert selected["structure_role"] == ("reference_condition", "partial_observation")
    assert [row["origin_id"] for row in selected["structure_provenance"]] == ["sample-1", "sample-2"]
    assert selected["attribute_origin"] == {"origin_id": "user"}
    torch.testing.assert_close(selected["input_ids"], values["input_ids"][1:])


def test_manifest_periodic_checkpoints_resume_at_optimizer_boundaries(
        tmp_path, monkeypatch, capsys, physical_manifest):
    _, manifest, _, _, _ = physical_manifest
    output = tmp_path / "periodic-training"
    saved = []
    original_save = Trainer.save

    def record_save(trainer, path):
        original_save(trainer, path)
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["microstep"] % checkpoint["training_config"]["gradient_accumulation"] == 0
        assert checkpoint["step"] == trainer.step
        assert not path.with_suffix(path.suffix + ".tmp").exists()
        saved.append(trainer.step)

    monkeypatch.setattr(Trainer, "save", record_save)
    reports = run_cli(
        monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
        "--steps", "3", "--batch-size", "2", "--gradient-accumulation", "2",
        "--checkpoint-every", "2", "--checkpoint-seconds", "0", "--output", output)
    assert saved == [2, 3]
    assert reports[-1]["step"] == 3
    assert [row["step"] for row in reports if row.get("status") == "checkpoint_saved"] == [2]
    metrics = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 6
    assert [row["optimizer_updated"] for row in metrics] == [0, 1, 0, 1, 0, 1]
    assert all(row["elapsed_seconds"] >= row["microstep_seconds"] >= 0 for row in metrics)
    assert all((row["step_seconds"] is None) == (not row["optimizer_updated"]) for row in metrics)
    restored = Trainer.load(output / "last.pt", resume_optimizer=True)
    assert restored.step == 3 and restored.microstep == 6 and restored.optimizer.state
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete" and run["completed_step"] == run["target_step"] == 3
    assert run["effective_batch_size"] == 4 and run["gradient_accumulation"] == 2
    assert run["steps_semantics"] == "additional_optimizer_updates"
    assert run["data"]["metadata_sha256"] == cli._file_digest(manifest)
    assert "clinical" not in json.dumps(run["args"])

    resumed = run_cli(
        monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
        "--steps", "2", "--batch-size", "2", "--resume", output / "last.pt",
        "--checkpoint-every", "2", "--checkpoint-seconds", "0", "--output", output)
    assert saved == [2, 3, 5]
    assert resumed[-1]["step"] == 5
    continued = Trainer.load(output / "last.pt", resume_optimizer=True)
    assert continued.step == 5 and continued.microstep == 10 and continued.optimizer.state
    assert continued.run_provenance["source_data_identity"] == restored.run_provenance["source_data_identity"]
    run = json.loads((output / "run.json").read_text())
    assert run["start_step"] == 3 and run["target_step"] == run["completed_step"] == 5
    assert run["resume_data_iterator"] == "restart"


def test_manifest_time_checkpoint_and_output_recipe_guards(
        tmp_path, monkeypatch, capsys, physical_manifest):
    _, manifest, _, _, _ = physical_manifest
    output = tmp_path / "time-checkpoint"
    clock = iter(range(100))
    monkeypatch.setattr(cli.time, "monotonic", lambda: float(next(clock)))
    reports = run_cli(
        monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
        "--steps", "2", "--batch-size", "2", "--gradient-accumulation", "2",
        "--checkpoint-every", "0", "--checkpoint-seconds", "0.1", "--output", output)
    assert [row["step"] for row in reports if row.get("status") == "checkpoint_saved"] == [1, 2]
    before = (output / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="nonempty"):
        run_cli(monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
                "--steps", "1", "--output", output)
    with pytest.raises(ValueError, match="training options differ"):
        run_cli(monkeypatch, capsys, "train", "--manifest", manifest, "--device", "cpu",
                "--resume", output / "last.pt", "--gradient-accumulation", "1",
                "--steps", "1", "--output", tmp_path / "conflicting-recipe")
    assert (output / "last.pt").read_bytes() == before
    modified_manifest = tmp_path / "modified-manifest.jsonl"
    modified_manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="data identity differs"):
        run_cli(monkeypatch, capsys, "train", "--manifest", modified_manifest, "--device", "cpu",
                "--resume", output / "last.pt", "--steps", "1", "--batch-size", "2",
                "--output", tmp_path / "conflicting-data")


def test_cuda_memory_fraction_is_explicit_and_applied_before_training(monkeypatch):
    applied = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: applied.append((fraction, device)))
    cli._set_cuda_memory_limit(SimpleNamespace(), torch.device("cpu"))
    assert applied == []
    for fraction in (0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fraction"):
            cli._set_cuda_memory_limit(SimpleNamespace(cuda_memory_fraction=fraction), torch.device("cuda"))
    with pytest.raises(ValueError, match="requires a CUDA"):
        cli._set_cuda_memory_limit(SimpleNamespace(cuda_memory_fraction=0.43), torch.device("cpu"))
    cli._set_cuda_memory_limit(SimpleNamespace(cuda_memory_fraction=0.43), torch.device("cuda:0"))
    assert applied == [(0.43, torch.device("cuda:0"))]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    cli._set_cuda_memory_limit(SimpleNamespace(cuda_memory_fraction=0.47), torch.device("cuda"))
    assert applied[-1] == (0.47, torch.device("cuda:2"))


def test_manifest_training_preserves_precreated_launcher_files(
        tmp_path, monkeypatch, capsys, physical_manifest):
    _, manifest, _, _, _ = physical_manifest
    output = tmp_path / "background-training"
    output.mkdir()
    log = output / "train.log"
    launch = output / "launcher.json"
    log.write_text("launcher stdout fixture\n")
    launch.write_text('{"background": true}\n')
    reports = run_cli(monkeypatch, capsys, "train", "--manifest", manifest,
                      "--device", "cpu", "--steps", "1", "--batch-size", "2", "--output", output)
    assert reports[-1]["status"] == "ok" and (output / "last.pt").is_file()
    assert log.read_text() == "launcher stdout fixture\n"
    assert launch.read_text() == '{"background": true}\n'


@pytest.mark.parametrize("stage", ["pretrain", "align"])
def test_representation_smoke_saves_without_untrained_diffusion_sampling(
        stage, tmp_path, monkeypatch, capsys):
    output = tmp_path / stage
    report = run_cli(monkeypatch, capsys, "smoke", "--device", "cpu", "--stage", stage,
                     "--steps", "1", "--output", output)[-1]
    assert report["status"] == "ok" and report["step"] == 1
    assert report["representation_only"] is True
    assert report["generated_shape"] is report["sampler"] is None
    assert (output / "last.pt").is_file() and not (output / "generated.npz").exists()
    restored = Trainer.load(output / "last.pt", resume_optimizer=True)
    assert restored.step == 1 and restored.active_stage == stage
