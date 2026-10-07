"""Complete CLI wiring using artificial physical-unit files, never clinical data."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from triwave import cli
from triwave.config import ModelConfig
from triwave.data.dataset import EEGManifestDataset, TrainNormalizer, tokenize_texts
from triwave.data.manifest import SampleRecord, write_manifest
from triwave.training import Trainer
from triwave.types import ConditionBatch


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
    np.testing.assert_allclose(saved.mean, source_values.mean(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(saved.std, source_values.std(), rtol=1e-12)

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
    conditions = ConditionBatch(
        batch_size=1, lead_geometry=torch.from_numpy(geometry[None]),
        reference_waveform=saved.transform(torch.from_numpy(physical_reference)),
        reference_valid=torch.ones(1, config.channels, dtype=torch.bool),
        input_ids=ids, text_valid=text_valid)
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
    first = run_cli(monkeypatch, capsys, "smoke", "--device", "cpu", "--objective", "flow",
                    "--steps", "1", "--sampling-steps", "2", "--output", output)[-1]
    assert first["status"] == "ok" and first["step"] == 1
    assert first["data_kind"] == "synthetic_engineering_check"
    checkpoint = output / "last.pt"
    restored = Trainer.load(checkpoint, resume_optimizer=True)
    assert restored.normalization is None and restored.training.objective == "flow"
    resumed = tmp_path / "smoke-resumed"
    second = run_cli(monkeypatch, capsys, "smoke", "--device", "cpu", "--resume", checkpoint,
                     "--steps", "1", "--sampling-steps", "2", "--output", resumed)[-1]
    assert second["status"] == "ok" and second["step"] == 2
    assert second["objective"] == "flow"  # Resume keeps the saved probability path.
    with np.load(resumed / "generated.npz", allow_pickle=False) as archive:
        assert str(archive["units"]) == "synthetic_arbitrary_units"
        assert "no clinical efficacy measured" in str(archive["provenance"])
        assert np.isfinite(archive["waveform"]).all()
