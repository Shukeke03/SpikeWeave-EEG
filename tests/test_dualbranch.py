"""Engineering fixtures for v2 isolation, atomic side tables and source scale."""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from triwave.config import ModelConfig
from triwave.data.build import INDEX_DTYPE
from triwave.data.dualbranch import (
    MANIFEST_NAME,
    TABLE_NAMES,
    DualBranchEEGDataset,
    PatientEventBalancedSampler,
    approve_hard_montage,
    build_dualbranch_dataset,
    validate_dualbranch_dataset,
)
from triwave.data.sharded import fit_sharded_statistics


def make_v1(root, *, unlabelled=False, broken_closure=False, native128=False):
    root.mkdir()
    content = ModelConfig.tiny().to_dict()
    content.update(samples=1024, sample_rate=256, cwt_frequencies=64, cwt_times=128,
                   cwt_max_frequency=80, attribute_dim=16, visual_patch=16,
                   montage_mode="audit", lead_names=("FP1-F7", "F7-T7", "T7-P7", "P7-FP1"))
    config = ModelConfig(**content)
    config.validate()
    records, rows, isolated = [], np.zeros(8, dtype=INDEX_DTYPE), np.empty(0, dtype=INDEX_DTYPE)
    for i, (patient, split, amplitude) in enumerate([
            ("chb02", "train", 10), ("chb03", "train", 30),
            ("chb04", "test", 1000), ("chb08", "validation", 500),
            *([("chb09", "train", 50)] if unlabelled else [])]):
        folder = root / "records" / str(i)
        folder.mkdir(parents=True)
        time = np.arange(2048) / config.sample_rate
        nodes = np.stack([amplitude * np.sin(2 * np.pi * (6 + j) * time + j) for j in range(4)])
        if native128:
            noise = np.random.default_rng(i).normal(size=nodes.shape)
            spectrum = np.fft.rfft(noise)
            spectrum[:, np.fft.rfftfreq(2048, 1 / config.sample_rate) >= 60] = 0
            nodes += amplitude * .1 * np.fft.irfft(spectrum, n=2048)
        x = (nodes - np.roll(nodes, -1, axis=0)).astype(np.float32)
        if broken_closure and i == 0:
            x[0] += 5
        np.save(folder / "waveform.npy", x)
        unknown = i == 4
        records.append({"index": i, "record_id": f"CHB-MIT:{patient}/fixture.edf", "subject_id": patient,
            "canonical_subject_id": patient, "split": split, "role": "source" if split == "train" else "query",
            "status": "isolated" if unknown else "ready", "source_sha256": hashlib.sha256(patient.encode()).hexdigest(),
            "source_file_id": f"{patient}/fixture.edf", "source_verified": True, "processed": True,
            "waveform_available": True, "waveform_path": f"records/{i}/waveform.npy", "waveform_shape": list(x.shape),
            "unit": "uV", "sample_rate": 256, "source_sample_rate": 128 if native128 and i == 0 else 256,
            "window_samples": 1024,
            "lead_names": list(config.lead_names), "lead_valid": [True] * 4,
            "lead_geometry": (np.arange(24).reshape(4, 6) / 100).tolist(), "annotation_known": not unknown,
            "annotation_source": "engineering_fixture", "isolation_reasons": ["unknown_annotation"] if unknown else [],
            "events": [] if unknown else [{"event_id": f"event-{i}", "onset": 4.5, "offset": 5.5}],
            "indexed_windows": 2, "incomplete_tail_samples": 0})
        if unknown:
            isolated = np.zeros(2, dtype=INDEX_DTYPE)
            isolated["record"], isolated["offset"], isolated["label"], isolated["ref_index"] = i, [0, 1024], -1, -1
        else:
            rows[2 * i:2 * i + 2]["record"] = i
            rows[2 * i:2 * i + 2]["offset"] = [0, 1024]
            rows[2 * i:2 * i + 2]["label"] = [0, 1]
    rows["ref_index"] = [2, 3, 0, 1, 0, 1, 0, 1]  # Legacy label-dependent bank must be ignored.
    protocol = {"unit": "uV", "sample_rate": 256, "window_samples": 1024,
                "lead_names": list(config.lead_names), "exclude_qc_bits": 3,
                "evaluation_protocol": "source-only", "bandpass": [.1, 80]}
    for name, value in {
            "records.json": {"records": records}, "protocol.json": protocol,
            "build_report.json": {"source_download_complete": True, "source_fully_processed": True},
            "split_lock.json": {"patients": {r["canonical_subject_id"]: r["split"] for r in records}}}.items():
        (root / name).write_text(json.dumps(value))
    np.save(root / "window_index.npy", rows)
    np.save(root / "isolated_index.npy", isolated)
    fit_sharded_statistics(root, config).save(root / "normalizer.json")
    return config


def test_streaming_migration_source_scale_tables_query_isolation_and_resume(tmp_path):
    source, output = tmp_path / "v1", tmp_path / "v2"
    config = make_v1(source)
    physical = source / "records/0/waveform.npy"
    mtime = physical.stat().st_mtime_ns
    build_dualbranch_dataset(source, output, config, batch_size=2)
    assert physical.stat().st_mtime_ns == mtime
    assert {path.stem for path in output.glob("*.parquet")} == set(TABLE_NAMES)
    report = validate_dualbranch_dataset(output, config, numerical_windows=2)
    assert report["valid"] and report["numerical"]["windows"] == 2
    assert report["numerical_query_reads"] == 0
    train = DualBranchEEGDataset(output, config, purpose="training")
    assert train.normalizer.std < 50
    item = train[0]
    assert item["teacher_attribute_valid"].any()
    assert not item["attribute_valid"].any() and not item["condition_attribute_valid"].any()
    assert item["teacher_attribute_origin"]["target_derived"]
    assert item["metadata"]["reference_audit"]["selection_used_query_label"] is False
    assert item["reference_provenance"]["origin_id"] != item["metadata"]["sample_id"]
    query = DualBranchEEGDataset(output, config, split="test")[0]
    assert query["metadata"]["purpose"] == "source_only"
    assert not query["text_valid"].any() and not query["reference_valid"].any()
    assert not query["teacher_attribute_valid"].any() and not query["attribute_valid"].any()
    assert not query["structure_visible"].any()
    before = (output / MANIFEST_NAME).stat().st_mtime_ns
    assert build_dualbranch_dataset(source, output, config, batch_size=1) == output
    assert (output / MANIFEST_NAME).stat().st_mtime_ns == before
    assert approve_hard_montage(output, config)


def test_partial_structure_strict_patch_visibility_and_collate(tmp_path):
    config = make_v1(tmp_path / "v1")
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config, batch_size=2)
    data = DualBranchEEGDataset(tmp_path / "v2", config, purpose="partial_observation", seed=4)
    item = data[0]
    assert item["structure_role"] == "partial_observation"
    assert item["structure_provenance"]["target_derived"]
    assert item["structure_visible"].any()
    assert (item["structure_visible"] <= item["structure_valid"]).all()
    assert (item["structure"][~item["structure_visible"]] == 0).all()
    provenance = item["structure_provenance"]
    assert abs(provenance["observed_patch_fraction"] - .2) < .05
    patch_valid = item["structure_valid"].reshape(4, 4, 16, 8, 16).all((2, 4))
    patch_visible = item["structure_visible"].reshape(4, 4, 16, 8, 16).all((2, 4))
    assert (patch_visible <= patch_valid).all()
    torch.testing.assert_close(data[0]["structure"], item["structure"])
    # Row-group readers and preindexed pools survive DataLoader spawn pickling.
    restored = pickle.loads(pickle.dumps(data))
    torch.testing.assert_close(restored[0]["target_waveform"], item["target_waveform"])
    from triwave.cli import _collate
    collated = _collate([data[0], data[1]])
    assert collated["structure"].shape == (2, 4, 64, 128)
    assert collated["structure_role"] == ["partial_observation"] * 2


def test_explicit_request_donor_class_independent_of_query_truth_and_leak_rejection(tmp_path):
    config = make_v1(tmp_path / "v1")
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config)
    data = DualBranchEEGDataset(tmp_path / "v2", config, split="test",
                               request={"reference": True, "requested_class": 1})
    item = data[0]  # Query's annotated class is 0, requested class is 1.
    assert item["class_ids"].item() == 0
    donor = item["metadata"]["reference_audit"]["reference_index"]
    assert data.window_index[donor]["label"] == 1
    assert item["text_valid"].any()
    with pytest.raises(ValueError, match="target-derived"):
        DualBranchEEGDataset(tmp_path / "v2", config, split="test", request={"target_derived": True})
    with pytest.raises(ValueError, match="source training"):
        DualBranchEEGDataset(tmp_path / "v2", config, split="test", purpose="representation")
    with pytest.raises(ValueError, match="source request bank"):
        DualBranchEEGDataset(tmp_path / "v2", config, request={"reference": True, "reference_index": 4})[0]
    with pytest.raises(ValueError, match="target patient"):
        DualBranchEEGDataset(tmp_path / "v2", config, request={"reference": True, "reference_index": 0})[0]


def test_unlabelled_is_explicit_source_representation_population_not_negative_label(tmp_path):
    config = make_v1(tmp_path / "v1", unlabelled=True)
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config,
                            include_unlabelled_source=True, batch_size=2)
    data = DualBranchEEGDataset(tmp_path / "v2", config, purpose="representation")
    assert len(data) == 6 and data.normalizer.count == 6 * 4 * 1024
    item = data[-1]
    assert item["class_ids"].item() == -1
    assert not item["text_valid"].any() and not item["alignment_metadata"]["semantic_reliable"]
    generation = DualBranchEEGDataset(tmp_path / "v2", config, purpose="training")
    assert len(generation) == 4
    sampler = PatientEventBalancedSampler(data, seed=3)
    assert len(list(sampler)) == 6
    labelled = PatientEventBalancedSampler(generation, num_samples=64, seed=2)
    assert set(list(labelled)) <= set(range(4))
    assert torch.isclose(labelled.weights[:2].sum(), labelled.weights[2:].sum())
    with pytest.raises(ValueError, match="training-source"):
        PatientEventBalancedSampler(DualBranchEEGDataset(tmp_path / "v2", config, split="test"))


def test_failed_publication_and_tampered_or_failed_audit_cannot_approve_hard(tmp_path, monkeypatch):
    from triwave.data import dualbranch
    config = make_v1(tmp_path / "v1", broken_closure=True)
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config)
    with pytest.raises(ValueError, match="whole-source"):
        approve_hard_montage(tmp_path / "v2", config)
    path = tmp_path / "v2" / "montage_audit.json"
    value = json.loads(path.read_text())
    value["hard_mode_approved"] = True
    path.write_text(json.dumps(value))
    assert not validate_dualbranch_dataset(tmp_path / "v2")["valid"]
    with pytest.raises(ValueError, match="integrity"):
        approve_hard_montage(tmp_path / "v2", config)
    monkeypatch.setattr(dualbranch, "waveform_attributes", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failure fixture")))
    with pytest.raises(RuntimeError, match="failure fixture"):
        build_dualbranch_dataset(tmp_path / "v1", tmp_path / "failed", config)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.building-*"))


def test_scale_identity_excludes_query_labels_and_legacy_reference_policy(tmp_path):
    source = tmp_path / "v1"
    config = make_v1(source)
    build_dualbranch_dataset(source, tmp_path / "a", config)
    a = DualBranchEEGDataset(tmp_path / "a", config)
    rows = np.load(source / "window_index.npy")
    rows["label"][4:] = 1 - rows["label"][4:]
    rows["ref_index"] = -1
    np.save(source / "window_index.npy", rows)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["reference_seed"] = 91
    protocol["reference_policy"] = "changed_nonphysical_policy"
    (source / "protocol.json").write_text(json.dumps(protocol))
    build_dualbranch_dataset(source, tmp_path / "b", config)
    b = DualBranchEEGDataset(tmp_path / "b", config)
    assert a.normalizer.std == pytest.approx(b.normalizer.std)
    assert a.normalizer.fit_manifest_digest == b.normalizer.fit_manifest_digest
    assert a.manifest["recipe"]["source_fingerprint"] != b.manifest["recipe"]["source_fingerprint"]


def test_side_tables_retain_truth_without_exporting_query_attributes_by_default(tmp_path):
    config = make_v1(tmp_path / "v1")
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config, batch_size=1)
    table = pq.read_table(tmp_path / "v2/attributes.parquet").to_pylist()
    assert all(row["computed"] for row in table[:4])
    assert all(not row["computed"] for row in table[4:])
    assert all(not any(row["valid"]) for row in table[4:])
    views = pq.read_table(tmp_path / "v2/views.parquet").to_pylist()
    assert {row["role"] for row in views} == {"target_teacher", "reference_condition", "partial_observation"}


@pytest.mark.parametrize("case", ["uncanonical_alias", "patient_alias_cross_split", "content_cross_split"])
def test_entire_population_identity_audit_precedes_eeg_processing(tmp_path, monkeypatch, case):
    from triwave.data import dualbranch
    source = tmp_path / "v1"
    config = make_v1(source)
    path = source / "records.json"
    content = json.loads(path.read_text())
    records = content["records"]
    if case == "uncanonical_alias":
        records[2]["subject_id"] = records[2]["canonical_subject_id"] = "chb21"
        expected = "canonical patient"
    elif case == "patient_alias_cross_split":
        records[0]["subject_id"] = records[0]["canonical_subject_id"] = "chb01"
        records[2]["subject_id"], records[2]["canonical_subject_id"] = "chb21", "chb01"
        expected = "patient identity crosses"
    else:
        records[2]["source_sha256"] = records[0]["source_sha256"]
        expected = "source content crosses"
    path.write_text(json.dumps(content))
    monkeypatch.setattr(dualbranch._PhysicalAssets, "load", lambda *args: pytest.fail("EEG read before identity audit"))
    with pytest.raises(ValueError, match=expected):
        build_dualbranch_dataset(source, tmp_path / "v2", config)
    assert not (tmp_path / "v2").exists()


def test_v2_cli_runs_explicit_partial_observation_train_step(tmp_path, capsys):
    from triwave.cli import build_parser
    config = make_v1(tmp_path / "v1")
    build_dualbranch_dataset(tmp_path / "v1", tmp_path / "v2", config, batch_size=2)
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(yaml.safe_dump({"model": config.to_dict(), "training": {
        "effective_batch_size": 2, "gradient_accumulation": 1, "warmup_steps": 0, "total_steps": 2}}))
    args = build_parser().parse_args([
        "train", "--manifest", str(tmp_path / "v2"), "--config", str(yaml_path),
        "--device", "cpu", "--stage", "diffusion", "--steps", "1", "--batch-size", "2",
        "--data-purpose", "partial_observation", "--output", str(tmp_path / "run")])
    args.function(args)
    assert (tmp_path / "run/last.pt").exists()
    metrics = json.loads((tmp_path / "run/metrics.jsonl").read_text())
    assert metrics["step"] == 1 and np.isfinite(metrics["loss"])
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert final["status"] == "ok" and final["clinical_efficacy_evaluated"] is False


def test_native_source_bandwidth_is_shared_by_teacher_and_control_loss(tmp_path):
    from triwave.attributes import ATTRIBUTE_SCHEMA
    from triwave.cli import _collate
    from triwave.losses import attribute_control_loss
    source = tmp_path / "v1"
    config = make_v1(source, native128=True)  # Upsampling does not restore 64–80 Hz.
    build_dualbranch_dataset(source, tmp_path / "v2", config)
    dataset = DualBranchEEGDataset(tmp_path / "v2", config)
    item = dataset[0]
    assert item["source_max_frequency"].item() == 64
    for name in ("gamma_fraction", "gamma_log_power_uV2"):
        slot = ATTRIBUTE_SCHEMA.index(name)
        assert not item["teacher_attribute_valid"][slot]
        assert item["teacher_attributes"][slot].item() == 0
    batch = _collate([item, dataset[2]])
    assert batch["source_max_frequency"].tolist() == [64, 80]
    loss = attribute_control_loss(
        batch["target_waveform"], batch["teacher_attributes"], batch["teacher_attribute_valid"],
        ATTRIBUTE_SCHEMA, config.sample_rate, dataset.normalizer.std, batch["target_valid"],
        max_frequency=batch["source_max_frequency"])
    torch.testing.assert_close(loss, torch.zeros(2), atol=2e-5, rtol=0)
