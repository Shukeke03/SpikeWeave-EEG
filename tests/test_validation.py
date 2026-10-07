"""Independent sharded audits, including held-out read and condition boundaries."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from triwave.config import ModelConfig
from triwave.data.build import INDEX_DTYPE
from triwave.data.qc import Q_FLATLINE, Q_HIGH_AMPLITUDE
from triwave.data.sharded import fit_sharded_statistics
from triwave.data.validation import validate_sharded_dataset


def save_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path, *, pending=False, normalizer=True):
    """Small compact dataset, not a claimed full clinical corpus."""
    config = ModelConfig.tiny()
    samples, channels = config.samples, config.channels
    leads = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
    geometry = np.arange(channels * 6).reshape(channels, 6) / 100
    records = []
    rows = np.zeros(6, dtype=INDEX_DTYPE)
    rows["record"] = np.repeat(np.arange(3), 2)
    rows["offset"] = np.tile([0, samples], 3)
    rows["label"] = np.tile([0, 1], 3)
    rows["ref_index"] = [2, 3, 0, 1, 0, 1]
    rows["qc_flags"][0] = Q_HIGH_AMPLITUDE
    for index, (subject, split) in enumerate(zip(["chb02", "chb03", "chb04"], ["train", "train", "test"])):
        directory = tmp_path / "records" / f"{index:04d}"
        directory.mkdir(parents=True)
        time = np.arange(samples * 2 + 3) / config.sample_rate
        waveform = np.stack([20 * (index + 1) * np.sin(2 * np.pi * (8 + channel) * time)
                             for channel in range(channels)]).astype(np.float32)
        np.save(directory / "waveform.npy", waveform)
        records.append({"index": index, "record_id": f"CHB-MIT:{subject}/example.edf", "subject_id": subject,
                        "canonical_subject_id": subject, "source_file_id": f"{subject}/example.edf",
                        "source_sha256": hashlib.sha256(subject.encode()).hexdigest(), "source_verified": True,
                        "split": split, "role": "source" if split == "train" else "query", "status": "ready",
                        "waveform_path": f"records/{index:04d}/waveform.npy", "waveform_shape": list(waveform.shape),
                        "sample_rate": config.sample_rate, "window_samples": samples,
                        "unit": "uV", "lead_names": leads, "lead_valid": [True] * channels,
                        "lead_geometry": geometry.tolist(), "processed": True,
                        "annotation_known": True, "annotation_source": "engineering_fixture",
                        "events": [{"event_id": f"event-{index}", "onset": 1.0, "offset": 1.5}],
                        "indexed_windows": 2, "incomplete_tail_samples": 3})
    if pending:
        records.append({"index": 3, "record_id": "CHB-MIT:chb05/pending.edf", "subject_id": "chb05",
                        "canonical_subject_id": "chb05", "source_file_id": "chb05/pending.edf",
                        "source_sha256": hashlib.sha256(b"pending").hexdigest(), "source_verified": False,
                        "split": "validation", "role": "query", "status": "pending_download",
                        "events": [{"event_id": "pending-event", "onset": 1.0, "offset": 2.0}],
                        "annotation_known": True, "indexed_windows": 0})
    protocol = {"unit": "uV", "evaluation_protocol": "source-only", "lead_names": leads,
                "window_samples": samples, "sample_rate": config.sample_rate, "exclude_qc_bits": 3}
    save_json(tmp_path / "protocol.json", protocol)
    save_json(tmp_path / "records.json", {"records": records})
    save_json(tmp_path / "split_lock.json", {"patients": {record["canonical_subject_id"]: record["split"]
                                                          for record in records}})
    save_json(tmp_path / "build_report.json", {"ready_windows": len(rows), "isolated_windows": 0,
                                               "source_edfs_expected": len(records),
                                               "source_download_complete": not pending,
                                               "source_fully_processed": not pending})
    np.save(tmp_path / "window_index.npy", rows)
    np.save(tmp_path / "isolated_index.npy", np.empty(0, dtype=INDEX_DTYPE))
    if normalizer:
        fit_sharded_statistics(tmp_path, config).save(tmp_path / "normalizer.json")
    return config


def codes(report):
    return {item["check"] for item in report["errors"]}


def test_complete_fixture_audit_is_valid_but_not_a_full_clinical_corpus(tmp_path):
    config = fixture(tmp_path)
    report = validate_sharded_dataset(tmp_path, config=config, numerical_windows=3)
    assert report["valid"] and report["status"] == "not_complete"
    assert report["normalizer"]["status"] == "valid"
    assert report["normalizer"]["count"] == 4 * config.channels * config.samples
    assert report["counts"]["events"]["with_ready_windows"] == 3
    assert report["counts"]["tails"]["samples"] == 9
    assert report["counts"]["qc"]["ready"]["high_amplitude"] == 1
    assert report["references"]["same_class"] == 6
    numerical = report["numerical"]
    assert numerical["windows"] == 3
    assert numerical["dwt_idwt"]["passed"] and numerical["cwt"]["finite"]
    assert numerical["cwt"]["invalid_bins_zero"]
    assert numerical["cwt"]["reference_windows"] == 3
    assert len(numerical["cwt"]["valid_fraction_by_frequency"]) == config.cwt_frequencies
    assert all(index < 4 for index in numerical["selected_indices"])
    saved = json.loads((tmp_path / "dataset_audit.json").read_text())
    assert saved == report
    assert report["storage"]["processed"]["by_extension"][".npy"]["apparent_bytes"] > 0
    assert not report["storage"]["raw"]["available"]


def test_partial_download_is_a_valid_structural_preview_without_fitting(tmp_path):
    config = fixture(tmp_path, pending=True, normalizer=False)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert report["valid"] and report["status"] == "not_complete"
    assert not report["completeness"]["source_download_complete"]
    assert report["counts"]["events"]["without_any_saved_windows"] == 1
    assert report["normalizer"]["status"] == "missing"
    assert report["numerical"]["status"] == "skipped"
    assert not (tmp_path / "normalizer.json").exists()
    assert validate_sharded_dataset(tmp_path, numerical_windows=0)["valid"]


@pytest.mark.parametrize("mutation, expected", [
    (lambda rows: rows["record"].__setitem__(0, 99), "record_bounds"),
    (lambda rows: rows["offset"].__setitem__(0, -128), "offset_bounds"),
    (lambda rows: rows["offset"].__setitem__(0, 1), "offset_bounds"),
    (lambda rows: rows["offset"].__setitem__(0, 256), "offset_bounds"),
    (lambda rows: rows["offset"].__setitem__(1, 0), "duplicate_windows"),
    (lambda rows: rows["label"].__setitem__(0, 7), "labels"),
    (lambda rows: rows["label"].__setitem__(0, 1), "annotation_labels"),
    (lambda rows: rows["qc_flags"].__setitem__(0, 8), "qc_bits"),
    (lambda rows: rows["qc_flags"].__setitem__(0, Q_FLATLINE), "ready_qc"),
    (lambda rows: rows["ref_index"].__setitem__(0, 99), "reference_bounds"),
    (lambda rows: rows["ref_index"].__setitem__(0, 0), "reference_same_patient"),
    (lambda rows: rows["ref_index"].__setitem__(0, 4), "reference_train_source"),
])
def test_full_index_corruption_is_reported_without_numerical_reads(tmp_path, mutation, expected):
    config = fixture(tmp_path, normalizer=False)
    rows = np.load(tmp_path / "window_index.npy")
    mutation(rows)
    np.save(tmp_path / "window_index.npy", rows)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert not report["valid"] and expected in codes(report)
    assert report["numerical"]["status"] == "not_run"


def test_patient_identity_and_split_lock_mismatch_are_rejected(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    payload = json.loads((tmp_path / "records.json").read_text())
    payload["records"][2].update(subject_id="chb02", canonical_subject_id="chb02")
    save_json(tmp_path / "records.json", payload)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert not report["valid"] and "record_metadata" in codes(report)


@pytest.mark.parametrize("change", ["count", "digest", "nontraining_range", "overlapping_ranges"])
def test_normalizer_lineage_is_verified_not_only_its_mean_and_std(tmp_path, change):
    config = fixture(tmp_path)
    path = tmp_path / "normalizer.json"
    value = json.loads(path.read_text())
    if change == "count":
        value["count"] += 1
    elif change == "digest":
        value["fit_manifest_digest"] = "stale"
    elif change == "nontraining_range":
        value["fit_sample_ids"][0] = "CHB-MIT:chb04/example.edf:0-256"
    else:
        value["fit_sample_ids"].append(value["fit_sample_ids"][0])
    save_json(path, value)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert not report["valid"] and "normalizer" in codes(report)
    assert report["numerical"]["status"] == "skipped"


def test_held_out_values_are_never_read_for_numerical_acceptance(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / "records" / "0002" / "waveform.npy"
    data = np.load(path)
    data[:] = np.nan
    np.save(path, data)
    report = validate_sharded_dataset(tmp_path, config=config, numerical_windows=4)
    assert report["valid"] and report["numerical"]["status"] == "complete"
    assert all(index < 4 for index in report["numerical"]["selected_indices"])
    assert all(origin["reference_index"] < 4 for origin in report["numerical"]["cwt"]["reference_origins"])


def test_index_union_detects_duplicates_and_unknown_isolated_labels(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    rows = np.load(tmp_path / "window_index.npy")
    isolated = rows[:1].copy()
    isolated["ref_index"] = -1
    isolated["label"] = -2
    np.save(tmp_path / "isolated_index.npy", isolated)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert {"duplicate_windows", "labels", "window_coverage"}.issubset(codes(report))


def test_isolated_windows_and_events_are_accounted_without_becoming_references(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    rows = np.load(tmp_path / "window_index.npy")
    isolated = rows[-2:].copy()
    isolated["ref_index"] = -1
    isolated["qc_flags"] = Q_FLATLINE
    np.save(tmp_path / "window_index.npy", rows[:-2])
    np.save(tmp_path / "isolated_index.npy", isolated)
    path = tmp_path / "build_report.json"
    value = json.loads(path.read_text())
    value.update(ready_windows=4, isolated_windows=2)
    save_json(path, value)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert report["valid"]
    assert report["counts"]["events"]["with_ready_windows"] == 2
    assert report["counts"]["events"]["only_isolated_windows"] == 1
    assert report["counts"]["qc"]["isolated"]["flatline"] == 2


def test_missing_references_are_allowed_and_preferred_class_fallback_is_counted(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    rows = np.load(tmp_path / "window_index.npy")
    rows["ref_index"][0] = -1
    rows["ref_index"][1] = 2  # Valid other patient, but an available matching class was ignored.
    np.save(tmp_path / "window_index.npy", rows)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert report["valid"]
    assert report["references"]["missing"] == 1
    assert report["references"]["cross_class_despite_available_same_class"] == 1


def test_unknown_reference_isolation_needs_no_fabricated_waveform(tmp_path):
    config = fixture(tmp_path)
    rows = np.load(tmp_path / "window_index.npy")
    np.save(tmp_path / "window_index.npy", rows[:4])
    path = tmp_path / "records.json"
    payload = json.loads(path.read_text())
    record = payload["records"][2]
    waveform = tmp_path / record.pop("waveform_path")
    waveform.unlink()
    record.update(status="isolated", waveform_available=False, indexed_windows=0,
                  isolation_reasons=["unsupported_montage", "unknown_reference"],
                  source_header_evidence={"source_samples": 259, "source_sample_rate": 128,
                                          "source_channel_names": ["F7", "T7"]})
    save_json(path, payload)
    path = tmp_path / "build_report.json"
    build = json.loads(path.read_text())
    build["ready_windows"] = 4
    save_json(path, build)
    report = validate_sharded_dataset(tmp_path, config=config, numerical_windows=2)
    assert report["valid"]
    assert report["counts"]["records"]["isolated"] == 1
    assert report["counts"]["events"]["without_any_saved_windows"] == 1
    assert report["counts"]["events"]["inventory_by_record_status"]["isolated"] == 1


def test_duplicate_ready_source_and_false_download_complete_ledger_are_rejected(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    path = tmp_path / "records.json"
    value = json.loads(path.read_text())
    value["records"][1]["source_sha256"] = value["records"][0]["source_sha256"]
    save_json(path, value)
    report = validate_sharded_dataset(tmp_path, config=config)
    assert {"duplicate_ready_sources", "reference_same_source_hash"}.issubset(codes(report))
    path = tmp_path / "build_report.json"
    value = json.loads(path.read_text())
    value["source_download_complete"] = False
    save_json(path, value)
    assert "build_report_counts" in codes(validate_sharded_dataset(tmp_path, config=config))


def test_unsafe_asset_path_and_malformed_index_schema_are_reported(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    payload = json.loads((tmp_path / "records.json").read_text())
    payload["records"][0]["waveform_path"] = "../outside.npy"
    save_json(tmp_path / "records.json", payload)
    assert "record_metadata" in codes(validate_sharded_dataset(tmp_path, config=config))
    np.save(tmp_path / "window_index.npy", np.zeros((2, 2)))
    assert "required_files_or_schema" in codes(validate_sharded_dataset(tmp_path, config=config))


def test_zero_numerical_windows_and_raw_storage_accounting(tmp_path):
    config = fixture(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "example.edf").write_bytes(b"synthetic header")
    path = tmp_path / "build_report.json"
    value = json.loads(path.read_text())
    value["raw_directory"] = str(raw)
    save_json(path, value)
    report = validate_sharded_dataset(tmp_path, config=config, numerical_windows=0)
    assert report["valid"] and report["numerical"]["status"] == "skipped"
    assert report["storage"]["raw"]["apparent_bytes"] == len(b"synthetic header")


def test_invalid_numerical_limit_and_missing_required_file(tmp_path):
    config = fixture(tmp_path, normalizer=False)
    with pytest.raises(ValueError, match="numerical_windows"):
        validate_sharded_dataset(tmp_path, numerical_windows=-1)
    (tmp_path / "records.json").unlink()
    assert "required_files_or_schema" in codes(validate_sharded_dataset(tmp_path, config=config))


def test_validator_accepts_actual_builder_output_and_train_statistics(tmp_path):
    from test_sharded import config as shard_config
    from test_sharded import fixture_source

    from triwave.data.build import build_chbmit

    source, output = tmp_path / "source", tmp_path / "built"
    splits = fixture_source(source, flatline=True, unlisted_background=True)
    build_chbmit(source, output, splits, samples=128, sample_rate=128,
                 bandpass=None, workers=1, expected_records=None)
    fit_sharded_statistics(output, shard_config()).save(output / "normalizer.json")
    report = validate_sharded_dataset(output, config=shard_config(), numerical_windows=2)
    assert report["valid"]
    assert report["counts"]["ready_windows"] == 11
    assert report["counts"]["isolated_windows"] == 1
    assert report["normalizer"]["count"] == 7 * 16 * 128
    assert report["numerical"]["cwt"]["reference_windows"] == 2
