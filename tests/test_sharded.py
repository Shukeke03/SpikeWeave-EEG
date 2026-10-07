"""Engineering EDF fixtures verify full-inventory, shard and leakage policies."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.build import INDEX_DTYPE, build_chbmit
from triwave.data.chbmit import STANDARD_BIPOLAR_LEADS, EDFRecording
from triwave.data.qc import Q_HIGH_AMPLITUDE
from triwave.data.sharded import (
    ShardedEEGDataset,
    fit_sharded_statistics,
    sharded_normalizer_digest,
)


def config():
    return ModelConfig(channels=16, samples=128, sample_rate=128.0,
                       waveform_patch=16, coefficient_patch=16, cwt_max_frequency=48.0)


def fixture_source(root, *, unlisted_background=False, flatline=False):
    pytest.importorskip("edfio")
    mne = pytest.importorskip("mne")
    root.mkdir()
    paths, positives = [], []
    for patient, amplitude in [("chb02", 10), ("chb03", 20), ("chb24", 10000)]:
        directory = root / patient
        directory.mkdir()
        filename = f"{patient}_01.edf"
        time = np.arange(512) / 128
        waveform = np.stack([amplitude * np.sin(2 * np.pi * (7 + channel / 8) * time)
                             for channel in range(16)])
        if flatline and patient == "chb02":
            waveform[0, :128] = 0
        raw = mne.io.RawArray(waveform * 1e-6, mne.create_info(list(STANDARD_BIPOLAR_LEADS), 128, "eeg"),
                             verbose="ERROR")
        raw.export(directory / filename, fmt="edf", physical_range=(-20000, 20000), verbose="ERROR")
        relative = f"{patient}/{filename}"
        paths.append(relative)
        lines = ["Data Sampling Rate: 128 Hz", "Channels in EDF Files:"]
        lines += [f"Channel {index + 1}: {lead}" for index, lead in enumerate(STANDARD_BIPOLAR_LEADS)]
        if not (unlisted_background and patient == "chb24"):
            positives.append(relative)
            lines += [f"File Name: {filename}", "Number of Seizures in File: 1",
                      "Seizure Start Time: 0.25 seconds", "Seizure End Time: 0.75 seconds"]
            # Only the inventory/summary relation is used in this engineering fixture.
            (directory / f"{filename}.seizures").write_bytes(b"synthetic-annotation-fixture")
        (directory / f"{patient}-summary.txt").write_text("\n".join(lines) + "\n")
    (root / "RECORDS").write_text("\n".join(paths) + "\n")
    (root / "RECORDS-WITH-SEIZURES").write_text("\n".join(positives) + "\n")
    (root / "SUBJECT-INFO").write_text("Case\tGender\tAge (years)\n")
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sums.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    return {"chb02": "train", "chb03": "train", "chb24": "test"}


def test_shards_reference_patient_isolation_train_statistics_and_resume(tmp_path):
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root, flatline=True)
    path = build_chbmit(root, output, splits, samples=128, sample_rate=128,
                       workers=2, bandpass=None, expected_records=None)
    index = np.load(path)
    assert index.dtype == INDEX_DTYPE
    assert len(index) == 11  # The full raw flatline window is isolated.
    records = json.loads((output / "records.json").read_text())["records"]
    for row in index:
        donor = index[int(row["ref_index"])]
        assert records[int(donor["record"])]["split"] == "train"
        assert records[int(donor["record"])]["canonical_subject_id"] != records[int(row["record"])]["canonical_subject_id"]
    test = ShardedEEGDataset(output, config(), split="test", mmap_cache_size=1)
    item = test[0]
    assert item["target_waveform"].shape == (16, 128)
    assert item["reference_valid"].all()
    assert item["metadata"]["qc_flags"] & Q_HIGH_AMPLITUDE
    assert "structure" not in item
    train = ShardedEEGDataset(output, config(), split="train")
    normalizer = fit_sharded_statistics(train)
    assert normalizer.count == 7 * 16 * 128
    assert normalizer.std < 30  # Large held-out amplitude cannot enter fitted scale.
    with pytest.raises(ValueError, match="source training"):
        fit_sharded_statistics(test)
    normalized = ShardedEEGDataset(output, config(), split="test", normalizer=normalizer)[0]
    torch.testing.assert_close(normalizer.inverse_transform(normalized["target_waveform"]),
                               item["target_waveform"], rtol=1e-5, atol=1e-3)
    shard = output / records[0]["waveform_path"]
    mtime = shard.stat().st_mtime_ns
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["resumed_records"] == 3 and report["source_download_complete"]
    assert shard.stat().st_mtime_ns == mtime
    np.testing.assert_array_equal(np.load(path), index)


def test_unlisted_background_requires_complete_authoritative_inventory(tmp_path):
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root, unlisted_background=True)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    records = json.loads((output / "records.json").read_text())["records"]
    assert not records[2]["summary_present"] and records[2]["annotation_known"]
    assert records[2]["annotation_source"] == "official_negative_membership_in_complete_seizure_file_list"
    dataset = ShardedEEGDataset(output, config(), split="test")
    assert all(int(dataset[index]["class_ids"]) == 0 for index in range(len(dataset)))
    (root / "RECORDS-WITH-SEIZURES").write_text("")
    with pytest.raises(ValueError, match="inventor|metadata"):
        build_chbmit(root, tmp_path / "corrupt", splits, samples=128, sample_rate=128,
                     workers=1, bandpass=None, expected_records=None)


def test_positive_list_typo_reconciles_only_with_interval_and_sidecar_evidence(tmp_path):
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    # An official-style list omission cannot overrule interval + sidecar agreement.
    positive = root / "RECORDS-WITH-SEIZURES"
    positive.write_text("chb02/chb02_01.edf\nchb24/chb24_01.edf\n")
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    records = json.loads((output / "records.json").read_text())["records"]
    reconciled = records[1]
    assert reconciled["annotation_known"] and reconciled["status"] == "ready"
    assert reconciled["positive_list_disagreement"] and len(reconciled["events"]) == 1
    audit = json.loads((output / "build_report.json").read_text())["inventory_audit"]
    assert audit["positive_list_sidecar_disagreements"] == ["chb03/chb03_01.edf"]


def test_conflicting_duplicate_lead_is_audited_isolation_not_an_unresolved_failure(tmp_path):
    import mne

    from triwave.data.chbmit import UnsupportedMontageError, read_edf_bipolar

    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    source = root / "chb02/chb02_01.edf"
    source.unlink()
    times = np.arange(512) / 128
    channels = list(STANDARD_BIPOLAR_LEADS) + ["T8-P8"]
    values = np.stack([10 * np.sin(2 * np.pi * 8 * times)] * 16 + [20 * np.sin(2 * np.pi * 12 * times)])
    raw = mne.io.RawArray(values * 1e-6, mne.create_info(channels, 128, "eeg"), verbose="ERROR")
    raw.export(source, fmt="edf", physical_range=(-100, 100), verbose="ERROR")
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    with pytest.raises(UnsupportedMontageError, match="ambiguous duplicate"):
        read_edf_bipolar(source, allow_missing=True)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    record = json.loads((output / "records.json").read_text())["records"][0]
    report = json.loads((output / "build_report.json").read_text())
    assert record["status"] == "isolated" and not record["waveform_available"]
    assert record["isolation_reasons"] == ["ambiguous_duplicate_bipolar_lead"]
    assert len(record["events"]) == 1 and record["source_header_evidence"]["ambiguous_lead"] == "T8-P8"
    assert report["source_fully_processed"] and report["records"]["failed"] == 0


def test_pending_download_retry_and_no_final_statistics(tmp_path):
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    pending_path = root / "chb24" / "chb24_01.edf"
    original = pending_path.read_bytes()
    pending_path.unlink()
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["records"]["pending_download"] == 1
    assert not (output / "records" / "0002" / "done.json").exists()
    with pytest.raises(ValueError, match="download is incomplete"):
        ShardedEEGDataset(output, config())
    preview = ShardedEEGDataset(output, config(), split="train", allow_partial=True)
    with pytest.raises(ValueError, match="complete source download"):
        fit_sharded_statistics(preview)
    pending_path.write_bytes(original)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["source_download_complete"] and report["resumed_records"] == 2


def test_aria2_active_download_skips_source_checksum_and_retries(tmp_path, monkeypatch):
    import triwave.data.build as build_module

    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    source = root / "chb24" / "chb24_01.edf"
    marker = source.with_suffix(".edf.aria2")
    marker.write_bytes(b"active-download-fixture")
    original_checksum = build_module.file_sha256

    def checksum(path):
        assert not (path == source and marker.exists()), "active EDF must not be hashed"
        return original_checksum(path)

    monkeypatch.setattr(build_module, "file_sha256", checksum)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    record = json.loads((output / "records.json").read_text())["records"][2]
    assert record["status"] == "pending_download" and not record["source_verified"]
    assert "aria2" in record["pending_reason"]
    assert not (output / "records" / "0002" / "done.json").exists()
    marker.unlink()
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["source_download_complete"] and report["resumed_records"] == 2


def test_unresolved_processing_failure_blocks_training_and_statistics(tmp_path, monkeypatch):
    import triwave.data.build as build_module

    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    original_reader = build_module.read_edf_bipolar

    def reader(path, **kwargs):
        if path.name == "chb03_01.edf":
            raise RuntimeError("synthetic unexpected reader failure")
        return original_reader(path, **kwargs)

    monkeypatch.setattr(build_module, "read_edf_bipolar", reader)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["source_download_complete"] and not report["source_fully_processed"]
    with pytest.raises(ValueError, match="source processing is incomplete"):
        ShardedEEGDataset(output, config(), split="train")
    preview = ShardedEEGDataset(output, config(), split="train", allow_partial=True)
    assert len(preview) > 0
    with pytest.raises(ValueError, match="complete source processing"):
        fit_sharded_statistics(preview)


def test_nonfinite_and_incomplete_montage_retained_but_isolated(tmp_path, monkeypatch):
    import triwave.data.build as build_module

    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    original_reader = build_module.read_edf_bipolar

    def reader(path, **kwargs):
        if path.name == "chb02_01.edf":
            values = np.ones((16, 512), dtype=np.float32)
            values[0, 100] = np.nan
            return EDFRecording(values, 128, STANDARD_BIPOLAR_LEADS, np.ones(16, dtype=bool),
                                np.zeros((16, 6), dtype=np.float32), {"synthetic": True})
        recording = original_reader(path, **kwargs)
        if path.name == "chb03_01.edf":
            recording.lead_valid[0] = False
            recording.waveform[0] = 0
        return recording

    monkeypatch.setattr(build_module, "read_edf_bipolar", reader)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=(0.1, 48), expected_records=None)
    records = json.loads((output / "records.json").read_text())["records"]
    assert records[0]["status"] == "isolated" and not records[0]["processed"]
    assert "nonfinite_raw_record_not_filtered" in records[0]["isolation_reasons"]
    assert "incomplete_standard_bipolar_montage" in records[1]["isolation_reasons"]
    assert np.isnan(np.load(output / records[0]["waveform_path"])[0, 100])
    ready = np.load(output / "window_index.npy")
    assert np.all(ready["record"] == 2) and np.all(ready["ref_index"] == -1)


def test_unknown_reference_header_is_auditable_isolation_without_derived_waveform(tmp_path):
    mne = pytest.importorskip("mne")
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    source = root / "chb03" / "chb03_01.edf"
    electrodes = ["FP1", "F7", "T3", "T5", "01", "FP2", "F8", "T4",
                  "T6", "O2", "F3", "C3", "P3", "F4", "C4", "P4"]
    waveform = 20e-6 * np.sin(2 * np.pi * 8 * np.arange(512) / 128)
    raw = mne.io.RawArray(np.tile(waveform, (16, 1)), mne.create_info(electrodes, 128, "eeg"),
                         verbose="ERROR")
    raw.export(source, fmt="edf", overwrite=True, verbose="ERROR")
    sums_path = root / "SHA256SUMS.txt"
    sums = []
    for line in sums_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name == "chb03/chb03_01.edf":
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        sums.append(f"{digest}  {name}")
    sums_path.write_text("\n".join(sums) + "\n")
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    record = json.loads((output / "records.json").read_text())["records"][1]
    assert record["status"] == "isolated" and record["source_verified"]
    assert not record["waveform_available"] and record["indexed_windows"] == 0
    assert record["isolation_reasons"] == ["unsupported_montage", "unknown_reference"]
    assert record["source_header_evidence"]["source_channel_names"] == electrodes
    assert record["events"] and record["annotation_known"]
    assert "waveform_path" not in record
    assert not (output / "records" / "0001" / "waveform.npy").exists()
    assert len(np.load(output / "records" / "0001" / "index.npy")) == 0
    report = json.loads((output / "build_report.json").read_text())
    assert report["source_download_complete"] and report["source_fully_processed"]
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    report = json.loads((output / "build_report.json").read_text())
    assert report["resumed_records"] == 3


@pytest.mark.parametrize("changed", ["training_index", "protocol", "split_lock", "training_source"])
def test_stale_normalizer_rejected_when_training_definition_changes(tmp_path, changed):
    root, output = tmp_path / "source", tmp_path / "shards"
    splits = fixture_source(root)
    build_chbmit(root, output, splits, samples=128, sample_rate=128,
                 workers=1, bandpass=None, expected_records=None)
    train = ShardedEEGDataset(output, config(), split="train")
    normalizer = fit_sharded_statistics(train)
    assert normalizer.fit_manifest_digest == sharded_normalizer_digest(train)
    # A held-out loader checks the complete source-training population, too.
    ShardedEEGDataset(output, config(), split="test", normalizer=normalizer)
    if changed == "training_index":
        path = output / "window_index.npy"
        rows = np.load(path)
        rows["label"][0] = 1 - rows["label"][0]
        np.save(path, rows)
    elif changed == "protocol":
        path = output / "protocol.json"
        protocol = json.loads(path.read_text())
        assert len(protocol["reader_sha256"]) == 64
        protocol["reader_sha256"] = "0" * 64
        path.write_text(json.dumps(protocol))
    elif changed == "split_lock":
        path = output / "split_lock.json"
        path.write_text(path.read_text() + "\n")
    else:
        path = output / "records.json"
        records = json.loads(path.read_text())
        records["records"][0]["source_sha256"] = "0" * 64
        path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="digest is stale"):
        ShardedEEGDataset(output, config(), split="test", normalizer=normalizer)
