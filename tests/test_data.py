"""Checks for clinical units, patient identity and prohibited target conditioning."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.chbmit import (
    STANDARD_BIPOLAR_LEADS,
    parse_summary_text,
    prepare_chbmit,
    read_edf_bipolar,
)
from triwave.data.dataset import EEGManifestDataset, TrainNormalizer
from triwave.data.manifest import (
    SampleRecord,
    attach_source_references,
    audit_reference,
    lock_patient_splits,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from triwave.data.synthetic import make_synthetic_batch


def sample(sample_id="a", subject="chb02", split="train", role="source", record_id="record-a",
           start=0.0, end=1.0, **kwargs):
    return SampleRecord(sample_id, "CHB-MIT", subject, record_id, f"{sample_id}.npz", start, end,
                        128.0, ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"], split, role, **kwargs)


def test_summary_events_channels_and_count_validation():
    text = """Data Sampling Rate: 256 Hz
Channels in EDF Files:
Channel 1: FP1-F7
Channel 2: F7-T3
File Name: chb01_01.edf
File Start Time: 03:42:54
File End Time: 04:42:54
Number of Seizures in File: 0
File Name: chb01_02.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 20 seconds
Seizure 2 Start Time: 25 seconds
Seizure 2 End Time: 30 seconds
"""
    records = parse_summary_text(text)
    assert len(records) == 2
    assert records[1].sample_rate == 256
    assert records[1].channels == ["FP1-F7", "F7-T3"]
    assert records[1].seizures[1].offset == 30
    with pytest.raises(ValueError, match="count mismatch"):
        parse_summary_text(text.replace("in File: 2", "in File: 3"))
    with pytest.raises(ValueError, match="no end"):
        parse_summary_text(text.replace("Seizure 2 End Time: 30 seconds", ""))


def test_same_patient_split_lock_and_duplicate_source(tmp_path):
    with pytest.raises(ValueError, match="same patient"):
        lock_patient_splits({"chb01": "train", "chb21": "test"}, tmp_path / "lock.json")
    assert lock_patient_splits({"chb01": "train", "chb21": "train"}, tmp_path / "lock.json") == {"chb01": "train"}
    with pytest.raises(ValueError, match="lock differs"):
        lock_patient_splits({"chb01": "test"}, tmp_path / "lock.json")
    with pytest.raises(ValueError, match="patient overlaps"):
        validate_manifest([sample(subject="chb01"), sample("b", "chb21", "test", "query")])
    with pytest.raises(ValueError, match="duplicate source"):
        validate_manifest([sample(provenance={"source_sha256": "same"}),
                           sample("b", "chb03", "test", "query", provenance={"source_sha256": "same"})])


def test_support_checks_processing_context_and_event():
    target = sample("query", "chb03", "test", "query", "same-record", start=10, end=11,
                    preprocessing_context=[0, 100], evaluation_protocol="few-shot-support")
    reference = sample("support", "chb03", "test", "support", "same-record", start=1, end=2,
                       preprocessing_context=[0, 100], evaluation_protocol="few-shot-support")
    with pytest.raises(ValueError, match="context overlaps"):
        audit_reference(target, reference)
    reference.record_id = "different-record"
    assert audit_reference(target, reference)["reference_role"] == "support"
    target.event_ids = reference.event_ids = ["same-event"]
    with pytest.raises(ValueError, match="share an event"):
        audit_reference(target, reference)
    with pytest.raises(ValueError, match="source training"):
        audit_reference(target, reference, "source-only")


def test_train_only_global_normalization_and_inverse():
    record = sample()
    waveform = np.arange(16, dtype=np.float32).reshape(4, 4)
    normalizer = TrainNormalizer().fit([record], lambda _: waveform)
    np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(waveform)), waveform, atol=1e-6)
    tensor = torch.from_numpy(waveform).to(torch.float64)
    restored = normalizer.inverse_transform(normalizer.transform(tensor))
    assert isinstance(restored, torch.Tensor) and restored.dtype == tensor.dtype
    torch.testing.assert_close(restored, tensor)
    copied = TrainNormalizer.from_dict(normalizer.to_dict())
    assert copied.fit_sample_ids == normalizer.fit_sample_ids
    for split, role in [("validation", "query"), ("test", "query")]:
        with pytest.raises(ValueError, match="only fit source training"):
            TrainNormalizer().fit([sample(split=split, role=role)], lambda _: waveform)


def test_manifest_dataset_uses_separate_reference_and_ignores_teacher(tmp_path):
    config = ModelConfig.tiny()
    target = sample("target", "chb03", "test", "query", "record-target", conditions={
        "reference": {"origin_id": "reference", "target_derived": False, "inference_available": True},
        "text": {"content": "an engineering test", "inference_available": True}})
    reference = sample("reference", "chb02", "train", "source", "record-reference")
    for record, value in [(target, 7), (reference, 2)]:
        np.savez(tmp_path / record.waveform_path, waveform=np.full((4, 128), value, dtype=np.float32),
                 lead_geometry=np.zeros((4, 6), dtype=np.float32), lead_valid=np.ones(4, dtype=bool),
                 teacher_cwt=np.full((4, 16, 32), 999))
    manifest = write_manifest([target, reference], tmp_path / "manifest.jsonl")
    assert len(read_manifest(manifest)) == 2
    item = EEGManifestDataset(manifest, config, split="test")[0]
    assert torch.all(item["target_waveform"] == 7)
    assert torch.all(item["reference_waveform"] == 2)
    assert "structure" not in item and "teacher_cwt" not in item
    assert item["metadata"]["reference_audit"]["reference_id"] == "reference"
    payload = target.to_dict()
    payload["conditions"]["reference"]["origin_id"] = "target"
    with pytest.raises(ValueError, match="target-derived"):
        SampleRecord.from_dict(payload)
    # A serialized record still cannot sneak target CWT into the structure slot.
    payload["conditions"].pop("reference")
    payload["conditions"]["structure"] = {"target_derived": True}
    with pytest.raises(ValueError, match="target-derived"):
        SampleRecord.from_dict(json.loads(json.dumps(payload)))


def test_synthetic_batch_is_reproducible_and_reference_independent():
    config = ModelConfig.tiny()
    batch = make_synthetic_batch(config, 3, seed=3)
    repeated = make_synthetic_batch(config, 3, seed=3)
    assert torch.equal(batch["target_waveform"], repeated["target_waveform"])
    assert not torch.equal(batch["target_waveform"], batch["reference_waveform"])
    assert batch["lead_geometry"].shape == (3, 4, 6)
    assert batch["metadata"]["synthetic"]
    assert "structure" not in batch


def test_synthetic_edf_units_aliases_and_known_reverse_polarity(tmp_path):
    pytest.importorskip("edfio")
    mne = pytest.importorskip("mne")
    time = np.arange(512) / 256
    physical_uv = np.stack([(index + 1) * 10 * np.sin(2 * np.pi * 5 * time)
                            for index in range(16)])
    source_names = [lead.replace("T7", "T3").replace("T8", "T4").replace("P7", "T5").replace("P8", "T6")
                    for lead in STANDARD_BIPOLAR_LEADS]
    source_names[0] = "F7-FP1"
    stored = physical_uv.copy()
    stored[0] *= -1
    raw = mne.io.RawArray(stored * 1e-6, mne.create_info(source_names, 256, "eeg"), verbose="ERROR")
    path = tmp_path / "synthetic.edf"
    raw.export(path, fmt="edf", physical_range=(-200, 200), verbose="ERROR")
    recording = read_edf_bipolar(path)
    np.testing.assert_allclose(recording.waveform, physical_uv, atol=0.01)
    assert recording.lead_geometry.shape == (16, 6)
    assert recording.provenance["lead_origins"][0]["sign"] == -1
    assert recording.provenance["output_unit"] == "uV"


def test_prepare_synthetic_edf_event_truncation_and_physical_dataset(tmp_path):
    pytest.importorskip("edfio")
    mne = pytest.importorskip("mne")
    root = tmp_path / "source"
    subject = root / "chb02"
    subject.mkdir(parents=True)
    filename = "chb02_01.edf"
    time = np.arange(2048) / 256
    data = np.stack([20e-6 * np.sin(2 * np.pi * (5 + channel / 10) * time)
                     for channel in range(16)])
    raw = mne.io.RawArray(data, mne.create_info(list(STANDARD_BIPOLAR_LEADS), 256, "eeg"), verbose="ERROR")
    raw.export(subject / filename, fmt="edf", physical_range=(-100, 100), verbose="ERROR")
    lines = ["Data Sampling Rate: 256 Hz", "Channels in EDF Files:"]
    lines += [f"Channel {index + 1}: {name}" for index, name in enumerate(STANDARD_BIPOLAR_LEADS)]
    lines += [f"File Name: {filename}", "Number of Seizures in File: 1",
              "Seizure Start Time: 3.5 seconds", "Seizure End Time: 4.5 seconds"]
    (subject / "chb02-summary.txt").write_text("\n".join(lines))
    output = tmp_path / "prepared"
    manifest = prepare_chbmit(root, output, {"chb02": "train"}, bandpass=None)
    records = read_manifest(manifest)
    assert len(records) == 2
    assert records[0].annotations[0]["right_truncated"]
    assert records[1].annotations[0]["left_truncated"]
    assert records[0].preprocessing_context == [0, 8]
    assert records[0].conditions["class"]["label"] == "annotated_seizure"
    dataset = EEGManifestDataset(manifest, ModelConfig(), split="train", role="source")
    item = dataset[0]
    assert item["target_waveform"].shape == (16, 1024)
    assert item["metadata"]["unit"] == "uV"
    assert not item["reference_valid"].any()
    normalizer = TrainNormalizer().fit(dataset)
    assert normalizer.count == 16 * 2048
    assert normalizer.std > 1  # Confirms actual uV rather than EDF SI volt scale.


def test_attach_source_references_same_class_and_heldout_exclusion(tmp_path):
    def label(class_id):
        return {"class": {"id": class_id, "namespace": "engineering.test"}}

    records = [sample("a", "chb02", record_id="record-a", conditions=label(0)),
               sample("b", "chb03", record_id="record-b", conditions=label(0)),
               sample("c", "chb04", record_id="record-c", conditions=label(1)),
               sample("query", "chb05", "test", "query", "record-query", conditions=label(0))]
    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    attached = read_manifest(attach_source_references(manifest, seed=11))
    by_id = {record.sample_id: record for record in attached}
    assert by_id["a"].conditions["reference"]["origin_id"] == "b"
    assert by_id["b"].conditions["reference"]["origin_id"] == "a"
    assert by_id["query"].conditions["reference"]["origin_id"] in {"a", "b"}
    for target in attached:
        reference = target.conditions["reference"]
        assert reference["origin_id"] != target.sample_id
        assert reference["origin_id"] != "query"
        audit_reference(target, by_id[reference["origin_id"]])
    repeated = read_manifest(attach_source_references(manifest, tmp_path / "repeat.jsonl", seed=11))
    assert [record.conditions for record in attached] == [record.conditions for record in repeated]
    # Existing audited choices survive a different seed.
    preserved = read_manifest(attach_source_references(tmp_path / "manifest-reference.jsonl", seed=99))
    assert [record.conditions for record in preserved] == [record.conditions for record in attached]
    with pytest.raises(ValueError, match="same asset directory"):
        attach_source_references(manifest, tmp_path / "elsewhere" / "manifest.jsonl")


def test_attach_source_references_insufficient_candidates_and_protocol_rejection(tmp_path):
    records = [sample("only", record_id="only-record"),
               sample("same-record", record_id="only-record", start=2, end=3)]
    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    attached = read_manifest(attach_source_references(manifest))
    assert all("reference" not in record.conditions for record in attached)
    records = [sample("few-shot", evaluation_protocol="few-shot-support")]
    manifest = write_manifest(records, tmp_path / "few-shot.jsonl")
    with pytest.raises(ValueError, match="source-only protocol only"):
        attach_source_references(manifest)


def test_scalar_normalization_keeps_cycle_closure_and_detects_legacy_affine():
    from triwave.montage import MontageSubspace

    waveform = np.arange(24, dtype=np.float32).reshape(3, 8)
    waveform[2] = waveform[0] + waveform[1]
    record = sample()
    record.lead_names = ["A-B", "B-C", "A-C"]
    record.lead_valid = [True, True, True]
    normalizer = TrainNormalizer().fit([record], lambda _: waveform)
    assert normalizer.mean == 0 and normalizer.scale_protocol == "source_rms_v2"
    normalized = normalizer.transform(waveform)
    montage = MontageSubspace.from_endpoints(torch.tensor([[0, 1], [1, 2], [0, 2]]))
    assert montage.audit(torch.from_numpy(normalized)[None])["passed"]
    payload = normalizer.to_dict()
    payload.pop("scale_protocol")
    payload["mean"] = 1.0
    legacy = TrainNormalizer.from_dict(payload)
    assert legacy.scale_protocol == "legacy_affine_v1"
