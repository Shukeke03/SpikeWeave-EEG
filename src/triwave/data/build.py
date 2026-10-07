"""Audited, resumable CHB-MIT record shards and compact nonoverlapping window indices."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from .chbmit import (
    STANDARD_BIPOLAR_LEADS,
    UnsupportedMontageError,
    file_sha256,
    parse_summary,
    read_edf_bipolar,
)
from .manifest import canonical_patient, lock_patient_splits
from .qc import DEFAULT_EXCLUDED_FLAGS, Q_NONFINITE, window_quality_flags

INDEX_DTYPE = np.dtype([("record", "<i4"), ("offset", "<i8"), ("label", "i1"),
                        ("qc_flags", "<u2"), ("ref_index", "<i8")])
PIPELINE_VERSION = "chbmit-record-shards-v1"
ANNOTATION_SEMANTICS_URL = "https://physionet.org/content/chbmit/1.0.0/"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _default_splits() -> dict[str, str]:
    path = Path(__file__).resolve().parents[3] / "configs" / "chbmit-patients-v1.json"
    if not path.is_file():
        raise ValueError("provide split_map; packaged runtime has no patient config file")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inventory(root: Path, expected_records: int | None) -> tuple[list[dict], dict]:
    sums_path = root / "SHA256SUMS.txt"
    checksums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            if name in checksums or len(digest) != 64:
                raise ValueError("malformed/duplicate SHA256 inventory entry")
            checksums[name] = digest
    names = [line.strip() for line in (root / "RECORDS").read_text().splitlines() if line.strip()]
    positives = {line.strip() for line in (root / "RECORDS-WITH-SEIZURES").read_text().splitlines()
                 if line.strip()}
    if len(names) != len(set(names)) or any(not name.endswith(".edf") for name in names):
        raise ValueError("RECORDS must uniquely list EDF assets")
    expected_edfs = {name for name in checksums if name.endswith(".edf")}
    if set(names) != expected_edfs:
        raise ValueError("RECORDS and SHA256 EDF inventories disagree")
    if expected_records is not None and len(names) != expected_records:
        raise ValueError(f"inventory has {len(names)} EDFs, expected {expected_records}")
    if not positives.issubset(names):
        raise ValueError("seizure-file inventory contains unknown recordings")
    expected_annotations = {name.removesuffix(".seizures") for name in checksums
                            if name.endswith(".edf.seizures")}
    if not expected_annotations.issubset(names):
        raise ValueError("seizure sidecar inventory contains unknown recordings")
    positive_list_disagreements = sorted(expected_annotations.symmetric_difference(positives))
    metadata_audit = {}
    for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO"):
        path = root / name
        if name not in checksums or not path.is_file() or file_sha256(path) != checksums[name]:
            raise ValueError(f"missing/corrupt official inventory metadata: {name}")
        metadata_audit[name] = checksums[name]
    auxiliary_errors = []
    for name, digest in checksums.items():
        if name.endswith(".edf"):
            continue
        path = root / name
        if not path.is_file() or file_sha256(path) != digest:
            auxiliary_errors.append(name)
    cases = sorted({Path(name).parts[0] for name in names})
    summary_by_name = {}
    summary_hashes = {}
    summary_errors = {}
    for case in cases:
        relative = f"{case}/{case}-summary.txt"
        path = root / relative
        try:
            if relative not in checksums or file_sha256(path) != checksums[relative]:
                raise ValueError("summary checksum mismatch")
            summary_hashes[case] = checksums[relative]
            for record in parse_summary(path):
                summary_by_name[f"{case}/{record.filename}"] = record
        except Exception as error:
            summary_errors[case] = f"{type(error).__name__}: {error}"
    inventory = []
    for index, name in enumerate(names):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe source asset path")
        case = Path(name).parts[0]
        summary = summary_by_name.get(name)
        known = False
        annotation_source = "unknown"
        events = []
        annotation_error = summary_errors.get(case, "")
        if summary is not None:
            has_event = bool(summary.seizures)
            if has_event != (name in expected_annotations):
                annotation_error = "summary and official seizure-sidecar inventory disagree"
            else:
                known = True
                events = [{"event_id": f"CHB-MIT:{name}:seizure:{event_index}",
                           "onset": event.onset, "offset": event.offset}
                          for event_index, event in enumerate(summary.seizures)]
                annotation_source = ("official_summary_and_seizure_sidecar_inventory_reconciled"
                                     if name in positive_list_disagreements else
                                     "official_summary_and_seizure_file_list")
        elif name not in positives and name not in expected_annotations and case not in summary_errors:
            # The official page explicitly defines the positive-file list as all
            # files containing >=1 seizure. Both list and annotation inventory
            # were checked above; this is not an assumption from a missing sidecar.
            known = True
            annotation_source = "official_negative_membership_in_complete_seizure_file_list"
        inventory.append({"index": index, "source_file_id": name,
                          "source_sha256": checksums[name], "subject_id": case,
                          "canonical_subject_id": canonical_patient(case),
                          "record_id": f"CHB-MIT:{name}", "events": events,
                          "annotation_known": known, "annotation_source": annotation_source,
                          "annotation_error": annotation_error,
                          "positive_list_disagreement": name in positive_list_disagreements,
                          "summary_present": summary is not None,
                          "summary_sample_rate": summary.sample_rate if summary is not None else None,
                          "summary_sha256": summary_hashes.get(case),
                          "annotation_semantics_source": ANNOTATION_SEMANTICS_URL})
    return inventory, {"source_edf_count": len(names), "positive_edf_count": len(positives),
                       "source_checksum_entries": len(checksums), "metadata_sha256": metadata_audit,
                       "positive_list_sidecar_disagreements": positive_list_disagreements,
                       "reconciliation_policy": "summary intervals and sidecar membership must agree; list differences retained",
                       "checksum_manifest_sha256": file_sha256(sums_path),
                       "auxiliary_missing_or_corrupt": auxiliary_errors,
                       "summary_errors": summary_errors,
                       "summary_unlisted_files": [r["source_file_id"] for r in inventory
                                                  if not r["summary_present"]]}


def _process_record(task: dict) -> tuple[dict, np.ndarray, bool]:
    """Workers return metadata/index only; whole EEG arrays stay inside a worker."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    root, output = Path(task["root"]), Path(task["output"])
    record, protocol = dict(task["record"]), task["protocol"]
    source = root / record["source_file_id"]
    folder = output / "records" / f"{record['index']:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    done_path = folder / "done.json"
    signature_protocol = {key: value for key, value in protocol.items() if key != "inventory"}
    signature_protocol["master_metadata_sha256"] = protocol["inventory"]["metadata_sha256"]
    signature_protocol["checksum_manifest_sha256"] = protocol["inventory"]["checksum_manifest_sha256"]
    signature_payload = {"source_sha256": record["source_sha256"], "record": record,
                         "protocol": signature_protocol}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    empty = np.empty(0, dtype=INDEX_DTYPE)
    try:
        if not source.is_file():
            record.update({"status": "pending_download", "pending_reason": "source EDF not yet present",
                           "source_verified": False, "indexed_windows": 0})
            return record, empty, False
        if Path(str(source) + ".aria2").exists():
            record.update({"status": "pending_download", "pending_reason": "source EDF aria2 download is active",
                           "source_verified": False, "indexed_windows": 0})
            return record, empty, False
        if file_sha256(source) != record["source_sha256"]:
            record.update({"status": "pending_download", "pending_reason": "source EDF checksum not yet valid",
                           "source_verified": False, "indexed_windows": 0})
            return record, empty, False
        record["source_verified"] = True
        if done_path.is_file():
            done = json.loads(done_path.read_text())
            if done.get("signature") == signature and done.get("complete", False):
                metadata = done["record"]
                if metadata.get("waveform_available", True):
                    waveform = np.load(folder / "waveform.npy", mmap_mode="r", allow_pickle=False)
                    if list(waveform.shape) != metadata["waveform_shape"] or waveform.dtype != np.float32:
                        raise ValueError("cached shard shape/dtype mismatch")
                elif metadata["status"] != "isolated" or metadata["indexed_windows"] != 0:
                    raise ValueError("cached record without waveform is not an empty isolation")
                rows = np.load(folder / "index.npy", allow_pickle=False)
                if rows.dtype != INDEX_DTYPE or len(rows) != metadata["indexed_windows"]:
                    raise ValueError("cached index shape/dtype mismatch")
                return metadata, rows, True
        policy = protocol["explicit_common_reference_policy"]
        common_reference = policy.get(source.name)
        try:
            physical = read_edf_bipolar(source, allow_missing=True, common_reference=common_reference)
        except UnsupportedMontageError as error:
            evidence = error.header_evidence
            reasons = list(evidence.get("isolation_reasons", ["unsupported_montage"]))
            if not evidence["has_explicit_channel_pairs"] and common_reference is None:
                reasons.append("unknown_reference")
            record.update({"status": "isolated", "isolation_reasons": reasons,
                           "waveform_available": False, "processed": False,
                           "indexed_windows": 0, "source_header_evidence": evidence,
                           "source_sample_rate": evidence["source_sample_rate"],
                           "preprocessing_context": [0.0, evidence["source_samples"]
                                                     / evidence["source_sample_rate"]],
                           "protocol_signature": signature})
            _atomic_npy(folder / "index.npy", empty)
            _atomic_json(done_path, {"complete": True, "signature": signature, "record": record})
            return record, empty, False
        if (record["summary_sample_rate"] is not None
                and not math.isclose(physical.sample_rate, record["summary_sample_rate"])):
            raise ValueError("summary/EDF sampling rate disagreement")
        duration = physical.waveform.shape[1] / physical.sample_rate
        if any(event["offset"] > duration for event in record["events"]):
            raise ValueError("event annotation exceeds source duration")
        samples, target_rate = protocol["window_samples"], protocol["sample_rate"]
        raw_window = samples * physical.sample_rate / target_rate
        if not math.isclose(raw_window, round(raw_window)):
            raise ValueError("window duration cannot align with source sample grid")
        observed = physical.waveform[physical.lead_valid]
        qc_fn: Callable = task.get("qc_fn") or window_quality_flags
        raw_flags = qc_fn(observed, physical.sample_rate, round(raw_window))
        nonfinite = bool(np.any(raw_flags & Q_NONFINITE)) or not np.isfinite(observed).all()
        del observed
        waveform = physical.waveform
        processed = False
        if not nonfinite:
            bandpass = protocol["bandpass"]
            if bandpass is not None:
                low, high = bandpass
                if not 0 < low < high < min(physical.sample_rate, target_rate) / 2:
                    raise ValueError("bandpass must lie below source and target Nyquist")
                sos = butter(4, bandpass, fs=physical.sample_rate, btype="bandpass", output="sos")
                # No explicit full-record float64 input copy. SciPy handles its
                # filter work arrays; the canonical float32 source remains audit input.
                waveform = sosfiltfilt(sos, waveform, axis=-1)
            if not math.isclose(physical.sample_rate, target_rate):
                ratio = Fraction(target_rate / physical.sample_rate).limit_denominator(10000)
                waveform = resample_poly(waveform, ratio.numerator, ratio.denominator, axis=-1)
            waveform = waveform.astype(np.float32, copy=False)
            processed = True
        shard_rate = target_rate if processed else physical.sample_rate
        indexed_samples = samples if processed else round(raw_window)
        filtered_flags = qc_fn(waveform[physical.lead_valid], shard_rate, indexed_samples)
        windows = min(len(raw_flags), len(filtered_flags))
        flags = np.asarray(raw_flags[:windows] | filtered_flags[:windows], dtype=np.uint16)
        rows = np.zeros(windows, dtype=INDEX_DTYPE)
        rows["record"] = record["index"]
        rows["offset"] = np.arange(windows, dtype=np.int64) * indexed_samples
        rows["label"] = 0 if record["annotation_known"] else -1
        rows["qc_flags"] = flags
        rows["ref_index"] = -1
        for event in record["events"]:
            starts = rows["offset"] / shard_rate
            overlaps = (starts < event["offset"]) & (starts + indexed_samples / shard_rate > event["onset"])
            rows["label"][overlaps] = 1
        complete_montage = bool(physical.lead_valid.all())
        reasons = []
        if not complete_montage:
            reasons.append("incomplete_standard_bipolar_montage")
        if not record["annotation_known"]:
            reasons.append("unknown_annotation")
        if nonfinite:
            reasons.append("nonfinite_raw_record_not_filtered")
        relative = str((folder / "waveform.npy").relative_to(output))
        record.update({"status": "ready" if not reasons else "isolated", "isolation_reasons": reasons,
                       "waveform_available": True, "waveform_path": relative,
                       "waveform_shape": list(waveform.shape),
                       "sample_rate": shard_rate, "source_sample_rate": physical.sample_rate,
                       "window_samples": indexed_samples, "unit": "uV", "dtype": "float32",
                       "lead_names": list(physical.lead_names), "lead_valid": physical.lead_valid.tolist(),
                       "lead_geometry": physical.lead_geometry.tolist(),
                       "lead_provenance": physical.provenance,
                       "preprocessing_context": [0.0, duration], "processed": processed,
                       "indexed_windows": windows,
                       "incomplete_tail_samples": waveform.shape[1] - windows * indexed_samples,
                       "qc_counts": {str(bit): int(np.count_nonzero(flags & bit)) for bit in (1, 2, 4)},
                       "protocol_signature": signature})
        _atomic_npy(folder / "waveform.npy", waveform)
        _atomic_npy(folder / "index.npy", rows)
        _atomic_json(done_path, {"complete": True, "signature": signature, "record": record})
        return record, rows, False
    except Exception as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                       "indexed_windows": 0, "protocol_signature": signature})
        _atomic_json(done_path, {"complete": False, "signature": signature, "record": record})
        return record, empty, False


def _attach_sharded_references(index: np.ndarray, records: list[dict], seed: int) -> None:
    source_records = np.asarray([record["split"] == "train" and record["role"] == "source"
                                 for record in records], dtype=bool)
    candidates = np.flatnonzero(source_records[index["record"]])
    patients = [record["canonical_subject_id"] for record in records]
    source_patient = np.asarray(patients)[index["record"][candidates]]
    pools = {label: {} for label in (0, 1)}
    all_pools = {}
    for patient in sorted(set(source_patient)):
        patient_candidates = candidates[source_patient == patient]
        all_pools[patient] = patient_candidates
        for label in (0, 1):
            same_label = patient_candidates[index["label"][patient_candidates] == label]
            if len(same_label):
                pools[label][patient] = same_label
    donor_cache = {}
    for target_index, target in enumerate(index):
        patient = patients[int(target["record"])]
        sample_id = f"{records[int(target['record'])]['record_id']}:{int(target['offset'])}"
        rotation = int.from_bytes(hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=8).digest(), "little")
        key = (patient, int(target["label"]))
        if key not in donor_cache:
            donors = [pool for donor, pool in pools[key[1]].items() if donor != patient]
            if not donors:
                donors = [pool for donor, pool in all_pools.items() if donor != patient]
            donor_cache[key] = donors
        donors = donor_cache[key]
        if donors:
            pool = donors[rotation % len(donors)]
            # Uniform donor-patient choice avoids a long single-patient search
            # while early builds have only one downloaded training subject.
            index["ref_index"][target_index] = int(pool[(rotation // len(donors)) % len(pool)])


def build_chbmit(root: str | Path, output: str | Path,
                 split_map: dict[str, str] | str | Path | None = None, *,
                 samples: int = 1024, sample_rate: float = 256.0, workers: int = 2,
                 bandpass: tuple[float, float] | None = (0.1, 80.0), seed: int = 0,
                 qc_fn: Callable | None = None, exclude_qc_bits: int = DEFAULT_EXCLUDED_FLAGS,
                 expected_records: int | None = 686) -> Path:
    """Build full source inventory, retained record shards, ready and isolated indices.

    Individual missing, corrupt or unsupported records are reported and isolated;
    invalid master inventories/split maps fail before any EEG processing. Setting
    expected_records=None is intended for small engineering fixtures only.
    """
    root, output = Path(root), Path(output)
    if samples <= 0 or sample_rate <= 0 or workers <= 0:
        raise ValueError("positive samples, sample_rate and workers are required")
    if split_map is None:
        split_map = _default_splits()
    elif isinstance(split_map, (str, Path)):
        split_map = json.loads(Path(split_map).read_text())
    output.mkdir(parents=True, exist_ok=True)
    splits = lock_patient_splits(split_map, output / "split_lock.json")
    inventory, inventory_audit = _load_inventory(root, expected_records)
    duplicate_hashes = {}
    for record in inventory:
        patient = record["canonical_subject_id"]
        if patient not in splits:
            raise ValueError(f"patient missing from locked split map: {patient}")
        record["split"] = splits[patient]
        record["role"] = "source" if record["split"] == "train" else "query"
        digest = record["source_sha256"]
        if digest in duplicate_hashes:
            record["duplicate_of"] = duplicate_hashes[digest]
        else:
            duplicate_hashes[digest] = record["source_file_id"]
    protocol = {"schema_version": "1.0", "pipeline": PIPELINE_VERSION,
                "dataset": "CHB-MIT", "dataset_version": "1.0.0", "evaluation_protocol": "source-only",
                "window_samples": samples, "sample_rate": sample_rate, "window_seconds": samples / sample_rate,
                "bandpass": list(bandpass) if bandpass is not None else None,
                "filter": "scipy butter order4 sosfiltfilt whole record",
                "resample": "scipy resample_poly whole record", "unit": "uV",
                "lead_names": list(STANDARD_BIPOLAR_LEADS), "exclude_qc_bits": exclude_qc_bits,
                "reference_seed": seed, "reference_policy": "different_patient_train_source_same_label_preferred",
                "explicit_common_reference_policy": {"chb12_27.edf": "CS2"},
                "processor_sha256": file_sha256(__file__),
                "reader_sha256": file_sha256(Path(__file__).with_name("chbmit.py")),
                "qc_sha256": file_sha256(Path(__file__).with_name("qc.py")),
                "inventory": inventory_audit}
    _atomic_json(output / "protocol.json", protocol)
    tasks = [{"root": str(root), "output": str(output), "record": record,
              "protocol": protocol, "qc_fn": qc_fn} for record in inventory]
    if workers == 1:
        results = list(map(_process_record, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process_record, tasks))
    records, ready_parts, isolated_parts = [], [], []
    resumed = 0
    for metadata, rows, reused in results:
        resumed += int(reused)
        if metadata.get("duplicate_of") and metadata["status"] in {"ready", "isolated"}:
            metadata["status"] = "isolated"
            metadata.setdefault("isolation_reasons", []).append("duplicate_source_content")
        records.append(metadata)
        eligible_record = metadata["status"] == "ready"
        accepted = (rows["qc_flags"] & exclude_qc_bits) == 0
        if not eligible_record:
            accepted[:] = False
        ready_parts.append(rows[accepted])
        isolated_parts.append(rows[~accepted])
    ready = np.concatenate(ready_parts) if ready_parts else np.empty(0, dtype=INDEX_DTYPE)
    isolated = np.concatenate(isolated_parts) if isolated_parts else np.empty(0, dtype=INDEX_DTYPE)
    _attach_sharded_references(ready, records, seed)
    _atomic_npy(output / "window_index.npy", ready)
    _atomic_npy(output / "isolated_index.npy", isolated)
    _atomic_json(output / "records.json", {"schema_version": "1.0", "dataset": "CHB-MIT",
                                          "dataset_version": "1.0.0", "unit": "uV", "records": records})
    statuses = {status: sum(record["status"] == status for record in records)
                for status in ("ready", "isolated", "failed", "pending_download")}
    report = {"schema_version": "1.0", "raw_directory": str(root.resolve()),
              "source_edfs_expected": len(inventory), "records": statuses,
              "resumed_records": resumed, "ready_windows": len(ready), "isolated_windows": len(isolated),
              "reference_missing_windows": int(np.count_nonzero(ready["ref_index"] < 0)),
              "split_ready_windows": {split: int(sum(np.count_nonzero(ready["record"] == record["index"])
                                                      for record in records if record["split"] == split))
                                      for split in ("train", "validation", "test")},
              "failed_records": [{"source_file_id": record["source_file_id"], "error": record["error"]}
                                 for record in records if record["status"] == "failed"],
              "pending_records": [{"source_file_id": record["source_file_id"],
                                   "reason": record["pending_reason"]}
                                  for record in records if record["status"] == "pending_download"],
              "isolated_records": [{"source_file_id": record["source_file_id"],
                                    "reasons": record["isolation_reasons"]}
                                   for record in records if record["status"] == "isolated"],
              "inventory_audit": inventory_audit,
              "source_download_complete": (all(record.get("source_verified", False) for record in records)
                                           and not inventory_audit["auxiliary_missing_or_corrupt"]),
              "source_edf_download_complete": all(record.get("source_verified", False) for record in records),
              "source_fully_processed": statuses["failed"] == 0 and statuses["pending_download"] == 0,
              "notes": ["No full-target CWT is used as an external condition.",
                        "Negative membership means no annotated seizure, not a healthy control.",
                        "Flagged high amplitude alone is retained; no new clinical QC judgment is claimed."]}
    _atomic_json(output / "build_report.json", report)
    return output / "window_index.npy"
