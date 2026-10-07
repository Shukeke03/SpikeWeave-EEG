"""Independent compact-index audits; numerical EEG reads are train/source only.

The persisted report describes engineering acceptance, not clinical validity.
Source checksum counts reuse the processor's verification ledger and do not
pretend that this inexpensive audit rehashed every original EDF.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from triwave.config import ModelConfig
from triwave.signal import MorletCWT, WaveletTransform

from .build import INDEX_DTYPE
from .dataset import TrainNormalizer
from .manifest import canonical_patient
from .qc import Q_FLATLINE, Q_HIGH_AMPLITUDE, Q_NONFINITE

OFFICIAL_CHBMIT_RECORDS = 686
OFFICIAL_CHBMIT_EVENTS = 198
KNOWN_QC_BITS = Q_NONFINITE | Q_FLATLINE | Q_HIGH_AMPLITUDE


def _write_report(root: Path, report: dict) -> dict:
    report["valid"] = not report["errors"]
    report["status"] = ("invalid" if not report["valid"] else
                        "complete" if report.get("completeness", {}).get("complete", False)
                        else "not_complete")
    temporary = root / "dataset_audit.json.partial"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, root / "dataset_audit.json")
    return report


def _asset(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("asset path escapes dataset root")
    return path


def _storage(root: Path | None) -> dict:
    if root is None or not root.is_dir():
        return {"available": False}
    kinds: dict[str, Counter] = defaultdict(Counter)
    seen = set()
    skipped = 0
    for folder, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(folder) / name
            try:
                if path.is_symlink():
                    skipped += 1
                    continue
                stat = path.stat()
            except OSError:
                skipped += 1
                continue
            kind = path.suffix.lower() or "no_extension"
            kinds[kind]["files"] += 1
            kinds[kind]["apparent_bytes"] += stat.st_size
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                kinds[kind]["allocated_bytes"] += getattr(stat, "st_blocks", 0) * 512
                seen.add(key)
    return {"available": True, "directory": str(root.resolve()),
            "files": sum(value["files"] for value in kinds.values()),
            "apparent_bytes": sum(value["apparent_bytes"] for value in kinds.values()),
            "allocated_bytes": sum(value["allocated_bytes"] for value in kinds.values()),
            "by_extension": {key: dict(value) for key, value in sorted(kinds.items())},
            "skipped_files_or_symlinks": skipped,
            "allocation_method": "unique inode st_blocks * 512"}


def _groups(index: np.ndarray, records: int) -> list[np.ndarray]:
    order = np.argsort(index["record"], kind="stable")
    keys = index["record"][order]
    boundaries = np.searchsorted(keys, np.arange(records + 1))
    return [order[boundaries[i]:boundaries[i + 1]] for i in range(records)]


def _normalizer(root: Path, records: list[dict], ready: np.ndarray,
                groups: list[np.ndarray], protocol: dict, error) -> tuple[dict, TrainNormalizer | None]:
    path = root / "train_scale_v2.json"
    if not path.is_file():
        path = root / "normalizer.json"
    if not path.is_file():
        alternate = root / "train_normalizer.json"
        if alternate.is_file():
            path = alternate
        else:
            return {"status": "missing", "fitted_during_audit": False}, None
    errors_before = error.count
    try:
        normalizer = TrainNormalizer.load(path)
        by_id = {record["record_id"]: i for i, record in enumerate(records)}
        ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        count = 0
        for declared in normalizer.fit_sample_ids:
            record_id, interval = declared.rsplit(":", 1)
            first, stop = map(int, interval.split("-"))
            record_index = by_id[record_id]
            record = records[record_index]
            samples = int(record["window_samples"])
            if record["split"] != "train" or record["role"] != "source":
                raise ValueError("normalizer fitted range includes a nontraining source")
            if first < 0 or stop <= first or first % samples or stop % samples:
                raise ValueError("normalizer fitted range is not an aligned positive interval")
            offsets = np.sort(ready["offset"][groups[record_index]])
            left, right = np.searchsorted(offsets, [first, stop])
            if (right - left != (stop - first) // samples
                    or not np.array_equal(offsets[left:right], np.arange(first, stop, samples))):
                raise ValueError("normalizer fitted range includes unavailable or excluded windows")
            ranges[record_index].append((first, stop))
            count += (stop - first) * len(record["lead_names"])
        for intervals in ranges.values():
            intervals.sort()
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError("normalizer fitted ranges overlap")
        train_mask = np.asarray([record["split"] == "train" and record["role"] == "source"
                                 for record in records])[ready["record"]]
        train_rows = ready[train_mask]
        expected_count = len(train_rows) * len(protocol["lead_names"]) * protocol["window_samples"]
        if count != normalizer.count or count != expected_count:
            raise ValueError("normalizer count/ranges do not cover exactly eligible source training windows")
        digest = hashlib.sha256(train_rows.tobytes())
        digest.update(json.dumps(protocol, sort_keys=True).encode())
        digest.update((root / "split_lock.json").read_bytes())
        for record_index in np.unique(train_rows["record"]):
            digest.update(records[int(record_index)]["source_sha256"].encode())
        if normalizer.fit_manifest_digest != digest.hexdigest():
            raise ValueError("normalizer training index/source checksum digest is stale")
        return {"status": "valid", "path": path.name, "count": normalizer.count,
                "expected_source_train_count": expected_count, "mean_uV": normalizer.mean,
                "std_uV": normalizer.std, "fit_range_count": len(normalizer.fit_sample_ids),
                "fit_record_count": len(ranges), "source_only": True,
                "scale_protocol": normalizer.scale_protocol,
                "v02_diffusion_compatible": normalizer.scale_protocol == "source_rms_v2",
                "fit_manifest_digest": normalizer.fit_manifest_digest,
                "fitted_during_audit": False}, normalizer
    except (OSError, ValueError, KeyError, TypeError) as exception:
        error("normalizer", str(exception))
        return {"status": "invalid", "path": path.name, "fitted_during_audit": False,
                "errors": error.count - errors_before}, None


def _numerical(root: Path, records: list[dict], ready: np.ndarray, config: ModelConfig,
               normalizer: TrainNormalizer | None, limit: int, device: str) -> dict:
    if limit == 0:
        return {"status": "skipped", "reason": "numerical_windows=0", "windows": 0}
    if normalizer is None:
        return {"status": "skipped", "reason": "valid saved source normalizer is unavailable", "windows": 0}
    train = np.asarray([record["split"] == "train" and record["role"] == "source"
                        for record in records])[ready["record"]]
    candidates = np.flatnonzero(train)
    if not len(candidates):
        return {"status": "skipped", "reason": "no eligible source training windows", "windows": 0}
    positions = np.linspace(0, len(candidates) - 1, min(limit, len(candidates)), dtype=np.int64)
    selected = candidates[positions]
    target_device = torch.device(device)
    wavelet = WaveletTransform(config.samples, config.wavelet_levels, config.wavelet).to(target_device)
    cwt = MorletCWT(config.sample_rate, config.cwt_frequencies, config.cwt_min_frequency,
                    config.cwt_max_frequency, config.cwt_times).to(target_device)
    frequency_valid = np.zeros(config.cwt_frequencies, dtype=np.int64)
    frequency_total = np.zeros_like(frequency_valid)
    reference_valid = np.zeros_like(frequency_valid)
    reference_total = np.zeros_like(frequency_valid)
    max_error, squared_error, values, amplitude = 0.0, 0.0, 0, 0.0
    references = []
    cwt_finite, invalid_bins_zero = True, True

    def load(row: np.void) -> torch.Tensor:
        record = records[int(row["record"])]
        # This guard applies to references as well as numerical target windows.
        if record["split"] != "train" or record["role"] != "source":
            raise ValueError("numerical audit cannot read nontraining EEG")
        data = np.load(_asset(root, record["waveform_path"]), mmap_mode="r", allow_pickle=False)
        first = int(row["offset"])
        waveform = np.array(data[:, first:first + config.samples], dtype=np.float32, copy=True)
        if waveform.shape != (config.channels, config.samples) or not np.isfinite(waveform).all():
            raise ValueError("numerical training window is nonfinite or has wrong shape")
        return torch.from_numpy(normalizer.transform(waveform)).to(target_device)[None]

    with torch.no_grad():
        # One window at a time bounds FFT work buffers independently of the
        # full dataset size and of the requested diagnostic sample count.
        for global_index in selected:
            row = ready[int(global_index)]
            waveform = load(row)
            residual = wavelet.decode(wavelet.encode(waveform)) - waveform
            max_error = max(max_error, float(residual.abs().max()))
            squared_error += float(residual.double().square().sum())
            values += residual.numel()
            amplitude = max(amplitude, float(waveform.abs().max()))
            power, valid = cwt(waveform)
            cwt_finite &= bool(torch.isfinite(power).all())
            invalid_bins_zero &= bool((power[~valid] == 0).all())
            frequency_valid += valid.sum(dim=(0, 1, 3)).cpu().numpy()
            frequency_total += valid.shape[0] * valid.shape[1] * valid.shape[3]
            reference_index = int(row["ref_index"])
            if reference_index >= 0:
                reference_row = ready[reference_index]
                reference_record = records[int(reference_row["record"])]
                target_record = records[int(row["record"])]
                if (reference_record["canonical_subject_id"] == target_record["canonical_subject_id"]
                        or reference_record["source_sha256"] == target_record["source_sha256"]):
                    raise ValueError("reference CWT origin is not independent")
                reference_power, mask = cwt(load(reference_row))
                cwt_finite &= bool(torch.isfinite(reference_power).all())
                invalid_bins_zero &= bool((reference_power[~mask] == 0).all())
                reference_valid += mask.sum(dim=(0, 1, 3)).cpu().numpy()
                reference_total += mask.shape[0] * mask.shape[1] * mask.shape[3]
                references.append({"target_index": int(global_index), "reference_index": reference_index,
                                   "target_record": int(row["record"]),
                                   "reference_record": int(reference_row["record"])})
    tolerance = max(2e-5, 5e-6 * amplitude)
    return {"status": "complete", "windows": len(selected), "selected_indices": selected.tolist(),
            "selection": "evenly spaced global train/source ready indices; no nontraining EEG reads",
            "device": str(target_device), "input_unit": "normalized_uV", "source_normalizer_used": True,
            "dwt_idwt": {"max_absolute_error": max_error, "rms_error": (squared_error / values) ** 0.5,
                         "max_absolute_error_uV": max_error * normalizer.std,
                         "acceptance_tolerance": tolerance, "passed": max_error <= tolerance},
            "cwt": {"frequencies_hz": cwt.frequencies.cpu().tolist(), "finite": cwt_finite,
                    "invalid_bins_zero": invalid_bins_zero,
                    "valid_fraction_by_frequency": (frequency_valid / frequency_total).tolist(),
                    "reference_windows": len(references),
                    "reference_valid_fraction_by_frequency": np.divide(
                        reference_valid, reference_total, out=np.zeros_like(reference_valid, dtype=float),
                        where=reference_total > 0).tolist(),
                    "reference_origins": references,
                    "reference_origin_policy": "independent train/source record, patient and source checksum",
                    "target_cwt_role": "diagnostic teacher only; never a generation condition"}}


def validate_sharded_dataset(root: str | Path, *, config: ModelConfig | None = None,
                             numerical_windows: int = 16, device: str = "cpu") -> dict:
    """Audit all compact rows, then at most N normalized training EEG windows.

    Partial downloads are valid previews and report ``not_complete``. Structural
    failures are collected in ``errors`` and report ``invalid`` rather than
    silently accepting the dataset. Missing normalization skips numerical checks;
    this function never fits statistics. The report is atomically saved beside
    the index as ``dataset_audit.json`` and should remain with private data.
    """
    if not isinstance(numerical_windows, int) or isinstance(numerical_windows, bool) or numerical_windows < 0:
        raise ValueError("numerical_windows must be a nonnegative integer")
    root = Path(root)
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = {"schema_version": "1.0", "audit_version": "sharded-audit-v1", "errors": [],
              "warnings": [], "numerical": {"status": "not_run"}}

    def error(check: str, message: str, count: int = 1) -> None:
        report["errors"].append({"check": check, "message": message, "count": int(count)})
        error.count += 1

    error.count = 0
    try:
        records = json.loads((root / "records.json").read_text())["records"]
        protocol = json.loads((root / "protocol.json").read_text())
        build = json.loads((root / "build_report.json").read_text())
        split_lock = json.loads((root / "split_lock.json").read_text())["patients"]
        ready = np.load(root / "window_index.npy", mmap_mode="r", allow_pickle=False)
        isolated = np.load(root / "isolated_index.npy", mmap_mode="r", allow_pickle=False)
        if any(index.ndim != 1 or index.dtype != INDEX_DTYPE for index in (ready, isolated)):
            raise ValueError("index must be a one-dimensional array with the supported structured dtype")
        if not records:
            raise ValueError("record inventory is empty")
        if protocol["unit"] != "uV" or protocol["evaluation_protocol"] != "source-only":
            raise ValueError("unsupported physical unit or evaluation protocol")
        if protocol["exclude_qc_bits"] < 0 or protocol["exclude_qc_bits"] & ~KNOWN_QC_BITS:
            raise ValueError("exclusion policy contains unknown QC bits")
    except (OSError, ValueError, KeyError, TypeError) as exception:
        error("required_files_or_schema", str(exception))
        return _write_report(root, report)

    patients: dict[str, set[str]] = defaultdict(set)
    statuses = Counter()
    event_records: dict[str, int] = {}
    source_hashes: dict[str, list[int]] = defaultdict(list)
    record_ids = set()
    ready_shape: dict[int, tuple[int, int]] = {}
    for position, record in enumerate(records):
        try:
            if record["index"] != position:
                raise ValueError("record index differs from its metadata table position")
            patient = canonical_patient(record["subject_id"])
            if record["canonical_subject_id"] != patient:
                raise ValueError("canonical patient identity disagrees with known identity rule")
            if record["split"] not in {"train", "validation", "test"}:
                raise ValueError("unknown patient split")
            if split_lock.get(patient) != record["split"]:
                raise ValueError("patient split disagrees with split lock")
            if record["role"] != ("source" if record["split"] == "train" else "query"):
                raise ValueError("source-only role disagrees with patient split")
            patients[patient].add(record["split"])
            statuses[record["status"]] += 1
            if record["status"] not in {"ready", "isolated", "failed", "pending_download"}:
                raise ValueError("unknown record processing status")
            if record["status"] in {"ready", "isolated"} and not record.get("source_verified", False):
                raise ValueError("saved recording has no verified source checksum ledger")
            if record["record_id"] in record_ids:
                raise ValueError("duplicate record ID")
            record_ids.add(record["record_id"])
            digest = record["source_sha256"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ValueError("missing or malformed source SHA256")
            source_hashes[digest].append(position)
            previous_offset = 0.0
            for event in record["events"]:
                if (not np.isfinite([event["onset"], event["offset"]]).all()
                        or event["onset"] < previous_offset or event["offset"] <= event["onset"]):
                    raise ValueError("unordered, overlapping or invalid event interval")
                previous_offset = event["offset"]
                if event["event_id"] in event_records:
                    raise ValueError("event ID is duplicated across metadata")
                event_records[event["event_id"]] = position
            if record["status"] in {"ready", "isolated"}:
                if not record.get("waveform_available", True):
                    if record["status"] != "isolated" or record["indexed_windows"] != 0:
                        raise ValueError("record without waveform must be an empty explicit isolation")
                    evidence = record["source_header_evidence"]
                    rate, length = evidence["source_sample_rate"], evidence["source_samples"]
                    if not np.isfinite(rate) or rate <= 0 or length <= 0:
                        raise ValueError("isolated source header has invalid duration metadata")
                    if any(event["offset"] > length / rate for event in record["events"]):
                        raise ValueError("isolated event exceeds original source duration")
                    continue
                waveform = np.load(_asset(root, record["waveform_path"]), mmap_mode="r", allow_pickle=False)
                if (waveform.ndim != 2 or list(waveform.shape) != record["waveform_shape"]
                        or waveform.dtype != np.float32 or record["unit"] != "uV"
                        or waveform.shape[0] != len(protocol["lead_names"])
                        or record["lead_names"] != protocol["lead_names"]):
                    raise ValueError("shard header does not match shape, dtype, unit or lead definition")
                ready_shape[position] = waveform.shape
                geometry = np.asarray(record["lead_geometry"])
                lead_mask = np.asarray(record["lead_valid"])
                if (geometry.shape != (waveform.shape[0], 6) or not np.isfinite(geometry).all()
                        or lead_mask.shape != (waveform.shape[0],) or lead_mask.dtype != bool):
                    raise ValueError("invalid lead geometry or observation mask")
                samples = record["window_samples"]
                if (not isinstance(samples, int) or samples <= 0
                        or not np.isfinite(record["sample_rate"]) or record["sample_rate"] <= 0):
                    raise ValueError("invalid record measurement rate/window size")
                if record["status"] == "ready" and (
                        not all(record["lead_valid"]) or not record["processed"]
                        or samples != protocol["window_samples"]
                        or not np.isclose(record["sample_rate"], protocol["sample_rate"])):
                    raise ValueError("ready recording is incomplete or has a mismatched measurement definition")
                if any(event["offset"] > waveform.shape[1] / record["sample_rate"]
                       for event in record["events"]):
                    raise ValueError("event extends beyond saved recording")
        except (OSError, ValueError, KeyError, TypeError) as exception:
            error("record_metadata", f"record {position}: {exception}")
    overlaps = [patient for patient, splits in patients.items() if len(splits) > 1]
    if overlaps:
        error("patient_split_overlap", ", ".join(overlaps), len(overlaps))
    if any(item["check"] == "record_metadata" for item in report["errors"]):
        return _write_report(root, report)

    for name, index in (("ready", ready), ("isolated", isolated)):
        bad_record = (index["record"] < 0) | (index["record"] >= len(records))
        if bad_record.any():
            error("record_bounds", f"{name} index contains out-of-range record indices", bad_record.sum())
        if np.any(index["qc_flags"] & (0xFFFF ^ KNOWN_QC_BITS)):
            error("qc_bits", f"{name} index contains unknown QC bits")
        if np.any(~np.isin(index["label"], (0, 1) if name == "ready" else (-1, 0, 1))):
            error("labels", f"{name} index contains unsupported labels")
        if name == "ready" and np.any(index["qc_flags"] & protocol["exclude_qc_bits"]):
            error("ready_qc", "ready index contains excluded quality flags")
        if name == "isolated" and np.any(index["ref_index"] != -1):
            error("isolated_reference", "isolated windows must not have generation references")
    # Bounds errors preclude safe table indexing and later numerical reads.
    if any(item["check"] == "record_bounds" for item in report["errors"]):
        return _write_report(root, report)
    ready_groups, isolated_groups = _groups(ready, len(records)), _groups(isolated, len(records))
    all_pairs = np.empty(len(ready) + len(isolated), dtype=[("record", "<i4"), ("offset", "<i8")])
    all_pairs["record"] = np.concatenate([ready["record"], isolated["record"]])
    all_pairs["offset"] = np.concatenate([ready["offset"], isolated["offset"]])
    _, repetitions = np.unique(all_pairs, return_counts=True)
    duplicates = int(np.sum(repetitions - 1))
    if duplicates:
        error("duplicate_windows", "record/offset pairs repeat within or across indices", duplicates)

    retained_events, isolated_events = set(), set()
    counts = {"ready_windows": len(ready), "isolated_windows": len(isolated), "records": dict(statuses),
              "canonical_patients": len(patients), "patient_split_overlap": len(overlaps),
              "duplicate_windows": duplicates, "by_split_label": {}, "qc": {}, "tails": {}}
    for name, index, groups in (("ready", ready, ready_groups), ("isolated", isolated, isolated_groups)):
        split_counts = {}
        for split in ("train", "validation", "test"):
            mask = np.asarray([record.get("split") == split for record in records])[index["record"]]
            split_counts[split] = {str(label): int(np.count_nonzero(mask & (index["label"] == label)))
                                   for label in (-1, 0, 1)}
        counts["by_split_label"][name] = split_counts
        counts["qc"][name] = {"nonfinite": int(np.count_nonzero(index["qc_flags"] & Q_NONFINITE)),
                              "flatline": int(np.count_nonzero(index["qc_flags"] & Q_FLATLINE)),
                              "high_amplitude": int(np.count_nonzero(index["qc_flags"] & Q_HIGH_AMPLITUDE))}
        for position, record in enumerate(records):
            rows = index[groups[position]]
            if not len(rows):
                continue
            if position not in ready_shape:
                error("unavailable_record", f"{name} windows refer to unsaved record {position}")
                continue
            samples = int(record["window_samples"])
            if np.any((rows["offset"] < 0) | (rows["offset"] % samples != 0)
                      | (rows["offset"] + samples > ready_shape[position][1])):
                error("offset_bounds", f"{name} record {position} has invalid window offsets")
            if name == "ready" and record["status"] != "ready":
                error("ready_status", f"ready index refers to isolated record {position}")
            expected = np.zeros(len(rows), dtype=np.int8) if record["annotation_known"] else np.full(len(rows), -1)
            starts = rows["offset"] / record["sample_rate"]
            for event in record["events"]:
                overlap = (starts < event["offset"]) & (starts + samples / record["sample_rate"] > event["onset"])
                expected[overlap] = 1
                if overlap.any():
                    (retained_events if name == "ready" else isolated_events).add(event["event_id"])
            if np.any(rows["label"] != expected):
                error("annotation_labels", f"{name} record {position} labels disagree with event intervals")

    tail_samples, tail_seconds, audited_hours = 0, 0.0, 0.0
    for position, shape in ready_shape.items():
        record = records[position]
        actual = len(ready_groups[position]) + len(isolated_groups[position])
        expected = shape[1] // record["window_samples"]
        tail = shape[1] - expected * record["window_samples"]
        if actual != expected or actual != record["indexed_windows"]:
            error("window_coverage", f"record {position} complete-window count disagrees with saved shard")
        if record["incomplete_tail_samples"] != tail:
            error("tail_accounting", f"record {position} tail count disagrees with saved shard")
        tail_samples += tail
        tail_seconds += tail / record["sample_rate"]
        audited_hours += shape[1] / record["sample_rate"] / 3600
    counts["tails"] = {"samples": tail_samples, "seconds": tail_seconds,
                        "records_with_saved_waveform": len(ready_shape)}
    counts["saved_waveform_hours"] = audited_hours
    counts["events"] = {"known_inventory": len(event_records), "with_ready_windows": len(retained_events),
                        "with_isolated_windows": len(isolated_events),
                        "only_isolated_windows": len(isolated_events - retained_events),
                        "without_any_saved_windows": len(set(event_records) - retained_events - isolated_events),
                        "inventory_by_record_status": {
                            status: sum(len(record["events"]) for record in records if record["status"] == status)
                            for status in ("ready", "isolated", "failed", "pending_download")},
                        "retained_by_split": {split: sum(records[event_records[event]]["split"] == split
                                                         for event in retained_events)
                                              for split in ("train", "validation", "test")}}
    counts["unknown_annotation_records"] = sum(not record.get("annotation_known", False) for record in records)
    report["counts"] = counts
    for key, actual in (("ready_windows", len(ready)), ("isolated_windows", len(isolated))):
        if build.get(key) != actual:
            error("build_report_counts", f"{key} disagrees with actual index")
    if build.get("source_edfs_expected") != len(records):
        error("build_report_counts", "source EDF inventory count disagrees with metadata")
    verified = sum(bool(record.get("source_verified", False)) for record in records)
    duplicate_sources = [positions for positions in source_hashes.values() if len(positions) > 1]
    report["checksums"] = {"verified_record_ledger": verified, "unverified_record_ledger": len(records) - verified,
                           "unique_source_sha256": len(source_hashes), "duplicate_source_groups": len(duplicate_sources),
                           "source_rehashed_during_audit": False,
                           "verification_basis": "per-record processor SHA256 verification ledger",
                           "official_inventory": build.get("inventory_audit", {})}
    declared_complete = bool(build.get("source_download_complete"))
    if declared_complete != (verified == len(records)):
        error("build_report_counts", "source_download_complete disagrees with verified record ledger")
    if report["checksums"]["duplicate_source_groups"]:
        report["warnings"].append("Duplicate source hashes are counted; duplicate source records must remain isolated.")
    for positions in duplicate_sources:
        active = [position for position in positions if len(ready_groups[position])]
        if len(active) > 1:
            error("duplicate_ready_sources", "duplicate source recordings both contribute ready windows", len(active))

    refs = ready["ref_index"]
    bad_reference = (refs < -1) | (refs >= len(ready))
    if bad_reference.any():
        error("reference_bounds", "reference index is out of ready-index bounds", bad_reference.sum())
    targets = np.flatnonzero((refs >= 0) & ~bad_reference)
    references = refs[targets]
    target_records = ready["record"][targets]
    reference_records = ready["record"][references]
    train_source = np.asarray([record.get("split") == "train" and record.get("role") == "source"
                               for record in records])
    patient_names = np.asarray([record.get("canonical_subject_id", "") for record in records])
    hashes = np.asarray([record.get("source_sha256", "") for record in records])
    bad_source = ~train_source[reference_records]
    same_patient = patient_names[target_records] == patient_names[reference_records]
    same_hash = hashes[target_records] == hashes[reference_records]
    for check, mask in (("reference_train_source", bad_source), ("reference_same_patient", same_patient),
                        ("reference_same_source_hash", same_hash)):
        if mask.any():
            error(check, "reference violates independent source-only conditioning", mask.sum())
    same_class = ready["label"][targets] == ready["label"][references]
    preferred_available = np.zeros(len(targets), dtype=bool)
    for label in (0, 1):
        pool = np.unique(patient_names[ready["record"][(ready["label"] == label) & train_source[ready["record"]]]])
        mask = ready["label"][targets] == label
        if len(pool) > 1:
            preferred_available[mask] = True
        elif len(pool) == 1:
            preferred_available[mask] = patient_names[target_records[mask]] != pool[0]
    report["references"] = {"assigned": len(targets), "missing": int(np.count_nonzero(refs == -1)),
                            "same_class": int(same_class.sum()), "fallback_class": int((~same_class).sum()),
                            "cross_class_despite_available_same_class": int((~same_class & preferred_available).sum()),
                            "policy": "train/source, different canonical patient and source checksum"}

    if report["errors"]:
        report["normalizer"] = {"status": "not_checked", "reason": "structural errors"}
    else:
        report["normalizer"], fitted = _normalizer(root, records, ready, ready_groups, protocol, error)
        try:
            rate, samples = protocol["sample_rate"], protocol["window_samples"]
            upper_frequency = min(80.0, rate * 0.45)
            measurement = config or ModelConfig(
                channels=len(protocol["lead_names"]), samples=samples, sample_rate=rate,
                waveform_patch=min(32, samples), coefficient_patch=min(32, samples // 8),
                cwt_min_frequency=min(2.0, upper_frequency / 4), cwt_max_frequency=upper_frequency)
            if (measurement.channels != len(protocol["lead_names"])
                    or measurement.samples != protocol["window_samples"]
                    or not np.isclose(measurement.sample_rate, protocol["sample_rate"])):
                raise ValueError("diagnostic model config does not match shard measurement definition")
            if fitted is not None and numerical_windows:
                measurement.validate()
            report["numerical"] = _numerical(root, records, ready, measurement, fitted, numerical_windows, device)
            numerical = report["numerical"]
            if numerical["status"] == "complete" and (
                    not numerical["dwt_idwt"]["passed"] or not numerical["cwt"]["finite"]
                    or not numerical["cwt"]["invalid_bins_zero"]):
                error("numerical_transforms", "DWT reconstruction or CWT finiteness/masking failed")
        except (RuntimeError, ValueError, OSError) as exception:
            error("numerical_transforms", str(exception))
            report["numerical"] = {"status": "failed", "error": str(exception)}
    raw_directory = build.get("raw_directory") or protocol.get("raw_directory")
    report["storage"] = {"processed": _storage(root),
                         "raw": _storage(Path(raw_directory)) if raw_directory else {"available": False}}
    complete_download = verified == len(records) and bool(build.get("source_download_complete"))
    full_inventory = len(records) == OFFICIAL_CHBMIT_RECORDS and len(event_records) == OFFICIAL_CHBMIT_EVENTS
    fully_processed = statuses["pending_download"] == 0 and statuses["failed"] == 0
    report["completeness"] = {"official_expected_records": OFFICIAL_CHBMIT_RECORDS,
                              "official_expected_events": OFFICIAL_CHBMIT_EVENTS,
                              "inventory_records": len(records), "known_inventory_events": len(event_records),
                              "full_official_inventory": full_inventory, "source_download_complete": complete_download,
                              "source_fully_processed": fully_processed,
                              "normalizer_available": report["normalizer"]["status"] == "valid",
                              "complete": full_inventory and complete_download and fully_processed
                              and report["normalizer"]["status"] == "valid"
                              and report["numerical"]["status"] == "complete"}
    report["notes"] = ["Engineering acceptance is not clinical model validation.",
                       "Test/validation EEG values were not read or used to fit statistics or construct conditions.",
                       "All splits were audited using indices, annotations and shard headers only; numerical reads were train/source.",
                       "QC counts use combined raw|processed flags; stage-specific counts are unavailable unless separately persisted.",
                       "Normalizer ranges/counts/digest are checked; its full-record mean/std are not recomputed by this audit.",
                       "High amplitude alone is a numerical flag and does not establish pathology."]
    return _write_report(root, report)
