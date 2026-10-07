"""Local CHB-MIT ingestion; never downloads data or fabricates clinical labels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .manifest import SampleRecord, canonical_patient, lock_patient_splits, write_manifest

STANDARD_BIPOLAR_LEADS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
ELECTRODE_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


@dataclass
class SeizureEvent:
    onset: float
    offset: float


@dataclass
class SummaryRecord:
    filename: str
    sample_rate: float
    channels: list[str]
    seizure_count: int | None = None
    seizures: list[SeizureEvent] = field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None


def parse_summary(path: str | Path) -> list[SummaryRecord]:
    return parse_summary_text(Path(path).read_text(encoding="utf-8-sig"))


def parse_summary_text(text: str) -> list[SummaryRecord]:
    sample_rate: float | None = None
    channels: list[str] = []
    records: list[SummaryRecord] = []
    current: SummaryRecord | None = None
    pending_onset: float | None = None
    channel_block = False
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", line, re.I)
        if match:
            sample_rate = float(match[1])
            continue
        if re.match(r"Channels(?:\s+changed)?\s+in EDF Files", line, re.I):
            channels = []
            channel_block = True
            continue
        match = re.match(r"Channel\s+\d+:\s*(.+)", line, re.I)
        if match:
            if not channel_block:
                channels = []
                channel_block = True
            channels.append(match[1].strip())
            continue
        match = re.match(r"File Name:\s*(.+)", line, re.I)
        if match:
            if pending_onset is not None:
                raise ValueError("seizure start has no end")
            if sample_rate is None or sample_rate <= 0:
                raise ValueError("summary is missing a positive sampling rate")
            current = SummaryRecord(match[1], sample_rate, channels.copy())
            records.append(current)
            channel_block = False
            continue
        if current is None:
            continue
        match = re.match(r"File (Start|End) Time:\s*(.+)", line, re.I)
        if match:
            setattr(current, match[1].lower() + "_time", match[2])
            continue
        match = re.match(r"Number of Seizures in File:\s*(\d+)", line, re.I)
        if match:
            if current.seizure_count is not None:
                raise ValueError("duplicate seizure count declaration")
            current.seizure_count = int(match[1])
            continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+(Start|End) Time:\s*([\d.]+)\s*seconds",
                         line, re.I)
        if match:
            seconds = float(match[2])
            if match[1].lower() == "start":
                if pending_onset is not None:
                    raise ValueError("seizure starts before previous event ends")
                pending_onset = seconds
            else:
                if pending_onset is None or seconds <= pending_onset:
                    raise ValueError("invalid seizure start/end pair")
                current.seizures.append(SeizureEvent(pending_onset, seconds))
                pending_onset = None
    if pending_onset is not None:
        raise ValueError("seizure start has no end")
    for record in records:
        if record.seizure_count is None or record.seizure_count != len(record.seizures):
            raise ValueError(f"seizure count mismatch in {record.filename}")
        if not record.channels:
            raise ValueError(f"missing channel declaration for {record.filename}")
        for first, second in zip(record.seizures, record.seizures[1:]):
            if second.onset < first.offset:
                raise ValueError("seizure events must be ordered and non-overlapping")
    if len({r.filename for r in records}) != len(records):
        raise ValueError("duplicate EDF filename in summary")
    return records


def canonical_lead(name: str) -> tuple[str, str] | None:
    name = name.upper().strip()
    name = re.sub(r"^EEG\s+", "", name)
    # MNE disambiguates duplicate EDF labels with a trailing integer suffix.
    name = re.sub(r"-\d+$", "", name)
    parts = [part.strip() for part in name.split("-")]
    if len(parts) != 2 or not all(parts):
        return None
    return tuple(ELECTRODE_ALIASES.get(part, part) for part in parts)


def standard_lead_geometry(leads: tuple[str, ...] = STANDARD_BIPOLAR_LEADS) -> np.ndarray:
    import mne

    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    lookup = {name.upper(): value for name, value in positions.items()}
    return np.asarray([np.concatenate([lookup[a], lookup[b]])
                       for a, b in (lead.split("-") for lead in leads)], dtype=np.float32)


@dataclass
class EDFRecording:
    waveform: np.ndarray
    sample_rate: float
    lead_names: tuple[str, ...]
    lead_valid: np.ndarray
    lead_geometry: np.ndarray
    provenance: dict


class UnsupportedMontageError(ValueError):
    """An EDF header provides no evidenced route to any supported bipolar lead."""

    def __init__(self, header_evidence: dict, message: str = "no supported bipolar lead is present"):
        super().__init__(message)
        self.header_evidence = header_evidence


def read_edf_bipolar(path: str | Path, *, allow_missing: bool = False,
                     common_reference: str | None = None) -> EDFRecording:
    """Read physical uV; select explicit bipolar pairs or declared common-reference pairs.

    Reversing an explicitly identified A-B lead is permitted with a recorded sign;
    reconstructing unknown monopolar potentials from bipolar data is never attempted.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    native_units = dict(getattr(raw, "_orig_units", {}))
    parsed = [canonical_lead(name) for name in raw.ch_names]
    header_evidence = {"source_channel_names": list(raw.ch_names), "native_units": native_units,
                       "source_sample_rate": float(raw.info["sfreq"]),
                       "source_samples": int(raw.n_times),
                       "has_explicit_channel_pairs": any(pair is not None for pair in parsed),
                       "declared_common_reference": common_reference}
    wanted = {tuple(lead.split("-")) for lead in STANDARD_BIPOLAR_LEADS}
    wanted |= {(b, a) for a, b in wanted.copy()}
    if common_reference is not None:
        ref = common_reference.upper()
        wanted |= {(electrode, ref) for lead in STANDARD_BIPOLAR_LEADS
                   for electrode in lead.split("-")}
    picks = [index for index, pair in enumerate(parsed) if pair in wanted]
    if not picks:
        raise UnsupportedMontageError(header_evidence)
    # Avoid a second full copy of dummy, ECG and unrelated EEG channels in 4h records.
    data = raw.get_data(picks=picks, units="uV")
    picked_index = {original: selected for selected, original in enumerate(picks)}

    def values(index: int) -> np.ndarray:
        return data[picked_index[index]]

    result = np.zeros((len(STANDARD_BIPOLAR_LEADS), raw.n_times), dtype=np.float32)
    valid = np.zeros(len(STANDARD_BIPOLAR_LEADS), dtype=bool)
    origins: list[dict] = []
    for out_index, lead in enumerate(STANDARD_BIPOLAR_LEADS):
        a, b = lead.split("-")
        forward = [i for i, pair in enumerate(parsed) if pair == (a, b)]
        reverse = [i for i, pair in enumerate(parsed) if pair == (b, a)]
        # Duplicate labels are only accepted if they actually represent the same waveform.
        choices = forward or reverse
        if choices:
            sign = 1 if forward else -1
            if any(not np.allclose(values(choices[0]), values(i), rtol=1e-5, atol=1e-3)
                   for i in choices[1:]):
                evidence = dict(header_evidence, isolation_reasons=["ambiguous_duplicate_bipolar_lead"],
                                ambiguous_lead=lead, ambiguous_channels=[raw.ch_names[i] for i in choices])
                raise UnsupportedMontageError(evidence, f"ambiguous duplicate EDF lead: {lead}")
            index = choices[0]
            result[out_index] = sign * values(index)
            valid[out_index] = True
            origins.append({"lead": lead, "method": "existing_bipolar",
                            "source_channels": [raw.ch_names[index]], "sign": sign})
        elif common_reference is not None:
            ref = common_reference.upper()
            pa = [i for i, pair in enumerate(parsed) if pair == (a, ref)]
            pb = [i for i, pair in enumerate(parsed) if pair == (b, ref)]
            if len(pa) == len(pb) == 1:
                result[out_index] = values(pa[0]) - values(pb[0])
                valid[out_index] = True
                origins.append({"lead": lead, "method": "declared_common_reference_difference",
                                "source_channels": [raw.ch_names[pa[0]], raw.ch_names[pb[0]]],
                                "common_reference": ref})
        if not valid[out_index]:
            origins.append({"lead": lead, "method": "missing", "source_channels": []})
    if not valid.all() and not allow_missing:
        missing = [lead for lead, present in zip(STANDARD_BIPOLAR_LEADS, valid) if not present]
        raise ValueError(f"missing standard bipolar leads: {missing}")
    if not valid.any():
        raise UnsupportedMontageError(header_evidence)
    for origin in origins:
        for name in origin["source_channels"]:
            native_unit = native_units.get(name, "").lower().replace("µ", "u").replace("μ", "u")
            if native_unit not in {"v", "mv", "uv", "nv"}:
                raise ValueError(f"EDF lead {name} has missing/unsupported physical units")
    provenance = {"source_channel_names": raw.ch_names, "lead_origins": origins,
                  "native_units": native_units, "output_unit": "uV",
                  "source_highpass": float(raw.info["highpass"]),
                  "source_lowpass": float(raw.info["lowpass"]),
                  "geometry_source": "MNE standard_1020 electrode endpoints in meters"}
    return EDFRecording(result, float(raw.info["sfreq"]), STANDARD_BIPOLAR_LEADS,
                        valid, standard_lead_geometry(), provenance)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_chbmit(root: str | Path, output: str | Path, split_map: dict[str, str], *,
                   window_seconds: float = 4.0, target_sfreq: float = 256.0,
                   bandpass: tuple[float, float] | None = (0.1, 80.0),
                   allow_missing: bool = False) -> Path:
    """Inventory/split first, then process continuous records and save physical windows.

    Processing context is the full record: support and query can never share a file
    under this pipeline. This entry point prepares source-only data, without support.
    """
    from scipy.signal import butter, resample_poly, sosfiltfilt

    root, output = Path(root), Path(output)
    if window_seconds <= 0 or target_sfreq <= 0:
        raise ValueError("window duration and sample rate must be positive")
    count = round(window_seconds * target_sfreq)
    if not math.isclose(count, window_seconds * target_sfreq):
        raise ValueError("window duration must contain an integer number of samples")
    splits = lock_patient_splits(split_map, output / "split_lock.json")
    inventory: list[tuple[Path, SummaryRecord, str, str]] = []
    seen_hashes: dict[str, str] = {}
    for summary_path in sorted(root.rglob("chb*-summary.txt")):
        subject = summary_path.name.split("-")[0].lower()
        patient = canonical_patient(subject)
        if patient not in splits:
            raise ValueError(f"patient missing from locked split map: {subject}")
        for summary in parse_summary(summary_path):
            source = summary_path.parent / summary.filename
            if not source.is_file():
                raise FileNotFoundError(source)
            digest = file_sha256(source)
            if digest in seen_hashes:
                raise ValueError(f"duplicate EDF content: {source} and {seen_hashes[digest]}")
            seen_hashes[digest] = str(source)
            inventory.append((source, summary, subject, digest))
    if not inventory:
        raise ValueError("no CHB-MIT summary/EDF records found")
    records: list[SampleRecord] = []
    for source, summary, subject, digest in inventory:
        physical = read_edf_bipolar(source, allow_missing=allow_missing)
        if not math.isclose(summary.sample_rate, physical.sample_rate):
            raise ValueError(f"summary/EDF sampling rate disagreement: {source}")
        duration = physical.waveform.shape[1] / physical.sample_rate
        if any(event.offset > duration for event in summary.seizures):
            raise ValueError("seizure annotation exceeds EDF duration")
        waveform = physical.waveform.astype(np.float64)
        if bandpass is not None:
            low, high = bandpass
            if not 0 < low < high < min(physical.sample_rate, target_sfreq) / 2:
                raise ValueError("bandpass must lie below source and target Nyquist")
            sos = butter(4, bandpass, fs=physical.sample_rate, btype="bandpass", output="sos")
            waveform = sosfiltfilt(sos, waveform, axis=-1)
        if not math.isclose(target_sfreq, physical.sample_rate):
            from fractions import Fraction

            ratio = Fraction(target_sfreq / physical.sample_rate).limit_denominator(10000)
            waveform = resample_poly(waveform, ratio.numerator, ratio.denominator, axis=-1)
        waveform = waveform.astype(np.float32)
        record_id = f"chbmit:{subject}:{source.name}"
        split = splits[canonical_patient(subject)]
        folder = output / "waveforms" / split / subject
        folder.mkdir(parents=True, exist_ok=True)
        for offset in range(0, waveform.shape[1] - count + 1, count):
            start, end = offset / target_sfreq, (offset + count) / target_sfreq
            sample_id = f"{record_id}:{offset}"
            sample_path = folder / f"{source.stem}_{offset:09d}.npz"
            np.savez_compressed(sample_path, waveform=waveform[:, offset:offset + count],
                                lead_geometry=physical.lead_geometry, lead_valid=physical.lead_valid)
            annotations = []
            for index, event in enumerate(summary.seizures):
                if max(start, event.onset) < min(end, event.offset):
                    annotations.append({"event_id": f"{record_id}:seizure:{index}",
                                        "namespace": "chbmit.annotated_seizure", "scope": "event",
                                        "source_onset": event.onset, "source_offset": event.offset,
                                        "window_onset": max(start, event.onset) - start,
                                        "window_offset": min(end, event.offset) - start,
                                        "left_truncated": event.onset < start,
                                        "right_truncated": event.offset > end})
            label = "annotated_seizure" if annotations else "no_annotated_seizure"
            text = ("This window overlaps an annotated seizure." if annotations else
                    "No seizure interval is annotated within this window.")
            provenance = {**physical.provenance, "source_file_id": str(source.relative_to(root)),
                          "source_sha256": digest, "summary_filename": source.name,
                          "summary_channels": summary.channels, "source_sample_rate": physical.sample_rate,
                          "access_terms": "https://physionet.org/content/chbmit/1.0.0/",
                          "processing": {"version": "chbmit-v1", "bandpass": bandpass,
                                         "filter": "scipy butter order4 sosfiltfilt whole record",
                                         "resample": "scipy resample_poly whole record",
                                         "target_sample_rate": target_sfreq}}
            conditions = {"text": {"content": text, "role": "annotation_template",
                                   "source": "official_event_annotation", "template_version": "1.0",
                                   "scope": "window", "origin_id": sample_id,
                                   "target_derived": False, "inference_available": True,
                                   "supported_label": label},
                          "class": {"id": int(bool(annotations)), "namespace": "chbmit.window_event",
                                    "label": label, "source": "official_event_annotation",
                                    "scope": f"{window_seconds:g}s_window_annotation_overlap", "reliable": True,
                                    "negative_eligible": True}}
            records.append(SampleRecord(
                sample_id, "CHB-MIT", subject, record_id,
                str(sample_path.relative_to(output)), start, end, target_sfreq,
                list(physical.lead_names), split, "source" if split == "train" else "query",
                event_ids=[annotation["event_id"] for annotation in annotations],
                lead_valid=physical.lead_valid.tolist(), preprocessing_context=[0.0, duration],
                provenance=provenance, annotations=annotations, conditions=conditions))
    manifest = write_manifest(records, output / "manifest.jsonl")
    (output / "inventory.json").write_text(json.dumps({
        "records": len(inventory), "windows": len(records), "unit": "uV",
        "split_counts": {split: sum(r.split == split for r in records) for split in sorted(set(splits.values()))},
        "notes": ["Non-seizure windows are not healthy controls.",
                  "Full-record preprocessing prohibits same-record support/query pairs."]}, indent=2) + "\n")
    return manifest


prepare = prepare_chbmit
