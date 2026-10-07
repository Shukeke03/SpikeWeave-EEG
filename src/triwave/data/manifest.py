"""Versioned sample manifests and auditable patient/support boundaries."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPLITS = {"train", "validation", "test"}
ROLES = {"source", "support", "query"}


def canonical_patient(subject_id: str) -> str:
    """CHB cases 01 and 21 are the same patient, recorded at different times."""
    subject_id = subject_id.lower().strip()
    return "chb01" if subject_id in {"chb01", "chb21"} else subject_id


@dataclass
class SampleRecord:
    sample_id: str
    dataset: str
    subject_id: str
    record_id: str
    waveform_path: str
    start_seconds: float
    end_seconds: float
    sample_rate: float
    lead_names: list[str]
    split: str
    role: str = "source"
    schema_version: str = "1.0"
    dataset_version: str = "1.0.0"
    unit: str = "uV"
    canonical_subject_id: str = ""
    event_ids: list[str] = field(default_factory=list)
    lead_valid: list[bool] = field(default_factory=list)
    preprocessing_context: list[float] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    evaluation_protocol: str = "source-only"

    def __post_init__(self) -> None:
        canonical = canonical_patient(self.subject_id)
        if self.canonical_subject_id and self.canonical_subject_id != canonical:
            raise ValueError("canonical_subject_id disagrees with known patient identity")
        self.canonical_subject_id = canonical
        if self.split not in SPLITS or self.role not in ROLES:
            raise ValueError("invalid split or source/support/query role")
        if self.end_seconds <= self.start_seconds or self.start_seconds < 0:
            raise ValueError("sample interval must have positive duration")
        if self.sample_rate <= 0 or self.unit != "uV":
            raise ValueError("physical manifest waveforms must use uV and a positive sample rate")
        if not self.sample_id or not self.record_id or not self.lead_names:
            raise ValueError("sample, record and lead identifiers are required")
        if not self.lead_valid:
            self.lead_valid = [True] * len(self.lead_names)
        if len(self.lead_valid) != len(self.lead_names):
            raise ValueError("lead_valid and lead_names must align")
        if not self.preprocessing_context:
            self.preprocessing_context = [self.start_seconds, self.end_seconds]
        if (len(self.preprocessing_context) != 2
                or self.preprocessing_context[0] > self.start_seconds
                or self.preprocessing_context[1] < self.end_seconds):
            raise ValueError("preprocessing context must contain the sample interval")
        if self.evaluation_protocol not in {"source-only", "few-shot-support",
                                            "observation-conditioned"}:
            raise ValueError("unknown evaluation protocol")
        if self.role == "support" and self.evaluation_protocol == "source-only":
            raise ValueError("target support is not available in source-only evaluation")
        for key in ("structure", "reference"):
            condition = self.conditions.get(key)
            if condition and (condition.get("target_derived", False)
                              or condition.get("origin_id") == self.sample_id):
                raise ValueError(f"target-derived {key} is not an external condition")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleRecord:
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_manifest(records: list[SampleRecord]) -> None:
    ids: set[str] = set()
    paths: set[str] = set()
    patients: dict[str, str] = {}
    assets: dict[str, str] = {}
    for record in records:
        if record.sample_id in ids:
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        ids.add(record.sample_id)
        if record.waveform_path in paths:
            raise ValueError(f"duplicate waveform asset path: {record.waveform_path}")
        paths.add(record.waveform_path)
        prior = patients.setdefault(record.canonical_subject_id, record.split)
        if prior != record.split:
            raise ValueError(f"patient overlaps splits: {record.canonical_subject_id}")
        digest = record.provenance.get("source_sha256")
        if digest:
            prior_split = assets.setdefault(digest, record.split)
            if prior_split != record.split:
                raise ValueError("duplicate source recording overlaps splits")


def write_manifest(records: list[SampleRecord], path: str | Path) -> Path:
    validate_manifest(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: str | Path) -> list[SampleRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        records = [SampleRecord.from_dict(json.loads(line)) for line in handle if line.strip()]
    validate_manifest(records)
    return records


def lock_patient_splits(split_map: dict[str, str], path: str | Path) -> dict[str, str]:
    """Persist a lock before EEG loading; later calls cannot silently change it."""
    canonical: dict[str, str] = {}
    for subject, split in sorted(split_map.items()):
        if split not in SPLITS:
            raise ValueError(f"invalid split: {split}")
        patient = canonical_patient(subject)
        previous = canonical.setdefault(patient, split)
        if previous != split:
            raise ValueError(f"same patient assigned to different splits: {subject}")
    path = Path(path)
    payload = {"schema_version": "1.0", "patients": canonical,
               "identity_rules": {"chb21": "chb01"}}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError("patient split lock differs from requested split map")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return canonical


def manifest_digest(records: list[SampleRecord]) -> str:
    content = json.dumps([r.to_dict() for r in records], sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def audit_reference(target: SampleRecord, reference: SampleRecord,
                    protocol: str | None = None) -> dict[str, Any]:
    """Check both the visible window and every preprocessing dependency."""
    protocol = protocol or target.evaluation_protocol
    same_person = target.canonical_subject_id == reference.canonical_subject_id
    if reference.sample_id == target.sample_id:
        raise ValueError("target cannot be its own reference")
    if reference.role == "query":
        raise ValueError("query samples cannot be reference/support conditions")
    if protocol == "source-only":
        if reference.split != "train" or reference.role != "source":
            raise ValueError("source-only references must be source training samples")
        if target.split != "train" and same_person:
            raise ValueError("source-only cannot use held-out patient EEG as a reference")
    elif protocol == "few-shot-support":
        if not (reference.split == "train" and reference.role == "source"):
            if (reference.role != "support"
                    or reference.evaluation_protocol != "few-shot-support"):
                raise ValueError("target-domain reference must be explicitly budgeted support")
    elif protocol != "observation-conditioned":
        raise ValueError("unknown reference protocol")
    if same_person and set(target.event_ids).intersection(reference.event_ids):
        raise ValueError("reference and target share an event")
    target_hash = target.provenance.get("source_sha256")
    reference_hash = reference.provenance.get("source_sha256")
    same_record = (target.record_id == reference.record_id
                   or bool(target_hash and target_hash == reference_hash))
    if same_record:
        a, b = target.preprocessing_context
        c, d = reference.preprocessing_context
        if max(a, c) < min(b, d):
            raise ValueError("reference preprocessing context overlaps target/query context")
    return {"target_id": target.sample_id, "reference_id": reference.sample_id,
            "protocol": protocol, "same_patient": same_person,
            "same_record": same_record, "context_overlap": False,
            "reference_role": reference.role, "reference_split": reference.split}


def select_reference(target: SampleRecord, candidates: list[SampleRecord],
                     protocol: str | None = None) -> tuple[SampleRecord, dict[str, Any]]:
    """Deterministic selection; excluded candidates never become implicit fallbacks."""
    for candidate in sorted(candidates, key=lambda r: r.sample_id):
        try:
            return candidate, audit_reference(target, candidate, protocol)
        except ValueError:
            continue
    raise ValueError("no permitted, context-isolated reference sample")


def attach_source_references(manifest_path: str | Path, output_path: str | Path | None = None,
                             seed: int = 0) -> Path:
    """Attach audited source references, preferring matching class namespace and ID.

    Selection depends only on the seed and sample ID, with rotated candidate lists
    to avoid a universal first-record reference. No usable reference remains absent.
    The output must share the input directory because assets use relative paths.
    Existing references are audited and retained rather than silently replaced.
    This function supports the strict source-only protocol only.
    """
    path = Path(manifest_path)
    output = Path(output_path) if output_path is not None else path.with_name(
        f"{path.stem}-reference.jsonl")
    if output.parent.resolve() != path.parent.resolve():
        raise ValueError("reference manifest must remain in the same asset directory")
    records = read_manifest(path)
    if any(record.evaluation_protocol != "source-only" for record in records):
        raise ValueError("attach_source_references supports source-only protocol only")
    if any(record.role not in {"source", "query"} for record in records):
        raise ValueError("source-only reference attachment accepts source/query records only")
    by_id = {record.sample_id: record for record in records}
    candidates = sorted((record for record in records
                         if record.split == "train" and record.role == "source"),
                        key=lambda record: record.sample_id)

    def check(target: SampleRecord, reference: SampleRecord) -> dict[str, Any]:
        if target.record_id == reference.record_id:
            raise ValueError("source reference must come from a different recording")
        target_hash = target.provenance.get("source_sha256")
        if target_hash and target_hash == reference.provenance.get("source_sha256"):
            raise ValueError("source reference has duplicate recording content")
        return audit_reference(target, reference, "source-only")

    def class_key(record: SampleRecord) -> tuple[str, int] | None:
        condition = record.conditions.get("class", {})
        namespace, class_id = condition.get("namespace"), condition.get("id")
        if namespace is None or class_id is None or class_id == -1:
            return None
        return namespace, class_id

    class_pools: dict[tuple[str, int], list[SampleRecord]] = {}
    for candidate in candidates:
        key = class_key(candidate)
        if key is not None:
            class_pools.setdefault(key, []).append(candidate)
    result: list[SampleRecord] = []
    for record in records:
        target = SampleRecord.from_dict(deepcopy(record.to_dict()))
        existing = target.conditions.get("reference")
        if existing is not None:
            origin = existing.get("origin_id")
            if (origin not in by_id or existing.get("target_derived", False)
                    or not existing.get("inference_available", False)):
                raise ValueError("existing reference has an invalid/unavailable origin")
            check(target, by_id[origin])
            result.append(target)
            continue
        key = class_key(target)
        preferred = class_pools.get(key, []) if key is not None else []
        digest = hashlib.sha256(f"{seed}:{target.sample_id}".encode()).digest()
        rotation = int.from_bytes(digest[:8], "little")
        for group in (preferred, candidates):
            if not group:
                continue
            offset = rotation % len(group)
            for position in range(len(group)):
                candidate = group[(offset + position) % len(group)]
                try:
                    audit = check(target, candidate)
                except ValueError:
                    continue
                target.conditions["reference"] = {
                    "origin_id": candidate.sample_id, "role": "reference_waveform",
                    "source": "audited_training_source", "target_derived": False,
                    "inference_available": True, "selection_seed": seed,
                    "selection_version": "source-reference-v1",
                    "same_class_preferred": key is not None and class_key(candidate) == key,
                    "audit": audit,
                }
                break
            if "reference" in target.conditions:
                break
        result.append(target)
    return write_manifest(result, output)
