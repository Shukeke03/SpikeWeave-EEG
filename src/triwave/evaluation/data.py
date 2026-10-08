"""Explicit physical EEG inputs for development evaluation, without conditions.

NPZ imports are a interchange format, not evidence that a named baseline was
faithfully reproduced. Identity/provenance declarations remain auditable in
the generated reports. No object arrays or pickle loading are accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from triwave.data.manifest import canonical_patient

if TYPE_CHECKING:
    from .gates import GateLedger


def json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass
class EvaluationData:
    waveforms: np.ndarray
    labels: np.ndarray
    patient_ids: np.ndarray
    record_ids: np.ndarray
    sample_ids: np.ndarray
    split: str
    sample_rate: float
    role: str = "source"
    unit: str = "uV"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waveforms = np.asarray(self.waveforms, dtype=np.float32)
        if (self.waveforms.ndim != 3 or min(self.waveforms.shape) < 1
                or self.waveforms.shape[-1] < 8 or not np.isfinite(self.waveforms).all()):
            raise ValueError("evaluation waveforms must be finite nonempty [N,C,T]")
        labels = np.asarray(self.labels)
        if labels.shape != (len(self.waveforms),) or not np.isin(labels, (0, 1)).all():
            raise ValueError("lightweight evaluation requires known binary labels")
        self.labels = labels.astype(np.int64)
        for name in ("patient_ids", "record_ids", "sample_ids"):
            values = np.asarray(getattr(self, name))
            if (values.shape != self.labels.shape or values.dtype.kind not in "US"
                    or any(not str(item).strip() for item in values)):
                raise ValueError(f"{name} must contain one nonempty string per window")
            setattr(self, name, values.astype(str))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique within an evaluation input")
        self.patient_ids = np.asarray([canonical_patient(p) for p in self.patient_ids])
        if self.split not in {"train", "validation", "test", "generated"}:
            raise ValueError("unknown evaluation split")
        if self.role not in {"source", "query", "generated"}:
            raise ValueError("support/adaptation is not allowed in strict source-only evaluation")
        if self.unit != "uV" or not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("evaluation requires physical uV and a positive sample rate")
        leads = self.metadata.get("lead_names")
        if (not isinstance(leads, list) or len(leads) != self.waveforms.shape[1]
                or len(set(leads)) != len(leads) or any(not isinstance(x, str) for x in leads)):
            raise ValueError("explicit unique lead_names must match the waveform order")
        if self.metadata.get("evaluation_protocol", "source-only") != "source-only":
            raise ValueError("this evaluator implements strict source-only only")
        json_digest(self.metadata)  # reject unserializable or nonfinite provenance
        if self.split == "generated" and self.role != "generated":
            raise ValueError("generated arrays must declare generated role")
        if self.role == "generated" and self.split != "generated":
            raise ValueError("generated arrays cannot masquerade as a real data split")

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for values in (self.waveforms, self.labels):
            digest.update(str(values.shape).encode())
            digest.update(np.ascontiguousarray(values).tobytes())
        digest.update(json.dumps([self.patient_ids.tolist(), self.record_ids.tolist(),
                                  self.sample_ids.tolist()], separators=(",", ":")).encode())
        digest.update(json_digest({"split": self.split, "role": self.role,
                                   "unit": self.unit, "sample_rate": self.sample_rate,
                                   "metadata": self.metadata}).encode())
        return digest.hexdigest()

    def counts(self) -> dict:
        return {"windows": len(self.labels), "patients": len(set(self.patient_ids)),
                "records": len(set(self.record_ids)), "positive": int(self.labels.sum()),
                "negative": int((self.labels == 0).sum()),
                "window_seconds": self.waveforms.shape[-1] / self.sample_rate,
                "loaded_seconds": len(self.labels) * self.waveforms.shape[-1] / self.sample_rate}


def load_evaluation_data(path: str | Path, *, stage: str = "G1",
                         ledger: GateLedger | None = None) -> EvaluationData:
    """Inspect split metadata before loading waveforms; locked test is G3-only."""
    if stage not in {"G0", "G1", "G2", "G3"}:
        raise ValueError("unknown evaluation stage")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        split = metadata["split"]
        if split == "test":
            if stage != "G3" or ledger is None:
                raise ValueError("locked test can only be read in G3 with a passing gate ledger")
            ledger.require_g3()
        required = ("waveforms", "labels", "patient_ids", "record_ids", "sample_ids")
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    metadata["import_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return EvaluationData(**arrays, split=split, role=metadata["role"],
                          unit=metadata["unit"], sample_rate=metadata["sample_rate"],
                          metadata=metadata)


def evaluation_data_from_dataset(dataset, *, max_windows: int | None = None,
                                 stage: str = "G1", ledger: GateLedger | None = None,
                                 seed: int = 0) -> EvaluationData:
    """Extract targets only; full split identity is audited before subsetting.

Subsets are uniformly selected without labels, and their limitation is recorded.
This does not assert that a label-stratified sample has natural prevalence.
"""
    split = dataset.split
    if split == "test":
        if stage != "G3" or ledger is None:
            raise ValueError("locked test can only be read in G3 with a passing gate ledger")
        ledger.require_g3()
    if split not in {"train", "validation", "test"}:
        raise ValueError("dataset adapter requires one explicit split")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    positions = np.arange(len(dataset))
    if max_windows is not None and len(positions) > max_windows:
        positions = np.sort(np.random.default_rng(seed).choice(positions, max_windows, replace=False))
    if not len(positions):
        raise ValueError("evaluation split is empty")
    waveforms, labels, patients, records, samples = [], [], [], [], []
    # Sharded and DualBranch loaders expose the complete record/row identities.
    if not all(hasattr(dataset, name) for name in ("records", "window_index", "indices")):
        raise ValueError("adapter supports explicit v1/v2 sharded datasets; use NPZ for other inputs")
    indices = dataset.indices
    population = [dataset.records[int(i)] for i in
                  np.unique(dataset.window_index["record"][indices])]
    for index in positions:
        row = dataset.window_index[int(indices[int(index)])]
        record = dataset.records[int(row["record"])]
        if record["split"] != split or record["unit"] != "uV":
            raise ValueError("missing patient identity or mixed splits")
        # Never call __getitem__: it can tokenize target annotation or load
        # reference/teacher assets even when the evaluator ignores those fields.
        if hasattr(dataset, "assets"):
            waveform = dataset.assets.load(row)
        elif hasattr(dataset, "_mmap"):
            offset = int(row["offset"])
            waveform = np.array(dataset._mmap(int(row["record"]))[:, offset:
                                offset + dataset.config.samples], copy=True)
        else:
            raise ValueError("sharded dataset does not expose a physical target-only reader")
        patient = record["canonical_subject_id"]
        sample = f"{record['record_id']}:{int(row['offset'])}"
        waveforms.append(waveform)
        labels.append(int(row["label"]))
        patients.append(patient)
        records.append(record["record_id"])
        samples.append(sample)
    config = dataset.config
    metadata = {"lead_names": list(population[0]["lead_names"]), "evaluation_protocol": "source-only",
                "natural_distribution": True, "label_namespace": "chbmit.window_event",
                "label_scope": f"{config.samples/config.sample_rate:g}s_window_annotation_overlap",
                "class_prompts": {"0": "No seizure interval is annotated within this window.",
                                  "1": "This window overlaps an annotated seizure."},
                "selection": "all" if len(positions) == len(dataset) else "uniform_without_labels",
                "selection_seed": seed if len(positions) < len(dataset) else None,
                "population_windows": len(dataset),
                "full_identity_available": bool(population),
                "population_patient_ids": sorted({canonical_patient(r["canonical_subject_id"])
                                                   for r in population}),
                "population_record_ids": sorted({r["record_id"] for r in population}),
                "population_source_hashes": sorted({r["source_sha256"] for r in population})}
    normalizer = getattr(dataset, "normalizer", None)
    if normalizer is not None:
        metadata["scale_fit_digest"] = normalizer.fit_manifest_digest
        metadata["scale_protocol"] = normalizer.scale_protocol
    return EvaluationData(np.stack(waveforms), np.asarray(labels), np.asarray(patients),
                          np.asarray(records), np.asarray(samples), split, config.sample_rate,
                          role="source" if split == "train" else "query", metadata=metadata)
