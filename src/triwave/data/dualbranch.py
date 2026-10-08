"""Streaming v2 side tables over immutable physical EEG record shards.

The query truth and condition request are separate objects. In particular,
neither the v1 ``ref_index`` nor a hidden query label selects a reference.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from triwave.attributes import ATTRIBUTE_PROTOCOL, ATTRIBUTE_SCHEMA, waveform_attributes
from triwave.config import ModelConfig
from triwave.montage import MontageSubspace
from triwave.signal import CWT_MASK_PROTOCOL, CWT_NUMERIC_PROTOCOL, MorletCWT, WaveletTransform

from .build import INDEX_DTYPE
from .dataset import TrainNormalizer, tokenize_texts
from .manifest import canonical_patient
from .sharded import ShardedEEGDataset, fit_sharded_statistics

SCHEMA_VERSION = 2
TABLE_NAMES = ("records", "windows", "texts", "views", "attributes", "relations", "condition_specs")
MANIFEST_NAME = "dualbranch_manifest.json"
NORMALIZER_NAME = "train_scale_v2.json"
PURPOSES = {"training", "representation", "source_only", "partial_observation"}


def _config(value: dict) -> ModelConfig:
    content = dict(value)
    for field in ("structure_depths", "snn_chunk_sizes", "lead_names", "attribute_schema"):
        if field in content:
            content[field] = tuple(content[field])
    config = ModelConfig(**content)
    config.validate()
    return config


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("records.json", "protocol.json", "build_report.json", "split_lock.json",
                 "window_index.npy", "isolated_index.npy"):
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(_hash_file(path).encode())
    for record in json.loads((root / "records.json").read_text())["records"]:
        if record.get("waveform_path"):
            path = (root / record["waveform_path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("physical shard path escapes the source root")
            stat = path.stat()
            digest.update(f"{record['waveform_path']}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _sample_id(record: dict, offset: int) -> str:
    return f"{record['record_id']}:{offset}"


def _events(record: dict, offset: int, config: ModelConfig) -> list[dict]:
    start, stop = offset / config.sample_rate, (offset + config.samples) / config.sample_rate
    return [{"event_id": event["event_id"],
             "window_onset": max(start, event["onset"]) - start,
             "window_offset": min(stop, event["offset"]) - start,
             "left_truncated": event["onset"] < start,
             "right_truncated": event["offset"] > stop}
            for event in record.get("events", [])
            if max(start, event["onset"]) < min(stop, event["offset"])]


def _event_group(record: dict, offset: int, config: ModelConfig) -> str:
    events = _events(record, offset, config)
    return ("|".join(sorted(event["event_id"] for event in events)) if events else
            f"{record['record_id']}:background-block:{offset // (16 * config.samples)}")


def _text(label: int) -> str:
    return ("This window overlaps an annotated seizure." if label == 1 else
            "No seizure interval is annotated within this window." if label == 0 else "")


def _source_rows(rows: np.ndarray, records: list[dict]) -> np.ndarray:
    mask = np.asarray([r["split"] == "train" and r["role"] == "source" for r in records])
    return np.flatnonzero(mask[rows["record"]])


def _audit_identities(records: list[dict]) -> None:
    patients, hashes = defaultdict(set), defaultdict(set)
    for record in records:
        patient = canonical_patient(record.get("subject_id", record["canonical_subject_id"]))
        if patient != record["canonical_subject_id"]:
            raise ValueError("canonical patient identity disagrees with the original subject alias")
        patients[patient].add(record["split"])
        hashes[record["source_sha256"]].add(record["split"])
        if (record["split"] == "train") != (record["role"] == "source"):
            raise ValueError("source/query role disagrees with the source-only split")
    if any(len(splits) > 1 for splits in patients.values()):
        raise ValueError("patient identity crosses locked splits")
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("identical source content crosses locked splits")


def _physical_protocol(protocol: dict) -> dict:
    keys = ("unit", "sample_rate", "window_samples", "lead_names", "bandpass", "filter", "resample",
            "processor_sha256", "reader_sha256", "qc_sha256", "exclude_qc_bits")
    return {name: protocol[name] for name in keys if name in protocol}


def _scale_digest(rows: np.ndarray, records: list[dict], physical_protocol: dict) -> str:
    source = np.array(rows[_source_rows(rows, records)], copy=True)
    source["ref_index"] = -1
    source["label"] = 0  # Scalar identity depends on physical population, not class names.
    digest = hashlib.sha256(source.tobytes())
    digest.update(json.dumps(physical_protocol, sort_keys=True).encode())
    for i in np.unique(source["record"]):
        record = records[int(i)]
        selected = {k: record[k] for k in ("source_sha256", "waveform_shape", "sample_rate",
                    "window_samples", "unit", "lead_names", "lead_valid", "lead_provenance") if k in record}
        digest.update(json.dumps(selected, sort_keys=True).encode())
    return digest.hexdigest()


def _measurement(config: ModelConfig) -> dict:
    return {"channels": config.channels, "samples": config.samples, "sample_rate": config.sample_rate,
            "lead_names": list(config.lead_names)}


def _audit_binding(recipe: dict, index_digest: str, scale_digest: str) -> str:
    config = _config(recipe["config"])
    body = {"source_fingerprint": recipe["source_fingerprint"], "window_index_sha256": index_digest,
            "scale_digest": scale_digest, "measurement": _measurement(config),
            "threshold": config.montage_residual_tolerance, "schema_version": SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class _PhysicalAssets:
    def __init__(self, root: Path, records: list[dict], config: ModelConfig):
        self.root, self.records, self.config = root.resolve(), records, config
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def load(self, row: np.void) -> np.ndarray:
        i, offset = int(row["record"]), int(row["offset"])
        record = self.records[i]
        path = (self.root / record["waveform_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("physical shard path escapes the declared source root")
        if (record["unit"] != "uV" or not all(record["lead_valid"])
                or tuple(record["lead_names"]) != tuple(self.config.lead_names)
                and self.config.montage_mode != "off"
                or not np.isclose(record["sample_rate"], self.config.sample_rate)):
            raise ValueError("physical unit, lead order, mask or sample rate mismatch")
        if i not in self.cache:
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.float32 or list(data.shape) != record["waveform_shape"]:
                raise ValueError("physical shard shape/dtype mismatch")
            self.cache[i] = data
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        x = np.array(self.cache[i][:, offset:offset + self.config.samples], copy=True)
        if x.shape != (self.config.channels, self.config.samples) or not np.isfinite(x).all():
            raise ValueError("physical window is incomplete or nonfinite")
        return x

    def __getstate__(self):
        return {"root": self.root, "records": self.records, "config": self.config}

    def __setstate__(self, state):
        self.__init__(state["root"], state["records"], state["config"])


def _fit_scale(assets: _PhysicalAssets, rows: np.ndarray, physical_protocol: dict) -> TrainNormalizer:
    total_sq, count, ranges = 0.0, 0, []
    for i in _source_rows(rows, assets.records):
        row = rows[i]
        x = assets.load(row).astype(np.float64)
        total_sq += float(np.square(x).sum())
        count += x.size
        record = assets.records[int(row["record"])]
        start = int(row["offset"])
        ranges.append(f"{record['record_id']}:{start}-{start + assets.config.samples}")
    if not count or total_sq / count <= 1e-12:
        raise ValueError("source scale needs nonzero eligible train/source waveforms")
    return TrainNormalizer(mean=0, std=math.sqrt(total_sq / count), count=count,
                           fit_manifest_digest=_scale_digest(rows, assets.records, physical_protocol),
                           fit_sample_ids=tuple(ranges))


def _reuse_or_fit_scale(source: ShardedEEGDataset, assets: _PhysicalAssets,
                       rows: np.ndarray, physical_protocol: dict, promoted: int) -> tuple[TrainNormalizer, str]:
    if not promoted:
        for name in (NORMALIZER_NAME, "normalizer.json", "train_normalizer.json"):
            if not (source.root / name).is_file():
                continue
            saved = TrainNormalizer.load(source.root / name)
            if saved.scale_protocol != "source_rms_v2":
                continue
            # This checks the complete training population, not the loader split.
            try:
                ShardedEEGDataset(source.root, source.config, normalizer=saved)
            except ValueError:
                # The old digest also includes reference policy. Never blindly
                # trust a stale value; source-only refitting preserves isolation.
                continue
            saved.fit_manifest_digest = _scale_digest(rows, assets.records, physical_protocol)
            return saved, "reused_verified_v1_source_rms"
        saved = fit_sharded_statistics(source)
        saved.fit_manifest_digest = _scale_digest(rows, assets.records, physical_protocol)
        return saved, "fitted_source_training"
    return _fit_scale(assets, rows, physical_protocol), "fitted_source_training_including_unlabelled"


def _promoted_unlabelled(source: ShardedEEGDataset) -> np.ndarray:
    path = source.root / "isolated_index.npy"
    if not path.exists():
        return np.empty(0, dtype=INDEX_DTYPE)
    isolated = np.load(path, mmap_mode="r", allow_pickle=False)
    if isolated.dtype != INDEX_DTYPE:
        raise ValueError("unsupported isolated index dtype")
    eligible = np.asarray([
        r["split"] == "train" and r["role"] == "source"
        and r.get("processed", False) and r.get("waveform_available", True)
        and set(r.get("isolation_reasons", [])) == {"unknown_annotation"}
        and all(r.get("lead_valid", [])) for r in source.records])
    mask = (eligible[isolated["record"]] & (isolated["label"] == -1)
            & ((isolated["qc_flags"] & source.protocol["exclude_qc_bits"]) == 0))
    return np.array(isolated[mask], copy=True)


def _schemas() -> dict[str, pa.Schema]:
    string, integer, boolean = pa.string(), pa.int64(), pa.bool_()
    return {
        "records": pa.schema([("record_index", integer), ("record_id", string),
                              ("patient_id", string), ("split", string), ("role", string),
                              ("source_sha256", string), ("metadata_json", string)]),
        "windows": pa.schema([("global_index", integer), ("sample_id", string),
                              ("record_index", integer), ("offset", integer), ("label", pa.int16()),
                              ("split", string), ("role", string), ("qc_flags", pa.uint16()),
                              ("event_ids", pa.list_(string)), ("event_group", string),
                              ("events_json", string)]),
        "texts": pa.schema([("global_index", integer), ("text_id", string), ("content", string),
                            ("namespace", string), ("scope", string), ("semantic_group", string),
                            ("reliable", boolean), ("target_derived", boolean)]),
        "views": pa.schema([("global_index", integer), ("view_id", string), ("parent_id", string),
                            ("role", string), ("numeric_protocol", string), ("mask_protocol", string),
                            ("shape", pa.list_(pa.int32())), ("target_derived", boolean)]),
        "attributes": pa.schema([("global_index", integer), ("values", pa.list_(pa.float32(), 16)),
                                 ("valid", pa.list_(boolean, 16)), ("computed", boolean),
                                 ("role", string), ("algorithm", string),
                                 ("max_frequency_hz", pa.float32())]),
        "relations": pa.schema([("global_index", integer), ("left_id", string),
                                ("right_id", string), ("relation_type", string),
                                ("patient_id", string), ("parent_id", string),
                                ("event_ids", pa.list_(string)), ("semantic_group", string)]),
        "condition_specs": pa.schema([("request_id", string), ("source_index", integer),
                                      ("source_window_id", string), ("source_patient_id", string),
                                      ("requested_class", pa.int16()), ("protocol", string),
                                      ("attribute_origin", string), ("target_derived", boolean),
                                      ("inference_available", boolean)])}


def build_dualbranch_dataset(v1_root: str | Path, output: str | Path,
                            config: ModelConfig | None = None, *, batch_size: int = 128,
                            include_query_attributes: bool = False,
                            include_unlabelled_source: bool = False, seed: int = 0) -> Path:
    """Atomically publish bounded-batch side tables; never copy physical EEG.

    Default migration reads only source EEG numerically. Exporting query teacher
    attributes is a separate, explicit evaluation-data operation, never training.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = config or ModelConfig()
    source = ShardedEEGDataset(v1_root, config)
    _audit_identities(source.records)  # Before fitting or reading any source EEG.
    source_root, output = source.root.resolve(), Path(output).resolve()
    if output == source_root or output.is_relative_to(source_root / "records"):
        raise ValueError("v2 output cannot replace physical source assets")
    fingerprint = _source_fingerprint(source_root)
    recipe = {"schema_version": SCHEMA_VERSION, "config": config.to_dict(), "seed": seed,
              "include_query_attributes": include_query_attributes,
              "include_unlabelled_source": include_unlabelled_source,
              "attribute_protocol": ATTRIBUTE_PROTOCOL, "attribute_schema": list(ATTRIBUTE_SCHEMA),
              "cwt_numeric_protocol": CWT_NUMERIC_PROTOCOL, "cwt_mask_protocol": CWT_MASK_PROTOCOL,
              "source_fingerprint": fingerprint,
              "physical_protocol": _physical_protocol(source.protocol),
              "processor_sha256": _hash_file(Path(__file__)),
              "attributes_sha256": _hash_file(Path(__file__).parents[1] / "attributes.py")}
    recipe = json.loads(json.dumps(recipe))  # Canonicalize tuple/list round trips.
    if output.exists():
        manifest = json.loads((output / MANIFEST_NAME).read_text())
        if manifest.get("recipe") != recipe:
            raise ValueError("existing v2 recipe is stale; use a new versioned output directory")
        report = validate_dualbranch_dataset(output)
        if not report["valid"]:
            raise ValueError("existing v2 data failed integrity validation")
        return output
    rows = np.array(source.window_index, copy=True)
    extra = _promoted_unlabelled(source) if include_unlabelled_source else np.empty(0, dtype=INDEX_DTYPE)
    if len(extra):
        rows = np.concatenate([rows, extra])
    rows["ref_index"] = -1  # The v1 label-derived selection is never migrated.
    keys = np.rec.fromarrays([rows["record"], rows["offset"]])
    if len(np.unique(keys)) != len(rows):
        raise ValueError("duplicate physical windows would bias training and source scale")
    assets = _PhysicalAssets(source_root, source.records, config)
    scale, scale_origin = _reuse_or_fit_scale(source, assets, rows, recipe["physical_protocol"], len(extra))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {name: 0 for name in TABLE_NAMES}
    schemas = _schemas()
    montage = None
    if config.montage_mode != "off":
        endpoints = [name.split("-") for name in config.lead_names]
        names = {name: i for i, name in enumerate(sorted({n for pair in endpoints for n in pair}))}
        montage = MontageSubspace.from_endpoints(
            torch.tensor([[names[a], names[b]] for a, b in endpoints]),
            max_relative_residual=config.montage_residual_tolerance)
    audit = {"scope": "all eligible train/source windows", "windows": 0, "passed_windows": 0,
             "max_residual_rms_uV": 0.0, "max_relative_residual": 0.0,
             "threshold": config.montage_residual_tolerance, "enforcement": "audit_only"}
    try:
        for name, schema in schemas.items():
            writers[name] = pq.ParquetWriter(staging / f"{name}.parquet", schema, compression="zstd")
        records_batch = [{"record_index": i, "record_id": r["record_id"],
                          "patient_id": r["canonical_subject_id"], "split": r["split"], "role": r["role"],
                          "source_sha256": r["source_sha256"], "metadata_json": json.dumps(r, sort_keys=True)}
                         for i, r in enumerate(source.records)]
        for start in range(0, len(records_batch), batch_size):
            batch = records_batch[start:start + batch_size]
            writers["records"].write_table(pa.Table.from_pylist(batch, schema=schemas["records"]))
            counts["records"] += len(batch)
        for start in range(0, len(rows), batch_size):
            batches = {name: [] for name in TABLE_NAMES if name != "records"}
            block = rows[start:start + batch_size]
            compute = [j for j, row in enumerate(block)
                       if source.records[int(row["record"])]["role"] == "source" or include_query_attributes]
            values = np.zeros((len(block), 16), np.float32)
            valid = np.zeros((len(block), 16), bool)
            maxima = np.full(len(block), min(80.0, config.sample_rate / 2), np.float32)
            if compute:
                physical = torch.from_numpy(np.stack([assets.load(block[j]) for j in compute]))
                for j in compute:
                    r = source.records[int(block[j]["record"])]
                    bandwidth = source.protocol.get("bandpass")
                    maxima[j] = min(maxima[j], float(r.get("source_sample_rate", config.sample_rate)) / 2,
                                    float(bandwidth[1]) if bandwidth else maxima[j])
                attr, av = waveform_attributes(physical, sample_rate=config.sample_rate,
                                              max_frequency=torch.from_numpy(maxima[compute]))
                values[compute], valid[compute] = attr.numpy(), av.numpy()
                if montage is not None:
                    source_local = [k for k, j in enumerate(compute)
                                    if source.records[int(block[j]["record"])]["role"] == "source"]
                    if source_local:
                        result = montage.audit(physical[source_local])
                        audit["windows"] += len(source_local)
                        audit["passed_windows"] += sum(result["passed_per_sample"])
                        audit["max_residual_rms_uV"] = max(audit["max_residual_rms_uV"], max(result["residual_rms"]))
                        audit["max_relative_residual"] = max(audit["max_relative_residual"], max(result["relative_residual"]))
            for local, row in enumerate(block):
                i, r = start + local, source.records[int(row["record"])]
                sid, label = _sample_id(r, int(row["offset"])), int(row["label"])
                events = _events(r, int(row["offset"]), config)
                event_ids = [e["event_id"] for e in events]
                group = "annotated_seizure" if label == 1 else "no_annotated_seizure" if label == 0 else ""
                scope = f"{config.samples / config.sample_rate:g}s_window_annotation_overlap"
                batches["windows"].append({"global_index": i, "sample_id": sid,
                    "record_index": int(row["record"]), "offset": int(row["offset"]), "label": label,
                    "split": r["split"], "role": r["role"], "qc_flags": int(row["qc_flags"]),
                    "event_ids": event_ids, "event_group": _event_group(r, int(row["offset"]), config),
                    "events_json": json.dumps(events)})
                batches["texts"].append({"global_index": i, "text_id": f"text:{sid}", "content": _text(label),
                    "namespace": "chbmit.window_event", "scope": scope, "semantic_group": group,
                    "reliable": label >= 0, "target_derived": True})
                for role in ("target_teacher", "partial_observation", *( ("reference_condition",) if r["role"] == "source" else ())):
                    batches["views"].append({"global_index": i, "view_id": f"{role}:{sid}", "parent_id": sid,
                        "role": role, "numeric_protocol": CWT_NUMERIC_PROTOCOL, "mask_protocol": CWT_MASK_PROTOCOL,
                        "shape": [config.channels, config.cwt_frequencies, config.cwt_times],
                        "target_derived": role != "reference_condition"})
                batches["attributes"].append({"global_index": i, "values": values[local].tolist(),
                    "valid": valid[local].tolist(), "computed": local in compute,
                    "role": "teacher_attributes", "algorithm": ATTRIBUTE_PROTOCOL,
                    "max_frequency_hz": float(maxima[local])})
                for other, relation in ((f"target_teacher:{sid}", "instance_structure"),
                                        (f"text:{sid}", "annotation_text")):
                    batches["relations"].append({"global_index": i, "left_id": sid, "right_id": other,
                        "relation_type": relation, "patient_id": r["canonical_subject_id"],
                        "parent_id": r["source_sha256"], "event_ids": event_ids, "semantic_group": group})
                if r["split"] == "train" and r["role"] == "source":
                    batches["condition_specs"].append({"request_id": f"source-reference:{sid}",
                        "source_index": i, "source_window_id": sid, "source_patient_id": r["canonical_subject_id"],
                        "requested_class": label, "protocol": "source-only", "attribute_origin": "reference",
                        "target_derived": False, "inference_available": True})
            for name, batch in batches.items():
                if batch:
                    writers[name].write_table(pa.Table.from_pylist(batch, schema=schemas[name]))
                    counts[name] += len(batch)
        for writer in writers.values():
            writer.close()
        writers.clear()
        np.save(staging / "window_index_v2.npy", rows, allow_pickle=False)
        scale.save(staging / NORMALIZER_NAME)
        audit["passed"] = bool(montage is not None and audit["windows"]
                               and audit["passed_windows"] == audit["windows"])
        audit["rank"] = montage.rank if montage else None
        audit["schema_version"] = SCHEMA_VERSION
        audit["source_fingerprint"] = fingerprint
        audit["measurement"] = _measurement(config)
        audit["eligible_source_windows"] = len(_source_rows(rows, source.records))
        audit["hard_mode_approved"] = audit["passed"] and audit["windows"] == audit["eligible_source_windows"]
        audit["audit_binding_digest"] = _audit_binding(recipe, _hash_file(staging / "window_index_v2.npy"),
                                                      scale.fit_manifest_digest)
        _json(staging / "montage_audit.json", audit)
        files = [*(f"{name}.parquet" for name in TABLE_NAMES), "window_index_v2.npy",
                 NORMALIZER_NAME, "montage_audit.json"]
        manifest = {"schema_version": SCHEMA_VERSION, "complete": True, "source_root": str(source_root),
                    "recipe": recipe, "table_rows": counts, "source_scale_origin": scale_origin,
                    "unlabelled_source_windows": len(extra), "files": {name: _hash_file(staging / name) for name in files}}
        _json(staging / MANIFEST_NAME, manifest)
        report = validate_dualbranch_dataset(staging)
        if not report["valid"]:
            raise ValueError(f"v2 integrity audit failed: {report['errors']}")
        _json(staging / "dataset_audit_v2.json", report)
        os.replace(staging, output)
        return output
    finally:
        for writer in writers.values():
            writer.close()
        if staging.exists():
            shutil.rmtree(staging)


def validate_dualbranch_dataset(root: str | Path, config: ModelConfig | None = None,
                               numerical_windows: int = 0) -> dict:
    """Full metadata audit plus optional bounded train/source numerical checks."""
    root, errors = Path(root), []
    numerical = {"status": "skipped", "windows": 0, "query_reads": 0}
    try:
        if numerical_windows < 0:
            raise ValueError("numerical_windows must be nonnegative")
        manifest = json.loads((root / MANIFEST_NAME).read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get("complete"):
            raise ValueError("v2 manifest is not complete or has an unsupported version")
        source_root = Path(manifest["source_root"])
        if _source_fingerprint(source_root) != manifest["recipe"]["source_fingerprint"]:
            raise ValueError("physical source metadata/index fingerprint is stale")
        for name, expected in manifest["files"].items():
            path = root / name
            if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("v2 file path escapes dataset root")
            if _hash_file(path) != expected:
                raise ValueError(f"v2 file digest mismatch: {name}")
        for name in TABLE_NAMES:
            table = pq.ParquetFile(root / f"{name}.parquet")
            if table.metadata.num_rows != manifest["table_rows"][name] or table.schema_arrow != _schemas()[name]:
                raise ValueError(f"side table count/schema mismatch: {name}")
        rows = np.load(root / "window_index_v2.npy", mmap_mode="r", allow_pickle=False)
        records = json.loads((source_root / "records.json").read_text())["records"]
        if rows.dtype != INDEX_DTYPE or len(rows) != manifest["table_rows"]["windows"]:
            raise ValueError("v2 window index shape/dtype/count mismatch")
        if (np.any(rows["record"] < 0) or np.any(rows["record"] >= len(records))
                or np.any(rows["offset"] < 0) or np.any(rows["ref_index"] != -1)
                or np.any(~np.isin(rows["label"], (-1, 0, 1)))):
            raise ValueError("v2 indices or label-free reference policy are invalid")
        _audit_identities(records)
        lengths = np.asarray([r.get("waveform_shape", [0, 0])[-1] for r in records])
        scale = TrainNormalizer.load(root / NORMALIZER_NAME)
        saved_config = _config(manifest["recipe"]["config"])
        if config is not None and _measurement(config) != _measurement(saved_config):
            raise ValueError("audit config disagrees with the physical measurement definition")
        config = saved_config
        if (np.any(rows["offset"] % config.samples)
                or np.any(rows["offset"] + config.samples > lengths[rows["record"]])
                or np.any(rows["qc_flags"] & manifest["recipe"]["physical_protocol"].get("exclude_qc_bits", 0))):
            raise ValueError("v2 window bounds, alignment or quality are invalid")
        unknown = rows["label"] == -1
        legal_unknown = np.asarray([r["split"] == "train" and r["role"] == "source"
            and set(r.get("isolation_reasons", [])) == {"unknown_annotation"}
            and all(r.get("lead_valid", [])) for r in records])
        if np.any(unknown & ~legal_unknown[rows["record"]]):
            raise ValueError("unlabelled windows are outside the explicit source representation protocol")
        source_count = len(_source_rows(rows, records)) * config.channels * config.samples
        if (scale.scale_protocol != "source_rms_v2" or scale.mean != 0 or scale.count != source_count
                or scale.fit_manifest_digest != _scale_digest(rows, records, manifest["recipe"]["physical_protocol"])):
            raise ValueError("source scale protocol/count/digest is invalid")
        audit = json.loads((root / "montage_audit.json").read_text())
        binding = _audit_binding(manifest["recipe"], manifest["files"]["window_index_v2.npy"],
                                 scale.fit_manifest_digest)
        if (audit.get("audit_binding_digest") != binding or audit.get("schema_version") != SCHEMA_VERSION
                or audit.get("eligible_source_windows") != len(_source_rows(rows, records))):
            raise ValueError("montage audit source/config/index binding is stale")
        for batch in pq.ParquetFile(root / "condition_specs.parquet").iter_batches(batch_size=1024):
            for spec in batch.to_pylist():
                i = spec["source_index"]
                if i < 0 or i >= len(rows):
                    raise ValueError("reference bank index is outside inventory")
                record = records[int(rows[i]["record"])]
                if (record["split"] != "train" or record["role"] != "source"
                        or spec["target_derived"] or not spec["inference_available"]
                        or spec["source_patient_id"] != record["canonical_subject_id"]):
                    raise ValueError("reference bank contains a non-source or target-derived candidate")
        if numerical_windows:
            assets = _PhysicalAssets(source_root, records, config)
            eligible = _source_rows(rows, records)
            positions = np.linspace(0, len(eligible) - 1, min(numerical_windows, len(eligible)), dtype=int)
            wavelet = WaveletTransform(config.samples, config.wavelet_levels, config.wavelet)
            cwt = MorletCWT(config.sample_rate, config.cwt_frequencies, config.cwt_min_frequency,
                            config.cwt_max_frequency, config.cwt_times)
            attributes = _ParquetRows(root / "attributes.parquet")
            max_error = 0.0
            for i in eligible[positions]:
                physical = torch.from_numpy(assets.load(rows[i]))[None]
                x = scale.transform(physical)
                error = float((wavelet.decode(wavelet.encode(x)) - x).abs().max())
                max_error = max(max_error, error)
                power, valid = cwt(x)
                if not torch.isfinite(power).all() or (power[~valid] != 0).any():
                    raise ValueError("CWT finite/invalid-zero numerical audit failed")
                saved_attr = attributes[int(i)]
                av, mask = waveform_attributes(physical, sample_rate=config.sample_rate,
                                                max_frequency=saved_attr["max_frequency_hz"])
                if (not torch.equal(mask[0], torch.tensor(saved_attr["valid"]))
                        or not torch.allclose(av[0], torch.tensor(saved_attr["values"]), rtol=2e-5, atol=2e-5)):
                    raise ValueError("saved physical attributes disagree with the shared extractor")
            if max_error > 2e-5:
                raise ValueError("DWT/IDWT numerical audit failed")
            numerical = {"status": "complete", "windows": len(positions), "query_reads": 0,
                         "dwt_max_absolute_error": max_error, "attribute_protocol": ATTRIBUTE_PROTOCOL,
                         "cwt_mask_protocol": CWT_MASK_PROTOCOL}
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
        errors.append(str(error))
    return {"valid": not errors, "complete": not errors, "errors": errors,
            "numerical": numerical, "numerical_query_reads": 0, "clinical_efficacy_evaluated": False}


def approve_hard_montage(root: str | Path, config: ModelConfig) -> bool:
    """Require an intact, bound full-source audit before hard enforcement."""
    root = Path(root)
    report = validate_dualbranch_dataset(root, config)
    if not report["valid"]:
        raise ValueError(f"hard montage data integrity audit failed: {report['errors']}")
    audit = json.loads((root / "montage_audit.json").read_text())
    if (not audit.get("hard_mode_approved") or not audit["windows"]
            or audit["windows"] != audit["eligible_source_windows"]
            or audit["passed_windows"] != audit["windows"]
            or audit["max_relative_residual"] > config.montage_residual_tolerance):
        raise ValueError("hard montage requires a passed whole-source measurement residual audit")
    return True


class _ParquetRows:
    """Bounded row-group cache; never materialize all attribute vectors."""
    def __init__(self, path: Path):
        self.path = path
        self.file = pq.ParquetFile(path)
        self.ends = np.cumsum([self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)])
        self.cache: OrderedDict[int, list[dict]] = OrderedDict()

    def __getitem__(self, index: int) -> dict:
        if index < 0 or not len(self.ends) or index >= self.ends[-1]:
            raise IndexError(index)
        group = bisect.bisect_right(self.ends, index)
        if group not in self.cache:
            self.cache[group] = self.file.read_row_group(group).to_pylist()
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(group)
        return self.cache[group][index - (int(self.ends[group - 1]) if group else 0)]

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])


class DualBranchEEGDataset(Dataset):
    def __init__(self, root: str | Path, config: ModelConfig | None = None, *, split: str = "train",
                 purpose: str | None = None, request: dict | None = None, seed: int = 0,
                 observation_fraction: float = 0.2, verify_integrity: bool = True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / MANIFEST_NAME).read_text())
        self.config = config or _config(self.manifest["recipe"]["config"])
        self.purpose = purpose or ("training" if split == "train" else "source_only")
        if split not in {"train", "validation", "test"} or self.purpose not in PURPOSES:
            raise ValueError("unknown split or v2 data purpose")
        if self.purpose in {"training", "representation"} and split != "train":
            raise ValueError("training/representation may read source training patients only")
        if not 0 < observation_fraction < 1:
            raise ValueError("partial observation must reveal a strict fraction in (0,1)")
        if verify_integrity and not validate_dualbranch_dataset(self.root)["valid"]:
            raise ValueError("v2 integrity audit failed")
        saved = _config(self.manifest["recipe"]["config"])
        if (self.config.channels != saved.channels or self.config.samples != saved.samples
                or self.config.sample_rate != saved.sample_rate or self.config.lead_names != saved.lead_names
                or self.config.cwt_frequencies != saved.cwt_frequencies or self.config.cwt_times != saved.cwt_times
                or self.config.cwt_min_frequency != saved.cwt_min_frequency
                or self.config.cwt_max_frequency != saved.cwt_max_frequency):
            raise ValueError("v2 measurement/wavelet definition disagrees with model config")
        self.source_root = Path(self.manifest["source_root"])
        self.records = json.loads((self.source_root / "records.json").read_text())["records"]
        self.window_index = np.load(self.root / "window_index_v2.npy", mmap_mode="r", allow_pickle=False)
        splits = np.asarray([r["split"] for r in self.records])
        select = splits[self.window_index["record"]] == split
        if self.purpose != "representation":
            select &= self.window_index["label"] >= 0
        self.indices = np.flatnonzero(select)
        self.split, self.seed, self.request = split, int(seed), dict(request or {})
        self.observation_fraction = observation_fraction
        self.normalizer = TrainNormalizer.load(self.root / NORMALIZER_NAME)
        self.assets = _PhysicalAssets(self.source_root, self.records, self.config)
        self.attributes = _ParquetRows(self.root / "attributes.parquet")
        bank = pq.read_table(self.root / "condition_specs.parquet", columns=["source_index"])
        self.reference_indices = bank.column(0).to_numpy().astype(np.int64, copy=False)
        self.reference_pools = defaultdict(lambda: defaultdict(list))
        for i in self.reference_indices:
            record = self.records[int(self.window_index[i]["record"])]
            patient = record["canonical_subject_id"]
            self.reference_pools[None][patient].append(int(i))
            self.reference_pools[int(self.window_index[i]["label"])][patient].append(int(i))
        self.reference_pools = {key: {p: np.asarray(ids, dtype=np.int64) for p, ids in pools.items()}
                                for key, pools in self.reference_pools.items()}
        self.cwt = MorletCWT(self.config.sample_rate, self.config.cwt_frequencies,
                            self.config.cwt_min_frequency, self.config.cwt_max_frequency, self.config.cwt_times)
        if self.request.get("target_derived") or self.request.get("attribute_origin", {}).get("target_derived"):
            raise ValueError("external requests cannot use hidden target-derived attributes or structure")

    def __len__(self) -> int:
        return len(self.indices)

    def _reference(self, global_index: int, rng: np.random.Generator) -> int | None:
        requested = self.request.get("reference", self.purpose == "training")
        if not requested:
            return None
        target_record = self.records[int(self.window_index[global_index]["record"])]
        class_id = self.request.get("requested_class", self.request.get("class_id"))
        pools = self.reference_pools.get(int(class_id) if class_id is not None else None, {})
        explicit = self.request.get("reference_index")
        if explicit is not None:
            i = int(explicit)
            if i not in self.reference_indices or (class_id is not None and self.window_index[i]["label"] != int(class_id)):
                raise ValueError("explicit reference is not in the legal source request bank")
            record = self.records[int(self.window_index[i]["record"])]
            if (record["canonical_subject_id"] == target_record["canonical_subject_id"]
                    or record["source_sha256"] == target_record["source_sha256"]):
                raise ValueError("reference shares the target patient or source record")
            return i
        # Patient-balanced donor choice is O(number of patients), not O(windows).
        patients = sorted(p for p in pools if p != target_record["canonical_subject_id"])
        if not patients:
            return None
        for _ in range(len(patients)):
            patient = patients.pop(int(rng.integers(len(patients))))
            donor = int(rng.choice(pools[patient]))
            record = self.records[int(self.window_index[donor]["record"])]
            if record["source_sha256"] != target_record["source_sha256"]:
                return donor
        return None

    def __getitem__(self, index: int) -> dict:
        i = int(self.indices[index])
        row, rng = self.window_index[i], np.random.default_rng(np.random.SeedSequence([self.seed, i]))
        record = self.records[int(row["record"])]
        sid, label = _sample_id(record, int(row["offset"])), int(row["label"])
        target = torch.from_numpy(self.normalizer.transform(self.assets.load(row)))
        reference, reference_valid = torch.zeros_like(target), torch.zeros(self.config.channels, dtype=torch.bool)
        structure = torch.zeros(self.config.channels, self.config.cwt_frequencies, self.config.cwt_times)
        structure_valid, visible = torch.zeros_like(structure, dtype=torch.bool), torch.zeros_like(structure, dtype=torch.bool)
        structure_role, provenance, reference_audit, reference_origin = "none", {}, {}, ""
        donor = self._reference(i, rng)
        if donor is not None:
            donor_row = self.window_index[donor]
            dr = self.records[int(donor_row["record"])]
            reference = torch.from_numpy(self.normalizer.transform(self.assets.load(donor_row)))
            reference_valid[:] = True
            value, valid = self.cwt(reference[None])
            structure, structure_valid = value[0], valid[0]
            visible, structure_role = structure_valid.clone(), "reference_condition"
            provenance = {"origin_id": _sample_id(dr, int(donor_row["offset"])), "target_derived": False,
                          "inference_available": True, "protocol": "source-only"}
            reference_origin = provenance["origin_id"]
            reference_audit = {"reference_index": donor, "reference_patient": dr["canonical_subject_id"],
                               "reference_record": dr["record_id"], "reference_split": "train",
                               "request_class": self.request.get("requested_class", self.request.get("class_id")),
                               "selection_used_query_label": False, "same_patient": False, "context_overlap": False}
        if self.purpose == "partial_observation":
            value, valid = self.cwt(target[None])
            structure, structure_valid = value[0], valid[0]
            patch = 16
            if structure.shape[-2] % patch or structure.shape[-1] % patch:
                raise ValueError("partial observation requires a 16x16 divisible CWT grid")
            allowed = structure_valid.reshape(self.config.channels, self.config.cwt_frequencies // patch,
                patch, self.config.cwt_times // patch, patch).all((2, 4))
            locations = np.argwhere(allowed.numpy())
            reveal = max(1, round(len(locations) * self.observation_fraction)) if len(locations) > 1 else 0
            reveal = min(reveal, max(len(locations) - 1, 0))
            visible = torch.zeros_like(structure_valid)
            if reveal:
                for c, f, t in locations[rng.choice(len(locations), reveal, replace=False)]:
                    visible[c, f * patch:(f + 1) * patch, t * patch:(t + 1) * patch] = True
            structure = torch.where(visible, structure, 0)
            structure_role = "partial_observation"
            provenance = {"origin_id": sid, "parent_window_id": sid, "target_derived": True,
                          "inference_available": True, "protocol": "observation-conditioned",
                          "available_valid_count": int(structure_valid.sum()), "visible_count": int(visible.sum()),
                          "available_patches": len(locations), "visible_patches": reveal,
                          "observed_patch_fraction": reveal / len(locations) if len(locations) else 0,
                          "seed": self.seed}
        attr_row = self.attributes[i]
        if attr_row["global_index"] != i:
            raise ValueError("teacher attribute alignment is corrupt")
        teacher = torch.tensor(attr_row["values"], dtype=torch.float32)
        teacher_valid = torch.tensor(attr_row["valid"], dtype=torch.bool)
        attributes = torch.zeros(self.config.attribute_dim, dtype=torch.float32)
        av = torch.zeros(self.config.attribute_dim, dtype=torch.bool)
        origin = {"kind": "none", "target_derived": False, "inference_available": False}
        teacher_origin = {"kind": "source_target_teacher", "origin_id": sid, "target_derived": True,
                          "source_role": record["split"], "inference_available": False}
        if "attribute_values" in self.request:
            attributes = torch.as_tensor(self.request["attribute_values"], dtype=torch.float32)
            av = torch.as_tensor(self.request.get("attribute_valid", [True] * len(attributes)), dtype=torch.bool)
            if (attributes.shape != (self.config.attribute_dim,) or av.shape != attributes.shape
                    or not torch.isfinite(attributes[av]).all()):
                raise ValueError("explicit attributes have invalid values, dimension or mask")
            origin = dict(self.request.get("attribute_origin", {}))
            if origin.get("target_derived") or not origin.get("inference_available", False):
                raise ValueError("explicit attributes require a legal inference-available origin")
        content = _text(label) if self.purpose in {"training", "representation"} else ""
        if "text" in self.request:
            content = str(self.request["text"])
        elif self.request.get("requested_class", self.request.get("class_id")) is not None:
            content = _text(int(self.request.get("requested_class", self.request.get("class_id"))))
        ids, tm = tokenize_texts([content], self.config)
        events = _events(record, int(row["offset"]), self.config)
        semantic = "annotated_seizure" if label == 1 else "no_annotated_seizure" if label == 0 else ""
        return {"target_waveform": target, "target_valid": torch.ones(self.config.channels, dtype=torch.bool),
                "reference_waveform": reference, "reference_valid": reference_valid,
                "reference_provenance": {"origin_id": reference_origin,
                    "target_derived": False, "inference_available": donor is not None, "protocol": "source-only"},
                "lead_geometry": torch.tensor(record["lead_geometry"], dtype=torch.float32),
                "structure": structure, "structure_valid": structure_valid, "structure_visible": visible,
                "structure_role": structure_role, "structure_provenance": provenance,
                "cwt_mask_protocol": CWT_MASK_PROTOCOL, "input_ids": ids[0], "text_valid": tm[0],
                "teacher_attributes": teacher, "teacher_attribute_valid": teacher_valid,
                "teacher_attribute_origin": teacher_origin,
                "source_max_frequency": torch.tensor(attr_row["max_frequency_hz"], dtype=torch.float32),
                "attribute_values": attributes, "attribute_valid": av,
                "condition_attributes": attributes.clone(), "condition_attribute_valid": av.clone(),
                "attribute_origin": origin, "class_ids": torch.tensor(label, dtype=torch.long),
                "alignment_metadata": {"instance_id": sid, "parent_id": record["source_sha256"],
                    "patient_id": record["canonical_subject_id"], "event_ids": [e["event_id"] for e in events],
                    "semantic_group": semantic, "namespace": "chbmit.window_event",
                    "scope": f"{self.config.samples / self.config.sample_rate:g}s_window_annotation_overlap",
                    "semantic_reliable": label >= 0, "negative_eligible": label >= 0},
                "metadata": {"sample_id": sid, "patient_id": record["canonical_subject_id"],
                    "canonical_subject_id": record["canonical_subject_id"], "record_id": record["record_id"],
                    "parent_id": record["source_sha256"], "source_sha256": record["source_sha256"],
                    "unit": "normalized_uV", "target_unit": "uV", "split": record["split"], "role": record["role"],
                    "purpose": self.purpose, "start_seconds": int(row["offset"]) / self.config.sample_rate,
                    "end_seconds": (int(row["offset"]) + self.config.samples) / self.config.sample_rate,
                    "events": events, "qc_flags": int(row["qc_flags"]), "reference_audit": reference_audit,
                    "text_source": "source_annotation_template" if self.purpose in {"training", "representation"} else "explicit_request",
                    "normalizer_digest": self.normalizer.fit_manifest_digest}}


class PatientEventBalancedSampler(Sampler[int]):
    """Equal class, then patient, then event/background-block mass on source."""
    def __init__(self, dataset: DualBranchEEGDataset, num_samples: int | None = None, *, seed: int = 0):
        if dataset.split != "train" or dataset.purpose == "source_only":
            raise ValueError("balanced sampling is training-source only")
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples < 1:
            raise ValueError("positive sampler length is required")
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for local, global_index in enumerate(dataset.indices):
            row = dataset.window_index[global_index]
            r = dataset.records[int(row["record"])]
            if r["split"] != "train" or r["role"] != "source":
                raise ValueError("sampler includes non-source training data")
            # Unknown is a separate unsupervised stratum, never a negative class.
            groups[int(row["label"])][r["canonical_subject_id"]][_event_group(r, int(row["offset"]), dataset.config)].append(local)
        weights = np.zeros(len(dataset), np.float64)
        for patients in groups.values():
            for events in patients.values():
                for members in events.values():
                    weights[members] = 1 / (len(groups) * len(patients) * len(events) * len(members))
        self.weights = torch.from_numpy(weights)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator).tolist()
