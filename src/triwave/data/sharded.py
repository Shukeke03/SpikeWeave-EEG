"""Memory-mapped clinical record shards, with target-independent source references."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from triwave.config import ModelConfig

from .build import INDEX_DTYPE
from .dataset import TrainNormalizer, tokenize_texts


class ShardedEEGDataset(Dataset):
    def __init__(self, root: str | Path, config: ModelConfig | None = None, *,
                 split: str | None = None, normalizer: TrainNormalizer | None = None,
                 mmap_cache_size: int = 4, allow_partial: bool = False):
        self.root = Path(root)
        if self.root.is_file():
            self.root = self.root.parent
        self.config = config or ModelConfig()
        self.normalizer, self.split = normalizer, split
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("invalid dataset split")
        if mmap_cache_size <= 0:
            raise ValueError("mmap_cache_size must be positive")
        self.mmap_cache_size = mmap_cache_size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.records = json.loads((self.root / "records.json").read_text())["records"]
        self.protocol = json.loads((self.root / "protocol.json").read_text())
        self.build_report = json.loads((self.root / "build_report.json").read_text())
        if not allow_partial and not self.build_report["source_download_complete"]:
            raise ValueError("source download is incomplete; allow_partial is preview-only")
        if not allow_partial and not self.build_report.get("source_fully_processed", False):
            raise ValueError("source processing is incomplete; allow_partial is preview-only")
        if self.protocol["evaluation_protocol"] != "source-only":
            raise ValueError("sharded loader supports source-only protocol only")
        if (self.config.channels != len(self.protocol["lead_names"])
                or self.config.samples != self.protocol["window_samples"]
                or not np.isclose(self.config.sample_rate, self.protocol["sample_rate"])):
            raise ValueError("shard measurement definition does not match model config")
        if self.config.montage_mode != "off" and tuple(self.protocol["lead_names"]) != tuple(self.config.lead_names):
            raise ValueError("configured montage lead order disagrees with shard measurement metadata")
        self.window_index = np.load(self.root / "window_index.npy", mmap_mode="r", allow_pickle=False)
        if self.window_index.dtype != INDEX_DTYPE:
            raise ValueError("unsupported window index dtype")
        if (np.any(self.window_index["record"] < 0)
                or np.any(self.window_index["record"] >= len(self.records))
                or np.any(self.window_index["offset"] < 0)
                or np.any(self.window_index["offset"] % self.config.samples)
                or np.any(~np.isin(self.window_index["label"], (0, 1)))
                or np.any(self.window_index["qc_flags"] & self.protocol["exclude_qc_bits"])
                or np.any(self.window_index["ref_index"] < -1)
                or np.any(self.window_index["ref_index"] >= len(self.window_index))):
            raise ValueError("invalid ready index bounds, labels or quality flags")
        splits = np.asarray([record["split"] for record in self.records])
        self.indices = (np.arange(len(self.window_index), dtype=np.int64) if split is None else
                        np.flatnonzero(splits[self.window_index["record"]] == split))
        if normalizer is not None:
            expected_count = len(_source_training_rows(self)) * self.config.channels * self.config.samples
            if normalizer.count != expected_count or normalizer.count <= 0 or normalizer.input_unit != "uV":
                raise ValueError("invalid fitted physical normalizer count/unit")
            if normalizer.fit_manifest_digest != sharded_normalizer_digest(self):
                raise ValueError("normalizer training index/source checksum digest is stale")

    def __len__(self) -> int:
        return len(self.indices)

    def _mmap(self, record_index: int) -> np.ndarray:
        if record_index not in self.cache:
            metadata = self.records[record_index]
            if metadata["status"] != "ready" or not all(metadata["lead_valid"]):
                raise ValueError("isolated recording is present in the ready index")
            self.cache[record_index] = np.load(self.root / metadata["waveform_path"],
                                               mmap_mode="r", allow_pickle=False)
            if (self.cache[record_index].dtype != np.float32 or metadata["unit"] != "uV"
                    or list(self.cache[record_index].shape) != metadata["waveform_shape"]
                    or not np.isclose(metadata["sample_rate"], self.config.sample_rate)):
                self.cache.pop(record_index)
                raise ValueError("shard dtype, shape or physical measurement metadata mismatch")
            if len(self.cache) > self.mmap_cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(record_index)
        return self.cache[record_index]

    def _waveform(self, row: np.void) -> np.ndarray:
        offset = int(row["offset"])
        data = self._mmap(int(row["record"]))
        waveform = np.array(data[:, offset:offset + self.config.samples], dtype=np.float32, copy=True)
        if waveform.shape != (self.config.channels, self.config.samples) or not np.isfinite(waveform).all():
            raise ValueError("invalid/nonfinite ready waveform")
        if self.normalizer is not None:
            waveform = self.normalizer.transform(waveform)
        return waveform

    def __getitem__(self, index: int) -> dict:
        global_index = int(self.indices[index])
        row = self.window_index[global_index]
        record = self.records[int(row["record"])]
        if row["label"] not in (0, 1) or row["qc_flags"] & self.protocol["exclude_qc_bits"]:
            raise ValueError("unknown or excluded QC sample is present in the ready index")
        target = self._waveform(row)
        reference = np.zeros_like(target)
        reference_valid = np.zeros(self.config.channels, dtype=bool)
        reference_audit = {}
        reference_index = int(row["ref_index"])
        if reference_index >= 0:
            if reference_index >= len(self.window_index):
                raise ValueError("reference index is outside ready inventory")
            reference_row = self.window_index[reference_index]
            reference_record = self.records[int(reference_row["record"])]
            if (reference_record["split"] != "train" or reference_record["role"] != "source"
                    or reference_record["canonical_subject_id"] == record["canonical_subject_id"]
                    or reference_record["source_sha256"] == record["source_sha256"]):
                raise ValueError("reference is not an independent training-source patient")
            if (reference_row["label"] not in (0, 1)
                    or reference_row["qc_flags"] & self.protocol["exclude_qc_bits"]):
                raise ValueError("reference is not an eligible ready sample")
            reference = self._waveform(reference_row)
            reference_valid[:] = True
            reference_audit = {"target_index": global_index, "reference_index": reference_index,
                               "reference_record": reference_record["record_id"],
                               "reference_patient": reference_record["canonical_subject_id"],
                               "reference_split": "train", "same_patient": False,
                               "context_overlap": False, "same_label": int(row["label"]) == int(reference_row["label"])}
        seizure = int(row["label"]) == 1
        text = ("This window overlaps an annotated seizure." if seizure else
                "No seizure interval is annotated within this window.")
        ids, masks = tokenize_texts([text], self.config)
        start = int(row["offset"]) / self.config.sample_rate
        end = start + self.config.samples / self.config.sample_rate
        events = [{"event_id": event["event_id"], "window_onset": max(start, event["onset"]) - start,
                   "window_offset": min(end, event["offset"]) - start,
                   "left_truncated": event["onset"] < start, "right_truncated": event["offset"] > end}
                  for event in record["events"] if max(start, event["onset"]) < min(end, event["offset"])]
        return {"target_waveform": torch.from_numpy(target),
                "target_valid": torch.ones(self.config.channels, dtype=torch.bool),
                "reference_waveform": torch.from_numpy(reference),
                "reference_valid": torch.from_numpy(reference_valid),
                "lead_geometry": torch.tensor(record["lead_geometry"], dtype=torch.float32),
                "input_ids": ids[0], "text_valid": masks[0],
                "attribute_values": torch.zeros(self.config.attribute_dim),
                "attribute_valid": torch.zeros(self.config.attribute_dim, dtype=torch.bool),
                "class_ids": torch.tensor(int(row["label"]), dtype=torch.long),
                "alignment_metadata": {
                    "instance_id": f"{record['record_id']}:{int(row['offset'])}",
                    "parent_id": record["source_sha256"],
                    "patient_id": record["canonical_subject_id"],
                    "event_ids": [event["event_id"] for event in events],
                    "semantic_group": "annotated_seizure" if seizure else "no_annotated_seizure",
                    "namespace": "chbmit.window_event", "scope": f"{self.config.samples/self.config.sample_rate:g}s_window_annotation_overlap",
                    "semantic_reliable": True, "negative_eligible": True,
                },
                "metadata": {"sample_id": f"{record['record_id']}:{int(row['offset'])}",
                             "unit": "normalized_uV" if self.normalizer else "uV", "target_unit": "uV",
                             "split": record["split"], "role": record["role"], "start_seconds": start,
                             "end_seconds": end, "events": events, "qc_flags": int(row["qc_flags"]),
                             "annotation_source": record["annotation_source"],
                             "label_namespace": "chbmit.window_event", "reference_audit": reference_audit,
                             "text_source": "official_annotation_template_v1"}}


def _source_training_rows(dataset: ShardedEEGDataset) -> np.ndarray:
    """Select the complete ready training population regardless of loader split."""
    train_record = np.asarray([record["split"] == "train" and record["role"] == "source"
                               for record in dataset.records])
    return dataset.window_index[train_record[dataset.window_index["record"]]]


def sharded_normalizer_digest(dataset: ShardedEEGDataset) -> str:
    """Bind normalization to ready train/source rows, processing, split and sources.

    The held-out loader uses the same complete training population as fitting.
    Record QC policy and reader/processor fingerprints enter through protocol.
    """
    rows = _source_training_rows(dataset)
    digest = hashlib.sha256(rows.tobytes())
    digest.update(json.dumps(dataset.protocol, sort_keys=True).encode())
    digest.update((dataset.root / "split_lock.json").read_bytes())
    for record_index in np.unique(rows["record"]):
        digest.update(dataset.records[int(record_index)]["source_sha256"].encode())
    return digest.hexdigest()


def fit_sharded_statistics(dataset_or_root: ShardedEEGDataset | str | Path,
                            config: ModelConfig | None = None) -> TrainNormalizer:
    """Fit one global affine uV scale on eligible source training windows only."""
    dataset = (dataset_or_root if isinstance(dataset_or_root, ShardedEEGDataset)
               else ShardedEEGDataset(dataset_or_root, config, split="train"))
    if dataset.normalizer is not None or dataset.split not in {None, "train"}:
        raise ValueError("statistics require unnormalized source training data")
    if not dataset.build_report["source_download_complete"]:
        raise ValueError("complete source download is required before final statistics fitting")
    if not dataset.build_report.get("source_fully_processed", False):
        raise ValueError("complete source processing is required before final statistics fitting")
    rows = _source_training_rows(dataset)
    if not len(rows):
        raise ValueError("no ready source training windows")
    total, total_sq, count = 0.0, 0.0, 0
    fitted_ranges = []
    for record_index in np.unique(rows["record"]):
        offsets = np.sort(rows["offset"][rows["record"] == record_index])
        if len(np.unique(offsets)) != len(offsets):
            raise ValueError("duplicated training windows would bias fitted statistics")
        boundaries = np.flatnonzero(np.diff(offsets) != dataset.config.samples) + 1
        waveform = dataset._mmap(int(record_index))
        for run in np.split(offsets, boundaries):
            start, stop = int(run[0]), int(run[-1]) + dataset.config.samples
            # Bound the float64 reduction buffer instead of copying a full record.
            for chunk_start in range(start, stop, dataset.config.samples * 256):
                chunk_stop = min(stop, chunk_start + dataset.config.samples * 256)
                block = np.asarray(waveform[:, chunk_start:chunk_stop], dtype=np.float64)
                if not np.isfinite(block).all():
                    raise ValueError("nonfinite waveform in ready training index")
                total += float(block.sum())
                total_sq += float(np.square(block).sum())
                count += block.size
            fitted_ranges.append(f"{dataset.records[int(record_index)]['record_id']}:{start}-{stop}")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    if variance <= 1e-12:
        raise ValueError("source training waveforms have zero variance")
    return TrainNormalizer(mean=0.0, std=float(np.sqrt(total_sq / count)), count=count,
                           fit_manifest_digest=sharded_normalizer_digest(dataset),
                           fit_sample_ids=tuple(fitted_ranges))
