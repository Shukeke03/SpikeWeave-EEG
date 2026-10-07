"""Torch loading and train-only physical normalization, without target-view conditions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from triwave.config import ModelConfig

from .manifest import SampleRecord, audit_reference, manifest_digest, read_manifest


def tokenize_text(text: str, vocab_size: int, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic engineering tokenizer, not pretrained clinical-language semantics."""
    if vocab_size < 4 or max_length < 2:
        raise ValueError("offline tokenizer requires vocab >=4 and length >=2")
    tokens = [1]
    for word in text.lower().split()[: max_length - 2]:
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        tokens.append(3 + int.from_bytes(digest, "little") % (vocab_size - 3))
    tokens.append(2)
    ids = np.zeros(max_length, dtype=np.int64)
    ids[: len(tokens)] = tokens
    return ids, ids != 0


@lru_cache(maxsize=4)
def _pretrained_tokenizer(name: str, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)


def tokenize_texts(texts: list[str], config: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize external prompts consistently for training and generation."""
    if not texts:
        raise ValueError("at least one prompt is required")
    if config.frontend_backend == "biomedclip" or config.text_backend == "clinicalbert":
        tokenizer = _pretrained_tokenizer(
            (
                config.frontend_text_config_path
                or (
                    str(Path(config.biomedclip_local_path) / "text_config")
                    if config.biomedclip_local_path
                    else config.text_model_name
                )
            ),
            config.text_local_files_only,
        )
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.text_max_length,
            return_tensors="pt",
        )
        ids, valid = encoded["input_ids"], encoded["attention_mask"].bool()
    else:
        encoded = [
            tokenize_text(text, config.text_vocab_size, config.text_max_length) for text in texts
        ]
        ids = torch.from_numpy(np.stack([value[0] for value in encoded]))
        valid = torch.from_numpy(np.stack([value[1] for value in encoded]))
    for index, text in enumerate(texts):
        if not text.strip():
            ids[index] = 0
            valid[index] = False
    return ids, valid


@dataclass
class TrainNormalizer:
    """Source-only RMS scalar, preserving measurement closure without centering.

    Legacy affine files are read explicitly as legacy_affine_v1 for inverse
    compatibility; the new generator refuses their use until source refitting.
    """

    mean: float = 0.0
    std: float = 1.0
    count: int = 0
    fit_manifest_digest: str = ""
    fit_sample_ids: tuple[str, ...] = ()
    input_unit: str = "uV"
    scale_protocol: str = "source_rms_v2"

    def fit(
        self,
        records: list[SampleRecord] | EEGManifestDataset,
        waveform_loader: Callable[[SampleRecord], np.ndarray] | None = None,
    ) -> TrainNormalizer:
        if isinstance(records, EEGManifestDataset):
            dataset = records
            if dataset.normalizer is not None:
                raise ValueError("fit requires physical, not already normalized, training data")
            records = dataset.records

            def load(record: SampleRecord) -> np.ndarray:
                with np.load(
                    dataset.manifest_path.parent / record.waveform_path, allow_pickle=False
                ) as data:
                    return data["waveform"]

            waveform_loader = load
        if waveform_loader is None:
            raise ValueError("provide a waveform loader or an EEGManifestDataset")
        if not records:
            raise ValueError("training records are required")
        if any(r.split != "train" or r.role != "source" for r in records):
            raise ValueError("normalization can only fit source training samples")
        total, total_sq, count = 0.0, 0.0, 0
        for record in records:
            waveform = np.asarray(waveform_loader(record), dtype=np.float64)
            if waveform.ndim != 2 or waveform.shape[0] != len(record.lead_names):
                raise ValueError("waveform must have shape [lead, time]")
            values = waveform[np.asarray(record.lead_valid, dtype=bool)]
            if not np.isfinite(values).all():
                raise ValueError("normalization input contains nonfinite values")
            total += float(values.sum())
            total_sq += float(np.square(values).sum())
            count += values.size
        if count == 0:
            raise ValueError("no valid training lead samples")
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        if variance <= 1e-12:
            raise ValueError("training waveforms have zero variance")
        self.mean, self.std, self.count = 0.0, float(np.sqrt(total_sq / count)), count
        self.scale_protocol = "source_rms_v2"
        self.fit_manifest_digest = manifest_digest(records)
        self.fit_sample_ids = tuple(record.sample_id for record in records)
        return self

    def transform(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not self.count:
            raise ValueError("normalizer has not been fitted")
        if isinstance(waveform, torch.Tensor):
            return (waveform - self.mean) / self.std
        return ((np.asarray(waveform) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if not self.count:
            raise ValueError("normalizer has not been fitted")
        if isinstance(waveform, torch.Tensor):
            return waveform * self.std + self.mean
        return (np.asarray(waveform) * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if not self.count:
            raise ValueError("cannot save an unfitted normalizer")
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> TrainNormalizer:
        value = dict(value)
        value.setdefault("scale_protocol", "legacy_affine_v1")
        value["fit_sample_ids"] = tuple(value.get("fit_sample_ids", ()))
        result = cls(**value)
        if (
            result.count <= 0
            or result.std <= 0
            or result.input_unit != "uV"
            or not np.isfinite([result.mean, result.std]).all()
        ):
            raise ValueError("invalid saved train normalizer")
        if result.scale_protocol not in {"source_rms_v2", "legacy_affine_v1"}:
            raise ValueError("unknown normalizer protocol")
        if result.scale_protocol == "source_rms_v2" and result.mean != 0:
            raise ValueError("source_rms_v2 must not subtract a mean")
        return result

    @classmethod
    def load(cls, path: str | Path) -> TrainNormalizer:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)


def fit_train_statistics(manifest_path: str | Path) -> TrainNormalizer:
    path = Path(manifest_path)
    records = [
        record
        for record in read_manifest(path)
        if record.split == "train" and record.role == "source"
    ]

    def load(record: SampleRecord) -> np.ndarray:
        with np.load(path.parent / record.waveform_path, allow_pickle=False) as data:
            return data["waveform"]

    return TrainNormalizer().fit(records, load)


class EEGManifestDataset(Dataset):
    """Return targets separately from independently sourced generation inputs.

    A saved teacher/full-target CWT is deliberately never read. Reference origins
    must resolve to audited manifest rows; no implicit target-as-reference fallback.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        config: ModelConfig | None = None,
        *,
        split: str | None = None,
        role: str | None = None,
        normalizer: TrainNormalizer | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.config = config or ModelConfig()
        self.normalizer = normalizer
        all_records = read_manifest(self.manifest_path)
        self.by_id = {record.sample_id: record for record in all_records}
        self.records = [
            record
            for record in all_records
            if (split is None or record.split == split) and (role is None or record.role == role)
        ]
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("invalid dataset split")
        if role is not None and role not in {"source", "support", "query"}:
            raise ValueError("invalid dataset role")
        if normalizer is not None:
            if not normalizer.count:
                raise ValueError("dataset received an unfitted normalizer")
            for sample_id in normalizer.fit_sample_ids:
                record = self.by_id.get(sample_id)
                if record is not None and (record.split != "train" or record.role != "source"):
                    raise ValueError("normalizer was fitted on nontraining samples")

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.manifest_path.parent / record.waveform_path, allow_pickle=False) as data:
            waveform = np.asarray(data["waveform"], dtype=np.float32)
            geometry = np.asarray(data["lead_geometry"], dtype=np.float32)
            valid = np.asarray(data.get("lead_valid", record.lead_valid), dtype=bool)
        if self.config.montage_mode != "off" and tuple(record.lead_names) != tuple(
            self.config.lead_names
        ):
            raise ValueError(
                "configured montage lead order disagrees with source measurement metadata"
            )
        shape = (self.config.channels, self.config.samples)
        if waveform.shape != shape:
            raise ValueError(f"waveform {record.sample_id} has {waveform.shape}, expected {shape}")
        if geometry.shape != (self.config.channels, 6) or valid.shape != (self.config.channels,):
            raise ValueError("lead geometry/mask does not match waveform")
        if not np.isclose(record.sample_rate, self.config.sample_rate):
            raise ValueError("manifest sample rate does not match model measurement definition")
        if not np.isfinite(waveform).all() or not np.isfinite(geometry).all():
            raise ValueError("waveform/geometry contains nonfinite values")
        if self.normalizer is not None:
            waveform = self.normalizer.transform(waveform)
        waveform[~valid] = 0
        return waveform, geometry, valid

    def __getitem__(self, index: int) -> dict:
        target = self.records[index]
        waveform, geometry, target_valid = self._load(target)
        reference_waveform = np.zeros_like(waveform)
        reference_valid = np.zeros(self.config.channels, dtype=bool)
        reference_audit: dict = {}
        reference = target.conditions.get("reference")
        if reference is not None:
            if not reference.get("inference_available", False):
                raise ValueError("reference is not declared available at inference")
            if reference.get("target_derived", False):
                raise ValueError("target-derived reference is forbidden")
            origin_id = reference.get("origin_id")
            if origin_id not in self.by_id:
                raise ValueError("reference origin must resolve to a manifest sample")
            reference_record = self.by_id[origin_id]
            reference_audit = audit_reference(target, reference_record)
            target_path = (self.manifest_path.parent / target.waveform_path).resolve()
            reference_path = (self.manifest_path.parent / reference_record.waveform_path).resolve()
            if target_path == reference_path:
                raise ValueError("reference and target resolve to the same waveform asset")
            if reference_record.lead_names != target.lead_names:
                raise ValueError("reference montage does not match target measurement definition")
            reference_waveform, reference_geometry, reference_valid = self._load(reference_record)
            if not np.allclose(reference_geometry, geometry):
                raise ValueError("reference endpoint coordinates do not match target montage")
        if target.conditions.get("structure") is not None:
            raise ValueError(
                "external structure loading is not implemented; never substitute target CWT"
            )
        text = target.conditions.get("text", {})
        if text.get("scope") in {"record", "session", "report"} and text.get("content"):
            raise ValueError(
                "record/session reports require bag-level pairing, not per-window labels"
            )
        content = text.get("content", "") if text.get("inference_available", False) else ""
        ids_batch, text_mask_batch = tokenize_texts([content], self.config)
        ids, text_valid = ids_batch[0], text_mask_batch[0]
        attributes = target.conditions.get("attributes", {})
        if attributes.get("target_derived", False):
            raise ValueError(
                "target-derived attributes require an explicit observation protocol loader"
            )
        values = np.asarray(
            attributes.get("values", [0.0] * self.config.attribute_dim), dtype=np.float32
        )
        masks = np.asarray(attributes.get("valid", [False] * self.config.attribute_dim), dtype=bool)
        if values.shape != (self.config.attribute_dim,) or masks.shape != values.shape:
            raise ValueError("attribute values/masks must match configured dimension")
        if masks.any() and not attributes.get("inference_available", False):
            raise ValueError("attributes are not declared available at inference")
        return {
            "target_waveform": torch.from_numpy(waveform),
            "target_valid": torch.from_numpy(target_valid),
            "reference_waveform": torch.from_numpy(reference_waveform),
            "reference_valid": torch.from_numpy(reference_valid),
            "lead_geometry": torch.from_numpy(geometry),
            "input_ids": ids,
            "text_valid": text_valid,
            "attribute_values": torch.from_numpy(values),
            "attribute_valid": torch.from_numpy(masks),
            "class_ids": torch.tensor(
                target.conditions.get("class", {}).get("id", -1), dtype=torch.long
            ),
            "alignment_metadata": {
                "instance_id": target.sample_id,
                "parent_id": target.record_id,
                "patient_id": target.canonical_subject_id,
                "event_ids": target.event_ids,
                "semantic_group": str(target.conditions.get("class", {}).get("id", "unknown")),
                "namespace": target.conditions.get("class", {}).get("namespace"),
                "scope": target.conditions.get("class", {}).get("scope"),
                "semantic_reliable": target.conditions.get("class", {}).get("reliable", False),
                "negative_eligible": target.conditions.get("class", {}).get(
                    "negative_eligible", False
                ),
            },
            "metadata": {
                "sample_id": target.sample_id,
                "unit": "normalized_uV" if self.normalizer else "uV",
                "target_unit": target.unit,
                "split": target.split,
                "role": target.role,
                "condition_provenance": target.conditions,
                "reference_audit": reference_audit,
            },
        }


ManifestDataset = EEGManifestDataset
