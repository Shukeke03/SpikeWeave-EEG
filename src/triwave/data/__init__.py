"""Clinical data ingestion, explicit manifests and engineering-only smoke examples."""

from .build import build_chbmit
from .chbmit import STANDARD_BIPOLAR_LEADS, parse_summary, prepare_chbmit, read_edf_bipolar
from .dataset import EEGManifestDataset, TrainNormalizer, fit_train_statistics, tokenize_texts
from .manifest import (
    SampleRecord,
    attach_source_references,
    audit_reference,
    read_manifest,
    select_reference,
    write_manifest,
)
from .sharded import ShardedEEGDataset, fit_sharded_statistics
from .synthetic import make_synthetic_batch

__all__ = ["STANDARD_BIPOLAR_LEADS", "EEGManifestDataset", "ShardedEEGDataset", "TrainNormalizer", "SampleRecord",
           "attach_source_references", "audit_reference", "fit_train_statistics",
           "build_chbmit", "fit_sharded_statistics", "make_synthetic_batch", "parse_summary",
           "prepare_chbmit", "read_edf_bipolar", "read_manifest", "select_reference", "tokenize_texts",
           "write_manifest"]
