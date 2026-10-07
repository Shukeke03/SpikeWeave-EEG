import json

import pytest

from triwave.config import ModelConfig
from triwave.data import pipeline


def test_partial_pipeline_never_fits_statistics(tmp_path, monkeypatch):
    def build(raw, output, *args, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "build_report.json").write_text(json.dumps({
            "records": {"ready": 1, "isolated": 0, "failed": 0, "pending_download": 685},
            "ready_windows": 100, "isolated_windows": 0,
            "source_download_complete": False, "source_fully_processed": False,
        }))

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("partial download must never fit final statistics")

    monkeypatch.setattr(pipeline, "build_chbmit", build)
    monkeypatch.setattr(pipeline, "fit_sharded_statistics", unexpected_fit)
    state = pipeline.run_pipeline(tmp_path, ModelConfig.tiny(), max_passes=1)
    assert state["state"] == "waiting_for_download"
    assert not (tmp_path / "processed/v1/train_scale_v2.json").exists()


def test_processing_failure_requires_attention(tmp_path, monkeypatch):
    def build(raw, output, *args, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "build_report.json").write_text(json.dumps({
            "records": {"ready": 685, "isolated": 0, "failed": 1, "pending_download": 0},
            "ready_windows": 100, "isolated_windows": 0, "failed_records": ["broken.edf"],
            "source_download_complete": True, "source_fully_processed": False,
        }))

    monkeypatch.setattr(pipeline, "build_chbmit", build)
    state = pipeline.run_pipeline(tmp_path, ModelConfig.tiny(), max_passes=1)
    assert state["state"] == "needs_attention"


def test_pipeline_writes_failure_status(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("invalid official inventory")

    monkeypatch.setattr(pipeline, "build_chbmit", broken)
    with pytest.raises(ValueError, match="inventory"):
        pipeline.run_pipeline(tmp_path, ModelConfig.tiny(), max_passes=1)
    assert json.loads((tmp_path / "metadata/dataset-status.json").read_text())["state"] == "failed"


def test_completed_pipeline_requires_numerical_acceptance(tmp_path, monkeypatch):
    def build(raw, output, *args, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "build_report.json").write_text(json.dumps({
            "records": {"ready": 684, "isolated": 2, "failed": 0, "pending_download": 0},
            "ready_windows": 100, "isolated_windows": 0,
            "source_download_complete": True, "source_fully_processed": True,
        }))

    class Fitted:
        def save(self, path):
            path.write_text('{"count": 100}')

    monkeypatch.setattr(pipeline, "build_chbmit", build)
    monkeypatch.setattr(pipeline, "fit_sharded_statistics", lambda *args: Fitted())
    monkeypatch.setattr(pipeline, "validate_sharded_dataset", lambda *args, **kwargs: {
        "status": "complete", "completeness": {"complete": True}, "errors": [], "storage": {},
    })
    state = pipeline.run_pipeline(tmp_path, ModelConfig.tiny(), max_passes=1)
    assert state["state"] == "complete"
    assert (tmp_path / "processed/v1/train_scale_v2.json").is_file()
