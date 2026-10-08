"""E/C reuse preserves semantics while generator capacities stay independent."""

import copy
import hashlib
from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.dataset import TrainNormalizer
from triwave.data.synthetic import make_synthetic_batch
from triwave.training import REPRESENTATION_TRANSFER_CONFIG_EXCEPTIONS, Trainer, TrainingConfig


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def clinical_metadata(trainer, seed=17):
    """Controlled source identities for transfer checks, not clinical data."""
    fit_digest = hashlib.sha256(b"unit-test-source-scale").hexdigest()
    trainer.normalization = TrainNormalizer(std=3.0, count=4096,
                                            fit_manifest_digest=fit_digest).to_dict()
    trainer.run_provenance = {
        "generator_training_seed": seed,
        "source_data_identity": {
            "kind": "manifest-jsonl",
            "metadata_sha256": hashlib.sha256(b"unit-test-source-records").hexdigest(),
            "source_scale_fit_digest": fit_digest,
        },
    }


def snapshot_encoders(trainer):
    return {name: {key: value.clone() for key, value in getattr(trainer.model, name).state_dict().items()}
            for name in ("eeg_encoder", "condition_encoder")}


def assert_encoders_equal(trainer, expected):
    for name, state in expected.items():
        for key, value in getattr(trainer.model, name).state_dict().items():
            torch.testing.assert_close(value, state[key])


def saved_representation(tmp_path, *, stage="align", trained=True, clinical=False):
    cfg = ModelConfig.tiny()
    source = Trainer(cfg, TrainingConfig(stage=stage, warmup_steps=0))
    if clinical:
        clinical_metadata(source)
    if trained:
        source.train_step(make_synthetic_batch(cfg, 2, seed=7))
    path = tmp_path / f"{stage}.pt"
    source.save(path)
    return source, path


def test_shared_a_old_large_capacity_then_b_initializes_two_independent_generators(tmp_path):
    # The running Stage A recipe historically records L24×1024 but allocates
    # only E/C. Its checkpoint is reusable by shared B and either generator.
    cfg = replace(ModelConfig.tiny(), denoiser_dim=1024, denoiser_depth=24, denoiser_heads=16)
    source_a = Trainer(cfg, TrainingConfig(stage="pretrain", warmup_steps=0))
    clinical_metadata(source_a)
    source_a.train_step(make_synthetic_batch(cfg, 2, seed=7))
    with torch.no_grad():
        source_a.model.geometry_projection.weight.fill_(123)
    a_path = tmp_path / "shared-a.pt"
    source_a.save(a_path)
    assert source_a.model.denoiser is None

    b_config = replace(cfg, denoiser_dim=48, denoiser_depth=3, denoiser_heads=4,
                       coefficient_patch=4, activation_checkpointing=True,
                       biomedclip_local_path="different-machine/weights",
                       frontend_text_config_path="different-machine/text-config")
    shared_b = Trainer(b_config, TrainingConfig(stage="align", warmup_steps=0))
    shared_b.run_provenance = {"generator_training_seed": 23, "stage": "align"}
    geometry_before = shared_b.model.geometry_projection.weight.clone()
    shared_b.initialize_representation(a_path)
    assert_encoders_equal(shared_b, snapshot_encoders(source_a))
    torch.testing.assert_close(shared_b.model.geometry_projection.weight, geometry_before)
    assert shared_b.normalization == source_a.normalization
    assert shared_b.run_provenance["source_data_identity"] == source_a.run_provenance["source_data_identity"]
    assert shared_b.run_provenance["generator_training_seed"] == 23
    assert shared_b.step == shared_b.microstep == 0 and not shared_b.optimizer.state
    assert shared_b.model.denoiser is None and shared_b.ema_denoiser is None
    parent_a = shared_b.run_provenance["representation_initialization"]
    assert parent_a["source_stage"] == "pretrain" and parent_a["source_optimizer_step"] == 1
    assert parent_a["source_capacity"]["denoiser_depth"] == 24
    assert parent_a["source_seed"] == 17
    assert parent_a["checkpoint_sha256"] == hashlib.sha256(a_path.read_bytes()).hexdigest()

    shared_b.train_step(make_synthetic_batch(b_config, 2, seed=11))
    b_path = tmp_path / "shared-b.pt"
    shared_b.save(b_path)
    shared_weights = snapshot_encoders(shared_b)
    generators = []
    for version, dim, depth in (("M", 32, 2), ("L", 48, 3)):
        config = replace(cfg, denoiser_dim=dim, denoiser_depth=depth, denoiser_heads=4)
        target = Trainer(config, TrainingConfig(warmup_steps=0))
        target.run_provenance = {"generator_training_seed": 31 if version == "M" else 37,
                                 "capacity_version": version}
        own_generator = {key: value.clone() for key, value in target.model.denoiser.state_dict().items()}
        own_ema = {key: value.clone() for key, value in target.ema_denoiser.state_dict().items()}
        geometry_before = target.model.geometry_projection.weight.clone()
        target.initialize_representation(b_path)
        assert_encoders_equal(target, shared_weights)
        torch.testing.assert_close(target.model.geometry_projection.weight, geometry_before)
        for key, value in own_generator.items():
            torch.testing.assert_close(target.model.denoiser.state_dict()[key], value)
            torch.testing.assert_close(target.ema_denoiser.state_dict()[key], own_ema[key])
        assert target.step == target.microstep == 0 and not target.optimizer.state
        assert target.run_provenance["capacity_version"] == version
        assert target.run_provenance["generator_training_seed"] == (31 if version == "M" else 37)
        parent_b = target.run_provenance["representation_initialization"]
        assert parent_b["source_stage"] == "align" and parent_b["source_seed"] == 23
        assert parent_b["source_run_provenance"]["representation_initialization"] == parent_a
        target.train_step(make_synthetic_batch(config, 2, seed=13))
        own_path = tmp_path / f"generation-{version}.pt"
        target.save(own_path)
        restored = Trainer.load(own_path, resume_optimizer=True)
        assert restored.config.denoiser_dim == dim and restored.config.denoiser_depth == depth
        assert restored.step == 1 and restored.optimizer.state
        assert restored.run_provenance == target.run_provenance
        for key, value in target.model.denoiser.state_dict().items():
            torch.testing.assert_close(restored.model.denoiser.state_dict()[key], value)
            torch.testing.assert_close(restored.ema_denoiser.state_dict()[key], target.ema_denoiser.state_dict()[key])
        generators.append(target)
    assert generators[0].model.eeg_encoder.patch.weight.data_ptr() != generators[1].model.eeg_encoder.patch.weight.data_ptr()


@pytest.mark.parametrize("changed", [
    {"sample_rate": 160.0}, {"heads": 8}, {"snn_chunk_sizes": (8,)},
    {"cwt_min_frequency": 3.0}, {"cwt_max_frequency": 40.0},
    {"lead_names": tuple(reversed(ModelConfig.tiny().lead_names))},
    {"attribute_schema": tuple(reversed(ModelConfig.tiny().attribute_schema))},
])
def test_same_shape_semantic_changes_rejected_before_any_encoder_is_loaded(tmp_path, changed):
    source, path = saved_representation(tmp_path)
    target = Trainer(replace(source.config, **changed), TrainingConfig())
    before = snapshot_encoders(target)
    with pytest.raises(ValueError, match="semantic fields"):
        target.initialize_representation(path)
    assert_encoders_equal(target, before)
    assert not target.representation_initialized


@pytest.mark.parametrize("corruption", ["missing_key", "extra_key", "wrong_shape"])
def test_condition_encoder_state_is_checked_before_eeg_encoder_copy(tmp_path, corruption):
    source, path = saved_representation(tmp_path)
    checkpoint = torch.load(path, weights_only=True)
    key = "condition_encoder.image_stem.weight"
    if corruption == "missing_key":
        checkpoint["model"].pop(key)
    elif corruption == "extra_key":
        checkpoint["model"]["condition_encoder.unexpected"] = torch.ones(1)
    else:
        checkpoint["model"][key] = torch.ones(1)
    torch.save(checkpoint, path)
    target = Trainer(source.config, TrainingConfig())
    before = snapshot_encoders(target)
    with pytest.raises(ValueError, match="condition_encoder checkpoint"):
        target.initialize_representation(path)
    assert_encoders_equal(target, before)


@pytest.mark.parametrize("corruption", ["scale", "fit_digest", "metadata_digest"])
def test_source_scale_or_data_identity_mismatch_rejected_without_loading(tmp_path, corruption):
    source, path = saved_representation(tmp_path, clinical=True)
    target = Trainer(source.config, TrainingConfig())
    clinical_metadata(target, seed=31)
    if corruption == "scale":
        target.normalization["std"] = 5.0
    elif corruption == "fit_digest":
        target.normalization["fit_manifest_digest"] = hashlib.sha256(b"other-scale").hexdigest()
    else:
        target.run_provenance["source_data_identity"]["metadata_sha256"] = hashlib.sha256(b"other-data").hexdigest()
    before = snapshot_encoders(target)
    provenance = copy.deepcopy(target.run_provenance)
    normalization = copy.deepcopy(target.normalization)
    with pytest.raises(ValueError, match="normalization differs|identity differs"):
        target.initialize_representation(path)
    assert_encoders_equal(target, before)
    assert target.run_provenance == provenance and target.normalization == normalization


@pytest.mark.parametrize("missing", ["normalization", "source_data_identity"])
def test_clinical_destination_rejects_source_missing_clinical_metadata(tmp_path, missing):
    source, path = saved_representation(tmp_path, clinical=True)
    checkpoint = torch.load(path, weights_only=True)
    if missing == "normalization":
        checkpoint["normalization"] = None
    else:
        checkpoint["run_provenance"].pop(missing)
    torch.save(checkpoint, path)
    target = Trainer(source.config, TrainingConfig())
    clinical_metadata(target)
    with pytest.raises(ValueError, match="source RMS normalization|source data identity"):
        target.initialize_representation(path)


@pytest.mark.parametrize("stage,trained", [("pretrain", True), ("align", False)])
def test_clinical_diffusion_requires_trained_stage_b(tmp_path, stage, trained):
    source, path = saved_representation(tmp_path, stage=stage, trained=trained, clinical=True)
    target = Trainer(source.config, TrainingConfig())
    with pytest.raises(ValueError, match="already trained align"):
        target.initialize_representation(path)
    assert target.normalization is None and not target.representation_initialized


def test_joint_smoke_keeps_explicit_unaligned_engineering_transfer(tmp_path):
    source, path = saved_representation(tmp_path, stage="pretrain", clinical=True)
    target = Trainer(source.config, TrainingConfig(stage="joint_smoke"))
    target.initialize_representation(path)
    assert target.representation_initialized
    assert target.run_provenance["representation_initialization"]["source_stage"] == "pretrain"


def test_lineage_hash_identifies_loaded_inode_when_source_job_replaces_last_pt(tmp_path, monkeypatch):
    source, path = saved_representation(tmp_path)
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = torch.load(path, weights_only=True)
    changed["model"]["eeg_encoder.patch.weight"].fill_(123)
    replacement = tmp_path / "new-source.pt"
    torch.save(changed, replacement)
    original_load = torch.load

    def source_job_replaces_checkpoint(file, *args, **kwargs):
        replacement.replace(path)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", source_job_replaces_checkpoint)
    target = Trainer(source.config, TrainingConfig())
    target.initialize_representation(path)
    assert_encoders_equal(target, snapshot_encoders(source))
    assert target.run_provenance["representation_initialization"]["checkpoint_sha256"] == original_digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original_digest


def test_capacity_exception_set_does_not_exempt_semantic_fields():
    assert REPRESENTATION_TRANSFER_CONFIG_EXCEPTIONS == {
        "denoiser_dim", "denoiser_heads", "denoiser_depth", "coefficient_patch",
        "activation_checkpointing", "biomedclip_local_path", "frontend_text_config_path",
    }
