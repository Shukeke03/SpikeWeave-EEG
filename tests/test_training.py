import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.training import Trainer, TrainingConfig


@pytest.mark.parametrize("objective", ["diffusion"])
def test_optimizer_checkpoint_and_conditional_sampling(tmp_path, objective):
    torch.set_num_threads(2)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(objective=objective))
    batch = make_synthetic_batch(trainer.config, 2, seed=11)
    before = trainer.model.denoiser.state_dict()
    snapshots = {key: value.clone() for key, value in before.items()}
    metrics = trainer.train_step(batch)
    assert metrics["gradient_norm"] > 0
    assert any(not torch.equal(value, snapshots[key]) for key, value in before.items())
    path = tmp_path / "check.pt"
    trainer.save(path)
    restored = Trainer.load(path, resume_optimizer=True)
    assert restored.step == 1
    generated = restored.sample(restored.conditions(batch), steps=2, seed=21, guidance=1.5)
    assert generated.shape == batch["target_waveform"].shape
    assert torch.isfinite(generated).all()
    again = restored.sample(restored.conditions(batch), steps=2, seed=21, guidance=1.5)
    torch.testing.assert_close(generated, again)


def test_condition_builder_is_invariant_to_query_target():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    batch = make_synthetic_batch(trainer.config, 2, seed=7)
    first = trainer.conditions(batch)
    batch["target_waveform"] = torch.randn_like(batch["target_waveform"]) * 100
    second = trainer.conditions(batch)
    torch.testing.assert_close(first.reference_waveform, second.reference_waveform)
    torch.testing.assert_close(first.structure, second.structure)


def test_missing_target_lead_is_rejected_before_generation_training():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    batch = make_synthetic_batch(trainer.config, 2, seed=7)
    batch["target_valid"] = torch.ones(2, trainer.config.channels, dtype=torch.bool)
    batch["target_valid"][0, 0] = False
    with pytest.raises(ValueError, match="complete target"):
        trainer.train_step(batch)


def test_representation_transfer_stage_freeze_and_accumulation_resume(tmp_path):
    cfg = ModelConfig.tiny()
    representation = Trainer(cfg, TrainingConfig(stage="align", warmup_steps=0))
    batch = make_synthetic_batch(cfg, 3, 7)
    metrics = representation.train_step(batch)
    assert metrics["alignment_valid_items"] == 3 and metrics["masked_patch"] > 0
    path = tmp_path / "repr.pt"
    representation.save(path)
    diffusion = Trainer(cfg, TrainingConfig(gradient_accumulation=2, warmup_steps=0))
    diffusion.initialize_representation(path)
    assert diffusion.representation_initialized
    assert all(not p.requires_grad for p in diffusion.model.eeg_encoder.parameters())
    frozen = {k: v.clone() for k, v in diffusion.model.eeg_encoder.state_dict().items()}
    assert diffusion.train_step(batch)["optimizer_updated"] == 0
    with pytest.raises(ValueError, match="accumulation boundaries"):
        diffusion.save(tmp_path / "mid.pt")
    assert diffusion.train_step(batch)["step"] == 1
    assert all(
        torch.equal(value, frozen[key])
        for key, value in diffusion.model.eeg_encoder.state_dict().items()
    )
    diffusion.save(tmp_path / "full.pt")
    restored = Trainer.load(tmp_path / "full.pt", resume_optimizer=True)
    assert restored.microstep == 2 and restored.representation_initialized
    restored.sample(restored.conditions(batch), steps=2, guidance=1.5)
    assert restored.last_sampler_stats["network_calls"] == 10


def test_legacy_checkpoint_and_legacy_normalizer_are_explicitly_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format_version": 1}, path)
    with pytest.raises(ValueError, match="v0.1 checkpoints"):
        Trainer.load(path)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig())
    trainer.normalization = {"std": 1.0, "mean": 0.1, "scale_protocol": "legacy_affine_v1"}
    with pytest.raises(ValueError, match="source_rms_v2"):
        trainer.train_step(make_synthetic_batch(trainer.config, 3, 2))


def test_stage_a_adapts_numeric_visual_stem_with_frozen_medical_backbone():
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(stage="pretrain", warmup_steps=0))
    before = trainer.model.condition_encoder.image_stem.weight.detach().clone()
    metrics = trainer.train_step(make_synthetic_batch(trainer.config, 3, 2))
    assert metrics["numeric_visual_reconstruction"] > 0 and metrics["masked_patch"] > 0
    assert metrics["alignment_valid_items"] == 0
    assert not torch.equal(before, trainer.model.condition_encoder.image_stem.weight)
