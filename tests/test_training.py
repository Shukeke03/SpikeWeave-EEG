import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.training import Trainer, TrainingConfig


@pytest.mark.parametrize("objective", ["diffusion", "flow"])
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
