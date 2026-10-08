from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.models.denoiser import MCWave
from triwave.models.snn import SpikingEEGEncoder
from triwave.training import Trainer, TrainingConfig


@pytest.mark.parametrize("stage", ["pretrain", "align"])
def test_representation_training_never_constructs_generator(monkeypatch, stage):
    def forbidden(*args, **kwargs):
        raise AssertionError("representation training must not allocate MCWave")

    monkeypatch.setattr("triwave.models.model.MCWave", forbidden)
    trainer = Trainer(ModelConfig.tiny(), TrainingConfig(stage=stage))
    assert trainer.model.denoiser is None and trainer.ema_denoiser is None
    metrics = trainer.train_step(make_synthetic_batch(trainer.config, 2, 11))
    assert metrics["optimizer_updated"] == 1 and metrics["loss"] > 0
    with pytest.raises(ValueError, match="representation checkpoints cannot sample"):
        trainer.sample(None)


def test_representation_resume_and_legacy_unused_generator_are_compatible(tmp_path):
    cfg = ModelConfig.tiny()
    trainer = Trainer(cfg, TrainingConfig(stage="pretrain", gradient_accumulation=2))
    batch = make_synthetic_batch(cfg, 2, 19)
    trainer.train_step(batch)
    trainer.train_step(batch)
    path = tmp_path / "pretrain.pt"
    trainer.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["ema"] is None
    assert not any(key.startswith("denoiser.") for key in checkpoint["model"])
    resumed = Trainer.load(path, resume_optimizer=True)
    assert resumed.step == 1 and resumed.microstep == 2
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name])
    resumed.train_step(batch)
    assert resumed.train_step(batch)["step"] == 2

    # Earlier format-2 Stage A/B files contained an unused random generator/EMA.
    unused = MCWave(cfg, trainer.model.band_lengths)
    checkpoint["model"].update({f"denoiser.{key}": value for key, value in unused.state_dict().items()})
    checkpoint["ema"] = unused.state_dict()
    checkpoint["training_config"]["stage"] = "diffusion"
    legacy = tmp_path / "old-pretrain.pt"
    torch.save(checkpoint, legacy)
    restored = Trainer.load(legacy, resume_optimizer=True)
    assert restored.active_stage == restored.training.stage == "pretrain"
    assert restored.model.denoiser is None and restored.ema_denoiser is None
    generation = Trainer(cfg, TrainingConfig())
    generation.initialize_representation(path)
    assert generation.representation_initialized
    assert generation.model.denoiser is not None and generation.ema_denoiser is not None


def test_snn_checkpoint_preserves_plif_gradients_and_drop_path_rng():
    cfg = replace(ModelConfig.tiny(), encoder_depth=3, snn_drop_path=0.4,
                  representation_dropout=0.2)
    plain = SpikingEEGEncoder(cfg).train()
    checked = SpikingEEGEncoder(replace(cfg, activation_checkpointing=True)).train()
    checked.load_state_dict(plain.state_dict())
    waveform = torch.randn(2, 4, 128)
    geometry = torch.randn(2, 4, 6)
    valid = torch.ones_like(waveform, dtype=torch.bool)
    valid[1, 3] = False
    patch_mask = plain.sample_patch_mask(valid)
    losses, random_states = [], []
    for model in (plain, checked):
        torch.manual_seed(42)
        output = model(waveform, geometry, valid, patch_mask)
        loss = output.reconstruction.square().mean() + output.tokens.square().mean()
        loss = loss + output.semantic[:, 0].sum() + output.structural[:, 1].sum()
        loss.backward()
        losses.append(loss.detach())
        random_states.append(torch.get_rng_state())
    torch.testing.assert_close(losses[0], losses[1])
    assert torch.equal(random_states[0], random_states[1])
    checked_parameters = dict(checked.named_parameters())
    for name, parameter in plain.named_parameters():
        other = checked_parameters[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)
    assert plain.blocks[0].attention.query_lif.decay_logit.grad is not None


def test_activation_checkpointing_requires_a_boolean():
    with pytest.raises(ValueError, match="activation_checkpointing must be boolean"):
        replace(ModelConfig.tiny(), activation_checkpointing="false").validate()
