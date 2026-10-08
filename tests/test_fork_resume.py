"""Stage A recipe changes preserve optimizer progress with explicit lineage."""

import copy
import hashlib
from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.training import Trainer, TrainingConfig


def source_checkpoint(tmp_path, *, stage="pretrain"):
    torch.set_num_threads(2)
    config = replace(ModelConfig.tiny(), denoiser_depth=24, denoiser_dim=1024, denoiser_heads=16)
    training = TrainingConfig(stage=stage, gradient_accumulation=4, effective_batch_size=8,
                              warmup_steps=0)
    trainer = Trainer(config, training)
    for _ in range(4):
        trainer.train_step(make_synthetic_batch(config, 2, 13))
    trainer.normalization = {"scale_protocol": "source_rms_v2", "std": 3.0}
    trainer.run_provenance = {"generator_training_seed": 17, "source_data_identity": {"digest": "source"}}
    path = tmp_path / "source.pt"
    trainer.save(path)
    return trainer, path


@pytest.mark.parametrize("accumulation,effective_batch,rebased", [(2, 8, False), (3, 9, True)])
def test_stage_a_recipe_fork_preserves_weights_adam_rng_and_progress(
        tmp_path, accumulation, effective_batch, rebased):
    source, path = source_checkpoint(tmp_path)
    checkpoint = torch.load(path, weights_only=True)
    configuration = replace(source.config, denoiser_dim=512, denoiser_heads=8,
                            denoiser_depth=12, activation_checkpointing=True)
    training = replace(source.training, gradient_accumulation=accumulation,
                       effective_batch_size=effective_batch)
    resumed = Trainer.fork_representation_resume(path, configuration, training)
    assert resumed.model.denoiser is resumed.ema_denoiser is None
    assert resumed.step == source.step == 1
    assert resumed.microstep == (3 if rebased else 4)
    assert resumed.normalization == source.normalization
    assert resumed.run_provenance["source_data_identity"] == source.run_provenance["source_data_identity"]
    assert resumed.run_provenance["generator_training_seed"] == 17
    assert torch.equal(torch.get_rng_state(), checkpoint["torch_rng"])
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value)
    source_adam = source.optimizer.state_dict()
    resumed_adam = resumed.optimizer.state_dict()
    assert resumed_adam["param_groups"] == source_adam["param_groups"]
    for index, values in source_adam["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(resumed_adam["state"][index][key], value)
    lineage = resumed.run_provenance["recipe_fork"]
    assert lineage["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert lineage["source_microstep"] == 4 and lineage["microstep_counter_rebased"] == rebased
    assert lineage["batch_size"] == effective_batch // accumulation
    assert not lineage["trajectory_equivalent"] and lineage["optimizer_resumed"]
    assert checkpoint["model_config"]["denoiser_depth"] == 24
    for micro in range(accumulation):
        metrics = resumed.train_step(make_synthetic_batch(configuration, effective_batch // accumulation, 19))
        assert bool(metrics["optimizer_updated"]) == (micro == accumulation - 1)
    assert resumed.step == 2
    target = tmp_path / "new-recipe.pt"
    resumed.save(target)
    restored = Trainer.load(target, resume_optimizer=True)
    assert restored.step == 2 and restored.microstep == resumed.microstep
    assert restored.run_provenance == resumed.run_provenance
    assert restored.training == training and restored.config == configuration


@pytest.mark.parametrize("changed", [{"sample_rate": 160.0}, {"heads": 8}, {"coefficient_patch": 4}])
def test_fork_rejects_semantic_configuration_changes(tmp_path, changed):
    source, path = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="semantic model fields"):
        Trainer.fork_representation_resume(path, replace(source.config, **changed), source.training)


@pytest.mark.parametrize("changed", [{"learning_rate": 2e-4}, {"total_steps": 2000}, {"stage": "align"}])
def test_fork_cannot_change_learning_protocol_or_stage(tmp_path, changed):
    source, path = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="fixed training fields|training stage"):
        Trainer.fork_representation_resume(path, source.config, replace(source.training, **changed))


def test_fork_rejects_partial_accumulation_or_nonintegral_batch(tmp_path):
    source, path = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="integral microbatches"):
        Trainer.fork_representation_resume(path, source.config,
                                          replace(source.training, gradient_accumulation=3))
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["microstep"] -= 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="complete accumulation boundary"):
        Trainer.fork_representation_resume(path, source.config, source.training)


def test_fork_rejects_alignment_and_legacy_unused_generator(tmp_path):
    source, path = source_checkpoint(tmp_path, stage="align")
    with pytest.raises(ValueError, match="only accepts pretrain"):
        Trainer.fork_representation_resume(path, source.config, source.training)
    source, path = source_checkpoint(tmp_path)
    original = torch.load(path, weights_only=True)
    for key in ("ema", "model"):
        altered = copy.deepcopy(original)
        if key == "ema":
            altered[key] = {"unused": torch.zeros(1)}
        else:
            altered[key]["denoiser.unused"] = torch.zeros(1)
        torch.save(altered, path)
        with pytest.raises(ValueError, match="no denoiser or EMA"):
            Trainer.fork_representation_resume(path, source.config, source.training)
