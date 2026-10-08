"""MCWave-M/L share representations but retain separate generator capacities."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.models.denoiser import MCWave
from triwave.training import Trainer, TrainingConfig

CAPACITY_FIELDS = {"denoiser_dim", "denoiser_heads", "denoiser_depth"}
CAPACITIES = {
    "m": {"denoiser_dim": 512, "denoiser_heads": 8, "denoiser_depth": 12},
    "l": {"denoiser_dim": 1024, "denoiser_heads": 16, "denoiser_depth": 24},
}


def test_parallel_configs_differ_only_in_generator_capacity():
    configs = [ModelConfig.from_yaml(f"configs/research-{version}.yaml") for version in ("m", "l")]
    left, right = (config.to_dict() for config in configs)
    assert {key for key in left if left[key] != right[key]} == CAPACITY_FIELDS
    assert left | CAPACITIES["l"] == right
    assert (configs[0].dim, configs[0].encoder_depth, configs[0].condition_dim,
            configs[0].fusion_depth, configs[0].frontend_backend) == (512, 8, 768, 6, "biomedclip")
    training = []
    for version in ("m", "l"):
        content = yaml.safe_load(Path(f"configs/research-{version}.yaml").read_text())
        training.append(content["training"])
    assert training[0] == training[1]
    assert training[0]["stage"] == "diffusion"


@pytest.mark.parametrize("version,expected", [("m", 105769728), ("l", 799622976)])
def test_full_generator_parameter_count_without_allocating_weights(version, expected):
    config = ModelConfig.from_yaml(f"configs/research-{version}.yaml")
    lengths = (128, 128, 256, 512)
    # Meta tensors verify every layer's shape without allocating 800M float weights.
    with torch.device("meta"):
        generator = MCWave(config, lengths)
    assert sum(parameter.numel() for parameter in generator.parameters()) == expected
    assert len(generator.blocks) == config.denoiser_depth
    assert generator.global_embedding.in_features == 768
    assert generator.global_embedding.out_features == config.denoiser_dim


def test_primary_defaults_and_shared_training_configs_select_m_metadata():
    default = ModelConfig().to_dict()
    assert {key: default[key] for key in CAPACITY_FIELDS} == CAPACITIES["m"]
    for name in ("research-5090", "legion-clinical", "research-pretrain",
                 "research-align", "legion-pretrain"):
        config = ModelConfig.from_yaml(f"configs/{name}.yaml").to_dict()
        assert {key: config[key] for key in CAPACITY_FIELDS} == CAPACITIES["m"]
    training = yaml.safe_load(Path("configs/legion-pretrain.yaml").read_text())["training"]
    assert training["stage"] == "pretrain"
    assert training["gradient_accumulation"] == 32
    assert training["effective_batch_size"] == 256
    assert training["effective_batch_size"] // training["gradient_accumulation"] == 8
    assert ModelConfig.from_yaml("configs/legion-pretrain.yaml").activation_checkpointing
    # The historical XL candidate remains an explicit future expansion.
    extra_large = ModelConfig.from_yaml("configs/research-xl.yaml")
    assert (extra_large.denoiser_depth, extra_large.denoiser_dim) == (28, 1152)


@pytest.mark.parametrize("stage", ["pretrain", "align"])
def test_shared_stage_has_identical_encoder_weights_and_no_generator(monkeypatch, stage):
    def forbidden(*args, **kwargs):
        raise AssertionError("shared Stage A/B must never construct MCWave-M or -L")

    monkeypatch.setattr("triwave.models.model.MCWave", forbidden)
    trainers = [Trainer(replace(ModelConfig.tiny(), **capacity), TrainingConfig(stage=stage))
                for capacity in CAPACITIES.values()]
    left, right = (trainer.model for trainer in trainers)
    right.load_state_dict(left.state_dict(), strict=True)
    assert left.state_dict().keys() == right.state_dict().keys()
    for trainer in trainers:
        assert trainer.model.denoiser is None and trainer.ema_denoiser is None
        assert not any(key.startswith("denoiser.") for key in trainer.model.state_dict())
        metrics = trainer.train_step(make_synthetic_batch(trainer.config, 2, 8))
        assert metrics["optimizer_updated"] == 1 and metrics["loss"] > 0


@pytest.mark.parametrize("width,depth,heads", [(64, 2, 4), (128, 4, 8)])
def test_scaled_runtime_accepts_condition_width_distinct_from_generator(width, depth, heads):
    torch.set_num_threads(2)
    config = replace(ModelConfig.tiny(), denoiser_dim=width,
                     denoiser_depth=depth, denoiser_heads=heads)
    assert config.condition_dim != config.denoiser_dim
    trainer = Trainer(config, TrainingConfig(warmup_steps=0))
    batch = make_synthetic_batch(config, 2, 5)
    metrics = trainer.train_step(batch)
    assert metrics["optimizer_updated"] == 1 and metrics["gradient_norm"] > 0
    generated = trainer.sample(trainer.conditions(batch), steps=2, guidance=1.5)
    assert generated.shape == batch["target_waveform"].shape
    assert torch.isfinite(generated).all()
