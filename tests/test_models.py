"""Two branches, shared C modes, physical masks and clean-target isolation."""

from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.data.synthetic import make_synthetic_batch
from triwave.models import SpikeWeaveModel
from triwave.signal import MorletCWT
from triwave.types import ConditionBatch


@pytest.fixture
def setup():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg = ModelConfig.tiny()
    model = SpikeWeaveModel(cfg)
    batch = make_synthetic_batch(cfg, 3, 4)
    cwt = MorletCWT(
        cfg.sample_rate,
        cfg.cwt_frequencies,
        cfg.cwt_min_frequency,
        cfg.cwt_max_frequency,
        cfg.cwt_times,
    )
    image, mask = cwt(batch["reference_waveform"])
    cond = ConditionBatch(
        3,
        batch["lead_geometry"],
        batch["reference_waveform"],
        batch["reference_valid"],
        image,
        mask,
        batch["input_ids"],
        batch["text_valid"],
        batch["attribute_values"],
        batch["attribute_valid"],
    )
    return cfg, model, cond


def equal(a, b):
    torch.testing.assert_close(a.tokens, b.tokens, atol=0, rtol=0)
    torch.testing.assert_close(a.global_vector, b.global_vector, atol=0, rtol=0)
    assert torch.equal(a.valid, b.valid)


def test_exactly_two_branches_and_complete_clean_coefficients(setup):
    cfg, m, c = setup
    assert {"eeg_encoder", "condition_encoder"} <= set(dict(m.named_children()))
    assert not hasattr(m, "text_encoder") and not hasattr(m, "structure_encoder")
    assert m.band_lengths == (16, 16, 32, 64)
    out = m(torch.randn(3, cfg.channels, cfg.samples), torch.rand(3), c)
    assert out.shape == (3, 4, 128) and torch.isfinite(out).all()
    (out - torch.randn_like(out)).square().mean().backward()
    assert all(head.weight.grad.abs().sum() > 0 for head in m.denoiser.output_heads)


def test_all_missing_is_finite_and_not_a_contrastive_candidate(setup):
    cfg, m, c = setup
    null = ConditionBatch(3, c.lead_geometry)
    assert m.encode_conditions(null).valid.any(-1).all()
    a = m.alignment(
        torch.full_like(c.reference_waveform, float("nan")),
        torch.full_like(c.structure, float("nan")),
        torch.zeros_like(c.structure_valid),
        torch.full_like(c.input_ids, -999),
        torch.zeros_like(c.text_valid),
        c.lead_geometry,
        torch.zeros_like(c.reference_valid),
    )
    assert not any(v.any() for v in (a.eeg_valid, a.text_valid, a.image_valid, a.joint_valid))
    assert all(torch.isfinite(x).all() for x in (a.eeg, a.eeg_structure, a.joint, a.text, a.image))


def test_dropped_inputs_and_masked_nan_values_do_not_leak(setup):
    _, m, c = setup
    m.eval()
    changed = replace(
        c,
        input_ids=torch.full_like(c.input_ids, -999),
        structure=torch.full_like(c.structure, float("nan")),
        reference_waveform=torch.full_like(c.reference_waveform, float("nan")),
        attribute_values=torch.full_like(c.attribute_values, float("nan")),
    )
    flags = dict(drop_text=True, drop_structure=True, drop_reference=True, drop_attributes=True)
    equal(m.encode_conditions(c, **flags), m.encode_conditions(changed, **flags))
    changed = replace(c, input_ids=torch.randint(0, m.config.text_vocab_size, c.input_ids.shape))
    equal(m.encode_conditions(c, drop_text=True), m.encode_conditions(changed, drop_text=True))
    assert not torch.allclose(m.encode_conditions(c).tokens, m.encode_conditions(changed).tokens)
    iv = c.structure_valid.clone()
    iv[:, :, 4:7] = False
    changed = replace(c, structure=c.structure.clone(), structure_valid=iv)
    changed.structure[~iv] = float("nan")
    equal(m.encode_conditions(replace(c, structure_valid=iv)), m.encode_conditions(changed))


def test_target_alignment_has_no_generation_cache_and_both_branches_receive_gradients(setup):
    _, m, c = setup
    m.eval()
    before = m.encode_conditions(c)
    a = m.alignment(
        c.reference_waveform,
        c.structure,
        c.structure_valid,
        c.input_ids,
        c.text_valid,
        c.lead_geometry,
    )
    loss = (
        (a.eeg - a.joint).square().mean()
        + (a.eeg_structure - a.image).square().mean()
        + (a.eeg - a.text).square().mean()
    )
    loss.backward()
    for branch in (m.eeg_encoder, m.condition_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None for p in m.denoiser.parameters())
    equal(before, m.encode_conditions(c))
    with pytest.raises(TypeError):
        ConditionBatch(3, c.lead_geometry, target_waveform=c.reference_waveform)
    with pytest.raises(TypeError):
        m(
            torch.randn_like(c.reference_waveform),
            torch.rand(3),
            c.reference_waveform,
            c.lead_geometry,
        )


def test_main_research_config_has_fixed_full_scale_dimensions():
    cfg = ModelConfig.from_yaml("configs/research-5090.yaml")
    assert (cfg.dim, cfg.encoder_depth, cfg.heads, cfg.waveform_patch) == (512, 8, 8, 16)
    assert (cfg.condition_dim, cfg.fusion_depth, cfg.condition_heads, cfg.text_max_length) == (
        768,
        6,
        12,
        256,
    )
    assert (cfg.denoiser_dim, cfg.denoiser_depth, cfg.denoiser_heads, cfg.coefficient_patch) == (
        1024,
        24,
        16,
        16,
    )
    assert cfg.frontend_backend == "biomedclip"
