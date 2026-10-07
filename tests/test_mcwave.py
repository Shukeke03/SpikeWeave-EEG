from dataclasses import replace

import pytest
import torch

from triwave.config import ModelConfig
from triwave.diffusion import VPCleanDiffusion, vp_factors
from triwave.losses import clean_constraints
from triwave.models import SpikeWeaveModel
from triwave.montage import MontageSubspace
from triwave.signal import WaveletTransform


def test_montage_rank_closure_and_real_residual_gate():
    m = MontageSubspace.from_endpoints(torch.tensor([[0, 1], [1, 2], [0, 2]]))
    assert m.rank == 2
    source = torch.randn(2, 3, 64)
    source[:, 2] = source[:, 0] + source[:, 1]
    torch.testing.assert_close(m.lift(m.reduce(source)), source, atol=2e-6, rtol=2e-6)
    damaged = source.clone()
    damaged[:, 2] += 3
    assert not m.audit(damaged)["passed"]
    with pytest.raises(ValueError, match="audit failed"):
        m.reduce(damaged)
    missing = MontageSubspace(m.lead_matrix, torch.tensor([True, True, False]))
    with pytest.raises(ValueError, match="patterns change"):
        missing.reduce(source, torch.ones(2, 3, dtype=torch.bool))


def test_double_banana_rank_is_14_without_calling_modes_electrodes():
    cfg = replace(ModelConfig.tiny(), channels=16, montage_mode="audit")
    m = SpikeWeaveModel(cfg).montage()
    assert m.rank == 14 and m.Q.shape == (16, 14)
    torch.testing.assert_close(m.Q.T @ m.Q, torch.eye(14), atol=1e-6, rtol=1e-6)


def test_vp_schedule_score_drift_and_actual_nfe():
    d = VPCleanDiffusion()
    time = torch.tensor([0.001, 0.3, 1.0])
    a, s = vp_factors(time)
    torch.testing.assert_close(a * a + s * s, torch.ones(3))
    assert a[-1] > 0 and a[-1] < 0.007
    clean = torch.randn(3, 2, 16)
    eps = torch.randn_like(clean)
    state = d.noise(clean, eps, time)
    torch.testing.assert_close(state, a[:, None, None] * clean + s[:, None, None] * eps)
    assert d.clean_loss(clean, clean, time) == 0
    assert d.clean_weight(time).max() <= 5
    calls = []

    def oracle(state, t):
        calls.append(t)
        return clean

    result, stats = d.sample_heun(oracle, eps, intervals=5, return_stats=True)
    torch.testing.assert_close(result, clean)
    assert stats["nfe"] == 11 == len(calls) and calls[-1][0] == pytest.approx(0.001)
    result, stats = d.sample_ddim(oracle, eps, steps=11, return_stats=True)
    torch.testing.assert_close(result, clean)
    assert stats["nfe"] == 11


def test_constraint_identity_flatline_masks_and_full_boundary_derivatives():
    wavelet = WaveletTransform(128)
    target = torch.randn(2, 3, 128)
    target[:, 2] = 0
    coeff = wavelet.encode(target)
    losses = clean_constraints(target, target, coeff, coeff, wavelet.band_lengths)
    assert all(value.abs().max() < 1e-5 for value in losses.values())
    changed = target.clone()
    changed[:, :, 64:] += 1
    losses = clean_constraints(
        changed, target, wavelet.encode(changed), coeff, wavelet.band_lengths
    )
    assert losses["derivative"].min() > 0  # Includes patch boundary at 64.
    flat = torch.zeros_like(target, requires_grad=True)
    losses = clean_constraints(flat, target, wavelet.encode(flat), coeff, wavelet.band_lengths)
    sum(v.mean() for v in losses.values()).backward()
    assert torch.isfinite(flat.grad).all()


def test_physical_attribute_controls_ignore_unknown_and_use_source_scale():
    from triwave.losses import attribute_control_loss

    fs = 128.0
    time = torch.arange(128) / fs
    waveform = (2 * torch.sin(2 * torch.pi * 8 * time))[None, None].repeat(2, 3, 1).requires_grad_()
    values = torch.tensor([[8.0, 2**0.5], [999.0, float("nan")]])
    valid = torch.tensor([[True, True], [False, False]])
    loss = attribute_control_loss(waveform, values, valid, ("spectral_centroid_hz", "rms_uV"), fs)
    assert loss[0] < 1e-6 and loss[1] == 0
    scaled = attribute_control_loss(
        waveform / 10, values, valid, ("spectral_centroid_hz", "rms_uV"), fs, source_scale=10
    )
    torch.testing.assert_close(loss, scaled)
    (loss.mean() + waveform.square().mean()).backward()
    assert torch.isfinite(waveform.grad).all()
