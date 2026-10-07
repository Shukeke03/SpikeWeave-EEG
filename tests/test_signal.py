"""Signal-level correctness tests, not model performance claims."""

import math

import numpy as np
import pytest
import pywt
import torch
from torch.nn import functional as F

from triwave.signal import MorletCWT, WaveletTransform, lead_cycle_residual


@pytest.mark.parametrize("samples,levels", [(128, 3), (256, 2), (1024, 3)])
def test_dwt_coefficients_and_reconstruction_match_pywavelets(samples, levels):
    rng = np.random.default_rng(31)
    original = rng.normal(size=(2, 3, samples))
    transform = WaveletTransform(samples, levels)
    expected = pywt.wavedec(original, "db4", mode="periodization", level=levels, axis=-1)
    encoded = transform.encode(torch.from_numpy(original))
    assert transform.band_lengths == tuple(part.shape[-1] for part in expected)
    assert transform.band_names == (f"A{levels}", *(f"D{j}" for j in range(levels, 0, -1)))
    np.testing.assert_allclose(encoded.numpy(), np.concatenate(expected, axis=-1), atol=2e-14)
    decoded = transform.decode(encoded)
    np.testing.assert_allclose(decoded.numpy(), original, atol=2e-14)
    np.testing.assert_allclose(
        decoded.numpy(), pywt.waverec(expected, "db4", mode="periodization", axis=-1),
        atol=2e-14,
    )


def test_dwt_signed_coefficients_and_autograd():
    transform = WaveletTransform(128)
    x = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
    y = transform(x)
    torch.testing.assert_close(transform(-x), -y)
    reconstruction = transform.decode(y)
    loss = reconstruction.square().sum()
    loss.backward()
    torch.testing.assert_close(x.grad, 2 * x.detach(), atol=1e-12, rtol=1e-12)


def test_dwt_masks_propagate_through_filter_support():
    transform = WaveletTransform(128)
    valid = torch.ones(1, 2, 128, dtype=torch.bool)
    valid[0, 0, 50] = False
    valid[0, 1] = False
    coefficient_valid = transform.coefficient_valid(valid)
    support = transform.analysis_matrix[:, 50].abs() > 1e-12
    assert torch.equal(coefficient_valid[0, 0], ~support)
    assert not coefficient_valid[0, 1].any()


@pytest.mark.parametrize("kwargs", [
    {"samples": 4097}, {"samples": 127}, {"samples": 128, "levels": 0},
    {"samples": 128, "mode": "periodic"}, {"samples": 128, "wavelet": "bior2.2"},
])
def test_dwt_rejects_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        WaveletTransform(**kwargs)


def test_dwt_float32_and_wrong_input_shape():
    transform = WaveletTransform(128)
    x = torch.randn(1, 2, 128)
    torch.testing.assert_close(transform.decode(transform(x)), x, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        transform(torch.randn(1, 2, 127))
    with pytest.raises(TypeError):
        transform(torch.zeros(1, 2, 128, dtype=torch.long))


def _sinusoid(frequency=20.0, samples=1024, sample_rate=256.0):
    time = torch.arange(samples, dtype=torch.float64) / sample_rate
    return torch.sin(2 * math.pi * frequency * time)[None, None]


def test_cwt_sinusoid_frequency_and_amplitude_scaling():
    cwt = MorletCWT(256.0, frequencies=48, min_frequency=4.0, max_frequency=80.0,
                    time_bins=128)
    x = _sinusoid()
    view, valid = cwt(x)
    louder, louder_valid = cwt(2 * x)
    assert view.shape == (1, 1, 48, 128)
    assert torch.equal(valid, louder_valid)
    mean_power = (view.exp() * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    peak_frequency = cwt.frequencies[mean_power[0, 0].argmax()]
    assert abs(float(peak_frequency) - 20.0) / 20.0 < 0.08
    # Doubling signal amplitude quadruples linear power. Include the fixed
    # logarithm floor rather than demanding log(4) at near-zero off-band power.
    expected_louder = torch.log(4 * (view.exp() - cwt.epsilon) + cwt.epsilon)
    torch.testing.assert_close(louder[valid], expected_louder[valid], atol=1e-10, rtol=1e-10)


def test_cwt_pooling_is_linear_power_before_log():
    full = MorletCWT(128.0, frequencies=8, min_frequency=6, max_frequency=48, time_bins=256)
    pooled = MorletCWT(128.0, frequencies=8, min_frequency=6, max_frequency=48, time_bins=16)
    x = torch.randn(1, 1, 256, dtype=torch.float64)
    fine, fine_valid = full(x)
    coarse, coarse_valid = pooled(x)
    power = (fine.exp() - full.epsilon).clamp_min(0) * fine_valid
    numerator = F.adaptive_avg_pool1d(power.flatten(0, 1), 16)
    denominator = F.adaptive_avg_pool1d(fine_valid.double().flatten(0, 1), 16)
    expected_valid = denominator > 0
    expected = torch.log(numerator / denominator.clamp_min(full.epsilon) + full.epsilon)
    expected = torch.where(expected_valid, expected, 0).reshape_as(coarse)
    assert torch.equal(coarse_valid, expected_valid.reshape_as(coarse_valid))
    torch.testing.assert_close(coarse, expected, atol=1e-9, rtol=1e-9)


def test_cwt_boundary_low_frequency_zero_and_fully_missing():
    cwt = MorletCWT(128, frequencies=16, min_frequency=2, max_frequency=48, time_bins=128)
    zeros = torch.zeros(2, 2, 128)
    view, valid = cwt(zeros)
    assert torch.isfinite(view).all()
    assert not valid[:, :, 0].any()  # The 2 Hz cone exceeds this 1-second window.
    assert not valid[..., 0].any()
    assert not valid[..., -1].any()
    assert valid[:, :, -1].any()
    assert (view[~valid] == 0).all()
    torch.testing.assert_close(view[valid], torch.full_like(view[valid], math.log(cwt.epsilon)))
    missing_view, missing_valid = cwt(zeros, torch.zeros(2, 2, dtype=torch.bool))
    assert not missing_valid.any()
    assert (missing_view == 0).all()


def test_cwt_missing_samples_nan_fill_and_autograd():
    cwt = MorletCWT(128, frequencies=8, min_frequency=8, max_frequency=48, time_bins=256)
    x = torch.randn(1, 2, 256, dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 128] = False
    with torch.no_grad():
        x[:, :, 128] = float("nan")
    view, valid = cwt(x, mask)
    assert not valid[..., 128].any()
    assert torch.isfinite(view).all()
    view[valid].sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[:, :, 128] == 0).all()
    assert x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="finite"):
        cwt(x)


def test_cwt_invalid_configuration_and_mask():
    with pytest.raises(ValueError):
        MorletCWT(128, max_frequency=80)
    cwt = MorletCWT(128, max_frequency=48)
    with pytest.raises(TypeError, match="boolean"):
        cwt(torch.zeros(1, 1, 128), torch.ones(1, 128))


def test_lead_cycle_projection_does_not_infer_absolute_potentials():
    # Two paths 0->1->2 and 0->2 measure the same electrode difference.
    matrix = torch.tensor([[1., -1., 0.], [0., 1., -1.], [1., 0., -1.]],
                          dtype=torch.float64)
    potentials = torch.randn(2, 3, 128, dtype=torch.float64)
    leads = torch.einsum("ce,bel->bcl", matrix, potentials)
    residual = lead_cycle_residual(leads, matrix)
    torch.testing.assert_close(residual, torch.zeros_like(leads), atol=1e-12, rtol=0)
    inconsistent = leads.clone()
    inconsistent[:, 0] += 1
    assert lead_cycle_residual(inconsistent, matrix).abs().max() > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_signal_transforms_cuda_gradients():
    x = torch.randn(1, 2, 256, device="cuda", requires_grad=True)
    wavelet = WaveletTransform(256).cuda()
    cwt = MorletCWT(128, frequencies=8, min_frequency=8, max_frequency=48,
                    time_bins=32).cuda()
    decoded = wavelet.decode(wavelet(x))
    view, valid = cwt(decoded)
    view[valid].sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(decoded, x, atol=1e-5, rtol=1e-5)
