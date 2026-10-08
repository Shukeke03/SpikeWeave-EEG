"""Physical measurement, incomplete support and differentiable control checks."""

import pytest
import torch

from triwave.attributes import ATTRIBUTE_SCHEMA, waveform_attributes
from triwave.config import ModelConfig


def test_source_physical_scale_and_complete_band_partition():
    t = torch.arange(1024) / 256
    x = (10 * torch.sin(2 * torch.pi * 10 * t) + 3)[None, None].repeat(2, 3, 1)
    values, valid = waveform_attributes(x)
    assert ATTRIBUTE_SCHEMA == ModelConfig().attribute_schema
    assert valid.all() and values.shape == (2, 16)
    torch.testing.assert_close(values[:, 0], torch.full((2,), 10.0), atol=1e-4, rtol=0)
    torch.testing.assert_close(values[:, 1], torch.full((2,), 59 ** .5), atol=1e-5, rtol=0)
    torch.testing.assert_close(values[:, 3], torch.full((2,), 3.), atol=1e-5, rtol=0)
    torch.testing.assert_close(values[:, 4:9].sum(-1), torch.ones(2), atol=1e-6, rtol=0)
    assert (values[:, 6] > .99).all()


def test_partial_lead_has_amplitude_but_no_spectral_measurements():
    x = torch.randn(1, 2, 1024)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 0] = False
    mask[0, 1, 200:250] = False
    x[~mask] = float("nan")
    values, valid = waveform_attributes(x, mask)
    assert torch.isfinite(values).all()
    assert not valid[0, 0] and not valid[0, 4:9].any() and not valid[0, 10:15].any()
    assert valid[0, 1:4].all() and valid[0, 9] and valid[0, 15]
    with pytest.raises(ValueError, match="nonfinite"):
        waveform_attributes(x)


def test_unavailable_gamma_is_not_a_measured_zero_and_constant_spectrum_invalid():
    values, valid = waveform_attributes(torch.randn(2, 3, 1024), max_frequency=48.)
    assert not valid[:, 8].any() and not valid[:, 14].any()
    assert (values[:, 8] == 0).all()
    _, constant_valid = waveform_attributes(torch.ones(1, 2, 1024))
    assert not constant_valid[:, 0].any() and not constant_valid[:, 4:9].any()


def test_invalid_all_mask_returns_zero_and_gradients_are_finite():
    x = torch.randn(2, 3, 1024, requires_grad=True)
    values, valid = waveform_attributes(x)
    values[valid].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    empty, empty_valid = waveform_attributes(x.detach(), torch.zeros(2, 3, dtype=torch.bool))
    assert not empty_valid.any() and torch.equal(empty, torch.zeros_like(empty))


def test_eighty_hz_bin_excluded_from_shared_partition():
    # The shared support and every band must partition exactly the same bins.
    t = torch.arange(1024) / 256
    x = (torch.sin(2 * torch.pi * 79.5 * t) + 5 * torch.sin(2 * torch.pi * 80 * t))[None, None]
    values, valid = waveform_attributes(x)
    assert valid.all()
    torch.testing.assert_close(values[:, 4:9].sum(-1), torch.ones(1), atol=1e-6, rtol=0)
