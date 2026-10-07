"""Numerical QC on physical continuous recordings, without clinical decisions."""

from __future__ import annotations

import numpy as np
import pytest

from triwave.data.qc import (
    DEFAULT_EXCLUDED_FLAGS,
    Q_FLATLINE,
    Q_HIGH_AMPLITUDE,
    Q_NONFINITE,
    WINDOW_BATCH_SIZE,
    window_quality_flags,
)


def signal(windows: int, channels: int = 2, samples: int = 32) -> np.ndarray:
    time = np.arange(windows * samples, dtype=np.float64)
    return np.stack([20 * np.sin(2 * np.pi * (time + lead) / 8)
                     for lead in range(channels)])


def test_independent_flags_and_high_amplitude_retention():
    waveform = signal(5)
    waveform[0, 32:64] = 5.0  # Sustained constant offset is also a flatline.
    waveform[1, 64] = 1500.1
    waveform[0, 96] = np.nan
    waveform[1, 96 + 1] = 2000.0
    waveform[0, 128:160] = 1600.0
    flags = window_quality_flags(waveform, 256, 32)
    np.testing.assert_array_equal(flags, [0, Q_FLATLINE, Q_HIGH_AMPLITUDE,
                                        Q_NONFINITE | Q_HIGH_AMPLITUDE,
                                        Q_FLATLINE | Q_HIGH_AMPLITUDE])
    assert flags.dtype == np.uint16
    eligible = (flags & DEFAULT_EXCLUDED_FLAGS) == 0
    np.testing.assert_array_equal(eligible, [True, False, True, False, False])


def test_flatline_threshold_applies_to_whole_finite_lead_window():
    waveform = signal(3, channels=1)
    waveform[0, :32] = np.tile([-0.5e-3, 0.5e-3], 16)
    waveform[0, 32:64] = np.tile([-2e-3, 2e-3], 16)
    waveform[0, 64:] = 0.0
    waveform[0, 64] = 1.0  # A nonconstant window is not an entire-window flatline.
    np.testing.assert_array_equal(window_quality_flags(waveform, 256, 32), [Q_FLATLINE, 0, 0])


def test_nonfinite_is_not_replaced_with_a_flatline():
    waveform = np.zeros((1, 64))
    waveform[0, 0] = np.nan
    waveform[0, 32] = np.inf
    np.testing.assert_array_equal(window_quality_flags(waveform, 256, 32),
                                  [Q_NONFINITE, Q_NONFINITE])
    both = np.concatenate([waveform, np.zeros_like(waveform)], axis=0)
    np.testing.assert_array_equal(window_quality_flags(both, 256, 32),
                                  [Q_NONFINITE | Q_FLATLINE] * 2)


def test_high_amplitude_is_strict_and_independent_of_polarity():
    waveform = signal(3, channels=1)
    waveform[0, 0] = 1500.0
    waveform[0, 32] = -1500.0
    waveform[0, 64] = -1500.1
    np.testing.assert_array_equal(window_quality_flags(waveform, 256, 32),
                                  [0, 0, Q_HIGH_AMPLITUDE])


def test_partial_tail_and_missing_leads_are_explicit_caller_policy():
    waveform = np.concatenate([signal(1), np.full((2, 7), np.nan)], axis=-1)
    np.testing.assert_array_equal(window_quality_flags(waveform, 256, 32), [0])
    missing = np.concatenate([signal(1), np.zeros((1, 32))], axis=0)
    np.testing.assert_array_equal(window_quality_flags(missing, 256, 32), [Q_FLATLINE])
    np.testing.assert_array_equal(window_quality_flags(missing[:2], 256, 32), [0])
    assert window_quality_flags(signal(1)[:, :7], 256, 32).shape == (0,)


def test_batched_memmap_input_is_unchanged(tmp_path):
    waveform = signal(WINDOW_BATCH_SIZE + 2)
    waveform[0, WINDOW_BATCH_SIZE * 32:(WINDOW_BATCH_SIZE + 1) * 32] = 0
    waveform[0, -1] = -2000
    path = tmp_path / "record.npy"
    np.save(path, waveform.astype(np.float32))
    memory = np.load(path, mmap_mode="r")
    before = memory.copy()
    flags = window_quality_flags(memory, 256, 32)
    assert np.count_nonzero(flags) == 2
    assert flags[WINDOW_BATCH_SIZE] == Q_FLATLINE
    assert flags[-1] == Q_HIGH_AMPLITUDE
    np.testing.assert_array_equal(memory, before)


def test_strided_integer_input_and_empty_record():
    waveform = signal(2).astype(np.int16)[:, ::2]
    np.testing.assert_array_equal(window_quality_flags(waveform, 128, 16), [0, 0])
    result = window_quality_flags(np.empty((2, 0)), 256)
    assert result.shape == (0,) and result.dtype == np.uint16


@pytest.mark.parametrize("sample_rate", [0, -1, np.nan, np.inf, True, "256"])
def test_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        window_quality_flags(signal(1), sample_rate, 32)


@pytest.mark.parametrize("window_samples", [0, -1, 32.0, True])
def test_invalid_window_length(window_samples):
    with pytest.raises(ValueError, match="window_samples"):
        window_quality_flags(signal(1), 256, window_samples)


@pytest.mark.parametrize("waveform", [np.zeros(32), np.empty((0, 32)),
                                    np.ones((2, 32), dtype=complex),
                                    np.full((2, 32), "x"), np.zeros((2, 32), dtype=bool)])
def test_invalid_waveform(waveform):
    with pytest.raises(ValueError, match="waveform"):
        window_quality_flags(waveform, 256, 32)
