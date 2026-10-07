"""Fixed numerical EEG quality flags, without diagnostic or fitted thresholds.

Inputs must be physical microvolts and contain only genuinely observed leads.
Run this separately on raw and processed recordings when retaining both audits:
filtering can hide flatlines and spread nonfinite samples into surrounding data.
The flags identify numerical concerns, not seizure status or clinical quality.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

Q_NONFINITE = 1
Q_FLATLINE = 2
Q_HIGH_AMPLITUDE = 4

QUALITY_FLAG_NAMES = {
    Q_NONFINITE: "nonfinite",
    Q_FLATLINE: "flatline",
    Q_HIGH_AMPLITUDE: "high_amplitude",
}
DEFAULT_EXCLUDED_FLAGS = Q_NONFINITE | Q_FLATLINE
FLATLINE_STD_UV = 1e-3
HIGH_AMPLITUDE_UV = 1500.0
WINDOW_BATCH_SIZE = 256


def window_quality_flags(waveform: np.ndarray, sample_rate: float,
                         window_samples: int = 1024) -> np.ndarray:
    """Return one uint16 bit field for each complete nonoverlapping window.

    ``waveform`` has shape [observed_lead, sample] in physical uV. A window is
    flagged if any observed lead has a nonfinite value, standard deviation at
    most 1e-3 uV throughout that entire window, or a finite absolute value above
    1500 uV. Flatline detection requires the entire lead window to be finite;
    an invalid value is never silently replaced with a synthetic flatline.

    The sample rate is validated as measurement metadata; ``window_samples``
    specifies the actual sample count (1024 is four seconds at 256 Hz). An
    incomplete final window is omitted. Missing leads must be removed by the
    caller, since zero-filled missing leads are indistinguishable from flatlines.
    Only small batches are promoted to float64; a full recording is never copied.

    This function does not discard anything. The preparation/training policy
    normally excludes NONFINITE and FLATLINE, while retaining HIGH_AMPLITUDE.
    """
    if (not isinstance(sample_rate, Real) or isinstance(sample_rate, bool)
            or not math.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("sample_rate must be finite and positive")
    if (not isinstance(window_samples, Integral) or isinstance(window_samples, bool)
            or window_samples <= 0):
        raise ValueError("window_samples must be a positive integer")
    waveform = np.asarray(waveform)
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise ValueError("waveform must have shape [observed_lead, sample] with at least one lead")
    if not np.issubdtype(waveform.dtype, np.number) or np.iscomplexobj(waveform):
        raise ValueError("waveform must contain real numeric physical-uV samples")

    window_samples = int(window_samples)
    windows = waveform.shape[1] // window_samples
    flags = np.zeros(windows, dtype=np.uint16)
    for first in range(0, windows, WINDOW_BATCH_SIZE):
        stop = min(first + WINDOW_BATCH_SIZE, windows)
        block = waveform[:, first * window_samples:stop * window_samples]
        block = np.asarray(block, dtype=np.float64).reshape(
            waveform.shape[0], stop - first, window_samples)
        finite = np.isfinite(block)
        lead_finite = finite.all(axis=-1)
        nonfinite = ~lead_finite.all(axis=0)
        # Nonfinite/overflowing inputs are explicitly flagged or fail the
        # flatline predicate; they do not require replacing signal values.
        with np.errstate(invalid="ignore", over="ignore"):
            deviation = np.std(block, axis=-1, dtype=np.float64)
        flatline = (lead_finite & (deviation <= FLATLINE_STD_UV)).any(axis=0)
        peak = np.max(np.abs(block), axis=-1, where=finite, initial=0.0)
        high_amplitude = (peak > HIGH_AMPLITUDE_UV).any(axis=0)
        flags[first:stop] = (
            nonfinite.astype(np.uint16) * Q_NONFINITE
            | flatline.astype(np.uint16) * Q_FLATLINE
            | high_amplitude.astype(np.uint16) * Q_HIGH_AMPLITUDE
        )
    return flags
