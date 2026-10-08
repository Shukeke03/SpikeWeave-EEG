"""Small, explicit EEG evaluation metrics with no learned self-evaluator.

Spectral TS-FID follows JET appendix A.2 equations 5–6. Pooling, FFT and
standardization conventions are recorded here; it is not claimed to reproduce
the authors' unpublished implementation or their differently processed scores.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import cdist


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Noninterpolated binary AP; ties share a threshold, independent of ordering."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all()):
        raise ValueError("AP requires finite scores and nonempty binary labels")
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    endpoints = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[endpoints].astype(np.float64)
    precision = tp / (endpoints + 1)
    return float(np.sum(np.diff(np.r_[0.0, tp]) * precision) / positives)


def patient_average_precision(labels: np.ndarray, scores: np.ndarray,
                              patient_ids: np.ndarray) -> dict:
    patients = np.asarray(patient_ids)
    if patients.shape != np.asarray(labels).shape:
        raise ValueError("patient identities must align with labels")
    rows = {}
    for patient in sorted(set(patients)):
        mask = patients == patient
        ap = average_precision(np.asarray(labels)[mask], np.asarray(scores)[mask])
        rows[str(patient)] = {"ap": ap, "windows": int(mask.sum()),
                              "positive": int(np.asarray(labels)[mask].sum()),
                              "background_above_0_5": int(np.sum(
                                  (np.asarray(labels)[mask] == 0)
                                  & (np.asarray(scores)[mask] >= 0.5)))}
    valid = [item["ap"] for item in rows.values() if item["ap"] is not None]
    return {"macro_patient_ap": float(np.mean(valid)) if valid else None,
            "pooled_ap": average_precision(labels, scores), "patients": rows,
            "computable_patients": len(valid), "total_patients": len(rows),
            "aggregation": "equal_weight_over_patients_with_positive_labels",
            "background_threshold": 0.5}


def _adaptive_pool(values: np.ndarray, bins: int, axis: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("pooling bins must be positive")
    length = values.shape[axis]
    result = []
    for index in range(bins):
        start = index * length // bins
        stop = ((index + 1) * length + bins - 1) // bins
        result.append(np.take(values, np.arange(start, stop), axis=axis).mean(axis=axis))
    return np.stack(result, axis=axis)


def spectral_ts_features(waveforms: np.ndarray, sample_rate: float, *,
                         spectral_bins: int = 256, spatial_bins: int = 4,
                         cutoff_hz: float | None = 80.0) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all() or sample_rate <= 0:
        raise ValueError("spectral features require finite [N,C,T] and physical sample rate")
    spectra = np.abs(np.fft.rfft(values, axis=-1))
    if cutoff_hz is not None:
        if cutoff_hz <= 0 or not np.isfinite(cutoff_hz):
            raise ValueError("cutoff must be finite and positive")
        spectra = spectra[..., np.fft.rfftfreq(values.shape[-1], 1 / sample_rate) <= cutoff_hz]
    pooled = _adaptive_pool(_adaptive_pool(spectra, spectral_bins, -1), spatial_bins, 1)
    features = np.log1p(pooled).reshape(len(values), -1)
    std = features.std(axis=1, keepdims=True)
    return (features - features.mean(axis=1, keepdims=True)) / np.maximum(std, 1e-8)


def frechet_distance(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Exact sample-covariance Fréchet via low-rank factors (works when n << d)."""
    real, generated = np.asarray(real_features, dtype=float), np.asarray(generated_features, dtype=float)
    if (real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]
            or min(len(real), len(generated)) < 2
            or not np.isfinite(real).all() or not np.isfinite(generated).all()):
        raise ValueError("Fréchet requires two finite feature matrices with >=2 samples")
    mean_r, mean_g = real.mean(0), generated.mean(0)
    a = (real - mean_r) / np.sqrt(len(real) - 1)
    b = (generated - mean_g) / np.sqrt(len(generated) - 1)
    # The small-sample factor is at most a few dozen rows in G1. Explicit
    # reduction avoids vendor-BLAS floating-status warnings on some macOS builds.
    cross = np.einsum("id,jd->ij", a, b, optimize=False)
    cross_trace = np.linalg.svd(cross, compute_uv=False).sum()
    result = np.square(mean_r - mean_g).sum() + np.square(a).sum() + np.square(b).sum() - 2 * cross_trace
    return float(max(result, 0.0))


def physical_features(waveforms: np.ndarray, sample_rate: float) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float64)
    frequencies, power = welch(values, fs=sample_rate, nperseg=min(256, values.shape[-1]), axis=-1)
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80))
    features = [values.mean((1, 2)), values.std((1, 2)),
                np.sqrt(np.square(values).mean((1, 2))),
                np.abs(values).max((1, 2)), np.abs(np.diff(values, axis=-1)).mean((1, 2))]
    for low, high in bands:
        selected = (frequencies >= low) & (frequencies < min(high, sample_rate / 2 + 1e-8))
        features.append(np.log1p(power[..., selected].sum((1, 2))) if selected.any()
                        else np.zeros(len(values)))
    return np.stack(features, axis=1)


def source_scaled_mmd(real: np.ndarray, generated: np.ndarray,
                      source: np.ndarray) -> dict:
    """Biased nonnegative RBF MMD; scale and bandwidth fitted to source only."""
    mean, scale = source.mean(0), np.maximum(source.std(0), 1e-8)
    fit = (source - mean) / scale
    distances = cdist(fit, fit, "sqeuclidean")
    positive = distances[np.triu_indices(len(fit), 1)]
    positive = positive[positive > 0]
    bandwidth_sq = float(np.median(positive)) if len(positive) else 1.0
    x, y = (real - mean) / scale, (generated - mean) / scale
    def kernel(left, right):
        return np.exp(-cdist(left, right, "sqeuclidean") / (2 * bandwidth_sq))
    mmd = kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
    return {"mmd_squared": float(max(mmd, 0.0)), "bandwidth_squared": bandwidth_sq,
            "scale_fit": "source_train_only", "estimator": "biased_rbf"}


def nearest_copy_audit(generated: np.ndarray, source: np.ndarray, *,
                        max_source: int = 256) -> dict:
    """Bounded original-waveform nearest-neighbor screen, not a privacy guarantee."""
    positions = np.linspace(0, len(source) - 1, min(len(source), max_source), dtype=int)
    candidates = np.asarray(source[positions], dtype=np.float64).reshape(len(positions), -1)
    query = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    distances = cdist(query, candidates, "sqeuclidean") / candidates.shape[1]
    chosen = distances.argmin(1)
    rmse = np.sqrt(np.maximum(distances[np.arange(len(query)), chosen], 0))
    return {"nearest_source_positions": positions[chosen].tolist(), "rmse_uV": rmse.tolist(),
            "near_exact_count": int((rmse <= 1e-5).sum()),
            "source_windows_screened": len(positions), "source_windows_total": len(source),
            "scope": "bounded_unshifted_waveform_screen_not_membership_inference"}
