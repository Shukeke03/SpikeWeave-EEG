"""One physical-unit attribute operator shared by data, control losses and audits.

Band energies use the historical Hann rFFT / sum(window**2) definition, not a
calibrated one-sided PSD. Missing measurements never become measured zeros.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

ATTRIBUTE_PROTOCOL = "hann-energy-v2"
ATTRIBUTE_SCHEMA = (
    "spectral_centroid_hz", "rms_uV", "std_uV", "mean_uV",
    "delta_fraction", "theta_fraction", "alpha_fraction", "beta_fraction", "gamma_fraction",
    "line_length_uV", "delta_log_power_uV2", "theta_log_power_uV2", "alpha_log_power_uV2",
    "beta_log_power_uV2", "gamma_log_power_uV2", "peak_abs_uV",
)
BANDS = (("delta", .5, 4.), ("theta", 4., 8.), ("alpha", 8., 13.),
         ("beta", 13., 30.), ("gamma", 30., 80.))


def waveform_attributes(
    waveform: Tensor,
    valid: Tensor | None = None,
    sample_rate: float = 256.0,
    max_frequency: float | Tensor = 80.0,
) -> tuple[Tensor, Tensor]:
    """Return [B,16] physical values and bool masks in ATTRIBUTE_SCHEMA order.

    Input is [B,C,T] in microvolts. Magnitudes are measured per lead and then
    averaged equally over eligible leads. Spectral measurements require a whole
    valid lead; a cropped/missing interval is not silently filled for an FFT.
    max_frequency declares source bandwidth, scalar or one value per record.
    """
    if waveform.ndim != 3 or min(waveform.shape[:2]) < 1 or waveform.shape[-1] < 4:
        raise ValueError("physical waveform must have shape [B,C,T>=4]")
    if not waveform.is_floating_point() or not math.isfinite(sample_rate) or sample_rate <= 1:
        raise ValueError("attributes require floating microvolt values and a valid sample rate")
    x = waveform.float()
    if valid is None:
        observed = torch.ones_like(x, dtype=torch.bool)
    else:
        if valid.dtype != torch.bool:
            raise ValueError("attribute validity must be boolean")
        if valid.shape == x.shape[:2]:
            observed = valid[..., None].expand_as(x)
        elif valid.shape == x.shape:
            observed = valid
        else:
            raise ValueError("attribute mask must be [B,C] or [B,C,T]")
        observed = observed.to(x.device)
    if not torch.isfinite(torch.where(observed, x, 0)).all():
        raise ValueError("observed physical attributes contain nonfinite input")
    x = torch.where(observed, x, 0)
    count = observed.sum(-1)
    amplitude_ok = count > 0
    mean = x.sum(-1) / count.clamp_min(1)
    centered = torch.where(observed, x - mean[..., None], 0)
    variance = centered.square().sum(-1) / count.clamp_min(1)
    rms = (x.square().sum(-1) / count.clamp_min(1) + 1e-8).sqrt()
    std = (variance + 1e-8).sqrt()
    pair_valid = observed[..., 1:] & observed[..., :-1]
    line_count = pair_valid.sum(-1)
    line = torch.where(pair_valid, x.diff().abs(), 0).sum(-1) / line_count.clamp_min(1)
    peak = x.abs().amax(-1)

    upper = torch.as_tensor(max_frequency, dtype=x.dtype, device=x.device)
    if upper.ndim == 0:
        upper = upper.expand(x.shape[0])
    if upper.shape != (x.shape[0],) or not torch.isfinite(upper).all() or (upper <= .5).any():
        raise ValueError("source bandwidth must be a finite scalar or [B] above 0.5 Hz")
    upper = upper.clamp(max=min(80.0, sample_rate / 2))
    frequencies = torch.fft.rfftfreq(x.shape[-1], 1 / sample_rate, device=x.device)
    window = torch.hann_window(x.shape[-1], periodic=True, device=x.device)
    power = torch.fft.rfft(centered * window).abs().square() / window.square().sum()
    support = ((frequencies[None] >= .5) & (frequencies[None] < upper[:, None]))
    complete = observed.all(-1)
    total = (power * support[:, None]).sum(-1)
    spectral_ok = complete & (total > 1e-8) & support.any(-1)[:, None]
    denominator = total.clamp_min(1e-8)
    centroid = (power * support[:, None] * frequencies).sum(-1) / denominator
    estimates = {"spectral_centroid_hz": centroid, "rms_uV": rms, "std_uV": std,
                 "mean_uV": mean, "line_length_uV": line, "peak_abs_uV": peak}
    validity = {"spectral_centroid_hz": spectral_ok, "rms_uV": amplitude_ok,
                "std_uV": amplitude_ok, "mean_uV": amplitude_ok,
                "line_length_uV": line_count > 0, "peak_abs_uV": amplitude_ok}
    for name, lower, high in BANDS:
        band = support & (frequencies[None] >= lower) & (frequencies[None] < high)
        energy = (power * band[:, None]).sum(-1)
        band_ok = spectral_ok & (upper >= high)[:, None] & band.any(-1)[:, None]
        estimates[name + "_fraction"] = energy / denominator
        estimates[name + "_log_power_uV2"] = torch.log(energy + 1e-8)
        validity[name + "_fraction"] = validity[name + "_log_power_uV2"] = band_ok
    values = torch.stack([estimates[name] for name in ATTRIBUTE_SCHEMA], -1)
    masks = torch.stack([validity[name] for name in ATTRIBUTE_SCHEMA], -1)
    result_valid = masks.any(1)
    result = torch.where(masks, values, 0).sum(1) / masks.sum(1).clamp_min(1)
    return torch.where(result_valid, result, 0), result_valid
