"""Clean physiological constraints and explicit historical comparison utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from triwave.types import AlignmentOutput


def masked_mse(
    prediction: Tensor, target: Tensor, valid: Tensor | None = None, reduction: str = "mean"
) -> Tensor:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    if valid is None:
        error = (prediction - target).square()
        return error.mean() if reduction == "mean" else error.flatten(1).mean(-1)
    mask = torch.broadcast_to(valid.bool(), target.shape)
    difference = torch.where(mask, prediction - target, 0)
    if reduction == "none":
        return difference.square().flatten(1).sum(-1) / mask.flatten(1).sum(-1).clamp_min(1)
    return difference.square().sum() / mask.sum().clamp_min(1)


def contrastive_pair(
    left: Tensor,
    right: Tensor,
    left_valid: Tensor,
    right_valid: Tensor,
    groups: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric multi-positive InfoNCE; candidates with a missing modality are excluded."""
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    common = left_valid.bool() & right_valid.bool()
    if common.sum() < 2:
        return (left.sum() + right.sum()) * 0
    left, right = F.normalize(left[common], dim=-1), F.normalize(right[common], dim=-1)
    logits = left @ right.T / temperature
    if groups is None:
        positives = torch.eye(len(left), device=left.device, dtype=torch.bool)
    else:
        labels = groups[common]
        positives = labels[:, None] == labels[None, :]

    def direction(scores: Tensor, positive_mask: Tensor) -> Tensor:
        positive_scores = scores.masked_fill(~positive_mask, -torch.inf)
        return (torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)).mean()

    return (direction(logits, positives) + direction(logits.T, positives.T)) / 2


def alignment_loss(output: AlignmentOutput, records: list[dict] | None = None) -> Tensor:
    from triwave.alignment import fixed_alignment_loss

    return fixed_alignment_loss(output, records)[0]


def signal_regularizers(
    prediction: Tensor, target: Tensor, valid: Tensor | None = None, reduction: str = "mean"
) -> dict[str, Tensor]:
    """PSD and cross-lead correlation are penalties, not guarantees of physiology."""
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    elif valid.ndim == 2:
        valid = valid[..., None].expand_as(target)
    valid = torch.broadcast_to(valid.bool(), target.shape)
    prediction = torch.where(valid, prediction, 0).float()
    target = torch.where(valid, target, 0).float()
    window = torch.hann_window(target.shape[-1], device=target.device)
    target_power = torch.fft.rfft(target * window).abs().square()
    prediction_power = torch.fft.rfft(prediction * window).abs().square()
    lead_valid = valid.all(dim=-1)
    psd = masked_mse(
        torch.log1p(prediction_power),
        torch.log1p(target_power),
        lead_valid[..., None],
        reduction=reduction,
    )

    def correlation(x: Tensor) -> Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-6)
        return normalized @ normalized.transpose(-1, -2)

    pair_valid = lead_valid[..., None] & lead_valid[:, None, :]
    topology = masked_mse(
        correlation(prediction), correlation(target), pair_valid, reduction=reduction
    )
    return {"psd": psd, "topology": topology}


def _masked_average(value: Tensor, valid: Tensor) -> Tensor:
    mask = torch.broadcast_to(valid.bool(), value.shape)
    return torch.where(mask, value, 0).flatten(1).sum(-1) / mask.flatten(1).sum(-1).clamp_min(1)


def clean_constraints(
    prediction: Tensor,
    target: Tensor,
    predicted_coefficients: Tensor,
    target_coefficients: Tensor,
    band_lengths: tuple[int, ...],
    valid: Tensor | None = None,
    local_samples: int = 128,
    control_prediction: Tensor | None = None,
    control_target: Tensor | None = None,
    control_valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """Six clean-estimate constraint families, averaged over observed elements.

    Inputs use the source-fitted scalar amplitude scale. L1 is the Laplace
    reconstruction prior; no graph Laplacian is implied. Derivatives cover ALL
    sample boundaries. Missing controls contribute zero, never pseudo labels.
    """
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    elif valid.ndim == 2:
        valid = valid[..., None].expand_as(target)
    valid = torch.broadcast_to(valid.bool(), target.shape)
    x, y = torch.where(valid, prediction.float(), 0), torch.where(valid, target.float(), 0)
    weights = valid.float()
    count = weights.sum(-1).clamp_min(1)
    lead_valid = valid.any(-1)
    mx, my = x.sum(-1) / count, y.sum(-1) / count
    cx, cy = torch.where(valid, x - mx[..., None], 0), torch.where(valid, y - my[..., None], 0)
    vx, vy = cx.square().sum(-1) / count, cy.square().sum(-1) / count
    stdx, stdy = (vx + 1e-8).sqrt(), (vy + 1e-8).sqrt()
    local_mean, local_std = [], []
    for start in range(0, target.shape[-1], local_samples):
        stop = min(start + local_samples, target.shape[-1])
        w = weights[..., start:stop]
        nx = w.sum(-1).clamp_min(1)
        ax = x[..., start:stop].sum(-1) / nx
        ay = y[..., start:stop].sum(-1) / nx
        bx = (((x[..., start:stop] - ax[..., None]).square() * w).sum(-1) / nx + 1e-8).sqrt()
        by = (((y[..., start:stop] - ay[..., None]).square() * w).sum(-1) / nx + 1e-8).sqrt()
        local_mean.append(_masked_average((ax - ay).abs(), w.any(-1)))
        local_std.append(_masked_average((bx - by).abs(), w.any(-1)))
    target_nonflat = lead_valid & (vy > 1e-8)
    pearson = (cx * cy).sum(-1) / (
        (cx.square().sum(-1) + 1e-8) * (cy.square().sum(-1) + 1e-8)
    ).sqrt()
    # Correlation matrices use each pair's common valid times.
    common = valid[:, :, None, :] & valid[:, None, :, :]

    def pair_correlation(values: Tensor) -> tuple[Tensor, Tensor]:
        a, b = values[:, :, None, :], values[:, None, :, :]
        n = common.sum(-1).clamp_min(1)
        am = torch.where(common, a, 0).sum(-1) / n
        bm = torch.where(common, b, 0).sum(-1) / n
        da, db = (
            torch.where(common, a - am[..., None], 0),
            torch.where(common, b - bm[..., None], 0),
        )
        va, vb = da.square().sum(-1), db.square().sum(-1)
        corr = (da * db).sum(-1) / ((va + 1e-8) * (vb + 1e-8)).sqrt()
        return corr, (common.sum(-1) > 1) & (va > 1e-8) & (vb > 1e-8)

    corrx, _ = pair_correlation(x)
    corry, pair_valid = pair_correlation(y)
    # Only completely observed leads have unambiguous dense DWT coefficients.
    coeff_valid = valid.all(-1)
    detail, energy = [], []
    for index, (px, py) in enumerate(
        zip(
            predicted_coefficients.split(band_lengths, -1),
            target_coefficients.split(band_lengths, -1),
        )
    ):
        if index:
            detail.append(_masked_average((px - py).abs(), coeff_valid[..., None]))
        ex, ey = px.square().mean(-1), py.square().mean(-1)
        energy.append(
            _masked_average((torch.log(ex + 1e-8) - torch.log(ey + 1e-8)).abs(), coeff_valid)
        )
    control = x.flatten(1).sum(-1) * 0
    if control_target is not None:
        if control_prediction is None or control_valid is None:
            raise ValueError("provided controls need prediction and observed/interval mask")
        control = _masked_average(
            (control_prediction.float() - control_target.float()).abs(), control_valid
        )
    return {
        "reconstruction": _masked_average((x - y).abs(), valid),
        "mean": 0.5
        * (_masked_average((mx - my).abs(), lead_valid) + torch.stack(local_mean).mean(0)),
        "std": 0.5
        * (_masked_average((stdx - stdy).abs(), lead_valid) + torch.stack(local_std).mean(0)),
        "derivative": _masked_average(
            (x.diff() - y.diff()).abs(), valid[..., 1:] & valid[..., :-1]
        ),
        "pearson": _masked_average(1 - pearson.clamp(-1, 1), target_nonflat),
        "cross_correlation": _masked_average((corrx - corry).square(), pair_valid),
        "wavelet_detail": torch.stack(detail).mean(0),
        "wavelet_energy": torch.stack(energy).mean(0),
        "control": control,
    }


CONSTRAINT_WEIGHTS = {
    "reconstruction": 0.1,
    "mean": 0.05,
    "std": 0.05,
    "derivative": 0.05,
    "pearson": 0.05,
    "cross_correlation": 0.05,
    "wavelet_detail": 0.05,
    "wavelet_energy": 0.05,
    "control": 0.1,
}


def attribute_control_loss(
    waveform: Tensor,
    values: Tensor | None,
    valid: Tensor | None,
    schema: tuple[str, ...],
    sample_rate: float,
    source_scale: float = 1.0,
) -> Tensor:
    """Explicit window-global numeric controls in declared physical units.

    Numeric slots follow the saved attribute_schema. Band power is Hann-window
    rFFT energy per bin divided by window squared norm; fractions share the
    0.5Hz..min(80Hz,Nyquist) denominator. No diagnosis is inferred from controls.
    """
    zero = waveform.flatten(1).sum(-1) * 0
    if values is None or valid is None or not valid.any():
        return zero
    if values.shape != valid.shape or values.shape[1] > len(schema):
        raise ValueError("numeric controls must match the saved attribute schema/masks")
    x = waveform.float() * source_scale
    mean = x.mean(-1)
    centered = x - mean[..., None]
    window = torch.hann_window(x.shape[-1], device=x.device)
    power = torch.fft.rfft(centered * window).abs().square() / window.square().sum().clamp_min(1)
    hz = torch.fft.rfftfreq(x.shape[-1], 1 / sample_rate, device=x.device)
    support = (hz >= 0.5) & (hz <= min(80.0, sample_rate / 2))
    total = power[..., support].sum(-1).clamp_min(1e-8)
    estimates = {
        "spectral_centroid_hz": (power[..., support] * hz[support]).sum(-1) / total,
        "rms_uV": (x.square().mean(-1) + 1e-8).sqrt(),
        "std_uV": (centered.square().mean(-1) + 1e-8).sqrt(),
        "mean_uV": mean,
        "line_length_uV": x.diff().abs().mean(-1),
        "peak_abs_uV": x.abs().amax(-1),
    }
    for name, lower, upper in (
        ("delta", 0.5, 4),
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("gamma", 30, 80),
    ):
        band = (hz >= lower) & (hz < upper) & support
        energy = power[..., band].sum(-1)
        estimates[name + "_fraction"] = energy / total
        estimates[name + "_log_power_uV2"] = torch.log(energy + 1e-8)
    errors = []
    for i, name in enumerate(schema[: values.shape[1]]):
        if name not in estimates:
            if valid[:, i].any():
                raise ValueError(f"unsupported supplied numeric control: {name}")
            estimate = zero
        else:
            estimate = estimates[name].mean(-1)
        denominator = (
            sample_rate / 2
            if name.endswith("_hz")
            else source_scale
            if name.endswith("_uV")
            else 1.0
        )
        truth = torch.where(valid[:, i], values[:, i], 0)
        errors.append(torch.where(valid[:, i], (estimate - truth).abs() / denominator, 0))
    return torch.stack(errors, -1).sum(-1) / valid.sum(-1).clamp_min(1)
