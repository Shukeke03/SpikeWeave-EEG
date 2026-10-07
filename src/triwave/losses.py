"""Masked tri-modal alignment and interpretable signal regularizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from triwave.types import AlignmentOutput


def masked_mse(prediction: Tensor, target: Tensor, valid: Tensor | None = None,
               reduction: str = "mean") -> Tensor:
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


def contrastive_pair(left: Tensor, right: Tensor, left_valid: Tensor, right_valid: Tensor,
                     groups: Tensor | None = None, temperature: float = 0.07) -> Tensor:
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
        return (torch.logsumexp(scores, dim=1)
                - torch.logsumexp(positive_scores, dim=1)).mean()

    return (direction(logits, positives) + direction(logits.T, positives.T)) / 2


def alignment_loss(output: AlignmentOutput, text_groups: Tensor | None = None) -> Tensor:
    # Two signal views retain instance correspondence. Repeated clinical prompts
    # are multiple positives, otherwise identical text becomes a false negative.
    signal = contrastive_pair(output.eeg, output.structure, output.eeg_valid,
                              output.structure_valid)
    eeg_text = contrastive_pair(output.eeg, output.text, output.eeg_valid,
                                output.text_valid, groups=text_groups)
    structure_text = contrastive_pair(output.structure, output.text, output.structure_valid,
                                      output.text_valid, groups=text_groups)
    return (signal + eeg_text + structure_text) / 3


def signal_regularizers(prediction: Tensor, target: Tensor,
                        valid: Tensor | None = None, reduction: str = "mean") -> dict[str, Tensor]:
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
    psd = masked_mse(torch.log1p(prediction_power), torch.log1p(target_power),
                     lead_valid[..., None], reduction=reduction)

    def correlation(x: Tensor) -> Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-6)
        return normalized @ normalized.transpose(-1, -2)

    pair_valid = lead_valid[..., None] & lead_valid[:, None, :]
    topology = masked_mse(correlation(prediction), correlation(target), pair_valid,
                          reduction=reduction)
    return {"psd": psd, "topology": topology}
