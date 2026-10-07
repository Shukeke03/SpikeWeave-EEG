"""Fixed two-branch P/N/U, group-balanced, multi-positive alignment.

Relations come from audited metadata, never from batch position or unmentioned
clinical findings. Unknown pairs are absent from the softmax denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from triwave.types import AlignmentOutput


@dataclass
class Relations:
    positive: Tensor
    negative: Tensor
    left_groups: list[str]
    right_groups: list[str]

    def validate(self, rows: int, columns: int) -> None:
        if self.positive.shape != (rows, columns) or self.negative.shape != (rows, columns):
            raise ValueError("relation matrices must match anchor/candidate dimensions")
        if self.positive.dtype != torch.bool or self.negative.dtype != torch.bool:
            raise TypeError("P/N matrices must be boolean")
        if (self.positive & self.negative).any():
            raise ValueError("a pair cannot be both positive and negative")
        if len(self.left_groups) != rows or len(self.right_groups) != columns:
            raise ValueError("relations require independent forward/reverse groups")


def relations_from_metadata(records: list[dict], head: str, device=None) -> Relations:
    if head not in {"semantic", "structure"}:
        raise ValueError("unknown alignment head")
    n = len(records)
    p = torch.zeros(n, n, dtype=torch.bool, device=device)
    neg = torch.zeros_like(p)
    key = "semantic_group" if head == "semantic" else "instance_id"
    groups = [str(row.get(key, f"unknown:{i}")) for i, row in enumerate(records)]
    for i, left in enumerate(records):
        for j, right in enumerate(records):
            if head == "semantic":
                reliable = left.get("semantic_reliable", False) and right.get(
                    "semantic_reliable", False
                )
                scope_equal = (
                    left.get("scope") is not None
                    and left.get("scope") == right.get("scope")
                    and left.get("namespace") == right.get("namespace")
                )
                if reliable and scope_equal:
                    p[i, j] = groups[i] == groups[j]
                    neg[i, j] = (
                        groups[i] != groups[j]
                        and left.get("negative_eligible", False)
                        and right.get("negative_eligible", False)
                    )
            else:
                same = left.get("instance_id") is not None and groups[i] == groups[j]
                p[i, j] = same
                if not same:
                    # Be conservative: same-parent views, shared events and same-patient
                    # near neighbours are unknown, regardless of clinical class.
                    different_patient = (
                        left.get("patient_id") is not None
                        and right.get("patient_id") is not None
                        and left["patient_id"] != right["patient_id"]
                    )
                    different_parent = (
                        left.get("parent_id") is not None
                        and right.get("parent_id") is not None
                        and left["parent_id"] != right["parent_id"]
                    )
                    shared_event = bool(
                        set(left.get("event_ids", [])) & set(right.get("event_ids", []))
                    )
                    neg[i, j] = different_patient and different_parent and not shared_event
    return Relations(p, neg, groups, groups.copy())


def _direction(
    scores: Tensor, p: Tensor, n: Tensor, groups: list[str]
) -> tuple[Tensor | None, int, int]:
    eligible = p.any(-1) & n.any(-1)
    if not eligible.any():
        return None, 0, 0
    # Slice before logsumexp: all-unknown rows must not create inf/NaN gradients.
    selected = torch.nonzero(eligible).flatten()
    ps, ns, logits = p[selected], n[selected], scores[selected]
    denominator = torch.logsumexp(logits.masked_fill(~(ps | ns), -torch.inf), -1)
    positive_mean = torch.where(ps, logits, 0).sum(-1) / ps.sum(-1)
    losses = denominator - positive_mean
    index_groups = [groups[int(index)] for index in selected]
    group_losses = [
        losses[torch.tensor([g == group for g in index_groups], device=losses.device)].mean()
        for group in sorted(set(index_groups))
    ]
    return torch.stack(group_losses).mean(), len(selected), len(group_losses)


def bidirectional_pair(
    left: Tensor,
    right: Tensor,
    relations: Relations,
    left_valid: Tensor,
    right_valid: Tensor,
    temperature: float = 0.07,
) -> tuple[Tensor | None, dict]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    relations.validate(len(left), len(right))
    present = left_valid[:, None].bool() & right_valid[None, :].bool()
    p = relations.positive.to(left.device) & present
    n = relations.negative.to(left.device) & present
    logits = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T / temperature
    forward, nf, gf = _direction(logits, p, n, relations.left_groups)
    reverse, nr, gr = _direction(logits.T, p.T, n.T, relations.right_groups)
    metrics = {
        "forward_anchors": nf,
        "reverse_anchors": nr,
        "forward_groups": gf,
        "reverse_groups": gr,
        "positive_pairs": int(p.sum()),
        "negative_pairs": int(n.sum()),
        "unknown_pairs": int((~(p | n)).sum()),
        "temperature": temperature,
        "skipped": int(forward is None or reverse is None),
    }
    if forward is None or reverse is None:
        return None, metrics
    return (forward + reverse) / 2, metrics


def _candidate_indices(records: list[dict], mode: str, available: Tensor) -> list[int]:
    seen, indices = set(), []
    for i, row in enumerate(records):
        if not bool(available[i]):
            continue
        # Unknown descriptions must not be accidentally merged.
        group = row.get("semantic_group", f"unknown:{i}")
        instance = row.get("instance_id", f"unknown:{i}")
        key = group if mode == "text" else (group, instance) if mode == "joint" else instance
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return indices


def fixed_alignment_loss(
    output: AlignmentOutput, records: list[dict] | None, temperature: float = 0.07
) -> tuple[Tensor, dict]:
    zero = (
        sum(
            value.sum()
            for value in (output.eeg, output.eeg_structure, output.joint, output.text, output.image)
        )
        * 0
    )
    if records is None:
        return zero, {"alignment_valid_items": 0, "alignment_missing_metadata": 1}
    if len(records) != len(output.eeg):
        raise ValueError("one audited alignment metadata row is required per EEG")
    sem = relations_from_metadata(records, "semantic", output.eeg.device)
    struct = relations_from_metadata(records, "structure", output.eeg.device)
    total, weight_sum, metrics = zero, 0.0, {}
    terms = (
        ("joint", output.eeg, output.joint, output.joint_valid, sem, 1.0),
        ("text", output.eeg, output.text, output.text_valid, sem, 0.5),
        ("image", output.eeg_structure, output.image, output.image_valid, struct, 0.5),
    )
    for mode, eeg, condition, valid, rel, weight in terms:
        indices = _candidate_indices(records, mode, valid)
        selected = torch.tensor(indices, device=eeg.device, dtype=torch.long)
        subset = Relations(
            rel.positive[:, selected],
            rel.negative[:, selected],
            rel.left_groups,
            [rel.right_groups[i] for i in indices],
        )
        item, stats = bidirectional_pair(
            eeg, condition[selected], subset, output.eeg_valid, valid[selected], temperature
        )
        metrics.update({f"alignment_{mode}_{key}": value for key, value in stats.items()})
        if item is not None:
            total, weight_sum = total + weight * item, weight_sum + weight
            metrics[f"alignment_{mode}_loss"] = float(item.detach())
    metrics["alignment_valid_items"] = sum(
        metrics[f"alignment_{r}_skipped"] == 0 for r in ("joint", "text", "image")
    )
    metrics["alignment_valid_weight"] = weight_sum
    return (total / weight_sum if weight_sum else zero), metrics
