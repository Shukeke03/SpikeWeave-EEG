
import torch

from triwave.alignment import (
    Relations,
    bidirectional_pair,
    fixed_alignment_loss,
    relations_from_metadata,
)
from triwave.types import AlignmentOutput


def test_positive_logs_are_averaged_instead_of_positive_set_logsumexp():
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    right = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], requires_grad=True)
    p = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    n = ~p
    rel = Relations(p, n, ["a", "b"], ["a", "a", "b"])
    loss, stats = bidirectional_pair(
        left, right, rel, torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool), 1.0
    )
    scores = left @ right.T
    forward = torch.stack(
        [
            torch.logsumexp(scores[0], 0) - scores[0, :2].mean(),
            torch.logsumexp(scores[1], 0) - scores[1, 2],
        ]
    ).mean()
    reverse = torch.stack(
        [
            ((torch.logsumexp(scores.T[:2], -1) - scores[0, :2]).mean()),
            torch.logsumexp(scores[:, 2], 0) - scores[1, 2],
        ]
    ).mean()
    torch.testing.assert_close(loss, (forward + reverse) / 2)
    assert stats["reverse_groups"] == 2
    loss.backward()
    assert torch.isfinite(left.grad).all()


def test_unknown_extreme_candidate_excluded_and_empty_reverse_skips_whole_item():
    left = torch.eye(2, requires_grad=True)
    right = torch.eye(3, 2, requires_grad=True)
    p = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
    n = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.bool)
    rel = Relations(p, n, ["a", "b"], ["a", "b", "unknown"])
    loss, _ = bidirectional_pair(
        left, right, rel, torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)
    )
    changed = right.detach().clone()
    changed[2] = 1e9
    other, _ = bidirectional_pair(
        left, changed, rel, torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)
    )
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert right.grad[2].abs().sum() == 0
    p = torch.tensor([[1, 0]], dtype=torch.bool)
    n = torch.tensor([[0, 1]], dtype=torch.bool)
    loss, stats = bidirectional_pair(
        left[:1],
        right[:2],
        Relations(p, n, ["a"], ["a", "b"]),
        torch.tensor([True]),
        torch.tensor([True, True]),
    )
    assert loss is None and stats["forward_anchors"] == 1 and stats["reverse_anchors"] == 0


def test_repeated_templates_deduplicated_and_unknown_labels_not_negative():
    records = [
        dict(
            instance_id=str(i),
            parent_id=str(i),
            patient_id=str(i),
            semantic_group=str(i // 2),
            namespace="annotated_event",
            scope="window",
            semantic_reliable=True,
            negative_eligible=True,
        )
        for i in range(4)
    ]
    x = torch.randn(4, 8, requires_grad=True)
    valid = torch.ones(4, dtype=torch.bool)
    output = AlignmentOutput(x, x, x, x, x, valid, valid, valid, valid)
    loss, stats = fixed_alignment_loss(output, records)
    assert stats["alignment_text_reverse_anchors"] == 2 and stats["alignment_valid_weight"] == 2.0
    assert stats["alignment_joint_positive_pairs"] == 8
    loss.backward()
    assert torch.isfinite(x.grad).all()
    records[0]["semantic_reliable"] = False
    rel = relations_from_metadata(records, "semantic")
    assert not (rel.positive[0] | rel.negative[0]).any()
    # The same clinical label is not a structural positive.
    structure = relations_from_metadata(records, "structure")
    assert not structure.positive[0, 1]


def test_no_valid_items_returns_differentiable_zero():
    x = torch.randn(2, 4, requires_grad=True)
    empty = torch.zeros(2, dtype=torch.bool)
    out = AlignmentOutput(x, x, x, x, x, empty, empty, empty, empty)
    loss, stats = fixed_alignment_loss(out, [{}, {}])
    assert loss == 0 and stats["alignment_valid_items"] == 0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
