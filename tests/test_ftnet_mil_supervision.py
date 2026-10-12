from __future__ import annotations

import pytest
import torch

from aic_video_highlight.ftnet.mil_supervision import (
    MilBag,
    MilSupervisionError,
    aggregate_members,
    build_video_bags,
    mil_bag_loss,
)
from aic_video_highlight.ftnet.youtube_highlights import MturkClip


def _clips() -> tuple[MturkClip, ...]:
    return (
        MturkClip(start_frame=0, end_frame=6, vote=5.0),
        MturkClip(start_frame=4, end_frame=10, vote=1.0),
        MturkClip(start_frame=20, end_frame=30, vote=3.0),
        MturkClip(start_frame=40, end_frame=50, vote=4.0),
    )


def test_bags_follow_annotation_units_and_loss_mask() -> None:
    frame_ids = torch.arange(0, 12) * 2
    loss_mask = torch.ones(12, dtype=torch.bool)
    loss_mask[8:] = False

    bags = build_video_bags("v", frame_ids, loss_mask, _clips())

    assert len(bags) == 2
    assert bags[0].member_indices.tolist() == [0, 1, 2]
    assert bags[0].target == 1.0
    assert bags[1].member_indices.tolist() == [2, 3, 4]
    assert bags[1].target == pytest.approx(0.2)


def test_zero_member_bags_are_dropped_not_fabricated() -> None:
    frame_ids = torch.arange(0, 12) * 2
    loss_mask = torch.zeros(12, dtype=torch.bool)
    loss_mask[0:3] = True

    bags = build_video_bags("v", frame_ids, loss_mask, _clips())

    assert len(bags) == 2
    assert bags[0].member_indices.tolist() == [0, 1, 2]
    assert bags[1].member_indices.tolist() == [2]


def test_bag_members_never_cross_video_boundary() -> None:
    frame_ids = torch.tensor([0, 2, 4, 6, 8, 10])
    loss_mask = torch.ones(6, dtype=torch.bool)

    bags = build_video_bags("v", frame_ids, loss_mask, _clips())
    for bag in bags:
        assert bag.member_indices.max().item() < frame_ids.numel()
        assert bag.video_id == "v"


def test_aggregators_bound_and_topk_rule() -> None:
    values = torch.tensor([0.1, 0.2, 0.9, 0.4, 0.3, 0.5, 0.2])
    members = torch.arange(7)

    mean = aggregate_members(values, members, mode="mean")
    topk = aggregate_members(values, members, mode="top_k", top_k_fraction=0.25)
    lse = aggregate_members(values, members, mode="logsumexp", lse_tau=0.25)

    assert mean == pytest.approx(float(values.mean()))
    assert topk == pytest.approx(0.7)
    assert float(mean) < float(topk) <= 1.0
    assert float(mean) <= float(lse) <= float(values.max()) + 1e-6


def test_logsumexp_is_numerically_stable_for_zero_members() -> None:
    values = torch.tensor([0.0, 0.0, 0.0])
    members = torch.arange(3)

    lse = aggregate_members(values, members, mode="logsumexp")

    assert torch.isfinite(lse)
    assert float(lse) == pytest.approx(0.0)


def test_mil_loss_gradient_only_touches_bag_members() -> None:
    probabilities = torch.tensor(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], requires_grad=True
    )
    bags = (
        MilBag("v", 0, torch.tensor([0, 1]), 1.0),
        MilBag("v", 1, torch.tensor([4, 5]), 0.0),
    )

    loss = mil_bag_loss(probabilities, bags, mode="mean")
    loss.backward()

    grad = probabilities.grad
    assert grad is not None
    assert torch.equal(grad[:2] != 0, torch.tensor([True, True]))
    assert torch.equal(grad[4:] != 0, torch.tensor([True, True]))
    assert torch.equal(grad[2:4], torch.zeros(2))


def test_mil_loss_is_lower_when_members_match_bag_label() -> None:
    bags = (MilBag("v", 0, torch.tensor([0, 1]), 1.0),)
    high = torch.tensor([0.9, 0.9])
    low = torch.tensor([0.1, 0.1])

    assert float(mil_bag_loss(high, bags, mode="mean")) < float(
        mil_bag_loss(low, bags, mode="mean")
    )


def test_empty_bag_rejected() -> None:
    with pytest.raises(MilSupervisionError):
        MilBag("v", 0, torch.zeros(0, dtype=torch.int64), 0.5)
