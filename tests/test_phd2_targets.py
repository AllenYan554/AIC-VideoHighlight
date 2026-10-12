"""Tests for PHD2_GENERIC_PRETRAIN_TARGET_V1 (adapter / sampler / loss)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.phd2_targets import (
    BOUNDARY_GUARD_SEC,
    PAIR_SEED_DEFAULT,
    Selection,
    load_train_annotations,
    pair_audit,
    sample_pairs,
    timestep_masks,
)


def _grid(step: float = 0.5, count: int = 40) -> list[float]:
    return [round(i * step, 6) for i in range(count)]


def test_interval_half_open_alignment() -> None:
    selections = [Selection("v", 1.0, 2.0, "u1")]
    masks = timestep_masks([0.5, 1.0, 1.5, 2.0, 2.5], selections)
    # [start, end): 0.5 no, 1.0 yes, 1.5 yes, 2.0 NO (end excluded), 2.5 no
    assert masks["positive_mask"] == [False, True, True, False, False]


def test_seconds_to_2fps_timestep() -> None:
    selections = [Selection("v", 3.0, 4.0, "u1")]
    stamps = _grid()
    masks = timestep_masks(stamps, selections)
    positives = [s for s, flag in zip(stamps, masks["positive_mask"]) if flag]
    assert positives == [3.0, 3.5]


def test_multi_user_union() -> None:
    selections = [
        Selection("v", 1.0, 2.0, "u1"),
        Selection("v", 3.0, 4.0, "u2"),
    ]
    masks = timestep_masks([1.5, 2.5, 3.5], selections)
    assert masks["positive_mask"] == [True, False, True]


def test_support_count_distinct_users() -> None:
    selections = [
        Selection("v", 1.0, 3.0, "u1"),
        Selection("v", 1.5, 3.5, "u2"),
        Selection("v", 1.5, 2.0, "u1"),  # same user twice -> still 1
    ]
    masks = timestep_masks([1.0, 2.0, 3.0], selections)
    assert masks["selection_support_count"] == [1, 2, 1]


def test_unselected_is_not_negative() -> None:
    selections = [Selection("v", 1.0, 1.5, "u1")]
    masks = timestep_masks([0.0, 1.0, 5.0], selections)
    assert masks["unlabeled_mask"] == [True, False, True]
    # no Y=0 field is ever produced by this adapter
    assert "y" not in masks and "negative_mask" not in masks


def test_boundary_guard_outside_neighbourhood() -> None:
    selections = [Selection("v", 2.0, 3.0, "u1")]
    stamps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    masks = timestep_masks(stamps, selections, boundary_guard_sec=0.5)
    # 1.5 in [start-0.5, start) -> guard ; 3.0 in [end, end+0.5) -> guard
    assert masks["boundary_guard_mask"] == [False, True, False, False, True, False, False]
    assert masks["legal_unlabeled_mask"] == [True, False, False, False, False, True, True]


def test_pair_sampling_deterministic() -> None:
    selections = [Selection("v", 2.0, 3.0, "u1")]
    stamps = _grid(count=20)
    masks = timestep_masks(stamps, selections)
    a = sample_pairs(stamps, masks, video_id="vidA", seed=PAIR_SEED_DEFAULT)
    b = sample_pairs(stamps, masks, video_id="vidA", seed=PAIR_SEED_DEFAULT)
    c = sample_pairs(stamps, masks, video_id="vidA", seed=PAIR_SEED_DEFAULT + 1)
    assert a == b
    assert isinstance(c, list)


def test_pair_sampling_one_to_one() -> None:
    selections = [Selection("v", 2.0, 3.0, "u1")]
    stamps = _grid(count=20)
    masks = timestep_masks(stamps, selections)
    pairs = sample_pairs(stamps, masks, video_id="vidA")
    audit = pair_audit(pairs, n=len(stamps))
    assert audit["one_to_one"]
    assert audit["unique_unlabeled"] == audit["pairs"]


def test_no_valid_unlabeled_skips() -> None:
    # the only unlabeled timestep sits inside the boundary guard
    stamps = [0.5, 1.0]
    selections = [Selection("v", 1.0, 1.4, "u1")]  # 0.5 in [start-0.5, start) -> guarded
    masks = timestep_masks(stamps, selections)
    assert masks["positive_mask"] == [False, True]
    assert masks["legal_unlabeled_mask"] == [False, False]
    pairs = sample_pairs(stamps, masks, video_id="v")
    assert pairs == []


def test_test_annotation_never_loaded(tmp_path: Path) -> None:
    train = tmp_path / "training.csv"
    with train.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["youtubeId", "start", "duration", "user_id", "video_duration", "is_last"])
        writer.writerow(["vid1", "0.0", "1.0", "u1", "10.0", "False"])
    testing = tmp_path / "testing.csv"
    testing.write_text("should not be read", encoding="utf-8")
    index = load_train_annotations(train, require_sha=False)
    assert index.unique_videos == 1
    assert index.source_sha256 == load_train_annotations(train, require_sha=False).source_sha256
    assert list(index.selections) == ["vid1"]


def test_pairwise_loss_numeric() -> None:
    torch = pytest.importorskip("torch")
    from aic_video_highlight.ftnet.phd2_targets import pairwise_ranking_loss

    z_pos = torch.tensor([2.0, 0.0, -1.0])
    z_u = torch.tensor([0.0, 0.0, 1.0])
    loss = pairwise_ranking_loss(z_pos, z_u)
    expected = math.log(1 + math.exp(-2.0)) + math.log(2) + math.log(1 + math.exp(2.0))
    expected /= 3.0
    assert abs(float(loss) - expected) < 1e-5


def test_no_hard_negative_masks() -> None:
    selections = [Selection("v", 1.0, 1.5, "u1")]
    masks = timestep_masks(_grid(), selections)
    assert set(masks) == {
        "positive_mask",
        "unlabeled_mask",
        "boundary_guard_mask",
        "selection_support_count",
        "legal_unlabeled_mask",
    }


def test_no_cross_video_pair() -> None:
    selections = [Selection("v", 2.0, 3.0, "u1")]
    stamps = _grid(count=20)
    masks = timestep_masks(stamps, selections)
    pairs = sample_pairs(stamps, masks, video_id="vidA")
    assert all(p["video_id"] == "vidA" for p in pairs)


def test_no_invalid_adjacency_pair() -> None:
    selections = [Selection("v", 2.0, 3.0, "u1")]
    stamps = _grid(count=20)
    masks = timestep_masks(stamps, selections)
    adjacency = [False] * len(stamps)
    adjacency[0] = True  # only index 0 is a valid unlabeled neighbor
    pairs = sample_pairs(stamps, masks, video_id="v", adjacency_mask=adjacency)
    assert pairs
    assert all(p["unlabeled_index"] == 0 for p in pairs)


def test_pair_index_schema_finite() -> None:
    selections = [Selection("v", 1.0, 2.0, "u1")]
    stamps = _grid(count=20)
    masks = timestep_masks(stamps, selections)
    pairs = sample_pairs(stamps, masks, video_id="v")
    for pair in pairs:
        assert math.isfinite(pair["positive_timestamp"])
        assert math.isfinite(pair["unlabeled_timestamp"])
        assert 0 <= pair["positive_index"] < len(stamps)
        assert 0 <= pair["unlabeled_index"] < len(stamps)


def test_boundary_guard_constant_is_half_second() -> None:
    assert BOUNDARY_GUARD_SEC == 0.5
