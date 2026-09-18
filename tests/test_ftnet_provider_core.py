from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import load_file

from aic_video_highlight.ftnet.materialize import SPLIT_DIRS, VideoRef, materialize_video
from aic_video_highlight.ftnet.native_schema import NATIVE_FIELDS
from aic_video_highlight.ftnet.provider_core import (
    ChunkSpan,
    DetectionArtifacts,
    IDX0_FALLBACK_NONE,
    IDX0_FALLBACK_NORM,
    MergedCandidate,
    RawCandidate,
    RetrievalContext,
    VideoArtifacts,
    apply_idx0_fallback,
    build_video_upstream,
)
from aic_video_highlight.ftnet.youtube_highlights import MturkClip

LENGTH = 10


def _ref() -> VideoRef:
    return VideoRef(
        canonical_video_id="vid-a",
        realized_video_id="vid-a",
        category="dog",
        stage7_split="TRAIN",
        relative_video_path="raw/dog/vid-a.mp4",
        source_sha256="a" * 64,
    )


def _artifacts(*, idx0_fallback: str = IDX0_FALLBACK_NONE) -> VideoArtifacts:
    timestamps = np.arange(LENGTH, dtype=np.float64) * 0.5
    frames = np.arange(LENGTH, dtype=np.int64)
    adjacency = np.ones(LENGTH, dtype=bool)
    adjacency[0] = False
    rng = np.random.default_rng(7)
    visual = rng.standard_normal((LENGTH, 256)).astype(np.float16)
    retrieval = RetrievalContext(
        chunks=(ChunkSpan(0, 0.0, 2.5), ChunkSpan(1, 2.5, 5.0)),
        raw_candidates=(
            RawCandidate(0, 0.5, 1.5, 0.9),
            RawCandidate(1, 3.0, 4.0, 0.8),
        ),
        merged_candidates=(
            MergedCandidate(0.5, 1.5, 0.9),
            MergedCandidate(3.0, 4.0, 0.8),
        ),
    )
    boxes = np.tile(
        np.array([[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]], dtype=np.float32),
        (LENGTH, 1),
    )
    scores = np.tile(np.array([0.9, 0.6], dtype=np.float32), LENGTH)
    labels = np.tile(np.array([1, 1], dtype=np.int32), LENGTH)
    offsets = np.arange(0, 2 * (LENGTH + 1), 2, dtype=np.int32)
    detections = DetectionArtifacts(
        offsets=offsets,
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=("__background__", "person"),
    )
    clips = (MturkClip(2, 6, 5.0), MturkClip(4, 8, 3.0))
    return VideoArtifacts(
        ref=_ref(),
        width=100,
        height=100,
        timestamps=timestamps,
        source_frame_ids=frames,
        adjacency_mask=adjacency,
        visual=visual,
        retrieval=retrieval,
        detections=detections,
        mturk_clips=clips,
        retrieval_metadata={},
    )


def test_provider_core_assembles_frozen_native_semantics(tmp_path) -> None:
    upstream = build_video_upstream(_artifacts())
    signals = upstream.native_signals
    assert tuple(signals) == NATIVE_FIELDS
    assert all(values.shape == (LENGTH,) for values in signals.values())

    assert signals["retrieval_support_ratio"].tolist() == pytest.approx(
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    )
    assert signals["candidate_time_from_start_log"][1] == pytest.approx(0.0)
    assert signals["candidate_time_from_start_log"][2] == pytest.approx(0.5)
    assert signals["candidate_time_to_end_log"][1] == pytest.approx(1.0)
    assert signals["candidate_duration_log"][2] == pytest.approx(1.0)
    assert signals["normalized_candidate_position"][2] == pytest.approx(0.5)
    assert np.isnan(signals["candidate_duration_log"][0])
    assert np.isnan(signals["candidate_duration_log"][5])

    assert signals["subject_track_present"].tolist() == pytest.approx([1.0] * LENGTH)
    assert signals["subject_selection_confidence"].tolist() == pytest.approx([0.9] * LENGTH)
    assert signals["subject_selection_margin"].tolist() == pytest.approx([0.3] * LENGTH)
    assert signals["bbox_area_ratio"].tolist() == pytest.approx([0.04] * LENGTH)
    assert signals["subject_frame_offset"][0] == pytest.approx(np.sqrt(0.18))
    assert signals["subject_scale_dynamics"][1] == pytest.approx(0.0)
    assert np.isnan(signals["subject_scale_dynamics"][0])
    assert signals["bbox_continuity_iou"][1] == pytest.approx(1.0)
    assert signals["bbox_continuity_iou"][0] == pytest.approx(0.0)
    assert signals["continuity_valid"].tolist() == pytest.approx(
        [0.0] + [1.0] * (LENGTH - 1)
    )
    assert signals["geometric_context_retention"][0] == pytest.approx(0.34 * 0.34)
    assert signals["stabilized_focus_velocity"].tolist() == pytest.approx([0.0] * LENGTH)
    assert signals["confidence_persistence"].tolist() == pytest.approx([1.0] * LENGTH)

    assert upstream.loss_mask.tolist() == [
        False, False, True, False, False, False, True, True, False, False
    ]
    assert upstream.candidate_missed_positive.tolist() == [
        False, False, False, True, True, True, False, False, False, False
    ]
    assert upstream.metadata["idx0_fallback"] == IDX0_FALLBACK_NONE


def test_provider_core_materializes_contract(tmp_path) -> None:
    upstream = build_video_upstream(_artifacts())
    record = materialize_video(upstream, tmp_path)
    tensors = load_file(str(tmp_path / SPLIT_DIRS["TRAIN"] / "vid-a.safetensors"))
    assert tensors["visual"].shape == (LENGTH, 256)
    assert tensors["native"].shape == (LENGTH, 16)
    assert np.isfinite(tensors["native"]).all()
    assert record["frame_count"] == LENGTH
    assert record["missed_positive_frames"] == 3
    # structural missing: candidate context absent outside merged candidates
    context_index = NATIVE_FIELDS.index("candidate_duration_log")
    assert tensors["native_missing"][0, context_index] == 1
    assert tensors["native_missing"][1, context_index] == 0


def test_idx0_fallback_is_pre_registered_and_deterministic() -> None:
    base = np.array([0.0, 0.5, 0.5, 0.25])
    assert apply_idx0_fallback(base, IDX0_FALLBACK_NONE).tolist() == base.tolist()
    assert apply_idx0_fallback(base, IDX0_FALLBACK_NORM).tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 0.5]
    )
    assert apply_idx0_fallback(np.zeros(3), IDX0_FALLBACK_NORM).tolist() == [0.0, 0.0, 0.0]
    upstream = build_video_upstream(_artifacts(), idx0_fallback=IDX0_FALLBACK_NORM)
    assert upstream.metadata["idx0_fallback"] == IDX0_FALLBACK_NORM
    assert upstream.native_signals["retrieval_support_ratio"].max() == pytest.approx(1.0)
