import pytest

from aic_video_highlight.spatial_composition.frame_projection import (
    VideoTiming,
    frame_timestamp,
    frames_in_segment,
    project_segments,
)


def cfr_timing(fps: float = 30.0, frame_count: int = 90) -> VideoTiming:
    return VideoTiming(
        video_id="synthetic",
        fps=fps,
        frame_count=frame_count,
        timestamp_mode="CFR_FPS",
    )


def pts_timing(
    pts: tuple[float, ...], frame_count: int | None = None
) -> VideoTiming:
    return VideoTiming(
        video_id="synthetic_pts",
        fps=30.0,
        frame_count=frame_count if frame_count is not None else len(pts),
        timestamp_mode="PTS_TABLE",
        pts_timestamps=pts,
    )


def test_segment_exactly_starting_on_frame_timestamp_includes_it() -> None:
    frames = frames_in_segment(1.0, 1.5, cfr_timing())

    assert frames[0] == 30
    assert frames == list(range(30, 45))


def test_segment_end_is_half_open_and_excludes_end_frame() -> None:
    frames = frames_in_segment(0.0, 1.0, cfr_timing())

    assert frames == list(range(0, 30))
    assert 30 not in frames


def test_adjacent_segments_share_no_frame() -> None:
    first = frames_in_segment(0.0, 1.0, cfr_timing())
    second = frames_in_segment(1.0, 2.0, cfr_timing())

    assert set(first).isdisjoint(second)
    projection = project_segments([(0.0, 1.0), (1.0, 2.0)], cfr_timing())
    assert list(projection.frames) == list(range(0, 60))


def test_overlapping_segments_are_unioned() -> None:
    projection = project_segments([(0.0, 2.0), (1.0, 3.0)], cfr_timing())

    assert list(projection.frames) == list(range(0, 90))


def test_duplicate_segments_deduplicate_to_sorted_unique_frames() -> None:
    projection = project_segments([(0.5, 1.0), (0.5, 1.0)], cfr_timing())

    assert list(projection.frames) == list(range(15, 30))
    assert len(projection.frames) == len(set(projection.frames))
    assert list(projection.frames) == sorted(projection.frames)


def test_first_frame_is_included_for_zero_start_segment() -> None:
    frames = frames_in_segment(0.0, 0.01, cfr_timing())

    assert frames == [0]


def test_final_frame_is_included_when_inside_segment() -> None:
    frames = frames_in_segment(2.9, 99.0, cfr_timing())

    assert frames == [87, 88, 89]


def test_segment_beyond_video_end_clips_by_default() -> None:
    frames = frames_in_segment(0.0, 120.0, cfr_timing())

    assert frames == list(range(0, 90))


def test_segment_beyond_video_end_raises_in_strict_mode() -> None:
    with pytest.raises(ValueError):
        frames_in_segment(0.0, 120.0, cfr_timing(), strict_beyond_end=True)


def test_fully_beyond_end_segment_is_empty_in_clip_mode() -> None:
    frames = frames_in_segment(95.0, 120.0, cfr_timing())

    assert frames == []


def test_empty_segments_project_to_no_frames() -> None:
    projection = project_segments([], cfr_timing())

    assert list(projection.frames) == []


def test_zero_and_inverted_duration_segments_are_empty() -> None:
    projection = project_segments([(1.0, 1.0), (2.0, 1.5)], cfr_timing())

    assert list(projection.frames) == []


def test_cfr_mapping_uses_exact_rational_membership() -> None:
    frames = frames_in_segment(1.0 / 3.0, 2.0 / 3.0, cfr_timing())

    assert frames == list(range(10, 20))


def test_non_integer_fps_membership_is_deterministic() -> None:
    timing = cfr_timing(fps=29.97, frame_count=90)
    first = project_segments([(0.0, 1.0)], timing)
    second = project_segments([(0.0, 1.0)], timing)

    assert first.frames == second.frames
    assert first.frames[0] == 0
    assert 30 not in first.frames


def test_pts_table_membership_is_half_open() -> None:
    timing = pts_timing((0.0, 0.5, 1.2, 2.0, 3.7))

    assert frames_in_segment(0.5, 2.0, timing) == [1, 2]
    assert frames_in_segment(2.0, 3.7, timing) == [3]
    assert frames_in_segment(0.0, 3.7, timing) == [0, 1, 2, 3]
    assert frames_in_segment(3.7, 9.9, timing) == [4]


def test_frame_timestamp_matches_membership_contract() -> None:
    timing = cfr_timing()

    assert frame_timestamp(30, timing) == pytest.approx(1.0)
    frames = frames_in_segment(0.0, 3.0, timing)
    assert all(0.0 <= frame_timestamp(f, timing) < 3.0 for f in frames)


def test_projection_clamps_to_frame_count() -> None:
    timing = cfr_timing(frame_count=45)

    projection = project_segments([(0.0, 5.0)], timing)

    assert list(projection.frames) == list(range(0, 45))
    assert projection.clipped_segment_count == 1


def test_invalid_timing_raises() -> None:
    with pytest.raises(ValueError):
        cfr_timing(fps=0.0)
    with pytest.raises(ValueError):
        cfr_timing(frame_count=0)
    with pytest.raises(ValueError):
        VideoTiming(
            video_id="x",
            fps=30.0,
            frame_count=10,
            timestamp_mode="PTS_TABLE",
            pts_timestamps=None,
        )
