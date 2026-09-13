"""Temporal segment to frame-index projection under the official contract."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

CFR_FPS = "CFR_FPS"
PTS_TABLE = "PTS_TABLE"


@dataclass(frozen=True, slots=True)
class VideoTiming:
    video_id: str
    fps: float
    frame_count: int
    timestamp_mode: str
    pts_timestamps: tuple[Fraction, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("fps must be a finite positive number")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
        ):
            raise ValueError("frame_count must be a positive integer")
        if self.timestamp_mode not in (CFR_FPS, PTS_TABLE):
            raise ValueError("timestamp_mode must be CFR_FPS or PTS_TABLE")
        if self.timestamp_mode == PTS_TABLE:
            pts = self.pts_timestamps
            if pts is None or len(pts) != self.frame_count:
                raise ValueError(
                    "PTS_TABLE timing requires one PTS entry per frame"
                )


@dataclass(frozen=True, slots=True)
class FrameProjection:
    frames: tuple[int, ...]
    per_segment_frames: tuple[tuple[int, ...], ...]
    clipped_segment_count: int


def _exact(value: int | float) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("segment bounds must be real numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("segment bounds must be finite")
    return Fraction(str(number))


def fps_rational(timing: VideoTiming) -> Fraction:
    return Fraction(str(timing.fps))


def frame_timestamp(frame: int, timing: VideoTiming) -> float:
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ValueError("frame must be an integer")
    if timing.timestamp_mode == CFR_FPS:
        if not 0 <= frame < timing.frame_count:
            raise ValueError("frame out of range")
        return frame / float(fps_rational(timing))
    pts = timing.pts_timestamps
    if pts is None or not 0 <= frame < len(pts):
        raise ValueError("frame out of range")
    return float(pts[frame])


def _raw_frame_range(
    start_sec: float, end_sec: float, timing: VideoTiming
) -> tuple[int, int] | None:
    if end_sec <= start_sec:
        return None
    if timing.timestamp_mode == CFR_FPS:
        rate = fps_rational(timing)
        lower = _exact(start_sec) * rate
        upper = _exact(end_sec) * rate
        first = math.ceil(lower)
        last = math.ceil(upper) - 1
    else:
        pts = timing.pts_timestamps
        if pts is None:
            raise ValueError("PTS_TABLE timing requires pts timestamps")
        first = bisect_left(pts, _exact(start_sec))
        last = bisect_left(pts, _exact(end_sec)) - 1
    return first, last


def _clamp_range(
    first: int, last: int, frame_count: int
) -> tuple[int, int] | None:
    first = max(first, 0)
    last = min(last, frame_count - 1)
    if first > last:
        return None
    return first, last


def frames_in_segment(
    start_sec: float,
    end_sec: float,
    timing: VideoTiming,
    *,
    strict_beyond_end: bool = False,
) -> list[int]:
    """Frames whose canonical timestamp lies in ``[start_sec, end_sec)``."""
    raw = _raw_frame_range(start_sec, end_sec, timing)
    if raw is None:
        return []
    first, last = raw
    if strict_beyond_end and last > timing.frame_count - 1:
        raise ValueError("segment extends beyond the final frame")
    clamped = _clamp_range(first, last, timing.frame_count)
    if clamped is None:
        return []
    return list(range(clamped[0], clamped[1] + 1))


def project_segments(
    segments: Iterable[tuple[float, float]],
    timing: VideoTiming,
    *,
    strict_beyond_end: bool = False,
) -> FrameProjection:
    """Deterministic segment union: deduplicated, ascending frame indices."""
    per_segment: list[tuple[int, ...]] = []
    collected: set[int] = set()
    clipped = 0
    for start_sec, end_sec in segments:
        raw = _raw_frame_range(start_sec, end_sec, timing)
        if raw is None:
            per_segment.append(())
            continue
        first, last = raw
        if last > timing.frame_count - 1:
            clipped += 1
            if strict_beyond_end:
                raise ValueError("segment extends beyond the final frame")
        clamped = _clamp_range(first, last, timing.frame_count)
        frames = (
            tuple(range(clamped[0], clamped[1] + 1)) if clamped is not None else ()
        )
        per_segment.append(frames)
        collected.update(frames)
    return FrameProjection(
        frames=tuple(sorted(collected)),
        per_segment_frames=tuple(per_segment),
        clipped_segment_count=clipped,
    )
