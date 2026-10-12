"""Uniform 2.0 fps source-time sampling for Stage 7 FTNet sequences.

The frozen preprocessing protocol (YouTube Highlights Preprocessing Protocol
v1, sections 2-3) fixes:

    sampling_mode   = uniform 2.0 fps in source time
    sample_period_s = 0.5
    frame_selection = nearest decoded source frame to each 0.5s grid point
    frame_timestamp = source PTS in seconds (float64)
    source_frame_id = decoded container frame index (int64)
    missing_frame   = keep the grid; invalid frames get adjacency=0

``adjacency_mask[t]`` is true iff source frames ``t-1`` and ``t`` are truly
adjacent (container index difference exactly one).  Duplicate grid points on
very low-fps sources and container gaps both produce ``False``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from aic_video_highlight.composition.frame_projection import CFR_FPS, VideoTiming, frame_timestamp

SAMPLE_FPS = 2.0
SAMPLE_PERIOD_SEC = 0.5
ADJACENCY_RULE = (
    "strictly increasing source frame id and 0 < dt <= 1.5 * sample_period; "
    "duplicate samples, decode gaps and video boundaries give adjacency=0"
)


class SamplingError(ValueError):
    """Raised when a source timing cannot produce a valid sampling grid."""


@dataclass(frozen=True)
class GridSample:
    frames: np.ndarray
    timestamps: np.ndarray
    adjacency_mask: np.ndarray
    target_timestamps: np.ndarray


def _nearest_cfr_frame(target: Fraction, timing: VideoTiming) -> int:
    rate = Fraction(str(timing.fps))
    position = target * rate
    nearest = math.floor(position + Fraction(1, 2))
    return max(0, min(int(nearest), timing.frame_count - 1))


def _nearest_pts_frame(target: Fraction, timing: VideoTiming) -> int:
    pts = timing.pts_timestamps
    if pts is None:
        raise SamplingError("PTS_TABLE timing requires pts timestamps")
    index = bisect_left(pts, target)
    if index <= 0:
        return 0
    if index >= len(pts):
        return len(pts) - 1
    before = pts[index - 1]
    after = pts[index]
    return index - 1 if (target - before) <= (after - target) else index


def build_uniform_grid(
    timing: VideoTiming,
    *,
    sample_fps: float = SAMPLE_FPS,
) -> GridSample:
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise SamplingError("sample_fps must be a finite positive number")
    if timing.frame_count <= 0:
        raise SamplingError("frame_count must be positive")
    period = Fraction(1, 1) / Fraction(str(sample_fps))
    frame_count = timing.frame_count
    last_timestamp = frame_timestamp(frame_count - 1, timing)
    frames: list[int] = []
    timestamps: list[float] = []
    target_times: list[float] = []
    grid_index = 0
    while True:
        target = period * grid_index
        if float(target) > last_timestamp + 1e-9:
            break
        frame = (
            _nearest_cfr_frame(target, timing)
            if timing.timestamp_mode == CFR_FPS
            else _nearest_pts_frame(target, timing)
        )
        frames.append(frame)
        timestamps.append(frame_timestamp(frame, timing))
        target_times.append(float(target))
        grid_index += 1
        if grid_index > 1_000_000:
            raise SamplingError("sampling grid exceeded one million frames")
    frame_array = np.asarray(frames, dtype=np.int64)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    adjacency = np.zeros(frame_array.shape[0], dtype=bool)
    max_dt = 1.5 * float(period)
    if frame_array.shape[0] > 1:
        frame_gap = frame_array[1:] > frame_array[:-1]
        dt = np.diff(timestamp_array)
        adjacency[1:] = frame_gap & (dt > 0.0) & (dt <= max_dt)
    return GridSample(
        frames=frame_array,
        timestamps=timestamp_array,
        adjacency_mask=adjacency,
        target_timestamps=np.asarray(target_times, dtype=np.float64),
    )


__all__ = [
    "ADJACENCY_RULE",
    "GridSample",
    "SAMPLE_FPS",
    "SAMPLE_PERIOD_SEC",
    "SamplingError",
    "build_uniform_grid",
]
