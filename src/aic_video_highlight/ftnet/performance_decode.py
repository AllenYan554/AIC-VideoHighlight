"""Bounded video decoding strategies for Stage 7 performance experiments.

Both strategies preserve the requested source-frame sequence exactly.  The
``windowed_prefix`` implementation intentionally reproduces the historical
engineering behaviour for baseline measurement, while ``one_pass`` keeps one
capture open and advances monotonically through the source video.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


WINDOWED_PREFIX = "windowed_prefix"
ONE_PASS = "one_pass"
DECODE_MODES = (WINDOWED_PREFIX, ONE_PASS)


@dataclass
class DecodeCounters:
    capture_open_count: int = 0
    grabbed_frame_count: int = 0
    retrieved_frame_count: int = 0
    emitted_frame_count: int = 0
    video_open_seconds: float = 0.0
    grab_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    decode_wall_seconds: float = 0.0
    repeated_prefix_decoded_frame_estimate: int = 0
    transcode_fallback_count: int = 0
    transcode_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _probe_decodable(video_path: Path) -> bool:
    """True when OpenCV can actually read a frame (open alone is not enough)."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return False
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


def _ensure_decodable(video_path: Path, counters: DecodeCounters) -> Path:
    """Return a path OpenCV can decode.

    Some benchmark sources are AV1, which this OpenCV/FFmpeg build cannot read.
    Mirror the retrieval layer's frozen compatibility path: transcode the
    offending source once with libx264/CRF18 into a sibling cache file, then
    decode from that copy.  Frame rate, frame count and timestamps are preserved
    by ffmpeg, so the requested source-frame sequence is unchanged.
    """

    if _probe_decodable(video_path):
        return video_path
    fallback = video_path.parent / f".decode-fallback-{video_path.stem}.mp4"
    if not fallback.is_file():
        started = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-an",
                str(fallback),
            ],
            check=True,
        )
        counters.transcode_seconds += time.perf_counter() - started
        counters.transcode_fallback_count += 1
    return fallback


def _validate_frame_ids(frame_ids: Sequence[int]) -> list[int]:
    requested = [int(value) for value in frame_ids]
    if not requested:
        return requested
    if requested[0] < 0 or requested != sorted(requested):
        raise ValueError("frame_ids must be non-negative and monotonically ordered")
    return requested


def _open_capture(video_path: Path, counters: DecodeCounters):
    import cv2

    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    counters.video_open_seconds += time.perf_counter() - started
    counters.capture_open_count += 1
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open {video_path}")
    return capture


def _read_prefix(
    video_path: Path,
    frame_ids: Sequence[int],
    counters: DecodeCounters,
) -> dict[int, np.ndarray]:
    import cv2

    needed = set(frame_ids)
    if not needed:
        return {}
    capture = _open_capture(video_path, counters)
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while index <= max(needed):
            started = time.perf_counter()
            ok = capture.grab()
            counters.grab_seconds += time.perf_counter() - started
            if not ok:
                break
            counters.grabbed_frame_count += 1
            if index in needed:
                started = time.perf_counter()
                retrieved, image = capture.retrieve()
                if retrieved:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                counters.retrieve_seconds += time.perf_counter() - started
                if not retrieved:
                    raise RuntimeError(f"frame {index} could not be retrieved from {video_path}")
                counters.retrieved_frame_count += 1
                frames[index] = image
            index += 1
    finally:
        capture.release()
    missing = needed - set(frames)
    if missing:
        raise RuntimeError(f"frames missing after decode of {video_path}: {sorted(missing)[:5]}")
    return frames


def iter_windowed_prefix_batches(
    video_path: str | Path,
    frame_ids: Sequence[int],
    *,
    batch_size: int,
    decode_window: int,
    counters: DecodeCounters | None = None,
) -> Iterator[tuple[list[int], list[np.ndarray]]]:
    """Historical baseline: reopen and decode from frame zero per window."""

    if batch_size <= 0 or decode_window <= 0:
        raise ValueError("batch_size and decode_window must be positive")
    requested = _validate_frame_ids(frame_ids)
    stats = counters if counters is not None else DecodeCounters()
    path = Path(video_path).expanduser().resolve()
    useful_single_pass = (max(requested) + 1) if requested else 0
    try:
        for position in range(0, len(requested), decode_window):
            window = requested[position : position + decode_window]
            decode_started = time.perf_counter()
            decoded = _read_prefix(path, window, stats)
            stats.decode_wall_seconds += time.perf_counter() - decode_started
            for batch_start in range(0, len(window), batch_size):
                batch_ids = window[batch_start : batch_start + batch_size]
                images = [decoded[frame_id] for frame_id in batch_ids]
                stats.emitted_frame_count += len(batch_ids)
                yield batch_ids, images
    finally:
        stats.repeated_prefix_decoded_frame_estimate = max(
            0, stats.grabbed_frame_count - useful_single_pass
        )


def iter_one_pass_batches(
    video_path: str | Path,
    frame_ids: Sequence[int],
    *,
    batch_size: int,
    counters: DecodeCounters | None = None,
) -> Iterator[tuple[list[int], list[np.ndarray]]]:
    """Open once, decode monotonically, and emit bounded ordered RGB batches."""

    import cv2

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    requested = _validate_frame_ids(frame_ids)
    if not requested:
        return
    stats = counters if counters is not None else DecodeCounters()
    path = Path(video_path).expanduser().resolve()
    capture = _open_capture(path, stats)
    requested_position = 0
    source_index = 0
    batch_ids: list[int] = []
    images: list[np.ndarray] = []
    try:
        while requested_position < len(requested):
            target = requested[requested_position]
            started = time.perf_counter()
            ok = capture.grab()
            stats.grab_seconds += time.perf_counter() - started
            if not ok:
                break
            stats.grabbed_frame_count += 1
            if source_index == target:
                started = time.perf_counter()
                retrieved, image = capture.retrieve()
                if retrieved:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                stats.retrieve_seconds += time.perf_counter() - started
                if not retrieved:
                    raise RuntimeError(
                        f"frame {source_index} could not be retrieved from {path}"
                    )
                stats.retrieved_frame_count += 1
                while (
                    requested_position < len(requested)
                    and requested[requested_position] == source_index
                ):
                    batch_ids.append(source_index)
                    images.append(image)
                    stats.emitted_frame_count += 1
                    requested_position += 1
                    if len(batch_ids) == batch_size:
                        yield batch_ids, images
                        batch_ids, images = [], []
            source_index += 1
        if requested_position != len(requested):
            missing = requested[requested_position:]
            raise RuntimeError(f"frames missing after decode of {path}: {missing[:5]}")
        if batch_ids:
            yield batch_ids, images
    finally:
        capture.release()
        stats.decode_wall_seconds = (
            stats.video_open_seconds + stats.grab_seconds + stats.retrieve_seconds
        )
        stats.repeated_prefix_decoded_frame_estimate = 0


def iter_decoded_batches(
    video_path: str | Path,
    frame_ids: Sequence[int],
    *,
    batch_size: int,
    decode_window: int,
    mode: str,
    counters: DecodeCounters | None = None,
) -> Iterator[tuple[list[int], list[np.ndarray]]]:
    if mode not in DECODE_MODES:
        raise ValueError(f"unknown decode mode {mode!r}; expected one of {DECODE_MODES}")
    stats = counters if counters is not None else DecodeCounters()
    decodable_path = _ensure_decodable(Path(video_path).expanduser().resolve(), stats)
    if mode == WINDOWED_PREFIX:
        yield from iter_windowed_prefix_batches(
            decodable_path,
            frame_ids,
            batch_size=batch_size,
            decode_window=decode_window,
            counters=stats,
        )
        return
    yield from iter_one_pass_batches(
        decodable_path,
        frame_ids,
        batch_size=batch_size,
        counters=stats,
    )


__all__ = [
    "DECODE_MODES",
    "ONE_PASS",
    "WINDOWED_PREFIX",
    "DecodeCounters",
    "iter_decoded_batches",
    "iter_one_pass_batches",
    "iter_windowed_prefix_batches",
]
