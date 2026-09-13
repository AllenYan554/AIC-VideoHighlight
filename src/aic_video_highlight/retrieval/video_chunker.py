"""Create overlapping time windows without decoding video frames."""

from __future__ import annotations

from .schemas import VideoChunk


def build_chunks(
    duration_sec: float,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 5.0,
) -> list[VideoChunk]:
    """Split a positive duration into bounded overlapping chunks."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be greater than zero")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")

    chunks: list[VideoChunk] = []
    start_sec = 0.0
    step_sec = chunk_seconds - overlap_seconds

    while start_sec < duration_sec:
        end_sec = min(start_sec + chunk_seconds, duration_sec)
        chunks.append(
            VideoChunk(
                index=len(chunks),
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
            )
        )
        if end_sec >= duration_sec:
            break
        start_sec += step_sec

    return chunks
