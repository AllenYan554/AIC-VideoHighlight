"""Internal data structures for the Stage 1 coarse-recall pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoMeta:
    video_id: str
    path: Path
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class VideoChunk:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class HighlightSegment:
    start_sec: float
    end_sec: float
    score: float
    reason: str = ""
    source_chunk: int | None = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(slots=True)
class Stage1Result:
    """Internal intermediate result; not the competition prediction schema."""

    video_id: str
    duration_sec: float
    segments: list[HighlightSegment] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    inference_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "duration_sec": self.duration_sec,
            "segments": [asdict(segment) for segment in self.segments],
            "raw_responses": self.raw_responses,
            "inference_time_sec": self.inference_time_sec,
        }
