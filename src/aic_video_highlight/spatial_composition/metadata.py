"""Per-video spatial metadata probing with deterministic caching."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .frame_projection import CFR_FPS, PTS_TABLE, VideoTiming

METADATA_SCHEMA = "aic.spatial-video-metadata/v1"


class VideoMetadataProbeError(RuntimeError):
    """Raised when ffprobe cannot provide usable spatial metadata."""


@dataclass(frozen=True, slots=True)
class VideoSpatialMeta:
    video_id: str
    video_path: str
    width: int
    height: int
    fps: float
    fps_rational: str
    frame_count: int
    duration_sec: float | None
    timestamp_mode: str
    rate_provenance: dict[str, Any]
    pts_timestamps: tuple[str, ...] | None = None


def _parse_rate(value: object) -> Fraction | None:
    if isinstance(value, str) and "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            denom = Fraction(denominator)
            numer = Fraction(numerator)
        except (ValueError, ZeroDivisionError):
            return None
        if denom <= 0:
            return None
        return numer / denom
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return Fraction(str(number)).limit_denominator(1_000_000)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return int(round(number))


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _run_ffprobe(command: list[str], path: Path, timeout_sec: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoMetadataProbeError(f"ffprobe binary not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoMetadataProbeError(
            f"ffprobe timed out after {timeout_sec:g}s: {path}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise VideoMetadataProbeError(f"ffprobe failed for {path}: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoMetadataProbeError(f"ffprobe returned invalid JSON for {path}") from exc


def extract_pts_timestamps(
    video_path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout_sec: float = 600.0,
) -> tuple[Fraction, ...]:
    """Canonical presentation timestamps for every frame, via packet scan."""
    path = Path(video_path).expanduser().resolve()
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "json",
        str(path),
    ]
    payload = _run_ffprobe(command, path, timeout_sec)
    frames = payload.get("frames") or []
    timestamps: list[Fraction] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        raw = frame.get("pts_time")
        if raw in (None, "N/A"):
            raise VideoMetadataProbeError(
                f"frame without pts_time in {path}; PTS_TABLE mode unavailable"
            )
        timestamps.append(Fraction(str(raw)))
    if not timestamps:
        raise VideoMetadataProbeError(f"ffprobe returned no frame timestamps for {path}")
    return tuple(timestamps)


def probe_spatial_meta(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    ffprobe_bin: str = "ffprobe",
    timeout_sec: float = 60.0,
) -> VideoSpatialMeta:
    """CPU-light metadata probe; never decodes pixel data."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = _run_ffprobe(command, path, timeout_sec)
    try:
        stream = payload["streams"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise VideoMetadataProbeError(f"ffprobe returned no video stream for {path}") from exc

    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    if width is None or height is None:
        raise VideoMetadataProbeError(f"invalid frame dimensions for {path}")

    r_rate = _parse_rate(stream.get("r_frame_rate"))
    avg_rate = _parse_rate(stream.get("avg_frame_rate"))
    selected_rate = avg_rate or r_rate
    nb_frames = _positive_int(stream.get("nb_frames"))
    duration = _positive_float(stream.get("duration"))
    if duration is None:
        duration = _positive_float(payload.get("format", {}).get("duration"))

    fps_rational_value = selected_rate
    if fps_rational_value is None and nb_frames is not None and duration is not None:
        fps_rational_value = Fraction(nb_frames) / Fraction(str(duration)).limit_denominator(
            1_000_000
        )
    if fps_rational_value is None:
        raise VideoMetadataProbeError(f"cannot determine frame rate for {path}")

    if r_rate is not None and avg_rate is not None and r_rate == avg_rate:
        timestamp_mode = CFR_FPS
    else:
        timestamp_mode = PTS_TABLE

    frame_count = nb_frames
    if frame_count is None and duration is not None:
        frame_count = int(round(duration * float(fps_rational_value)))
    if frame_count is None:
        raise VideoMetadataProbeError(f"cannot determine frame count for {path}")

    resolved_id = video_id or path.stem
    fps_value = float(fps_rational_value)
    provenance = {
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "selected": "avg_frame_rate" if avg_rate is not None else "r_frame_rate",
        "nb_frames_source": nb_frames is not None,
    }
    return VideoSpatialMeta(
        video_id=resolved_id,
        video_path=str(path),
        width=width,
        height=height,
        fps=fps_value,
        fps_rational=f"{fps_rational_value.numerator}/{fps_rational_value.denominator}",
        frame_count=frame_count,
        duration_sec=duration,
        timestamp_mode=timestamp_mode,
        rate_provenance=provenance,
        pts_timestamps=None,
    )


def timing_from_metadata(
    meta: VideoSpatialMeta, *, pts_timestamps: tuple[Fraction, ...] | None = None
) -> VideoTiming:
    pts = pts_timestamps
    if meta.timestamp_mode == PTS_TABLE and pts is None and meta.pts_timestamps:
        pts = tuple(Fraction(value) for value in meta.pts_timestamps)
    return VideoTiming(
        video_id=meta.video_id,
        fps=meta.fps,
        frame_count=meta.frame_count,
        timestamp_mode=meta.timestamp_mode,
        pts_timestamps=pts,
    )


def save_metadata_cache(
    records: Mapping[str, VideoSpatialMeta], path: str | Path
) -> None:
    payload = {
        "schema": METADATA_SCHEMA,
        "records": {video_id: asdict(meta) for video_id, meta in sorted(records.items())},
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_metadata_cache(path: str | Path) -> dict[str, VideoSpatialMeta]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != METADATA_SCHEMA:
        raise VideoMetadataProbeError(f"unknown metadata cache schema in {path}")
    records: dict[str, VideoSpatialMeta] = {}
    for video_id, raw in payload.get("records", {}).items():
        pts = raw.get("pts_timestamps")
        records[video_id] = VideoSpatialMeta(
            video_id=raw["video_id"],
            video_path=raw["video_path"],
            width=raw["width"],
            height=raw["height"],
            fps=raw["fps"],
            fps_rational=raw["fps_rational"],
            frame_count=raw["frame_count"],
            duration_sec=raw.get("duration_sec"),
            timestamp_mode=raw["timestamp_mode"],
            rate_provenance=raw.get("rate_provenance", {}),
            pts_timestamps=tuple(pts) if pts else None,
        )
    return records


def get_or_probe_metadata(
    video_id: str,
    video_path: str | Path,
    cache: dict[str, VideoSpatialMeta],
    *,
    ffprobe_bin: str = "ffprobe",
) -> VideoSpatialMeta:
    if video_id in cache:
        return cache[video_id]
    meta = probe_spatial_meta(video_path, video_id=video_id, ffprobe_bin=ffprobe_bin)
    cache[video_id] = meta
    return meta
