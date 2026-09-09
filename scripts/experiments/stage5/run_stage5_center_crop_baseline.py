"""Stage 5.1 center-crop baseline runner: frozen temporal -> official JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from aic_video_highlight.spatial_composition import (
    PTS_TABLE,
    VideoSpatialMeta,
    build_submission_record,
    compute_center_crop,
    extract_pts_timestamps,
    get_or_probe_metadata,
    load_metadata_cache,
    project_segments,
    save_metadata_cache,
    timing_from_metadata,
    validate_submission_file,
    write_predictions_jsonl,
)


def load_index(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        entries = json.loads(text)
    else:
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(entries, list):
        raise ValueError(f"index must be a JSON array or JSONL: {path}")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or "video_id" not in entry:
            raise ValueError(f"index entry missing video_id in {path}")
        if "targetRatioWH" not in entry:
            raise ValueError(
                f"index entry {entry.get('video_id')!r} missing targetRatioWH"
            )
        normalized.append(entry)
    return normalized


def load_frozen_segments(
    path: Path,
) -> dict[str, list[tuple[float, float]]]:
    segments: dict[str, list[tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = record.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise ValueError(f"replay line {line_number} missing video_id")
            merged = record.get("merged_prediction_segments", [])
            pairs: list[tuple[float, float]] = []
            for segment in merged:
                start_sec = segment.get("start_sec")
                end_sec = segment.get("end_sec")
                if start_sec is None or end_sec is None:
                    raise ValueError(
                        f"replay line {line_number} segment missing start/end"
                    )
                pairs.append((float(start_sec), float(end_sec)))
            segments[video_id] = pairs
    return segments


def index_video_path(
    entry: dict[str, Any], video_dir: Path | None
) -> Path | None:
    if video_dir is None:
        return None
    for key in ("video_path", "video", "file", "filename"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = video_dir / candidate
            if candidate.is_file():
                return candidate
    return video_dir / f"{entry['video_id']}.mp4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    index_entries = load_index(Path(args.index))
    if args.limit and args.limit > 0:
        index_entries = index_entries[: args.limit]
    segments = (
        load_frozen_segments(Path(args.segments)) if args.segments else {}
    )
    cache: dict[str, VideoSpatialMeta] = {}
    if args.metadata_cache and Path(args.metadata_cache).is_file():
        cache = load_metadata_cache(args.metadata_cache)

    records: list[dict[str, Any]] = []
    metadata_map: dict[str, VideoSpatialMeta] = {}
    temporal_map: dict[str, set[int]] = {}
    index_map: dict[str, tuple[Any, Any]] = {}
    warnings: list[str] = []
    total_predictions = 0

    for entry in index_entries:
        video_id = str(entry["video_id"])
        target_ratio = entry["targetRatioWH"]
        index_map[video_id] = (target_ratio[0], target_ratio[1])
        frozen = segments.get(video_id, [])
        video_path = index_video_path(entry, Path(args.video_dir) if args.video_dir else None)

        if video_path is None or (video_id not in cache and not video_path.is_file()):
            if args.allow_missing_video:
                warnings.append(f"no video for {video_id}; emitting empty predictions")
                records.append(
                    build_submission_record(
                        video_id=video_id,
                        target_ratio=target_ratio,
                        predictions=[],
                    )
                )
                temporal_map[video_id] = set()
                continue
            raise FileNotFoundError(
                f"video file missing for {video_id}: {video_path}"
            )

        meta = get_or_probe_metadata(
            video_id, video_path, cache, ffprobe_bin=args.ffprobe_bin
        )
        metadata_map[video_id] = meta
        pts = None
        if meta.timestamp_mode == PTS_TABLE:
            pts = tuple(
                Fraction(value) for value in (meta.pts_timestamps or ())
            )
            if not pts:
                if not args.pts_extract:
                    raise RuntimeError(
                        f"{video_id} is VFR; re-run with --pts-extract to build the PTS table"
                    )
                pts = extract_pts_timestamps(
                    meta.video_path, ffprobe_bin=args.ffprobe_bin
                )
        timing = timing_from_metadata(meta, pts_timestamps=pts)
        projection = project_segments(
            frozen, timing, strict_beyond_end=args.strict_beyond_end
        )
        temporal_map[video_id] = set(projection.frames)
        box = compute_center_crop(
            meta.width, meta.height, target_ratio[0], target_ratio[1]
        )
        predictions = [
            {"frame": frame, "bboxes": [box.x, box.y, box.w]}
            for frame in projection.frames
        ]
        total_predictions += len(predictions)
        records.append(
            build_submission_record(
                video_id=video_id,
                target_ratio=target_ratio,
                predictions=predictions,
                frame_size=(meta.width, meta.height),
                frame_count=meta.frame_count,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_jsonl(records, out_path)

    if args.metadata_cache:
        cache_path = Path(args.metadata_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_cache(cache, cache_path)

    report = validate_submission_file(
        out_path,
        index=index_map,
        metadata=metadata_map,
        temporal_frames=temporal_map,
    )

    summary = {
        "index_videos": len(index_entries),
        "predictions": total_predictions,
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "metadata_cache": str(args.metadata_cache) if args.metadata_cache else None,
        "validator": {
            "is_valid": report.is_valid,
            "stats": report.stats,
            "issue_count": len(report.issues),
            "first_issues": [
                {"line": issue.line_number, "video_id": issue.video_id, "code": issue.code, "detail": issue.detail}
                for issue in report.issues[:20]
            ],
        },
        "warnings": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report.is_valid else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-cache", default=None)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict-beyond-end", action="store_true")
    parser.add_argument("--pts-extract", action="store_true")
    parser.add_argument("--allow-missing-video", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
