"""Dataset input materialization for release inference profiles.

Only file names, sizes and index fields are used here; video content is never
inspected.  The official test set is a flat directory of numbered videos with no
external index, so the release inference layer materializes the manifest/index
rows it needs from directory metadata alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


class DatasetInputError(RuntimeError):
    """Raised when a profile's dataset inputs cannot be materialized."""


def validate_target_ratio(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DatasetInputError("target_ratio_wh must contain exactly two components")
    width, height = float(value[0]), float(value[1])
    if not (width > 0 and height > 0):
        raise DatasetInputError("target_ratio_wh components must be positive")
    return width, height


def scan_numbered_videos(video_root: Path) -> list[Path]:
    root = Path(video_root).expanduser()
    if not root.is_dir():
        raise DatasetInputError(f"video root does not exist: {root}")
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"]
    files.sort(key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem.lower()))
    if not files:
        raise DatasetInputError(f"no .mp4 videos found in {root}")
    return files


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def materialize_numbered_video_inputs(
    video_root: Path,
    input_dir: Path,
    target_ratio: Any,
    *,
    video_ids: Sequence[str] | None = None,
    clip_end_sec: float = 1.0e9,
) -> dict[str, Any]:
    """Build manifest/index rows from a flat numbered-video directory.

    ``clip_end_sec`` intentionally exceeds any real video: the retrieval runner
    probes the extracted clip and uses its true duration, so chunking remains
    driven by the frozen scientific config, not by this placeholder bound.
    """
    width, height = validate_target_ratio(target_ratio)
    files = scan_numbered_videos(video_root)
    if video_ids is not None:
        wanted = [str(value) for value in video_ids]
        by_id = {path.stem: path for path in files}
        missing = [value for value in wanted if value not in by_id]
        if missing:
            raise DatasetInputError(f"requested video ids are absent: {missing[:3]}")
        files = [by_id[value] for value in wanted]

    manifest_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for position, path in enumerate(files, start=1):
        video_id = path.stem
        manifest_rows.append(
            {
                "video_id": video_id,
                "relative_video_path": path.name,
                "split": "test",
                "clip_start_sec": 0.0,
                "clip_end_sec": float(clip_end_sec),
                "weak_reference_segments": [],
                "group": "official_test",
                "source_group": "official_test",
                "sample_index": position,
            }
        )
        index_rows.append(
            {
                "video_id": video_id,
                "video_path": path.name,
                "targetRatioWH": [width, height],
            }
        )
        records.append({"video_id": video_id})

    manifest_path = input_dir / "official_test_manifest.jsonl"
    index_path = input_dir / "official_test_index.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(index_path, index_rows)
    return {
        "manifest_path": manifest_path,
        "index_path": index_path,
        "records": records,
        "video_files": files,
        "video_count": len(files),
    }
