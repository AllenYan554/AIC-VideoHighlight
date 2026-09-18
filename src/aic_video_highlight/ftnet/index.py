"""Frozen, portable index for the YouTube Highlights FTNet dataset.

The index is the machine-readable identity of every video that Stage 7 is
allowed to materialize.  It is generated once from the frozen split manifest
plus the upstream raw manifest, then enriched on the compute host with ffprobe
metadata and a decode/sha audit.  It stores only dataset-relative paths, so the
same index maps onto the Windows warehouse root and the AutoDL dataset root.

This module never reads labels or ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .split_manifest import SPLIT_MANIFEST_NAME, YOUTUBE_HIGHLIGHTS_DIR

INDEX_SCHEMA = "aic.stage7.ftnet.frozen-index/v1"
DATASET_ID = "youtube_highlights"

DECODE_PENDING = "PENDING"
DECODE_OK = "OK"
DECODE_SHA_MISMATCH = "SHA_MISMATCH"
DECODE_PROBE_FAILED = "PROBE_FAILED"
DECODE_FAILED = "DECODE_FAILED"


class FrozenIndexError(RuntimeError):
    """Raised when the frozen index is missing, invalid or inconsistent."""


@dataclass(frozen=True)
class IndexEntry:
    dataset_id: str
    video_id: str
    realized_video_id: str
    category: str
    split: str
    relative_video_path: str
    source_sha256: str
    annotation_identity: str
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_sec: float | None = None
    fps: float | None = None
    fps_rational: str | None = None
    timestamp_mode: str | None = None
    pts_timestamps: tuple[str, ...] | None = None
    decode_status: str = DECODE_PENDING
    decode_error: str | None = None
    probe_source: str | None = None

    @property
    def split_dir_name(self) -> str:
        return {"TRAIN": "train", "VALIDATION": "validation", "CALIBRATION": "calibration"}[
            self.split
        ]


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest_payload(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FrozenIndexError(f"split manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aic.stage7.ftnet.split-manifest/v1":
        raise FrozenIndexError(f"unexpected split manifest schema: {manifest_path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FrozenIndexError("split manifest contains no entries")
    return payload


def build_index_entries(
    split_manifest_payload: Mapping[str, Any],
    *,
    dataset_id: str = DATASET_ID,
) -> tuple[list[IndexEntry], dict[str, Any]]:
    entries: list[IndexEntry] = []
    seen: set[str] = set()
    for raw in split_manifest_payload["entries"]:
        video_id = str(raw["canonical_video_id"])
        if video_id in seen:
            raise FrozenIndexError(f"duplicate canonical_video_id in split manifest: {video_id}")
        seen.add(video_id)
        relative_path = str(raw.get("relative_video_path") or "")
        if not relative_path or ".." in Path(relative_path).parts or Path(relative_path).is_absolute():
            raise FrozenIndexError(f"invalid relative_video_path for {video_id}: {relative_path!r}")
        source_sha256 = str(raw.get("source_sha256") or "")
        if len(source_sha256) != 64:
            raise FrozenIndexError(f"missing source SHA-256 for {video_id}")
        annotation_identity = str(raw.get("annotation_identity") or "")
        if not annotation_identity:
            raise FrozenIndexError(f"missing annotation_identity for {video_id}")
        entries.append(
            IndexEntry(
                dataset_id=dataset_id,
                video_id=video_id,
                realized_video_id=str(raw.get("realized_video_id") or video_id),
                category=str(raw["category"]),
                split=str(raw["stage7_split"]),
                relative_video_path=relative_path.replace("\\", "/"),
                source_sha256=source_sha256,
                annotation_identity=annotation_identity.replace("\\", "/"),
            )
        )
    entries.sort(key=lambda entry: (entry.split, entry.category, entry.video_id))
    metadata = {
        "dataset_id": dataset_id,
        "protocol_name": split_manifest_payload.get("protocol_name"),
        "protocol_version": split_manifest_payload.get("protocol_version"),
        "seed": split_manifest_payload.get("seed"),
        "total": len(entries),
        "counts": {
            split: sum(1 for entry in entries if entry.split == split)
            for split in ("TRAIN", "VALIDATION", "CALIBRATION")
        },
        "split_manifest_sha256": split_manifest_payload.get("manifest_sha256"),
        "split_content_sha256": split_manifest_payload.get("content_sha256"),
    }
    return entries, metadata


def index_payload(entries: Iterable[IndexEntry], *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    entry_list = [asdict(entry) for entry in entries]
    payload: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "dataset_id": metadata.get("dataset_id", DATASET_ID),
        "protocol_name": metadata.get("protocol_name"),
        "protocol_version": metadata.get("protocol_version"),
        "seed": metadata.get("seed"),
        "total": len(entry_list),
        "counts": {
            split: sum(1 for entry in entry_list if entry["split"] == split)
            for split in ("TRAIN", "VALIDATION", "CALIBRATION")
        },
        "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        "split_content_sha256": metadata.get("split_content_sha256"),
        "entries": entry_list,
    }
    return payload


def write_index(
    entries: Iterable[IndexEntry],
    *,
    metadata: Mapping[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path]:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = index_payload(entries, metadata=metadata)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    file_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{file_sha256}  {path.name}\n", encoding="utf-8", newline="\n")
    return path, sidecar


def load_index(path: str | Path) -> tuple[list[IndexEntry], dict[str, Any]]:
    index_path = Path(path).expanduser().resolve()
    if not index_path.is_file():
        raise FrozenIndexError(f"frozen index does not exist: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != INDEX_SCHEMA:
        raise FrozenIndexError(f"unexpected index schema: {index_path}")
    entries: list[IndexEntry] = []
    for raw in payload.get("entries", []):
        values = dict(raw)
        pts = values.get("pts_timestamps")
        if pts is not None:
            values["pts_timestamps"] = tuple(pts)
        entries.append(IndexEntry(**values))
    metadata = {key: value for key, value in payload.items() if key != "entries"}
    metadata["index_path"] = str(index_path)
    return entries, metadata


def entries_for_split(entries: Iterable[IndexEntry], split: str) -> list[IndexEntry]:
    if split not in ("TRAIN", "VALIDATION", "CALIBRATION"):
        raise FrozenIndexError(f"unknown Stage 7 split: {split}")
    return [entry for entry in entries if entry.split == split]


def resolve_video_path(dataset_root: str | Path, entry: IndexEntry) -> Path:
    """Map a dataset-relative index entry onto the host dataset root."""

    root = Path(dataset_root).expanduser().resolve()
    return root / entry.dataset_id / entry.relative_video_path


def resolve_annotation_dir(dataset_root: str | Path, entry: IndexEntry) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    return root / entry.dataset_id / "annotations" / "upstream_repo" / entry.annotation_identity


def probe_index_entry(
    entry: IndexEntry,
    dataset_root: str | Path,
    *,
    verify_sha256: bool = False,
    extract_pts: bool = True,
    ffprobe_bin: str = "ffprobe",
) -> IndexEntry:
    """Enrich one entry with real decode/probe status; never raises for data errors."""

    from aic_video_highlight.composition.frame_projection import PTS_TABLE
    from aic_video_highlight.composition.metadata import (
        VideoMetadataProbeError,
        extract_pts_timestamps,
        probe_spatial_meta,
    )

    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        return replace(
            entry,
            decode_status=DECODE_FAILED,
            decode_error=f"video file missing: {video_path}",
            probe_source="filesystem",
        )
    if verify_sha256:
        actual = _sha256_file(video_path)
        if actual != entry.source_sha256:
            return replace(
                entry,
                decode_status=DECODE_SHA_MISMATCH,
                decode_error=f"sha256 mismatch: expected {entry.source_sha256}, got {actual}",
                probe_source="sha256+ffprobe",
            )
    try:
        meta = probe_spatial_meta(
            video_path, video_id=entry.video_id, ffprobe_bin=ffprobe_bin
        )
    except (VideoMetadataProbeError, OSError) as exc:
        return replace(
            entry,
            decode_status=DECODE_PROBE_FAILED,
            decode_error=f"ffprobe failed: {type(exc).__name__}: {exc}",
            probe_source="ffprobe",
        )
    pts_serialized: tuple[str, ...] | None = None
    if meta.timestamp_mode == PTS_TABLE and extract_pts:
        try:
            pts = extract_pts_timestamps(video_path, ffprobe_bin=ffprobe_bin)
        except (VideoMetadataProbeError, OSError) as exc:
            return replace(
                entry,
                width=meta.width,
                height=meta.height,
                frame_count=meta.frame_count,
                duration_sec=meta.duration_sec,
                fps=meta.fps,
                fps_rational=meta.fps_rational,
                timestamp_mode=meta.timestamp_mode,
                decode_status=DECODE_PROBE_FAILED,
                decode_error=f"PTS extraction failed: {type(exc).__name__}: {exc}",
                probe_source="ffprobe",
            )
        pts_serialized = tuple(str(value) for value in pts)
    return replace(
        entry,
        width=meta.width,
        height=meta.height,
        frame_count=meta.frame_count,
        duration_sec=meta.duration_sec,
        fps=meta.fps,
        fps_rational=meta.fps_rational,
        timestamp_mode=meta.timestamp_mode,
        pts_timestamps=pts_serialized,
        decode_status=DECODE_OK,
        decode_error=None,
        probe_source="sha256+ffprobe" if verify_sha256 else "ffprobe",
    )


def audit_index(entries: Iterable[IndexEntry]) -> dict[str, Any]:
    entry_list = list(entries)
    counts = {
        split: sum(1 for entry in entry_list if entry.split == split)
        for split in ("TRAIN", "VALIDATION", "CALIBRATION")
    }
    statuses: dict[str, int] = {}
    for entry in entry_list:
        statuses[entry.decode_status] = statuses.get(entry.decode_status, 0) + 1
    return {
        "total": len(entry_list),
        "counts": counts,
        "decode_status": statuses,
        "all_ok": statuses.get(DECODE_OK, 0) == len(entry_list),
    }


def index_sha256(path: str | Path) -> str:
    return _sha256_file(Path(path).expanduser().resolve())


__all__ = [
    "DATASET_ID",
    "DECODE_FAILED",
    "DECODE_OK",
    "DECODE_PENDING",
    "DECODE_PROBE_FAILED",
    "DECODE_SHA_MISMATCH",
    "FrozenIndexError",
    "INDEX_SCHEMA",
    "IndexEntry",
    "audit_index",
    "build_index_entries",
    "entries_for_split",
    "index_payload",
    "index_sha256",
    "load_index",
    "load_split_manifest_payload",
    "probe_index_entry",
    "resolve_annotation_dir",
    "resolve_video_path",
    "write_index",
    "SPLIT_MANIFEST_NAME",
    "YOUTUBE_HIGHLIGHTS_DIR",
]
