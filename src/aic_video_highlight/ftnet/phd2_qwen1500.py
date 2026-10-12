"""PHD2 Full1500 Qwen retrieval production run (retrieval only, no detection).

This module drives the frozen retrieval implementation over the frozen 1500-video
PHD2 raw manifest with:

* the promoted execution mode (``concurrency = 1`` + one-chunk render look-ahead),
* in-place chunk-level recovery (never a whole-video re-run),
* per-video atomic persistence and resume (completed videos are skipped),
* per-video failure isolation (one bad video never stops the run),
* progress.json / progress.jsonl with throughput and ETA, and
* an auditable manifest + summary of the run.

It never touches RT-DETR, materialization, LOC/CMP/TS, Native16 or supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .index import IndexEntry, resolve_video_path

RAW1500_SCHEMA = "aic.phd2.raw-1500-manifest/v1"
RUN_MANIFEST_SCHEMA = "aic.phd2.qwen1500-run-manifest/v1"
RUN_SUMMARY_SCHEMA = "aic.phd2.qwen1500-summary/v1"
STATUS_SCHEMA = "aic.phd2.qwen1500-status/v1"
RETRIEVAL_SCHEMA = "aic.stage7.ftnet.retrieval-artifact/v1"

QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROMPT_VERSION = "high_recall_retrieval_v0"

SYSTEMIC_ERROR_MARKERS = (
    "Connection refused",
    "connection refused",
    "ConnectionError",
    "ConnectError",
    "timed out",
    "health check",
)


class Qwen1500Error(RuntimeError):
    """Raised for run-level failures that must stop the run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return target


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def chunk_count_for_duration(duration_sec: float, chunk_seconds: float = 30.0, overlap_seconds: float = 5.0) -> int:
    if duration_sec <= 0:
        raise ValueError("duration must be positive")
    if duration_sec <= chunk_seconds:
        return 1
    step = chunk_seconds - overlap_seconds
    return int(math.ceil((duration_sec - chunk_seconds) / step)) + 1


@dataclass(frozen=True)
class RawRecord:
    video_id: str
    relative_path: str
    size_bytes: int
    duration_sec: float
    sha256: str
    arm: str
    duration_bin: str
    vcodec: str


def load_raw1500_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None,
    dataset_root: str | Path,
    verify_file_sizes: bool = True,
    expected_records: int = 1500,
) -> tuple[list[RawRecord], dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise Qwen1500Error(f"raw 1500 manifest not found: {path}")
    actual = sha256_file(path) if expected_sha256 else None
    if expected_sha256 and actual != expected_sha256:
        raise Qwen1500Error(
            f"raw 1500 manifest SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != RAW1500_SCHEMA:
        raise Qwen1500Error(f"unexpected raw manifest schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_records:
        raise Qwen1500Error(
            f"raw manifest must hold {expected_records} records, found {len(records or [])}"
        )
    ids = [str(row["youtubeId"]) for row in records]
    if len(set(ids)) != len(ids):
        raise Qwen1500Error("raw manifest contains duplicate youtubeId values")
    root = Path(dataset_root)
    parsed: list[RawRecord] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for row in records:
        relative = str(row["relative_path"]).replace("\\", "/")
        video_path = root / relative
        if not video_path.is_file():
            missing.append(relative)
            continue
        if verify_file_sizes and video_path.stat().st_size != int(row["actual_bytes"]):
            mismatched.append(relative)
            continue
        parsed.append(
            RawRecord(
                video_id=str(row["youtubeId"]),
                relative_path=relative,
                size_bytes=int(row["actual_bytes"]),
                duration_sec=float(row["actual_duration_s"]),
                sha256=str(row.get("sha256") or ""),
                arm=relative.split("/")[1] if "/" in relative else "unknown",
                duration_bin=str(row.get("duration_bin") or ""),
                vcodec=str(row.get("vcodec") or ""),
            )
        )
    if missing:
        raise Qwen1500Error(
            f"{len(missing)} raw videos missing under {root}; first: {missing[:3]}"
        )
    if mismatched:
        raise Qwen1500Error(
            f"{len(mismatched)} raw videos have size mismatch; first: {mismatched[:3]}"
        )
    meta = {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": actual,
        "schema": payload.get("schema"),
        "record_count": len(parsed),
        "total_bytes": sum(record.size_bytes for record in parsed),
        "dataset_root": str(root.resolve()),
        "raw_total_gib": payload.get("raw_total_gib"),
        "declared_total_videos": payload.get("total_videos"),
        "verified_file_sizes": verify_file_sizes,
    }
    return parsed, meta


def entries_from_records(
    records: Sequence[RawRecord],
    *,
    dataset_root: str | Path,
    dataset_id: str = ".",
    split: str = "TRAIN",
) -> list[IndexEntry]:
    """Build retrieval-only index entries without ffprobe/PTS preprocessing.

    The Qwen stage consumes raw video bytes plus the frozen chunking settings; it
    requires no frame grid, so the expensive per-video PTS extraction belongs to
    the (not yet authorized) detection stage.

    ``dataset_id`` must stay ``"."`` because ``resolve_video_path`` maps entries
    as ``dataset_root / dataset_id / relative_video_path`` and the stage-7
    production convention treats the passed root as the raw root itself.
    """

    root = Path(dataset_root)
    entries = []
    for record in records:
        entries.append(
            IndexEntry(
                dataset_id=dataset_id,
                video_id=record.video_id,
                realized_video_id=record.video_id,
                category=record.arm,
                split=split,
                relative_video_path=record.relative_path,
                source_sha256=record.sha256 or "0" * 64,
                annotation_identity="QWEN1500_RETRIEVAL_ONLY",
                duration_sec=record.duration_sec,
            )
        )
    del root
    return entries


def artifact_paths(
    root: str | Path, video_id: str
) -> tuple[Path, Path]:
    root = Path(root)
    return (
        root / "retrieval" / f"{video_id}.json",
        root / "raw_responses" / f"{video_id}.chunks.json",
    )


def validate_artifact(path: str | Path, video_id: str) -> tuple[bool, str | None]:
    target = Path(path)
    if not target.is_file():
        return False, "missing"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, f"unreadable: {type(exc).__name__}"
    if payload.get("schema") != RETRIEVAL_SCHEMA:
        return False, f"bad schema: {payload.get('schema')!r}"
    if payload.get("video_id") != video_id:
        return False, "foreign video_id"
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return False, "no chunks"
    for row in chunks:
        if not row.get("parse_success"):
            return False, f"chunk {row.get('chunk_index')} parse_success=false"
    if payload.get("merged_candidates") is None or payload.get("raw_candidates") is None:
        return False, "missing candidate lists"
    return True, None


@dataclass
class RunCounters:
    videos_total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    chunks: int = 0
    retries: int = 0
    expected_chunks: int = 0
    started_monotonic: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        throughput = self.chunks / self.elapsed_seconds if self.elapsed_seconds > 0 else None
        remaining = max(self.videos_total - self.completed, 0)
        eta = None
        if throughput and self.completed:
            per_video = self.elapsed_seconds / max(self.completed, 1)
            eta = per_video * remaining
        return {
            "schema": STATUS_SCHEMA,
            "videos_total": self.videos_total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "remaining": remaining,
            "chunks": self.chunks,
            "expected_chunks": self.expected_chunks,
            "retries": self.retries,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "elapsed_hours": round(self.elapsed_seconds / 3600.0, 4),
            "chunks_per_second": throughput,
            "video_seconds_per_video": (
                self.elapsed_seconds / self.completed if self.completed else None
            ),
            "eta_seconds": eta,
            "eta_hours": eta / 3600.0 if eta else None,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def is_systemic_error(message: str) -> bool:
    return any(marker in message for marker in SYSTEMIC_ERROR_MARKERS)


__all__ = [
    "PROMPT_VERSION",
    "QWEN_MODEL",
    "QWEN_REVISION",
    "RAW1500_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RUN_SUMMARY_SCHEMA",
    "STATUS_SCHEMA",
    "Qwen1500Error",
    "RawRecord",
    "RunCounters",
    "append_jsonl",
    "artifact_paths",
    "atomic_json",
    "chunk_count_for_duration",
    "entries_from_records",
    "is_systemic_error",
    "load_raw1500_manifest",
    "sha256_file",
    "validate_artifact",
]
