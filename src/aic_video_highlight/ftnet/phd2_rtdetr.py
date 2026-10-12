"""PHD2 RT-DETR full-cohort detection stage (detection + Visual256 only).

This stage consumes the frozen PHD2 raw manifest plus the Qwen retrieval
artifacts and produces, for every eligible video:

    grid/<video_id>.npz        the frozen full-video uniform 2.0 fps grid identity
    detections/<video_id>.npz  detection candidates + encoder level-0 GAP (Visual256)

Frozen promoted configuration (validated by the upstream performance experiment;
no sweep is re-run here):

    model            PekingU/rtdetr_r50vd, task-level load once
    sampling         full-video uniform 2.0 fps
    decode           one_pass monotonic decode (bounded memory)
    detection batch  8   (the only correctness-passing configuration)
    top_k            100
    feature          encoder level-0 GAP -> Visual256 (float16)

Videos whose Qwen retrieval artifact is missing/invalid are excluded from the
derived eligibility cohort and are never fabricated (no empty candidates, no
negative relabeling).  Every stage is per-video atomic and resume-safe; one bad
video never aborts the cohort.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .index import (
    DECODE_OK,
    INDEX_SCHEMA,
    IndexEntry,
    audit_index,
    load_index,
    probe_index_entry,
    resolve_video_path,
    write_index,
)
from .performance_decode import ONE_PASS
from .phd2_qwen1500 import (
    RawRecord,
    append_jsonl,
    atomic_json,
    load_raw1500_manifest,
    sha256_file,
    validate_artifact,
)
from .real_provider import (
    DETECTION_SCHEMA,
    RTDETR_MODEL,
    RTDETR_REVISION,
    RealProviderError,
    create_rtdetr_localizer,
    detection_artifact_path,
    grid_artifact_path,
    run_detection_for_entry,
)

COHORT_SCHEMA = "aic.phd2.qwen-success-cohort/v1"
FAILURE_AUDIT_SCHEMA = "aic.phd2.qwen-failure-audit/v1"
GRID_EXPECTATIONS_SCHEMA = "aic.phd2.grid-expectations/v1"
DETECTION_RUN_MANIFEST_SCHEMA = "aic.phd2.rtdetr-run-manifest/v1"
DETECTION_RUN_SUMMARY_SCHEMA = "aic.phd2.rtdetr-summary/v1"
DETECTION_STATUS_SCHEMA = "aic.phd2.rtdetr-status/v1"

FROZEN_BATCH_SIZE = 8
FROZEN_TOP_K = 100
FROZEN_DECODE_MODE = ONE_PASS

SYSTEMIC_ERROR_MARKERS = (
    "CUDA out of memory",
    "CUDA error",
    "no CUDA-capable device",
    "device-side assert",
    "out of memory",
)


class Phd2RtdetrError(RuntimeError):
    """Raised for cohort-level failures that must stop the stage."""


def is_systemic_detection_error(message: str) -> bool:
    return any(marker in message for marker in SYSTEMIC_ERROR_MARKERS)


def build_success_cohort(
    records: Sequence[RawRecord],
    *,
    retrieval_root: str | Path,
) -> tuple[list[RawRecord], list[dict[str, Any]]]:
    """Split the frozen cohort into Qwen-success and QWEN_RETRIEVAL_UNAVAILABLE."""

    root = Path(retrieval_root)
    eligible: list[RawRecord] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        artifact = root / "retrieval" / f"{record.video_id}.json"
        ok, reason = validate_artifact(artifact, record.video_id)
        if ok:
            eligible.append(record)
        else:
            excluded.append(
                {
                    "video_id": record.video_id,
                    "relative_path": record.relative_path,
                    "classification": "QWEN_RETRIEVAL_UNAVAILABLE",
                    "reason": reason,
                    "artifact_path": str(artifact),
                    "artifact_present": artifact.is_file(),
                }
            )
    return eligible, excluded


def cohort_payload(
    *,
    run_id: str,
    raw_manifest_meta: Mapping[str, Any],
    total_records: int,
    eligible: Sequence[RawRecord],
    excluded: Sequence[Mapping[str, Any]],
    retrieval_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    root = Path(retrieval_root)
    return {
        "schema": COHORT_SCHEMA,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_frozen_manifest": {
            "path": raw_manifest_meta.get("manifest_path"),
            "sha256": raw_manifest_meta.get("manifest_sha256"),
            "schema": raw_manifest_meta.get("schema"),
        },
        "dataset_root": str(Path(dataset_root).resolve()),
        "retrieval_root": str(root.resolve()),
        "original_records": total_records,
        "eligible_records": len(eligible),
        "excluded_records": len(excluded),
        "failure_rate": (len(excluded) / total_records) if total_records else None,
        "status": "COMPLETE_WITH_EXCLUSIONS" if excluded else "COMPLETE",
        "excluded_ids": [row["video_id"] for row in excluded],
        "excluded": list(excluded),
        "eligible": [
            {
                "video_id": record.video_id,
                "relative_video_path": record.relative_path,
                "retrieval_artifact": str(root / "retrieval" / f"{record.video_id}.json"),
                "duration_sec": record.duration_sec,
                "quality_arm": record.arm,
                "vcodec": record.vcodec,
                "duration_bin": record.duration_bin,
                "source_sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
            for record in eligible
        ],
    }


def entries_from_cohort(records: Sequence[RawRecord], *, dataset_id: str = ".") -> list[IndexEntry]:
    """Retrieval-only entries for the Qwen stage map ``dataset_id`` to the raw root."""

    return [
        IndexEntry(
            dataset_id=dataset_id,
            video_id=record.video_id,
            realized_video_id=record.video_id,
            category=record.arm or "phd2",
            split="TRAIN",
            relative_video_path=record.relative_path.replace("\\", "/"),
            source_sha256=record.sha256,
            annotation_identity=".",
            duration_sec=record.duration_sec,
        )
        for record in records
    ]


@dataclass(frozen=True)
class IndexBuildResult:
    index_path: Path
    index_sha256: str
    expectations_path: Path
    audit: dict[str, Any]
    metadata: dict[str, Any]


def build_probed_index(
    entries: Sequence[IndexEntry],
    *,
    dataset_root: str | Path,
    index_path: str | Path,
    grid_expectations_path: str | Path,
    verify_sha256: bool = True,
    extract_pts: bool = True,
    ffprobe_bin: str = "ffprobe",
    workers: int = 1,
    log: Any | None = None,
) -> IndexBuildResult:
    """Probe each entry with ffprobe/PTS and write the frozen index + grid expectations.

    Resume-safe at two levels: previously OK entries in an existing index are
    reused, and a per-entry partial journal survives an interrupted build
    (``<index>.partial.jsonl``).  ``workers > 1`` runs the identical per-entry
    probe concurrently; it never changes the probe semantics or the output order.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    from .real_provider import build_grid_for_entry

    target = Path(index_path)
    partial_path = Path(str(target) + ".partial.jsonl")
    metadata: dict[str, Any] = {
        "dataset_id": "phd2_raw_1500",
        "protocol_name": "phd2_full_cohort",
        "protocol_version": "frozen-2fps-uniform-one-pass-batch8",
        "seed": None,
        "split_manifest_sha256": None,
        "split_content_sha256": None,
    }
    existing: dict[str, IndexEntry] = {}
    if target.is_file():
        loaded, _meta = load_index(target)
        existing = {entry.video_id: entry for entry in loaded}
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = json.loads(line)
            pts = values.get("pts_timestamps")
            if pts is not None:
                values["pts_timestamps"] = tuple(pts)
            entry = IndexEntry(**values)
            existing.setdefault(entry.video_id, entry)

    probed: dict[str, IndexEntry] = {}
    todo: list[IndexEntry] = []
    for entry in entries:
        prior = existing.get(entry.video_id)
        if prior is not None and prior.decode_status == DECODE_OK:
            probed[entry.video_id] = prior
        else:
            todo.append(entry)

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    journal = partial_path.open("a", encoding="utf-8", newline="\n")
    journal_lock = threading.Lock()
    log_lock = threading.Lock()
    done = len(probed)

    def _record(enriched: IndexEntry) -> None:
        nonlocal done
        from dataclasses import asdict

        with journal_lock:
            journal.write(json.dumps(asdict(enriched), ensure_ascii=False, sort_keys=True) + "\n")
            journal.flush()
        probed[enriched.video_id] = enriched
        done += 1
        if log is not None and done % 100 == 0:
            with log_lock:
                log(f"probed {done}/{len(entries)} last={enriched.video_id} status={enriched.decode_status}")

    if todo and workers > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = {
                pool.submit(
                    probe_index_entry,
                    entry,
                    dataset_root,
                    verify_sha256=verify_sha256,
                    extract_pts=extract_pts,
                    ffprobe_bin=ffprobe_bin,
                ): entry
                for entry in todo
            }
            for future in as_completed(futures):
                _record(future.result())
    else:
        for entry in todo:
            _record(
                probe_index_entry(
                    entry,
                    dataset_root,
                    verify_sha256=verify_sha256,
                    extract_pts=extract_pts,
                    ffprobe_bin=ffprobe_bin,
                )
            )
    journal.close()

    ordered = sorted(probed.values(), key=lambda item: item.video_id)
    if len(ordered) != len(entries):
        raise Phd2RtdetrError(
            f"index probe produced {len(ordered)}/{len(entries)} entries; partial journal kept for resume"
        )
    path, _sidecar = write_index(ordered, metadata=metadata, output_path=target)
    if partial_path.is_file():
        partial_path.unlink()
    audit = audit_index(ordered)
    expectations: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for entry in ordered:
        if entry.decode_status != DECODE_OK:
            problems.append(
                {"video_id": entry.video_id, "status": entry.decode_status, "error": entry.decode_error}
            )
            continue
        try:
            sample = build_grid_for_entry(entry)
        except RealProviderError as exc:
            problems.append({"video_id": entry.video_id, "status": "GRID_FAILED", "error": str(exc)})
            continue
        expectations.append(
            {
                "video_id": entry.video_id,
                "sampled_frames": int(sample.frames.shape[0]),
                "first_timestamp": float(sample.timestamps[0]) if sample.frames.shape[0] else None,
                "last_timestamp": float(sample.timestamps[-1]) if sample.frames.shape[0] else None,
                "frame_count": entry.frame_count,
                "fps": entry.fps,
                "timestamp_mode": entry.timestamp_mode,
            }
        )
    expectations_path = Path(grid_expectations_path)
    atomic_json(
        expectations_path,
        {
            "schema": GRID_EXPECTATIONS_SCHEMA,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "index_path": str(path),
            "grid": "full-video uniform 2.0 fps",
            "total": len(expectations),
            "problems": problems,
            "entries": expectations,
        },
    )
    return IndexBuildResult(
        index_path=path,
        index_sha256=sha256_file(path),
        expectations_path=expectations_path,
        audit={**audit, "grid_problems": len(problems), "grid_ok": len(expectations)},
        metadata=metadata,
    )


def validate_detection_artifact(path: str | Path, entry: IndexEntry) -> tuple[bool, str]:
    target = Path(path)
    if not target.is_file():
        return False, "missing"
    try:
        payload = np.load(target, allow_pickle=False)
        schema = str(payload["schema"])
        if schema != DETECTION_SCHEMA:
            return False, f"schema={schema!r}"
        visual = np.asarray(payload["visual"])
        frames = np.asarray(payload["frames"])
        offsets = np.asarray(payload["offsets"])
        timestamps = np.asarray(payload["timestamps"])
        if visual.ndim != 2 or visual.shape[1] != 256:
            return False, f"visual shape {visual.shape}"
        if frames.ndim != 1 or visual.shape[0] != frames.shape[0]:
            return False, "frame axis mismatch"
        if timestamps.shape[0] != frames.shape[0]:
            return False, "timestamp axis mismatch"
        if offsets.shape[0] != frames.shape[0] + 1:
            return False, "offsets length mismatch"
        if frames.shape[0] == 0:
            return False, "empty grid"
        if entry.frame_count is not None and int(frames.max()) >= int(entry.frame_count):
            return False, "frame id exceeds frame_count"
        if not np.isfinite(visual).all():
            return False, "non-finite visual"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def audit_detections(run_root: str | Path, entries: Sequence[IndexEntry]) -> tuple[list[str], list[str], list[str]]:
    detections = Path(run_root) / "detections"
    present = {path.stem for path in detections.glob("*.npz")} if detections.is_dir() else set()
    expected = {entry.video_id for entry in entries}
    missing = sorted(expected - present)
    duplicate = sorted(present - expected)
    corrupt: list[str] = []
    by_id = {entry.video_id: entry for entry in entries}
    for video_id in sorted(expected & present):
        ok, _why = validate_detection_artifact(detections / f"{video_id}.npz", by_id[video_id])
        if not ok:
            corrupt.append(video_id)
    return missing, duplicate, corrupt


def run_detection_cohort(
    entries: Sequence[IndexEntry],
    *,
    dataset_root: str | Path,
    run_root: str | Path,
    run_id: str,
    rtdetr_snapshot: str | Path,
    rtdetr_model_id: str = RTDETR_MODEL,
    device: str = "cuda",
    batch_size: int = FROZEN_BATCH_SIZE,
    top_k: int = FROZEN_TOP_K,
    decode_mode: str = FROZEN_DECODE_MODE,
    retry_failed_rounds: int = 2,
    code_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    for name in ("detections", "grid", "logs", "summary", "failures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    progress_jsonl = root / "progress.jsonl"
    log_path = root / "logs" / "run.log"
    total = len(entries)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()

    manifest = {
        "schema": DETECTION_RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_utc": started_wall,
        "stage": "RTDETR_DETECTION_VISUAL256_ONLY",
        "forbidden_stages_not_run": ["LOC", "CMP", "TS", "native16", "pairwise_pu_supervision", "ftnet"],
        "model": {"id": rtdetr_model_id, "revision": RTDETR_REVISION},
        "execution": {
            "model_load": "task_level_once",
            "decode_mode": decode_mode,
            "batch_size": int(batch_size),
            "top_k": int(top_k),
            "sampling": "full-video uniform 2.0 fps",
            "whole_video_rerun_on_failure": False,
            "failure_isolation": True,
        },
        "cohort_size": total,
        "code_identity": dict(code_identity or {}),
    }
    atomic_json(root / "manifest" / "run_manifest.json", manifest)

    log_handle = log_path.open("a", encoding="utf-8", newline="\n")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"=== rtdetr run start run_id={run_id} videos={total} batch={batch_size} decode={decode_mode} ===")

    load_started = time.perf_counter()
    localizer = create_rtdetr_localizer(
        rtdetr_snapshot=rtdetr_snapshot, rtdetr_model_id=rtdetr_model_id, device=device
    )
    model_load_s = time.perf_counter() - load_started
    log(f"RT-DETR loaded once in {model_load_s:.1f}s device={device}")

    expected = {entry.video_id: entry for entry in entries}
    progress: dict[str, Any] = {
        "schema": DETECTION_STATUS_SCHEMA,
        "run_id": run_id,
        "completed": 0,
        "failed": 0,
        "videos_total": total,
        "frames_processed": 0,
        "current_video": None,
        "elapsed_seconds": 0.0,
        "videos_per_second": None,
        "eta_seconds": None,
        "model_load_seconds": round(model_load_s, 3),
        "gpu_util_percent": None,
        "vram_used_mib": None,
        "disk_free_mib": None,
        "updated_utc": None,
    }

    stop_reason: str | None = None
    rounds = max(int(retry_failed_rounds), 0) + 1
    for round_index in range(rounds):
        pending = [
            entry
            for entry in entries
            if not validate_detection_artifact(detection_artifact_path(root, entry.video_id), entry)[0]
        ]
        if round_index > 0:
            log(f"=== retry round {round_index}/{rounds - 1}: {len(pending)} videos pending ===")
        if not pending:
            log("no pending videos; detection set complete")
            break
        for entry in pending:
            video_started = time.perf_counter()
            status = "completed"
            error_message = None
            frames_processed = 0
            try:
                result = run_detection_for_entry(
                    entry,
                    dataset_root=dataset_root,
                    work_root=root,
                    rtdetr_snapshot=rtdetr_snapshot,
                    rtdetr_model_id=rtdetr_model_id,
                    device=device,
                    batch_size=batch_size,
                    top_k=top_k,
                    decode_window=64,
                    decode_mode=decode_mode,
                    localizer=localizer,
                    model_load_s=0.0,
                    model_load_count=0,
                )
                frames_processed = int(result.get("frames", 0))
                progress["frames_processed"] = int(progress["frames_processed"]) + frames_processed
                valid, why = validate_detection_artifact(
                    detection_artifact_path(root, entry.video_id), entry
                )
                if not valid:
                    raise Phd2RtdetrError(f"artifact failed post-write validation: {why}")
            except Exception as exc:  # noqa: BLE001 - per-video failure isolation
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                progress["failed"] = int(progress["failed"]) + 1
                atomic_json(
                    root / "failures" / f"{entry.video_id}.json",
                    {
                        "video_id": entry.video_id,
                        "relative_path": entry.relative_video_path,
                        "error": error_message,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                log(f"FAILED video={entry.video_id} error={error_message[:200]}")
                if is_systemic_detection_error(error_message):
                    stop_reason = f"systemic error: {error_message[:200]}"
                    log(f"STOPPING RUN: {stop_reason}")
                    break
            else:
                progress["completed"] = int(progress["completed"]) + 1
            elapsed_video = time.perf_counter() - video_started
            elapsed = time.monotonic() - started_monotonic
            completed = int(progress["completed"])
            rate = completed / elapsed if elapsed > 0 else None
            remaining = max(total - completed, 0)
            progress.update(
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "videos_per_second": rate,
                    "eta_seconds": (remaining / rate) if rate else None,
                    "current_video": entry.video_id,
                    "last_video_status": status,
                    "last_video_seconds": round(elapsed_video, 3),
                    "last_video_frames": frames_processed,
                    "last_video_error": error_message,
                    "gpu_util_percent": _gpu_metric(0),
                    "vram_used_mib": _gpu_metric(1),
                    "disk_free_mib": _disk_free_mib(),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            atomic_json(progress_path, progress)
            append_jsonl(
                progress_jsonl,
                {
                    "video_id": entry.video_id,
                    "status": status,
                    "frames": frames_processed,
                    "elapsed_video_seconds": round(elapsed_video, 3),
                    "completed": completed,
                    "failed": progress["failed"],
                    "frames_total": progress["frames_processed"],
                    "elapsed_seconds": round(elapsed, 3),
                    "eta_seconds": progress["eta_seconds"],
                    "error": error_message,
                    "timestamp_utc": progress["updated_utc"],
                },
            )
            if status == "completed":
                log(
                    f"video={entry.video_id} frames={frames_processed} elapsed={elapsed_video:.1f}s "
                    f"completed={completed}/{total} failed={progress['failed']} "
                    f"eta={progress['eta_seconds'] if progress['eta_seconds'] is None else round(progress['eta_seconds']/3600, 3)}h "
                    f"gpu={progress['gpu_util_percent']}% vram={progress['vram_used_mib']}MiB"
                )
        if stop_reason:
            break

    elapsed = time.monotonic() - started_monotonic
    missing, duplicate, corrupt = audit_detections(root, entries)
    summary = {
        "schema": DETECTION_RUN_SUMMARY_SCHEMA,
        "run_id": run_id,
        "stage": "RTDETR_DETECTION_VISUAL256_ONLY",
        "status": "COMPLETE" if not missing and not corrupt else "INCOMPLETE",
        "stop_reason": stop_reason,
        "videos_total": total,
        "completed": int(progress["completed"]),
        "failed": int(progress["failed"]),
        "frames_processed": int(progress["frames_processed"]),
        "wall_seconds": round(elapsed, 3),
        "wall_hours": round(elapsed / 3600.0, 4),
        "videos_per_second": (int(progress["completed"]) / elapsed) if elapsed else None,
        "model_load_seconds": round(model_load_s, 3),
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "decode_mode": decode_mode,
        "missing": len(missing),
        "duplicate": len(duplicate),
        "corrupt": len(corrupt),
        "missing_ids": missing[:20],
        "corrupt_ids": corrupt[:20],
        "code_identity": dict(code_identity or {}),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / "summary" / "run_summary.json", summary)
    log(
        f"=== rtdetr run end status={summary['status']} completed={summary['completed']} failed={summary['failed']} "
        f"wall={summary['wall_hours']:.3f}h missing={summary['missing']} corrupt={summary['corrupt']} ==="
    )
    del localizer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - optional cleanup only
        pass
    log_handle.close()
    return summary


def _gpu_metric(index: int) -> Any:
    import subprocess

    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).splitlines()[0]
        values = [item.strip() for item in raw.split(",")]
        return values[index] if index < len(values) else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _disk_free_mib() -> int | None:
    import subprocess

    try:
        for line in subprocess.check_output(["df", "-Pk", "/root/autodl-tmp"], text=True).splitlines()[1:]:
            return int(line.split()[3]) // 1024
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return None


__all__ = [
    "COHORT_SCHEMA",
    "DETECTION_RUN_MANIFEST_SCHEMA",
    "DETECTION_RUN_SUMMARY_SCHEMA",
    "DETECTION_STATUS_SCHEMA",
    "FAILURE_AUDIT_SCHEMA",
    "FROZEN_BATCH_SIZE",
    "FROZEN_DECODE_MODE",
    "FROZEN_TOP_K",
    "GRID_EXPECTATIONS_SCHEMA",
    "Phd2RtdetrError",
    "audit_detections",
    "build_probed_index",
    "build_success_cohort",
    "cohort_payload",
    "entries_from_cohort",
    "is_systemic_detection_error",
    "run_detection_cohort",
    "validate_detection_artifact",
]
