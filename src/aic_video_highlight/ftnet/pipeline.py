"""Stage 7 FTNet materialization pipeline: retrieval -> detection -> assemble.

This module is the operational driver for the AutoDL GPU host.  It owns
resumable per-video staging, the failure journal, progress reporting and the
materialized-video manifest.  All scientific computations live in
``provider_core`` (CPU) and the heavy component calls live in
``real_provider`` (GPU); nothing here alters frozen component code.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from aic_video_highlight.experiment_runtime.progress import ProgressReporter
from aic_video_highlight.runtime.paths import EnvironmentPaths

from .index import (
    DECODE_OK,
    IndexEntry,
    build_index_entries,
    load_index,
    load_split_manifest_payload,
    probe_index_entry,
    write_index,
)
from .materialize import (
    MATERIALIZE_SCHEMA,
    materialize_video,
    verify_materialized,
    write_manifest,
)
from .native_schema import NATIVE_SCHEMA_VERSION
from .provider_core import IDX0_FALLBACK_NONE, IDX0_FALLBACKS
from .real_provider import (
    QWEN_MODEL,
    RTDETR_MODEL,
    RealUpstreamProvider,
    RetrievalSettings,
    detection_artifact_path,
    retrieval_artifact_path,
    run_detection_for_entry,
    run_retrieval_for_entry,
    video_ref_from_entry,
)

STATUS_SCHEMA = "aic.stage7.ftnet.production-status/v1"
FAILURE_SCHEMA = "aic.stage7.ftnet.failures/v1"
SPLIT_MANIFEST_RELATIVE = Path("manifests") / "stage7_ftnet_split_manifest.json"

STAGE_RETRIEVAL = "retrieval"
STAGE_DETECTION = "detection"
STAGE_ASSEMBLE = "assemble"
STAGES = (STAGE_RETRIEVAL, STAGE_DETECTION, STAGE_ASSEMBLE)


class MaterializationError(RuntimeError):
    """Raised when the production run cannot proceed at all."""


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _fingerprint(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MaterializationSettings:
    dataset_root: Path
    work_root: Path
    output_root: Path
    index_path: Path
    qwen_snapshot: Path
    rtdetr_snapshot: Path
    environment_path: Path
    split_manifest_path: Path | None = None
    rtdetr_model_id: str = RTDETR_MODEL
    device: str = "cuda"
    detection_batch_size: int = 8
    detection_top_k: int = 100
    decode_window: int = 64
    retrieval_timeout_sec: float = 120.0
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_gpu_memory_utilization: float = 0.80
    vllm_max_model_len: int = 32768
    vllm_start_timeout_sec: float = 900.0
    python: str = field(default_factory=lambda: sys.executable)
    idx0_fallback: str = IDX0_FALLBACK_NONE
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.idx0_fallback not in IDX0_FALLBACKS:
            raise MaterializationError(f"unknown idx0 fallback: {self.idx0_fallback}")
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.work_root = Path(self.work_root).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        self.index_path = Path(self.index_path).expanduser().resolve()


class FailureJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.items: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for row in payload.get("failures", []):
                self.items[str(row["video_id"])] = row

    def record(self, video_id: str, stage: str, exc: BaseException) -> None:
        self.items[video_id] = {
            "video_id": video_id,
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.save()

    def clear(self, video_id: str) -> None:
        if video_id in self.items:
            del self.items[video_id]
            self.save()

    def failed_ids(self, stage: str | None = None) -> list[str]:
        return sorted(
            video_id
            for video_id, row in self.items.items()
            if stage is None or row.get("stage") == stage
        )

    def save(self) -> None:
        _atomic_json(
            self.path,
            {"schema": FAILURE_SCHEMA, "count": len(self.items), "failures": list(self.items.values())},
        )


class StatusStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for row in payload.get("videos", []):
                self.records[str(row["video_id"])] = row

    def get(self, video_id: str) -> dict[str, Any]:
        return self.records.get(video_id, {})

    def update(self, video_id: str, **fields: Any) -> None:
        row = self.records.setdefault(video_id, {"video_id": video_id})
        row.update(fields)
        row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()

    def save(self) -> None:
        _atomic_json(
            self.path,
            {
                "schema": STATUS_SCHEMA,
                "count": len(self.records),
                "videos": [self.records[key] for key in sorted(self.records)],
            },
        )


@dataclass
class ProgressState:
    reporter: ProgressReporter
    total: int
    retrieval_done: int = 0
    detection_done: int = 0
    assembled: int = 0
    skipped: int = 0
    failed: int = 0
    current_video: str | None = None
    stage: str = STAGE_RETRIEVAL

    def emit(self, *, gpu_gb: float | None = None, detail: str | None = None) -> None:
        display: dict[str, Any] = {
            "stage": self.stage,
            "retrieval": f"{self.retrieval_done}/{self.total}",
            "detection": f"{self.detection_done}/{self.total}",
            "assemble": f"{self.assembled}/{self.total}",
            "failed": self.failed,
            "skipped": self.skipped,
        }
        if gpu_gb is not None:
            display["gpu_gb"] = f"{gpu_gb:.1f}"
        if detail:
            display["detail"] = detail
        self.reporter.update(
            self.assembled,
            current_video=self.current_video,
            errors=self.failed,
            display=display,
        )


def _gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024**3))
    except Exception:
        return None
    return None


def resolve_split_manifest(settings: MaterializationSettings) -> Path:
    if settings.split_manifest_path is not None:
        return Path(settings.split_manifest_path).expanduser().resolve()
    return (
        settings.dataset_root
        / "youtube_highlights"
        / "manifests"
        / "stage7_ftnet_split_manifest.json"
    )


def build_fresh_index(settings: MaterializationSettings) -> tuple[list[IndexEntry], dict[str, Any]]:
    manifest_path = resolve_split_manifest(settings)
    payload = load_split_manifest_payload(manifest_path)
    entries, metadata = build_index_entries(payload)
    metadata = {**metadata, "split_manifest_path": str(manifest_path)}
    return entries, metadata


def ensure_probed_index(
    settings: MaterializationSettings,
    *,
    rebuild: bool = False,
    verify_sha256: bool = True,
    only_video_ids: set[str] | None = None,
    ffprobe_bin: str = "ffprobe",
) -> tuple[list[IndexEntry], dict[str, Any]]:
    """Load the frozen index, build it when absent, and probe missing metadata."""

    if settings.index_path.is_file() and not rebuild:
        entries, metadata = load_index(settings.index_path)
    else:
        entries, metadata = build_fresh_index(settings)

    updated = False
    result: list[IndexEntry] = []
    for entry in entries:
        if only_video_ids is not None and entry.video_id not in only_video_ids:
            result.append(entry)
            continue
        if entry.decode_status == DECODE_OK and not rebuild:
            result.append(entry)
            continue
        probed = probe_index_entry(
            entry,
            settings.dataset_root,
            verify_sha256=verify_sha256,
            ffprobe_bin=ffprobe_bin,
        )
        result.append(probed)
        updated = True
    if updated or not settings.index_path.is_file():
        write_index(result, metadata=metadata, output_path=settings.index_path)
    return result, metadata


def _select_entries(
    entries: Iterable[IndexEntry],
    *,
    splits: Sequence[str] | None,
    video_ids: Sequence[str] | None,
) -> list[IndexEntry]:
    selected = list(entries)
    if splits:
        allowed = set(splits)
        selected = [entry for entry in selected if entry.split in allowed]
    if video_ids:
        wanted = set(video_ids)
        selected = [entry for entry in selected if entry.video_id in wanted]
        missing = wanted - {entry.video_id for entry in selected}
        if missing:
            raise MaterializationError(f"unknown video ids: {sorted(missing)}")
    if not selected:
        raise MaterializationError("no videos selected for materialization")
    return sorted(selected, key=lambda entry: (entry.split, entry.category, entry.video_id))


def run_retrieval_stage(
    settings: MaterializationSettings,
    entries: Sequence[IndexEntry],
    *,
    progress: ProgressState,
    journal: FailureJournal,
    status: StatusStore,
) -> int:
    from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient
    from aic_video_highlight.runtime.orchestrator import start_vllm, stop_vllm

    progress.stage = STAGE_RETRIEVAL
    todo = [
        entry
        for entry in entries
        if settings.overwrite or not retrieval_artifact_path(settings.work_root, entry.video_id).is_file()
    ]
    progress.retrieval_done = len(entries) - len(todo)
    progress.emit()
    if not todo:
        return 0

    client = QwenVLLMClient(
        base_url=settings.vllm_base_url, model=QWEN_MODEL, timeout_sec=settings.retrieval_timeout_sec
    )
    owned = None
    environment = EnvironmentPaths.from_json(settings.environment_path)
    if not client.health_check():
        config = {
            "inference": {
                "python": settings.python,
                "qwen_snapshot": {"base": "models", "path": str(settings.qwen_snapshot)},
                "video_root": {"base": "datasets", "path": str(settings.dataset_root)},
                "vllm_base_url": settings.vllm_base_url,
                "vllm_gpu_memory_utilization": settings.vllm_gpu_memory_utilization,
                "vllm_max_model_len": settings.vllm_max_model_len,
                "vllm_start_timeout_sec": settings.vllm_start_timeout_sec,
            }
        }
        log_path = settings.work_root / "logs" / "vllm.log"
        owned, base_url = start_vllm(config, environment, log_path)
        client = QwenVLLMClient(
            base_url=base_url, model=QWEN_MODEL, timeout_sec=settings.retrieval_timeout_sec
        )
    failures = 0
    try:
        retrieval_settings = RetrievalSettings(
            base_url=settings.vllm_base_url, timeout_sec=settings.retrieval_timeout_sec
        )
        for entry in todo:
            progress.current_video = entry.video_id
            started = time.perf_counter()
            try:
                payload = run_retrieval_for_entry(
                    entry,
                    dataset_root=settings.dataset_root,
                    client=client,
                    settings=retrieval_settings,
                )
                _atomic_json(
                    retrieval_artifact_path(settings.work_root, entry.video_id), payload
                )
                status.update(
                    entry.video_id,
                    retrieval="OK",
                    retrieval_wall_sec=round(time.perf_counter() - started, 3),
                    retrieval_chunks=len(payload["chunks"]),
                    retrieval_merged=len(payload["merged_candidates"]),
                )
                journal.clear(entry.video_id)
                progress.retrieval_done += 1
            except Exception as exc:  # noqa: BLE001 - per-video isolation is the contract
                failures += 1
                progress.failed = len(journal.items) + failures
                journal.record(entry.video_id, STAGE_RETRIEVAL, exc)
                status.update(entry.video_id, retrieval="FAILED", retrieval_error=str(exc))
            progress.emit(detail=entry.video_id)
    finally:
        if owned is not None:
            stop_vllm(owned)
    progress.failed = len(journal.items)
    return failures


def run_detection_stage(
    settings: MaterializationSettings,
    entries: Sequence[IndexEntry],
    *,
    progress: ProgressState,
    journal: FailureJournal,
    status: StatusStore,
) -> int:
    progress.stage = STAGE_DETECTION
    todo = [
        entry
        for entry in entries
        if settings.overwrite or not detection_artifact_path(settings.work_root, entry.video_id).is_file()
    ]
    progress.detection_done = len(entries) - len(todo)
    progress.emit()
    if not todo:
        return 0
    failures = 0
    for entry in todo:
        progress.current_video = entry.video_id
        started = time.perf_counter()
        try:
            result = run_detection_for_entry(
                entry,
                dataset_root=settings.dataset_root,
                work_root=settings.work_root,
                rtdetr_snapshot=settings.rtdetr_snapshot,
                rtdetr_model_id=settings.rtdetr_model_id,
                device=settings.device,
                batch_size=settings.detection_batch_size,
                top_k=settings.detection_top_k,
                decode_window=settings.decode_window,
            )
            status.update(
                entry.video_id,
                detection="OK",
                detection_wall_sec=round(float(result["wall_sec"]), 3),
                detection_frames=int(result["frames"]),
                detection_candidates=int(result["candidates"]),
            )
            journal.clear(entry.video_id)
            progress.detection_done += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            journal.record(entry.video_id, STAGE_DETECTION, exc)
            status.update(entry.video_id, detection="FAILED", detection_error=str(exc))
        progress.failed = len(journal.items)
        progress.emit(gpu_gb=_gpu_memory_gb(), detail=entry.video_id)
    return failures


def assemble_fingerprint(settings: MaterializationSettings) -> str:
    return _fingerprint(
        {
            "schema": MATERIALIZE_SCHEMA,
            "native_schema_version": NATIVE_SCHEMA_VERSION,
            "idx0_fallback": settings.idx0_fallback,
        }
    )


def run_assemble_stage(
    settings: MaterializationSettings,
    entries: Sequence[IndexEntry],
    *,
    progress: ProgressState,
    journal: FailureJournal,
    status: StatusStore,
    records_out: list[dict[str, Any]] | None = None,
) -> int:
    progress.stage = STAGE_ASSEMBLE
    provider = RealUpstreamProvider(
        entries, dataset_root=settings.dataset_root, work_root=settings.work_root
    )
    fingerprint = assemble_fingerprint(settings)
    failures = 0
    records = records_out if records_out is not None else []
    for entry in entries:
        progress.current_video = entry.video_id
        ref = video_ref_from_entry(entry)
        target_path = settings.output_root / entry.split_dir_name / f"{entry.video_id}.safetensors"
        row = status.get(entry.video_id)
        if (
            not settings.overwrite
            and row.get("assemble") == "OK"
            and row.get("assemble_fingerprint") == fingerprint
            and verify_materialized(target_path, source_sha256=entry.source_sha256)
        ):
            progress.skipped += 1
            progress.assembled += 1
            records.append(
                {
                    "canonical_video_id": entry.video_id,
                    "category": entry.category,
                    "stage7_split": entry.split,
                    "relative_path": f"{entry.split_dir_name}/{entry.video_id}.safetensors",
                    "frame_count": int(row.get("assemble_frames", 0)),
                    "skipped": True,
                    "missed_positive_frames": int(row.get("assemble_missed_positive", 0)),
                }
            )
            progress.emit(detail=entry.video_id)
            continue
        try:
            if not provider.has_artifacts(entry.video_id):
                raise MaterializationError(
                    f"cannot assemble {entry.video_id}: staged artifacts missing"
                )
            upstream = provider.produce(ref)
            overwrite = settings.overwrite or row.get("assemble_fingerprint") != fingerprint
            record = materialize_video(upstream, settings.output_root, overwrite=overwrite)
            records.append(record)
            status.update(
                entry.video_id,
                assemble="OK",
                assemble_fingerprint=fingerprint,
                assemble_frames=record["frame_count"],
                assemble_missed_positive=record["missed_positive_frames"],
                materialized_relative_path=record["relative_path"],
                idx0_fallback=settings.idx0_fallback,
            )
            journal.clear(entry.video_id)
            progress.assembled += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            journal.record(entry.video_id, STAGE_ASSEMBLE, exc)
            status.update(entry.video_id, assemble="FAILED", assemble_error=str(exc))
        progress.failed = len(journal.items)
        progress.emit(detail=entry.video_id)
    if records:
        write_manifest(
            settings.output_root,
            records,
            extra={
                "idx0_fallback": settings.idx0_fallback,
                "assemble_fingerprint": fingerprint,
            },
        )
    return failures


def load_raw_entries(
    settings: MaterializationSettings, *, rebuild_index: bool = False
) -> tuple[list[IndexEntry], dict[str, Any]]:
    if settings.index_path.is_file() and not rebuild_index:
        return load_index(settings.index_path)
    return build_fresh_index(settings)


def select_and_probe(
    settings: MaterializationSettings,
    *,
    splits: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    limit: int | None = None,
    rebuild_index: bool = False,
    verify_sha256: bool = True,
    ffprobe_bin: str = "ffprobe",
) -> tuple[list[IndexEntry], dict[str, Any]]:
    """Select the run scope first, then probe only the selected videos."""

    if settings.index_path.is_file() and not rebuild_index:
        entries, metadata = load_index(settings.index_path)
    else:
        entries, metadata = build_fresh_index(settings)
    selected = _select_entries(entries, splits=splits, video_ids=video_ids)
    if limit is not None:
        if limit <= 0:
            raise MaterializationError("limit must be a positive integer")
        ordered = sorted(selected, key=lambda entry: (entry.split, entry.category, entry.video_id))
        selected = ordered[:limit]
    by_id = {entry.video_id: entry for entry in entries}
    probed: list[IndexEntry] = []
    changed = False
    for entry in selected:
        if entry.decode_status == DECODE_OK and not rebuild_index:
            probed.append(entry)
            continue
        updated = probe_index_entry(
            entry, settings.dataset_root, verify_sha256=verify_sha256, ffprobe_bin=ffprobe_bin
        )
        probed.append(updated)
        by_id[entry.video_id] = updated
        changed = True
    if changed or not settings.index_path.is_file():
        merged = [by_id[entry.video_id] for entry in entries]
        write_index(merged, metadata=metadata, output_path=settings.index_path)
    bad = [entry.video_id for entry in probed if entry.decode_status != DECODE_OK]
    if bad:
        raise MaterializationError(
            f"selected index entries are not decodable: {len(bad)} video(s), first={bad[:5]}"
        )
    return probed, metadata


def run_materialization(
    settings: MaterializationSettings,
    *,
    splits: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    limit: int | None = None,
    rebuild_index: bool = False,
    verify_sha256: bool = True,
    stages: Sequence[str] = STAGES,
) -> dict[str, Any]:
    selected, index_metadata = select_and_probe(
        settings,
        splits=splits,
        video_ids=video_ids,
        limit=limit,
        rebuild_index=rebuild_index,
        verify_sha256=verify_sha256,
    )
    journal = FailureJournal(settings.work_root / "failures.json")
    status = StatusStore(settings.work_root / "status.json")
    log_dir = settings.work_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(
        experiment_id="ftnet_data_materialization", total=len(selected), log_dir=log_dir
    )
    progress = ProgressState(reporter=reporter, total=len(selected))
    started = time.monotonic()
    records: list[dict[str, Any]] = []

    if STAGE_RETRIEVAL in stages:
        run_retrieval_stage(settings, selected, progress=progress, journal=journal, status=status)
    if STAGE_DETECTION in stages:
        run_detection_stage(settings, selected, progress=progress, journal=journal, status=status)
    if STAGE_ASSEMBLE in stages:
        run_assemble_stage(
            settings, selected, progress=progress, journal=journal, status=status, records_out=records
        )

    elapsed = time.monotonic() - started
    summary = {
        "schema": "aic.stage7.ftnet.materialization-summary/v1",
        "status": "PASS" if not journal.failed_ids() else "COMPLETED_WITH_FAILURES",
        "work_root": str(settings.work_root),
        "output_root": str(settings.output_root),
        "index_path": str(settings.index_path),
        "index_metadata": index_metadata,
        "selected": len(selected),
        "selected_video_ids": [entry.video_id for entry in selected],
        "counts": {
            split: sum(1 for entry in selected if entry.split == split)
            for split in ("TRAIN", "VALIDATION", "CALIBRATION")
        },
        "retrieval_done": progress.retrieval_done,
        "detection_done": progress.detection_done,
        "assembled": progress.assembled,
        "skipped": progress.skipped,
        "failed": len(journal.failed_ids()),
        "failed_video_ids": journal.failed_ids(),
        "idx0_fallback": settings.idx0_fallback,
        "elapsed_sec": round(elapsed, 3),
    }
    _atomic_json(settings.work_root / "summary.json", summary)
    return summary


def retry_failed(settings: MaterializationSettings, *, splits: Sequence[str] | None = None) -> dict[str, Any]:
    journal = FailureJournal(settings.work_root / "failures.json")
    failed = journal.failed_ids()
    if not failed:
        return {"status": "NO_FAILURES", "retried": 0}
    status = StatusStore(settings.work_root / "status.json")
    for video_id in failed:
        for stage in STAGES:
            if stage == STAGE_ASSEMBLE:
                row = status.get(video_id)
                path = row.get("materialized_relative_path")
                if path:
                    try:
                        (settings.output_root / path).unlink()
                    except FileNotFoundError:
                        pass
                continue
            artifact = (
                retrieval_artifact_path(settings.work_root, video_id)
                if stage == STAGE_RETRIEVAL
                else detection_artifact_path(settings.work_root, video_id)
            )
            try:
                artifact.unlink()
            except FileNotFoundError:
                pass
    return run_materialization(settings, splits=splits, video_ids=failed, verify_sha256=False)


__all__ = [
    "FAILURE_SCHEMA",
    "MaterializationError",
    "MaterializationSettings",
    "STAGE_ASSEMBLE",
    "STAGE_DETECTION",
    "STAGE_RETRIEVAL",
    "STAGES",
    "STATUS_SCHEMA",
    "assemble_fingerprint",
    "build_fresh_index",
    "ensure_probed_index",
    "retry_failed",
    "run_materialization",
]
