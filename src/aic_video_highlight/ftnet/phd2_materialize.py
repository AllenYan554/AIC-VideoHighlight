"""PHD2 materialization: upstream features + frozen PHD2 pretrain targets.

Consumes per-video staged artifacts produced by the earlier stages:

    retrieval   Qwen high-recall candidates            (retrieval/<video_id>.json)
    detection   2 fps RT-DETR + Visual256 + grid       (detections/<video_id>.npz)

and the frozen TRAIN annotation CSV, then writes one safetensors per video with
the contract required by ``PHD2_GENERIC_PRETRAIN_TARGET_V1``:

    visual            float16 [T,256]   (Visual256; Delta256 is computed at train time)
    native            float32 [T,16]    (raw Native16)
    native_missing    float32 [T,16]
    timestamps        float64 [T]
    source_frame_ids  int64   [T]
    adjacency_mask    bool    [T]
    positive_mask     bool    [T]       (TRAIN selection coverage)
    unlabeled_mask    bool    [T]       (not negative)
    boundary_guard_mask bool  [T]
    selection_support_count int32 [T]
    pair_positive_index  int64 [P]
    pair_unlabeled_index int64 [P]

LOC / CMP / TS are the frozen upstream transforms already applied by
``provider_core.build_video_upstream`` when it produces Native16.  No label is
ever fabricated; videos without retrieval or detection artifacts fail closed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .index import DECODE_OK, IndexEntry
from .materialize import assemble_native
from .performance_decode import ONE_PASS
from .phd2_qwen1500 import atomic_json, sha256_file
from .phd2_rtdetr import validate_detection_artifact
from .phd2_targets import (
    PAIR_SEED_DEFAULT,
    PROTOCOL_VERSION,
    AnnotationIndex,
    load_train_annotations,
    sample_pairs,
    timestep_masks,
)
from .provider_core import VideoArtifacts, build_video_upstream
from .real_provider import (
    RealProviderError,
    detection_artifact_path,
    load_detection,
    load_retrieval_context,
    retrieval_artifact_path,
    video_ref_from_entry,
)

MATERIALIZED_SCHEMA = "aic.phd2.materialized/v1"
MATERIALIZE_RUN_SUMMARY_SCHEMA = "aic.phd2.materialize-summary/v1"


class Phd2MaterializeError(RuntimeError):
    """Raised when a video cannot be materialized under the frozen contract."""


def materialized_path(run_root: str | Path, video_id: str) -> Path:
    return Path(run_root) / "materialized" / f"{video_id}.safetensors"


def validate_materialized(path: str | Path, entry: IndexEntry) -> tuple[bool, str]:
    import torch
    from safetensors import safe_open

    target = Path(path)
    if not target.is_file():
        return False, "missing"
    try:
        with safe_open(str(target), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("protocol") != PROTOCOL_VERSION:
                return False, f"protocol metadata {metadata.get('protocol')!r}"
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        visual = tensors["visual"]
        native = tensors["native"]
        timestamps = tensors["timestamps"]
        pos = tensors["positive_mask"]
        unl = tensors["unlabeled_mask"]
        guard = tensors["boundary_guard_mask"]
        support = tensors["selection_support_count"]
        frames = tensors["source_frame_ids"]
        if tuple(visual.shape) != (timestamps.shape[0], 256):
            return False, f"visual shape {tuple(visual.shape)}"
        if tuple(native.shape) != (timestamps.shape[0], 16):
            return False, f"native shape {tuple(native.shape)}"
        for name, value in (("positive_mask", pos), ("unlabeled_mask", unl), ("boundary_guard_mask", guard)):
            if value.shape[0] != timestamps.shape[0]:
                return False, f"{name} length mismatch"
        if support.shape[0] != timestamps.shape[0]:
            return False, "support length mismatch"
        if frames.shape[0] != timestamps.shape[0]:
            return False, "source_frame_ids length mismatch"
        if bool((pos & unl).any()):
            return False, "positive and unlabeled overlap"
        if bool((pos & guard).any()):
            return False, "positive inside boundary guard"
        if not bool(torch.isfinite(timestamps).all()):
            return False, "non-finite timestamps"
        if not bool(torch.isfinite(visual).all()):
            return False, "non-finite visual"
        pi = tensors["pair_positive_index"]
        ui = tensors["pair_unlabeled_index"]
        if pi.shape != ui.shape:
            return False, "pair ledger shape mismatch"
        n = int(timestamps.shape[0])
        if pi.numel() and (int(pi.max()) >= n or int(ui.max()) >= n or int(pi.min()) < 0 or int(ui.min()) < 0):
            return False, "pair index out of range"
    except (OSError, KeyError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def materialize_one(
    entry: IndexEntry,
    *,
    dataset_root: str | Path,
    retrieval_root: str | Path,
    rtdetr_root: str | Path,
    run_root: str | Path,
    annotations: AnnotationIndex,
    pair_seed: int = PAIR_SEED_DEFAULT,
) -> dict[str, Any]:
    """Build one PHD2 materialized example; never fabricates anything."""

    import torch
    from safetensors.torch import save_file

    video_id = entry.video_id
    retrieval_path = retrieval_artifact_path(retrieval_root, video_id)
    detection_path = detection_artifact_path(rtdetr_root, video_id)
    if not retrieval_path.is_file():
        raise Phd2MaterializeError(f"retrieval artifact missing: {retrieval_path}")
    ok, why = validate_detection_artifact(detection_path, entry)
    if not ok:
        raise Phd2MaterializeError(f"detection artifact invalid: {why}")

    retrieval, retrieval_payload = load_retrieval_context(retrieval_path)
    detections, sample, visual = load_detection(detection_path)

    artifacts = VideoArtifacts(
        ref=video_ref_from_entry(entry),
        width=int(entry.width or 0),
        height=int(entry.height or 0),
        timestamps=sample.timestamps,
        source_frame_ids=sample.frames,
        adjacency_mask=sample.adjacency_mask,
        visual=visual,
        retrieval=retrieval,
        detections=detections,
        mturk_clips=(),
        retrieval_metadata={
            "retrieval_provenance": retrieval_payload.get("provenance", {}),
            "retrieval_timing": retrieval_payload.get("timing", {}),
        },
    )
    upstream = build_video_upstream(artifacts)
    native_raw, native_missing = assemble_native(upstream)

    timestamps = np.asarray(upstream.timestamps, dtype=np.float64)
    masks = timestep_masks(timestamps, annotations.intervals_for(video_id))
    pairs = sample_pairs(
        timestamps,
        masks,
        video_id=video_id,
        adjacency_mask=np.asarray(upstream.adjacency_mask, dtype=bool),
        seed=pair_seed,
    )
    pair_pos = np.asarray([p["positive_index"] for p in pairs], dtype=np.int64)
    pair_unl = np.asarray([p["unlabeled_index"] for p in pairs], dtype=np.int64)

    tensors = {
        "visual": torch.from_numpy(np.asarray(upstream.visual, dtype=np.float16)),
        "native": torch.from_numpy(np.asarray(native_raw, dtype=np.float32)),
        "native_missing": torch.from_numpy(np.asarray(native_missing, dtype=np.float32)),
        "timestamps": torch.from_numpy(timestamps),
        "source_frame_ids": torch.from_numpy(np.asarray(upstream.source_frame_id, dtype=np.int64)),
        "adjacency_mask": torch.from_numpy(np.asarray(upstream.adjacency_mask, dtype=bool)),
        "positive_mask": torch.from_numpy(np.asarray(masks["positive_mask"], dtype=bool)),
        "unlabeled_mask": torch.from_numpy(np.asarray(masks["unlabeled_mask"], dtype=bool)),
        "boundary_guard_mask": torch.from_numpy(np.asarray(masks["boundary_guard_mask"], dtype=bool)),
        "selection_support_count": torch.from_numpy(
            np.asarray(masks["selection_support_count"], dtype=np.int32)
        ),
        "pair_positive_index": torch.from_numpy(pair_pos),
        "pair_unlabeled_index": torch.from_numpy(pair_unl),
    }
    path = materialized_path(run_root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.safetensors")
    save_file(tensors, str(temporary), metadata={"protocol": PROTOCOL_VERSION, "video_id": video_id})
    temporary.replace(path)

    n = int(timestamps.shape[0])
    return {
        "video_id": video_id,
        "frames": n,
        "positive_frames": int(np.asarray(masks["positive_mask"]).sum()),
        "unlabeled_frames": int(np.asarray(masks["unlabeled_mask"]).sum()),
        "boundary_guard_frames": int(np.asarray(masks["boundary_guard_mask"]).sum()),
        "legal_unlabeled_frames": int(np.asarray(masks["legal_unlabeled_mask"]).sum()),
        "pairs": len(pairs),
        "native_missing_cells": int(np.asarray(native_missing).sum()),
        "artifact": str(path),
        "retrieval_artifact": str(retrieval_path),
        "detection_artifact": str(detection_path),
        "pair_seed": int(pair_seed),
        "protocol_version": PROTOCOL_VERSION,
    }


def load_annotations_for_run(path: str | Path) -> AnnotationIndex:
    return load_train_annotations(path)


def materialize_cohort(
    entries: Sequence[IndexEntry],
    *,
    dataset_root: str | Path,
    retrieval_root: str | Path,
    rtdetr_root: str | Path,
    run_root: str | Path,
    annotation_path: str | Path,
    pair_seed: int = PAIR_SEED_DEFAULT,
    retry_failed_rounds: int = 1,
    code_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    for name in ("materialized", "materialized_meta", "logs", "summary", "failures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / "materialize.log"
    started_monotonic = time.monotonic()
    annotations = load_annotations_for_run(annotation_path)
    total = len(entries)
    log_handle = log_path.open("a", encoding="utf-8", newline="\n")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"=== materialize start videos={total} annotation_sha={annotations.source_sha256[:16]} protocol={PROTOCOL_VERSION} ===")
    progress = {"completed": 0, "failed": 0, "videos_total": total, "current_video": None,
                "elapsed_seconds": 0.0, "eta_seconds": None, "protocol": PROTOCOL_VERSION}
    stop_reason: str | None = None
    rounds = max(int(retry_failed_rounds), 0) + 1
    for round_index in range(rounds):
        pending = [
            entry for entry in entries
            if not validate_materialized(materialized_path(root, entry.video_id), entry)[0]
        ]
        if round_index > 0:
            log(f"=== retry round {round_index}/{rounds - 1}: {len(pending)} pending ===")
        if not pending:
            break
        for entry in pending:
            video_started = time.perf_counter()
            status = "completed"
            error_message = None
            try:
                meta = materialize_one(
                    entry,
                    dataset_root=dataset_root,
                    retrieval_root=retrieval_root,
                    rtdetr_root=rtdetr_root,
                    run_root=root,
                    annotations=annotations,
                    pair_seed=pair_seed,
                )
                valid, why = validate_materialized(materialized_path(root, entry.video_id), entry)
                if not valid:
                    raise Phd2MaterializeError(f"post-write validation failed: {why}")
                atomic_json(root / "materialized_meta" / f"{entry.video_id}.json", meta)
                if int(meta["positive_frames"]) == 0:
                    log(f"video={entry.video_id} frames={meta['frames']} NO_POSITIVE (no pairs; not fabricated)")
            except Exception as exc:  # noqa: BLE001 - per-video isolation
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                progress["failed"] = int(progress["failed"]) + 1
                atomic_json(
                    root / "failures" / f"{entry.video_id}.json",
                    {"video_id": entry.video_id, "stage": "materialize", "error": error_message,
                     "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                )
                log(f"FAILED video={entry.video_id} error={error_message[:200]}")
            else:
                progress["completed"] = int(progress["completed"]) + 1
            elapsed = time.monotonic() - started_monotonic
            progress.update({
                "current_video": entry.video_id,
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": ((total - int(progress["completed"])) / (int(progress["completed"]) / elapsed))
                if int(progress["completed"]) and elapsed else None,
                "last_video_status": status,
                "last_video_seconds": round(time.perf_counter() - video_started, 3),
                "last_video_error": error_message,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            atomic_json(root / "progress.json", progress)
        if stop_reason:
            break

    elapsed = time.monotonic() - started_monotonic
    missing = [
        entry.video_id for entry in entries
        if not materialized_path(root, entry.video_id).is_file()
    ]
    corrupt = [
        entry.video_id for entry in entries
        if materialized_path(root, entry.video_id).is_file()
        and not validate_materialized(materialized_path(root, entry.video_id), entry)[0]
    ]
    summary = {
        "schema": MATERIALIZE_RUN_SUMMARY_SCHEMA,
        "status": "COMPLETE" if not missing and not corrupt else "INCOMPLETE",
        "protocol_version": PROTOCOL_VERSION,
        "annotation_source": annotations.source_path,
        "annotation_sha256": annotations.source_sha256,
        "videos_total": total,
        "completed": int(progress["completed"]),
        "failed": int(progress["failed"]),
        "missing": len(missing),
        "missing_ids": missing[:20],
        "corrupt": len(corrupt),
        "corrupt_ids": corrupt[:20],
        "wall_seconds": round(elapsed, 3),
        "pair_seed": int(pair_seed),
        "code_identity": dict(code_identity or {}),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / "summary" / "materialize_summary.json", summary)
    log(f"=== materialize end status={summary['status']} completed={summary['completed']} failed={summary['failed']} missing={summary['missing']} corrupt={summary['corrupt']} ===")
    log_handle.close()
    return summary


__all__ = [
    "MATERIALIZE_RUN_SUMMARY_SCHEMA",
    "MATERIALIZED_SCHEMA",
    "Phd2MaterializeError",
    "materialize_cohort",
    "materialize_one",
    "materialized_path",
    "validate_materialized",
]
