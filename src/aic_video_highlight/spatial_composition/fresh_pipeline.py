"""VHiCraft-v1 true-fresh deterministic core (Stage 5.6 corrective integration).

This module contains ONLY deterministic, GPU-free transformations that the fresh
VC-1/VC-A0 orchestrator composes.  It never re-implements a scientific method:

* frame projection / composition  -> ``composition_pipeline`` (CMP-1)
* temporal stabilization          -> ``temporal_smoothing`` (TS-5 Revised / TS-0)
* cache construction / replay     -> ``highlight_retrieval.candidate_cache``

The frozen Stage 4 candidate cache (``4b515a6d...``) is deliberately NOT
referenced anywhere in this module: fresh VC-1 builds its own provenance chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.spatial_composition.temporal_diagnostics import (
    observations_from_records,
)
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    SmoothedFrame,
    smooth_video_sequence_canonical_center_projected_state_bbox_guarded,
)

TS5_REVISED_METHOD = "projected_state_canonical_center_ema_v1"
TS0_METHOD = "stage5_3_final_frozen_cmp1"
FRESH_SHARD_SCHEMA_VERSION = "aic.vhicraft.fresh-stage5_4-shard/v1"
FRESH_CACHE_BINDING_SCHEMA_VERSION = "aic.vhicraft.fresh-cache-binding/v1"


class FreshPipelineError(RuntimeError):
    """Raised when a fresh-orchestration invariant is violated."""


def validate_fresh_candidate_cache_binding(
    manifest: Mapping[str, Any],
    *,
    stage1_predictions_path: Path,
    expected_video_ids: Sequence[str],
    forbidden_global_sha256: str,
) -> dict[str, Any]:
    """Prove that a candidate cache was built from this run's Stage 1 output.

    The cache builder already records the source predictions byte hash.  This
    gate makes that link mandatory and separately rejects the historical
    FROZEN_REPLAY cache identity.
    """
    global_sha = str(manifest.get("global_semantic_sha256", ""))
    if not global_sha or global_sha == forbidden_global_sha256:
        raise FreshPipelineError("fresh cache is missing or uses the frozen cache identity")
    source_sets = manifest.get("source_sets")
    if not isinstance(source_sets, list) or len(source_sets) != 1:
        raise FreshPipelineError("fresh cache must have exactly one Dev source set")
    source = source_sets[0]
    hashes = source.get("source_artifact_hashes", {})
    predictions_sha = file_sha256(stage1_predictions_path)
    if hashes.get("predictions_jsonl_sha256") != predictions_sha:
        raise FreshPipelineError("fresh cache does not consume this run's Stage 1 predictions")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise FreshPipelineError("fresh cache records are missing")
    actual_ids = [str(record.get("video_id")) for record in records]
    if actual_ids != [str(video_id) for video_id in expected_video_ids]:
        raise FreshPipelineError("fresh cache video membership/order mismatch")
    return {
        "schema_version": FRESH_CACHE_BINDING_SCHEMA_VERSION,
        "stage1_predictions_sha256": predictions_sha,
        "fresh_cache_global_semantic_sha256": global_sha,
        "record_count": len(actual_ids),
        "downstream_consumer": "stage4_candidate_cache",
        "frozen_fallback": False,
    }


def build_fresh_index(
    role_records: Sequence[Mapping[str, Any]],
    frozen_index_entries: Sequence[Mapping[str, Any]],
    *,
    video_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Project frozen Dev membership onto a fresh Stage 5 index.

    Schema transform only: membership + ``video_path`` / ``targetRatioWH`` are
    copied verbatim from the frozen index; no scientific selection is performed.
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in frozen_index_entries:
        video_id = str(entry["video_id"])
        if video_id in by_id:
            raise FreshPipelineError(f"duplicate frozen index video_id: {video_id}")
        by_id[video_id] = entry
    order = [str(record["video_id"]) for record in role_records]
    if video_ids is not None:
        allowed = {str(value) for value in video_ids}
        order = [video_id for video_id in order if video_id in allowed]
    seen: set[str] = set()
    index: list[dict[str, Any]] = []
    for video_id in order:
        if video_id in seen:
            raise FreshPipelineError(f"duplicate role video_id: {video_id}")
        seen.add(video_id)
        entry = by_id.get(video_id)
        if entry is None:
            raise FreshPipelineError(f"role video missing from frozen index: {video_id}")
        if "targetRatioWH" not in entry:
            raise FreshPipelineError(f"frozen index entry lacks targetRatioWH: {video_id}")
        index.append(dict(entry))
    if not index:
        raise FreshPipelineError("fresh index is empty")
    return index


def build_ts5_by_frame(
    composed_records: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    target_ratio: Sequence[int | float],
    alpha: float = 0.5,
) -> dict[int, SmoothedFrame]:
    """Apply the canonical TS-5 Revised smoother to one video's CMP-1 records."""
    if not composed_records:
        raise FreshPipelineError("cannot stabilize an empty frame sequence")
    records = sorted(composed_records, key=lambda item: int(item["frame"]))
    observations = observations_from_records(
        [
            {"frame": record["frame"], "ts0": record["cmp1"], "sanitized": record["sanitized"]}
            for record in records
        ]
    )
    primary_bboxes = {
        int(record["frame"]): tuple(float(value) for value in record["sanitized"]["xyxy"])
        for record in records
        if not record["cmp1"]["fallback"]
    }
    smoothed = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        int(width),
        int(height),
        float(target_ratio[0]),
        float(target_ratio[1]),
        observations,
        primary_bboxes,
        alpha=float(alpha),
    )
    by_frame = {item.frame: item for item in smoothed}
    if sorted(by_frame) != [int(record["frame"]) for record in records]:
        raise FreshPipelineError("TS-5 Revised changed the frame set")
    return by_frame


def build_shard_records(
    composed_records: Sequence[Mapping[str, Any]],
    ts5_by_frame: Mapping[int, SmoothedFrame],
) -> list[dict[str, Any]]:
    """Serialize one video's fresh Stage 5.4 shard (``ts0`` = CMP-1, ``ts5`` = TS-5 Revised).

    Schema-compatible with ``vhicraft_pipeline.load_stage5_4_shard`` so Stage 5.6
    can consume fresh geometry exactly like frozen geometry.
    """
    records: list[dict[str, Any]] = []
    for record in sorted(composed_records, key=lambda item: int(item["frame"])):
        frame = int(record["frame"])
        cmp1 = record["cmp1"]
        smoothed = ts5_by_frame.get(frame)
        if smoothed is None:
            raise FreshPipelineError(f"missing TS-5 Revised output for frame {frame}")
        records.append(
            {
                "schema_version": FRESH_SHARD_SCHEMA_VERSION,
                "frame": frame,
                "ts0": {
                    "x": int(cmp1["x"]),
                    "y": int(cmp1["y"]),
                    "w": int(cmp1["w"]),
                    "h": float(cmp1["h"]),
                    "fallback": bool(cmp1["fallback"]),
                    "placement_status": str(cmp1["placement_status"]),
                },
                "ts5": {
                    "x": int(smoothed.x),
                    "y": int(smoothed.y),
                    "w": int(smoothed.w),
                    "h": float(smoothed.h),
                    "fallback": bool(cmp1["fallback"]),
                    "placement_status": str(smoothed.placement_status),
                },
            }
        )
    frames = [int(item["frame"]) for item in records]
    if len(frames) != len(set(frames)):
        raise FreshPipelineError("fresh shard contains duplicate frames")
    return records


def shared_upstream_identity(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Machine-readable proof that VC-1 and VC-A0 consumed identical upstream."""
    if not paths:
        raise FreshPipelineError("shared upstream identity requires at least one artifact")
    artifacts = {
        str(name): file_sha256(Path(path))
        for name, path in sorted(paths.items(), key=lambda item: str(item[0]))
    }
    return {
        "shared": True,
        "artifacts": artifacts,
        "shared_upstream_sha256": canonical_sha256(artifacts),
    }


@dataclass(frozen=True, slots=True)
class FreshShardAudit:
    video_id: str
    frame_count: int
    ts0_sha256: str
    ts5_sha256: str
    frames_sha256: str


def audit_fresh_shard(video_id: str, shard_records: Sequence[Mapping[str, Any]]) -> FreshShardAudit:
    """Deterministic content audit for resume/identity propagation."""
    ordered = sorted(shard_records, key=lambda item: int(item["frame"]))
    frames = [int(item["frame"]) for item in ordered]
    if len(frames) != len(set(frames)):
        raise FreshPipelineError(f"duplicate frame in fresh shard audit: {video_id}")
    ts0 = [
        [int(item["ts0"]["x"]), int(item["ts0"]["y"]), int(item["ts0"]["w"])]
        for item in ordered
    ]
    ts5 = [
        [int(item["ts5"]["x"]), int(item["ts5"]["y"]), int(item["ts5"]["w"])]
        for item in ordered
    ]
    return FreshShardAudit(
        video_id=video_id,
        frame_count=len(frames),
        ts0_sha256=canonical_sha256(ts0),
        ts5_sha256=canonical_sha256(ts5),
        frames_sha256=canonical_sha256(frames),
    )
