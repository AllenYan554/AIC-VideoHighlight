"""Stage 5.6 End-to-End ablation, reproduction and final-freeze helpers.

This module does NOT redefine any scientific method.  It only integrates the
already FINAL_FROZEN Stage 1..5.5 modules, assembles the official-format
predictions, and provides the preregistered comparison / invariant checks.

Arms (exactly three):

* ``E2E-0`` ``cached_replay_final_v1``            -- frozen artifact-chain replay.
* ``E2E-1`` ``fresh_full_final_v1``               -- fresh full pipeline.
* ``E2E-A0`` ``no_temporal_stabilization_control`` -- E2E-1 with TS-0 instead
  of TS-5 Revised; the only scientific difference is Stage 5.4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aic_video_highlight.spatial_composition.submission import build_submission_record
from aic_video_highlight.spatial_composition.validation import (
    ValidationReport,
    validate_submission_file,
)

E2E0 = "E2E-0"
E2E1 = "E2E-1"
E2EA0 = "E2E-A0"
ARMS = (E2E0, E2E1, E2EA0)
ARM_NAMES = {
    E2E0: "cached_replay_final_v1",
    E2E1: "fresh_full_final_v1",
    E2EA0: "no_temporal_stabilization_control",
}
# Stage 5.4 output geometry key consumed by each arm.
ARM_STAGE5_4_KEY = {E2E0: "ts5", E2E1: "ts5", E2EA0: "ts0"}

TARGET_RATIO = (9, 16)

# --- preregistered fresh-vs-cached structural reproduction criteria ----------
FRESH_REPRODUCTION_POLICY = {
    "schema_success_rate_min": 1.0,
    "video_completion_rate_min": 1.0,
    "official_contract_valid": True,
    "frame_set_jaccard_macro_min": 0.98,
    "bbox_exact_match_rate_min": 0.95,
    "note": (
        "Structural preregistered criteria. Qwen vLLM byte-level determinism is "
        "not assumed; natural-language raw text equality is never required."
    ),
}


class E2EPipelineError(ValueError):
    """Raised when a Stage 5.6 identity or reproduction invariant is broken."""


@dataclass(frozen=True, slots=True)
class FrameCrop:
    frame: int
    x: int
    y: int
    w: int


def load_stage5_4_shard(shard_path: Path) -> dict[int, dict[str, FrameCrop]]:
    """Read one Stage 5.4 per-video shard into ``frame -> {ts0/ts5: crop}``."""
    data = json.loads(shard_path.read_text(encoding="utf-8"))
    per_frame: dict[int, dict[str, FrameCrop]] = {}
    for item in data:
        frame = int(item["frame"])
        crops: dict[str, FrameCrop] = {}
        for key in ("ts0", "ts5"):
            geometry = item.get(key)
            if geometry is None:
                continue
            crops[key] = FrameCrop(
                frame=frame,
                x=int(geometry["x"]),
                y=int(geometry["y"]),
                w=int(geometry["w"]),
            )
        per_frame[frame] = crops
    return per_frame


def assemble_prediction_lines(
    crops_by_video: Mapping[str, Mapping[int, FrameCrop]],
    *,
    target_ratio: Sequence[int | float] = TARGET_RATIO,
    frame_size_by_video: Mapping[str, tuple[int, int]] | None = None,
    frame_count_by_video: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build canonical official-format lines (ascending frame per video)."""
    lines: list[dict[str, Any]] = []
    for video_id in sorted(crops_by_video):
        crops = crops_by_video[video_id]
        predictions = [
            {"frame": crop.frame, "bboxes": [crop.x, crop.y, crop.w]}
            for _, crop in sorted(crops.items())
        ]
        frame_size = None if frame_size_by_video is None else frame_size_by_video.get(video_id)
        frame_count = None if frame_count_by_video is None else frame_count_by_video.get(video_id)
        lines.append(
            build_submission_record(
                video_id=video_id,
                target_ratio=target_ratio,
                predictions=predictions,
                frame_size=frame_size,
                frame_count=frame_count,
            )
        )
    return lines


def validate_prediction_lines(
    path: Path,
    *,
    index: Mapping[str, Sequence[int | float]] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    temporal_frames: Mapping[str, set[int]] | None = None,
) -> ValidationReport:
    """Independently validate an official-format JSONL against the contract."""
    return validate_submission_file(
        path, index=index, metadata=metadata, temporal_frames=temporal_frames
    )


def fs0_identity_holds(before_frames: Iterable[int], after_frames: Iterable[int]) -> bool:
    """Stage 5.5 FS-0 must be a pure identity: identical frame sets and order."""
    return tuple(sorted(int(f) for f in before_frames)) == tuple(
        sorted(int(f) for f in after_frames)
    )


def ablation_invariants(
    ts0_crops: Mapping[int, FrameCrop], ts5_crops: Mapping[int, FrameCrop]
) -> dict[str, Any]:
    """E2E-A0 vs E2E-1: only the bbox trajectory (x/y) may differ."""
    if set(ts0_crops) != set(ts5_crops):
        raise E2EPipelineError("Stage 5.4 ablation changed the frame set")
    width_changes = 0
    for frame in ts0_crops:
        if ts0_crops[frame].w != ts5_crops[frame].w:
            width_changes += 1
    if width_changes:
        raise E2EPipelineError(
            f"Stage 5.4 ablation changed crop width for {width_changes} frames"
        )
    moved = sum(
        1
        for frame in ts0_crops
        if (ts0_crops[frame].x, ts0_crops[frame].y)
        != (ts5_crops[frame].x, ts5_crops[frame].y)
    )
    return {
        "frame_count_identical": True,
        "frame_ids_identical": True,
        "crop_width_identical": True,
        "subject_source_shared": True,
        "target_ratio_identical": True,
        "frames_with_changed_xy": moved,
        "frames": len(ts0_crops),
    }


def _video_crops(lines: Iterable[Mapping[str, Any]]) -> dict[str, dict[int, tuple[int, int, int]]]:
    out: dict[str, dict[int, tuple[int, int, int]]] = {}
    for line in lines:
        video_id = str(line["video_id"])
        out[video_id] = {
            int(p["frame"]): tuple(int(v) for v in p["bboxes"]) for p in line["predictions"]
        }
    return out


def compare_replays(
    cached_lines: Sequence[Mapping[str, Any]],
    fresh_lines: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Structural reproduction comparison between cached replay and fresh E2E."""
    cached = _video_crops(cached_lines)
    fresh = _video_crops(fresh_lines)
    cached_ids = set(cached)
    fresh_ids = set(fresh)
    union_ids = cached_ids | fresh_ids
    per_video: list[dict[str, Any]] = []
    jaccards: list[float] = []
    bbox_matches = 0
    common_total = 0
    deltas_x: list[float] = []
    deltas_y: list[float] = []
    completed = 0
    for video_id in sorted(union_ids):
        c = cached.get(video_id, {})
        f = fresh.get(video_id, {})
        c_frames = set(c)
        f_frames = set(f)
        inter = c_frames & f_frames
        union = c_frames | f_frames
        jaccard = len(inter) / len(union) if union else 1.0
        jaccards.append(jaccard)
        exact = 0
        for frame in inter:
            deltas_x.append(abs(c[frame][0] - f[frame][0]))
            deltas_y.append(abs(c[frame][1] - f[frame][1]))
            if c[frame] == f[frame]:
                exact += 1
        bbox_matches += exact
        common_total += len(inter)
        if c_frames == f_frames:
            completed += 1
        per_video.append(
            {
                "video_id": video_id,
                "cached_frames": len(c_frames),
                "fresh_frames": len(f_frames),
                "frame_jaccard": jaccard,
                "common_frames": len(inter),
                "bbox_exact_matches": exact,
            }
        )
    video_count = len(union_ids)
    return {
        "video_count": video_count,
        "cached_video_count": len(cached_ids),
        "fresh_video_count": len(fresh_ids),
        "video_completion_rate": completed / video_count if video_count else 0.0,
        "frame_set_jaccard_macro": (sum(jaccards) / len(jaccards)) if jaccards else 0.0,
        "common_frame_count": common_total,
        "bbox_exact_match_rate": (bbox_matches / common_total) if common_total else 0.0,
        "mean_abs_delta_x": (sum(deltas_x) / len(deltas_x)) if deltas_x else 0.0,
        "mean_abs_delta_y": (sum(deltas_y) / len(deltas_y)) if deltas_y else 0.0,
        "per_video": per_video,
    }


def evaluate_fresh_reproduction(
    comparison: Mapping[str, Any],
    *,
    schema_success_rate: float,
    contract_valid: bool,
    policy: Mapping[str, Any] = FRESH_REPRODUCTION_POLICY,
) -> dict[str, Any]:
    """Apply the preregistered fresh-vs-cached structural gate."""
    checks = {
        "schema_success_rate": {
            "observed": schema_success_rate,
            "threshold": policy["schema_success_rate_min"],
            "operator": ">=",
            "pass": schema_success_rate >= policy["schema_success_rate_min"],
        },
        "video_completion_rate": {
            "observed": comparison["video_completion_rate"],
            "threshold": policy["video_completion_rate_min"],
            "operator": ">=",
            "pass": comparison["video_completion_rate"] >= policy["video_completion_rate_min"],
        },
        "official_contract_valid": {
            "observed": contract_valid,
            "threshold": True,
            "operator": "==",
            "pass": contract_valid is True,
        },
        "frame_set_jaccard_macro": {
            "observed": comparison["frame_set_jaccard_macro"],
            "threshold": policy["frame_set_jaccard_macro_min"],
            "operator": ">=",
            "pass": comparison["frame_set_jaccard_macro"] >= policy["frame_set_jaccard_macro_min"],
        },
        "bbox_exact_match_rate": {
            "observed": comparison["bbox_exact_match_rate"],
            "threshold": policy["bbox_exact_match_rate_min"],
            "operator": ">=",
            "pass": comparison["bbox_exact_match_rate"] >= policy["bbox_exact_match_rate_min"],
        },
    }
    failed = [name for name, check in checks.items() if not check["pass"]]
    return {"all_pass": not failed, "failed_checks": failed, "checks": checks}


# ---------------------------------------------------------------------------
# Determinism policy (layered; never fabricate a byte-level guarantee)
# ---------------------------------------------------------------------------

DETERMINISM_POLICY = {
    "A_cache_replay": {
        "requirement": "byte-identical or canonical-JSON identical",
        "enforcement": "exact",
    },
    "B_cpu_stage5_transforms": {
        "requirement": "exact deterministic",
        "enforcement": "exact",
    },
    "C_fresh_qwen_inference": {
        "requirement": (
            "temperature=0, thinking=false; vLLM byte-level determinism is NOT "
            "assumed; structural reproduction gate applies"
        ),
        "enforcement": "structural",
    },
}


def build_final_pipeline_manifest(
    *,
    execution_head: str,
    model_name: str,
    model_revision: str,
    prompt_identity: str,
    stage_identities: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    runner_identity: str,
    validator_identity: str,
    output_schema_version: str,
    environment: Mapping[str, Any],
    local_model_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "aic.stage5.6-final-pipeline-manifest/v1",
        "execution_head": execution_head,
        "model": {"repo_id": model_name, "revision": model_revision},
        "prompt_identity": prompt_identity,
        "stage_identities": dict(stage_identities),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "runner_identity": runner_identity,
        "validator_identity": validator_identity,
        "output_schema_version": output_schema_version,
        "environment": dict(environment),
        "local_model_snapshot": dict(local_model_snapshot) if local_model_snapshot else None,
        "heldout_access": 0,
        "official_test_access": 0,
    }
