#!/usr/bin/env python3
"""Stage 5.5 Frame-Level Prediction Calibration runner (preregistration-ready).

This development HEAD ships the frozen preregistration, three arms, the
weak-frame temporal-proxy evaluator, dev guardrails, the immutable Dev->Hard
promotion marker and the conditional Hard229 confirmation.  It intentionally
does NOT execute any real experiment on Windows: ``--execute`` is only honored
on a non-Windows (AutoDL) host, and the Hard runner is fail-closed unless the
exact immutable Dev promotion marker exists.

Registered experiment ids:
- ``stage5_5_dev_formal``         : Dev166 FS-0 / FS-1 / FS-2 (AutoDL CPU)
- ``stage5_5_hard_confirmation``  : Hard229 FS-0 vs frozen Dev winner (AutoDL CPU)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from aic_video_highlight.experiment_runtime.hashing import (
    canonical_sha256,
    file_sha256,
)
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputError,
)
from aic_video_highlight.spatial_composition.frame_calibration_metrics import (
    adjudicate_stage5_5,
    evaluate_video_set,
    hard_gates,
    select_dev_winner,
)
from fractions import Fraction

from aic_video_highlight.spatial_composition.frame_projection import (
    VideoTiming,
    frames_in_segment,
)
from aic_video_highlight.spatial_composition.frame_selection import (
    ARM_NAMES,
    FS0,
    FS1,
    FS2,
    POLICIES,
    ChunkWindow,
    FinalSegment,
    FrameSelection,
    RawCandidateSpan,
    select_emit_frames,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE5_CONFIGS = REPO_ROOT / "configs" / "experiments" / "stage5"

MASTER_PREREGISTRATION_PATH = STAGE5_CONFIGS / "stage5_5_master_preregistration.json"
DEV_PROTOCOL_PATH = STAGE5_CONFIGS / "stage5_5_dev_protocol.json"
HARD_PROTOCOL_PATH = STAGE5_CONFIGS / "stage5_5_hard_protocol.json"
FS0_CONFIG_PATH = STAGE5_CONFIGS / "stage5_5_fs0_all_frames.json"
FS1_CONFIG_PATH = STAGE5_CONFIGS / "stage5_5_fs1_singleton_edge_prune.json"
FS2_CONFIG_PATH = STAGE5_CONFIGS / "stage5_5_fs2_nonunanimous_edge_prune.json"
DEV_CONFIG_PATH = STAGE5_CONFIGS / "stage5_5_dev_formal.json"
HARD_CONFIG_PATH = STAGE5_CONFIGS / "stage5_5_hard_confirmation.json"

STAGE5_5_METHOD = "frame_level_prediction_calibration_v1"
FROZEN_TEMPORAL_BASELINE = "projected_state_canonical_center_ema_v1"
PROXY_LABEL = "weak-frame temporal proxy (Macro-F1); NOT an official score"
MASTER_STATUS = "PREREGISTERED_BEFORE_ANY_STAGE5_5_EXPERIMENT"
PROTOCOL_SCHEMA_VERSION = "aic.stage5.5-experiment-protocol/v1"
ARM_CONFIG_SCHEMA_VERSION = "aic.stage5.5-arm-config/v1"
EXECUTION_CONFIG_SCHEMA_VERSION = "aic.stage5.5-execution-config/v1"
DEV_MARKER_SCHEMA_VERSION = "aic.stage5.5-dev-promotion-marker/v1"

FROZEN_CANDIDATE_CACHE_GLOBAL_SHA256 = (
    "4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246"
)
DEV166_WEAK_REFERENCE_SHA256 = (
    "702f0bf1c74c33b3240a1d24409640d7b4bab8d62a4b84a1857838c6dcb5d5e2"
)

STAGE5_5_REPORT_SECTIONS = (
    "Stage 5.5 Scientific Question",
    "Frozen Inputs",
    "Arms",
    "Dev166 Protocol",
    "Hard229 Conditional Protocol",
    "Weak-Frame Proxy Metrics",
    "Dev166 Result",
    "Hard229 Result",
    "Final Decision",
    "Spatial Invariants",
    "Provenance",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenInputError(f"expected a JSON object: {path}")
    return payload


def _semantic_sha256(payload: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != field})


def _expected_fs_definitions() -> dict[str, dict[str, Any]]:
    """Canonical arm definitions shared by the protocol, configs and runner."""
    return {
        FS0: {
            "policy": FS0,
            "arm": "all_frames_v1",
            "description": "identity control: keep every candidate frame; bbox unchanged",
            "drop_condition": None,
            "edge_only": False,
            "internal_holes": "KEEP",
            "extra_hyperparameters": [],
        },
        FS1: {
            "policy": FS1,
            "arm": "cross_chunk_singleton_edge_prune_v1",
            "description": (
                "drop left/right edge-connected runs where E_t >= 2 and S_t == 1, "
                "only when a higher-consensus core (S_t >= 2) exists"
            ),
            "drop_condition": "E_t >= 2 and S_t == 1",
            "core_requirement": "exists frame with S_t >= 2",
            "edge_only": True,
            "internal_holes": "KEEP",
            "all_singleton_segment": "KEEP_ALL",
            "extra_hyperparameters": [],
        },
        FS2: {
            "policy": FS2,
            "arm": "cross_chunk_nonunanimous_edge_prune_v1",
            "description": (
                "drop left/right edge-connected runs where E_t >= 2 and S_t < E_t, "
                "only when a unanimity core (S_t == E_t) exists"
            ),
            "drop_condition": "E_t >= 2 and S_t < E_t",
            "core_requirement": "exists frame with S_t == E_t",
            "edge_only": True,
            "internal_holes": "KEEP",
            "no_unanimity_core": "KEEP_ALL",
            "extra_hyperparameters": [],
        },
    }


def _arm_config_path(policy: str) -> Path:
    return {FS0: FS0_CONFIG_PATH, FS1: FS1_CONFIG_PATH, FS2: FS2_CONFIG_PATH}[policy]


# ---------------------------------------------------------------------------
# Preregistration identity validation
# ---------------------------------------------------------------------------

def validate_protocol(protocol: Mapping[str, Any], *, expected_status: str) -> dict[str, Any]:
    if protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise FrozenInputError("unsupported Stage 5.5 protocol schema")
    if protocol.get("status") != expected_status:
        raise FrozenInputError(
            f"protocol status must be {expected_status!r} at this HEAD"
        )
    claimed = protocol.get("protocol_semantic_sha256")
    if claimed != _semantic_sha256(protocol, "protocol_semantic_sha256"):
        raise FrozenInputError("protocol semantic hash mismatch")
    policies = protocol.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != set(POLICIES):
        raise FrozenInputError("protocol must register exactly FS-0/FS-1/FS-2")
    for policy, definition in _expected_fs_definitions().items():
        if policies[policy] != definition:
            raise FrozenInputError(f"protocol definition drift for {policy}")
    return {
        "protocol_id": protocol.get("protocol_id"),
        "status": expected_status,
        "protocol_semantic_sha256": claimed,
    }


def validate_arm_config(policy: str) -> dict[str, Any]:
    config = _read_json(_arm_config_path(policy))
    if config.get("arm_config_schema_version") != ARM_CONFIG_SCHEMA_VERSION:
        raise FrozenInputError(f"unsupported arm config schema for {policy}")
    if config.get("definition") != _expected_fs_definitions()[policy]:
        raise FrozenInputError(f"arm config definition drift for {policy}")
    protocol_path = REPO_ROOT / str(config["dev_protocol"])
    if config.get("protocol_sha256") != file_sha256(protocol_path):
        raise FrozenInputError(f"arm config protocol byte hash mismatch for {policy}")
    protocol = _read_json(protocol_path)
    if config.get("protocol_semantic_sha256") != protocol.get("protocol_semantic_sha256"):
        raise FrozenInputError(f"arm config protocol semantic hash mismatch for {policy}")
    return config


def validate_master_preregistration() -> dict[str, Any]:
    master = _read_json(MASTER_PREREGISTRATION_PATH)
    if master.get("status") != MASTER_STATUS:
        raise FrozenInputError("Stage 5.5 master preregistration status is wrong")
    artifacts = master.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FrozenInputError("master preregistration artifacts are missing")
    for name, binding in artifacts.items():
        if name in {"dev_protocol", "hard_protocol"}:
            path = REPO_ROOT / str(binding["path"])
            if binding["byte_sha256"] != file_sha256(path):
                raise FrozenInputError(f"master protocol byte hash mismatch: {name}")
            protocol = _read_json(path)
            if binding["semantic_sha256"] != protocol.get("protocol_semantic_sha256"):
                raise FrozenInputError(f"master protocol semantic hash mismatch: {name}")
        elif name in {"fs0", "fs1", "fs2", "dev_formal", "hard_confirmation"}:
            path = REPO_ROOT / str(binding["path"])
            if binding["byte_sha256"] != file_sha256(path):
                raise FrozenInputError(f"master config byte hash mismatch: {name}")
    return master


def validate_execution_config(config: Mapping[str, Any], protocol_path: Path) -> dict[str, Any]:
    if config.get("execution_config_schema_version") != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise FrozenInputError("unsupported Stage 5.5 execution config schema")
    if config.get("protocol_sha256") != file_sha256(protocol_path):
        raise FrozenInputError("execution config protocol byte hash mismatch")
    protocol = _read_json(protocol_path)
    if config.get("protocol_semantic_sha256") != protocol.get("protocol_semantic_sha256"):
        raise FrozenInputError("execution config protocol semantic hash mismatch")
    return {"experiment_id": config.get("experiment_id"), "protocol": protocol}


# ---------------------------------------------------------------------------
# Frozen input resolution / preflight
# ---------------------------------------------------------------------------

def _resolve(base: str, path: str, environment: EnvironmentPaths) -> Path:
    if base == "repo":
        return REPO_ROOT / path
    if base == "outputs":
        return environment.outputs / path
    if base == "datasets":
        return environment.datasets / path
    raise FrozenInputError(f"unknown input base: {base}")


def preflight(config: Mapping[str, Any], environment: EnvironmentPaths) -> dict[str, Any]:
    """Verify declared frozen inputs resolve; never fabricate or recompute science."""
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, spec in config.get("inputs", {}).items():
        path = _resolve(str(spec["base"]), str(spec["path"]), environment)
        entry = {"name": name, "path": str(path), "exists": path.exists()}
        if spec.get("sha256"):
            entry["expected_sha256"] = spec["sha256"]
            if path.is_file():
                entry["actual_sha256"] = file_sha256(path)
                entry["sha256_match"] = entry["actual_sha256"] == spec["sha256"]
        if not path.exists():
            missing.append(name)
        resolved.append(entry)
    return {
        "status": "PASS" if not missing else "FAIL",
        "missing_inputs": missing,
        "resolved_inputs": resolved,
    }


# ---------------------------------------------------------------------------
# Dev -> Hard immutable promotion marker
# ---------------------------------------------------------------------------

def _marker_path(environment: EnvironmentPaths) -> Path:
    return (
        environment.outputs
        / "stage5"
        / "stage5_5_dev_formal"
        / "machine"
        / "dev_promotion_marker.json"
    )


def build_dev_promotion_marker(
    *,
    execution_head: str,
    dev_protocol_sha256: str,
    dev_config_sha256: str,
    evaluation_sha256: str,
    selected_candidate: str,
    guard_results: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_candidate not in {FS1, FS2}:
        raise FrozenInputError("a Dev promotion marker may only freeze FS-1 or FS-2")
    marker: dict[str, Any] = {
        "marker_schema_version": DEV_MARKER_SCHEMA_VERSION,
        "method": STAGE5_5_METHOD,
        "frozen_temporal_baseline": FROZEN_TEMPORAL_BASELINE,
        "execution_head": execution_head,
        "dev_protocol_sha256": dev_protocol_sha256,
        "dev_config_sha256": dev_config_sha256,
        "evaluation_sha256": evaluation_sha256,
        "selected_candidate": selected_candidate,
        "selected_arm": ARM_NAMES[selected_candidate],
        "guard_results": dict(guard_results),
        "all_pass": bool(guard_results.get("all_pass")) and selected_candidate in {FS1, FS2},
        "hard_candidate_immutable": True,
        "override_allowed": False,
        "amendment_allowed": False,
    }
    marker["marker_semantic_sha256"] = canonical_sha256(marker)
    return marker


def load_dev_promotion_marker(
    config: Mapping[str, Any], environment: EnvironmentPaths
) -> dict[str, Any]:
    """Fail-closed Hard229 gate: require the exact immutable Dev winner marker."""
    binding = config.get("dev_promotion_marker")
    if not isinstance(binding, Mapping):
        raise FrozenInputError("Hard config lacks a dev_promotion_marker binding")
    marker_path = _resolve(str(binding["base"]), str(binding["path"]), environment)
    if not marker_path.is_file():
        raise FrozenInputError(
            "BLOCKED_BEFORE_EXECUTION: immutable Dev promotion marker is absent"
        )
    marker = _read_json(marker_path)
    if marker.get("marker_schema_version") != DEV_MARKER_SCHEMA_VERSION:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: wrong Dev marker schema")
    if marker.get("method") != STAGE5_5_METHOD:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: Dev marker method mismatch")
    if marker.get("all_pass") is not True:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: Dev marker did not pass")
    if marker.get("selected_candidate") not in {FS1, FS2}:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: Dev marker candidate is invalid")
    claimed = marker.get("marker_semantic_sha256")
    if claimed != _semantic_sha256(marker, "marker_semantic_sha256"):
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: Dev marker hash mismatch")
    return marker


# ---------------------------------------------------------------------------
# Frozen artifact adapters (used only by a real AutoDL execution)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrozenVideoInputs:
    video_id: str
    timing: VideoTiming
    segments: tuple[FinalSegment, ...]
    chunk_windows: tuple[ChunkWindow, ...]
    raw_spans: tuple[RawCandidateSpan, ...]
    reference_frames: tuple[int, ...]
    ts5_frames: tuple[int, ...]
    ts5_bboxes: tuple[tuple[int, int, int, int, int], ...]


def parse_chunk_evidence(
    record: Mapping[str, Any],
) -> tuple[tuple[ChunkWindow, ...], tuple[RawCandidateSpan, ...]]:
    """Extract E_t/S_t evidence from one frozen Stage 4 candidate cache record."""
    windows = tuple(
        ChunkWindow(
            chunk_index=int(chunk["chunk_index"]),
            start_sec=float(chunk["chunk_start_sec"]),
            end_sec=float(chunk["chunk_end_sec"]),
        )
        for chunk in record["source_chunks"]
    )
    spans = tuple(
        RawCandidateSpan(
            chunk_index=int(candidate["chunk_index"]),
            start_sec=float(candidate["start_sec"]),
            end_sec=float(candidate["end_sec"]),
        )
        for candidate in record["raw_candidates"]
    )
    return windows, spans


def final_segments_from_record(
    record: Mapping[str, Any], selected_candidate_ids: Sequence[str] | None = None
) -> tuple[FinalSegment, ...]:
    """Build frozen final segments from merged candidates (optionally post-selection)."""
    selected = set(selected_candidate_ids) if selected_candidate_ids is not None else None
    segments = []
    for candidate in record["merged_candidates"]:
        candidate_id = str(candidate["merged_candidate_id"])
        if selected is not None and candidate_id not in selected:
            continue
        segments.append(
            FinalSegment(
                segment_id=candidate_id,
                start_sec=float(candidate["start_sec"]),
                end_sec=float(candidate["end_sec"]),
            )
        )
    return tuple(segments)


def run_frame_selection(
    inputs: FrozenVideoInputs, policy: str
) -> FrameSelection:
    return select_emit_frames(
        policy,
        inputs.segments,
        inputs.timing,
        inputs.chunk_windows,
        inputs.raw_spans,
    )


def evaluate_arm_on_video_set(
    videos: Sequence[FrozenVideoInputs], policy: str
) -> dict[str, object]:
    predictions: dict[str, list[int]] = {}
    references: dict[str, list[int]] = {}
    for video in videos:
        selection = run_frame_selection(video, policy)
        frozen = set(video.ts5_frames)
        emitted = tuple(frame for frame in selection.emitted_frames if frame in frozen)
        if len(emitted) != len(selection.emitted_frames):
            raise FrozenInputError(
                f"policy {policy} emitted a frame without a frozen TS-5 bbox: {video.video_id}"
            )
        predictions[video.video_id] = list(emitted)
        references[video.video_id] = list(video.reference_frames)
    return evaluate_video_set(predictions, references)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_stage5_5_report(
    output: Path,
    *,
    config: Mapping[str, Any],
    validation: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
) -> None:
    lines = [
        f"# Stage 5.5 Frame-Level Prediction Calibration: {config.get('experiment_id', '?')}",
        "",
        f"> {PROXY_LABEL}",
        "",
        "> Machine-generated factual evidence. Scientific interpretation is excluded.",
        "",
    ]
    body = {
        "validation": validation,
        "metrics": metrics,
        "runtime": runtime,
    }
    for section in STAGE5_5_REPORT_SECTIONS:
        lines.extend([f"## {section}", ""])
        key = section.split()[0].lower()
        if section == "Stage 5.5 Scientific Question":
            lines.extend(
                [
                    "Which candidate frames that already own a frozen Stage 5.4 "
                    "TS-5 Revised bbox should be emitted, judged only by frozen "
                    "cross-chunk temporal evidence?",
                    "",
                ]
            )
        elif section == "Final Decision":
            lines.extend(
                [
                    "```json",
                    json.dumps(validation.get("decision"), ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        else:
            payload = next((value for name, value in body.items() if name in key), None)
            if payload is not None:
                lines.extend(
                    ["```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""]
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_progress(machine_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write the resumable machine-readable progress snapshot."""
    _write_json(machine_dir / "progress.json", payload)


def write_final_freeze(output_dir: Path, adjudication: Mapping[str, Any]) -> None:
    """Persist the terminal Stage 5.5 freeze decision (never overwrites)."""
    path = output_dir / "final_freeze.json"
    if path.exists():
        raise FileExistsError(f"final freeze already written: {path}")
    _write_json(path, {"stage5_5_terminal": "STAGE5_5_CLOSED", **adjudication})


def render_official_contract_proxy(
    path: Path, emitted_frames_by_video: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    """Emit a clearly-labelled NON-official proxy over already-frozen bboxes.

    The proxy never re-derives geometry and never claims an official score; it
    only exposes which frozen Stage 5.4 TS-5 Revised frames survive the emit
    mask, so a future submission step cannot silently add or move frames.
    """
    lines: list[str] = []
    for video_id in sorted(emitted_frames_by_video):
        frames = [
            {
                "frame": int(item["frame"]),
                "x": int(item["x"]),
                "y": int(item["y"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
            }
            for item in emitted_frames_by_video[video_id]
        ]
        lines.append(
            json.dumps(
                {
                    "video_id": video_id,
                    "frames": frames,
                    "proxy": True,
                    "proxy_label": PROXY_LABEL,
                    "official_format": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise FrozenInputError(f"invalid JSONL object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _timing_from_metadata(video_id: str, meta: Mapping[str, Any]) -> VideoTiming:
    mode = str(meta.get("timestamp_mode", "CFR_FPS"))
    pts = meta.get("pts_timestamps")
    return VideoTiming(
        video_id=video_id,
        fps=float(meta["fps"]),
        frame_count=int(meta["frame_count"]),
        timestamp_mode=mode,
        pts_timestamps=(
            tuple(Fraction(str(value)) for value in pts) if pts is not None else None
        ),
    )


def _frames_from_segments(
    segments: Sequence[Mapping[str, Any]], timing: VideoTiming
) -> tuple[int, ...]:
    frames: set[int] = set()
    for segment in segments:
        frames.update(
            frames_in_segment(float(segment["start_sec"]), float(segment["end_sec"]), timing)
        )
    return tuple(sorted(frames))


def load_stage5_5_inputs(
    config: Mapping[str, Any], environment: EnvironmentPaths
) -> tuple[FrozenVideoInputs, ...]:
    """Load the frozen Stage 4 / 5.1 / 5.4 provenance chain for one analysis set."""
    specs = config["inputs"]

    def resolved(name: str) -> Path:
        spec = specs[name]
        return _resolve(str(spec["base"]), str(spec["path"]), environment)

    cache_dir = resolved("frozen_candidate_cache")
    manifest = _read_json(cache_dir / "cache_manifest.json")
    if manifest.get("global_semantic_sha256") != FROZEN_CANDIDATE_CACHE_GLOBAL_SHA256:
        raise FrozenInputError("frozen candidate cache global hash mismatch")
    records = {
        entry["video_id"]: _read_json(cache_dir / entry["path"])
        for entry in manifest["records"]
    }
    role_manifest = _read_json(resolved("role_manifest"))
    metadata = _read_json(resolved("stage5_1_metadata_cache"))
    predictions = {
        row["video_id"]: row for row in _load_jsonl(resolved("stage5_1_predictions"))
    }
    ts5_rows = {
        row["video_id"]: row
        for row in _load_jsonl(resolved("stage5_4_ts5_revised_predictions"))
    }

    videos: list[FrozenVideoInputs] = []
    for identity in role_manifest["records"]:
        video_id = str(identity["video_id"])
        if video_id not in records:
            raise FrozenInputError(f"role video missing from frozen cache: {video_id}")
        record = records[video_id]
        windows, spans = parse_chunk_evidence(record)
        timing = _timing_from_metadata(video_id, metadata[video_id])
        prediction = predictions[video_id]
        segments = tuple(
            FinalSegment(
                segment_id=f"{video_id}#seg{index}",
                start_sec=float(segment["start_sec"]),
                end_sec=float(segment["end_sec"]),
            )
            for index, segment in enumerate(prediction.get("merged_prediction_segments", []))
        )
        reference = _frames_from_segments(
            prediction.get("weak_reference_segments", []), timing
        )
        ts5 = ts5_rows[video_id]
        frames: list[int] = []
        bboxes: list[tuple[int, int, int, int, int]] = []
        for item in ts5.get("frames", []):
            frames.append(int(item["frame"]))
            bboxes.append(
                (
                    int(item["frame"]),
                    int(item["x"]),
                    int(item["y"]),
                    int(item["w"]),
                    int(item["h"]),
                )
            )
        videos.append(
            FrozenVideoInputs(
                video_id=video_id,
                timing=timing,
                segments=segments,
                chunk_windows=windows,
                raw_spans=spans,
                reference_frames=reference,
                ts5_frames=tuple(frames),
                ts5_bboxes=tuple(bboxes),
            )
        )
    return tuple(videos)


def _stage5_5_paths(config: Mapping[str, Any], environment: EnvironmentPaths) -> tuple[Path, Path]:
    run_id = str(config["output_run_id"])
    output = environment.outputs / "stage5" / run_id
    return output, output / "machine"


def _write_stage5_5_artifacts(
    output_dir: Path,
    machine_dir: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    _write_json(machine_dir / "summary.json", summary)
    _write_json(machine_dir / "metrics.json", metrics)
    _write_json(machine_dir / "validation.json", validation)
    _write_json(machine_dir / "decision.json", validation["decision"])
    render_stage5_5_report(
        output_dir / "experiment_report.md",
        config=config,
        validation=validation,
        metrics=metrics,
        runtime={"heldout_access": 0, "official_test_access": 0},
    )


def _run_dev(config: Mapping[str, Any], environment: EnvironmentPaths, *, execution_head: str) -> int:
    preflight_result = preflight(config, environment)
    if preflight_result["status"] != "PASS":
        raise FrozenInputError(f"preflight failed: {preflight_result['missing_inputs']}")
    videos = load_stage5_5_inputs(config, environment)
    evaluations = {
        policy: evaluate_arm_on_video_set(videos, policy) for policy in POLICIES
    }
    baseline = {**evaluations[FS0], "arm": ARM_NAMES[FS0]}
    candidates = {
        policy: {**evaluations[policy], "arm": ARM_NAMES[policy]}
        for policy in (FS1, FS2)
    }
    selection = select_dev_winner(baseline, candidates)
    winner = selection["winner_policy"]
    decision = adjudicate_stage5_5(winner, None)
    if winner is not None:
        guard = candidates[winner]["objective"]["guardrails"]  # type: ignore[index]
        marker = build_dev_promotion_marker(
            execution_head=execution_head,
            dev_protocol_sha256=file_sha256(DEV_PROTOCOL_PATH),
            dev_config_sha256=file_sha256(DEV_CONFIG_PATH),
            evaluation_sha256=canonical_sha256(candidates[winner]),
            selected_candidate=winner,
            guard_results={
                "all_pass": bool(guard["all_pass"]),
                "checks": guard["checks"],
            },
        )
        output_dir, machine_dir = _stage5_5_paths(config, environment)
        _write_json(machine_dir / "dev_promotion_marker.json", marker)
    output_dir, machine_dir = _stage5_5_paths(config, environment)
    validation = {
        "status": "PASS",
        "proxy_label": PROXY_LABEL,
        "decision": decision,
        "preflight": preflight_result,
    }
    summary = {"policies": list(POLICIES), "selection": selection}
    metrics = {policy: evaluations[policy] for policy in POLICIES}
    _write_stage5_5_artifacts(
        output_dir, machine_dir, config=config, summary=summary, metrics=metrics, validation=validation
    )
    return 0


def _run_hard(config: Mapping[str, Any], environment: EnvironmentPaths, *, execution_head: str) -> int:
    marker = load_dev_promotion_marker(config, environment)
    preflight_result = preflight(config, environment)
    if preflight_result["status"] != "PASS":
        raise FrozenInputError(f"preflight failed: {preflight_result['missing_inputs']}")
    winner = str(marker["selected_candidate"])
    videos = load_stage5_5_inputs(config, environment)
    baseline = evaluate_arm_on_video_set(videos, FS0)
    candidate = evaluate_arm_on_video_set(videos, winner)
    result = hard_gates(baseline, candidate)
    decision = adjudicate_stage5_5(winner, result)
    output_dir, machine_dir = _stage5_5_paths(config, environment)
    validation = {
        "status": "PASS",
        "proxy_label": PROXY_LABEL,
        "decision": decision,
        "dev_promotion_marker": marker,
        "preflight": preflight_result,
    }
    summary = {"dev_winner": winner, "hard_gates": result}
    metrics = {FS0: baseline, winner: candidate}
    _write_stage5_5_artifacts(
        output_dir, machine_dir, config=config, summary=summary, metrics=metrics, validation=validation
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    environment = EnvironmentPaths.from_json(args.environment)
    experiment_id = str(config.get("experiment_id", ""))
    is_hard = experiment_id == "stage5_5_hard_confirmation"
    protocol_path = HARD_PROTOCOL_PATH if is_hard else DEV_PROTOCOL_PATH
    validate_master_preregistration()
    protocol = _read_json(protocol_path)
    validate_protocol(protocol, expected_status="PREREGISTERED_BEFORE_DEV166")
    validate_execution_config(config, protocol_path)
    if not is_hard:
        for policy in POLICIES:
            validate_arm_config(policy)
    if is_hard:
        # Hard is fail-closed before any execution attempt.
        load_dev_promotion_marker(config, environment)

    if args.validate_only or args.dry_run or not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "status": "VALIDATED_BEFORE_EXECUTION",
                    "executed": False,
                    "note": "development HEAD: no real Stage 5.5 experiment was run",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if platform.system() == "Windows":
        raise FrozenInputError(
            "Stage 5.5 formal execution is AutoDL-only and is refused on Windows"
        )
    execution_head = str(config.get("_execution_head", "")) or "UNKNOWN"
    return (
        _run_hard(config, environment, execution_head=execution_head)
        if is_hard
        else _run_dev(config, environment, execution_head=execution_head)
    )


if __name__ == "__main__":
    raise SystemExit(main())
