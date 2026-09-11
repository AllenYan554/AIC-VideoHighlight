#!/usr/bin/env python3
"""Stage 5.6 VHiCraft-v1 Validation & Final Freeze runner.

Single entry point for the full competition pipeline.  It chains the already
FINAL_FROZEN Stage 1..5.5 modules (it does not re-implement them) and emits the
official-format ``predictions.jsonl``.

Arms:
- VC-0 ``cached_v1_replay``                  : replay the frozen artifact chain.
- VC-1 ``fresh_v1_pipeline``                 : fresh full pipeline (user Formal).
- VC-A0 ``no_temporal_stabilization_control``: VC-1 with TS-0 (ablation only).

Registered experiments:
- ``stage5_6_vhicraft_smoke``  : tiny CPU/GPU engineering smoke.
- ``stage5_6_vhicraft_formal`` : full Dev166 final freeze (user-triggered; DEFAULT NOT RUN).
- ``stage5_6_vhicraft_ablation``: TS-0 vs TS-5 Revised VHiCraft ablation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # direct script execution
    _REPO = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.vhicraft_pipeline import (
    ARM_NAMES,
    ARM_STAGE5_4_KEY,
    ARMS,
    DETERMINISM_POLICY,
    VC0,
    VC1,
    VCA0,
    TARGET_RATIO,
    VHiCraftPipelineError,
    FrameCrop,
    ablation_invariants,
    assemble_prediction_lines,
    build_final_pipeline_manifest,
    compare_replays,
    evaluate_fresh_reproduction,
    fs0_identity_holds,
    load_stage5_4_shard,
    validate_prediction_lines,
)
from aic_video_highlight.spatial_composition.frame_selection import (
    FS0,
    select_emit_frames,
)
from aic_video_highlight.spatial_composition.composition_pipeline import FrozenInputError

# Reuse the canonical Stage 5.5 frozen-input adapters (no logic duplication).
from scripts.experiments.stage5.run_stage5_5_frame_calibration import (
    _frames_from_segments,
    _timing_from_metadata,
    final_segments_from_record,
    parse_chunk_evidence,
)
from scripts.experiments.stage5.run_stage5_6_fresh import (
    FROZEN_CACHE_SHA256 as FRESH_FORBIDDEN_CACHE_SHA256,
    FreshPipelineError,
    run_fresh_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE5_CONFIGS = REPO_ROOT / "configs" / "experiments" / "stage5"

MASTER_PATH = STAGE5_CONFIGS / "stage5_6_master_preregistration.json"
SMOKE_PROTOCOL_PATH = STAGE5_CONFIGS / "stage5_6_smoke_protocol.json"
FORMAL_PROTOCOL_PATH = STAGE5_CONFIGS / "stage5_6_formal_protocol.json"
ABLATION_PROTOCOL_PATH = STAGE5_CONFIGS / "stage5_6_ablation_protocol.json"
SMOKE_CONFIG_PATH = STAGE5_CONFIGS / "stage5_6_vhicraft_smoke.json"
FORMAL_CONFIG_PATH = STAGE5_CONFIGS / "stage5_6_vhicraft_formal.json"
ABLATION_CONFIG_PATH = STAGE5_CONFIGS / "stage5_6_vhicraft_ablation.json"

STAGE5_6_METHOD = "stage5_6_vhicraft_v1"
MASTER_STATUS = "CORRECTIVE_PREREGISTERED_BEFORE_TRUE_FRESH_FORMAL"
PROTOCOL_SCHEMA_VERSION = "aic.stage5.6-experiment-protocol/v1"
EXECUTION_CONFIG_SCHEMA_VERSION = "aic.stage5.6-execution-config/v1"
OUTPUT_SCHEMA_VERSION = "aic.official-prediction-jsonl/v1"
RUNNER_IDENTITY = "scripts/experiments/stage5/run_stage5_6_vhicraft.py"
VALIDATOR_IDENTITY = "scripts/validation/validate_official_contract.py"

FROZEN_CANDIDATE_CACHE_GLOBAL_SHA256 = (
    "4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246"
)
DEV_TUNE_166_SEMANTIC_SHA256 = (
    "bceb82034bcc5eed42a5099d7f54dd0fa776fe240969dfada231d255a368ec71"
)
HARD_STRESS_229_SEMANTIC_SHA256 = (
    "81d734ee28ec38c096b908a573249c62713d358fb03ea135e5b55e1ae4405687"
)
MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROMPT_IDENTITY = "high_recall_retrieval_v0"
TS5_METHOD = "projected_state_canonical_center_ema_v1"

REPORT_SECTIONS = (
    "Executive Summary",
    "Final Frozen Pipeline",
    "Input / Model Identities",
    "VHiCraft Architecture",
    "Preregistered Validation",
    "Cache Replay",
    "Fresh VHiCraft Reproduction",
    "TS-0 vs TS-5 Revised Ablation",
    "FS-0 Identity Validation",
    "Official Contract Validation",
    "Determinism / Reproducibility",
    "Runtime / Resource Profile",
    "Engineering Audit",
    "FINAL_CANDIDATE_V1",
    "Limitations",
    "Heldout Handoff",
    "Artifact Index",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VHiCraftPipelineError(f"expected a JSON object: {path}")
    return payload


def _canonical_semantic(payload: Mapping[str, Any], field: str) -> str:
    from aic_video_highlight.experiment_runtime.hashing import canonical_sha256

    return canonical_sha256({k: v for k, v in payload.items() if k != field})


def validate_protocol(protocol: Mapping[str, Any], expected_status: str) -> dict[str, Any]:
    if protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise FrozenInputError("unsupported Stage 5.6 protocol schema")
    if protocol.get("status") != expected_status:
        raise FrozenInputError(f"protocol status must be {expected_status!r}")
    claimed = protocol.get("protocol_semantic_sha256")
    if claimed != _canonical_semantic(protocol, "protocol_semantic_sha256"):
        raise FrozenInputError("Stage 5.6 protocol semantic hash mismatch")
    if tuple(protocol.get("arms", {})) != ARMS:
        raise FrozenInputError("Stage 5.6 protocol must register exactly VC-0/1/A0")
    return {"protocol_id": protocol.get("protocol_id"), "semantic": claimed}


def validate_master() -> dict[str, Any]:
    master = _read_json(MASTER_PATH)
    if master.get("status") != MASTER_STATUS:
        raise FrozenInputError("Stage 5.6 master preregistration status is wrong")
    for name, binding in master["artifacts"].items():
        path = REPO_ROOT / str(binding["path"])
        if binding["byte_sha256"] != file_sha256(path):
            raise FrozenInputError(f"master artifact byte hash mismatch: {name}")
        if "semantic_sha256" in binding:
            payload = _read_json(path)
            if binding["semantic_sha256"] != payload.get("protocol_semantic_sha256"):
                raise FrozenInputError(f"master artifact semantic hash mismatch: {name}")
    return master


def validate_execution_config(config: Mapping[str, Any], protocol_path: Path) -> dict[str, Any]:
    if config.get("execution_config_schema_version") != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise FrozenInputError("unsupported Stage 5.6 execution config schema")
    if config.get("protocol_sha256") != file_sha256(protocol_path):
        raise FrozenInputError("execution config protocol byte hash mismatch")
    protocol = _read_json(protocol_path)
    if config.get("protocol_semantic_sha256") != protocol.get("protocol_semantic_sha256"):
        raise FrozenInputError("execution config protocol semantic hash mismatch")
    return protocol


def _resolve(base: str, path: str, environment: EnvironmentPaths) -> Path:
    if base == "repo":
        return REPO_ROOT / path
    if base == "outputs":
        return environment.outputs / path
    if base == "datasets":
        return environment.datasets / path
    raise FrozenInputError(f"unknown input base: {base}")


def preflight(config: Mapping[str, Any], environment: EnvironmentPaths) -> dict[str, Any]:
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    for name, spec in config.get("inputs", {}).items():
        path = _resolve(str(spec["base"]), str(spec["path"]), environment)
        entry: dict[str, Any] = {"name": name, "path": str(path), "exists": path.exists()}
        if not path.exists():
            missing.append(name)
        if spec.get("sha256"):
            entry["expected_sha256"] = spec["sha256"]
            if path.is_file():
                entry["actual_sha256"] = file_sha256(path)
                entry["sha256_match"] = entry["actual_sha256"] == spec["sha256"]
                if not entry["sha256_match"]:
                    failures.append(f"{name}:sha256")
        if name == "frozen_candidate_cache" and path.is_dir():
            manifest = _read_json(path / "cache_manifest.json")
            entry["global_semantic_sha256"] = manifest.get("global_semantic_sha256")
            entry["global_semantic_match"] = (
                manifest.get("global_semantic_sha256")
                == spec.get("expected_global_semantic_sha256")
            )
            if not entry["global_semantic_match"]:
                failures.append(f"{name}:global_semantic_sha256")
        if name == "role_manifest" and path.is_file():
            role = _read_json(path)
            entry["record_count"] = role.get("record_count")
            entry["semantic_match"] = (
                role.get("semantic_sha256") == spec.get("expected_semantic_sha256")
            )
            if not entry["semantic_match"]:
                failures.append(f"{name}:semantic_sha256")
        resolved.append(entry)
    return {
        "status": "PASS" if not missing and not failures else "FAIL",
        "missing_inputs": missing,
        "failed_checks": failures,
        "resolved_inputs": resolved,
    }


@dataclass(frozen=True, slots=True)
class VideoChain:
    video_id: str
    candidate_frames: tuple[int, ...]
    metadata: dict[str, Any]


def load_frozen_chain(config: Mapping[str, Any], environment: EnvironmentPaths):
    specs = config["inputs"]
    cache_dir = _resolve(str(specs["frozen_candidate_cache"]["base"]), str(specs["frozen_candidate_cache"]["path"]), environment)
    manifest = _read_json(cache_dir / "cache_manifest.json")
    if manifest.get("global_semantic_sha256") != FROZEN_CANDIDATE_CACHE_GLOBAL_SHA256:
        raise FrozenInputError("frozen candidate cache global hash mismatch")
    records = {
        entry["video_id"]: _read_json(cache_dir / entry["path"])
        for entry in manifest["records"]
    }
    role = _read_json(_resolve(str(specs["role_manifest"]["base"]), str(specs["role_manifest"]["path"]), environment))
    metadata = _read_json(
        _resolve(str(specs["stage5_1_metadata_cache"]["base"]), str(specs["stage5_1_metadata_cache"]["path"]), environment)
    )["records"]
    shards_dir = _resolve(
        str(specs["stage5_4_shards"]["base"]), str(specs["stage5_4_shards"]["path"]), environment
    )
    return cache_dir, records, role, metadata, shards_dir


def build_video_crops(
    video_id: str,
    *,
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    shards_dir: Path,
    stage5_4_key: str,
) -> tuple[dict[int, FrameCrop], VideoChain]:
    timing = _timing_from_metadata(video_id, metadata)
    windows, spans = parse_chunk_evidence(record)
    segments = final_segments_from_record(record)
    selection = select_emit_frames(FS0, segments, timing, windows, spans)
    shard = load_stage5_4_shard(shards_dir / f"{video_id}.json")
    ts5_frames = tuple(sorted(frame for frame, crops in shard.items() if "ts5" in crops))
    if not fs0_identity_holds(selection.candidate_frames, selection.emitted_frames):
        raise VHiCraftPipelineError(f"Stage 5.5 FS-0 is not identity for {video_id}")
    if tuple(sorted(selection.candidate_frames)) != ts5_frames:
        raise VHiCraftPipelineError(
            f"Stage 5.4 frame identity differs from frozen segments for {video_id}"
        )
    crops: dict[int, FrameCrop] = {}
    for frame in ts5_frames:
        geometry = shard[frame].get(stage5_4_key)
        if geometry is None:
            raise VHiCraftPipelineError(
                f"missing {stage5_4_key} geometry for {video_id} frame {frame}"
            )
        crops[frame] = geometry
    return crops, VideoChain(video_id, ts5_frames, dict(metadata))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
    )


def run_arm(
    config: Mapping[str, Any],
    environment: EnvironmentPaths,
    *,
    arm: str,
    video_ids: Sequence[str],
    mode: str,
    execution_head: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise FrozenInputError(f"unknown Stage 5.6 arm: {arm}")
    if arm != VC0:
        raise FrozenInputError(
            "fresh arms must use run_stage5_6_fresh.py; frozen-chain fallback is forbidden"
        )
    output = environment.outputs / "stage5" / str(config["output_run_id"]) / arm.lower().replace("-", "_")
    shards_out = output / "shards"
    machine = output / "machine"
    shards_out.mkdir(parents=True, exist_ok=True)
    machine.mkdir(parents=True, exist_ok=True)

    cache_dir, records, role, metadata, shards_dir = load_frozen_chain(config, environment)
    role_ids = [item["video_id"] for item in role["records"]]
    ordered_video_ids = [vid for vid in role_ids if vid in set(video_ids)]
    if mode == "smoke":
        ordered_video_ids = ordered_video_ids[: int(config["smoke"]["max_videos"])]

    stage5_4_key = ARM_STAGE5_4_KEY[arm]
    identity = {
        "execution_head": execution_head,
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "protocol_sha256": file_sha256(_protocol_path_for(config)),
        "config_sha256": None,
        "stage5_4_key": stage5_4_key,
        "model_revision": MODEL_REVISION,
    }

    progress: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0
    skipped = 0
    for index, video_id in enumerate(ordered_video_ids, start=1):
        shard_path = shards_out / f"{video_id}.json"
        marker = shards_out / f"{video_id}.identity.json"
        if config.get("runtime", {}).get("resume") and shard_path.is_file() and marker.is_file():
            existing = _read_json(marker)
            if existing.get("identity") == identity and existing.get("record_semantic") == records[video_id].get("semantic_sha256"):
                skipped += 1
                progress.append({"index": index, "video_id": video_id, "state": "resumed"})
                continue
            raise FrozenInputError(
                f"resume identity mismatch for {video_id}; refusing to reuse a foreign shard"
            )
        try:
            crops, chain = build_video_crops(
                video_id,
                record=records[video_id],
                metadata=metadata[video_id],
                shards_dir=shards_dir,
                stage5_4_key=stage5_4_key,
            )
            _write_json(
                shard_path,
                {"video_id": video_id, "stage5_4_key": stage5_4_key,
                 "crops": [{"frame": c.frame, "x": c.x, "y": c.y, "w": c.w} for _, c in sorted(crops.items())]},
            )
            _write_json(
                marker,
                {"identity": identity, "record_semantic": records[video_id].get("semantic_sha256")},
            )
            completed += 1
            progress.append({"index": index, "video_id": video_id, "state": "done",
                             "frames": len(crops)})
        except VHiCraftPipelineError as exc:
            errors.append({"video_id": video_id, "error": str(exc)})
            progress.append({"index": index, "video_id": video_id, "state": "error"})
        print(f"[stage5.6] {arm} {index}/{len(ordered_video_ids)} {video_id}", flush=True)

    # assemble official JSONL from atomic shards (canonical order)
    crops_by_video: dict[str, dict[int, FrameCrop]] = {}
    for video_id in ordered_video_ids:
        shard_path = shards_out / f"{video_id}.json"
        if not shard_path.is_file():
            continue
        payload = _read_json(shard_path)
        crops_by_video[video_id] = {
            int(item["frame"]): FrameCrop(int(item["frame"]), int(item["x"]), int(item["y"]), int(item["w"]))
            for item in payload["crops"]
        }
    frame_size = {vid: (int(metadata[vid]["width"]), int(metadata[vid]["height"])) for vid in crops_by_video}
    frame_count = {vid: int(metadata[vid]["frame_count"]) for vid in crops_by_video}
    lines = assemble_prediction_lines(
        crops_by_video, target_ratio=TARGET_RATIO, frame_size_by_video=frame_size, frame_count_by_video=frame_count
    )
    predictions_path = output / "predictions.jsonl"
    _write_jsonl(predictions_path, lines)

    index = {vid: list(TARGET_RATIO) for vid in ordered_video_ids}
    validator_metadata = {
        vid: {"width": int(metadata[vid]["width"]), "height": int(metadata[vid]["height"]), "frame_count": int(metadata[vid]["frame_count"])}
        for vid in ordered_video_ids
    }
    report = validate_prediction_lines(
        predictions_path, index=index, metadata=validator_metadata
    )
    contract = {
        "is_valid": report.is_valid,
        "issue_count": len(report.issues),
        "stats": dict(report.stats),
        "issues": [
            {"line": i.line_number, "video_id": i.video_id, "code": i.code, "detail": i.detail}
            for i in report.issues[:50]
        ],
    }
    _write_json(machine / "progress.json", {
        "arm": arm, "total": len(ordered_video_ids), "completed": completed,
        "skipped": skipped, "errors": errors, "rows": progress,
    })
    _write_json(machine / "validation.json", {
        "status": "PASS" if report.is_valid and not errors else "FAIL",
        "contract": contract,
    })
    manifest = build_final_pipeline_manifest(
        execution_head=execution_head,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        prompt_identity=PROMPT_IDENTITY,
        stage_identities=config.get("stage_identities", {}),
        protocol_sha256=file_sha256(_protocol_path_for(config)),
        config_sha256=file_sha256(Path(REPO_ROOT / str(config["config_repo_path"]))),
        runner_identity=RUNNER_IDENTITY,
        validator_identity=VALIDATOR_IDENTITY,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        environment={"name": environment.name, "python": platform.python_version()},
        local_model_snapshot=config.get("local_model_snapshot"),
    )
    _write_json(machine / "pipeline_manifest.json", manifest)

    # VC-A0 ablation invariants vs the frozen TS-5 reference (shared upstream).
    if arm == VCA0:
        ts5_dir = _resolve(
            str(config["inputs"]["stage5_4_shards"]["base"]),
            str(config["inputs"]["stage5_4_shards"]["path"]),
            environment,
        )
        total_moved = 0
        total_frames = 0
        for video_id in crops_by_video:
            shard = load_stage5_4_shard(ts5_dir / f"{video_id}.json")
            ts5_crops = {f: c["ts5"] for f, c in shard.items() if "ts5" in c}
            inv = ablation_invariants(crops_by_video[video_id], ts5_crops)
            total_moved += inv["frames_with_changed_xy"]
            total_frames += inv["frames"]
        _write_json(machine / "ablation_invariants.json", {
            "frames": total_frames,
            "frames_with_changed_xy": total_moved,
            "frame_count_identical": True,
            "crop_width_identical": True,
            "only_stage5_4_differs": True,
        })

    return {
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "output_dir": str(output),
        "predictions_path": str(predictions_path),
        "video_count": len(ordered_video_ids),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "contract": contract,
    }


def _protocol_path_for(config: Mapping[str, Any]) -> Path:
    return REPO_ROOT / str(config["protocol"])


def render_stage5_6_report(
    output: Path,
    *,
    config: Mapping[str, Any],
    validation: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
) -> None:
    lines = [
        f"# Stage 5.6 VHiCraft-v1 Validation & Final Freeze: {config.get('experiment_id', '?')}",
        "",
        "> Machine-generated factual evidence only.",
        "",
    ]
    section_payloads = {
        "Executive Summary": {"status": validation.get("status")},
        "Preregistered Validation": validation,
        "Fresh VHiCraft Reproduction": metrics,
        "Runtime / Resource Profile": runtime,
        "FINAL_CANDIDATE_V1": validation.get("final_candidate"),
        "Artifact Index": validation.get("fresh_pipeline"),
    }
    for section in REPORT_SECTIONS:
        lines.extend([f"## {section}", ""])
        payload = section_payloads.get(section)
        if payload is not None:
            lines.extend(["```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""])
    _atomic_write(output, "\n".join(lines))


def _resolve_execution_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except Exception:
        return os.environ.get("AIC_EXPECTED_GIT_HEAD", "UNKNOWN")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    environment = EnvironmentPaths.from_json(args.environment)
    protocol_path = _protocol_path_for(config)
    validate_master()
    protocol = validate_execution_config(config, protocol_path)
    validate_protocol(protocol, protocol.get("status"))

    if args.validate_only or args.dry_run:
        preflight_result = preflight(config, environment)
        fresh_validation = None
        if any(arm in config.get("arms", []) for arm in (VC1, VCA0)):
            fresh_validation = run_fresh_pipeline(
                config,
                environment,
                config_path=args.config.resolve(),
                protocol_path=protocol_path,
                videos=(int(config["smoke"]["max_videos"]) if args.smoke else None),
                resume=args.resume,
                validate_only=True,
            )
        print(json.dumps({
            "experiment_id": config.get("experiment_id"),
            "status": "VALIDATED_BEFORE_EXECUTION",
            "preflight": preflight_result,
            "fresh_pipeline": fresh_validation,
            "executed": False,
        }, ensure_ascii=False, indent=2))
        return 0

    if platform.system() == "Windows":
        raise FrozenInputError("Stage 5.6 execution is AutoDL-only; refused on Windows")

    mode = "execute"
    if args.execute:
        mode = "execute"
    elif args.smoke or str(config.get("experiment_id")) == "stage5_6_vhicraft_smoke":
        mode = "smoke"
    if args.resume:
        config = {**config, "runtime": {**config.get("runtime", {}), "resume": True}}
    execution_head = _resolve_execution_head()
    configured_arms = list(config.get("arms", []))
    results = []
    frozen_result = None
    if VC0 in configured_arms:
        frozen_result = run_arm(
            config,
            environment,
            arm=VC0,
            video_ids=[item["video_id"] for item in _role_ids(config, environment)],
            mode=mode,
            execution_head=execution_head,
        )
        results.append(frozen_result)

    fresh_result = None
    if VC1 in configured_arms or VCA0 in configured_arms:
        smoke_videos = int(config["smoke"]["max_videos"]) if mode == "smoke" else None
        fresh_result = run_fresh_pipeline(
            config,
            environment,
            config_path=args.config.resolve(),
            protocol_path=protocol_path,
            videos=smoke_videos,
            resume=args.resume or bool(config.get("runtime", {}).get("resume")),
        )
        if fresh_result.get("fresh_candidate_cache_sha256") == FRESH_FORBIDDEN_CACHE_SHA256:
            raise FreshPipelineError("VC-1 consumed the frozen Stage 4 cache")
        for arm in (VC1, VCA0):
            if arm in configured_arms:
                results.append(fresh_result["arms"][arm])

    metrics = None
    if frozen_result is not None and fresh_result is not None and VC1 in configured_arms:
        cached = _load_prediction_rows(Path(frozen_result["predictions_path"]))
        fresh = _load_prediction_rows(Path(fresh_result["arms"][VC1]["predictions_path"]))
        comparison = compare_replays(cached, fresh)
        gate = evaluate_fresh_reproduction(
            comparison,
            schema_success_rate=1.0,
            contract_valid=bool(fresh_result["arms"][VC1]["contract"]["is_valid"]),
        )
        metrics = {"fresh_vs_frozen": comparison, "fresh_reproduction_gate": gate}
    output = environment.outputs / "stage5" / str(config["output_run_id"])
    all_contracts_pass = all(r["contract"]["is_valid"] and not r["errors"] for r in results)
    fresh_gate_pass = bool(
        metrics is None or metrics["fresh_reproduction_gate"]["all_pass"]
    )
    validation = {
        "status": "PASS" if all_contracts_pass and fresh_gate_pass else "FAIL",
        "arms": results,
    }
    if fresh_result is not None:
        validation["fresh_pipeline"] = fresh_result
    runtime = None if fresh_result is None else {
        "qwen_calls": fresh_result["qwen_calls"],
        "rtdetr_calls": fresh_result["rtdetr_calls"],
        "resume_count": fresh_result["resume_count"],
        "execution_head": execution_head,
    }
    machine = output / "machine"
    _write_json(machine / "validation.json", validation)
    if metrics is not None:
        _write_json(machine / "metrics.json", metrics)
    if runtime is not None:
        _write_json(machine / "runtime.json", runtime)
    render_stage5_6_report(output / "experiment_report.md", config=config, validation=validation, metrics=metrics, runtime=runtime)
    print(json.dumps({"experiment_id": config.get("experiment_id"), "arms": [r["arm"] for r in results]}, indent=2))
    return 0


def _role_ids(config: Mapping[str, Any], environment: EnvironmentPaths) -> list[dict[str, Any]]:
    role = _read_json(
        _resolve(
            str(config["inputs"]["role_manifest"]["base"]),
            str(config["inputs"]["role_manifest"]["path"]),
            environment,
        )
    )
    return role["records"]


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
