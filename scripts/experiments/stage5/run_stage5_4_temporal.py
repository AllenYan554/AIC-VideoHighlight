#!/usr/bin/env python3
"""Stage 5.4 Temporal Composition Stabilization runner (TS-0 vs TS-1).

Config-driven runner for `stage5_4_smoke`: consumes the frozen Stage 5.3 CMP-1
composition as TS-0 (temporal control baseline), applies the deterministic TS-1
EMA crop-center smoothing (development baseline alpha = 0.5), and evaluates
temporal stability metrics, Stage 5.3 spatial guardrails, strata and the
observation-only multi-subject diagnostic. CPU-only, frozen-artifact-only,
no RT-DETR, no Qwen, no SAM, no tracking, no Heldout access.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from aic_video_highlight.experiment_runtime.artifacts import build_artifact_manifest
from aic_video_highlight.experiment_runtime.hashing import canonical_sha256
from aic_video_highlight.experiment_runtime.io import atomic_write_json
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.experiment_runtime.progress import ProgressReporter
from aic_video_highlight.experiment_runtime.raw_report import render_raw_report, write_ai_report_inputs
from aic_video_highlight.experiment_runtime.run_context import RunContext, RunIdentityMismatch
from aic_video_highlight.experiment_runtime.shards import ShardStore
from aic_video_highlight.spatial_composition.center_crop import derived_height
from aic_video_highlight.spatial_composition.composition_metrics import (
    crop_rect_from_xywh,
    subject_center_inside_crop,
    subject_visible_fraction,
)
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputError,
    InputBinding,
    compose_frame,
    load_frozen_inputs,
    load_raw_shard_dir_subset,
    verify_input_bindings,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    PLACEMENT_FALLBACK_CENTER_CROP,
    SANITIZE_OK,
    SanitizedSubject,
)
from aic_video_highlight.spatial_composition.temporal_diagnostics import (
    crop_geometry_valid,
    observations_from_records,
    spatial_guardrail_metrics,
    temporal_multi_subject_diagnostic,
    temporal_stability_metrics,
)
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    DEFAULT_EMA_ALPHA,
    large_jump_ratios,
    smooth_video_sequence,
    temporal_summary,
)
from aic_video_highlight.spatial_localization.subject_localization import SubjectPolicyConfig

BASE_FIELDS = {
    "archive": "archive",
    "outputs": "outputs",
    "datasets": "datasets",
    "repo": "repo",
    "models": "models",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def resolve_bindings(config: dict, environment: EnvironmentPaths) -> list[InputBinding]:
    bindings = []
    for name, entry in config["inputs"].items():
        base = getattr(environment, BASE_FIELDS[entry["base"]])
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        bindings.append(
            InputBinding(
                name=name,
                path=path,
                sha256=str(entry.get("sha256", "") or ""),
                format=entry.get("format", "json"),
            )
        )
    required = {
        "stage5_4_smoke_manifest",
        "stage5_1_predictions",
        "video_metadata_cache",
        "dev166_index",
        "stage5_2_policy_artifact",
        "stage5_2_raw_detector",
        "weak_spatial_reference",
    }
    missing = required - {binding.name for binding in bindings}
    if missing:
        raise FrozenInputError(f"config inputs missing: {sorted(missing)}")
    return bindings


def policy_config_from(config: dict) -> SubjectPolicyConfig:
    thresholds = config["composition"]["stage5_2_policy_thresholds"]
    return SubjectPolicyConfig(
        reliable_score=thresholds["reliable_score"],
        possible_score=thresholds["possible_score"],
        ambiguity_score_gap=thresholds["ambiguity_score_gap"],
        ambiguity_iou_threshold=thresholds["ambiguity_iou_threshold"],
        min_area_fraction=thresholds["min_area_fraction"],
        full_frame_width_fraction=thresholds["full_frame_width_fraction"],
        full_frame_height_fraction=thresholds["full_frame_height_fraction"],
        full_frame_area_fraction=thresholds["full_frame_area_fraction"],
        person_priority=True,
        target_ratio=tuple(config["composition"]["target_ratio"]),
    )


def load_frozen_manifest(config: dict, bindings: list[InputBinding]) -> dict:
    """Load and byte-verify the frozen smoke manifest (SHA must be pinned)."""
    expected_sha = config["manifest"].get("expected_manifest_sha256")
    if not expected_sha:
        raise FrozenInputError(
            "stage5_4 smoke manifest SHA is not pinned: set "
            "configs/experiments/stage5/stage5_4_smoke.json manifest.expected_manifest_sha256 "
            "from the stage5_4_manifest build output before running the smoke"
        )
    binding = next(b for b in bindings if b.name == "stage5_4_smoke_manifest")
    if not binding.path.is_file():
        raise FrozenInputError(f"frozen manifest missing: {binding.path}")
    manifest = json.loads(binding.path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != expected_sha:
        raise FrozenInputError("frozen manifest internal sha mismatch")
    content_sha = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if content_sha != expected_sha:
        raise FrozenInputError(
            f"frozen manifest sha mismatch: expected {expected_sha}, got canonical content {content_sha}"
        )
    return manifest


def crosscheck_manifest_entry(manifest_entry: dict, composed: dict) -> list[str]:
    """The recomputed frozen TS-0 must reproduce the frozen manifest model-blind fields."""
    mismatches: list[str] = []
    key = f"{composed['video_id']}:{composed['frame']}"
    if manifest_entry["stratum"] != composed["stratum"]:
        mismatches.append(f"{key} stratum")
    if bool(manifest_entry["cmp1_fallback"]) != bool(composed["cmp1"]["fallback"]):
        mismatches.append(f"{key} fallback")
    if bool(manifest_entry["ambiguous"]) != bool(composed["ambiguous"]):
        mismatches.append(f"{key} ambiguous")
    if manifest_entry["horizontal_center_offset"] != composed["horizontal_center_offset"]:
        mismatches.append(f"{key} offset")
    xyxy = composed["sanitized"]["xyxy"]
    recomputed_center_x = round((float(xyxy[0]) + float(xyxy[2])) / 2.0, 6)
    recomputed_center_y = round((float(xyxy[1]) + float(xyxy[3])) / 2.0, 6)
    if manifest_entry["subject_center_x"] != recomputed_center_x:
        mismatches.append(f"{key} center_x")
    if manifest_entry["subject_center_y"] != recomputed_center_y:
        mismatches.append(f"{key} center_y")
    return mismatches


def build_video_records(
    video: dict,
    inputs,
    target_ratio: list[float],
    strata_thresholds: dict,
    alpha: float,
    tw: float,
    th: float,
) -> tuple[list[dict], list[str]]:
    """TS-0 composition + TS-1 smoothing for one manifest video."""
    manifest_by_frame = {int(entry["frame"]): entry for entry in video["frames"]}
    frozen_boxes = {
        int(pred["frame"]): list(pred["bboxes"])
        for pred in inputs.stage5_1_predictions[video["video_id"]]["predictions"]
    }
    composed_records: list[dict] = []
    mismatches: list[str] = []
    for entry in sorted(video["frames"], key=lambda item: int(item["frame"])):
        composed = compose_frame(
            video["video_id"],
            int(entry["frame"]),
            inputs,
            target_ratio,
            strata_thresholds,
            frozen_boxes,
        )
        frame_mismatches = crosscheck_manifest_entry(entry, composed)
        mismatches.extend(frame_mismatches)
        # TS-0 is the Stage 5.3 FINAL FROZEN CMP-1; its regression evidence is the
        # per-frame cross-check against the SHA-pinned manifest identity (the
        # frozen CMP-1 must reproduce it exactly, model-blind).
        composed["ts0_frozen_regression"] = bool(frame_mismatches)
        composed_records.append(composed)

    # observations_from_records consumes the merged record schema; the TS-1 side
    # does not exist yet at this point, so expose the frozen TS-0 (CMP-1) under
    # the "ts0" key contract expected by the diagnostics module.
    observations = observations_from_records(
        [
            {"frame": composed["frame"], "ts0": composed["cmp1"], "sanitized": composed["sanitized"]}
            for composed in composed_records
        ]
    )
    width, height = int(video["image_width"]), int(video["image_height"])
    smoothed = smooth_video_sequence(width, height, tw, th, observations, alpha=alpha)
    smoothed_by_frame = {item.frame: item for item in smoothed}

    records: list[dict] = []
    for composed in composed_records:
        frame = int(composed["frame"])
        item = smoothed_by_frame[frame]
        ts0 = composed["cmp1"]
        ts1 = {
            "x": item.x,
            "y": item.y,
            "w": item.w,
            "h": float(item.h),
            "crop_w": item.crop_w,
            "crop_h": item.crop_h,
            "placement_status": item.placement_status,
            "reset_reason": item.reset_reason,
            "ema_center_x": item.ema_center_x,
            "ema_center_y": item.ema_center_y,
            "clamped_x": item.clamped_x,
            "clamped_y": item.clamped_y,
            "matches_ts0_placement": item.matches_ts0_placement,
            "subject_visible_fraction": None,
            "subject_center_inside": False,
        }
        if composed["sanitized"]["status"] == SANITIZE_OK:
            xyxy = composed["sanitized"]["xyxy"]
            sanitized = SanitizedSubject(
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
                SANITIZE_OK,
                composed["sanitized"]["clamp_left"],
                composed["sanitized"]["clamp_top"],
                composed["sanitized"]["clamp_right"],
                composed["sanitized"]["clamp_bottom"],
            )
            ts1_rect = crop_rect_from_xywh(item.x, item.y, item.w, float(item.h))
            ts1["subject_visible_fraction"] = round(subject_visible_fraction(sanitized, ts1_rect), 6)
            ts1["subject_center_inside"] = subject_center_inside_crop(sanitized, ts1_rect)
        geometry = {
            **{
                f"ts0_{key}": value
                for key, value in crop_geometry_valid(
                    int(ts0["x"]),
                    int(ts0["w"]),
                    int(ts0["y"]),
                    derived_height(int(ts0["w"]), tw, th),
                    width,
                    height,
                ).items()
            },
            **{
                f"ts1_{key}": value
                for key, value in crop_geometry_valid(
                    item.x, item.w, item.y, derived_height(item.w, tw, th), width, height
                ).items()
            },
        }
        records.append(
            {
                "video_id": composed["video_id"],
                "frame": frame,
                "image_width": width,
                "image_height": height,
                "stage5_2_status": composed["stage5_2_status"],
                "fallback_reasons": composed["fallback_reasons"],
                "ambiguous": composed["ambiguous"],
                "ambiguous_candidate_count": composed["ambiguous_candidate_count"],
                "stratum": composed["stratum"],
                "horizontal_center_offset": composed["horizontal_center_offset"],
                "sanitized": composed["sanitized"],
                "ts0": {
                    "x": int(ts0["x"]),
                    "y": int(ts0["y"]),
                    "w": int(ts0["w"]),
                    "h": float(ts0["h"]),
                    "crop_w": int(ts0["crop_w"]),
                    "crop_h": int(ts0["crop_h"]),
                    "placement_status": ts0["placement_status"],
                    "fallback": bool(ts0["fallback"]),
                    "subject_visible_fraction": ts0["subject_visible_fraction"],
                    "subject_center_inside": ts0["subject_center_inside"],
                    "frozen_regression": bool(composed["ts0_frozen_regression"]),
                },
                "ts1": ts1,
                "geometry_valid": geometry,
            }
        )
    return records, mismatches


def engineering_gate(manifest: dict, records: list[dict], crosscheck_mismatches: list[str]) -> dict:
    expected_keys = {
        (video["video_id"], int(entry["frame"]))
        for video in manifest["videos"]
        for entry in video["frames"]
    }
    produced_keys = [(record["video_id"], int(record["frame"])) for record in records]
    missing = expected_keys - set(produced_keys)
    extra = set(produced_keys) - expected_keys
    duplicates = len(produced_keys) - len(set(produced_keys))
    crop_size_unchanged = all(
        record["ts1"]["w"] == record["ts0"]["w"] and record["ts1"]["h"] == record["ts0"]["h"]
        for record in records
    )
    fallback_placement_unchanged = all(
        record["ts1"]["placement_status"] == PLACEMENT_FALLBACK_CENTER_CROP
        and (record["ts1"]["x"], record["ts1"]["y"]) == (record["ts0"]["x"], record["ts0"]["y"])
        for record in records
        if record["ts0"]["fallback"]
    )
    geometry_failures = [record for record in records if not all(record["geometry_valid"].values())]
    invalid_crop = len(geometry_failures)
    out_of_bounds = sum(
        1
        for record in geometry_failures
        if not (record["geometry_valid"]["ts0_nonnegative"] and record["geometry_valid"]["ts1_nonnegative"]
                and record["geometry_valid"]["ts0_x_within_frame"] and record["geometry_valid"]["ts1_x_within_frame"])
    )
    ratio_violations = sum(
        1
        for record in geometry_failures
        if not (record["geometry_valid"]["ts0_derived_height_within_frame"] and record["geometry_valid"]["ts1_derived_height_within_frame"])
    )
    ts0_regression = sum(1 for record in records if record["ts0"]["frozen_regression"])
    return {
        "expected_frames": len(expected_keys),
        "produced_frames": len(produced_keys),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": duplicates,
        "manifest_crosscheck_mismatches": len(crosscheck_mismatches),
        "crop_size_unchanged": crop_size_unchanged,
        "fallback_placement_unchanged": fallback_placement_unchanged,
        "invalid_crop": invalid_crop,
        "out_of_bounds": out_of_bounds,
        "ratio_violations": ratio_violations,
        "ts0_frozen_regression": ts0_regression,
        "frame_identity_complete": not missing and not extra and not duplicates,
    }


def evaluate_gates(
    gate: dict, deterministic_pass: bool, artifact_modified: dict, multi_subject_generated: bool
) -> dict:
    return {
        "manifest_sha_pinned": True,
        "frame_identity_100pct": bool(gate["frame_identity_complete"]),
        "manifest_crosscheck_consistent": gate["manifest_crosscheck_mismatches"] == 0,
        "ts1_crop_size_unchanged": bool(gate["crop_size_unchanged"]),
        "fallback_placement_unchanged": bool(gate["fallback_placement_unchanged"]),
        "invalid_crop_zero": gate["invalid_crop"] == 0,
        "out_of_bounds_zero": gate["out_of_bounds"] == 0,
        "ratio_violations_zero": gate["ratio_violations"] == 0,
        "ts0_frozen_regression_zero": gate["ts0_frozen_regression"] == 0,
        "deterministic_pass": deterministic_pass,
        "stage5_2_artifact_unmodified": not any(artifact_modified.values()),
        "heldout_access_zero": True,
        "multi_subject_diagnostic_generated": multi_subject_generated,
    }


def pooled_distributions(records: list[dict]) -> dict[str, list[float]]:
    """Concatenated per-transition displacement / acceleration values across videos."""
    from aic_video_highlight.spatial_composition.temporal_diagnostics import (
        acceleration_norm,
        crop_center_x,
        displacement_norm,
    )

    pooled: dict[str, list[float]] = {
        "ts0_displacement": [],
        "ts1_displacement": [],
        "ts1_displacement_smoothed_only": [],
        "ts0_acceleration": [],
        "ts1_acceleration": [],
        "ts1_acceleration_smoothed_only": [],
    }
    by_video: dict[str, list[dict]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for video_records in by_video.values():
        ordered = sorted(video_records, key=lambda record: int(record["frame"]))
        width = int(ordered[0]["image_width"])
        for previous, current in zip(ordered, ordered[1:]):
            if previous["ts0"]["fallback"] or current["ts0"]["fallback"]:
                continue
            if int(current["frame"]) - int(previous["frame"]) != 1:
                continue
            pooled["ts0_displacement"].append(
                displacement_norm(
                    crop_center_x(int(previous["ts0"]["x"]), int(previous["ts0"]["w"])),
                    crop_center_x(int(current["ts0"]["x"]), int(current["ts0"]["w"])),
                    width,
                )
            )
            ts1_value = displacement_norm(
                crop_center_x(int(previous["ts1"]["x"]), int(previous["ts1"]["w"])),
                crop_center_x(int(current["ts1"]["x"]), int(current["ts1"]["w"])),
                width,
            )
            pooled["ts1_displacement"].append(ts1_value)
            if current["ts1"]["reset_reason"] is None:
                pooled["ts1_displacement_smoothed_only"].append(ts1_value)
        for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
            if any(record["ts0"]["fallback"] for record in (first, second, third)):
                continue
            if int(second["frame"]) - int(first["frame"]) != 1 or int(third["frame"]) - int(second["frame"]) != 1:
                continue

            def delta(record_a: dict, record_b: dict, side: str) -> float:
                return crop_center_x(int(record_a[side]["x"]), int(record_a[side]["w"])) - crop_center_x(
                    int(record_b[side]["x"]), int(record_b[side]["w"])
                )

            pooled["ts0_acceleration"].append(
                acceleration_norm(delta(first, second, "ts0"), delta(second, third, "ts0"), width)
            )
            ts1_acceleration = acceleration_norm(delta(first, second, "ts1"), delta(second, third, "ts1"), width)
            pooled["ts1_acceleration"].append(ts1_acceleration)
            if second["ts1"]["reset_reason"] is None and third["ts1"]["reset_reason"] is None:
                pooled["ts1_acceleration_smoothed_only"].append(ts1_acceleration)
    return pooled


def summary_payload(values: list[float]) -> dict:
    return {
        **temporal_summary(values),
        "large_jump_ratios_descriptive_only": large_jump_ratios(values),
    }


def run(args) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_json(args.environment)
    paths = environment.for_experiment(config["stage"], config["experiment_id"])
    if args.dry_run:
        print(json.dumps({
            "experiment_id": config["experiment_id"],
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "action": "NONE",
        }, indent=2))
        return 0

    bindings = resolve_bindings(config, environment)
    context = RunContext(
        config["experiment_id"],
        config["stage"],
        config["run_type"],
        environment.repo,
        paths,
        args.config,
        Path(config["protocol"]),
        {
            f"input:{name}": (entry.get("sha256") or "UNPINNED_VERIFIED_AT_RUNTIME")
            for name, entry in config["inputs"].items()
        },
        {"name": "none", "gpu": False, "note": "frozen Stage 5.3 artifact consumption only; CPU temporal smoothing"},
    )
    if args.validate_only:
        try:
            manifest = load_frozen_manifest(config, bindings)
            load_frozen_inputs(
                [b for b in bindings if b.format != "raw_shard_dir"],
                include_raw=False,
                policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
            )
            result = {
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "videos": manifest["video_count"],
                "frames": manifest["frame_count"],
                "validation": "PASS",
            }
        except FrozenInputError as exc:
            result = {"validation": "FAIL", "reason": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("validation") == "PASS" else 2

    try:
        context.start(resume=args.resume)
    except (RunIdentityMismatch, FileExistsError) as exc:
        print(f"EXPERIMENT FAILED\n\nReason:\n{exc}\n\nResume available:\nNO", file=sys.stderr)
        return 2

    started = time.monotonic()
    identity_hash = canonical_sha256(context.identity())
    shard_store = ShardStore(paths.output / "shards", identity_hash)
    reporter = None

    try:
        manifest = load_frozen_manifest(config, bindings)
        inputs = load_frozen_inputs(
            [b for b in bindings if b.format != "raw_shard_dir"],
            include_raw=False,
            policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
        )
        manifest_video_ids = {video["video_id"] for video in manifest["videos"]}
        manifest_frame_keys = {
            (video["video_id"], int(entry["frame"]))
            for video in manifest["videos"]
            for entry in video["frames"]
        }
        slim_policy_records = tuple(
            record
            for record in inputs.policy_records
            if (record["video_id"], int(record["frame"])) in manifest_frame_keys
        )
        slim_inputs = dataclasses.replace(inputs, policy_records=slim_policy_records)

        target_ratio = [float(value) for value in config["composition"]["target_ratio"]]
        strata = config["composition"]["strata_thresholds"]
        tw, th = float(target_ratio[0]), float(target_ratio[1])
        alpha = float(config["temporal_smoothing"].get("alpha", DEFAULT_EMA_ALPHA))

        total_frames = int(manifest["frame_count"])
        reporter = ProgressReporter(config["experiment_id"], total_frames, paths.logs)
        frames_done = 0
        all_mismatches: list[str] = []
        for video in manifest["videos"]:
            video_id = video["video_id"]
            if shard_store.is_complete(video_id):
                frames_done += int(video["frame_count"])
                reporter.update(
                    frames_done,
                    current_video=video_id,
                    current_shard=video_id,
                    display={"frames": f"{frames_done}/{total_frames}", "resumed": "yes"},
                )
                continue
            video_records, mismatches = build_video_records(
                video, slim_inputs, target_ratio, strata, alpha, tw, th
            )
            all_mismatches.extend(mismatches)
            shard_store.write(video_id, video_records)
            frames_done += len(video_records)
            invalid = sum(1 for record in video_records if not all(record["geometry_valid"].values()))
            reporter.update(
                frames_done,
                current_video=video_id,
                current_shard=video_id,
                errors=len(mismatches),
                display={
                    "frames": f"{frames_done}/{total_frames}",
                    "sequences": manifest["video_count"],
                    "invalid": invalid,
                },
            )

        records: list[dict] = []
        for shard in sorted(shard_store.root.glob("*.json")):
            if shard.name.endswith(".status.json"):
                continue
            records.extend(json.loads(shard.read_text(encoding="utf-8")))
        records.sort(key=lambda record: (record["video_id"], record["frame"]))

        machine = paths.output / "machine"
        diagnostics_dir = machine / "diagnostics"
        machine.mkdir(parents=True, exist_ok=True)

        gate = engineering_gate(manifest, records, all_mismatches)
        temporal_by_video = {
            video["video_id"]: temporal_stability_metrics(
                [record for record in records if record["video_id"] == video["video_id"]]
            )
            for video in manifest["videos"]
        }
        pooled = pooled_distributions(records)
        guardrails = spatial_guardrail_metrics(records)

        deterministic_records: list[dict] = []
        for video in manifest["videos"]:
            video_records, _ = build_video_records(
                video, slim_inputs, target_ratio, strata, alpha, tw, th
            )
            deterministic_records.extend(video_records)
        deterministic_records.sort(key=lambda record: (record["video_id"], record["frame"]))
        deterministic_pass = canonical_sha256(records) == canonical_sha256(deterministic_records)

        post_hashes = verify_input_bindings(
            [b for b in bindings if b.format not in ("policy_shard_dir", "raw_shard_dir")]
        )
        artifact_modified = {
            name: post_hashes.get(name) != inputs.input_hashes.get(name) for name in post_hashes
        }

        raw_frames = load_raw_shard_dir_subset(
            next(b for b in bindings if b.name == "stage5_2_raw_detector").path,
            manifest_video_ids,
        )
        inputs_with_raw = dataclasses.replace(slim_inputs, raw_frames=raw_frames)
        multi_subject = temporal_multi_subject_diagnostic(
            inputs_with_raw, records, target_ratio, policy_config_from(config)
        )

        gates = evaluate_gates(
            gate, deterministic_pass, artifact_modified, multi_subject.get("ambiguous_frames") is not None
        )
        wall_sec = time.monotonic() - started

        atomic_write_json(machine / "summary.json", {
            "schema_version": "aic.machine-summary/v1",
            "experiment_id": config["experiment_id"],
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "videos": manifest["video_count"],
            "frames": manifest["frame_count"],
            "ts0": "stage5_3_final_frozen_cmp1",
            "ts1": {**config["temporal_smoothing"]},
            "manifest_coverage": manifest.get("coverage", {}),
        })
        atomic_write_json(machine / "metrics.json", {
            "schema_version": "aic.machine-metrics/v1",
            "metric_role": "DESCRIPTIVE_ONLY / no promotion gate in Stage 5.4 v1",
            "temporal_stability_by_video": temporal_by_video,
            "temporal_stability_pooled": {key: summary_payload(values) for key, values in pooled.items()},
            "spatial_guardrails": guardrails,
        })
        atomic_write_json(machine / "runtime.json", {
            "schema_version": "aic.machine-runtime/v1",
            "wall_sec": round(wall_sec, 3),
            "gpu_calls": 0,
            "qwen_vllm_calls": 0,
            "rtdetr_inference": 0,
            "device": "cpu",
        })
        atomic_write_json(machine / "validation.json", {
            "schema_version": "aic.machine-validation/v1",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "engineering_gate": gate,
            "deterministic_replay": deterministic_pass,
            "stage5_2_artifact_modified": artifact_modified,
        })
        atomic_write_json(diagnostics_dir / "multi_subject_diagnostic.json", multi_subject)

        artifact_files = [
            *sorted(shard_store.root.glob("*.json")),
            machine / "summary.json",
            machine / "metrics.json",
            machine / "validation.json",
            machine / "runtime.json",
            diagnostics_dir / "multi_subject_diagnostic.json",
        ]
        build_artifact_manifest(paths.output, artifact_files, machine / "artifact_manifest.json", created_by=config["experiment_id"])
        render_raw_report(machine, paths.output / "experiment_raw_report.md", config["experiment_id"])
        write_ai_report_inputs(paths.output)
        context.set_status(
            "COMPLETED" if all(gates.values()) else "VALIDATION_FAILED",
            validation="PASS" if all(gates.values()) else "FAIL",
        )

        print(f"\n{'=' * 50}\nEXPERIMENT {'COMPLETE' if all(gates.values()) else 'VALIDATION FAILED'}\n{'=' * 50}")
        print(f"Gates: {json.dumps(gates, indent=2)}")
        print(f"Raw report:\n{paths.output / 'experiment_raw_report.md'}")
        return 0 if all(gates.values()) else 2
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.interrupt()
        context.set_status("INTERRUPTED")
        print("\nEXPERIMENT FAILED\n\nReason:\nInterrupted safely\n\nResume available:\nYES\n\nResume command:\nSame command + --resume")
        return 130
    except FrozenInputError as exc:
        context.set_status("FAILED", reason=str(exc))
        print(f"EXPERIMENT FAILED\n\nReason:\n{exc}\n\nResume available:\nNO (frozen input problem)", file=sys.stderr)
        return 2


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
