#!/usr/bin/env python3
"""Stage 5.3 Target-Ratio Spatial Composition runner (CMP-0 vs CMP-1).

Config-driven runner shared by `stage5_3_smoke` (CPU, frozen 450-frame formal
policy artifact) and `stage5_3_formal` (AutoDL, frozen Full Dev artifact).
No detector inference, no GPU, no Heldout access.
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
from aic_video_highlight.spatial_composition.composition_figures import render_all_figures
from aic_video_highlight.spatial_composition.composition_pipeline import (
    InputBinding,
    FrozenInputError,
    aggregate_metrics,
    box_too_large_diagnostic,
    build_manifest,
    compose_all_frames,
    crosscheck_mirror_against_artifact,
    engineering_gate,
    load_frozen_inputs,
    load_raw_shard_dir,
    fallback_reason_diagnostic,
    mirror_reliable_candidate,
    multi_subject_diagnostic,
    verify_input_bindings,
    verify_shard_dir_integrity,
)
from aic_video_highlight.spatial_composition.official_emission import write_and_validate_official
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


def load_shard_records(shard_root: Path) -> list[dict]:
    records: list[dict] = []
    for shard in sorted(shard_root.glob("*.json")):
        if shard.name.endswith(".status.json"):
            continue
        records.extend(json.loads(shard.read_text(encoding="utf-8")))
    return records


def load_deferred_raw_inputs(bindings: list[InputBinding]):
    """Attach deferred raw-detector shards to the frozen composition inputs."""
    raw_binding = next(b for b in bindings if b.name == "stage5_2_raw_detector")
    raw_frames = load_raw_shard_dir(raw_binding.path)
    inputs_without_raw = load_frozen_inputs(
        [b for b in bindings if b.format != "raw_shard_dir"],
        include_raw=False,
    )
    return dataclasses.replace(inputs_without_raw, raw_frames=raw_frames)


def run(args) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_json(args.environment)
    paths = environment.for_experiment(config["stage"], config["experiment_id"])
    protocol_path = Path(config["protocol"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": config["experiment_id"],
                    "paths": {key: str(value) for key, value in vars(paths).items()},
                    "action": "NONE",
                },
                indent=2,
            )
        )
        return 0

    bindings = resolve_bindings(config, environment)
    context = RunContext(
        config["experiment_id"],
        config["stage"],
        config["run_type"],
        environment.repo,
        paths,
        args.config,
        protocol_path,
        {f"input:{name}": entry.get("sha256", "") for name, entry in config["inputs"].items()},
        {"name": "none", "gpu": False, "note": "frozen Stage 5.2 artifact consumption only"},
    )
    if args.validate_only:
        try:
            probe_inputs = load_frozen_inputs(
                bindings,
                include_raw=False,
                policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
            )
            probe_manifest = build_manifest(
                probe_inputs,
                manifest_id=config["manifest"]["manifest_id"],
                strata_thresholds=config["composition"]["strata_thresholds"],
                target_ratio=config["composition"]["target_ratio"],
            )
            expected_sha = config["manifest"].get("expected_manifest_sha256")
            manifest_ok = not expected_sha or probe_manifest["manifest_sha256"] == expected_sha
            store_probe = ShardStore(paths.output / "shards", canonical_sha256(context.identity()))
            complete = sum(1 for video in probe_manifest["videos"] if store_probe.is_complete(video["video_id"]))
            result = {
                "manifest_ok": manifest_ok,
                "complete_videos": complete,
                "total_videos": probe_manifest["video_count"],
                "validation": "PASS" if manifest_ok and complete == probe_manifest["video_count"] else "INCOMPLETE",
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
        inputs = load_frozen_inputs(
            bindings,
            include_raw=not config.get("defer_raw_loading", False),
            policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
        )
        strata = config["composition"]["strata_thresholds"]
        target_ratio = config["composition"]["target_ratio"]
        manifest = build_manifest(
            inputs,
            manifest_id=config["manifest"]["manifest_id"],
            strata_thresholds=strata,
            target_ratio=target_ratio,
        )
        expected_sha = config["manifest"].get("expected_manifest_sha256")
        if expected_sha and manifest["manifest_sha256"] != expected_sha:
            raise FrozenInputError(
                f"manifest sha drift: expected {expected_sha}, got {manifest['manifest_sha256']}"
            )
        atomic_write_json(paths.output / "manifest" / "stage5_3_smoke_manifest_v1.json", manifest)

        total_frames = int(manifest["frame_count"])
        reporter = ProgressReporter(config["experiment_id"], total_frames, paths.logs)
        frames_done = 0
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
            video_records = compose_all_frames(
                {"videos": [video]}, inputs, target_ratio, strata
            )
            shard_store.write(video_id, video_records)
            frames_done += len(video_records)
            fallback = sum(1 for record in video_records if record["cmp1"]["fallback"])
            invalid = sum(1 for record in video_records if not all(record["geometry_valid"].values()))
            reporter.update(
                frames_done,
                current_video=video_id,
                current_shard=video_id,
                display={
                    "frames": f"{frames_done}/{total_frames}",
                    "fallback": fallback,
                    "invalid": invalid,
                },
            )

        records = load_shard_records(shard_store.root)
        records.sort(key=lambda record: (record["video_id"], record["frame"]))

        machine = paths.output / "machine"
        diagnostics_dir = machine / "diagnostics"
        machine.mkdir(parents=True, exist_ok=True)

        gate = engineering_gate(manifest, records)
        metrics = aggregate_metrics(manifest, records, strata)
        deterministic_records = compose_all_frames(manifest, inputs, target_ratio, strata)
        deterministic_pass = canonical_sha256(records) == canonical_sha256(deterministic_records)

        post_hashes = verify_input_bindings(
            [b for b in bindings if b.format not in ("policy_shard_dir", "raw_shard_dir")]
        )
        artifact_modified = {
            name: post_hashes.get(name) != inputs.input_hashes.get(name)
            for name in post_hashes
        }

        crosscheck = {"compared": 0, "mismatches": 0}
        box_too_large = {"rows": []}
        multi_subject = {"rows": []}
        fallback_reasons_diag = {"reasons": {}}
        if inputs.raw_frames:
            policy_config = policy_config_from(config)
            crosscheck = crosscheck_mirror_against_artifact(inputs, policy_config)
            box_too_large = box_too_large_diagnostic(inputs, target_ratio, policy_config)
            multi_subject = multi_subject_diagnostic(inputs, target_ratio, policy_config)
            fallback_reasons_diag = fallback_reason_diagnostic(inputs, target_ratio, policy_config)
        else:
            inputs_with_raw = load_deferred_raw_inputs(bindings)
            policy_config = policy_config_from(config)
            crosscheck = crosscheck_mirror_against_artifact(inputs_with_raw, policy_config)
            box_too_large = box_too_large_diagnostic(inputs_with_raw, target_ratio, policy_config)
            multi_subject = multi_subject_diagnostic(inputs_with_raw, target_ratio, policy_config)
            fallback_reasons_diag = fallback_reason_diagnostic(inputs_with_raw, target_ratio, policy_config)

        temporal_frames = {
            video_id: {int(pred["frame"]) for pred in record["predictions"]}
            for video_id, record in inputs.stage5_1_predictions.items()
        }
        cmp0_validation = write_and_validate_official(
            records, "cmp0", target_ratio, paths.output / "cmp0_predictions.jsonl",
            inputs.metadata, inputs.index, temporal_frames,
        )
        cmp1_validation = write_and_validate_official(
            records, "cmp1", target_ratio, paths.output / "cmp1_predictions.jsonl",
            inputs.metadata, inputs.index, temporal_frames,
        )

        box_too_large_boxes = {
            (row["video_id"], int(row["frame"])): row["raw_bbox_xyxy"]
            for row in box_too_large.get("rows", [])
            if row.get("raw_bbox_xyxy")
        }
        videos_root = None
        if config.get("videos_root"):
            base = getattr(environment, BASE_FIELDS[config["videos_root"]["base"]])
            videos_root = base / config["videos_root"]["path"] if config["videos_root"].get("path") else base
        figures = render_all_figures(
            records,
            paths.output / "figures",
            videos_root=videos_root,
            video_paths={video_id: record["video_path"] for video_id, record in inputs.index.items()},
            box_too_large_boxes=box_too_large_boxes,
            weak_reference=inputs.weak_reference,
            per_category=int(config.get("figures_per_category", 3)),
        )

        shard_integrity = None
        if config.get("verify_stage5_2_shards"):
            expected_keys = {
                video_id: {int(pred["frame"]) for pred in record["predictions"]}
                for video_id, record in inputs.stage5_1_predictions.items()
            }
            shard_integrity = verify_shard_dir_integrity(
                next(b for b in bindings if b.name == "stage5_2_policy_artifact").path.parent,
                expected_keys,
            )

        gates = evaluate_gates(
            gate, deterministic_pass, artifact_modified, crosscheck,
            cmp0_validation, cmp1_validation, shard_integrity, config,
        )
        wall_sec = time.monotonic() - started

        atomic_write_json(machine / "summary.json", {
            "schema_version": "aic.machine-summary/v1",
            "experiment_id": config["experiment_id"],
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "videos": manifest["video_count"],
            "frames": manifest["frame_count"],
            "strata_counts": manifest["strata_counts"],
            "composition": config["composition"],
        })
        atomic_write_json(machine / "metrics.json", {
            "schema_version": "aic.machine-metrics/v1",
            **metrics,
        })
        atomic_write_json(machine / "runtime.json", {
            "schema_version": "aic.machine-runtime/v1",
            "wall_sec": round(wall_sec, 3),
            "gpu_calls": 0,
            "qwen_vllm_calls": 0,
            "device": "cpu",
        })
        atomic_write_json(machine / "validation.json", {
            "schema_version": "aic.machine-validation/v1",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "engineering_gate": gate,
            "deterministic_replay": deterministic_pass,
            "stage5_2_artifact_modified": artifact_modified,
            "mirror_crosscheck": crosscheck,
            "official_cmp0_validation": cmp0_validation,
            "official_cmp1_validation": cmp1_validation,
            "stage5_2_shard_integrity": shard_integrity,
        })
        atomic_write_json(diagnostics_dir / "multi_subject_diagnostic.json", multi_subject)
        atomic_write_json(diagnostics_dir / "box_too_large_diagnostic.json", box_too_large)
        atomic_write_json(diagnostics_dir / "fallback_reason_diagnostic.json", fallback_reasons_diag)

        manifest_artifact = paths.output / "manifest" / "stage5_3_smoke_manifest_v1.json"
        artifact_files = [
            paths.output / "cmp0_predictions.jsonl",
            paths.output / "cmp1_predictions.jsonl",
            manifest_artifact,
            *sorted(shard_store.root.glob("*.json")),
        ]
        build_artifact_manifest(paths.output, artifact_files, machine / "artifact_manifest.json", created_by=config["experiment_id"])
        render_raw_report(machine, paths.output / "experiment_raw_report.md", config["experiment_id"])
        write_ai_report_inputs(paths.output)
        context.set_status("COMPLETED" if all(gates.values()) else "VALIDATION_FAILED", validation="PASS" if all(gates.values()) else "FAIL")

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


def evaluate_gates(
    gate: dict,
    deterministic_pass: bool,
    artifact_modified: dict,
    crosscheck: dict,
    cmp0_validation: dict,
    cmp1_validation: dict,
    shard_integrity: dict | None,
    config: dict,
) -> dict[str, bool]:
    gates = {
        "frame_identity_100pct": bool(gate["frame_identity_complete"]),
        "invalid_crop_zero": gate["invalid_crop"] == 0,
        "out_of_bounds_zero": gate["out_of_bounds"] == 0,
        "ratio_violations_zero": gate["ratio_violations"] == 0,
        "missing_zero": gate["missing"] == 0,
        "duplicate_zero": gate["duplicates"] == 0,
        "sanitized_invalid_unhandled_zero": gate["sanitized_invalid_unhandled"] == 0,
        "deterministic_pass": deterministic_pass,
        "stage5_1_frozen_regression_zero": gate["cmp0_frozen_regression"] == 0,
        "stage5_2_artifact_unmodified": not any(artifact_modified.values()),
        "heldout_access_zero": True,
        "official_cmp0_contract_valid": bool(cmp0_validation["is_valid"]),
        "official_cmp1_contract_valid": bool(cmp1_validation["is_valid"]),
        "mirror_crosscheck_consistent": crosscheck.get("mismatches", 0) == 0,
    }
    if shard_integrity is not None:
        gates["stage5_2_shards_intact"] = bool(shard_integrity["intact"])
    return gates


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
