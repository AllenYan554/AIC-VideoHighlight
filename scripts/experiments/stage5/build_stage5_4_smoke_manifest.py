#!/usr/bin/env python3
"""Build the frozen Stage 5.4 smoke manifest (model-blind, deterministic).

The manifest is selected from the FULL frozen Dev166 frame identity (frozen
Stage 5.1 frame set + Stage 5.2 policy artifact) and must be built and
SHA-frozen BEFORE the stage5_4_smoke run. Selection is model-blind: it measures
temporal jitter on the frozen CMP-1 geometry only and never executes or
consults any Stage 5.4 smoothing output.

Frozen selection rule (all thresholds come from the experiment config):
1. per video, split the frozen frame identity into dense runs (consecutive
   frame ids); videos are eligible when their longest dense run has at least
   ``min_dense_run_length`` frames;
2. per-video jitter = mean |delta subject-center-x| / W over paired transitions
   (adjacent frozen frames, both non-fallback) across all dense runs;
3. rank eligible videos by (jitter, video_id), split into equal terciles
   low / moderate / high, and take the first ``videos_per_tercile`` of each;
4. coverage top-ups (deterministic, by video_id): ensure at least
   ``min_videos_with_fallback_transitions`` videos containing fallback
   transitions and at least ``min_videos_with_ambiguous_frames`` videos
   containing ambiguous (multi-subject) frames;
5. per selected video emit the longest dense run truncated to its first
   ``max_frames_per_video`` frozen frames (fallback frames inside the run are
   kept: they exercise the Stage 5.4 fallback reset rule);
6. scale guards: at most ``max_videos`` videos and ``max_frames`` frames.

The manifest content is deterministic (no timestamps); its canonical SHA-256
must be pinned into configs/experiments/stage5/stage5_4_smoke.json before the
smoke run (the smoke runner refuses to start otherwise).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.io import atomic_write_json
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.center_crop import compute_center_crop
from aic_video_highlight.spatial_composition.composition_metrics import (
    STRATUM_NO_SUBJECT,
    classify_center_stratum,
    horizontal_center_offset,
)
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputError,
    InputBinding,
    load_frozen_inputs,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    compute_subject_shifted_crop,
    sanitize_primary_bbox,
)
from aic_video_highlight.spatial_localization.subject_localization import STATUS_PRIMARY

BASE_FIELDS = {
    "archive": "archive",
    "outputs": "outputs",
    "datasets": "datasets",
    "repo": "repo",
    "models": "models",
    "logs": "logs",
    "cache": "cache",
    "tmp": "tmp",
}

TERCILES = ("low", "moderate", "high")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="accepted; the builder is deterministic and stateless")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="accepted; the builder is deterministic and stateless")
    return parser.parse_args(argv)


def resolve_bindings(config: dict, environment: EnvironmentPaths) -> list[InputBinding]:
    bindings = []
    for name, entry in config["inputs"].items():
        if name == "stage5_2_raw_detector":
            continue
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
        "weak_spatial_reference",
    }
    missing = required - {binding.name for binding in bindings}
    if missing:
        raise FrozenInputError(f"builder config inputs missing: {sorted(missing)}")
    return bindings


def round6(value):
    return None if value is None else round(value, 6)


def dense_runs(frames: list[int]) -> list[list[int]]:
    """Maximal runs of consecutive frame ids from an ascending frame list."""
    runs: list[list[int]] = []
    current: list[int] = []
    for frame in frames:
        if current and frame - current[-1] == 1:
            current.append(frame)
        else:
            if current:
                runs.append(current)
            current = [frame]
    if current:
        runs.append(current)
    return runs


def build_video_entry(
    video_id: str,
    policy_frames: list[dict],
    metadata: dict,
    index_record: dict,
    target_ratio: list[float],
    strata_thresholds: dict,
) -> dict:
    meta = metadata[video_id]
    width, height = int(meta["width"]), int(meta["height"])
    tw, th = float(target_ratio[0]), float(target_ratio[1])
    center_box = compute_center_crop(width, height, tw, th)
    entries: list[dict] = []
    for record in sorted(policy_frames, key=lambda item: int(item["frame"])):
        frame = int(record["frame"])
        primary = record.get("primary")
        raw_bbox = [float(value) for value in primary["xyxy"]] if primary else None
        sanitized = sanitize_primary_bbox(raw_bbox, width, height)
        fallback = record["status"] != STATUS_PRIMARY or raw_bbox is None or not sanitized.valid
        if fallback:
            subject_center_x = None
            subject_center_y = None
        else:
            subject_center_x = round6(sanitized.center_x)
            subject_center_y = round6(sanitized.center_y)
        if sanitized.valid:
            offset = round6(horizontal_center_offset(sanitized, width))
            stratum = classify_center_stratum(
                offset,
                float(strata_thresholds["near_center_lt"]),
                float(strata_thresholds["strongly_off_center_gte"]),
            )
        else:
            offset = None
            stratum = STRATUM_NO_SUBJECT
        entries.append(
            {
                "frame": frame,
                "stratum": stratum,
                "horizontal_center_offset": offset,
                "cmp1_fallback": fallback,
                "ambiguous": bool(record.get("ambiguous")),
                "subject_center_x": subject_center_x,
                "subject_center_y": subject_center_y,
            }
        )
    runs = dense_runs([entry["frame"] for entry in entries])
    by_frame = {entry["frame"]: entry for entry in entries}
    displacements: list[float] = []
    fallback_transitions = 0
    for run in runs:
        for previous_frame, current_frame in zip(run, run[1:]):
            previous_entry = by_frame[previous_frame]
            current_entry = by_frame[current_frame]
            if previous_entry["cmp1_fallback"] != current_entry["cmp1_fallback"]:
                fallback_transitions += 1
            if previous_entry["cmp1_fallback"] or current_entry["cmp1_fallback"]:
                continue
            displacements.append(
                abs(current_entry["subject_center_x"] - previous_entry["subject_center_x"]) / float(width)
            )
    longest_run = max(runs, key=len) if runs else []
    jitter_mean = round6(sum(displacements) / len(displacements)) if displacements else 0.0
    return {
        "video_id": video_id,
        "image_width": width,
        "image_height": height,
        "frame_count": len(entries),
        "longest_dense_run_length": len(longest_run),
        "jitter": {
            "paired_transitions": len(displacements),
            "mean_subject_center_displacement_norm": jitter_mean,
            "max_subject_center_displacement_norm": round6(max(displacements)) if displacements else 0.0,
            "fallback_transition_count": fallback_transitions,
            "fallback_frame_count": sum(1 for entry in entries if entry["cmp1_fallback"]),
            "ambiguous_frame_count": sum(1 for entry in entries if entry["ambiguous"]),
        },
        "_entries": entries,
        "_longest_run": longest_run,
        "_fallback_transition_count": fallback_transitions,
        "_ambiguous_frame_count": sum(1 for entry in entries if entry["ambiguous"]),
    }


def select_videos(video_entries: list[dict], selection: dict) -> tuple[list[dict], dict]:
    min_run = int(selection["min_dense_run_length"])
    eligible = [entry for entry in video_entries if entry["longest_dense_run_length"] >= min_run]
    if len(eligible) < 3:
        raise FrozenInputError(f"only {len(eligible)} eligible videos (longest dense run >= {min_run})")
    ranked = sorted(eligible, key=lambda entry: (entry["jitter"]["mean_subject_center_displacement_norm"], entry["video_id"]))
    third = len(ranked) // 3
    tercile_by_video: dict[str, str] = {}
    for index, entry in enumerate(ranked):
        tercile_by_video[entry["video_id"]] = TERCILES[0] if index < third else (
            TERCILES[1] if index < 2 * third else TERCILES[2]
        )
    selected: dict[str, dict] = {}
    for tercile in TERCILES:
        picked = 0
        for entry in ranked:
            if tercile_by_video[entry["video_id"]] != tercile or picked >= int(selection["videos_per_tercile"]):
                continue
            selected[entry["video_id"]] = entry
            picked += 1
    by_id = {entry["video_id"]: entry for entry in ranked}

    def top_up(predicate_key: str, minimum: int) -> None:
        have = sum(1 for entry in selected.values() if entry[predicate_key] > 0)
        for entry in sorted(ranked, key=lambda item: item["video_id"]):
            if have >= minimum:
                break
            if entry[predicate_key] > 0 and entry["video_id"] not in selected:
                selected[entry["video_id"]] = entry
                have += 1

    top_up("_fallback_transition_count", int(selection["min_videos_with_fallback_transitions"]))
    top_up("_ambiguous_frame_count", int(selection["min_videos_with_ambiguous_frames"]))

    tercile_counts = {tercile: sum(1 for entry in selected.values() if tercile_by_video[entry["video_id"]] == tercile) for tercile in TERCILES}
    coverage = {
        "eligible_videos": len(eligible),
        "selected_videos": len(selected),
        "jitter_terciles": tercile_counts,
        "videos_with_fallback_transitions": sum(1 for entry in selected.values() if entry["_fallback_transition_count"] > 0),
        "videos_with_ambiguous_frames": sum(1 for entry in selected.values() if entry["_ambiguous_frame_count"] > 0),
    }
    failures = []
    if any(count == 0 for count in tercile_counts.values()):
        failures.append("a jitter tercile is empty")
    if coverage["videos_with_fallback_transitions"] < int(selection["min_videos_with_fallback_transitions"]):
        failures.append("not enough videos with fallback transitions")
    if coverage["videos_with_ambiguous_frames"] < int(selection["min_videos_with_ambiguous_frames"]):
        failures.append("not enough videos with ambiguous (multi-subject) frames")
    if failures:
        raise FrozenInputError("smoke selection coverage insufficient: " + "; ".join(failures))
    return list(selected.values()), coverage


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.resume or args.validate_only:
        print("note: the builder is deterministic and stateless; flags ignored")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_json(args.environment)
    bindings = resolve_bindings(config, environment)
    if args.dry_run:
        print(json.dumps({
            "experiment_id": config["experiment_id"],
            "inputs": {binding.name: str(binding.path) for binding in bindings},
            "output": str(Path(config["output"]["path"]) if Path(config["output"]["path"]).is_absolute() else getattr(environment, BASE_FIELDS[config["output"]["base"]]) / config["output"]["path"]),
            "action": "NONE",
        }, ensure_ascii=False, indent=2))
        return 0

    inputs = load_frozen_inputs(
        bindings,
        include_raw=False,
        policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
    )

    policy_by_video: dict[str, list[dict]] = {}
    for record in inputs.policy_records:
        policy_by_video.setdefault(record["video_id"], []).append(record)
    for video_id in policy_by_video:
        if video_id not in inputs.metadata:
            raise FrozenInputError(f"policy video missing from metadata cache: {video_id}")

    strata_thresholds = config["composition"]["strata_thresholds"]
    target_ratio = [float(value) for value in config["composition"]["target_ratio"]]
    video_entries = [
        build_video_entry(
            video_id,
            policy_frames,
            inputs.metadata,
            inputs.index[video_id],
            target_ratio,
            strata_thresholds,
        )
        for video_id, policy_frames in sorted(policy_by_video.items())
    ]
    selected, coverage = select_videos(video_entries, config["selection"])

    max_frames = int(config["selection"]["max_frames_per_video"])
    videos_payload = []
    for entry in sorted(selected, key=lambda item: item["video_id"]):
        run_frames = set(entry["_longest_run"][:max_frames])
        frames = [frame_entry for frame_entry in entry["_entries"] if frame_entry["frame"] in run_frames]
        videos_payload.append(
            {
                "video_id": entry["video_id"],
                "image_width": entry["image_width"],
                "image_height": entry["image_height"],
                "target_ratio_wh": target_ratio,
                "frame_count": len(frames),
                "longest_dense_run_length": entry["longest_dense_run_length"],
                "jitter": entry["jitter"],
                "frames": frames,
            }
        )
    frame_count = sum(video["frame_count"] for video in videos_payload)
    if len(videos_payload) > int(config["selection"]["max_videos"]):
        raise FrozenInputError(f"scale guard exceeded: {len(videos_payload)} videos")
    if frame_count > int(config["selection"]["max_frames"]):
        raise FrozenInputError(f"scale guard exceeded: {frame_count} frames")

    manifest: dict = {
        "manifest_id": config["manifest"]["manifest_id"],
        "model_blind": True,
        "selection_rule": (
            "model-blind selection over the full frozen Dev166 frame identity: eligibility by "
            "dense-run length; per-video jitter measured on the frozen CMP-1 subject centers "
            "(adjacent frozen frames, both non-fallback); rank terciles low/moderate/high with "
            "deterministic coverage top-ups (fallback transitions, ambiguous frames); per video "
            "the longest dense run truncated to the first max_frames_per_video frozen frames; "
            "no Stage 5.4 smoothing output is executed or consulted at build time"
        ),
        "strata_thresholds": strata_thresholds,
        "target_ratio_wh": target_ratio,
        "video_count": len(videos_payload),
        "frame_count": frame_count,
        "coverage": coverage,
        "videos": videos_payload,
        "frozen_bindings": {name: inputs.input_hashes[name] for name in sorted(inputs.input_hashes)},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)

    output_base = getattr(environment, BASE_FIELDS[config["output"]["base"]])
    output_path = Path(config["output"]["path"])
    output_path = output_path if output_path.is_absolute() else output_base / output_path
    atomic_write_json(output_path, manifest)
    print(json.dumps({
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "video_count": manifest["video_count"],
        "frame_count": manifest["frame_count"],
        "coverage": coverage,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FrozenInputError as exc:
        print(f"manifest build failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
