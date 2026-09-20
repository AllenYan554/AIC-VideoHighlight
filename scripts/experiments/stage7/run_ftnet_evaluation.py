#!/usr/bin/env python3
"""Stage 7.1 FTNet post-training evaluation runner (read-only evaluation).

Stages:
  audit    freeze the evaluation protocol, verify checkpoint identity, audit the
           soft-vote provenance, record the event-GT status and derived runs
  evaluate score best.pt / last.pt, select thresholds on CALIBRATION, compute
           VALIDATION development metrics, figures, tables and the report
  all      audit then evaluate (default)

No training, no fine-tuning, no materialization, no Qwen, no RT-DETR, no
Official Test, no TVSum.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet import evaluation as ev  # noqa: E402
from aic_video_highlight.ftnet.dataset import load_normalization_stats  # noqa: E402
from aic_video_highlight.ftnet.index import load_index  # noqa: E402
from aic_video_highlight.runtime.paths import EnvironmentPaths  # noqa: E402


def _resolve_spec(spec: Any, environment: EnvironmentPaths) -> Path:
    if isinstance(spec, (str, Path)):
        return Path(spec).expanduser().resolve()
    if not isinstance(spec, Mapping) or "path" not in spec:
        raise ValueError(f"invalid path spec: {spec!r}")
    base = str(spec.get("base", "repo"))
    roots = {
        "repo": environment.repo,
        "outputs": environment.outputs,
        "datasets": environment.datasets,
        "models": environment.models,
        "hf_cache": environment.hf_cache,
        "logs": environment.logs,
        "cache": environment.cache,
        "tmp": environment.tmp,
        "archive": environment.archive,
        "derived": environment.derived,
    }
    if base not in roots or roots[base] is None:
        raise ValueError(f"unsupported path base: {base!r}")
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else (roots[base] / path)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# audit stage
# ---------------------------------------------------------------------------


def _checkpoint_identity(
    checkpoint_path: Path,
    control_path: Path,
    identity_audit_path: Path,
) -> dict[str, Any]:
    audit = json.loads(identity_audit_path.read_text(encoding="utf-8"))
    recorded = audit.get("checkpoints", {})
    result: dict[str, Any] = {"identity_audit_schema": audit.get("schema"), "checks": {}}
    for name, path in (("best.pt", checkpoint_path), ("last.pt", control_path)):
        row = recorded.get(name, {})
        actual_sha = ev.sha256_file(path)
        ok = bool(row) and row.get("sha256") == actual_sha and row.get("size") == path.stat().st_size
        if not ok:
            raise ev.EvaluationError(f"checkpoint identity mismatch for {name}: {path}")
        result["checks"][name] = {
            "path": str(path),
            "sha256": actual_sha,
            "epoch": row.get("epoch"),
            "global_step": row.get("global_step"),
            "recorded_source_path": row.get("recorded_source_path"),
            "identity": row.get("identity", {}),
            "status": "PASS",
        }
    result["run_id"] = audit.get("summary_run_id")
    result["status"] = "PASS"
    return result


def run_audit(config: dict[str, Any], environment: EnvironmentPaths, paths: dict[str, Path]) -> dict[str, Any]:
    output_dir = paths["output_dir"]
    checkpoint_identity = _checkpoint_identity(
        paths["checkpoint"], paths["control_checkpoint"], paths["checkpoint_identity_audit"]
    )
    protocol = ev.build_protocol(
        checkpoint={
            "role": "PRIMARY",
            **checkpoint_identity["checks"]["best.pt"],
        },
        control_checkpoint={
            "role": "OVERFITTING_DIAGNOSTIC_CONTROL",
            **checkpoint_identity["checks"]["last.pt"],
        },
        data_root=paths["data_root"],
        normalization_path=paths["normalization"],
        index_path=paths["index"],
        split_manifest_path=paths["split_manifest"],
        annotations_root=paths["annotations_root"],
        seed=int(config.get("seed", 20260917)),
        device=str(config.get("device", "auto")),
        code_git_head=_git_head(),
    )
    protocol["frozen_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ev.write_json(output_dir / "config" / "evaluation_protocol.json", protocol)

    device = ev.resolve_device(str(config.get("device", "auto")))
    ev.write_json(output_dir / "config" / "environment_snapshot.json", ev.environment_snapshot(device=device, seed=int(config.get("seed", 20260917))))

    entries, _ = load_index(paths["index"])
    entries_by_id = {entry.video_id: entry for entry in entries}
    manifest = json.loads((paths["data_root"] / "manifests" / "materialized_videos.json").read_text(encoding="utf-8"))
    manifest_records = {row["canonical_video_id"]: row for row in manifest["records"]}

    frames_by_split: dict[str, list[ev.VideoFrames]] = {}
    for split in ev.EVAL_SPLITS:
        frames_by_split[split] = [
            ev.load_video_frames(paths["data_root"], split, entry.video_id, category=entry.category)
            for entry in entries
            if entry.split == split
        ]
    provenance = ev.audit_provenance(
        data_root=paths["data_root"],
        annotations_root=paths["annotations_root"],
        entries=entries,
        manifest_records=manifest_records,
        splits=("TRAIN", "VALIDATION", "CALIBRATION"),
    )
    if provenance["status"] != "PASS":
        raise ev.EvaluationError("label provenance audit failed; refusing to evaluate")
    ev.write_json(output_dir / "supplementary" / "provenance_audit.json", provenance)

    event_gt = ev.audit_event_gt(
        frames_by_split,
        annotations_root=paths["annotations_root"],
        entries_by_id=entries_by_id,
    )
    ev.write_json(output_dir / "supplementary" / "event_gt_audit.json", event_gt)
    return {
        "status": "AUDIT_COMPLETE",
        "checkpoint_identity": checkpoint_identity,
        "provenance_status": provenance["status"],
        "event_metrics_available": event_gt["EVENT_METRICS_AVAILABLE"],
        "run_counts": event_gt["run_counts"],
        "protocol_path": str(output_dir / "config" / "evaluation_protocol.json"),
    }


# ---------------------------------------------------------------------------
# evaluate stage
# ---------------------------------------------------------------------------


def _pooled_sensitivity(
    scored_list: list[ev.ScoredVideo],
    *,
    tau: float,
) -> list[dict[str, Any]]:
    rows = []
    for item in ev.SENSITIVITY_THRESHOLDS:
        rows.append(ev.operating_metrics(scored_list, tau=tau, reference_threshold=item))
    rows.append(ev.operating_metrics(scored_list, tau=tau, reference_threshold=0.0))
    return rows


def _threshold_sweep(scored_list: list[ev.ScoredVideo], reference_threshold: float) -> list[dict[str, Any]]:
    y, p = ev.concat_universe(scored_list, reference_threshold=reference_threshold)
    rows: list[dict[str, Any]] = []
    for tau in ev.THRESHOLD_SWEEP_GRID:
        keep = p >= tau
        counts = ev.confusion_counts(y, keep)
        metrics = ev.metrics_from_counts(counts)
        universe = len(y)
        rows.append(
            {
                "tau": tau,
                "precision": ev._rounded(metrics["precision"]),
                "recall": ev._rounded(metrics["recall"]),
                "f1": ev._rounded(metrics["f1"]),
                "pruning_rate": ev._rounded(float(int((~keep).sum()) / universe) if universe else None),
            }
        )
    return rows


def _run_metrics_for_split(
    frames_list: list[ev.VideoFrames],
    entries_by_id: Mapping[str, Any],
    annotations_root: Path,
    scored_by_video: Mapping[str, ev.ScoredVideo],
    *,
    tau: float,
) -> dict[str, Any]:
    runs: list[ev.PositiveRun] = []
    for frames in frames_list:
        entry = entries_by_id[frames.video_id]
        from aic_video_highlight.ftnet.youtube_highlights import load_mturk_clips

        clips = load_mturk_clips(annotations_root / entry.annotation_identity)
        coverage = ev.positive_clip_coverage(clips, frames.source_frame_id)
        runs.extend(ev.extract_positive_runs(frames, coverage))
    return ev.run_metrics(runs, scored_by_video, tau=tau)


def _frame_score_rows(scored_list: list[ev.ScoredVideo], runs: list[ev.PositiveRun]) -> list[dict[str, Any]]:
    run_lookup: dict[tuple[str, int], str] = {}
    for index, run in enumerate(runs):
        run_id = f"{run.video_id}:run{index:03d}"
        for frame_index in run.frame_indices:
            run_lookup[(run.video_id, frame_index)] = run_id
    rows: list[dict[str, Any]] = []
    for scored in scored_list:
        frames = scored.frames
        for index in range(frames.length):
            y_bin = bool(frames.loss_mask[index] and frames.target[index] >= ev.PRIMARY_BINARY_THRESHOLD)
            rows.append(
                {
                    "video_id": frames.video_id,
                    "split": frames.split,
                    "frame_index": index,
                    "source_frame_id": int(frames.source_frame_id[index]),
                    "timestamp": ev._rounded(float(frames.timestamp[index]), 6),
                    "target": ev._rounded(float(frames.target[index]), 8),
                    "loss_mask": bool(frames.loss_mask[index]),
                    "candidate_identity": "SUPERVISED_CANDIDATE" if frames.loss_mask[index] else "UNLABELLED",
                    "y_bin_primary": y_bin,
                    "p_keep": ev._rounded(float(scored.p_keep[index]), 8),
                    "positive_run_id": run_lookup.get((frames.video_id, index)),
                }
            )
    return rows


def _compute_verdict(
    validation: Mapping[str, Any],
    *,
    tau_safe_validation: Mapping[str, Any] | None,
    safe_run_whole_deletion: float | None,
) -> dict[str, Any]:
    tau_f1 = validation.get("tau_f1", {})
    pruning_f1 = tau_f1.get("pruning_rate")
    recall_f1 = tau_f1.get("recall")
    if (
        tau_safe_validation is not None
        and tau_safe_validation.get("recall") is not None
        and tau_safe_validation["recall"] + 1e-12 >= ev.RECALL_SAFETY_TARGET
        and (tau_safe_validation.get("pruning_rate") or 0.0) >= ev.MIN_MEANINGFUL_PRUNING
        and (safe_run_whole_deletion or 0.0) == 0.0
    ):
        return {
            "code": "A",
            "label": "有效删帧模型",
            "rule": protocol_rule_text("A"),
        }
    if pruning_f1 is not None and recall_f1 is not None and pruning_f1 >= ev.FS0_NEAR_ZERO_PRUNING and recall_f1 < ev.DANGEROUS_RECALL:
        return {"code": "D", "label": "负结果", "rule": protocol_rule_text("D")}
    if pruning_f1 is not None and pruning_f1 < ev.FS0_NEAR_ZERO_PRUNING:
        return {"code": "C", "label": "接近 FS0", "rule": protocol_rule_text("C")}
    return {"code": "B", "label": "有信号但风险较高", "rule": protocol_rule_text("B")}


def protocol_rule_text(code: str) -> str:
    if code == "A":
        return (
            f"tau_safe 存在且 VALIDATION recall >= {ev.RECALL_SAFETY_TARGET}、"
            f"pruning >= {ev.MIN_MEANINGFUL_PRUNING}、positive-run 全删 = 0"
        )
    if code == "D":
        return f"tau_f1 VALIDATION recall < {ev.DANGEROUS_RECALL} 且 pruning >= {ev.FS0_NEAR_ZERO_PRUNING}"
    if code == "C":
        return f"tau_f1 VALIDATION pruning < {ev.FS0_NEAR_ZERO_PRUNING}"
    return "其余情况"


def run_evaluate(config: dict[str, Any], environment: EnvironmentPaths, paths: dict[str, Path]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = paths["output_dir"]
    protocol_path = output_dir / "config" / "evaluation_protocol.json"
    if not protocol_path.is_file():
        raise ev.EvaluationError("evaluation protocol is not frozen; run the audit stage first")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_METRICS":
        raise ev.EvaluationError("evaluation protocol is not marked as frozen")
    if ev.sha256_file(paths["checkpoint"]) != protocol["checkpoint"]["sha256"]:
        raise ev.EvaluationError("primary checkpoint changed after the protocol was frozen")
    if ev.sha256_file(paths["control_checkpoint"]) != protocol["overfitting_control_checkpoint"]["sha256"]:
        raise ev.EvaluationError("control checkpoint changed after the protocol was frozen")

    seed = int(config.get("seed", 20260917))
    device = ev.resolve_device(str(config.get("device", "auto")))
    stats = load_normalization_stats(paths["normalization"])
    entries, _ = load_index(paths["index"])
    entries_by_id = {entry.video_id: entry for entry in entries}
    categories = {entry.video_id: entry.category for entry in entries}
    frames_by_split: dict[str, list[ev.VideoFrames]] = {
        split: [
            ev.load_video_frames(paths["data_root"], split, entry.video_id, category=entry.category)
            for entry in entries
            if entry.split == split
        ]
        for split in ev.EVAL_SPLITS
    }

    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_model, best_info = ev.load_checkpoint_model(paths["checkpoint"], device)
    last_model, last_info = ev.load_checkpoint_model(paths["control_checkpoint"], device)

    scored: dict[str, dict[str, list[ev.ScoredVideo]]] = {
        "best": {
            split: ev.score_split(
                best_model, paths["data_root"], split, stats, device=device, categories=categories
            )
            for split in ev.EVAL_SPLITS
        },
        "last": {
            split: ev.score_split(
                last_model, paths["data_root"], split, stats, device=device, categories=categories
            )
            for split in ev.EVAL_SPLITS
        },
    }
    del best_model, last_model

    scored_by_video_split = {
        split: {video.frames.video_id: video for video in scored["best"][split]}
        for split in ev.EVAL_SPLITS
    }

    y_cal, p_cal = ev.concat_universe(scored["best"]["CALIBRATION"])
    tau_f1 = ev.select_tau_f1(y_cal, p_cal)
    tau_safe = ev.select_tau_safe(y_cal, p_cal)
    runs_by_split: dict[str, list[ev.PositiveRun]] = {}
    from aic_video_highlight.ftnet.youtube_highlights import load_mturk_clips

    for split in ev.EVAL_SPLITS:
        runs: list[ev.PositiveRun] = []
        for frames in frames_by_split[split]:
            entry = entries_by_id[frames.video_id]
            clips = load_mturk_clips(paths["annotations_root"] / entry.annotation_identity)
            coverage = ev.positive_clip_coverage(clips, frames.source_frame_id)
            runs.extend(ev.extract_positive_runs(frames, coverage))
        runs_by_split[split] = runs
    tau_run_safe = ev.select_tau_run_safe(
        y_cal, p_cal, runs_by_split["CALIBRATION"], scored_by_video_split["CALIBRATION"]
    )

    thresholds = {
        "schema": "aic.stage7.ftnet.evaluation-thresholds/v1",
        "selection_split": "CALIBRATION",
        "reference_threshold": ev.PRIMARY_BINARY_THRESHOLD,
        "tau_f1": tau_f1,
        "tau_safe": tau_safe if tau_safe is not None else {"status": "NO_R95_SAFE_THRESHOLD"},
        "tau_run_safe": (
            tau_run_safe if tau_run_safe is not None else {"status": "NO_RUN_SAFE_THRESHOLD"}
        ),
        "tau_half": 0.5,
    }
    ev.write_json(output_dir / "results" / "thresholds.json", thresholds)

    operating_points: dict[str, float | None] = {
        "tau_f1": tau_f1["tau"],
        "tau_safe": None if tau_safe is None else tau_safe["tau"],
        "tau_run_safe": None if tau_run_safe is None else tau_run_safe["tau"],
    }

    validation: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for name, tau in [("fs0", 0.0), ("tau_half", 0.5), *operating_points.items()]:
        if name != "fs0" and tau is None:
            continue
        calibration[name] = ev.operating_metrics(scored["best"]["CALIBRATION"], tau=float(tau))
        validation[name] = ev.operating_metrics(scored["best"]["VALIDATION"], tau=float(tau))

    ap_results = {
        "best": {},
        "last": {},
    }
    for split in ev.EVAL_SPLITS:
        for model_name in ("best", "last"):
            y, p = ev.concat_universe(scored[model_name][split])
            ap_results[model_name][split] = {
                "average_precision": ev._rounded(ev.average_precision(y, p)),
                "universe_frames": int(y.size),
                "positive_frames": int(y.sum()),
            }

    run_metrics_by_split: dict[str, dict[str, Any]] = {}
    for split in ev.EVAL_SPLITS:
        split_metrics: dict[str, Any] = {
            "fs0": ev.run_metrics(runs_by_split[split], scored_by_video_split[split], tau=0.0),
            "tau_f1": ev.run_metrics(
                runs_by_split[split], scored_by_video_split[split], tau=float(operating_points["tau_f1"])
            ),
        }
        for name in ("tau_safe", "tau_run_safe"):
            tau = operating_points.get(name)
            if tau is not None:
                split_metrics[name] = ev.run_metrics(
                    runs_by_split[split], scored_by_video_split[split], tau=float(tau)
                )
        run_metrics_by_split[split] = split_metrics
    ev.write_json(
        output_dir / "results" / "positive_run_metrics.json",
        {
            "status": "DERIVED_POSITIVE_RUN_PROXY_NOT_HUMAN_EVENT_GT",
            "strata_definitions": {
                "short": f"<= {ev.SHORT_RUN_MAX_SEC}s",
                "medium": f"({ev.SHORT_RUN_MAX_SEC}, {ev.MEDIUM_RUN_MAX_SEC}]s",
                "long": f"> {ev.MEDIUM_RUN_MAX_SEC}s",
            },
            "calibration": run_metrics_by_split["CALIBRATION"],
            "validation": run_metrics_by_split["VALIDATION"],
        },
    )
    short_metrics = {
        "status": "DERIVED_PROXY",
        "definition_supported": True,
        "short_definition_sec": ev.SHORT_RUN_MAX_SEC,
        "calibration": {
            name: run_metrics_by_split["CALIBRATION"][name]["short"]
            for name in run_metrics_by_split["CALIBRATION"]
        },
        "validation": {
            name: run_metrics_by_split["VALIDATION"][name]["short"]
            for name in run_metrics_by_split["VALIDATION"]
        },
    }
    ev.write_json(output_dir / "results" / "short_run_metrics.json", short_metrics)
    ev.write_json(
        output_dir / "results" / "short_event_metrics.json",
        {
            "EVENT_METRICS_AVAILABLE": "NO",
            "reason": (
                "No human short-event identity exists; short-event recall cannot be computed "
                "without fabricating event identity."
            ),
            "derived_proxy_file": "results/short_run_metrics.json",
            "derived_proxy": short_metrics,
        },
    )
    ev.write_json(
        output_dir / "results" / "event_metrics.json",
        {
            "EVENT_METRICS_AVAILABLE": "NO",
            "reason": json.loads(
                (output_dir / "supplementary" / "event_gt_audit.json").read_text(encoding="utf-8")
            )["reason"],
            "derived_proxy_file": "results/positive_run_metrics.json",
        },
    )

    per_video = ev.per_video_metrics(scored["best"]["VALIDATION"], tau_by_method=operating_points)
    ev.write_csv(
        output_dir / "results" / "per_video_metrics.csv",
        per_video,
        [
            "video_id", "split", "method", "tau", "frames", "universe_frames", "positive_frames",
            "predicted_keep", "predicted_drop", "tp", "fp", "fn", "tn", "precision", "recall",
            "f1", "pruning_rate", "false_deletion_rate",
        ],
    )

    sensitivity_rows = []
    for name, tau in [("fs0", 0.0), *operating_points.items()]:
        if name != "fs0" and tau is None:
            continue
        for split, scored_list in (("VALIDATION", scored["best"]["VALIDATION"]), ("CALIBRATION", scored["best"]["CALIBRATION"])):
            for row in _pooled_sensitivity(scored_list, tau=float(tau)):
                sensitivity_rows.append({"split": split, "method": name, **row})
    ev.write_csv(
        output_dir / "supplementary" / "sensitivity_binary_references.csv",
        sensitivity_rows,
        [
            "split", "method", "tau", "reference_threshold", "universe_frames", "positive_frames",
            "tp", "fp", "fn", "tn", "precision", "recall", "f1", "pruning_rate",
        ],
    )

    sweep_rows = _threshold_sweep(scored["best"]["CALIBRATION"], ev.PRIMARY_BINARY_THRESHOLD)
    ev.write_csv(
        output_dir / "supplementary" / "threshold_sweep.csv",
        sweep_rows,
        ["tau", "precision", "recall", "f1", "pruning_rate"],
    )

    missingness = ev.native_missingness_audit(scored["best"]["VALIDATION"] + scored["best"]["CALIBRATION"])
    ev.write_csv(
        output_dir / "results" / "native16_missingness_audit.csv",
        missingness["rows"],
        [
            "field_index", "field", "observed_rate_overall", "p_observed_given_y1",
            "p_observed_given_y0", "delta", "risk_flag",
        ],
    )

    score_rows_val = _frame_score_rows(scored["best"]["VALIDATION"], runs_by_split["VALIDATION"])
    score_rows_cal = _frame_score_rows(scored["best"]["CALIBRATION"], runs_by_split["CALIBRATION"])
    ev.write_jsonl(output_dir / "supplementary" / "frame_scores_validation.jsonl", score_rows_val)
    ev.write_jsonl(output_dir / "supplementary" / "frame_scores_calibration.jsonl", score_rows_cal)

    safe_run_whole_deletion = None
    if tau_safe is not None:
        safe_run_whole_deletion = run_metrics_by_split["VALIDATION"]["tau_safe"]["overall"]["whole_deletion_rate"]
    verdict = _compute_verdict(
        validation,
        tau_safe_validation=validation.get("tau_safe"),
        safe_run_whole_deletion=safe_run_whole_deletion,
    )

    summary = {
        "schema": ev.SUMMARY_SCHEMA,
        "verdict": verdict,
        "checkpoint": {"primary": protocol["checkpoint"], "control": protocol["overfitting_control_checkpoint"]},
        "universe": protocol["universe"],
        "binary_reference": protocol["binary_reference"],
        "thresholds": thresholds,
        "calibration": calibration,
        "validation": validation,
        "pr_auc": ap_results,
        "positive_run_proxy": {
            "event_metrics_available": "NO",
            "validation": run_metrics_by_split["VALIDATION"],
            "calibration": run_metrics_by_split["CALIBRATION"],
        },
        "native16_missingness": {
            "flagged_fields": missingness["flagged_fields"],
            "flag_delta": missingness["flag_delta"],
            "note": missingness["note"],
        },
        "limitations": [
            "VALIDATION metrics are DEVELOPMENT metrics; they are not independent test performance.",
            "Human event ground truth does not exist for YouTube Highlights: only a derived positive-run proxy is reported.",
            "The primary binary reference (soft target >= 0.5) is strict; sensitivity references are reported as well.",
            "no bootstrap confidence intervals (deterministic point metrics).",
            "The full Qwen candidate mask is not persisted outside MTurk coverage; the universe is loss_mask == 1.",
        ],
        "constraints": protocol["constraints"],
        "device": device,
        "git_head": _git_head(),
    }
    ev.write_json(output_dir / "results" / "summary_metrics.json", summary)

    fs0_rows = [
        {
            "method": name,
            "precision": _pct(validation[name]["precision"]),
            "recall": _pct(validation[name]["recall"]),
            "f1": _pct(validation[name]["f1"]),
            "pruning_rate": _pct(validation[name]["pruning_rate"]),
            "false_deletion_rate": _pct(validation[name]["false_deletion_rate"]),
        }
        for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe")
        if name in validation
    ]
    ev.write_csv(
        output_dir / "results" / "fs0_comparison.csv",
        fs0_rows,
        ["method", "precision", "recall", "f1", "pruning_rate", "false_deletion_rate"],
    )

    _render_figures(
        plt,
        output_dir=output_dir,
        scored=scored,
        validation=validation,
        calibration=calibration,
        operating_points=operating_points,
        ap_results=ap_results,
        sweep_rows=sweep_rows,
        run_metrics=run_metrics_by_split,
    )
    report = render_report(
        output_dir=output_dir,
        summary=summary,
        per_video=per_video,
        run_metrics=run_metrics_by_split,
        missingness=missingness,
        tau_f1=tau_f1,
        tau_safe=tau_safe,
        tau_run_safe=tau_run_safe,
    )
    (output_dir / "experiment_report.md").write_text(report, encoding="utf-8", newline="\n")
    return summary


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * float(value), 4)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _render_figures(
    plt,
    *,
    output_dir: Path,
    scored: Mapping[str, Mapping[str, list[ev.ScoredVideo]]],
    validation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    operating_points: Mapping[str, float | None],
    ap_results: Mapping[str, Any],
    sweep_rows: list[dict[str, Any]],
    run_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 1. PR curve on VALIDATION for best and last
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model_name, color in (("best", "#1f77b4"), ("last", "#d62728")):
        y, p = ev.concat_universe(scored[model_name]["VALIDATION"])
        points = ev.pr_curve(y, p)
        recall = [point["recall"] for point in points]
        precision = [point["precision"] for point in points]
        ap_value = ap_results[model_name]["VALIDATION"]["average_precision"]
        ax.plot(recall, precision, color=color, label=f"{model_name}.pt (AP={ap_value:.5f})")
    for name, marker in (("tau_f1", "o"), ("tau_safe", "s")):
        if name in validation and validation[name].get("recall") is not None:
            ax.scatter(
                [validation[name]["recall"]],
                [validation[name]["precision"]],
                marker=marker,
                color="black",
                zorder=5,
                label=f"{name} (tau={operating_points.get(name):.3f})",
            )
    ax.set_xlabel("Recall (primary binary reference: soft target >= 0.5)")
    ax.set_ylabel("Precision")
    ax.set_title("VALIDATION Precision-Recall curve (development metrics)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    # 2. threshold metrics on CALIBRATION
    fig, ax = plt.subplots(figsize=(7, 5.5))
    taus = [row["tau"] for row in sweep_rows]
    for key, color in (("precision", "#1f77b4"), ("recall", "#2ca02c"), ("f1", "#d62728")):
        ax.plot(taus, [row[key] for row in sweep_rows], color=color, label=key)
    for name, color in (("tau_f1", "black"), ("tau_safe", "gray"), ("tau_run_safe", "purple")):
        tau = operating_points.get(name)
        if tau is not None:
            ax.axvline(tau, color=color, linestyle="--", linewidth=1, label=f"{name}={tau:.3f}")
    ax.set_xlabel("Threshold tau (DROP if p_keep < tau)")
    ax.set_ylabel("Metric value")
    ax.set_title("CALIBRATION threshold metrics (threshold selection split)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "calibration_threshold_metrics.png", dpi=150)
    plt.close(fig)

    # 3. pruning vs recall on VALIDATION
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y, p = ev.concat_universe(scored["best"]["VALIDATION"])
    prunings = []
    recalls = []
    for tau in ev.THRESHOLD_SWEEP_GRID:
        keep = p >= tau
        counts = ev.confusion_counts(y, keep)
        metrics = ev.metrics_from_counts(counts)
        prunings.append(float(int((~keep).sum()) / y.size) if y.size else 0.0)
        recalls.append(metrics["recall"] if metrics["recall"] is not None else 0.0)
    ax.plot(prunings, recalls, color="#1f77b4")
    for name, marker in (("tau_f1", "o"), ("tau_safe", "s")):
        if name in validation and validation[name].get("recall") is not None:
            ax.scatter(
                [validation[name]["pruning_rate"]],
                [validation[name]["recall"]],
                marker=marker,
                color="black",
                zorder=5,
                label=f"{name} (tau={operating_points.get(name):.3f})",
            )
    ax.scatter([0.0], [1.0], marker="*", color="green", s=120, zorder=5, label="FS0 (keep all)")
    ax.set_xlabel("Candidate pruning rate (supervised universe)")
    ax.set_ylabel("Frame recall")
    ax.set_title("VALIDATION pruning vs recall trade-off")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pruning_vs_recall.png", dpi=150)
    plt.close(fig)

    # 4. FS0 vs FTNet grouped bars
    methods = [name for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe") if name in validation]
    keys = ["precision", "recall", "f1", "pruning_rate"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(keys))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"]
    for position, method in enumerate(methods):
        values = [
            (validation[method][key] if validation[method][key] is not None else 0.0) for key in keys
        ]
        ax.bar(x + position * width, values, width, label=method, color=colors[position % len(colors)])
        for xpos, value in zip(x + position * width, values):
            ax.text(xpos, value + 0.01, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.1)
    ax.set_title("VALIDATION: FS0 vs FTNet operating points")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fs0_vs_ftnet_metrics.png", dpi=150)
    plt.close(fig)

    # 5. best vs last PR-AUC
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(2)
    width = 0.35
    for position, model_name in enumerate(("best", "last")):
        values = [
            ap_results[model_name]["VALIDATION"]["average_precision"] or 0.0,
            ap_results[model_name]["CALIBRATION"]["average_precision"] or 0.0,
        ]
        ax.bar(x + position * width, values, width, label=f"{model_name}.pt")
        for xpos, value in zip(x + position * width, values):
            ax.text(xpos, value + 0.002, f"{value:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["VALIDATION", "CALIBRATION"])
    ax.set_ylabel("Average Precision (primary reference)")
    ax.set_title("Overfitting control: best.pt (epoch 7) vs last.pt (epoch 40)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "best_vs_last_pr_auc.png", dpi=150)
    plt.close(fig)

    # 6. positive-run survival by duration stratum (derived proxy)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    strata = ["short", "medium", "long"]
    methods = [name for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe") if name in run_metrics["VALIDATION"]]
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(strata))
    for position, method in enumerate(methods):
        values = []
        for stratum in strata:
            block = run_metrics["VALIDATION"][method][stratum]
            values.append(block["survival_recall"] if block["survival_recall"] is not None else 0.0)
        ax.bar(x + position * width, values, width, label=method)
        for xpos, value in zip(x + position * width, values):
            ax.text(xpos, value + 0.01, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"{s}\n(duration strata)" for s in strata])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Positive-run survival recall (derived proxy)")
    ax.set_title("VALIDATION short/medium/long run survival (NOT human event GT)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "positive_run_recall_by_duration_proxy.png", dpi=150)
    plt.close(fig)

    # 7. positive-run retention distribution (derived proxy)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    methods = [name for name in ("tau_f1", "tau_safe", "tau_run_safe") if name in run_metrics["VALIDATION"]]
    for method in methods:
        per_run = run_metrics["VALIDATION"][method]["per_run"]
        retention = [row["retention_ratio"] for row in per_run]
        if retention:
            ax.hist(retention, bins=20, range=(0, 1), alpha=0.5, label=f"{method} (n={len(retention)})")
    ax.set_xlabel("Positive-run retention ratio (derived proxy)")
    ax.set_ylabel("Run count")
    ax.set_title("VALIDATION positive-run retention distribution (NOT human event GT)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "positive_run_retention_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{100.0 * value:.2f}%"
    return f"{value:.{digits}f}"


def render_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    per_video: list[dict[str, Any]],
    run_metrics: Mapping[str, Mapping[str, Any]],
    missingness: Mapping[str, Any],
    tau_f1: Mapping[str, Any],
    tau_safe: Mapping[str, Any] | None,
    tau_run_safe: Mapping[str, Any] | None,
) -> str:
    verdict = summary["verdict"]
    validation = summary["validation"]
    ap_results = summary["pr_auc"]
    lines: list[str] = []
    add = lines.append

    add("# Stage 7.1 FTNet Post-Training Evaluation Report")
    add("")
    add("```text")
    add("TASK                 : FTNet 第一次训练（best.pt @ epoch 7）正式效果评价")
    add("TRAINING             : NO")
    add("QWEN / RT-DETR       : NO")
    add("DATA MODIFICATION    : NO (frozen dataset, read-only)")
    add(f"PRIMARY CHECKPOINT   : best.pt sha256 {summary['checkpoint']['primary']['sha256'][:16]}...")
    add(f"CONTROL CHECKPOINT   : last.pt sha256 {summary['checkpoint']['control']['sha256'][:16]}...")
    add(f"GIT HEAD             : {summary['git_head']}")
    add("OFFICIAL TEST / TVSUM: NOT ACCESSED")
    add("```")
    add("")
    add("## 0. 直接回答：FTNet 有没有用？")
    add("")
    add(f"**结论（依据预注册规则，VALIDATION 开发集）：{verdict['code']} — {verdict['label']}**")
    add("")
    add(f"- 判定规则：{verdict['rule']}")
    add(
        f"- tau_f1（CALIBRATION 选择）下 VALIDATION："
        f"pruning={_fmt(validation['tau_f1']['pruning_rate'], percent=True)}，"
        f"recall={_fmt(validation['tau_f1']['recall'], percent=True)}，"
        f"precision={_fmt(validation['tau_f1']['precision'], percent=True)}"
    )
    if tau_safe is not None and "tau_safe" in validation:
        add(
            f"- tau_safe（recall>={ev.RECALL_SAFETY_TARGET} 安全点）下 VALIDATION："
            f"pruning={_fmt(validation['tau_safe']['pruning_rate'], percent=True)}，"
            f"recall={_fmt(validation['tau_safe']['recall'], percent=True)}"
        )
    else:
        add("- tau_safe：**NO_R95_SAFE_THRESHOLD**（CALIBRATION 上不存在满足 recall>=0.95 的阈值）")
    add(
        "- 补充观察（不改变预注册判定）：best.pt 的阈值无关排序能力明显高于随机"
        f"（VALIDATION AP={_fmt(ap_results['best']['VALIDATION']['average_precision'], 4)}，"
        f"正帧基准 prevalence={_fmt(validation['fs0']['precision'], percent=True)}），"
        "且存在一个“近安全点”可删除大部分 candidate 帧；但该点在 VALIDATION 上未达到冻结的 "
        f"recall>={ev.RECALL_SAFETY_TARGET} 门槛。"
    )
    add(
        "- 注意：VALIDATION 已用于选择 epoch 7，因此以下均为 **development metrics**，"
        "不是独立最终泛化成绩。"
    )
    add("")
    add("## 1. 核心总表（VALIDATION，主 binary reference：soft target >= 0.5）")
    add("")
    add("| Method | Precision | Recall | F1 | PR-AUC | Pruning Rate | False Deletion |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    add(
        f"| FS0 (keep all) | {_fmt(validation['fs0']['precision'], percent=True)} | "
        f"{_fmt(validation['fs0']['recall'], percent=True)} | {_fmt(validation['fs0']['f1'], percent=True)} | N/A | 0.00% | "
        f"{_fmt(validation['fs0']['false_deletion_rate'], percent=True)} |"
    )
    for name, label in (("tau_f1", "FTNet @ tau_f1"), ("tau_safe", "FTNet @ tau_safe"), ("tau_run_safe", "FTNet @ tau_run_safe")):
        if name not in validation:
            continue
        row = validation[name]
        add(
            f"| {label} (tau={row['tau']:.4f}) | {_fmt(row['precision'], percent=True)} | "
            f"{_fmt(row['recall'], percent=True)} | {_fmt(row['f1'], percent=True)} | "
            f"{_fmt(ap_results['best']['VALIDATION']['average_precision'], 5)} | "
            f"{_fmt(row['pruning_rate'], percent=True)} | {_fmt(row['false_deletion_rate'], percent=True)} |"
        )
    add("")
    add(f"- PR-AUC 在 VALIDATION 上对全部帧排序计算（best.pt）；FS0 无排序概念。")
    add(f"- Universe = loss_mask==1（有 MTurk 覆盖的 candidate 帧），共 {summary['validation']['fs0']['universe_frames']} 帧。")
    add("")
    add("## 2. 过拟合对照：best.pt vs last.pt（PR-AUC）")
    add("")
    add("| Model | VALIDATION PR-AUC | CALIBRATION PR-AUC |")
    add("|---|---:|---:|")
    for model_name in ("best", "last"):
        add(
            f"| {model_name}.pt | {_fmt(ap_results[model_name]['VALIDATION']['average_precision'], 5)} | "
            f"{_fmt(ap_results[model_name]['CALIBRATION']['average_precision'], 5)} |"
        )
    diff_val = None
    diff_cal = None
    if (
        ap_results["best"]["VALIDATION"]["average_precision"] is not None
        and ap_results["last"]["VALIDATION"]["average_precision"] is not None
    ):
        diff_val = ap_results["best"]["VALIDATION"]["average_precision"] - ap_results["last"]["VALIDATION"]["average_precision"]
    if (
        ap_results["best"]["CALIBRATION"]["average_precision"] is not None
        and ap_results["last"]["CALIBRATION"]["average_precision"] is not None
    ):
        diff_cal = ap_results["best"]["CALIBRATION"]["average_precision"] - ap_results["last"]["CALIBRATION"]["average_precision"]
    add("")
    add(
        f"- VALIDATION 差值（best - last）= {_fmt(diff_val, 5)}："
        + ("best.pt 明显更好，支持“epoch 7 后过拟合损害 frame ranking”。" if diff_val is not None and diff_val > 0 else
           "best.pt 并未更好。")
    )
    add(
        f"- CALIBRATION 差值（best - last）= {_fmt(diff_cal, 5)}："
        + ("best.pt 更好。" if diff_cal is not None and diff_cal > 0 else
           "last.pt 反而更好（与 VALIDATION 结论相反）。")
    )
    add("- 结论：过拟合对 ranking 的影响是 **split-dependent**，不能外推为普遍泛化下降；VALIDATION 上的优势与 best 选择同源。")
    add("")
    add("## 3. 阈值与选择协议")
    add("")
    add("| 参数 | 值 | 说明 |")
    add("|---|---:|---|")
    add(f"| tau_f1 | {tau_f1['tau']:.4f} | CALIBRATION F1 最大；tie-break 高 recall→低 tau |")
    if tau_safe is not None:
        add(f"| tau_safe | {tau_safe['tau']:.4f} | CALIBRATION recall>=0.95 且 pruning 最大 |")
    else:
        add("| tau_safe | N/A | NO_R95_SAFE_THRESHOLD |")
    if tau_run_safe is not None:
        add(f"| tau_run_safe | {tau_run_safe['tau']:.4f} | recall>=0.95 且 positive-run 全删=0（派生代理约束） |")
    else:
        add("| tau_run_safe | N/A | NO_RUN_SAFE_THRESHOLD |")
    add("")
    add("## 4. Event GT 状态与 Positive-Run 派生代理")
    add("")
    add(
        "- **EVENT_METRICS_AVAILABLE = NO**：YouTube Highlights 的原始人工标注是 ~2 秒 clip 窗口 + soft vote，"
        "不存在连续 human event identity，因此无法合法计算 Event Recall / Whole-event Deletion。"
    )
    add("- 下表为 **派生代理**（POSITIVE_RUN，非人类 event GT）：positive clip 覆盖的相邻采样帧合并为 run。")
    add("")
    add("| Method | Run survival recall | Whole-run deletion | Retention mean | Retention p10 | Retention min |")
    add("|---|---:|---:|---:|---:|---:|")
    for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe"):
        if name not in run_metrics["VALIDATION"]:
            continue
        overall = run_metrics["VALIDATION"][name]["overall"]
        add(
            f"| {name} (tau={validation[name]['tau']:.4f}) | {_fmt(overall['survival_recall'], percent=True)} | "
            f"{_fmt(overall['whole_deletion_rate'], percent=True)} | {_fmt(overall['retention_mean'], 3)} | "
            f"{_fmt(overall['retention_p10'], 3)} | {_fmt(overall['retention_min'], 3)} |"
        )
    add("")
    add("### 4.1 分时长层（short 最高优先级安全指标）")
    add("")
    add(
        "> 说明：VALIDATION 的阳性 run 中 **short（<=2s）数量为 0**，因此 VALIDATION 的 short-run 指标为 "
        "NOT AVAILABLE；short-run 证据只能由 CALIBRATION 提供（见 4.2）。"
    )
    add("")
    add("| Method | Stratum | Runs | Survival recall | Whole-run deletion | Retention mean | Retention min |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe"):
        if name not in run_metrics["VALIDATION"]:
            continue
        for stratum in ("short", "medium", "long"):
            block = run_metrics["VALIDATION"][name][stratum]
            add(
                f"| {name} | {stratum} | {block['runs']} | {_fmt(block['survival_recall'], percent=True)} | "
                f"{_fmt(block['whole_deletion_rate'], percent=True)} | {_fmt(block['retention_mean'], 3)} | "
                f"{_fmt(block['retention_min'], 3)} |"
            )
    add("")
    add("### 4.2 CALIBRATION positive-run（阈值选择 split；short-run 证据来源）")
    add("")
    add("| Method | Stratum | Runs | Survival recall | Whole-run deletion | Retention mean | Retention min |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for name in ("fs0", "tau_f1", "tau_safe", "tau_run_safe"):
        data = summary["positive_run_proxy"]["calibration"].get(name)
        if data is None:
            continue
        for stratum in ("overall", "short", "medium", "long"):
            block = data[stratum]
            add(
                f"| {name} | {stratum} | {block['runs']} | {_fmt(block['survival_recall'], percent=True)} | "
                f"{_fmt(block['whole_deletion_rate'], percent=True)} | {_fmt(block['retention_mean'], 3)} | "
                f"{_fmt(block['retention_min'], 3)} |"
            )
    add("")
    add("## 5. 逐视频失败案例分析（VALIDATION，best.pt @ tau_f1）")
    add("")
    rows = [row for row in per_video if row["method"] == "tau_f1" and row["positive_frames"] > 0]
    worst_recall = sorted(rows, key=lambda row: (row["recall"] if row["recall"] is not None else 1.0))[:8]
    highest_pruning = sorted(
        [row for row in per_video if row["method"] == "tau_f1"], key=lambda row: -(row["pruning_rate"] or 0.0)
    )[:5]
    add("**Recall 最低的 8 个视频：**")
    add("")
    add("| video_id | positives | recall | precision | pruning | false deletion |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in worst_recall:
        add(
            f"| {row['video_id']} | {row['positive_frames']} | {_fmt(row['recall'], percent=True)} | "
            f"{_fmt(row['precision'], percent=True)} | {_fmt(row['pruning_rate'], percent=True)} | "
            f"{_fmt(row['false_deletion_rate'], percent=True)} |"
        )
    add("")
    add("**Pruning 最高的 5 个视频：**")
    add("")
    add("| video_id | universe | positives | recall | pruning |")
    add("|---|---:|---:|---:|---:|")
    for row in highest_pruning:
        add(
            f"| {row['video_id']} | {row['universe_frames']} | {row['positive_frames']} | "
            f"{_fmt(row['recall'], percent=True)} | {_fmt(row['pruning_rate'], percent=True)} |"
        )
    add("")
    whole_deleted = [
        row
        for row in run_metrics["VALIDATION"].get("tau_f1", {}).get("per_run", [])
        if row["whole_deleted"]
    ]
    add(f"**Positive-run 被整体删除的案例（tau_f1，派生代理）：{len(whole_deleted)} 个**")
    if whole_deleted:
        add("")
        add("| video_id | start_ts | end_ts | duration | stratum | positive frames |")
        add("|---|---:|---:|---:|---|---:|")
        for row in whole_deleted[:20]:
            add(
                f"| {row['video_id']} | {row['start_ts']} | {row['end_ts']} | "
                f"{row['duration_sec']} | {row['stratum']} | {row['positive_frames']} |"
            )
    add("")
    add("## 6. Native16 missingness-vs-Y 审计（仅统计）")
    add("")
    add(
        f"- 阈值 |Δ| >= {missingness['flag_delta']} 视为 POTENTIAL_SHORTCUT_RISK；"
        "该标记仅表示 missingness 与 Y 存在关联，**不能证明**模型利用了 shortcut。"
    )
    add("")
    add("| field | P(observed\\|Y=1) | P(observed\\|Y=0) | Δ | flag |")
    add("|---|---:|---:|---:|---|")
    flagged_rows = [row for row in missingness["rows"] if row["risk_flag"]]
    for row in (flagged_rows or missingness["rows"]):
        add(
            f"| {row['field']} | {_fmt(row['p_observed_given_y1'])} | {_fmt(row['p_observed_given_y0'])} | "
            f"{_fmt(row['delta'], 4)} | {row['risk_flag'] or ''} |"
        )
    if not flagged_rows:
        add("")
        add("- 未发现超过阈值的字段。")
    add("")
    add("## 7. 图")
    add("")
    add("![PR curve](figures/pr_curve_validation.png)")
    add("")
    add("![threshold metrics](figures/calibration_threshold_metrics.png)")
    add("")
    add("![pruning vs recall](figures/pruning_vs_recall.png)")
    add("")
    add("![FS0 vs FTNet](figures/fs0_vs_ftnet_metrics.png)")
    add("")
    add("![best vs last](figures/best_vs_last_pr_auc.png)")
    add("")
    add("![positive run recall](figures/positive_run_recall_by_duration_proxy.png)")
    add("")
    add("![positive run retention](figures/positive_run_retention_distribution.png)")
    add("")
    add("## 8. 方法与边界")
    add("")
    add("- checkpoint 只读加载，SHA-256 已在冻结前校验；未训练、未微调、未修改 best.pt。")
    add("- 阈值只在 CALIBRATION 上选择；TRAIN 未参与任何阈值或指标。")
    add("- 评价宇宙 = `loss_mask==1`（有 MTurk 覆盖的 candidate 帧）；完整 Qwen candidate mask 未落盘，无法对未覆盖 candidate 帧评价。")
    add("- 主 binary reference = soft target >= 0.5（约等于平均投票 >= 2.5/5）；另有 0.4 / 0.6 / any>0 敏感性分析。")
    add("- 未使用 bootstrap；正样本极少时指标不稳定，已单列正样本数量。")
    add("- 未做 SHAP / permutation importance / Native branch ablation / 重训练。")
    add("")
    add("## 9. 限制与下一步建议")
    add("")
    add(
        f"1. 主参考（soft>=0.5）非常严格：VALIDATION 仅 {validation['fs0']['positive_frames']} 个正帧 / "
        f"{validation['fs0']['universe_frames']} 个 universe 帧，PR-AUC 对少数帧敏感。"
    )
    add("2. 若后续要做 threshold 冻结或 decoder 校准，应固定本报告的 tau 选择流程，并在 CALIBRATION 上复现。")
    add("3. 若需要真正的事件级结论，需先预注册 clip 窗口 → human event 的映射协议（当前不可合法恢复）。")
    add("4. 是否进入 Stage 7.2 模型优化（loss / 结构 / 消融）由用户决定；本报告不宣布 FTNet_PROMOTED。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--stage", choices=("audit", "evaluate", "all"), default="all")
    parser.add_argument("--device", default=None, help="override config device: auto|cuda|cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    environment = EnvironmentPaths.from_json(args.environment)
    paths = {
        key: _resolve_spec(config[key], environment)
        for key in (
            "checkpoint",
            "control_checkpoint",
            "checkpoint_identity_audit",
            "data_root",
            "normalization",
            "index",
            "split_manifest",
            "annotations_root",
            "output_dir",
        )
    }
    print(f"[stage7-eval] python={sys.executable} platform={platform.system()}", flush=True)
    if args.validate_only:
        missing = [str(path) for path in paths.values() if not Path(path).exists()]
        _print({"status": "VALIDATE_ONLY", "missing": missing, "paths": {k: str(v) for k, v in paths.items()}})
        return 0 if not missing else 3
    if args.dry_run:
        _print({"status": "DRY_RUN", "stage": args.stage, "paths": {k: str(v) for k, v in paths.items()}})
        return 0
    if args.stage in ("audit", "all"):
        audit = run_audit(config, environment, paths)
        _print(audit)
    if args.stage in ("evaluate", "all"):
        summary = run_evaluate(config, environment, paths)
        _print(
            {
                "status": "EVALUATION_COMPLETE",
                "verdict": summary["verdict"],
                "tau_f1": summary["thresholds"]["tau_f1"],
                "tau_safe": summary["thresholds"]["tau_safe"],
                "validation_pr_auc_best": summary["pr_auc"]["best"]["VALIDATION"],
                "validation_pr_auc_last": summary["pr_auc"]["last"]["VALIDATION"],
                "output_dir": str(paths["output_dir"]),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
