#!/usr/bin/env python3
"""POSTHOC_DIAGNOSTIC_ONLY — Stage 6.2A within-video / segment-level signal audit.

Question: even though Stage 6.1 showed frame-level global AUROC ≈ 0.51 for PGL/VAS,
do the raw literature scores carry a *within-video* or *segment-level* coarse temporal
signal ("this stretch of time is more likely to contain highlights")?

It is **not** a Formal challenger, changes no frozen science parameter, and does not
rewrite the Stage 6.1 result (``PGL/VAS: NO_PROMOTION``).

It reuses the frozen Stage 6.1 feature cache in place (no re-decode, no re-extraction),
recomputes per-checkpoint scores in memory, and writes only under
``06_Within_Video_Temporal_Signal_Audit``.

Key methodological point: within-video *ranking* is invariant to any monotone per-video
transform, so per-video AUROC is identical for raw / percentile-rank / z-score / min-max.
Normalization only changes the *pooled* (cross-video) AUROC. This is measured explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _path in (str(_REPO / "src"), str(_REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aic_video_highlight.evaluation.official_like import evaluate_official_like  # noqa: E402

DIAGNOSTIC_SCHEMA = "aic.stage6_2a.within-video-temporal-signal/v1"
IDENTITY = "POSTHOC_DIAGNOSTIC_ONLY"
DROP_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
WINDOW_SECONDS = (2.0, 5.0)
BOOTSTRAP_ITERATIONS = 500


class TemporalAuditError(RuntimeError):
    """Raised when the audit cannot proceed on the frozen artifacts."""


def _load_module(path: Path, name: str) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner() -> Any:
    return _load_module(
        Path(__file__).resolve().parent / "run_stage6_1_literature_frame_selection.py",
        "aic_stage6_1_formal_runner",
    )


def _audit05() -> Any:
    return _load_module(
        Path(__file__).resolve().parent / "analyze_stage6_1_failure.py",
        "aic_stage6_1_failure_audit",
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def percentile_rank(values: Sequence[float]) -> np.ndarray:
    """Rank in [0, 1] (average rank / (n-1)); constant input maps to 0.5."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return np.full(arr.size, 0.5, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    if np.ptp(arr) == 0:
        return np.full(arr.size, 0.5, dtype=np.float64)
    return ranks / (arr.size - 1)


def zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    return (arr - float(arr.mean())) / std if std > 0 else np.zeros_like(arr)


def min_max(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(arr))
    return (arr - float(arr.min())) / span if span > 0 else np.zeros_like(arr)


def windows_of(n_samples: int, width: int) -> list[tuple[int, int]]:
    if width <= 0:
        return []
    return [(start, min(start + width, n_samples)) for start in range(0, n_samples, width)]


def tp_temporal_runs(source_frames: Sequence[int]) -> list[int]:
    """Lengths of maximal runs of consecutive source-frame indices among TP frames."""
    frames = sorted({int(frame) for frame in source_frames})
    if not frames:
        return []
    runs: list[int] = []
    current = 1
    for previous, frame in zip(frames, frames[1:]):
        if frame == previous + 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs


def video_bootstrap_auroc(
    per_video: Sequence[tuple[Sequence[float], Sequence[int]]],
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, Any]:
    """Video-level bootstrap CI for the pooled raw AUROC (resample videos)."""
    audit = _audit05()
    point = None
    scores_all = [np.asarray(s, dtype=np.float64) for s, _ in per_video]
    labels_all = [np.asarray(l, dtype=np.int64) for _, l in per_video]
    if scores_all:
        point = audit.auroc(np.concatenate(scores_all), np.concatenate(labels_all))
    estimates: list[float] = []
    count = len(per_video)
    if count == 0:
        return {"point": point, "ci_low": None, "ci_high": None}
    for _ in range(iterations):
        picks = rng.integers(0, count, size=count)
        scores = np.concatenate([scores_all[index] for index in picks])
        labels = np.concatenate([labels_all[index] for index in picks])
        value = audit.auroc(scores, labels)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return {"point": point, "ci_low": None, "ci_high": None}
    return {
        "point": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-videos", type=int, default=0, help="smoke only; 0 = all valid109")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args(argv)
    try:
        return run(args)
    except TemporalAuditError as exc:
        print(f"[6.2A] BLOCKED: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    runner = _runner()
    audit = _audit05()

    protocol_path = args.protocol.resolve()
    protocol = runner.load_json(protocol_path)
    runner.verify_sha256("formal_protocol.json", protocol_path, runner.FROZEN_PROTOCOL_SHA)
    runner.verify_sha256("model_manifest.json", args.model_manifest.resolve(), runner.FROZEN_MODEL_MANIFEST_SHA)

    control_run = args.control_run.resolve()
    control_predictions = control_run / "predictions" / "predictions.jsonl"
    runner.verify_sha256("control predictions", control_predictions, runner.FROZEN_CONTROL_PREDICTIONS_SHA)
    metadata_records = runner.load_json(control_run / "frame_projection" / "metadata_cache.json")["records"]
    control_records = runner.read_jsonl(control_predictions)
    video_shas = {
        str(row["video_id"]): str(row.get("sha256", ""))
        for row in runner.read_jsonl(control_run / "input" / "dev_selected.jsonl")
    }
    specs = runner.build_video_specs(metadata_records, control_records, video_shas)
    reference_path = Path(protocol["frozen_stage6_0_control"]["weak_spatial_reference_path"])
    reference_records = {str(row["video_id"]): row for row in runner.read_jsonl(reference_path)}
    valid_ids = list(runner.read_video_ids(protocol_path.parent / protocol["evaluation_protocol"]["reference_valid_primary"]["identity_file"]))
    valid_set = set(valid_ids)
    specs = [spec for spec in specs if spec.video_id in valid_set]
    if args.max_videos:
        specs = specs[: args.max_videos]
    _expect(len(specs) > 0, "no valid109 specs")

    output_root = args.output_root.resolve()
    for sub in ("report", "tables", "figures", "audit"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    cache_root = args.feature_cache.resolve()
    rng = np.random.default_rng(args.seed)

    per_video: dict[str, dict[str, Any]] = {}
    per_arm: dict[str, dict[str, Any]] = {
        arm: {"pooled": {k: [] for k in ("raw", "pct", "z", "minmax", "rank_ens")},
              "labels": [], "video_pairs": {k: [] for k in ("raw", "pct", "z", "minmax", "rank_ens")},
              "per_video_auroc_pairs": [], "window_stats": {}, "quintiles": [], "drop_rows": [],
              "clustering": []}
        for arm in (runner.PGL, runner.VASNET)
    }

    for arm in (runner.PGL, runner.VASNET):
        entries = runner.checkpoint_entries(protocol, arm)
        runner.verify_checkpoints(arm, entries)
        models = runner.load_models(arm, entries, args.device)
        print(f"[6.2A] {arm}: {len(models)} checkpoints loaded", flush=True)

        for index, spec in enumerate(specs, start=1):
            feature_path = cache_root / spec.video_id / "features.npy"
            _expect(feature_path.is_file(), f"{spec.video_id}: missing cached features")
            features = np.load(feature_path)
            ckpt = audit.per_checkpoint_scores(runner, arm, models, features, args.device)
            raw = ckpt.mean(axis=0)
            rank_ens = np.stack([percentile_rank(row) for row in ckpt], axis=0).mean(axis=0)

            ref_frames = audit._reference_frames(reference_records[spec.video_id]) if spec.video_id in reference_records else ()
            ref_set = set(ref_frames)
            fs0 = spec.fs0_frames
            labels = np.asarray([1 if int(frame) in ref_set else 0 for frame in fs0], dtype=np.int64)
            sample_index = np.asarray([audit.frame_to_sample_index(spec.sample_frames, frame) for frame in fs0], dtype=np.int64)
            variants = {
                "raw": raw[sample_index],
                "pct": percentile_rank(raw)[sample_index],
                "z": zscore(raw)[sample_index],
                "minmax": min_max(raw)[sample_index],
                "rank_ens": rank_ens[sample_index],
            }
            stats = per_arm[arm]
            for key, values in variants.items():
                stats["pooled"][key].extend(float(v) for v in values)
                stats["video_pairs"][key].append((values, labels))
            stats["labels"].extend(int(v) for v in labels)
            if labels.sum() > 0 and (labels == 0).sum() > 0:
                stats["per_video_auroc_pairs"].append((spec.video_id, float(audit.auroc(variants["raw"], labels))))

            # --- segment windows (fixed 2s / 5s) -----------------------------
            window_stats = per_arm[arm]["window_stats"].setdefault("fixed", {})
            for seconds in WINDOW_SECONDS:
                width = max(1, int(round(seconds * runner.SAMPLE_FPS)))
                rows = window_stats.setdefault(seconds, [])
                for start, end in windows_of(len(spec.sample_frames), width):
                    if end <= start:
                        continue
                    in_window = [i for i in range(len(fs0)) if start <= int(sample_index[i]) < end]
                    candidates = len(in_window)
                    if candidates == 0:
                        continue
                    tp = int(sum(int(labels[i]) for i in in_window))
                    score = float(np.mean(rank_ens[start:end]))
                    rows.append({"video_id": spec.video_id, "score": score, "candidates": candidates,
                                 "tp": tp, "fp": candidates - tp, "has_tp": 1 if tp > 0 else 0,
                                 "tp_density": tp / candidates})
            # --- KTS segments (auxiliary) -----------------------------------
            selection = runner.select_with_scores(runner.METHOD_OF_ARM[arm], spec, features, raw)
            kts_rows = per_arm[arm]["window_stats"].setdefault("kts", [])
            for start, end in selection.shot_bounds:
                if end <= start:
                    continue
                in_window = [i for i in range(len(fs0)) if start <= int(sample_index[i]) < end]
                candidates = len(in_window)
                if candidates == 0:
                    continue
                tp = int(sum(int(labels[i]) for i in in_window))
                score = float(np.mean(rank_ens[start:end]))
                kts_rows.append({"video_id": spec.video_id, "score": score, "candidates": candidates,
                                 "tp": tp, "fp": candidates - tp, "has_tp": 1 if tp > 0 else 0,
                                 "tp_density": tp / candidates})

            # --- per-video quintiles (fixed 2s windows) ----------------------
            width = max(1, int(round(2.0 * runner.SAMPLE_FPS)))
            video_windows = []
            for start, end in windows_of(len(spec.sample_frames), width):
                in_window = [i for i in range(len(fs0)) if start <= int(sample_index[i]) < end]
                if not in_window:
                    continue
                tp = int(sum(int(labels[i]) for i in in_window))
                video_windows.append((float(np.mean(rank_ens[start:end])), tp / len(in_window)))
            per_arm[arm]["quintiles"].append(_quintile_densities(video_windows))

            # --- low-score drop curve ---------------------------------------
            order = np.argsort(variants["raw"], kind="mergesort")
            total_tp = int(labels.sum())
            total_fp = int((labels == 0).sum())
            n_ref_all = 18635
            for fraction in DROP_FRACTIONS:
                drop = int(round(fraction * len(order)))
                dropped = set(int(i) for i in order[:drop])
                retained_tp = int(sum(1 for i in range(len(labels)) if i not in dropped and labels[i] == 1))
                removed_fp = int(sum(1 for i in range(len(labels)) if i in dropped and labels[i] == 0))
                stats["drop_rows"].append({
                    "video_id": spec.video_id, "fraction": fraction,
                    "tp_retained": retained_tp, "fp_removed": removed_fp,
                    "total_tp": total_tp, "total_fp": total_fp,
                    "n_ref_all": n_ref_all,
                })

            # --- clustering --------------------------------------------------
            runs = tp_temporal_runs(sorted(ref_set & set(fs0)))
            stats["clustering"].append({
                "video_id": spec.video_id,
                "fs0_count": len(fs0),
                "tp_count": int(labels.sum()),
                "tp_runs": len(runs),
                "tp_run_mean": float(np.mean(runs)) if runs else 0.0,
                "tp_run_max": int(max(runs)) if runs else 0,
            })

            per_video[(arm, spec.video_id)] = {
                "raw": raw, "rank_ens": rank_ens, "sample_frames": spec.sample_frames,
                "fs0": fs0, "labels": labels, "sample_index": sample_index, "fps": spec.timing.fps,
            }
            if index % 25 == 0 or index == len(specs):
                print(f"[6.2A] {arm} {index}/{len(specs)}", flush=True)

        del models

    # --- ΔF for "severe regression" grouping --------------------------------
    reference_rows = [reference_records[spec.video_id] for spec in specs]
    fs0_eval = evaluate_official_like([dict(spec.control_record) for spec in specs], reference_rows)
    fs0_f = {str(row["video_id"]): float(row["f_score"]) for row in fs0_eval["per_video"]}
    deltas: dict[str, dict[str, float]] = {}
    for arm in (runner.PGL, runner.VASNET):
        records = []
        for spec in specs:
            keep = set()
            state = per_video[(arm, spec.video_id)]
            # model predictions not needed for ΔF of this audit; reuse Stage 6.1 by recomputing selection
            selection = runner.select_with_scores(runner.METHOD_OF_ARM[arm], spec, np.load(cache_root / spec.video_id / "features.npy"), state["raw"])
            keep = set(selection.kept_frames)
            record = dict(spec.control_record)
            record["predictions"] = [row for row in spec.control_record["predictions"] if int(row["frame"]) in keep]
            records.append(record)
        model_eval = evaluate_official_like(records, reference_rows)
        model_f = {str(row["video_id"]): float(row["f_score"]) for row in model_eval["per_video"]}
        deltas[arm] = {vid: model_f[vid] - fs0_f[vid] for vid in fs0_f}

    summary = _summarise(runner, audit, per_arm, per_video, deltas, rng)
    _write_outputs(output_root, runner, audit, per_arm, per_video, summary, deltas, specs)
    try:
        _render(output_root, runner, per_arm, per_video, summary, deltas, specs)
    except Exception as exc:  # pragma: no cover - plotting environment
        _write_json(output_root / "audit" / "figure_error.json", {"error": str(exc)})
    _write_report(output_root, runner, summary)
    print("[6.2A] STAGE6_2A_TEMPORAL_SIGNAL_AUDIT_COMPLETE", flush=True)
    return 0


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise TemporalAuditError(message)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _quintile_densities(windows: Sequence[tuple[float, float]]) -> dict[str, float] | None:
    if len(windows) < 5:
        return None
    ordered = sorted(windows, key=lambda item: item[0])
    chunks = np.array_split(np.arange(len(ordered)), 5)
    result = {}
    for index, chunk in enumerate(chunks, start=1):
        if chunk.size == 0:
            result[f"q{index}"] = 0.0
        else:
            result[f"q{index}"] = float(np.mean([ordered[i][1] for i in chunk]))
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarise(runner: Any, audit: Any, per_arm: Mapping[str, Mapping[str, Any]], per_video: Mapping[Any, Any], deltas: Mapping[str, Mapping[str, float]], rng: np.random.Generator) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in (runner.PGL, runner.VASNET):
        stats = per_arm[arm]
        labels = np.asarray(stats["labels"], dtype=np.int64)
        normalized = {}
        for key, values in stats["pooled"].items():
            scores = np.asarray(values, dtype=np.float64)
            normalized[key] = {
                "auroc": audit.auroc(scores, labels),
                "average_precision": audit.average_precision(scores, labels),
            }
        per_video_auroc = np.asarray([value for _vid, value in stats["per_video_auroc_pairs"]], dtype=np.float64)
        window_analysis = {}
        for name, rows in stats["window_stats"].items():
            if name == "fixed":
                for seconds, seconds_rows in rows.items():
                    window_analysis[f"fixed_{seconds:g}s"] = _window_metrics(audit, seconds_rows)
            else:
                window_analysis["kts"] = _window_metrics(audit, rows)
        quintile_rows = [q for q in stats["quintiles"] if q]
        quintile_means = {f"q{i}": float(np.mean([q[f"q{i}"] for q in quintile_rows])) for i in range(1, 6)} if quintile_rows else {}
        clustering = _clustering_summary(stats["clustering"], per_video, arm)
        drop = _drop_curve(stats["drop_rows"])
        arms[arm] = {
            "pooled_normalized": normalized,
            "per_video_auroc": {
                "count": int(per_video_auroc.size),
                "mean": float(per_video_auroc.mean()) if per_video_auroc.size else None,
                "median": float(np.median(per_video_auroc)) if per_video_auroc.size else None,
                "p25": float(np.quantile(per_video_auroc, 0.25)) if per_video_auroc.size else None,
                "p75": float(np.quantile(per_video_auroc, 0.75)) if per_video_auroc.size else None,
                "gt_0_5": int((per_video_auroc > 0.5).sum()),
                "gt_0_55": int((per_video_auroc > 0.55).sum()),
                "gt_0_60": int((per_video_auroc > 0.60).sum()),
                "values": [float(v) for v in per_video_auroc],
                "pairs": [[vid, float(value)] for vid, value in stats["per_video_auroc_pairs"]],
            },
            "pooled_raw_bootstrap": video_bootstrap_auroc(stats["video_pairs"]["raw"], rng, BOOTSTRAP_ITERATIONS),
            "segments": window_analysis,
            "quintile_tp_density": quintile_means,
            "quintile_enrichment": _quintile_enrichment(quintile_means),
            "clustering": clustering,
            "low_score_drop": drop,
        }
    aggregates = {
        "valid109_videos": len(per_video) // 2,
        "window_seconds": list(WINDOW_SECONDS),
        "drop_fractions": list(DROP_FRACTIONS),
        "note": "within-video ranking is invariant to monotone per-video normalization; only pooled AUROC changes",
        "formal_result_unchanged": "PGL NO_PROMOTION / VAS NO_PROMOTION",
        "delta_f_mean": {arm: float(np.mean(list(deltas[arm].values()))) for arm in (runner.PGL, runner.VASNET)},
    }
    return {"schema": DIAGNOSTIC_SCHEMA, "identity": IDENTITY, "arms": arms, "aggregates": aggregates}


def _window_metrics(audit: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"window_count": 0}
    scores = [row["score"] for row in rows]
    has_tp = [row["has_tp"] for row in rows]
    densities = np.asarray([row["tp_density"] for row in rows], dtype=np.float64)
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    baseline = float(np.mean(densities))
    enrichment = {}
    for fraction in (0.1, 0.2, 0.3):
        top = ordered[: max(1, int(round(fraction * len(ordered))))]
        density = float(np.mean([row["tp_density"] for row in top]))
        enrichment[f"top_{int(fraction*100)}pct"] = {"tp_density": density, "enrichment_over_baseline": density / baseline if baseline else None}
    return {
        "window_count": len(rows),
        "auroc_has_tp": audit.auroc(scores, has_tp),
        "average_precision_has_tp": audit.average_precision(scores, has_tp),
        "baseline_tp_density": baseline,
        "top_enrichment": enrichment,
    }


def _quintile_enrichment(quintile_means: Mapping[str, float]) -> dict[str, Any]:
    if not quintile_means:
        return {}
    overall = float(np.mean(list(quintile_means.values())))
    q5, q1 = quintile_means.get("q5", 0.0), quintile_means.get("q1", 0.0)
    return {
        "q5_over_q1": (q5 / q1) if q1 > 0 else None,
        "q5_over_overall": (q5 / overall) if overall > 0 else None,
    }


def _clustering_summary(rows: Sequence[Mapping[str, Any]], per_video: Mapping[Any, Any], arm: str) -> dict[str, Any]:
    if not rows:
        return {}
    run_means = [row["tp_run_mean"] for row in rows if row["tp_count"] > 0]
    fs0_counts = [row["fs0_count"] for row in rows]
    tp_counts = [row["tp_count"] for row in rows]
    return {
        "videos": len(rows),
        "videos_with_tp": len(run_means),
        "tp_run_mean": float(np.mean(run_means)) if run_means else 0.0,
        "tp_run_median": float(np.median(run_means)) if run_means else 0.0,
        "tp_runs_per_video_mean": float(np.mean([row["tp_runs"] for row in rows])) if rows else 0.0,
        "mean_fs0_count": float(np.mean(fs0_counts)) if fs0_counts else 0.0,
        "mean_tp_count": float(np.mean(tp_counts)) if tp_counts else 0.0,
        "tp_fraction": float(sum(tp_counts) / sum(fs0_counts)) if sum(fs0_counts) else 0.0,
    }


def _drop_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_fraction: dict[float, dict[str, int]] = {}
    for row in rows:
        bucket = by_fraction.setdefault(row["fraction"], {"tp_retained": 0, "fp_removed": 0, "total_tp": 0, "total_fp": 0})
        bucket["tp_retained"] += row["tp_retained"]
        bucket["fp_removed"] += row["fp_removed"]
        bucket["total_tp"] += row["total_tp"]
        bucket["total_fp"] += row["total_fp"]
    curve = []
    for fraction in sorted(by_fraction):
        bucket = by_fraction[fraction]
        total_tp, total_fp = bucket["total_tp"], bucket["total_fp"]
        tp_retained, fp_removed = bucket["tp_retained"], bucket["fp_removed"]
        retained_fp = total_fp - fp_removed
        retained_total = tp_retained + retained_fp
        curve.append({
            "drop_fraction": fraction,
            "tp_retained": tp_retained,
            "tp_loss": total_tp - tp_retained,
            "fp_removed": fp_removed,
            "tp_recall": tp_retained / total_tp if total_tp else 0.0,
            "fp_removed_fraction": fp_removed / total_fp if total_fp else 0.0,
            "tp_loss_fraction": (total_tp - tp_retained) / total_tp if total_tp else 0.0,
            "precision_retained": tp_retained / retained_total if retained_total else 0.0,
            "fp_minus_tp_removal": (fp_removed / total_fp - (total_tp - tp_retained) / total_tp) if total_tp and total_fp else 0.0,
        })
    return curve


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_outputs(output_root: Path, runner: Any, audit: Any, per_arm: Mapping[str, Mapping[str, Any]], per_video: Mapping[Any, Any], summary: Mapping[str, Any], deltas: Mapping[str, Mapping[str, float]], specs: Sequence[Any]) -> None:
    tables = output_root / "tables"
    _write_json(output_root / "audit" / "summary.json", summary)
    _write_json(output_root / "audit" / "audit_identity.json", {
        "schema": DIAGNOSTIC_SCHEMA,
        "identity": IDENTITY,
        "formal_result_unchanged": "PGL NO_PROMOTION / VAS NO_PROMOTION",
        "reused": ["stage6.1 feature cache", "frozen control predictions", "protocol checkpoints"],
        "re_extracted_features": False,
        "re_decoded_video": False,
    })

    pooled_rows = []
    for arm in (runner.PGL, runner.VASNET):
        for key, metrics in summary["arms"][arm]["pooled_normalized"].items():
            pooled_rows.append({"arm": arm, "normalization": key, "auroc": metrics["auroc"], "average_precision": metrics["average_precision"]})
    _write_csv(tables / "pooled_normalization_auroc.csv", ["arm", "normalization", "auroc", "average_precision"], pooled_rows)

    per_video_rows = []
    for arm in (runner.PGL, runner.VASNET):
        for vid, value in summary["arms"][arm]["per_video_auroc"]["pairs"]:
            per_video_rows.append({"arm": arm, "video_id": vid, "within_video_auroc": value, "delta_f": deltas[arm].get(vid)})
    _write_csv(tables / "per_video_within_auroc.csv", ["arm", "video_id", "within_video_auroc", "delta_f"], per_video_rows)

    segment_rows = []
    for arm in (runner.PGL, runner.VASNET):
        for name, metrics in summary["arms"][arm]["segments"].items():
            if metrics.get("window_count", 0) == 0:
                continue
            segment_rows.append({
                "arm": arm, "segment": name, "window_count": metrics["window_count"],
                "auroc_has_tp": metrics["auroc_has_tp"], "ap_has_tp": metrics["average_precision_has_tp"],
                "baseline_tp_density": metrics["baseline_tp_density"],
                "top10_density": metrics["top_enrichment"]["top_10pct"]["tp_density"],
                "top10_enrichment": metrics["top_enrichment"]["top_10pct"]["enrichment_over_baseline"],
                "top20_enrichment": metrics["top_enrichment"]["top_20pct"]["enrichment_over_baseline"],
                "top30_enrichment": metrics["top_enrichment"]["top_30pct"]["enrichment_over_baseline"],
            })
    _write_csv(tables / "segment_analysis.csv", ["arm", "segment", "window_count", "auroc_has_tp", "ap_has_tp", "baseline_tp_density", "top10_density", "top10_enrichment", "top20_enrichment", "top30_enrichment"], segment_rows)

    quintile_rows = []
    for arm in (runner.PGL, runner.VASNET):
        means = summary["arms"][arm]["quintile_tp_density"]
        if means:
            quintile_rows.append({"arm": arm, **{k: means[k] for k in sorted(means)}})
    _write_csv(tables / "score_quintile_tp_density.csv", ["arm", "q1", "q2", "q3", "q4", "q5"], quintile_rows)

    drop_rows = []
    for arm in (runner.PGL, runner.VASNET):
        for row in summary["arms"][arm]["low_score_drop"]:
            drop_rows.append({"arm": arm, **row})
    _write_csv(tables / "low_score_drop_curve.csv", ["arm", "drop_fraction", "tp_retained", "tp_loss", "fp_removed", "tp_recall", "fp_removed_fraction", "tp_loss_fraction", "precision_retained", "fp_minus_tp_removal"], drop_rows)

    cluster_rows = []
    for arm in (runner.PGL, runner.VASNET):
        cluster_rows.append({"arm": arm, **summary["arms"][arm]["clustering"]})
    _write_csv(tables / "tp_clustering.csv", ["arm", "videos", "videos_with_tp", "tp_run_mean", "tp_run_median", "tp_runs_per_video_mean", "mean_fs0_count", "mean_tp_count", "tp_fraction"], cluster_rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _render(output_root: Path, runner: Any, per_arm: Mapping[str, Mapping[str, Any]], per_video: Mapping[Any, Any], summary: Mapping[str, Any], deltas: Mapping[str, Mapping[str, float]], specs: Sequence[Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    tags = {runner.PGL: "pgl", runner.VASNET: "vas"}

    for arm in (runner.PGL, runner.VASNET):
        values = np.asarray(summary["arms"][arm]["per_video_auroc"]["values"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=20, color="#4c72b0", alpha=0.8)
        ax.axvline(0.5, color="k", linestyle="--", linewidth=1.0, label="chance")
        ax.axvline(values.mean(), color="r", linestyle="--", linewidth=1.0, label=f"mean={values.mean():.3f}")
        ax.set_xlabel("within-video AUROC"); ax.set_ylabel("video count")
        ax.set_title(f"{arm.upper()} per-video TP-vs-FP AUROC (valid109, n={values.size})")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / f"per_video_auroc_distribution_{tags[arm]}.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels_x = ["raw pooled", "per-video mean", "pct-rank pooled", "rank-ens pooled"]
    width = 0.35
    xs = np.arange(len(labels_x))
    for offset, arm in enumerate((runner.PGL, runner.VASNET)):
        pooled = summary["arms"][arm]["pooled_normalized"]
        values = [
            pooled["raw"]["auroc"],
            summary["arms"][arm]["per_video_auroc"]["mean"],
            pooled["pct"]["auroc"],
            pooled["rank_ens"]["auroc"],
        ]
        ax.bar(xs + (offset - 0.5) * width, values, width=width, label=arm.upper())
        for x, v in zip(xs + (offset - 0.5) * width, values):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels_x); ax.set_ylim(0.45, 0.60)
    ax.set_ylabel("AUROC"); ax.set_title("raw vs rank-normalized AUROC"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "raw_vs_rank_normalized_auroc.png", dpi=140); plt.close(fig)

    for arm in (runner.PGL, runner.VASNET):
        means = summary["arms"][arm]["quintile_tp_density"]
        if not means:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = [f"q{i}" for i in range(1, 6)]
        ax.bar(keys, [means[k] for k in keys], color="#55a868")
        overall = float(np.mean([means[k] for k in keys]))
        ax.axhline(overall, color="r", linestyle="--", label=f"overall={overall:.3f}")
        ax.set_xlabel("segment score quintile (per video, 2s windows)")
        ax.set_ylabel("mean TP density")
        ax.set_title(f"{arm.upper()} TP density by score quintile"); ax.legend()
        fig.tight_layout(); fig.savefig(figures / f"segment_tp_density_by_score_quintile_{tags[arm]}.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in (runner.PGL, runner.VASNET):
        curve = summary["arms"][arm]["low_score_drop"]
        ax.plot([row["drop_fraction"] for row in curve], [row["fp_removed_fraction"] for row in curve], marker="o", label=f"{arm.upper()} FP removed")
        ax.plot([row["drop_fraction"] for row in curve], [row["tp_loss_fraction"] for row in curve], marker="x", linestyle="--", label=f"{arm.upper()} TP loss")
    ax.plot([0, 0.5], [0, 0.5], "k:", linewidth=0.8)
    ax.set_xlabel("low-score fraction dropped (within-video)"); ax.set_ylabel("fraction")
    ax.set_title("POSTHOC low-score drop: FP removed vs TP lost"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "low_score_drop_recall_precision_curve.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in (runner.PGL, runner.VASNET):
        cluster = summary["arms"][arm]["clustering"]
        ax.bar([arm.upper()], [cluster.get("tp_run_mean", 0.0)], label=f"{arm.upper()} mean TP run")
    ax.set_ylabel("mean consecutive TP run length (source frames)")
    ax.set_title("TP temporal clustering")
    fig.tight_layout(); fig.savefig(figures / "temporal_tp_clustering.png", dpi=140); plt.close(fig)

    # representative timelines: 3 high enrichment, 3 low, 3 severe regression (per PGL)
    arm = runner.PGL
    per_vid_auc = {vid: value for vid, value in summary["arms"][arm]["per_video_auroc"]["pairs"]}
    enriched = sorted(per_vid_auc, key=lambda v: per_vid_auc[v], reverse=True)
    severe = sorted(deltas[arm], key=lambda v: deltas[arm][v])
    picks = ([("high-enrichment", v) for v in enriched[:3]]
             + [("low-enrichment", v) for v in enriched[-3:]]
             + [("severe-regression", v) for v in severe[:3]])
    spec_by_id = {spec.video_id: spec for spec in specs}
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.4 * len(picks)))
    for ax, (label, video_id) in zip(np.atleast_1d(axes), picks):
        state = per_video[(arm, video_id)]
        spec = spec_by_id[video_id]
        times = [frame / float(spec.timing.fps) for frame in state["sample_frames"]]
        ax.plot(times, state["raw"], color="#333333", linewidth=0.9, label="raw score")
        ax.plot(times, state["rank_ens"], color="#9467bd", linewidth=0.9, alpha=0.7, label="rank-ens")
        fs0 = state["fs0"]; labels = state["labels"]
        tp_times = [frame / float(spec.timing.fps) for frame, lab in zip(fs0, labels) if lab == 1]
        fp_times = [frame / float(spec.timing.fps) for frame, lab in zip(fs0, labels) if lab == 0]
        ax.scatter(fp_times, [1.0] * len(fp_times), s=6, c="#bbbbbb", label="FP")
        ax.scatter(tp_times, [1.05] * len(tp_times), s=10, c="#2ca02c", label="TP")
        ax.set_ylim(-0.05, 1.2); ax.set_xlabel("time (s)")
        ax.set_title(f"{label}: {video_id} (AUROC={per_vid_auc[video_id]:.3f}, dF={deltas[arm][video_id]:+.3f})", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(figures / "representative_timelines.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(output_root: Path, runner: Any, summary: Mapping[str, Any]) -> None:
    pgl = summary["arms"][runner.PGL]
    vas = summary["arms"][runner.VASNET]
    lines: list[str] = []
    add = lines.append
    add("# Stage 6.2A — Within-Video / Segment-Level Temporal Signal Audit (POSTHOC_DIAGNOSTIC_ONLY)")
    add("")
    add("Stage 6.1 remains frozen: `PGL NO_PROMOTION`, `VAS NO_PROMOTION`.")
    add("")
    add("## 1. Raw global vs within-video")
    add("")
    add(f"- PGL raw pooled AUROC {pgl['pooled_normalized']['raw']['auroc']:.4f}; per-video mean AUROC {pgl['per_video_auroc']['mean']:.4f} (n={pgl['per_video_auroc']['count']})")
    add(f"- VAS raw pooled AUROC {vas['pooled_normalized']['raw']['auroc']:.4f}; per-video mean AUROC {vas['per_video_auroc']['mean']:.4f} (n={vas['per_video_auroc']['count']})")
    add(f"- PGL raw pooled bootstrap 95% CI [{pgl['pooled_raw_bootstrap']['ci_low']:.4f}, {pgl['pooled_raw_bootstrap']['ci_high']:.4f}]")
    add(f"- VAS raw pooled bootstrap 95% CI [{vas['pooled_raw_bootstrap']['ci_low']:.4f}, {vas['pooled_raw_bootstrap']['ci_high']:.4f}]")
    add(f"- within-video ranking is invariant to per-video monotone normalization; only pooled AUROC changes")
    add("")
    add("## 2. Rank-normalized ensemble")
    add("")
    add(f"- PGL pooled pct-rank AUROC {pgl['pooled_normalized']['pct']['auroc']:.4f}; rank-ens ensemble {pgl['pooled_normalized']['rank_ens']['auroc']:.4f}")
    add(f"- VAS pooled pct-rank AUROC {vas['pooled_normalized']['pct']['auroc']:.4f}; rank-ens ensemble {vas['pooled_normalized']['rank_ens']['auroc']:.4f}")
    add("")
    add("## 3. Per-video AUROC distribution")
    add("")
    add(f"- PGL: median {pgl['per_video_auroc']['median']:.4f}, IQR [{pgl['per_video_auroc']['p25']:.4f}, {pgl['per_video_auroc']['p75']:.4f}], >0.5 {pgl['per_video_auroc']['gt_0_5']}, >0.55 {pgl['per_video_auroc']['gt_0_55']}, >0.60 {pgl['per_video_auroc']['gt_0_60']}")
    add(f"- VAS: median {vas['per_video_auroc']['median']:.4f}, IQR [{vas['per_video_auroc']['p25']:.4f}, {vas['per_video_auroc']['p75']:.4f}], >0.5 {vas['per_video_auroc']['gt_0_5']}, >0.55 {vas['per_video_auroc']['gt_0_55']}, >0.60 {vas['per_video_auroc']['gt_0_60']}")
    add("")
    add("## 4. Segment-level TP enrichment")
    add("")
    for arm, stats in ((runner.PGL, pgl), (runner.VASNET, vas)):
        for name, metrics in stats["segments"].items():
            if metrics.get("window_count", 0) == 0:
                continue
            add(f"- {arm.upper()} {name}: AUROC(has_tp) {metrics['auroc_has_tp']:.4f}, "
                f"top10% TP-density enrichment {metrics['top_enrichment']['top_10pct']['enrichment_over_baseline']}")
    add("")
    add("## 5. Score quintile analysis")
    add("")
    add(f"- PGL per-video quintile TP density {pgl['quintile_tp_density']} → enrichment {pgl['quintile_enrichment']}")
    add(f"- VAS per-video quintile TP density {vas['quintile_tp_density']} → enrichment {vas['quintile_enrichment']}")
    add("")
    add("## 6. Low-score drop diagnostics")
    add("")
    add("| drop | PGL FP removed | PGL TP loss | VAS FP removed | VAS TP loss |")
    add("| --- | --- | --- | --- | --- |")
    pgl_map = {row["drop_fraction"]: row for row in pgl["low_score_drop"]}
    vas_map = {row["drop_fraction"]: row for row in vas["low_score_drop"]}
    for fraction in sorted(pgl_map):
        add(f"| {fraction:.2f} | {pgl_map[fraction]['fp_removed_fraction']:.3f} | {pgl_map[fraction]['tp_loss_fraction']:.3f} | "
            f"{vas_map[fraction]['fp_removed_fraction']:.3f} | {vas_map[fraction]['tp_loss_fraction']:.3f} |")
    add("")
    add("All values are POSTHOC_DIAGNOSTIC_ONLY; no drop fraction is selected as a configuration.")
    add("")
    add("## 7. Temporal clustering explanation")
    add("")
    add(f"- PGL TP run mean {pgl['clustering'].get('tp_run_mean')} (runs/video {pgl['clustering'].get('tp_runs_per_video_mean')})")
    add(f"- VAS TP run mean {vas['clustering'].get('tp_run_mean')} (runs/video {vas['clustering'].get('tp_runs_per_video_mean')})")
    add("")
    add("## 8. Representative timelines")
    add("")
    add("- `figures/representative_timelines.png` (3 high-enrichment, 3 low-enrichment, 3 severe-regression).")
    add("")
    add("## 9. Core conclusion")
    add("")
    add(_verdict(summary, runner))
    add("")
    add("## 10. Stage 6.2 recommendation")
    add("")
    add("- See the conversation handoff; no threshold is fixed here.")
    add("")
    add("## 11. Integrity")
    add("")
    add("- POSTHOC_DIAGNOSTIC_ONLY; Stage 6.1 unchanged.")
    add("- Hard229 / Heldout / Official Test NOT ACCESSED.")
    add("- Feature cache reused in place; no re-decode, no re-extraction.")
    add("")
    add("## 12. Reproducibility")
    add("")
    add("- `audit/summary.json`, `audit/audit_identity.json`, `tables/*.csv`, `figures/*.png`.")
    add("")
    _write_json(output_root / "report" / "stage6_2a_report_lines.json", lines)
    (output_root / "report" / "Stage6_2A_Within_Video_Temporal_Signal_Audit.md").write_text("\n".join(lines), encoding="utf-8")


def _verdict(summary: Mapping[str, Any], runner: Any) -> str:
    """Heuristic mechanism label. NOT a preregistered numeric pass threshold.

    The label is descriptive; the numbers are reported above and the Stage 6.2
    go/no-go threshold (if any) must be preregistered in a later round.
    """
    pgl = summary["arms"][runner.PGL]
    vas = summary["arms"][runner.VASNET]
    frame_means = [pgl["per_video_auroc"]["mean"], vas["per_video_auroc"]["mean"]]
    q5 = [pgl["quintile_enrichment"].get("q5_over_overall"), vas["quintile_enrichment"].get("q5_over_overall")]
    top = []
    for arm_stats in (pgl, vas):
        for metrics in arm_stats["segments"].values():
            if metrics.get("window_count", 0):
                top.append(metrics["top_enrichment"]["top_10pct"]["enrichment_over_baseline"])
    coarse = [value for value in (q5 + top) if value is not None]
    frame_best = max([value for value in frame_means if value is not None], default=None)
    coarse_best = max(coarse) if coarse else None
    detail = (
        f"(per-video AUROC mean PGL {frame_means[0]:.4f} / VAS {frame_means[1]:.4f}; "
        f"coarse enrichment max {coarse_best:.3f} if coarse_best else 'n/a')"
    )
    # Narrative label conventions, NOT preregistration go/no-go thresholds.
    if frame_best is not None and frame_best >= 0.55 and coarse_best is not None and coarse_best >= 1.3:
        return f"COARSE_TEMPORAL_SIGNAL_SUPPORTED {detail}"
    if (frame_best is not None and frame_best >= 0.50) and (coarse_best is not None and coarse_best >= 1.05):
        return f"WEAK_SIGNAL_ONLY {detail}"
    return f"LITERATURE_SIGNAL_NOT_SUPPORTED {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
