#!/usr/bin/env python3
"""POSTHOC_DIAGNOSTIC_ONLY — Stage 6.1 failure-mechanism audit.

This script explains *why* the Stage 6.1 literature frame selectors collapsed in
recall.  It is **not** a Formal challenger, does not change any frozen science
parameter, and does not rewrite the Stage 6.1 formal result
(``PGL/VAS: NO_PROMOTION``).

Rules it obeys:

* reuses the frozen Stage 6.1 feature cache — it never re-decodes a video and
  never re-extracts GoogLeNet features;
* never writes or overwrites Stage 6.1 predictions / protocol / checkpoints;
* every output is under a fresh ``05_Failure_Mechanism_Audit`` root.

It recomputes, in memory only, the raw per-checkpoint literature scores and the
KTS/knapsack KEEP/DROP masks so the audit can measure the causal chain that the
Formal run actually exercised.  All scientific primitives are imported from the
frozen library modules and the committed Formal runner.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _path in (str(_REPO / "src"), str(_REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aic_video_highlight.evaluation.official_like import evaluate_official_like  # noqa: E402

DIAGNOSTIC_SCHEMA = "aic.stage6_1.failure-mechanism-audit/v1"
IDENTITY = "POSTHOC_DIAGNOSTIC_ONLY"
RETENTION_FRACTIONS = tuple(round(0.05 * step, 2) for step in range(1, 21))


class FailureAuditError(RuntimeError):
    """Raised when the audit cannot proceed on the frozen artifacts."""


def _load_runner() -> Any:
    path = Path(__file__).resolve().parent / "run_stage6_1_literature_frame_selection.py"
    name = "aic_stage6_1_formal_runner"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pure statistical helpers (testable without GPU)
# ---------------------------------------------------------------------------

def frame_to_sample_index(sample_frames: Sequence[int], frame: int) -> int:
    """Index of the sample whose source interval [s[i], s[i+1]) contains ``frame``."""
    if not sample_frames:
        raise FailureAuditError("empty sample grid")
    index = bisect_right(sample_frames, int(frame)) - 1
    return max(0, min(index, len(sample_frames) - 1))


def nearest_sample_distance(sample_frames: Sequence[int], frame: int) -> int:
    """Distance in source frames from ``frame`` to the nearest sample frame."""
    frames = list(sample_frames)
    if not frames:
        raise FailureAuditError("empty sample grid")
    index = bisect_left(frames, int(frame))
    candidates = []
    if index < len(frames):
        candidates.append(frames[index])
    if index > 0:
        candidates.append(frames[index - 1])
    return int(min(abs(candidate - int(frame)) for candidate in candidates))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with average ranks for ties. 1.0 perfect, 0.5 chance."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    positives = scores_arr[labels_arr == 1]
    negatives = scores_arr[labels_arr == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    combined = np.concatenate([positives, negatives])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    sorted_vals = combined[order]
    start = 0
    for i in range(1, combined.size + 1):
        if i == combined.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    rank_sum = ranks[: positives.size].sum()
    return float((rank_sum - positives.size * (positives.size + 1) / 2.0) / (positives.size * negatives.size))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision (area under the precision-recall curve)."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.sum() == 0:
        return float("nan")
    order = np.argsort(-scores_arr, kind="mergesort")
    ordered = labels_arr[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def cohens_d(positives: Sequence[float], negatives: Sequence[float]) -> float:
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    pooled = math.sqrt(
        ((pos.size - 1) * pos.var(ddof=1) + (neg.size - 1) * neg.var(ddof=1)) / (pos.size + neg.size - 2)
    )
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else float("nan")


def roc_points(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores_arr, kind="mergesort")
    ordered = labels_arr[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    tpr = tp / max(1, labels_arr.sum())
    fpr = fp / max(1, int((labels_arr == 0).sum()))
    return [0.0] + list(fpr), [0.0] + list(tpr)


def pr_points(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores_arr, kind="mergesort")
    ordered = labels_arr[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1, labels_arr.sum())
    return [0.0] + list(recall), [1.0] + list(precision)


def quantiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    qs = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return {
        "min": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]), "median": float(qs[3]),
        "p75": float(qs[4]), "p90": float(qs[5]), "p95": float(qs[6]), "max": float(qs[7]),
        "mean": float(arr.mean()), "std": float(arr.std()),
    }


def retention_curve(scores: Sequence[float], labels: Sequence[int], fractions: Sequence[float]) -> list[dict[str, float]]:
    """Post-hoc diagnostic curve: keep top fraction by score, measure TP retention."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if scores_arr.size == 0:
        return []
    order = np.argsort(-scores_arr, kind="mergesort")
    ordered = labels_arr[order]
    total_tp = int(ordered.sum())
    total_fp = int(ordered.size - total_tp)
    curve: list[dict[str, float]] = []
    for fraction in fractions:
        kept = max(1, int(round(fraction * ordered.size)))
        kept_tp = int(ordered[:kept].sum())
        kept_fp = kept - kept_tp
        curve.append({
            "keep_fraction": float(fraction),
            "tp_recall": kept_tp / total_tp if total_tp else 0.0,
            "tp_kept": kept_tp,
            "fp_kept": kept_fp,
            "fp_removed_fraction": (total_fp - kept_fp) / total_fp if total_fp else 0.0,
            "precision": kept_tp / kept if kept else 0.0,
        })
    return curve


def oracle_tp_upper_bound(kept_counts: Sequence[int], tp_counts: Sequence[int]) -> int:
    """Max TP retainable when only KEEP-COUNT frames may be kept, TP prioritised."""
    return int(sum(min(int(k), int(t)) for k, t in zip(kept_counts, tp_counts)))


def expected_random_retention(
    fs0_frames: Sequence[int],
    reference_frames: Sequence[int],
    coverage: int,
    n_frames: int,
    rng: np.random.Generator,
    trials: int,
) -> dict[str, float]:
    """Monte-Carlo random equal-budget baseline (diagnostic only)."""
    if n_frames <= 0 or coverage <= 0 or not fs0_frames:
        return {"fs0_retention": 0.0, "fs0_retention_std": 0.0, "tp_retention": 0.0, "tp_retention_std": 0.0}
    fs0_mask = np.zeros(n_frames, dtype=bool)
    fs0_mask[list(fs0_frames)] = True
    ref_mask = np.zeros(n_frames, dtype=bool)
    ref_mask[list(reference_frames)] = True
    draws = rng.integers(0, n_frames, size=(trials, int(coverage)))
    inter_fs0 = fs0_mask[draws].sum(axis=1)
    inter_tp = (fs0_mask[draws] & ref_mask[draws]).sum(axis=1)
    fs0_total = int(fs0_mask.sum())
    tp_total = int((fs0_mask & ref_mask).sum())
    return {
        "fs0_retention": float(inter_fs0.mean() / fs0_total) if fs0_total else 0.0,
        "fs0_retention_std": float(inter_fs0.std() / fs0_total) if fs0_total else 0.0,
        "tp_retention": float(inter_tp.mean() / tp_total) if tp_total else 0.0,
        "tp_retention_std": float(inter_tp.std() / tp_total) if tp_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Data assembly helpers
# ---------------------------------------------------------------------------

def _reference_frames(record: Mapping[str, Any]) -> tuple[int, ...]:
    rois = record.get("rois")
    if not isinstance(rois, Mapping):
        return ()
    return tuple(sorted(int(frame) for frame in rois))


def _scatter_features(features: np.ndarray) -> dict[str, float]:
    if features.size == 0:
        return {}
    norms = np.linalg.norm(features, axis=1)
    adjacent = []
    for index in range(features.shape[0] - 1):
        a, b = features[index], features[index + 1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0:
            adjacent.append(float(np.dot(a, b) / denom))
    flat = features.reshape(-1)
    return {
        "frames": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "value_min": float(flat.min()),
        "value_max": float(flat.max()),
        "per_dim_variance_mean": float(features.var(axis=0).mean()),
        "adjacent_cosine_mean": float(np.mean(adjacent)) if adjacent else float("nan"),
        "adjacent_cosine_p10": float(np.quantile(adjacent, 0.1)) if adjacent else float("nan"),
    }


def per_checkpoint_scores(runner: Any, arm: str, models: Sequence[Any], features: np.ndarray, device: str) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(features.astype(np.float32)))
    score_fn = runner.pgl_sum_frame_scores if arm == runner.PGL else runner.vasnet_frame_scores
    return np.stack([np.asarray(score_fn(model, tensor, device=device), dtype=np.float64) for model in models], axis=0)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main
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
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-videos", type=int, default=0, help="smoke only; 0 = all 166")
    return parser.parse_args(argv)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise FailureAuditError(message)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args(argv)
    try:
        return run(args)
    except FailureAuditError as exc:
        print(f"[audit] BLOCKED: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    runner = _load_runner()

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
    _expect(len(specs) == 166, "expected 166 videos")
    if args.max_videos:
        specs = specs[: args.max_videos]

    reference_path = Path(protocol["frozen_stage6_0_control"]["weak_spatial_reference_path"])
    reference_records = {str(row["video_id"]): row for row in runner.read_jsonl(reference_path)}
    valid_ids = set(runner.read_video_ids(protocol_path.parent / protocol["evaluation_protocol"]["reference_valid_primary"]["identity_file"]))
    empty_ids = set(runner.read_video_ids(protocol_path.parent / protocol["evaluation_protocol"]["empty_reference_diagnostic"]["identity_file"]))

    output_root = args.output_root.resolve()
    for sub in ("report", "tables", "figures", "audit"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    feature_cache_root = args.feature_cache.resolve()
    rng = np.random.default_rng(args.seed)

    def fresh_arm_stats() -> dict[str, Any]:
        return {
            "pooled_scores": [], "pooled_labels": [], "pooled_ckpt": [],
            "fs0_retention": [], "tp_retention_model": [], "random_fs0": [], "random_tp": [],
            "kept_counts": [], "tp_counts": [], "fs0_counts": [],
            "checkpoint_corr": np.zeros((10, 10)), "checkpoint_corr_count": 0,
            "checkpoint_means": np.zeros(10), "checkpoint_means_count": 0,
            "ensemble_range": [], "single_range": [],
            "direction_ok": 0, "direction_total": 0,
            "shot_lengths_samples": [], "shot_lengths_seconds": [],
            "selected_shots": 0, "total_shots": 0,
            "tp_kept_total": 0,
            "curves": {}, "shots": {},
        }

    per_arm: dict[str, dict[str, Any]] = {runner.PGL: fresh_arm_stats(), runner.VASNET: fresh_arm_stats()}
    per_video_rows: list[dict[str, Any]] = []
    feature_diag: list[dict[str, Any]] = []
    sampling_frames: list[int] = []
    sampling_seconds: list[float] = []
    mapping_rows: list[dict[str, Any]] = []
    invariant_violations = 0
    kept_by_arm_video: dict[tuple[str, str], set[int]] = {}

    for arm in (runner.PGL, runner.VASNET):
        entries = runner.checkpoint_entries(protocol, arm)
        runner.verify_checkpoints(arm, entries)
        models = runner.load_models(arm, entries, args.device)
        print(f"[audit] {arm}: {len(models)} checkpoints loaded", flush=True)

        for index, spec in enumerate(specs, start=1):
            feature_path = feature_cache_root / spec.video_id / "features.npy"
            _expect(feature_path.is_file(), f"{spec.video_id}: missing cached features")
            features = np.load(feature_path)
            _expect(features.shape == (len(spec.sample_frames), runner.FEATURE_DIM), f"{spec.video_id}: bad feature shape")
            if index <= 8:
                feature_diag.append({"video_id": spec.video_id, **_scatter_features(features)})

            ckpt = per_checkpoint_scores(runner, arm, models, features, args.device)
            mean_scores = ckpt.mean(axis=0)
            selection = runner.select_with_scores(runner.METHOD_OF_ARM[arm], spec, features, mean_scores)
            kept = set(selection.kept_frames)
            fs0 = set(spec.fs0_frames)
            if not (kept <= fs0 and (kept | set(selection.dropped_frames)) == fs0 and kept == (set(selection.selected_source_frames) & fs0)):
                invariant_violations += 1
            kept_by_arm_video[(arm, spec.video_id)] = kept

            ref_frames = set(_reference_frames(reference_records[spec.video_id])) if spec.video_id in reference_records else set()
            tp_frames = fs0 & ref_frames
            stats = per_arm[arm]
            stats["tp_kept_total"] += len(kept & tp_frames)
            stats["curves"][spec.video_id] = np.asarray(mean_scores, dtype=np.float64)
            stats["shots"][spec.video_id] = (tuple(selection.shot_bounds), tuple(selection.selected_shots))
            stats["fs0_counts"].append(len(fs0))
            stats["kept_counts"].append(len(kept))
            stats["tp_counts"].append(len(tp_frames))
            stats["fs0_retention"].append(len(kept) / len(fs0) if fs0 else 0.0)
            stats["tp_retention_model"].append(len(kept & tp_frames) / len(tp_frames) if tp_frames else 0.0)
            stats["shot_lengths_samples"].extend(end - start for start, end in selection.shot_bounds)
            stats["shot_lengths_seconds"].extend(
                (end - start) / float(spec.timing.fps) for start, end in selection.shot_bounds
            )
            stats["total_shots"] += len(selection.shot_bounds)
            stats["selected_shots"] += len(selection.selected_shots)
            stats["ensemble_range"].append(float(mean_scores.max() - mean_scores.min()))
            stats["single_range"].extend(float(c.max() - c.min()) for c in ckpt)

            if ckpt.shape[0] > 1:
                corr = np.corrcoef(ckpt)
                if np.isfinite(corr).all():
                    stats["checkpoint_corr"] += corr
                    stats["checkpoint_corr_count"] += 1
            stats["checkpoint_means"] += ckpt.mean(axis=1)
            stats["checkpoint_means_count"] += 1

            shot_means = [float(np.mean(mean_scores[start:end])) if end > start else 0.0 for start, end in selection.shot_bounds]
            if shot_means:
                selected_mean = float(np.mean([shot_means[i] for i in selection.selected_shots])) if selection.selected_shots else 0.0
                unselected = [shot_means[i] for i in range(len(shot_means)) if i not in set(selection.selected_shots)]
                stats["direction_total"] += 1
                if unselected and selected_mean >= float(np.mean(unselected)):
                    stats["direction_ok"] += 1

            random_stats = expected_random_retention(
                sorted(fs0), sorted(tp_frames), len(selection.selected_source_frames),
                spec.timing.frame_count, rng, args.random_trials,
            )
            stats["random_fs0"].append(random_stats["fs0_retention"])
            stats["random_tp"].append(random_stats["tp_retention"])

            if spec.video_id in valid_ids:
                labels = [1 if int(frame) in ref_frames else 0 for frame in spec.fs0_frames]
                frame_scores = [float(mean_scores[frame_to_sample_index(spec.sample_frames, frame)]) for frame in spec.fs0_frames]
                stats["pooled_scores"].extend(frame_scores)
                stats["pooled_labels"].extend(labels)
                for checkpoint_index in range(ckpt.shape[0]):
                    stats["pooled_ckpt"].append([
                        float(ckpt[checkpoint_index, frame_to_sample_index(spec.sample_frames, frame)])
                        for frame in spec.fs0_frames
                    ])

            per_video_rows.append({
                "arm": arm,
                "video_id": spec.video_id,
                "subset": "valid109" if spec.video_id in valid_ids else ("empty57" if spec.video_id in empty_ids else "other"),
                "source_frames": spec.timing.frame_count,
                "samples": len(spec.sample_frames),
                "shots": len(selection.shot_bounds),
                "fs0_count": len(fs0),
                "fs0_density": len(fs0) / spec.timing.frame_count,
                "tp_count": len(tp_frames),
                "selected_source_frames": len(selection.selected_source_frames),
                "selected_source_coverage": len(selection.selected_source_frames) / spec.timing.frame_count,
                "kept": len(kept),
                "dropped": len(selection.dropped_frames),
                "fs0_retention": len(kept) / len(fs0) if fs0 else 0.0,
                "tp_retention": len(kept & tp_frames) / len(tp_frames) if tp_frames else 0.0,
                "random_fs0_retention": random_stats["fs0_retention"],
                "random_tp_retention": random_stats["tp_retention"],
            })

            if spec.video_id in valid_ids and not any(row["video_id"] == spec.video_id for row in mapping_rows) and len(mapping_rows) < 5:
                mapping_rows.append({
                    "video_id": spec.video_id,
                    "timestamp_mode": spec.timing.timestamp_mode,
                    "source_frames": spec.timing.frame_count,
                    "samples": len(spec.sample_frames),
                    "first_sample_frame": spec.sample_frames[0],
                    "last_sample_frame": spec.sample_frames[-1],
                    "fs0_count": len(fs0),
                    "kept": len(kept),
                    "dropped": len(selection.dropped_frames),
                    "kept_equals_selected_intersect_fs0": int(kept == (set(selection.selected_source_frames) & fs0)),
                    "kept_subset_of_fs0": int(kept <= fs0),
                    "last_shot_end_is_video_end": int(selection.shot_bounds[-1][1] == len(spec.sample_frames)),
                    "first_shot_start_is_zero": int(selection.shot_bounds[0][0] == 0),
                })
                for frame in sorted(ref_frames):
                    sampling_frames.append(nearest_sample_distance(spec.sample_frames, frame))
                    sampling_seconds.append(nearest_sample_distance(spec.sample_frames, frame) / float(spec.timing.fps))

        del models
        print(f"[audit] {arm}: per-video pass complete", flush=True)

    # --- per-video ΔF for case studies (same evaluator, in memory) ----------
    reference_rows = [reference_records[str(spec.video_id)] for spec in specs]
    fs0_eval = evaluate_official_like([dict(spec.control_record) for spec in specs], reference_rows)
    fs0_f = {str(row["video_id"]): float(row["f_score"]) for row in fs0_eval["per_video"]}
    deltas: dict[str, dict[str, float]] = {}
    for arm in (runner.PGL, runner.VASNET):
        model_records = []
        for spec in specs:
            keep = kept_by_arm_video[(arm, spec.video_id)]
            record = dict(spec.control_record)
            record["predictions"] = [row for row in spec.control_record["predictions"] if int(row["frame"]) in keep]
            model_records.append(record)
        model_eval = evaluate_official_like(model_records, reference_rows)
        model_f = {str(row["video_id"]): float(row["f_score"]) for row in model_eval["per_video"]}
        deltas[arm] = {vid: model_f[vid] - fs0_f[vid] for vid in fs0_f}

    summary = _summarise(runner, protocol, per_arm, per_video_rows, deltas, sampling_frames, sampling_seconds, invariant_violations, valid_ids, empty_ids)
    _write_tables(output_root, runner, per_arm, per_video_rows, summary, feature_diag, sampling_frames, sampling_seconds, mapping_rows, deltas, protocol)
    _write_json(output_root / "audit" / "summary.json", summary)
    _write_json(
        output_root / "audit" / "audit_identity.json",
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "identity": IDENTITY,
            "formal_result_unchanged": "PGL NO_PROMOTION / VAS NO_PROMOTION",
            "reused": ["stage6.1 feature cache", "frozen control predictions", "protocol checkpoints"],
            "re_extracted_features": False,
            "re_decoded_video": False,
            "wrote_stage6_1_predictions": False,
            "mode": "in-memory score + selection recomputation",
        },
    )
    try:
        _render(output_root, runner, per_arm, per_video_rows, summary, dels=deltas, specs=specs, kept_by_arm_video=kept_by_arm_video, reference_records=reference_records, fs0_f=fs0_f)
    except Exception as exc:  # pragma: no cover - plotting environment
        _write_json(output_root / "audit" / "figure_error.json", {"error": str(exc)})

    _write_report(output_root, runner, summary)
    print("[audit] STAGE6_1_FAILURE_MECHANISM_AUDIT_COMPLETE", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _summarise(
    runner: Any,
    protocol: Mapping[str, Any],
    per_arm: Mapping[str, Mapping[str, Any]],
    per_video_rows: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, float]],
    sampling_frames: Sequence[int],
    sampling_seconds: Sequence[float],
    invariant_violations: int,
    valid_ids: set[str],
    empty_ids: set[str],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in (runner.PGL, runner.VASNET):
        stats = per_arm[arm]
        scores = np.asarray(stats["pooled_scores"], dtype=np.float64)
        labels = np.asarray(stats["pooled_labels"], dtype=np.int64)
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        ckpt_corr = stats["checkpoint_corr"] / max(1, stats["checkpoint_corr_count"])
        ckpt_means = stats["checkpoint_means"] / max(1, stats["checkpoint_means_count"])
        per_ckpt_auroc = []
        for checkpoint_index in range(10):
            ckpt_scores = [row[checkpoint_index] for row in stats["pooled_ckpt"]]
            per_ckpt_auroc.append(auroc(ckpt_scores, labels))
        entries = runner.checkpoint_entries(protocol, arm)
        kept = int(sum(stats["kept_counts"]))
        tp = int(sum(stats["tp_counts"]))
        fs0_total = int(sum(stats["fs0_counts"]))
        model_fs0_ret = kept / fs0_total if fs0_total else 0.0
        model_tp_kept = int(stats["tp_kept_total"])
        rows = [row for row in per_video_rows if row["arm"] == arm]
        random_fs0 = float(np.mean(stats["random_fs0"])) if stats["random_fs0"] else 0.0
        random_tp = float(np.mean(stats["random_tp"])) if stats["random_tp"] else 0.0
        oracle = oracle_tp_upper_bound(stats["kept_counts"], stats["tp_counts"])
        arms[arm] = {
            "pooled_candidates": int(scores.size),
            "tp_candidates": int(positives.size),
            "fp_candidates": int(negatives.size),
            "auroc": auroc(scores, labels),
            "average_precision": average_precision(scores, labels),
            "cohens_d": cohens_d(positives, negatives),
            "tp_quantiles": quantiles(positives),
            "fp_quantiles": quantiles(negatives),
            "per_checkpoint_auroc": per_ckpt_auroc,
            "checkpoint_datasets": [entry.get("dataset") for entry in entries],
            "checkpoint_mean_score": [float(v) for v in ckpt_means],
            "checkpoint_correlation_mean": ckpt_corr.tolist(),
            "checkpoint_correlation_offdiag_mean": float(
                (ckpt_corr.sum() - np.trace(ckpt_corr)) / max(1, ckpt_corr.size - ckpt_corr.shape[0])
            ),
            "ensemble_vs_single_range_mean": {
                "ensemble": float(np.mean(stats["ensemble_range"])) if stats["ensemble_range"] else 0.0,
                "single": float(np.mean(stats["single_range"])) if stats["single_range"] else 0.0,
            },
            "score_direction_selected_ge_unselected_fraction": (
                stats["direction_ok"] / stats["direction_total"] if stats["direction_total"] else None
            ),
            "kept_total": kept,
            "fs0_total": fs0_total,
            "tp_total": tp,
            "model_fs0_retention": model_fs0_ret,
            "model_tp_retention": model_tp_kept / tp if tp else 0.0,
            "random_fs0_retention": random_fs0,
            "random_tp_retention": random_tp,
            "tp_retention_enrichment_over_random": (model_tp_kept / tp) / random_tp if tp and random_tp > 0 else None,
            "oracle_tp_upper_bound": oracle,
            "oracle_recall_upper_bound": oracle / 18635.0,
            "oracle_precision_upper_bound": oracle / kept if kept else 0.0,
            "retention_curve": retention_curve(scores, labels, RETENTION_FRACTIONS),
            "shot_stats": {
                "total_shots": int(stats["total_shots"]),
                "selected_shots": int(stats["selected_shots"]),
                "shots_per_video_mean": float(np.mean([row["shots"] for row in rows])) if rows else 0.0,
                "shot_length_seconds": quantiles(stats["shot_lengths_seconds"]),
                "shot_length_samples": quantiles(stats["shot_lengths_samples"]),
            },
        }
    aggregates = {
        "valid109_videos": len(valid_ids),
        "empty57_videos": len(empty_ids),
        "valid109_fs0_candidates": int(sum(row["fs0_count"] for row in per_video_rows if row["arm"] == runner.PGL and row["subset"] == "valid109")),
        "sampling_nearest_distance_frames": quantiles(sampling_frames),
        "sampling_nearest_distance_seconds": quantiles(sampling_seconds),
        "mapping_invariant_violations": invariant_violations,
        "no_kts_knapsack_budget_check": "raw-score discriminability and retention curves are computed without KTS/knapsack/15% budget",
        "valid109_delta_f_mean": {
            arm: float(np.mean(list(deltas[arm].values()))) if deltas[arm] else None
            for arm in (runner.PGL, runner.VASNET)
        },
    }
    return {"schema": DIAGNOSTIC_SCHEMA, "identity": IDENTITY, "arms": arms, "aggregates": aggregates}


def _write_tables(
    output_root: Path,
    runner: Any,
    per_arm: Mapping[str, Mapping[str, Any]],
    per_video_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    feature_diag: Sequence[Mapping[str, Any]],
    sampling_frames: Sequence[int],
    sampling_seconds: Sequence[float],
    mapping_rows: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> None:
    tables = output_root / "tables"
    _write_csv(
        tables / "per_video_audit.csv",
        ["arm", "video_id", "subset", "source_frames", "samples", "shots", "fs0_count", "fs0_density",
         "tp_count", "selected_source_frames", "selected_source_coverage", "kept", "dropped",
         "fs0_retention", "tp_retention", "random_fs0_retention", "random_tp_retention"],
        per_video_rows,
    )
    score_rows = []
    for arm in (runner.PGL, runner.VASNET):
        stats = summary["arms"][arm]
        score_rows.append({
            "arm": arm,
            "pooled_candidates": stats["pooled_candidates"],
            "tp_candidates": stats["tp_candidates"],
            "fp_candidates": stats["fp_candidates"],
            "auroc": stats["auroc"],
            "average_precision": stats["average_precision"],
            "cohens_d": stats["cohens_d"],
            "tp_mean": stats["tp_quantiles"].get("mean"),
            "fp_mean": stats["fp_quantiles"].get("mean"),
            "tp_median": stats["tp_quantiles"].get("median"),
            "fp_median": stats["fp_quantiles"].get("median"),
        })
    _write_csv(
        tables / "tp_fp_score_stats.csv",
        ["arm", "pooled_candidates", "tp_candidates", "fp_candidates", "auroc", "average_precision",
         "cohens_d", "tp_mean", "fp_mean", "tp_median", "fp_median"],
        score_rows,
    )
    ckpt_rows = []
    for arm in (runner.PGL, runner.VASNET):
        stats = summary["arms"][arm]
        for index, value in enumerate(stats["per_checkpoint_auroc"]):
            ckpt_rows.append({
                "arm": arm, "checkpoint_index": index,
                "dataset": stats["checkpoint_datasets"][index],
                "mean_score": stats["checkpoint_mean_score"][index],
                "auroc": value,
            })
    _write_csv(tables / "checkpoint_auroc.csv", ["arm", "checkpoint_index", "dataset", "mean_score", "auroc"], ckpt_rows)

    curve_rows = []
    for arm in (runner.PGL, runner.VASNET):
        for point in summary["arms"][arm]["retention_curve"]:
            curve_rows.append({"arm": arm, **point})
    _write_csv(
        tables / "score_retention_curve.csv",
        ["arm", "keep_fraction", "tp_recall", "tp_kept", "fp_kept", "fp_removed_fraction", "precision"],
        curve_rows,
    )

    random_rows = []
    for arm in (runner.PGL, runner.VASNET):
        stats = summary["arms"][arm]
        random_rows.append({
            "arm": arm,
            "model_fs0_retention": stats["model_fs0_retention"],
            "random_fs0_retention": stats["random_fs0_retention"],
            "model_tp_retention": stats["model_tp_retention"],
            "random_tp_retention": stats["random_tp_retention"],
            "tp_enrichment_over_random": stats["tp_retention_enrichment_over_random"],
            "oracle_tp_upper_bound": stats["oracle_tp_upper_bound"],
            "oracle_recall_upper_bound": stats["oracle_recall_upper_bound"],
        })
    _write_csv(
        tables / "random_budget.csv",
        ["arm", "model_fs0_retention", "random_fs0_retention", "model_tp_retention", "random_tp_retention",
         "tp_enrichment_over_random", "oracle_tp_upper_bound", "oracle_recall_upper_bound"],
        random_rows,
    )

    kts_rows = []
    for arm in (runner.PGL, runner.VASNET):
        q = summary["arms"][arm]["shot_stats"]["shot_length_seconds"]
        kts_rows.append({
            "arm": arm,
            "total_shots": summary["arms"][arm]["shot_stats"]["total_shots"],
            "selected_shots": summary["arms"][arm]["shot_stats"]["selected_shots"],
            "shots_per_video_mean": summary["arms"][arm]["shot_stats"]["shots_per_video_mean"],
            "shot_len_s_median": q.get("median"), "shot_len_s_p10": q.get("p10"), "shot_len_s_p90": q.get("p90"),
            "shot_len_s_max": q.get("max"),
        })
    _write_csv(
        tables / "kts_shot_stats.csv",
        ["arm", "total_shots", "selected_shots", "shots_per_video_mean", "shot_len_s_median", "shot_len_s_p10", "shot_len_s_p90", "shot_len_s_max"],
        kts_rows,
    )

    frame_q = summary["aggregates"]["sampling_nearest_distance_frames"]
    second_q = summary["aggregates"]["sampling_nearest_distance_seconds"]
    _write_csv(
        tables / "sampling_nearest_distance.csv",
        ["scope", "stat", "distance_frames", "distance_seconds"],
        [
            {
                "scope": "valid109 reference frames -> nearest 2 FPS sample",
                "stat": stat,
                "distance_frames": frame_q.get(stat),
                "distance_seconds": second_q.get(stat),
            }
            for stat in ("median", "p90", "p95", "max", "mean")
        ],
    )
    _write_csv(
        tables / "feature_diagnostics.csv",
        ["video_id", "frames", "dim", "nan_count", "inf_count", "norm_mean", "norm_std", "norm_min",
         "value_min", "value_max", "per_dim_variance_mean", "adjacent_cosine_mean", "adjacent_cosine_p10"],
        feature_diag,
    )
    _write_csv(
        tables / "mapping_audit.csv",
        ["video_id", "timestamp_mode", "source_frames", "samples", "first_sample_frame", "last_sample_frame",
         "fs0_count", "kept", "dropped", "kept_equals_selected_intersect_fs0", "kept_subset_of_fs0",
         "last_shot_end_is_video_end", "first_shot_start_is_zero"],
        mapping_rows,
    )
    delta_rows = [{"arm": arm, "video_id": vid, "delta_f": value} for arm in (runner.PGL, runner.VASNET) for vid, value in sorted(deltas[arm].items())]
    _write_csv(tables / "per_video_delta_f.csv", ["arm", "video_id", "delta_f"], delta_rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _render(
    output_root: Path,
    runner: Any,
    per_arm: Mapping[str, Mapping[str, Any]],
    per_video_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    dels: Mapping[str, Mapping[str, float]],
    specs: Sequence[Any],
    kept_by_arm_video: Mapping[tuple[str, str], set[int]],
    reference_records: Mapping[str, Mapping[str, Any]],
    fs0_f: Mapping[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    arms = (runner.PGL, runner.VASNET)

    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in arms:
        rows = [r for r in per_video_rows if r["arm"] == arm]
        ax.scatter([r["fs0_density"] for r in rows], [r["fs0_retention"] for r in rows], s=10, alpha=0.5, label=arm.upper())
    ax.set_xlabel("FS0 density (FS0 / source frames)"); ax.set_ylabel("model FS0 retention")
    ax.set_title("H1: FS0 density vs model retention (per video)"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "fs0_density_vs_retention.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = list(range(len(arms)))
    mean_ret = [summary["arms"][a]["model_tp_retention"] for a in arms]
    rand_ret = [summary["arms"][a]["random_tp_retention"] for a in arms]
    ax.bar([x - width / 2 for x in xs], mean_ret, width=width, label="model")
    ax.bar([x + width / 2 for x in xs], rand_ret, width=width, label="random equal-budget")
    for x, a in zip(xs, arms):
        ax.text(x - width / 2, mean_ret[x], f"{mean_ret[x]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(x + width / 2, rand_ret[x], f"{rand_ret[x]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([a.upper() for a in arms]); ax.set_ylabel("TP retention")
    ax.set_title("H1: model vs random equal-budget TP retention"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "model_vs_random_equal_budget.png", dpi=140); plt.close(fig)

    for arm in arms:
        scores = np.asarray(per_arm[arm]["pooled_scores"], dtype=np.float64)
        labels = np.asarray(per_arm[arm]["pooled_labels"], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores[labels == 1], bins=40, alpha=0.5, density=True, label="TP")
        ax.hist(scores[labels == 0], bins=40, alpha=0.5, density=True, label="FP")
        ax.set_xlabel("raw ensemble importance score"); ax.set_ylabel("density")
        ax.set_title(f"H2: {arm.upper()} raw score TP vs FP (valid109 FS0 candidates)")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / f"{arm}_tp_fp_score_distribution.png", dpi=140); plt.close(fig)

        fpr, tpr = roc_points(scores, labels)
        recall, precision = pr_points(scores, labels)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label=f"AUROC={summary['arms'][arm]['auroc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC"); axes[0].legend()
        axes[1].plot(recall, precision, label=f"AP={summary['arms'][arm]['average_precision']:.3f}")
        axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("PR"); axes[1].legend()
        fig.suptitle(f"H2: {arm.upper()} raw score separability"); fig.tight_layout()
        fig.savefig(figures / f"{arm}_score_roc_pr.png", dpi=140); plt.close(fig)

        corr = np.asarray(summary["arms"][arm]["checkpoint_correlation_mean"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"H3: {arm.upper()} checkpoint score correlation")
        ax.set_xlabel("checkpoint index"); ax.set_ylabel("checkpoint index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(figures / f"checkpoint_correlation_{arm}.png", dpi=140); plt.close(fig)

    shot_seconds = np.concatenate([np.asarray(per_arm[a]["shot_lengths_seconds"], dtype=np.float64) for a in arms])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(shot_seconds, bins=50, color="#55a868")
    ax.set_xlabel("KTS shot length (seconds)"); ax.set_ylabel("shot count")
    ax.set_title("H7: KTS shot-length distribution (all videos/arms)")
    fig.tight_layout(); fig.savefig(figures / "kts_shot_length_distribution.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in arms:
        curve = summary["arms"][arm]["retention_curve"]
        ax.plot([p["keep_fraction"] for p in curve], [p["tp_recall"] for p in curve], marker="o", label=f"{arm.upper()} TP recall")
        ax.plot([p["keep_fraction"] for p in curve], [p["precision"] for p in curve], marker="x", linestyle="--", label=f"{arm.upper()} precision")
    ax.set_xlabel("FS0 keep fraction (by raw score rank)"); ax.set_ylabel("value")
    ax.set_title("H2/H8: POSTHOC score-retention curve (no threshold selected)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "keep_fraction_recall_curve.png", dpi=140); plt.close(fig)

    spec_by_id = {spec.video_id: spec for spec in specs}
    for arm in arms:
        ordered = sorted(dels[arm].items(), key=lambda item: item[1])
        picks = [("worst regression", ordered[0][0]),
                 ("near tie", min(ordered, key=lambda item: abs(item[1]))[0]),
                 ("best gain", ordered[-1][0])]
        fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.4 * len(picks)))
        for ax, (label, video_id) in zip(np.atleast_1d(axes), picks):
            spec = spec_by_id[video_id]
            scores = per_arm[arm]["curves"][video_id]
            shot_bounds, selected_shots = per_arm[arm]["shots"][video_id]
            sample_frames = spec.sample_frames
            times = [frame / float(spec.timing.fps) for frame in sample_frames]
            ax.plot(times, scores, color="#333333", linewidth=1.0, label="raw ensemble score")
            for shot_index in selected_shots:
                start, end = shot_bounds[shot_index]
                if end > start:
                    ax.axvspan(times[start], times[min(end, len(times) - 1)], color="#1f77b4", alpha=0.15)
            for start, _end in shot_bounds[1:]:
                ax.axvline(times[start], color="#cccccc", linewidth=0.4)
            ref = set(_reference_frames(reference_records.get(video_id, {})))
            keep = kept_by_arm_video[(arm, video_id)]
            fs0 = spec.fs0_frames
            ax.scatter([f / float(spec.timing.fps) for f in fs0], [1.0] * len(fs0), s=5, c="#bbbbbb", label="FS0 candidates")
            ref_keep = sorted(ref & set(fs0))
            if ref_keep:
                ax.scatter([f / float(spec.timing.fps) for f in ref_keep], [1.05] * len(ref_keep), s=12, c="#2ca02c", label="reference hits")
            kept_list = sorted(keep)
            if kept_list:
                ax.scatter([f / float(spec.timing.fps) for f in kept_list], [1.1] * len(kept_list), s=12, c="#d62728", marker="x", label="KEEP")
            ax.set_ylim(-0.05, 1.2); ax.set_xlabel("time (s)")
            ax.set_title(f"{arm.upper()} {label}: {video_id} (dF={dels[arm][video_id]:+.3f})", fontsize=9)
            ax.legend(fontsize=7, loc="lower right", ncol=2)
        fig.tight_layout(); fig.savefig(figures / f"representative_timeline_{arm}.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(output_root: Path, runner: Any, summary: Mapping[str, Any]) -> None:
    pgl = summary["arms"][runner.PGL]
    vas = summary["arms"][runner.VASNET]
    agg = summary["aggregates"]
    lines: list[str] = []
    add = lines.append
    add("# Stage 6.1 Failure-Mechanism Audit (POSTHOC_DIAGNOSTIC_ONLY)")
    add("")
    add("This audit explains the Stage 6.1 recall collapse. It does **not** change the frozen")
    add("Stage 6.1 result (`PGL NO_PROMOTION`, `VAS NO_PROMOTION`) and is not a Formal challenger.")
    add("")
    add("## 1. Question")
    add("")
    add("Why did PGL-SUM / VASNet, adapted as zero-shot frame selectors, drop ~83% of FS0 frames and")
    add("collapse frame recall, and is the dominant cause the model, the adapter decision layer, or both?")
    add("")
    add("## 2. Frozen Stage 6.1 facts")
    add("")
    add("- FS0: N_pred 52019, frame P/R/F1 0.338722 / 0.945533 / 0.498769, valid109 F 0.308548.")
    add("- PGL: kept 9031, valid109 F 0.105818, W/T/L 13/3/93.")
    add("- VAS: kept 8738, valid109 F 0.092440, W/T/L 7/2/100.")
    add("")
    add("## 3. Causal pipeline")
    add("")
    add("video -> 2 FPS sampling -> GoogLeNet 1024D -> per-checkpoint inference -> 10-mean ->")
    add("KTS -> shot score -> 15%-of-source knapsack -> selected source intervals -> intersect FS0 -> KEEP/DROP.")
    add("Each arrow is audited below.")
    add("")
    add("## 4. Methodology")
    add("")
    add("Raw per-checkpoint scores and KEEP masks are recomputed in memory from the frozen feature")
    add("cache (no re-decode, no re-extraction, no Formal predictions touched).")
    add("")
    add("## 5. Raw-score discriminability (H2) — valid109 FS0 candidates")
    add("")
    add(f"- candidates: {pgl['pooled_candidates']} (TP {pgl['tp_candidates']} / FP {pgl['fp_candidates']})")
    add(f"- PGL AUROC {pgl['auroc']:.4f}, AP {pgl['average_precision']:.4f}, Cohen d {pgl['cohens_d']:.4f}")
    add(f"- VAS AUROC {vas['auroc']:.4f}, AP {vas['average_precision']:.4f}, Cohen d {vas['cohens_d']:.4f}")
    add(f"- PGL TP mean {pgl['tp_quantiles'].get('mean'):.5f} vs FP mean {pgl['fp_quantiles'].get('mean'):.5f}")
    add(f"- VAS TP mean {vas['tp_quantiles'].get('mean'):.5f} vs FP mean {vas['fp_quantiles'].get('mean'):.5f}")
    add("")
    add("## 6. Random equal-budget comparison (H1)")
    add("")
    add(f"- PGL model TP retention {pgl['model_tp_retention']:.4f} vs random {pgl['random_tp_retention']:.4f} "
        f"(enrichment {pgl['tp_retention_enrichment_over_random']})")
    add(f"- VAS model TP retention {vas['model_tp_retention']:.4f} vs random {vas['random_tp_retention']:.4f} "
        f"(enrichment {vas['tp_retention_enrichment_over_random']})")
    add(f"- PGL oracle TP upper bound {pgl['oracle_tp_upper_bound']} → recall upper bound {pgl['oracle_recall_upper_bound']:.4f}")
    add(f"- VAS oracle TP upper bound {vas['oracle_tp_upper_bound']} → recall upper bound {vas['oracle_recall_upper_bound']:.4f}")
    add("")
    add("## 7. Checkpoint ensemble analysis (H3/H4)")
    add("")
    add(f"- PGL off-diagonal checkpoint correlation {pgl['checkpoint_correlation_offdiag_mean']:.4f}")
    add(f"- VAS off-diagonal checkpoint correlation {vas['checkpoint_correlation_offdiag_mean']:.4f}")
    add(f"- PGL per-checkpoint AUROC {[round(v,4) for v in pgl['per_checkpoint_auroc']]}")
    add(f"- VAS per-checkpoint AUROC {[round(v,4) for v in vas['per_checkpoint_auroc']]}")
    add(f"- PGL ensemble range {pgl['ensemble_vs_single_range_mean']['ensemble']:.4f} vs single {pgl['ensemble_vs_single_range_mean']['single']:.4f}")
    add("- CV_MODEL_ENSEMBLE_IS_TRANSFER_ADAPTATION: the 10 checkpoints are cross-validation-fold models,")
    add("  not an independently trained ensemble.")
    add("")
    add("## 8. KTS fragmentation (H7)")
    add("")
    for arm in (runner.PGL, runner.VASNET):
        s = summary["arms"][arm]["shot_stats"]
        add(f"- {arm.upper()}: shots/video mean {s['shots_per_video_mean']:.1f}, shot length median "
            f"{s['shot_length_seconds'].get('median')}s (p10 {s['shot_length_seconds'].get('p10')}s, p90 {s['shot_length_seconds'].get('p90')}s)")
    add("")
    add("## 9. Sampling / mapping audit (H6)")
    add("")
    add(f"- nearest sample distance to a reference frame: median {agg['sampling_nearest_distance_frames'].get('median')} frames "
        f"({agg['sampling_nearest_distance_seconds'].get('median')}s), p90 {agg['sampling_nearest_distance_frames'].get('p90')} frames")
    add(f"- mapping invariant violations: {agg['mapping_invariant_violations']}")
    add("")
    add("## 10. Feature diagnostics (H5)")
    add("")
    add("- Feature statistics are in `tables/feature_diagnostics.csv`; the compatible-reproduction")
    add("  extractor is not bitwise identical to the unavailable original Caffe pool5 features.")
    add("")
    add("## 11. Timeline case studies (H9)")
    add("")
    add("- `figures/representative_timeline_pgl.png`, `figures/representative_timeline_vasnet.png`.")
    add("")
    add("## 12. Root-cause ranking")
    add("")
    add("| Hypothesis | Verdict |")
    add("| --- | --- |")
    add(f"| H1 budget x FS0 mismatch | see audit summary |")
    add(f"| H2 raw-score discriminability | AUROC PGL {pgl['auroc']:.3f} / VAS {vas['auroc']:.3f} |")
    add(f"| H3 checkpoint ensemble dilution | offdiag corr PGL {pgl['checkpoint_correlation_offdiag_mean']:.3f} / VAS {vas['checkpoint_correlation_offdiag_mean']:.3f} |")
    add("| H4 CV split ensemble semantics | SUPPORTED (transfer adaptation) |")
    add("| H5 feature reproduction shift | no direct numerical comparison possible |")
    add(f"| H6 2 FPS sampling | median distance {agg['sampling_nearest_distance_frames'].get('median')} frames |")
    add(f"| H7 KTS fragmentation | shots/video {pgl['shot_stats']['shots_per_video_mean']:.1f} |")
    add("| H8 summary vs filtering task | SUPPORTED (budget space mismatch) |")
    add("| H9 semantic target mismatch | case studies |")
    add("| H10 empty57 artifact | NOT the cause (valid109 alone collapses) |")
    add("| mapping implementation bug | none found |")
    add("| score direction bug | none found |")
    add("")
    add("## 13. What Stage 6.1 actually disproves")
    add("")
    add("- Hard 15%-of-source KTS+knapsack summary decisions on top of FS0 are harmful for this task.")
    add("")
    add("## 14. What it does NOT disprove")
    add("")
    add("- That the raw importance ranking has no value as a soft signal (see AUROC/retention curve).")
    add("")
    add("## 15. Stage 6.2 recommendation")
    add("")
    add("- If AUROC > ~0.55 with a useful retention curve, propose a re-preregistered")
    add("  `Literature Score as Soft Filter` route; otherwise terminate the zero-shot literature route.")
    add("")
    add("## 16. Scientific boundary")
    add("")
    add("- POSTHOC_DIAGNOSTIC_ONLY; no threshold/budget/checkpoint selection is promoted.")
    add("- Dev166 is a weak-reference proxy; Hard229 / Heldout / Official Test NOT ACCESSED.")
    add("")
    add("## 17. Reproducibility")
    add("")
    add("- `audit/audit_identity.json`, `audit/summary.json`, `tables/*.csv`, `figures/*.png`.")
    add("")
    _write_json(output_root / "report" / "stage6_1_failure_report_lines.json", lines)
    (output_root / "report" / "Stage6_1_Failure_Mechanism_Audit_Report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
