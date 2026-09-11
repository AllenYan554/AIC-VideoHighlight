"""Stage 5.5 weak-frame temporal proxy metrics, guardrails and decisions.

The official competition metric couples frame selection with same-frame spatial
IoU.  Stage 5.5 freezes the spatial side and studies only frame cardinality, so
every number produced here is a **weak-frame temporal proxy** and must never be
reported as an official score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from aic_video_highlight.spatial_composition.frame_selection import FS0, FS1, FS2

LOW_RECALL_THRESHOLD = 0.80
DEV_MIN_RECALL_DELTA = -0.02
DEV_MAX_LOW_RECALL_RATE_DELTA = 0.02
DEV_MAX_MISSED_REFERENCE_RATE_DELTA = 0.02
DEV_MIN_F1_DELTA = 0.01
HARD_MIN_RECALL_DELTA = -0.02
HARD_MAX_LOW_RECALL_RATE_DELTA = 0.02

# Pre-registered conservatism order: FS-0 > FS-1 > FS-2 (smaller rank wins).
CONSERVATISM_RANK = {FS0: 0, FS1: 1, FS2: 2}

PROXY_LABEL = "weak-frame temporal proxy (Macro-F1); NOT an official score"


class FrameCalibrationError(ValueError):
    """Raised when a Stage 5.5 metric or decision input is malformed."""


@dataclass(frozen=True, slots=True)
class VideoFrameMetrics:
    """Per-video weak-frame temporal proxy counts and ratios (macro inputs)."""

    video_id: str
    predicted_frames: int
    reference_frames: int
    true_positive_frames: int
    precision: float
    recall: float
    f1: float
    missed_reference_frames: int
    low_recall: bool
    empty_prediction: bool


def weak_frame_video_metrics(
    predicted_frames: Iterable[int], reference_frames: Iterable[int]
) -> VideoFrameMetrics:
    """Exact per-video intersection cardinality metrics (set semantics)."""
    predicted = frozenset(int(frame) for frame in predicted_frames)
    reference = frozenset(int(frame) for frame in reference_frames)
    tp = len(predicted & reference)
    n_pred = len(predicted)
    n_ref = len(reference)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_ref if n_ref else 0.0
    denominator = n_pred + n_ref
    f1 = (2.0 * tp / denominator) if denominator else 0.0
    return VideoFrameMetrics(
        video_id="",
        predicted_frames=n_pred,
        reference_frames=n_ref,
        true_positive_frames=tp,
        precision=precision,
        recall=recall,
        f1=f1,
        missed_reference_frames=n_ref - tp,
        low_recall=recall < LOW_RECALL_THRESHOLD,
        empty_prediction=n_pred == 0,
    )


def macro_metrics(videos: Sequence[VideoFrameMetrics]) -> dict[str, float | int]:
    """Video-macro Precision / Recall / F1 plus guardrail aggregates.

    Macro metrics average the per-video ratios, so every video contributes
    equally.  Pooled ratios are exposed separately and are DIAGNOSTIC_ONLY.
    """
    if not videos:
        raise FrameCalibrationError("macro metrics require at least one video")
    count = len(videos)
    tp_total = sum(item.true_positive_frames for item in videos)
    pred_total = sum(item.predicted_frames for item in videos)
    ref_total = sum(item.reference_frames for item in videos)
    missed_total = sum(item.missed_reference_frames for item in videos)
    return {
        "video_count": count,
        "macro_precision": statistics.fmean(item.precision for item in videos),
        "macro_recall": statistics.fmean(item.recall for item in videos),
        "macro_f1": statistics.fmean(item.f1 for item in videos),
        "low_recall_rate": sum(item.low_recall for item in videos) / count,
        "missed_reference_frame_rate": (
            missed_total / ref_total if ref_total else 0.0
        ),
        "empty_prediction_count": sum(item.empty_prediction for item in videos),
        "total_predicted_frames": pred_total,
        "total_reference_frames": ref_total,
        "total_true_positive_frames": tp_total,
        # Diagnostics only; the preregistered decision never consumes these.
        "DIAGNOSTIC_ONLY_pooled_precision": tp_total / pred_total if pred_total else 0.0,
        "DIAGNOSTIC_ONLY_pooled_recall": tp_total / ref_total if ref_total else 0.0,
        "DIAGNOSTIC_ONLY_pooled_f1": (
            (2.0 * tp_total / (pred_total + ref_total))
            if (pred_total + ref_total)
            else 0.0
        ),
    }


def evaluate_video_set(
    predictions_by_video: Mapping[str, Iterable[int]],
    references_by_video: Mapping[str, Iterable[int]],
) -> dict[str, object]:
    """Evaluate one arm on one frozen video set; returns macro + per-video rows."""
    if set(predictions_by_video) != set(references_by_video):
        raise FrameCalibrationError("prediction/reference video identities differ")
    videos = []
    for video_id in sorted(predictions_by_video):
        metrics = weak_frame_video_metrics(
            predictions_by_video[video_id], references_by_video[video_id]
        )
        videos.append(
            VideoFrameMetrics(
                video_id=video_id,
                predicted_frames=metrics.predicted_frames,
                reference_frames=metrics.reference_frames,
                true_positive_frames=metrics.true_positive_frames,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                missed_reference_frames=metrics.missed_reference_frames,
                low_recall=metrics.low_recall,
                empty_prediction=metrics.empty_prediction,
            )
        )
    return {
        "proxy_label": PROXY_LABEL,
        "macro": macro_metrics(videos),
        "per_video": [
            {
                "video_id": item.video_id,
                "predicted_frames": item.predicted_frames,
                "reference_frames": item.reference_frames,
                "true_positive_frames": item.true_positive_frames,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "missed_reference_frames": item.missed_reference_frames,
                "low_recall": item.low_recall,
                "empty_prediction": item.empty_prediction,
            }
            for item in videos
        ],
    }


def recall_guardrails(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """Preregistered Dev166 Recall Guardrails G1-G4 (machine-readable)."""
    base = baseline["macro"]
    cand = candidate["macro"]
    g1_observed = float(cand["macro_recall"]) - float(base["macro_recall"])
    g2_observed = float(cand["low_recall_rate"]) - float(base["low_recall_rate"])
    g3_observed = float(cand["missed_reference_frame_rate"]) - float(
        base["missed_reference_frame_rate"]
    )
    g4_observed = int(cand["empty_prediction_count"])
    checks = {
        "G1_macro_recall_delta": {
            "observed": g1_observed,
            "operator": ">=",
            "threshold": DEV_MIN_RECALL_DELTA,
            "pass": g1_observed >= DEV_MIN_RECALL_DELTA,
        },
        "G2_low_recall_rate_delta": {
            "observed": g2_observed,
            "operator": "<=",
            "threshold": DEV_MAX_LOW_RECALL_RATE_DELTA,
            "pass": g2_observed <= DEV_MAX_LOW_RECALL_RATE_DELTA,
        },
        "G3_missed_reference_frame_rate_delta": {
            "observed": g3_observed,
            "operator": "<=",
            "threshold": DEV_MAX_MISSED_REFERENCE_RATE_DELTA,
            "pass": g3_observed <= DEV_MAX_MISSED_REFERENCE_RATE_DELTA,
        },
        "G4_empty_prediction_count": {
            "observed": g4_observed,
            "operator": "==",
            "threshold": 0,
            "pass": g4_observed == 0,
        },
    }
    return {"all_pass": all(check["pass"] for check in checks.values()), "checks": checks}


def dev_promotion_objective(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """Guardrails + Precision strict gain + Macro-F1 gain >= +0.01."""
    guardrails = recall_guardrails(baseline, candidate)
    base = baseline["macro"]
    cand = candidate["macro"]
    precision_gain = float(cand["macro_precision"]) - float(base["macro_precision"])
    f1_delta = float(cand["macro_f1"]) - float(base["macro_f1"])
    objective = {
        "guardrails": guardrails,
        "precision_gain": precision_gain,
        "f1_delta": f1_delta,
        "precision_strict_gain": precision_gain > 0.0,
        "f1_gain_ge_threshold": f1_delta >= DEV_MIN_F1_DELTA,
    }
    objective["eligible"] = (
        guardrails["all_pass"]
        and objective["precision_strict_gain"]
        and objective["f1_gain_ge_threshold"]
    )
    return objective


def dev_tie_break_key(metrics: Mapping[str, object], policy: str) -> tuple:
    """Preregistered Dev winner ordering (smaller key sorts first)."""
    if policy not in CONSERVATISM_RANK:
        raise FrameCalibrationError(f"unknown policy in tie-break: {policy}")
    macro = metrics["macro"]
    return (
        -float(macro["macro_f1"]),
        -float(macro["macro_recall"]),
        float(macro["low_recall_rate"]),
        float(macro["missed_reference_frame_rate"]),
        int(macro["total_predicted_frames"]),
        CONSERVATISM_RANK[policy],
    )


def select_dev_winner(
    baseline: Mapping[str, object],
    candidate_evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Freeze at most one Dev166 winner: only fully eligible arms compete."""
    rows: dict[str, dict[str, object]] = {}
    for policy, evaluation in candidate_evaluations.items():
        objective = dev_promotion_objective(baseline, evaluation)
        rows[policy] = {
            "objective": objective,
            "tie_break_key": list(dev_tie_break_key(evaluation, policy)),
        }
    eligible = [
        policy
        for policy in candidate_evaluations
        if rows[policy]["objective"]["eligible"]
    ]
    winner = (
        min(eligible, key=lambda policy: dev_tie_break_key(candidate_evaluations[policy], policy))
        if eligible
        else None
    )
    return {
        "eligible_policies": sorted(eligible),
        "winner_policy": winner,
        "winner_arm": candidate_evaluations[winner]["arm"] if winner else None,
        "rows": rows,
        "selection_rule": (
            "eligible -> higher MacroF1, higher MacroRecall, lower low-R rate, "
            "lower missed-reference rate, fewer N_pred, more conservative "
            "(FS-0 > FS-1 > FS-2)"
        ),
    }


def hard_gates(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """Hard229 non-regression gates H1-H5 (no +1pp requirement)."""
    base = baseline["macro"]
    cand = candidate["macro"]
    recall_delta = float(cand["macro_recall"]) - float(base["macro_recall"])
    low_recall_delta = float(cand["low_recall_rate"]) - float(base["low_recall_rate"])
    checks = {
        "H1_macro_recall_delta": {
            "observed": recall_delta,
            "operator": ">=",
            "threshold": HARD_MIN_RECALL_DELTA,
            "pass": recall_delta >= HARD_MIN_RECALL_DELTA,
        },
        "H2_low_recall_rate_delta": {
            "observed": low_recall_delta,
            "operator": "<=",
            "threshold": HARD_MAX_LOW_RECALL_RATE_DELTA,
            "pass": low_recall_delta <= HARD_MAX_LOW_RECALL_RATE_DELTA,
        },
        "H3_empty_prediction_count": {
            "observed": int(cand["empty_prediction_count"]),
            "operator": "==",
            "threshold": 0,
            "pass": int(cand["empty_prediction_count"]) == 0,
        },
        "H4_precision_non_regression": {
            "observed": float(cand["macro_precision"]) - float(base["macro_precision"]),
            "operator": ">=",
            "threshold": 0.0,
            "pass": float(cand["macro_precision"]) >= float(base["macro_precision"]),
        },
        "H5_f1_non_regression": {
            "observed": float(cand["macro_f1"]) - float(base["macro_f1"]),
            "operator": ">=",
            "threshold": 0.0,
            "pass": float(cand["macro_f1"]) >= float(base["macro_f1"]),
        },
    }
    failed = [name for name, check in checks.items() if not check["pass"]]
    return {
        "all_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def adjudicate_stage5_5(
    dev_winner: str | None, hard_result: Mapping[str, object] | None
) -> dict[str, object]:
    """Exact preregistered decision tree; no near-pass, no amendment, no override."""
    if dev_winner is None:
        return {
            "status": "FS0_FINAL_FROZEN",
            "frozen_policy": FS0,
            "stage5_5_terminal": "STAGE5_5_CLOSED",
            "reason": "no Dev166 candidate passed guardrails + Precision gain + F1 >= +0.01",
            "near_pass_allowed": False,
            "amendment_allowed": False,
            "override_allowed": False,
        }
    if hard_result is None:
        return {
            "status": "DEV_WINNER_PENDING_HARD",
            "frozen_policy": None,
            "dev_winner": dev_winner,
            "stage5_5_terminal": None,
            "reason": "Hard229 confirmation has not been executed",
            "near_pass_allowed": False,
            "amendment_allowed": False,
            "override_allowed": False,
        }
    if hard_result["all_pass"]:
        return {
            "status": "FS_CANDIDATE_FINAL_FROZEN",
            "frozen_policy": dev_winner,
            "dev_winner": dev_winner,
            "stage5_5_terminal": "STAGE5_5_CLOSED",
            "reason": "Hard229 gates H1-H5 all passed for the frozen Dev winner",
            "near_pass_allowed": False,
            "amendment_allowed": False,
            "override_allowed": False,
        }
    return {
        "status": "FS0_FINAL_FROZEN",
        "frozen_policy": FS0,
        "dev_winner": dev_winner,
        "stage5_5_terminal": "STAGE5_5_CLOSED",
        "reason": "Hard229 reapplied FS-0 after one or more Hard gates failed",
        "failed_hard_checks": hard_result.get("failed_checks", []),
        "near_pass_allowed": False,
        "amendment_allowed": False,
        "override_allowed": False,
    }
