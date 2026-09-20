"""Stage 7.1 FTNet post-training evaluation.

This module is read-only over the frozen artifacts:

    * checkpoint            (never modified, identity verified by SHA-256)
    * materialized dataset  (safetensors are only read)
    * MTurk annotations     (read for provenance verification only)

It implements the pre-registered evaluation protocol: a supervised frame
universe, a frozen binary reference on the soft vote target, calibration-only
threshold selection, FS0 (keep-all) baselines, threshold-free ranking metrics,
derived positive-run safety proxies and a Native16 missingness audit.

Nothing here retrains, fine-tunes, re-materializes or touches Official Test /
TVSum.  Human *event* ground truth does not exist for this dataset; only a
derived positive-run proxy is reported and it is never labelled as human event
recall.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .materialize import SPLIT_DIRS
from .native_schema import NATIVE_DIM, NATIVE_FIELDS, NormalizationStats, standardize_native
from .sampling import SAMPLE_PERIOD_SEC

PROTOCOL_SCHEMA = "aic.stage7.ftnet.posttraining-evaluation-protocol/v1"
SUMMARY_SCHEMA = "aic.stage7.ftnet.posttraining-evaluation-summary/v1"
PROVENANCE_SCHEMA = "aic.stage7.ftnet.evaluation-provenance-audit/v1"
EVENT_GT_SCHEMA = "aic.stage7.ftnet.event-gt-audit/v1"

PRIMARY_BINARY_THRESHOLD = 0.5
SENSITIVITY_THRESHOLDS = (0.4, 0.6)
RECALL_SAFETY_TARGET = 0.95
SHORT_RUN_MAX_SEC = 2.0
MEDIUM_RUN_MAX_SEC = 5.0
MIN_MEANINGFUL_PRUNING = 0.25
FS0_NEAR_ZERO_PRUNING = 0.10
DANGEROUS_RECALL = 0.90
MISSINGNESS_FLAG_DELTA = 0.10

EVAL_SPLITS = ("VALIDATION", "CALIBRATION")
THRESHOLD_SWEEP_GRID = tuple(round(0.01 * index, 2) for index in range(101))


class EvaluationError(RuntimeError):
    """Raised when the evaluation protocol cannot be satisfied."""


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    _atomic_text(
        target,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return target


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    _atomic_text(target, "".join(line + "\n" for line in lines))
    return target


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, target)
    return target


def _rounded(value: float | None, digits: int = 10) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# frame container and model scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoFrames:
    video_id: str
    split: str
    category: str
    source_frame_id: np.ndarray
    timestamp: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    adjacency_mask: np.ndarray
    native_missing: np.ndarray

    @property
    def length(self) -> int:
        return int(self.target.shape[0])


@dataclass(frozen=True)
class ScoredVideo:
    frames: VideoFrames
    p_keep: np.ndarray


def resolve_device(name: str = "auto") -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("cuda requested but is not available")
    return name


def load_checkpoint_model(checkpoint_path: str | Path, device: str):
    """Load the frozen FTNet checkpoint without modifying or re-saving it."""

    import torch

    from .checkpoint import load_checkpoint
    from .model import FTNet, FTNetConfig

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise EvaluationError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "aic.stage7.ftnet.checkpoint/v1":
        raise EvaluationError(f"unexpected checkpoint schema: {path}")
    config_values = dict(payload.get("config", {}))
    if config_values.get("temporal_dilations") is not None:
        config_values["temporal_dilations"] = tuple(config_values["temporal_dilations"])
    try:
        config = FTNetConfig(**config_values)
    except TypeError as exc:
        raise EvaluationError(f"checkpoint config is not a valid FTNetConfig: {exc}") from exc
    model = FTNet(config)
    info = load_checkpoint(path, model=model, restore_rng=False)
    model.to(device)
    model.eval()
    return model, info


def load_video_frames(data_root: str | Path, split: str, video_id: str, *, category: str) -> VideoFrames:
    from safetensors.numpy import load_file

    root = Path(data_root).expanduser().resolve()
    path = root / SPLIT_DIRS[split] / f"{video_id}.safetensors"
    tensors = load_file(str(path))
    return VideoFrames(
        video_id=video_id,
        split=split,
        category=category,
        source_frame_id=np.asarray(tensors["source_frame_id"], dtype=np.int64),
        timestamp=np.asarray(tensors["timestamp"], dtype=np.float64),
        target=np.asarray(tensors["target"], dtype=np.float64),
        loss_mask=np.asarray(tensors["loss_mask"], dtype=bool),
        adjacency_mask=np.asarray(tensors["adjacency_mask"], dtype=bool),
        native_missing=np.asarray(tensors["native_missing"], dtype=bool),
    )


def score_video(
    model,
    frames: VideoFrames,
    data_root: str | Path,
    stats: NormalizationStats,
    *,
    device: str,
) -> np.ndarray:
    """Deterministic single-video forward pass; returns ``p_keep`` per frame."""

    import torch
    from safetensors.numpy import load_file

    path = Path(data_root).expanduser().resolve() / SPLIT_DIRS[frames.split] / f"{frames.video_id}.safetensors"
    tensors = load_file(str(path))
    visual = torch.from_numpy(np.asarray(tensors["visual"], dtype=np.float32)).unsqueeze(0)
    raw_native = torch.from_numpy(np.asarray(tensors["native"], dtype=np.float32))
    missing = torch.from_numpy(np.asarray(tensors["native_missing"], dtype=bool))
    native = standardize_native(raw_native, stats, missing).unsqueeze(0)
    length = int(visual.shape[1])
    sequence_mask = torch.ones(1, length, dtype=torch.bool)
    adjacency = torch.from_numpy(frames.adjacency_mask.astype(bool)).unsqueeze(0)
    visual = visual.to(device)
    native = native.to(device)
    sequence_mask = sequence_mask.to(device)
    adjacency = adjacency.to(device)
    with torch.inference_mode():
        output = model(
            visual,
            native_features=native,
            sequence_mask=sequence_mask,
            adjacency_mask=adjacency,
        )
    return output.probabilities[0].detach().cpu().numpy().astype(np.float64)


def score_split(
    model,
    data_root: str | Path,
    split: str,
    stats: NormalizationStats,
    *,
    device: str,
    categories: Mapping[str, str] | None = None,
    video_ids: Sequence[str] | None = None,
) -> list[ScoredVideo]:
    root = Path(data_root).expanduser().resolve() / SPLIT_DIRS[split]
    if video_ids is None:
        paths = sorted(root.glob("*.safetensors"))
    else:
        paths = [root / f"{video_id}.safetensors" for video_id in sorted(video_ids)]
    if not paths:
        raise EvaluationError(f"no materialized videos for split {split} in {root}")
    scored: list[ScoredVideo] = []
    for path in paths:
        video_id = path.stem
        category = "" if categories is None else categories.get(video_id, "")
        frames = load_video_frames(data_root, split, video_id, category=category)
        probabilities = score_video(model, frames, data_root, stats, device=device)
        if probabilities.shape != frames.target.shape:
            raise EvaluationError(f"score shape mismatch for {video_id}")
        scored.append(ScoredVideo(frames=frames, p_keep=probabilities))
    return scored


# ---------------------------------------------------------------------------
# frozen binary reference + pure metrics
# ---------------------------------------------------------------------------


def binary_reference(target: np.ndarray, loss_mask: np.ndarray, threshold: float) -> np.ndarray:
    """Return the binary reference on the supervised frame universe."""

    supervised = np.asarray(loss_mask, dtype=bool)
    values = np.asarray(target, dtype=np.float64)
    if threshold <= 0.0:
        reference = values > 0.0
    else:
        reference = values >= float(threshold)
    return (reference & supervised).astype(bool)


def universe_mask(loss_mask: np.ndarray) -> np.ndarray:
    return np.asarray(loss_mask, dtype=bool)


def confusion_counts(y_true: np.ndarray, keep: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true, dtype=bool)
    k = np.asarray(keep, dtype=bool)
    if y.shape != k.shape:
        raise EvaluationError("y_true and keep must share the shape")
    tp = int(np.sum(y & k))
    fp = int(np.sum((~y) & k))
    fn = int(np.sum(y & (~k)))
    tn = int(np.sum((~y) & (~k)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | None]:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = None
    false_deletion = (1.0 - recall) if recall is not None else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_deletion_rate": false_deletion,
    }


def operating_metrics(
    subsets: Sequence[ScoredVideo],
    *,
    tau: float,
    reference_threshold: float = PRIMARY_BINARY_THRESHOLD,
) -> dict[str, Any]:
    """Pooled frame-level metrics at one threshold on the supervised universe."""

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    universe = 0
    dropped = 0
    positives = 0
    for scored in subsets:
        mask = universe_mask(scored.frames.loss_mask)
        y = binary_reference(scored.frames.target, scored.frames.loss_mask, reference_threshold)
        keep = (scored.p_keep >= tau) & mask
        local = confusion_counts(y, keep)
        for key in counts:
            counts[key] += local[key]
        universe += int(mask.sum())
        dropped += int((mask & (scored.p_keep < tau)).sum())
        positives += int(y.sum())
    metrics = metrics_from_counts(counts)
    pruning_rate = dropped / universe if universe else None
    return {
        "tau": float(tau),
        "reference_threshold": float(reference_threshold),
        "universe_frames": universe,
        "positive_frames": positives,
        "predicted_keep": int(counts["tp"] + counts["fp"]),
        "predicted_drop": int(counts["fn"] + counts["tn"]),
        "pruning_rate": pruning_rate,
        **{key: counts[key] for key in ("tp", "fp", "fn", "tn")},
        **metrics,
    }


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Step-wise average precision with correct handling of tied scores."""

    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    if y.shape != p.shape:
        raise EvaluationError("y_true and scores must share the shape")
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    p_sorted = p[order]
    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    length = len(y_sorted)
    while index < length:
        end = index
        while end < length and p_sorted[end] == p_sorted[index]:
            end += 1
        group = y_sorted[index:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ap += precision * (recall - previous_recall)
        previous_recall = recall
        index = end
    return float(ap)


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return []
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    p_sorted = p[order]
    points: list[dict[str, float]] = []
    tp = 0
    fp = 0
    index = 0
    length = len(y_sorted)
    while index < length:
        end = index
        while end < length and p_sorted[end] == p_sorted[index]:
            end += 1
        group = y_sorted[index:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / positives
        points.append({"threshold": float(p_sorted[index]), "precision": float(precision), "recall": float(recall)})
        index = end
    if points and points[-1]["recall"] < 1.0:
        points.append({"threshold": 1.0, "precision": 0.0, "recall": 1.0})
    return points


def threshold_candidates(scores: np.ndarray) -> list[float]:
    values = np.asarray(scores, dtype=np.float64)
    candidates = {0.0}
    candidates.update(float(value) for value in np.unique(values))
    return sorted(candidates)


def select_tau_f1(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """F1-optimal threshold; tie-break: higher recall, then lower threshold."""

    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    best: dict[str, Any] | None = None
    for tau in threshold_candidates(p):
        keep = p >= tau
        counts = confusion_counts(y, keep)
        metrics = metrics_from_counts(counts)
        f1 = metrics["f1"]
        if f1 is None:
            continue
        record = {
            "tau": float(tau),
            "f1": float(f1),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        if best is None:
            best = record
            continue
        if record["f1"] > best["f1"] + 1e-12:
            best = record
        elif abs(record["f1"] - best["f1"]) <= 1e-12:
            if record["recall"] > best["recall"] + 1e-12:
                best = record
            elif abs(record["recall"] - best["recall"]) <= 1e-12 and record["tau"] < best["tau"]:
                best = record
    if best is None:
        raise EvaluationError("no valid F1 threshold candidates")
    return best


def select_tau_safe(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float = RECALL_SAFETY_TARGET,
) -> dict[str, Any] | None:
    """Max-pruning threshold subject to recall >= min_recall."""

    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    if int(y.sum()) == 0:
        return None
    candidates: list[dict[str, Any]] = []
    universe = len(y)
    for tau in threshold_candidates(p):
        keep = p >= tau
        counts = confusion_counts(y, keep)
        metrics = metrics_from_counts(counts)
        if metrics["recall"] is None or metrics["recall"] + 1e-12 < min_recall:
            continue
        pruning = float(int((~keep).sum()) / universe) if universe else 0.0
        candidates.append(
            {
                "tau": float(tau),
                "recall": float(metrics["recall"]),
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "pruning_rate": pruning,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row["pruning_rate"], -row["recall"], -row["tau"]))
    best = candidates[0]
    best["status"] = "FOUND"
    best["min_recall"] = float(min_recall)
    return best


# ---------------------------------------------------------------------------
# derived positive-run proxy (never called human event ground truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveRun:
    video_id: str
    split: str
    frame_indices: tuple[int, ...]
    start_ts: float
    end_ts: float
    duration_sec: float

    def stratum(self) -> str:
        if self.duration_sec <= SHORT_RUN_MAX_SEC:
            return "short"
        if self.duration_sec <= MEDIUM_RUN_MAX_SEC:
            return "medium"
        return "long"


def positive_clip_coverage(
    clips: Sequence[Any],
    source_frame_ids: np.ndarray,
    *,
    vote_fraction: float = PRIMARY_BINARY_THRESHOLD,
) -> np.ndarray:
    """Sampled frames covered by at least one clip with mean vote >= threshold."""

    covered = np.zeros(len(source_frame_ids), dtype=bool)
    frame_ids = np.asarray(source_frame_ids, dtype=np.int64)
    for clip in clips:
        if (clip.vote / 5.0) < vote_fraction:
            continue
        covered |= (frame_ids >= clip.start_frame) & (frame_ids < clip.end_frame)
    return covered


def extract_positive_runs(
    frames: VideoFrames,
    coverage: np.ndarray,
) -> list[PositiveRun]:
    runs: list[PositiveRun] = []
    indices = [index for index, flag in enumerate(coverage) if flag]
    if not indices:
        return runs
    current: list[int] = [indices[0]]
    for index in indices[1:]:
        adjacent = bool(frames.adjacency_mask[index]) and index == current[-1] + 1
        if adjacent:
            current.append(index)
        else:
            runs.append(_make_run(frames, current))
            current = [index]
    runs.append(_make_run(frames, current))
    return runs


def _make_run(frames: VideoFrames, indices: Sequence[int]) -> PositiveRun:
    start_ts = float(frames.timestamp[indices[0]])
    end_ts = float(frames.timestamp[indices[-1]])
    duration = end_ts - start_ts + SAMPLE_PERIOD_SEC
    return PositiveRun(
        video_id=frames.video_id,
        split=frames.split,
        frame_indices=tuple(int(index) for index in indices),
        start_ts=start_ts,
        end_ts=end_ts,
        duration_sec=duration,
    )


def run_metrics(
    runs: Sequence[PositiveRun],
    scored_by_video: Mapping[str, ScoredVideo],
    *,
    tau: float,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in runs:
        scored = scored_by_video.get(run.video_id)
        if scored is None:
            raise EvaluationError(f"missing scored video for run: {run.video_id}")
        keep_flags = [bool(scored.p_keep[index] >= tau) for index in run.frame_indices]
        total = len(keep_flags)
        kept = int(sum(keep_flags))
        per_run.append(
            {
                "video_id": run.video_id,
                "split": run.split,
                "start_ts": _rounded(run.start_ts, 6),
                "end_ts": _rounded(run.end_ts, 6),
                "duration_sec": _rounded(run.duration_sec, 6),
                "stratum": run.stratum(),
                "positive_frames": total,
                "kept_frames": kept,
                "survived": kept > 0,
                "whole_deleted": kept == 0,
                "retention_ratio": kept / total if total else 0.0,
            }
        )
    aggregate = _aggregate_runs(per_run)
    aggregate["per_run"] = per_run
    aggregate["tau"] = float(tau)
    aggregate["status"] = "DERIVED_POSITIVE_RUN_PROXY"
    aggregate["not_human_event_gt"] = True
    return aggregate


def _aggregate_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {
                "runs": 0,
                "survival_recall": None,
                "whole_deletion_rate": None,
                "retention_mean": None,
                "retention_median": None,
                "retention_p10": None,
                "retention_p25": None,
                "retention_min": None,
            }
        retention = np.asarray([row["retention_ratio"] for row in subset], dtype=np.float64)
        survived = sum(1 for row in subset if row["survived"])
        deleted = sum(1 for row in subset if row["whole_deleted"])
        return {
            "runs": len(subset),
            "survival_recall": survived / len(subset),
            "whole_deletion_rate": deleted / len(subset),
            "retention_mean": float(retention.mean()),
            "retention_median": float(np.median(retention)),
            "retention_p10": float(np.percentile(retention, 10)),
            "retention_p25": float(np.percentile(retention, 25)),
            "retention_min": float(retention.min()),
        }

    result = {"overall": _block(list(rows))}
    for stratum in ("short", "medium", "long"):
        result[stratum] = _block([row for row in rows if row["stratum"] == stratum])
    return result


def select_tau_run_safe(
    y_true: np.ndarray,
    scores: np.ndarray,
    runs: Sequence[PositiveRun],
    scored_by_video: Mapping[str, ScoredVideo],
    *,
    min_recall: float = RECALL_SAFETY_TARGET,
) -> dict[str, Any] | None:
    """Derived-proxy safe point: universe recall >= target AND zero whole-run deletion."""

    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    if int(y.sum()) == 0 or not runs:
        return None
    candidates: list[dict[str, Any]] = []
    for tau in threshold_candidates(p):
        keep = p >= tau
        counts = confusion_counts(y, keep)
        metrics = metrics_from_counts(counts)
        if metrics["recall"] is None or metrics["recall"] + 1e-12 < min_recall:
            continue
        blocked = False
        for run in runs:
            scored = scored_by_video.get(run.video_id)
            if scored is None:
                blocked = True
                break
            if not any(scored.p_keep[index] >= tau for index in run.frame_indices):
                blocked = True
                break
        if blocked:
            continue
        pruning = float(int((~keep).sum()) / len(y)) if len(y) else 0.0
        candidates.append(
            {
                "tau": float(tau),
                "recall": float(metrics["recall"]),
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "pruning_rate": pruning,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row["pruning_rate"], -row["recall"], -row["tau"]))
    best = candidates[0]
    best["status"] = "FOUND"
    best["constraint"] = "recall>=target and whole_positive_run_deletion==0"
    return best


def concat_universe(
    scored_list: Sequence[ScoredVideo],
    *,
    reference_threshold: float = PRIMARY_BINARY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (y_bin, p_keep) arrays restricted to the supervised universe."""

    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for scored in scored_list:
        mask = universe_mask(scored.frames.loss_mask)
        reference = binary_reference(
            scored.frames.target, scored.frames.loss_mask, reference_threshold
        )
        y_parts.append(reference[mask])
        p_parts.append(scored.p_keep[mask])
    if not y_parts:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    return np.concatenate(y_parts).astype(bool), np.concatenate(p_parts).astype(np.float64)

# ---------------------------------------------------------------------------
# auditing: provenance, event GT status, native missingness, per-video rows
# ---------------------------------------------------------------------------


def _load_clips(annotations_root: Path, annotation_identity: str) -> tuple[Any, ...]:
    from .youtube_highlights import load_mturk_clips

    return load_mturk_clips(annotations_root / annotation_identity)


def audit_provenance(
    *,
    data_root: str | Path,
    annotations_root: str | Path,
    entries: Sequence[Any],
    manifest_records: Mapping[str, Mapping[str, Any]],
    splits: Sequence[str] = EVAL_SPLITS,
) -> dict[str, Any]:
    """Verify that stored targets are exactly the frozen MTurk soft projection."""

    root = Path(data_root).expanduser().resolve()
    annotations = Path(annotations_root).expanduser().resolve()
    videos: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for entry in entries:
        if entry.split not in splits:
            continue
        frames = load_video_frames(root, entry.split, entry.video_id, category=entry.category)
        clips = _load_clips(annotations, entry.annotation_identity)
        vote_sum = np.zeros(frames.length, dtype=np.float64)
        count = np.zeros(frames.length, dtype=np.int64)
        for clip in clips:
            member = (frames.source_frame_id >= clip.start_frame) & (
                frames.source_frame_id < clip.end_frame
            )
            vote_sum[member] += clip.vote
            count[member] += 1
        covered = count > 0
        recomputed = np.zeros(frames.length, dtype=np.float64)
        recomputed[covered] = (vote_sum[covered] / count[covered]) / 5.0
        target_delta = float(np.abs(frames.target - recomputed).max()) if frames.length else 0.0
        missed_calc = int(((~frames.loss_mask) & covered & (recomputed >= PRIMARY_BINARY_THRESHOLD)).sum())
        record = manifest_records.get(entry.video_id, {})
        missed_recorded = int(record.get("missed_positive_frames", -1))
        loss_outside_covered = int((frames.loss_mask & ~covered).sum())
        ok = (
            target_delta <= 1e-5
            and missed_calc == missed_recorded
            and loss_outside_covered == 0
        )
        row = {
            "video_id": entry.video_id,
            "split": entry.split,
            "category": entry.category,
            "frames": frames.length,
            "clips": len(clips),
            "covered_frames": int(covered.sum()),
            "covered_fraction": _rounded(float(covered.mean()) if frames.length else 0.0),
            "supervised_frames": int(frames.loss_mask.sum()),
            "positive_frames_ge_0.5": int(
                ((frames.target >= PRIMARY_BINARY_THRESHOLD) & frames.loss_mask).sum()
            ),
            "target_max_abs_delta": target_delta,
            "missed_positive_calc": missed_calc,
            "missed_positive_recorded": missed_recorded,
            "loss_mask_outside_coverage": loss_outside_covered,
            "provenance_ok": bool(ok),
            "clips_are_2s_windows": True,
        }
        videos.append(row)
        if not ok:
            mismatches.append(row)
    return {
        "schema": PROVENANCE_SCHEMA,
        "status": "PASS" if not mismatches else "FAIL",
        "splits": list(splits),
        "video_count": len(videos),
        "mismatch_count": len(mismatches),
        "label_semantics": {
            "target": "mean(mturk soft vote of containing clips) / 5.0, clipped [0,1]",
            "binary_reference_primary": PRIMARY_BINARY_THRESHOLD,
            "binary_reference_sensitivity": list(SENSITIVITY_THRESHOLDS),
            "universe": "loss_mask == 1 (candidate frames covered by MTurk clip windows)",
            "loss_mask": "candidate interval AND covered by >= 1 MTurk clip window",
        },
        "videos": videos,
        "mismatches": mismatches,
    }


def audit_event_gt(
    frames_by_split: Mapping[str, Sequence[VideoFrames]],
    *,
    annotations_root: str | Path,
    entries_by_id: Mapping[str, Any],
    vote_fraction: float = PRIMARY_BINARY_THRESHOLD,
) -> dict[str, Any]:
    """Establish that human event identities do not exist; build derived run proxy."""

    annotations = Path(annotations_root).expanduser().resolve()
    runs_by_split: dict[str, list[PositiveRun]] = {}
    for split, frames_list in frames_by_split.items():
        runs: list[PositiveRun] = []
        for frames in frames_list:
            entry = entries_by_id[frames.video_id]
            clips = _load_clips(annotations, entry.annotation_identity)
            coverage = positive_clip_coverage(
                clips, frames.source_frame_id, vote_fraction=vote_fraction
            )
            runs.extend(extract_positive_runs(frames, coverage))
        runs_by_split[split] = runs
    stratum_counts = {split: {"short": 0, "medium": 0, "long": 0} for split in runs_by_split}
    for split, runs in runs_by_split.items():
        for run in runs:
            stratum_counts[split][run.stratum()] += 1
    return {
        "schema": EVENT_GT_SCHEMA,
        "EVENT_METRICS_AVAILABLE": "NO",
        "reason": (
            "The upstream YouTube Highlights annotations are ~2s MTurk clip windows with soft "
            "votes. They do not identify contiguous human highlight events, so event recall / "
            "whole-event deletion cannot be computed without fabricating event identities. "
            "Only a derived positive-run proxy is reported."
        ),
        "derived_proxy_definition": {
            "name": "POSITIVE_RUN (DERIVED PROXY, NOT HUMAN EVENT GT)",
            "coverage": f"sampled frames covered by a clip with vote/5 >= {vote_fraction}",
            "merge_rule": "maximal adjacent sampled frames with coverage and adjacency_mask",
            "duration_sec": "last_ts - first_ts + 0.5 sampling period",
            "strata": {
                "short": f"<= {SHORT_RUN_MAX_SEC}s",
                "medium": f"({SHORT_RUN_MAX_SEC}, {MEDIUM_RUN_MAX_SEC}]s",
                "long": f"> {MEDIUM_RUN_MAX_SEC}s",
            },
        },
        "run_counts": {
            split: {"total": len(runs), **stratum_counts[split]}
            for split, runs in runs_by_split.items()
        },
    }


def native_missingness_audit(
    scored_list: Sequence[ScoredVideo],
    *,
    reference_threshold: float = PRIMARY_BINARY_THRESHOLD,
) -> dict[str, Any]:
    """P(observed | Y=1) vs P(observed | Y=0) per native field, universe-only."""

    rows: list[dict[str, Any]] = []
    positives_parts: list[np.ndarray] = []
    missing_parts: list[np.ndarray] = []
    for scored in scored_list:
        mask = universe_mask(scored.frames.loss_mask)
        reference = binary_reference(
            scored.frames.target, scored.frames.loss_mask, reference_threshold
        )
        positives_parts.append(reference[mask])
        missing_parts.append(scored.frames.native_missing[mask])
    positives = (
        np.concatenate(positives_parts) if positives_parts else np.zeros(0, dtype=bool)
    )
    missing = (
        np.concatenate(missing_parts, axis=0)
        if missing_parts
        else np.zeros((0, NATIVE_DIM), dtype=bool)
    )
    observed = ~missing
    for index, name in enumerate(NATIVE_FIELDS):
        pos = positives
        neg = ~positives
        p_obs_pos = float(observed[pos, index].mean()) if int(pos.sum()) else None
        p_obs_neg = float(observed[neg, index].mean()) if int(neg.sum()) else None
        delta = None if (p_obs_pos is None or p_obs_neg is None) else p_obs_pos - p_obs_neg
        rows.append(
            {
                "field_index": index,
                "field": name,
                "observed_rate_overall": _rounded(float(observed[:, index].mean())),
                "p_observed_given_y1": _rounded(p_obs_pos),
                "p_observed_given_y0": _rounded(p_obs_neg),
                "delta": _rounded(delta),
                "risk_flag": (
                    "POTENTIAL_SHORTCUT_RISK"
                    if delta is not None and abs(delta) >= MISSINGNESS_FLAG_DELTA
                    else ""
                ),
            }
        )
    flagged = [row["field"] for row in rows if row["risk_flag"]]
    return {
        "reference_threshold": reference_threshold,
        "universe_frames": int(positives.size),
        "flag_delta": MISSINGNESS_FLAG_DELTA,
        "rows": rows,
        "flagged_fields": flagged,
        "note": (
            "A missingness/Y association is a statistical audit only; it does not prove the "
            "model used the missingness pattern as a shortcut."
        ),
    }


def per_video_metrics(
    scored_list: Sequence[ScoredVideo],
    *,
    tau_by_method: Mapping[str, float | None],
    reference_threshold: float = PRIMARY_BINARY_THRESHOLD,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = {"fs0": 0.0}
    methods.update({name: tau for name, tau in tau_by_method.items() if tau is not None})
    for scored in scored_list:
        frames = scored.frames
        mask = universe_mask(frames.loss_mask)
        y = binary_reference(frames.target, frames.loss_mask, reference_threshold)
        universe = int(mask.sum())
        for method, tau in methods.items():
            keep = (scored.p_keep >= tau) & mask
            counts = confusion_counts(y, keep)
            metrics = metrics_from_counts(counts)
            dropped = int((mask & (scored.p_keep < tau)).sum())
            rows.append(
                {
                    "video_id": frames.video_id,
                    "split": frames.split,
                    "method": method,
                    "tau": tau,
                    "frames": frames.length,
                    "universe_frames": universe,
                    "positive_frames": int(y.sum()),
                    "predicted_keep": int(counts["tp"] + counts["fp"]),
                    "predicted_drop": int(counts["fn"] + counts["tn"]),
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "tn": counts["tn"],
                    "precision": _rounded(metrics["precision"]),
                    "recall": _rounded(metrics["recall"]),
                    "f1": _rounded(metrics["f1"]),
                    "pruning_rate": _rounded(dropped / universe if universe else None),
                    "false_deletion_rate": _rounded(metrics["false_deletion_rate"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# protocol freezing and identity snapshot
# ---------------------------------------------------------------------------


def build_protocol(
    *,
    checkpoint: Mapping[str, Any],
    control_checkpoint: Mapping[str, Any],
    data_root: str | Path,
    normalization_path: str | Path,
    index_path: str | Path,
    split_manifest_path: str | Path,
    annotations_root: str | Path,
    seed: int,
    device: str,
    code_git_head: str,
) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "FROZEN_BEFORE_METRICS",
        "frozen_at_utc": None,
        "seed": int(seed),
        "device": device,
        "code_git_head": code_git_head,
        "checkpoint": dict(checkpoint),
        "overfitting_control_checkpoint": dict(control_checkpoint),
        "dataset": {
            "data_root": str(Path(data_root).expanduser().resolve()),
            "normalization_path": str(Path(normalization_path).expanduser().resolve()),
            "normalization_sha256": sha256_file(normalization_path),
            "index_path": str(Path(index_path).expanduser().resolve()),
            "index_sha256": sha256_file(index_path),
            "split_manifest_path": str(Path(split_manifest_path).expanduser().resolve()),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "annotations_root": str(Path(annotations_root).expanduser().resolve()),
            "splits": {"TRAIN": 111, "VALIDATION": 23, "CALIBRATION": 20, "TOTAL": 154},
        },
        "split_responsibilities": {
            "TRAIN": "never read for thresholds or metrics",
            "CALIBRATION": "threshold / operating-point selection only",
            "VALIDATION": "development evaluation only; already used to select the best epoch",
        },
        "universe": {
            "definition": "loss_mask == 1 (candidate frames covered by MTurk clip windows)",
            "reason": (
                "The full Qwen candidate mask is not persisted for candidate frames outside MTurk "
                "clip coverage, so frame-level ground truth exists exactly on loss_mask == 1."
            ),
        },
        "binary_reference": {
            "primary": {
                "name": "soft_target_majority",
                "threshold": PRIMARY_BINARY_THRESHOLD,
                "rule": "positive iff soft target >= 0.5 (mean annotator vote >= 2.5/5)",
            },
            "sensitivity": [
                {"name": "soft_target_0.4", "threshold": 0.4},
                {"name": "soft_target_0.6", "threshold": 0.6},
                {"name": "any_human_annotation", "threshold": 0.0, "rule": "soft target > 0"},
            ],
        },
        "metrics": {
            "primary_ranking": "average precision (step-wise, tie-aware) on VALIDATION",
            "operating_points": ["fs0", "tau_f1", "tau_safe", "tau_run_safe"],
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "f1": "2PR / (P + R)",
            "pruning_rate": "DROP frames / supervised universe frames",
            "false_deletion_rate": "FN / (TP + FN) = 1 - recall",
            "drop_rule": "DROP iff p_keep < tau",
        },
        "threshold_selection": {
            "method": "exact search over observed p_keep values on CALIBRATION",
            "tau_f1": "max F1; tie-break higher recall, then lower tau",
            "tau_safe": (
                f"max pruning subject to recall >= {RECALL_SAFETY_TARGET}; "
                "tie-break higher recall, then higher tau"
            ),
            "tau_run_safe": (
                "derived-proxy constraint: recall >= target and zero whole-positive-run deletion; "
                "max pruning"
            ),
            "cannot_relax_after_results": True,
        },
        "positive_run_proxy": {
            "EVENT_METRICS_AVAILABLE": "NO",
            "name": "POSITIVE_RUN (DERIVED PROXY, NOT HUMAN EVENT GT)",
            "short_run_sec": SHORT_RUN_MAX_SEC,
            "medium_run_sec": MEDIUM_RUN_MAX_SEC,
        },
        "verdict_rules": {
            "order": ["A", "D", "C", "B"],
            "A_effective_pruning": (
                f"tau_safe exists with validation recall >= {RECALL_SAFETY_TARGET}, "
                f"pruning >= {MIN_MEANINGFUL_PRUNING}, whole-run deletion == 0"
            ),
            "D_negative_result": (
                f"tau_f1 validation recall < {DANGEROUS_RECALL} and pruning >= {FS0_NEAR_ZERO_PRUNING}"
            ),
            "C_close_to_fs0": f"tau_f1 validation pruning < {FS0_NEAR_ZERO_PRUNING}",
            "B_signal_with_risk": "otherwise",
        },
        "uncertainty": {
            "bootstrap": False,
            "reason": "deterministic point metrics; positive frame count is small and reported explicitly",
        },
        "constraints": {
            "no_retraining": True,
            "no_finetuning": True,
            "no_materialization": True,
            "no_qwen": True,
            "no_rtdetr": True,
            "official_test": "NOT ACCESSED",
            "tvsum": "NOT ACCESSED",
            "checkpoint_read_only": True,
        },
    }


def environment_snapshot(*, device: str, seed: int) -> dict[str, Any]:
    import platform
    import sys

    import matplotlib
    import torch

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": seed,
    }
    return payload


__all__ = [
    "DANGEROUS_RECALL",
    "EVAL_SPLITS",
    "EVENT_GT_SCHEMA",
    "EvaluationError",
    "FS0_NEAR_ZERO_PRUNING",
    "MEDIUM_RUN_MAX_SEC",
    "MIN_MEANINGFUL_PRUNING",
    "PRIMARY_BINARY_THRESHOLD",
    "PROTOCOL_SCHEMA",
    "PositiveRun",
    "RECALL_SAFETY_TARGET",
    "SENSITIVITY_THRESHOLDS",
    "SHORT_RUN_MAX_SEC",
    "SUMMARY_SCHEMA",
    "ScoredVideo",
    "VideoFrames",
    "audit_event_gt",
    "audit_provenance",
    "average_precision",
    "binary_reference",
    "build_protocol",
    "concat_universe",
    "confusion_counts",
    "environment_snapshot",
    "extract_positive_runs",
    "load_checkpoint_model",
    "load_video_frames",
    "metrics_from_counts",
    "native_missingness_audit",
    "operating_metrics",
    "per_video_metrics",
    "positive_clip_coverage",
    "pr_curve",
    "resolve_device",
    "run_metrics",
    "score_split",
    "score_video",
    "select_tau_f1",
    "select_tau_run_safe",
    "select_tau_safe",
    "sha256_file",
    "threshold_candidates",
    "universe_mask",
    "write_csv",
    "write_json",
    "write_jsonl",
]
