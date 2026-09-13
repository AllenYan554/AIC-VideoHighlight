"""Matplotlib figures built only from real run artifacts (no fabricated data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WEAK_REFERENCE_NOTE = "weak-reference official-like metric (NOT_OFFICIAL_SCORE)"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def prediction_distribution(counts: Mapping[str, int], path: Path) -> None:
    values = sorted(int(v) for v in counts.values())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if values:
        bins = min(40, max(5, len(set(values))))
        ax.hist(values, bins=bins, color="#3b6ea5", edgecolor="white")
    ax.set_title("Per-video prediction frame count")
    ax.set_xlabel("predicted frames per video")
    ax.set_ylabel("videos")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def runtime_distribution(values: Sequence[float], path: Path) -> None:
    values = [float(v) for v in values]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if values:
        bins = min(40, max(5, len(set(round(v, 1) for v in values))))
        ax.hist(values, bins=bins, color="#4f8f5a", edgecolor="white")
    ax.set_title("Per-video retrieval runtime")
    ax.set_xlabel("seconds")
    ax.set_ylabel("videos")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def qwen_call_distribution(latencies: Sequence[float], path: Path) -> None:
    values = [float(v) for v in latencies]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if values:
        bins = min(40, max(5, len(set(round(v, 1) for v in values))))
        ax.hist(values, bins=bins, color="#b07a3a", edgecolor="white")
    ax.set_title("Qwen call duration")
    ax.set_xlabel("seconds per chunk call")
    ax.set_ylabel("calls")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def score_summary(evaluation: Mapping[str, Any], path: Path) -> None:
    macro_p = float(evaluation.get("Official-like Weak-Reference Precision", 0.0))
    macro_r = float(evaluation.get("Official-like Weak-Reference Recall", 0.0))
    macro_f = float(evaluation.get("Official-like Weak-Reference F-score", 0.0))
    per_video_f = [float(item.get("f_score", 0.0)) for item in evaluation.get("per_video", [])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    bars = left.bar(["P", "R", "F"], [macro_p, macro_r, macro_f], color=["#3b6ea5", "#4f8f5a", "#b07a3a"])
    for bar, value in zip(bars, [macro_p, macro_r, macro_f]):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    left.set_ylim(0, max(0.1, 1.05 * max(macro_p, macro_r, macro_f)))
    left.set_title("Macro metrics")
    left.grid(axis="y", alpha=0.25)
    if per_video_f:
        right.hist(per_video_f, bins=min(40, max(5, len(set(round(v, 3) for v in per_video_f)))), color="#6b5b95", edgecolor="white")
    right.set_title("Per-video F-score")
    right.set_xlabel("F-score")
    right.set_ylabel("videos")
    right.grid(axis="y", alpha=0.25)
    fig.suptitle(WEAK_REFERENCE_NOTE, fontsize=10)
    _save(fig, path)


def variant_score_comparison(evaluations: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    labels = list(evaluations)
    metrics = ["Precision", "Recall", "F-score"]
    keys = {
        "Precision": "Official-like Weak-Reference Precision",
        "Recall": "Official-like Weak-Reference Recall",
        "F-score": "Official-like Weak-Reference F-score",
    }
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for index, label in enumerate(labels):
        values = [float(evaluations[label].get(keys[metric], 0.0)) for metric in metrics]
        positions = [i + index * width for i in range(len(metrics))]
        ax.bar(positions, values, width=width, label=label)
    ax.set_xticks([i + width * (len(labels) - 1) / 2 for i in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Variant comparison: {WEAK_REFERENCE_NOTE}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def variant_prediction_comparison(
    stabilized_counts: Mapping[str, int], control_counts: Mapping[str, int], path: Path
) -> None:
    ids = sorted(set(stabilized_counts) & set(control_counts))
    left = [int(stabilized_counts[v]) for v in ids]
    right = [int(control_counts[v]) for v in ids]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if ids:
        ax.scatter(right, left, s=14, alpha=0.65, color="#3b6ea5")
        limit = max(max(left, default=0), max(right, default=1), 1)
        ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="#888888")
    ax.set_xlabel("control_no_stabilization frames")
    ax.set_ylabel("stabilized frames")
    ax.set_title("Prediction count per video (geometry only; no score)")
    ax.grid(alpha=0.25)
    _save(fig, path)


def bbox_change_distribution(delta_pairs: Sequence[tuple[int, int]], path: Path) -> None:
    dx = [pair[0] for pair in delta_pairs]
    dy = [pair[1] for pair in delta_pairs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, values, label in ((left, dx, "|delta x| px"), (right, dy, "|delta y| px")):
        if values:
            ax.hist(values, bins=min(50, max(5, len(set(values)))), color="#b07a3a", edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("pixels per common frame")
        ax.set_ylabel("frames")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("stabilized vs control_no_stabilization bbox change (common frames)", fontsize=10)
    _save(fig, path)
