"""Reproducible experiment outputs for the Baseline 20 development set."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .temporal_metrics import ZERO_DURATION_EPSILON_SEC, duration_based_metrics


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_run_config(
    path: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Protect the original experiment snapshot unless overwrite is explicit."""
    if path.is_file():
        if resume:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("existing run_config.json must contain a JSON object")
            return existing
        if not overwrite:
            raise FileExistsError(
                "run_config.json already exists; use --resume or --overwrite"
            )
    atomic_write_json(path, payload)
    return payload


def load_baseline_samples(
    manifest_path: str | Path,
    video_root: str | Path,
    *,
    video_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    root = Path(video_root).expanduser().resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("baseline manifest must be a JSON array")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    selected: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    seen_filenames: set[str] = set()
    for index, raw_item in enumerate(payload, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        item = dict(raw_item)
        item_video_id = item.get("video_id")
        filename = item.get("filename")
        if not isinstance(item_video_id, str) or not item_video_id:
            raise ValueError(f"manifest item {index} has an invalid video_id")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"manifest item {index} has an invalid filename")
        if item.get("dataset_split") != "train":
            raise ValueError(f"manifest item {index} is not in the train split")
        if item_video_id in seen_video_ids or filename in seen_filenames:
            raise ValueError(f"manifest contains a duplicate item: {item_video_id}")
        seen_video_ids.add(item_video_id)
        seen_filenames.add(filename)
        if video_id is not None and item_video_id != video_id:
            continue

        clip_start = float(item["clip_start_sec"])
        clip_end = float(item["clip_end_sec"])
        if clip_start < 0 or clip_end <= clip_start:
            raise ValueError(f"invalid clip bounds for {item_video_id}")
        references = item.get("weak_reference_segments")
        if not isinstance(references, list):
            raise ValueError(f"weak_reference_segments must be an array for {item_video_id}")
        clip_duration = clip_end - clip_start
        for segment in references:
            start = float(segment["start_sec"])
            end = float(segment["end_sec"])
            if start < 0 or end < start or end > clip_duration + ZERO_DURATION_EPSILON_SEC:
                raise ValueError(f"weak reference is outside clip-local bounds for {item_video_id}")

        item["source_video_path"] = str((root / filename).resolve())
        item["clip_duration_sec"] = clip_duration
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break

    if video_id is not None and not selected:
        raise ValueError(f"video_id is not present in manifest: {video_id}")
    return selected


def resume_decision(
    result_path: Path,
    *,
    raw_path: Path,
    resume: bool,
    overwrite: bool = False,
) -> str:
    if overwrite:
        return "run_model"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result = None
        if isinstance(result, dict) and result.get("success") is True:
            return "skip" if resume else "blocked"
        if not resume:
            return "blocked"
    if resume and raw_path.is_file() and raw_path.stat().st_size > 0:
        return "recover_raw"
    return "run_model"


class RawOutputJournal:
    """Atomically persist every model response before downstream processing."""

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = path
        self.payload = {**metadata, "chunks": []}
        atomic_write_json(self.path, self.payload)

    def record(self, chunk_record: dict[str, Any]) -> None:
        snapshot = json.loads(json.dumps(chunk_record, ensure_ascii=False))
        chunks = self.payload["chunks"]
        for index, existing in enumerate(chunks):
            if existing["chunk_index"] == snapshot["chunk_index"]:
                chunks[index] = snapshot
                break
        else:
            chunks.append(snapshot)
        chunks.sort(key=lambda item: item["chunk_index"])
        atomic_write_json(self.path, self.payload)


def evaluate_weak_references(
    prediction_segments: Iterable[dict[str, Any]],
    weak_reference_segments: Iterable[dict[str, Any]],
    *,
    zero_duration_epsilon_sec: float = ZERO_DURATION_EPSILON_SEC,
) -> dict[str, float | int]:
    predicted = [
        (float(segment["start_sec"]), float(segment["end_sec"]))
        for segment in prediction_segments
    ]
    references = [
        (float(segment["start_sec"]), float(segment["end_sec"]))
        for segment in weak_reference_segments
    ]
    metrics = duration_based_metrics(
        predicted,
        references,
        zero_duration_epsilon_sec=zero_duration_epsilon_sec,
    )
    return {
        "weak_ref_precision": metrics["precision"],
        "weak_ref_recall": metrics["recall"],
        "weak_ref_f1": metrics["f1"],
        "temporal_iou": metrics["temporal_iou"],
        "intersection_sec": metrics["intersection_sec"],
        "prediction_duration_sec": metrics["predicted_sec"],
        "reference_duration_sec": metrics["reference_sec"],
        "duration_union_sec": metrics["union_sec"],
        "zero_duration_reference_count": metrics["zero_duration_reference_count"],
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(results)
    successful = [item for item in records if item.get("success") is True]
    metric_sources = {
        "weak_ref_precision": lambda item: item["metrics"]["weak_ref_precision"],
        "weak_ref_recall": lambda item: item["metrics"]["weak_ref_recall"],
        "weak_ref_f1": lambda item: item["metrics"]["weak_ref_f1"],
        "temporal_iou": lambda item: item["metrics"]["temporal_iou"],
        "inference_time_sec": lambda item: item["timing"]["model_inference_sec"],
    }
    summaries: dict[str, dict[str, float] | None] = {}
    for name, getter in metric_sources.items():
        values = [float(getter(item)) for item in successful]
        summaries[name] = (
            {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            if values
            else None
        )
    return {
        "processed": len(records),
        "success": len(successful),
        "failed": len(records) - len(successful),
        "metrics": summaries,
    }


def run_with_failure_isolation(
    items: Iterable[dict[str, Any]],
    processor: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            results.append(processor(item))
        except Exception as exc:
            results.append(
                {
                    "video_id": item.get("video_id"),
                    "group": item.get("group"),
                    "success": False,
                    "warnings": [],
                    "errors": [
                        {
                            "stage": stage,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                }
            )
    return results


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for experiment figures") from exc
    return plt


def write_timeline_figure(
    *,
    video_id: str,
    duration_sec: float,
    weak_reference_segments: list[dict[str, Any]],
    prediction_segments: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(10, 2.8))
    for segment in weak_reference_segments:
        start = float(segment["start_sec"])
        duration = float(segment["end_sec"]) - start
        if duration > 0:
            axis.broken_barh([(start, duration)], (1.25, 0.5), facecolors="tab:blue")
        else:
            axis.axvline(start, ymin=0.62, ymax=0.82, color="tab:blue", linewidth=2)
    for segment in prediction_segments:
        start = float(segment["start_sec"])
        duration = float(segment["end_sec"]) - start
        axis.broken_barh([(start, duration)], (0.25, 0.5), facecolors="tab:orange")
    axis.set_xlim(0, max(duration_sec, 1e-6))
    axis.set_ylim(0, 2)
    axis.set_yticks([0.5, 1.5], labels=["Prediction", "Weak reference"])
    axis.set_xlabel("Clip-local seconds")
    axis.set_title(
        f"{video_id} | F1={metrics['weak_ref_f1']:.3f} "
        f"R={metrics['weak_ref_recall']:.3f} P={metrics['weak_ref_precision']:.3f}"
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_aggregate_figures(results: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    successful = [item for item in results if item.get("success") is True]
    specifications = [
        ("weak_ref_recall", "Weak-reference recall", lambda item: item["metrics"]["weak_ref_recall"]),
        ("weak_ref_precision", "Weak-reference precision", lambda item: item["metrics"]["weak_ref_precision"]),
        ("weak_ref_f1", "Weak-reference F1", lambda item: item["metrics"]["weak_ref_f1"]),
        ("temporal_iou", "Duration-set temporal IoU", lambda item: item["metrics"]["temporal_iou"]),
        ("latency", "Model inference latency (seconds)", lambda item: item["timing"]["model_inference_sec"]),
    ]
    plt = _load_pyplot()
    outputs: list[Path] = []
    labels = [str(item.get("sample_index", item["video_id"])) for item in successful]
    for filename, title, getter in specifications:
        figure, axis = plt.subplots(figsize=(max(6, len(successful) * 0.55), 3.6))
        values = [float(getter(item)) for item in successful]
        axis.bar(labels, values, color="tab:blue")
        axis.set_xlabel("Sample index")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        output = figures_dir / f"{filename}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        plt.close(figure)
        outputs.append(output)
    return outputs


def write_summary_files(
    results: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    aggregate = aggregate_results(results)
    atomic_write_json(output_dir / "metrics.json", aggregate)

    fields = [
        "video_id",
        "group",
        "success",
        "segment_count_ref",
        "segment_count_pred",
        "weak_ref_precision",
        "weak_ref_recall",
        "weak_ref_f1",
        "temporal_iou",
        "inference_time_sec",
        "total_time_sec",
        "warning_count",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            metrics = item.get("metrics", {})
            timing = item.get("timing", {})
            writer.writerow(
                {
                    "video_id": item.get("video_id"),
                    "group": item.get("group"),
                    "success": item.get("success", False),
                    "segment_count_ref": len(item.get("weak_reference_segments", [])),
                    "segment_count_pred": len(item.get("merged_prediction_segments", [])),
                    "weak_ref_precision": metrics.get("weak_ref_precision", ""),
                    "weak_ref_recall": metrics.get("weak_ref_recall", ""),
                    "weak_ref_f1": metrics.get("weak_ref_f1", ""),
                    "temporal_iou": metrics.get("temporal_iou", ""),
                    "inference_time_sec": timing.get("model_inference_sec", ""),
                    "total_time_sec": timing.get("total_sec", ""),
                    "warning_count": len(item.get("warnings", [])),
                }
            )
    return aggregate
