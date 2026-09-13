"""CSV/markdown tables built only from real run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_SUMMARY_HEADER = [
    "run_id",
    "generated_at",
    "git_head",
    "profile",
    "scope",
    "runtime_profile",
    "attention_backend",
    "gqa_execution_mode",
    "video_count",
    "non_empty_videos",
    "empty_videos",
    "qwen_calls",
    "rtdetr_frames",
    "wall_time_sec",
    "retrieval_time_sec",
    "localization_time_sec",
    "qwen_peak_vram_mib",
    "rtdetr_peak_allocated_mib",
    "oom",
    "contract_valid",
    "official_score",
    "evaluation_labels",
]

PER_VIDEO_HEADER = [
    "video_id",
    "prediction_count",
    "prediction_count_control",
    "empty",
    "localized_frames",
    "retrieval_total_sec",
    "qwen_chunks",
    "retrieval_success",
]

RUNTIME_HEADER = [
    "video_id",
    "chunk_index",
    "latency_sec",
    "response_chars",
    "finish_reason",
    "experiment_clip_duration_sec",
]

COMPARISON_HEADER = [
    "video_id",
    "frame_count",
    "common_frames",
    "bbox_exact_matches",
    "bbox_exact_rate",
    "mean_abs_delta_x",
    "mean_abs_delta_y",
]


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prediction_counts(lines: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(line["video_id"]): len(line.get("predictions", [])) for line in lines}


def localization_frame_counts(policy_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not policy_dir.is_dir():
        return counts
    for path in sorted(policy_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, list):
            counts[path.stem] = len(payload)
    return counts


def build_run_summary(
    *,
    run_id: str,
    generated_at: str,
    result: Mapping[str, Any],
    evaluation_labels: Sequence[str],
) -> dict[str, Any]:
    identity = result.get("identity", {})
    resources = result.get("resources", {})
    variants = result.get("variants", {})
    stabilized = variants.get("stabilized", {}) if isinstance(variants, Mapping) else {}
    processes = resources.get("processes", [])
    retrieval_wall = next(
        (float(p.get("wall_sec", 0.0)) for p in processes if p.get("label") == "Retrieval Qwen fresh inference"),
        None,
    )
    localization_wall = next(
        (float(p.get("wall_sec", 0.0)) for p in processes if p.get("label") == "Subject localization RT-DETR"),
        None,
    )
    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "git_head": identity.get("git_head"),
        "profile": result.get("profile"),
        "scope": result.get("scope", "unspecified"),
        "runtime_profile": result.get("runtime_profile", "default"),
        "attention_backend": result.get(
            "attention_backend", "transformers_sdpa_auto"
        ),
        "gqa_execution_mode": result.get(
            "gqa_execution_mode", "transformers_default"
        ),
        "video_count": result.get("video_count"),
        "non_empty_videos": int(result.get("video_count", 0)) - int(result.get("empty_video_count", 0)),
        "empty_videos": result.get("empty_video_count"),
        "qwen_calls": result.get("qwen_calls"),
        "rtdetr_frames": result.get("rtdetr_calls"),
        "wall_time_sec": round(float(resources.get("total_wall_sec", 0.0)), 1),
        "retrieval_time_sec": round(retrieval_wall, 1) if retrieval_wall is not None else "",
        "localization_time_sec": round(localization_wall, 1) if localization_wall is not None else "",
        "qwen_peak_vram_mib": resources.get("qwen", {}).get("peak_vram_mib"),
        "rtdetr_peak_allocated_mib": resources.get("rtdetr", {}).get("peak_allocated_mib"),
        "oom": resources.get("oom"),
        "contract_valid": stabilized.get("contract", {}).get("is_valid"),
        "official_score": " / ".join(
            str(result.get("official_score", {}).get(key, ""))
            for key in ("status", "reason")
        ).strip(" /"),
        "evaluation_labels": ",".join(evaluation_labels),
    }


def build_per_video_rows(
    *,
    video_ids: Sequence[str],
    stabilized_counts: Mapping[str, int],
    control_counts: Mapping[str, int] | None,
    empty_ids: Sequence[str],
    localized_counts: Mapping[str, int],
    samples: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    empty_set = {str(value) for value in empty_ids}
    sample_by_id = {str(sample.get("video_id")): sample for sample in samples}
    rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        sample = sample_by_id.get(video_id, {})
        timing = sample.get("timing", {}) if isinstance(sample, Mapping) else {}
        row: dict[str, Any] = {
            "video_id": video_id,
            "prediction_count": stabilized_counts.get(video_id, 0),
            "prediction_count_control": None if control_counts is None else control_counts.get(video_id, 0),
            "empty": video_id in empty_set,
            "localized_frames": localized_counts.get(video_id, 0),
            "retrieval_total_sec": round(float(timing.get("total_sec", 0.0)), 2) if timing else None,
            "qwen_chunks": len(sample.get("raw_chunk_outputs", [])) if sample else None,
            "retrieval_success": sample.get("success") if sample else None,
        }
        for label, evaluation in evaluations.items():
            per_video = next(
                (item for item in evaluation.get("per_video", []) if str(item.get("video_id")) == video_id),
                None,
            )
            if per_video is None:
                continue
            row[f"{label}_precision"] = per_video.get("precision")
            row[f"{label}_recall"] = per_video.get("recall")
            row[f"{label}_f_score"] = per_video.get("f_score")
            row[f"{label}_sum_iou"] = per_video.get("sum_spatial_iou")
            row[f"{label}_matched_frames"] = per_video.get("exact_common_frames")
            matched = int(per_video.get("exact_common_frames", 0))
            row[f"{label}_mean_matched_iou"] = (
                float(per_video.get("sum_spatial_iou", 0.0)) / matched if matched else 0.0
            )
        rows.append(row)
    return rows


def build_runtime_rows(
    raw_calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in raw_calls:
        video_id = str(payload.get("video_id"))
        clip_duration = payload.get("experiment_clip_duration_sec")
        for chunk in payload.get("chunks", []):
            rows.append(
                {
                    "video_id": video_id,
                    "chunk_index": chunk.get("chunk_index"),
                    "latency_sec": round(float(chunk.get("request_latency_sec", 0.0)), 2),
                    "response_chars": len(str(chunk.get("raw_response", ""))),
                    "finish_reason": chunk.get("finish_reason"),
                    "experiment_clip_duration_sec": clip_duration,
                }
            )
    return rows


def build_comparison_rows(
    stabilized_lines: Sequence[Mapping[str, Any]],
    control_lines: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stabilized = {
        str(line["video_id"]): {
            int(item["frame"]): tuple(int(v) for v in item["bboxes"])
            for item in line.get("predictions", [])
        }
        for line in stabilized_lines
    }
    control = {
        str(line["video_id"]): {
            int(item["frame"]): tuple(int(v) for v in item["bboxes"])
            for item in line.get("predictions", [])
        }
        for line in control_lines
    }
    rows: list[dict[str, Any]] = []
    for video_id in sorted(set(stabilized) | set(control)):
        left = control.get(video_id, {})
        right = stabilized.get(video_id, {})
        common = sorted(set(left) & set(right))
        exact = sum(1 for frame in common if left[frame] == right[frame])
        deltas_x = [abs(left[frame][0] - right[frame][0]) for frame in common]
        deltas_y = [abs(left[frame][1] - right[frame][1]) for frame in common]
        rows.append(
            {
                "video_id": video_id,
                "frame_count": len(right),
                "common_frames": len(common),
                "bbox_exact_matches": exact,
                "bbox_exact_rate": (exact / len(common)) if common else None,
                "mean_abs_delta_x": (sum(deltas_x) / len(deltas_x)) if deltas_x else None,
                "mean_abs_delta_y": (sum(deltas_y) / len(deltas_y)) if deltas_y else None,
            }
        )
    return rows


def overall_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    common = sum(int(row["common_frames"]) for row in rows)
    exact = sum(int(row["bbox_exact_matches"]) for row in rows)
    dx_values = [row["mean_abs_delta_x"] for row in rows if row["mean_abs_delta_x"] is not None]
    dy_values = [row["mean_abs_delta_y"] for row in rows if row["mean_abs_delta_y"] is not None]
    return {
        "video_id": "__overall__",
        "frame_count": sum(int(row["frame_count"]) for row in rows),
        "common_frames": common,
        "bbox_exact_matches": exact,
        "bbox_exact_rate": (exact / common) if common else None,
        "mean_abs_delta_x": (sum(dx_values) / len(dx_values)) if dx_values else None,
        "mean_abs_delta_y": (sum(dy_values) / len(dy_values)) if dy_values else None,
    }
