"""Render tables, figures and a markdown report for one finished run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import figures as fig
from . import tables as tab

CONTROL_DEFINITION = (
    "control_no_stabilization shares retrieval, temporal refinement, frame projection, "
    "subject localization and spatial composition with the stabilized variant; the only "
    "difference is temporal stabilization (TS-5 Revised ts5 vs TS-0/ts0). This is bound in "
    "the run's shared_upstream_identity.json (fork_component=stabilization)."
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_samples(samples_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(samples_dir.glob("*.json"))] if samples_dir.is_dir() else []


def _read_raw(raw_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))] if raw_dir.is_dir() else []


def _frame_bboxes(lines: list[dict[str, Any]]) -> dict[str, dict[int, tuple[int, int, int]]]:
    result: dict[str, dict[int, tuple[int, int, int]]] = {}
    for line in lines:
        result[str(line["video_id"])] = {
            int(item["frame"]): tuple(int(v) for v in item["bboxes"])
            for item in line.get("predictions", [])
        }
    return result


def _delta_pairs(stabilized_lines: list[dict[str, Any]], control_lines: list[dict[str, Any]]) -> list[tuple[int, int]]:
    left = _frame_bboxes(control_lines)
    right = _frame_bboxes(stabilized_lines)
    pairs: list[tuple[int, int]] = []
    for video_id in set(left) & set(right):
        for frame in set(left[video_id]) & set(right[video_id]):
            pairs.append(
                (
                    abs(left[video_id][frame][0] - right[video_id][frame][0]),
                    abs(left[video_id][frame][1] - right[video_id][frame][1]),
                )
            )
    return pairs


def _md_table(header: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in header) + " |")
    return "\n".join(lines)


def render_run_report(
    run_dir: Path,
    *,
    evaluations: Mapping[str, Path] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    result_path = run_dir / "inference_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"run has no inference_result.json: {run_dir}")
    result = _read_json(result_path)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()

    evaluation_payloads: dict[str, dict[str, Any]] = {}
    for label, path in (evaluations or {}).items():
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"evaluation report does not exist: {candidate}")
        evaluation_payloads[str(label)] = _read_json(candidate)

    samples = _read_samples(run_dir / "retrieval" / "samples")
    raw_calls = _read_raw(run_dir / "retrieval" / "raw")
    stabilized = tab.read_jsonl(run_dir / "predictions" / "predictions.jsonl")
    if not stabilized:
        raise FileNotFoundError(f"run has no predictions: {run_dir / 'predictions' / 'predictions.jsonl'}")
    control_path = run_dir / "control_no_stabilization" / "predictions.jsonl"
    control = tab.read_jsonl(control_path) if control_path.is_file() else []

    stabilized_counts = tab.prediction_counts(stabilized)
    control_counts = tab.prediction_counts(control) if control else None
    localized_counts = tab.localization_frame_counts(run_dir / "localization" / "policy_v1")

    variant_blob = result.get("variants", {}).get("stabilized", {})
    video_ids = [str(value) for value in result.get("video_ids", [])]
    if not video_ids:
        video_ids = sorted(stabilized_counts)

    summary = tab.build_run_summary(
        run_id=run_dir.name,
        generated_at=timestamp,
        result=result,
        evaluation_labels=list(evaluation_payloads),
    )
    per_video = tab.build_per_video_rows(
        video_ids=video_ids,
        stabilized_counts=stabilized_counts,
        control_counts=control_counts,
        empty_ids=result.get("empty_video_ids", []),
        localized_counts=localized_counts,
        samples=samples,
        evaluations=evaluation_payloads,
    )
    runtime_rows = tab.build_runtime_rows(raw_calls)
    comparison_rows = tab.build_comparison_rows(stabilized, control) if control else []
    if comparison_rows:
        comparison_rows.append(tab.overall_comparison(comparison_rows))

    per_video_header = list(tab.PER_VIDEO_HEADER)
    for label in evaluation_payloads:
        per_video_header.extend(
            [
                f"{label}_precision",
                f"{label}_recall",
                f"{label}_f_score",
                f"{label}_sum_iou",
                f"{label}_matched_frames",
                f"{label}_mean_matched_iou",
            ]
        )

    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    report_dir = run_dir / "report"
    tab.write_csv(tables_dir / "summary.csv", tab.RUN_SUMMARY_HEADER, [summary])
    tab.write_csv(tables_dir / "per_video.csv", per_video_header, per_video)
    tab.write_csv(tables_dir / "runtime.csv", tab.RUNTIME_HEADER, runtime_rows)
    if comparison_rows:
        tab.write_csv(tables_dir / "comparison.csv", tab.COMPARISON_HEADER, comparison_rows)

    fig.prediction_distribution(stabilized_counts, figures_dir / "prediction_distribution.png")
    fig.runtime_distribution(
        [row["retrieval_total_sec"] for row in per_video if row["retrieval_total_sec"] is not None],
        figures_dir / "runtime_distribution.png",
    )
    fig.qwen_call_distribution(
        [row["latency_sec"] for row in runtime_rows],
        figures_dir / "qwen_call_distribution.png",
    )
    if evaluation_payloads:
        first_label = next(iter(evaluation_payloads))
        fig.score_summary(evaluation_payloads[first_label], figures_dir / "score_summary.png")
    if len(evaluation_payloads) >= 2:
        fig.variant_score_comparison(evaluation_payloads, figures_dir / "score_comparison.png")
    if control_counts is not None:
        fig.variant_prediction_comparison(
            stabilized_counts, control_counts, figures_dir / "variant_prediction_comparison.png"
        )
        fig.bbox_change_distribution(_delta_pairs(stabilized, control), figures_dir / "bbox_change_distribution.png")

    macro_rows = [
        {
            "variant": label,
            "Precision": evaluation.get("Official-like Weak-Reference Precision"),
            "Recall": evaluation.get("Official-like Weak-Reference Recall"),
            "F_score": evaluation.get("Official-like Weak-Reference F-score"),
            "score_identity": evaluation.get("score_identity"),
        }
        for label, evaluation in evaluation_payloads.items()
    ]

    report_lines = [
        "# VHiCraft Run Report",
        "",
        f"- run_id: `{run_dir.name}`",
        f"- generated_at: `{timestamp}`",
        f"- git_head: `{summary['git_head']}`",
        f"- profile: `{summary['profile']}`",
        f"- scope: `{summary['scope']}`",
        f"- runtime_profile: `{summary['runtime_profile']}`",
        f"- attention_backend: `{summary['attention_backend']}`",
        f"- gqa_execution_mode: `{summary['gqa_execution_mode']}`",
        f"- videos: `{summary['video_count']}` (empty `{summary['empty_videos']}`)",
        f"- Official score: `{summary['official_score'] or 'not reported'}`",
        "",
        "## Summary",
        "",
        _md_table(tab.RUN_SUMMARY_HEADER, [summary]),
        "",
    ]
    if macro_rows:
        report_lines.extend(
            [
                "## Evaluation",
                "",
                "Weak-reference official-like metrics; identity is `NOT_OFFICIAL_SCORE` and these are",
                "not official competition scores.",
                "",
                _md_table(["variant", "Precision", "Recall", "F_score", "score_identity"], macro_rows),
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "## Evaluation",
                "",
                "No reference evaluation was provided for this run; no score is reported.",
                "",
            ]
        )
    report_lines.extend(["## Variant comparison", ""])
    if control:
        report_lines.extend(
            [
                CONTROL_DEFINITION,
                "",
                f"- common frames: `{comparison_rows[-1]['common_frames']}`",
                f"- exact bbox matches: `{comparison_rows[-1]['bbox_exact_matches']}` "
                f"(`{comparison_rows[-1]['bbox_exact_rate']}` rate)",
                f"- mean |dx| = `{comparison_rows[-1]['mean_abs_delta_x']}`, "
                f"mean |dy| = `{comparison_rows[-1]['mean_abs_delta_y']}` px",
                "",
            ]
        )
    else:
        report_lines.extend(["No control_no_stabilization predictions in this run.", ""])
    report_lines.extend(["## Tables", ""])
    for name in sorted(path.name for path in tables_dir.glob("*.csv")):
        report_lines.append(f"- `tables/{name}`")
    report_lines.extend(["", "## Figures", ""])
    for name in sorted(path.name for path in figures_dir.glob("*.png")):
        report_lines.append(f"- `figures/{name}`")
    report_lines.append("")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "status": "PASS",
        "run_id": run_dir.name,
        "tables": sorted(path.name for path in tables_dir.glob("*.csv")),
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
        "report": str(report_dir / "run_report.md"),
        "evaluation_labels": list(evaluation_payloads),
        "variant_blob_keys": sorted(variant_blob) if isinstance(variant_blob, Mapping) else [],
    }
