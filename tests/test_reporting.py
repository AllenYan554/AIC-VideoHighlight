"""Reporting tests: real tables/figures/markdown from synthetic artifacts only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.reporting import render_run_report  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "vhicraft_smoke_202601010000"
    (run / "predictions").mkdir(parents=True)
    (run / "control_no_stabilization").mkdir(parents=True)
    (run / "retrieval" / "samples").mkdir(parents=True)
    (run / "retrieval" / "raw").mkdir(parents=True)
    (run / "localization" / "policy_v1").mkdir(parents=True)

    stabilized = [
        {"video_id": "v1", "targetRatioWH": [9, 16], "predictions": [
            {"frame": 0, "bboxes": [10, 0, 100]}, {"frame": 1, "bboxes": [12, 0, 100]},
        ]},
        {"video_id": "v2", "targetRatioWH": [9, 16], "predictions": []},
    ]
    control = [
        {"video_id": "v1", "targetRatioWH": [9, 16], "predictions": [
            {"frame": 0, "bboxes": [10, 0, 100]}, {"frame": 1, "bboxes": [13, 1, 100]},
        ]},
        {"video_id": "v2", "targetRatioWH": [9, 16], "predictions": []},
    ]
    _write_jsonl(run / "predictions" / "predictions.jsonl", stabilized)
    _write_jsonl(run / "control_no_stabilization" / "predictions.jsonl", control)
    _write_json(run / "inference_result.json", {
        "status": "PASS",
        "profile": "smoke",
        "runtime_profile": "local_efficient_sdpa",
        "attention_backend": "pytorch_efficient_attention",
        "gqa_execution_mode": "temporary_kv_head_expansion",
        "video_ids": ["v1", "v2"],
        "video_count": 2,
        "qwen_calls": 1,
        "rtdetr_calls": 2,
        "empty_video_count": 1,
        "empty_video_ids": ["v2"],
        "identity": {"git_head": "deadbeef"},
        "resources": {
            "processes": [
                {"label": "Retrieval Qwen fresh inference", "wall_sec": 5.0},
                {"label": "Subject localization RT-DETR", "wall_sec": 2.0},
            ],
            "qwen": {"peak_vram_mib": 1000.0},
            "rtdetr": {"peak_allocated_mib": 300.0},
            "total_wall_sec": 8.0,
            "oom": False,
        },
        "variants": {"stabilized": {"contract": {"is_valid": True}}},
    })
    _write_json(run / "retrieval" / "samples" / "v1.json", {
        "video_id": "v1",
        "success": True,
        "timing": {"total_sec": 5.0},
        "raw_chunk_outputs": [{"chunk_index": 0}],
    })
    _write_json(run / "retrieval" / "samples" / "v2.json", {
        "video_id": "v2",
        "success": True,
        "timing": {"total_sec": 1.0},
        "raw_chunk_outputs": [{"chunk_index": 0}],
    })
    _write_json(run / "retrieval" / "raw" / "v1.json", {
        "video_id": "v1",
        "experiment_clip_duration_sec": 3.0,
        "chunks": [{
            "chunk_index": 0, "request_latency_sec": 2.5,
            "raw_response": "x" * 80, "finish_reason": "stop",
        }],
    })
    _write_json(run / "localization" / "policy_v1" / "v1.json", [
        {"video_id": "v1", "frame": 0}, {"video_id": "v1", "frame": 1},
    ])
    evaluation = {
        "score_identity": "NOT_OFFICIAL_SCORE",
        "Official-like Weak-Reference Precision": 0.25,
        "Official-like Weak-Reference Recall": 0.5,
        "Official-like Weak-Reference F-score": 1 / 3,
        "per_video": [
            {
                "video_id": "v1",
                "precision": 0.5, "recall": 1.0, "f_score": 2 / 3,
                "sum_spatial_iou": 1.0, "exact_common_frames": 1,
            },
            {
                "video_id": "v2",
                "precision": 0.0, "recall": 0.0, "f_score": 0.0,
                "sum_spatial_iou": 0.0, "exact_common_frames": 0,
            },
        ],
    }
    evaluation_path = tmp_path / "evaluation.json"
    _write_json(evaluation_path, evaluation)
    return run, evaluation_path


def test_render_run_report_with_evaluation(tmp_path):
    run, evaluation_path = _fixture(tmp_path)
    rendered = render_run_report(run, evaluations={"stabilized": evaluation_path})
    assert rendered["status"] == "PASS"
    for name in ("summary.csv", "per_video.csv", "runtime.csv", "comparison.csv"):
        assert (run / "tables" / name).is_file(), name
    for name in (
        "prediction_distribution.png",
        "runtime_distribution.png",
        "qwen_call_distribution.png",
        "score_summary.png",
        "variant_prediction_comparison.png",
        "bbox_change_distribution.png",
    ):
        assert (run / "figures" / name).is_file(), name
    assert (run / "report" / "run_report.md").is_file()

    with (run / "tables" / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["run_id"] == run.name
    assert summary["git_head"] == "deadbeef"
    assert summary["video_count"] == "2"
    assert summary["evaluation_labels"] == "stabilized"
    assert summary["runtime_profile"] == "local_efficient_sdpa"
    assert summary["attention_backend"] == "pytorch_efficient_attention"

    with (run / "tables" / "per_video.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["video_id"]: row for row in csv.DictReader(handle)}
    assert rows["v1"]["stabilized_f_score"] != ""
    assert rows["v2"]["empty"] == "True"

    with (run / "tables" / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        comparison = {row["video_id"]: row for row in csv.DictReader(handle)}
    assert comparison["v1"]["bbox_exact_matches"] == "1"
    assert "__overall__" in comparison

    report_text = (run / "report" / "run_report.md").read_text(encoding="utf-8")
    assert "NOT_OFFICIAL_SCORE" in report_text
    assert "control_no_stabilization shares retrieval" in report_text
    assert "runtime_profile: `local_efficient_sdpa`" in report_text
    assert "gqa_execution_mode: `temporary_kv_head_expansion`" in report_text


def test_no_reference_means_no_score(tmp_path):
    run, _ = _fixture(tmp_path)
    render_run_report(run)
    assert not (run / "figures" / "score_summary.png").exists()
    assert not (run / "figures" / "score_comparison.png").exists()
    report_text = (run / "report" / "run_report.md").read_text(encoding="utf-8")
    assert "no score is reported" in report_text


def test_official_test_report_declares_isolated_scope_and_hidden_gt(tmp_path):
    run, _ = _fixture(tmp_path)
    result_path = run / "inference_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["scope"] = "isolated_validation"
    result["official_score"] = {"status": "UNAVAILABLE", "reason": "HIDDEN_GT"}
    result_path.write_text(json.dumps(result), encoding="utf-8")

    render_run_report(run)

    with (run / "tables" / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    report_text = (run / "report" / "run_report.md").read_text(encoding="utf-8")
    assert summary["scope"] == "isolated_validation"
    assert summary["official_score"] == "UNAVAILABLE / HIDDEN_GT"
    assert "scope: `isolated_validation`" in report_text
    assert "Official score: `UNAVAILABLE / HIDDEN_GT`" in report_text


def test_variant_score_comparison_requires_two_evaluations(tmp_path):
    run, evaluation_path = _fixture(tmp_path)
    second = tmp_path / "evaluation_control.json"
    second.write_text(evaluation_path.read_text(encoding="utf-8"), encoding="utf-8")
    render_run_report(run, evaluations={"stabilized": evaluation_path, "control": second})
    assert (run / "figures" / "score_comparison.png").is_file()


def test_missing_run_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        render_run_report(tmp_path / "missing_run")
