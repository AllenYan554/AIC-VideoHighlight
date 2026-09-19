from __future__ import annotations

import json
from pathlib import Path

from aic_video_highlight.ftnet.formal_reporting import (
    INSUFFICIENT,
    TRAINING_FIGURES,
    generate_formal_deliverables,
)

REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    formal = tmp_path / "archive" / "stage7_1"
    data = tmp_path / "derived" / "youtube_highlights_ftnet"
    formal.mkdir(parents=True)
    history = [
        {"epoch": epoch, "train_loss": 0.3 - epoch * 0.02, "val_loss": 0.2 - epoch * 0.01,
         "lr": 3e-4 - epoch * 5e-5, "grad_norm": 0.5 - epoch * 0.05}
        for epoch in range(3)
    ]
    _write_json(formal / "training_history.json", history)
    _write_json(
        formal / "training_summary.json",
        {
            "run_id": "formal-test",
            "git_head": "abc123",
            "device": "cuda",
            "best_epoch": 3,
            "best_val_masked_bce": 0.18,
            "epochs_run": 3,
            "steps": 6,
            "checkpoint_best": "/remote/best.pt",
            "checkpoint_last": "/remote/last.pt",
        },
    )
    _write_json(
        data / "manifests" / "materialized_videos.json",
        {"count": 1, "records": [{"canonical_video_id": "video-a", "relative_path": "train/video-a.safetensors"}]},
    )
    (data / "train").mkdir(parents=True)
    (data / "train" / "video-a.safetensors").write_bytes(b"fixture")
    _write_json(
        data / "audits" / "integrity_report.json",
        {
            "status": "PASS",
            "counts": {"TRAIN": 1, "VALIDATION": 0, "CALIBRATION": 0},
            "missing": [], "extra": [], "duplicate": [], "corrupt": [],
            "official_test_rows": 0, "tvsum_rows": 0,
        },
    )
    _write_json(
        data / "audits" / "native_feature_audit.json",
        {
            "video_count": 1,
            "frame_count": 10,
            "native_schema_version": "stage7-native-v1.1",
            "fields": {
                "field_a": {"mean": 0.4, "std": 0.1, "count": 10, "unique_count": 8, "missing_rate": 0.0},
                "field_b": {"mean": 0.7, "std": 0.2, "count": 8, "unique_count": 4, "missing_rate": 0.2},
            },
        },
    )
    _write_json(
        data / "normalization" / "normalization_stats.json",
        {"train_video_count": 1, "native_dim": 16, "missing_policy": "standardize_then_zero"},
    )
    (data / "normalization" / "normalization_stats.json.sha256").write_text("fixture  normalization_stats.json\n")
    return formal, data


def test_report_and_real_data_figures_are_generated_offline(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    status = generate_formal_deliverables(
        formal,
        data_root=data,
        repo_root=REPO,
        training_config_path=REPO / "configs" / "models" / "ftnet_reference.yaml",
    )

    assert status["status"] == "INCOMPLETE"
    assert (formal / "experiment_report.md").is_file()
    report = (formal / "experiment_report.md").read_text(encoding="utf-8")
    assert "Stage 7.1 Real-Data FTNet Baseline Experiment Report" in report
    assert INSUFFICIENT in report
    for name in TRAINING_FIGURES:
        assert (formal / "figures" / name).stat().st_size > 0
        assert f"figures/{name}" in report
    assert (formal / "figures" / "native16_mean_std.png").stat().st_size > 0
    assert (formal / "figures" / "native16_unique_or_variability.png").stat().st_size > 0
    audit = json.loads((formal / "artifact_audit.json").read_text(encoding="utf-8"))
    missing = {item for item in audit["missing"] if item is not None}
    assert {"best.pt", "last.pt"}.issubset(missing)
    manifest = json.loads((formal / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_file_count"] == 6


def test_formal_completion_requires_checkpoints_report_figures_and_manifest(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    (formal / "best.pt").write_bytes(b"best")
    (formal / "last.pt").write_bytes(b"last")

    status = generate_formal_deliverables(
        formal,
        data_root=data,
        repo_root=REPO,
        training_config_path=REPO / "configs" / "models" / "ftnet_reference.yaml",
    )

    assert status["status"] == "COMPLETE"
    assert status["missing_required"] == []


def test_windows_environment_separates_data_runtime_and_archive_paths() -> None:
    environment = json.loads(
        (REPO / "configs" / "environments" / "windows_local.json").read_text(encoding="utf-8")
    )

    assert environment["derived"] == "E:/ResearchData/derived"
    assert environment["outputs"].endswith("/实验运行/outputs")
    assert environment["archive"].endswith("/实验记录")
    assert environment["outputs"] != environment["archive"]
