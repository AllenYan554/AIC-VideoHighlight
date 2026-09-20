from __future__ import annotations

import json
from pathlib import Path

import torch

from aic_video_highlight.ftnet.formal_reporting import (
    ARCHIVE_DIRECTORIES,
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
    for directory in ARCHIVE_DIRECTORIES:
        (formal / directory).mkdir(parents=True, exist_ok=True)
    history = [
        {
            "epoch": epoch,
            "train_loss": 0.3 - epoch * 0.02,
            "val_loss": 0.2 - epoch * 0.01,
            "lr": 3e-4 - epoch * 5e-5,
            "grad_norm": 0.5 - epoch * 0.05,
        }
        for epoch in range(3)
    ]
    _write_json(formal / "results" / "training_history.json", history)
    _write_json(
        formal / "results" / "training_summary.json",
        {
            "run_id": "formal-test",
            "git_head": "abc123",
            "device": "cuda",
            "best_epoch": 3,
            "best_val_masked_bce": 0.18,
            "epochs_run": 3,
            "steps": 6,
            "checkpoint_best": "/remote/formal-test/best.pt",
            "checkpoint_last": "/remote/formal-test/last.pt",
        },
    )
    _write_json(formal / "results" / "performance_summary.json", {"status": "NOT_BENCHMARKED"})
    _write_json(formal / "supplementary" / "frozen_index.json", {"total": 1})
    _write_json(formal / "supplementary" / "idx0_gate_decision.json", {"decision": "PASS"})
    _write_json(
        data / "manifests" / "materialized_videos.json",
        {
            "count": 1,
            "records": [
                {
                    "canonical_video_id": "video-a",
                    "relative_path": "train/video-a.safetensors",
                }
            ],
        },
    )
    (data / "train").mkdir(parents=True)
    (data / "train" / "video-a.safetensors").write_bytes(b"fixture")
    _write_json(
        data / "audits" / "integrity_report.json",
        {
            "status": "PASS",
            "counts": {"TRAIN": 1, "VALIDATION": 0, "CALIBRATION": 0},
            "missing": [],
            "extra": [],
            "duplicate": [],
            "corrupt": [],
            "official_test_rows": 0,
            "tvsum_rows": 0,
        },
    )
    _write_json(
        data / "audits" / "native_feature_audit.json",
        {
            "video_count": 1,
            "frame_count": 10,
            "native_schema_version": "stage7-native-v1.1",
            "fields": {
                "field_a": {
                    "mean": 0.4,
                    "std": 0.1,
                    "count": 10,
                    "unique_count": 8,
                    "missing_rate": 0.0,
                },
                "field_b": {
                    "mean": 0.7,
                    "std": 0.2,
                    "count": 8,
                    "unique_count": 4,
                    "missing_rate": 0.2,
                },
            },
        },
    )
    _write_json(
        data / "normalization" / "normalization_stats.json",
        {"train_video_count": 1, "native_dim": 16, "missing_policy": "standardize_then_zero"},
    )
    (data / "normalization" / "normalization_stats.json.sha256").write_text(
        "fixture  normalization_stats.json\n"
    )
    return formal, data


def _generate(formal: Path, data: Path):
    return generate_formal_deliverables(
        formal,
        data_root=data,
        repo_root=REPO,
        training_config_path=REPO / "configs" / "models" / "ftnet_reference.yaml",
        config_snapshot_paths=(
            REPO / "configs" / "experiments" / "stage7" / "ftnet_train_formal.json",
            REPO / "configs" / "environments" / "windows_local.json",
        ),
    )


def _write_checkpoint(path: Path, *, epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "aic.stage7.ftnet.checkpoint/v1",
            "epoch": epoch,
            "global_step": step,
            "identity": {"git_head": "abc123"},
            "config": {},
        },
        path,
    )


def _write_required_logs(formal: Path) -> None:
    for name in (
        "training_progress.json",
        "materialization_status.json",
        "materialization_progress.json",
        "materialization_summary.json",
        "materialization_failures.json",
    ):
        _write_json(formal / "supplementary" / name, {"elapsed_sec": 1.0})
    for name in ("training_progress.jsonl", "materialization_progress.jsonl"):
        (formal / "supplementary" / name).write_text("{}\n", encoding="utf-8")
    (formal / "supplementary" / "materialization_vllm.log").write_text(
        "existing log\n", encoding="utf-8"
    )
    (formal / "supplementary" / "path_and_storage_audit.md").write_text(
        "# Path audit\n", encoding="utf-8"
    )
    (formal / "supplementary" / "performance_audit.md").write_text(
        "# Performance audit\n", encoding="utf-8"
    )


def test_report_and_real_data_figures_are_generated_offline(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    status = _generate(formal, data)

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
    audit = json.loads(
        (formal / "supplementary" / "artifact_audit.json").read_text(encoding="utf-8")
    )
    missing = {item for item in audit["missing"] if item is not None}
    assert {
        "results/checkpoints/best.pt",
        "results/checkpoints/last.pt",
    }.issubset(missing)
    manifest = json.loads(
        (formal / "supplementary" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_file_count"] == 6


def test_formal_archive_layout_has_no_flat_json_and_report_links_resolve(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    _generate(formal, data)

    assert {path.name for path in formal.iterdir() if path.is_dir()} == set(ARCHIVE_DIRECTORIES)
    assert {path.name for path in formal.iterdir() if path.is_file()} == {"experiment_report.md"}
    report = (formal / "experiment_report.md").read_text(encoding="utf-8")
    for name in TRAINING_FIGURES:
        assert (formal / "figures" / name).is_file()
        assert f"](figures/{name})" in report
    assert "`results/materialized_videos.json`" in report
    assert "`supplementary/frozen_index.json`" in report
    assert "figures/native16_mean_std.png" in report
    assert "figures/native16_unique_or_variability.png" in report


def test_formal_completion_requires_valid_nested_checkpoints_and_logs(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    _write_checkpoint(formal / "results" / "checkpoints" / "best.pt", epoch=3, step=6)
    _write_checkpoint(formal / "results" / "checkpoints" / "last.pt", epoch=3, step=6)
    _write_required_logs(formal)

    status = _generate(formal, data)

    assert status["status"] == "COMPLETE"
    assert status["missing_required"] == []
    assert status["checkpoint_validation"] == "PASS"
    checkpoint_audit = json.loads(
        (formal / "supplementary" / "checkpoint_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint_audit["checkpoints"]["best.pt"]["relative_path"] == (
        "results/checkpoints/best.pt"
    )


def test_checkpoint_identity_conflict_fails_closed(tmp_path: Path) -> None:
    formal, data = _prepare_inputs(tmp_path)
    _write_checkpoint(formal / "results" / "checkpoints" / "best.pt", epoch=2, step=4)
    _write_checkpoint(formal / "results" / "checkpoints" / "last.pt", epoch=3, step=6)

    try:
        _generate(formal, data)
    except ValueError as exc:
        assert "best checkpoint epoch conflicts" in str(exc)
    else:  # pragma: no cover - explicit fail-closed contract
        raise AssertionError("checkpoint conflict did not fail closed")


def test_runtime_evidence_is_collected_without_flat_archive_json(tmp_path: Path) -> None:
    _, data = _prepare_inputs(tmp_path)
    formal = tmp_path / "collected_archive"
    runtime = tmp_path / "runtime" / "formal-test"
    materialization = tmp_path / "runtime" / "full154"
    _write_json(
        runtime / "history.json",
        [{"epoch": 0, "train_loss": 0.2, "val_loss": 0.1, "lr": 1e-4, "grad_norm": 0.3}],
    )
    _write_json(
        runtime / "summary.json",
        {
            "run_id": "formal-test",
            "git_head": "abc123",
            "best_epoch": 1,
            "epochs_run": 1,
            "steps": 1,
            "checkpoint_best": "/remote/formal-test/best.pt",
            "checkpoint_last": "/remote/formal-test/last.pt",
        },
    )
    _write_json(materialization / "status.json", {"count": 1})
    _write_json(materialization / "summary.json", {"status": "PASS"})
    _write_json(materialization / "failures.json", {"count": 0})
    _write_json(materialization / "logs" / "progress.json", {"processed": 1})
    (materialization / "logs" / "progress.jsonl").write_text("{}\n", encoding="utf-8")
    (materialization / "logs" / "vllm.log").write_text("existing\n", encoding="utf-8")
    _write_json(materialization / "performance_summary.json", {"status": "RECORDED"})
    idx0 = tmp_path / "runtime" / "idx0_gate_decision.json"
    _write_json(idx0, {"decision": "PASS"})

    generate_formal_deliverables(
        formal,
        data_root=data,
        training_run_dir=runtime,
        materialization_run_dir=materialization,
        idx0_gate_path=idx0,
        training_config_path=REPO / "configs" / "models" / "ftnet_reference.yaml",
    )

    assert (runtime / "history.json").is_file()
    assert (materialization / "status.json").is_file()
    assert (formal / "results" / "training_history.json").is_file()
    assert (formal / "results" / "performance_summary.json").is_file()
    assert (formal / "supplementary" / "materialization_status.json").is_file()
    assert (formal / "supplementary" / "materialization_vllm.log").is_file()
    assert not list(formal.glob("*.json"))


def test_windows_environment_separates_data_runtime_and_archive_paths() -> None:
    environment = json.loads(
        (REPO / "configs" / "environments" / "windows_local.json").read_text(encoding="utf-8")
    )

    assert environment["derived"] == "E:/ResearchData/derived"
    assert environment["outputs"].endswith("/实验运行/outputs")
    assert environment["archive"].endswith("/实验记录")
    assert environment["outputs"] != environment["archive"]
    formal_config = json.loads(
        (REPO / "configs" / "experiments" / "stage7" / "ftnet_train_formal.json").read_text(
            encoding="utf-8"
        )
    )
    assert formal_config["output_root"] == {
        "base": "derived",
        "path": "VHiCraFTNet/youtube_highlights_ftnet",
    }
