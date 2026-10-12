"""Unit tests for the PHD2 Full1500 Qwen retrieval production module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.phd2_qwen1500 import (
    RAW1500_SCHEMA,
    Qwen1500Error,
    RunCounters,
    artifact_paths,
    chunk_count_for_duration,
    entries_from_records,
    is_systemic_error,
    load_raw1500_manifest,
    validate_artifact,
)


def _manifest(tmp_path: Path, count: int = 2, size: int = 1024) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    records = []
    for index in range(count):
        relative = f"raw/360p/vid{index}.mp4"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        records.append(
            {
                "youtubeId": f"vid{index}",
                "relative_path": relative,
                "actual_bytes": size,
                "actual_duration_s": 100.0 + index,
                "sha256": "a" * 64,
                "duration_bin": "1-3min",
                "vcodec": "vp9",
            }
        )
    manifest = tmp_path / "raw1500.json"
    manifest.write_text(
        json.dumps({"schema": RAW1500_SCHEMA, "records": records, "total_videos": count}),
        encoding="utf-8",
    )
    return manifest, root


def test_chunk_count_matches_frozen_formula() -> None:
    assert chunk_count_for_duration(0.5) == 1
    assert chunk_count_for_duration(30.0) == 1
    assert chunk_count_for_duration(30.5) == 2
    assert chunk_count_for_duration(55.0) == 2
    assert chunk_count_for_duration(55.2) == 3
    assert chunk_count_for_duration(160.9) == 7
    assert chunk_count_for_duration(862.6) == 35


def test_load_raw1500_manifest_validates_and_builds_entries(tmp_path: Path) -> None:
    manifest, root = _manifest(tmp_path, count=3)
    records, meta = load_raw1500_manifest(
        manifest, expected_sha256=None, dataset_root=root, expected_records=3
    )
    assert len(records) == 3
    assert meta["record_count"] == 3
    assert meta["total_bytes"] == 3 * 1024
    entries = entries_from_records(records, dataset_root=root)
    assert [entry.video_id for entry in entries] == ["vid0", "vid1", "vid2"]
    assert all(entry.category == "360p" for entry in entries)
    assert entries[0].duration_sec == 100.0
    assert entries[0].relative_video_path == "raw/360p/vid0.mp4"
    assert entries[0].dataset_id == "."


def test_load_raw1500_manifest_rejects_sha_mismatch(tmp_path: Path) -> None:
    manifest, root = _manifest(tmp_path)
    for call in (
        lambda: load_raw1500_manifest(
            manifest, expected_sha256="0" * 64, dataset_root=root, expected_records=2
        ),
    ):
        with pytest.raises(Qwen1500Error, match="SHA256 mismatch"):
            call()


def test_load_raw1500_manifest_rejects_size_mismatch(tmp_path: Path) -> None:
    manifest, root = _manifest(tmp_path)
    (root / "raw/360p/vid0.mp4").write_bytes(b"yy")
    with pytest.raises(Qwen1500Error, match="size mismatch"):
        load_raw1500_manifest(
            manifest, expected_sha256=None, dataset_root=root, expected_records=2
        )


def test_load_raw1500_manifest_rejects_missing_raw(tmp_path: Path) -> None:
    manifest, root = _manifest(tmp_path)
    (root / "raw/360p/vid1.mp4").unlink()
    with pytest.raises(Qwen1500Error, match="missing"):
        load_raw1500_manifest(
            manifest, expected_sha256=None, dataset_root=root, expected_records=2
        )


def test_validate_artifact_detects_corruption(tmp_path: Path) -> None:
    artifact, _sidecar = artifact_paths(tmp_path, "vid")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{not json", encoding="utf-8")
    assert validate_artifact(artifact, "vid")[0] is False

    artifact.write_text(
        json.dumps(
            {
                "schema": "aic.stage7.ftnet.retrieval-artifact/v1",
                "video_id": "other",
                "chunks": [{"chunk_index": 0, "parse_success": True}],
                "raw_candidates": [],
                "merged_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    valid, reason = validate_artifact(artifact, "vid")
    assert valid is False and reason == "foreign video_id"

    artifact.write_text(
        json.dumps(
            {
                "schema": "aic.stage7.ftnet.retrieval-artifact/v1",
                "video_id": "vid",
                "chunks": [{"chunk_index": 0, "parse_success": False}],
                "raw_candidates": [],
                "merged_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    valid, reason = validate_artifact(artifact, "vid")
    assert valid is False and "parse_success" in str(reason)

    artifact.write_text(
        json.dumps(
            {
                "schema": "aic.stage7.ftnet.retrieval-artifact/v1",
                "video_id": "vid",
                "chunks": [{"chunk_index": 0, "parse_success": True}],
                "raw_candidates": [{"start_sec": 0.0}],
                "merged_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    assert validate_artifact(artifact, "vid") == (True, None)


def test_counters_report_eta_and_throughput() -> None:
    counters = RunCounters(
        videos_total=100,
        expected_chunks=1000,
        started_monotonic=0.0,
        completed=10,
        chunks=100,
        elapsed_seconds=200.0,
    )
    payload = counters.to_dict()
    assert payload["chunks_per_second"] == pytest.approx(0.5)
    assert payload["remaining"] == 90
    assert payload["eta_seconds"] == pytest.approx(1800.0)


def test_systemic_error_classification() -> None:
    assert is_systemic_error("ConnectionError: Connection refused")
    assert is_systemic_error("openai.APITimeoutError: Request timed out")
    assert not is_systemic_error("ResponseParseError: response is not strict JSON")
