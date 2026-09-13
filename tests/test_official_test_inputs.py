"""Official-test input materialization tests (file names only; no video content)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.runtime.datasets import (  # noqa: E402
    DatasetInputError,
    materialize_numbered_video_inputs,
    validate_target_ratio,
)


def _numbered_root(tmp_path: Path) -> Path:
    root = tmp_path / "tests"
    root.mkdir()
    for name in ("10.mp4", "2.mp4", "A.mp4"):
        (root / name).write_bytes(b"x")
    return root


def test_materialize_numbered_videos_numeric_order(tmp_path):
    root = _numbered_root(tmp_path)
    result = materialize_numbered_video_inputs(root, tmp_path / "input", [9, 16])
    assert [record["video_id"] for record in result["records"]] == ["2", "10", "A"]
    manifest = [
        json.loads(line)
        for line in result["manifest_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest[0]["relative_video_path"] == "2.mp4"
    assert manifest[0]["split"] == "test"
    assert manifest[0]["clip_start_sec"] == 0.0
    assert manifest[0]["clip_end_sec"] > 0
    assert manifest[0]["weak_reference_segments"] == []
    index = [
        json.loads(line)
        for line in result["index_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert index[0]["targetRatioWH"] == [9.0, 16.0]
    assert index[0]["video_path"] == "2.mp4"


def test_materialize_respects_requested_video_ids(tmp_path):
    root = _numbered_root(tmp_path)
    result = materialize_numbered_video_inputs(root, tmp_path / "input", [9, 16], video_ids=["10", "2"])
    assert [record["video_id"] for record in result["records"]] == ["10", "2"]


def test_materialize_rejects_unknown_video_id(tmp_path):
    root = _numbered_root(tmp_path)
    with pytest.raises(DatasetInputError):
        materialize_numbered_video_inputs(root, tmp_path / "input", [9, 16], video_ids=["999"])


def test_materialize_rejects_empty_root(tmp_path):
    empty = tmp_path / "tests"
    empty.mkdir()
    with pytest.raises(DatasetInputError):
        materialize_numbered_video_inputs(empty, tmp_path / "input", [9, 16])


def test_validate_target_ratio():
    assert validate_target_ratio([9, 16]) == (9.0, 16.0)
    with pytest.raises(DatasetInputError):
        validate_target_ratio([9])
    with pytest.raises(DatasetInputError):
        validate_target_ratio([0, 16])
