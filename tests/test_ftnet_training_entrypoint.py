from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aic_video_highlight.ftnet.dataset import compute_train_normalization
from aic_video_highlight.ftnet.materialize import (
    VideoRef,
    VideoUpstream,
    materialize_video,
)
from aic_video_highlight.ftnet.native_schema import NATIVE_FIELDS


REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts" / "train_ftnet.py"


def _ref(video_id: str, split: str) -> VideoRef:
    return VideoRef(
        canonical_video_id=video_id,
        realized_video_id=video_id,
        category="dog",
        stage7_split=split,
        relative_video_path=f"raw/dog/{video_id}.mp4",
        source_sha256="b" * 64,
    )


def _upstream(ref: VideoRef, length: int) -> VideoUpstream:
    rng = np.random.default_rng(hash(ref.canonical_video_id) % (2**31))
    adjacency = np.ones(length, dtype=bool)
    adjacency[0] = False
    signals = {}
    for index, name in enumerate(NATIVE_FIELDS):
        if name in ("subject_track_present", "continuity_valid"):
            signals[name] = (rng.random(length) > 0.3).astype(np.float32)
        else:
            signals[name] = rng.random(length).astype(np.float32) * (index + 1)
    return VideoUpstream(
        ref=ref,
        timestamps=np.arange(length, dtype=np.float64) * 0.5,
        source_frame_id=np.arange(length, dtype=np.int64),
        adjacency_mask=adjacency,
        visual=rng.standard_normal((length, 256)).astype(np.float32),
        native_signals=signals,
        target=rng.random(length).astype(np.float32),
        loss_mask=np.ones(length, dtype=bool),
        metadata={"run_id": "synthetic-entrypoint"},
    )


def _materialize_fixture(root: Path) -> Path:
    data_root = root / "youtube_highlights_ftnet"
    for split, names in {
        "TRAIN": ["v0", "v1", "v2"],
        "VALIDATION": ["v3", "v4"],
        "CALIBRATION": ["v5"],
    }.items():
        for name in names:
            materialize_video(_upstream(_ref(name, split), length=6 + len(name)), data_root)
    compute_train_normalization(
        data_root, output_path=data_root / "normalization" / "normalization_stats.json"
    )
    return data_root


def test_train_entrypoint_runs_and_writes_checkpoints(tmp_path: Path) -> None:
    data_root = _materialize_fixture(tmp_path)
    output_root = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(data_root),
            "--output-root", str(output_root),
            "--run-id", "ftnet_test_20260918",
            "--device", "cpu",
            "--num-workers", "0",
            "--max-epochs", "1",
            "--max-steps", "2",
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    run_dir = output_root / "ftnet_test_20260918"
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "logs" / "progress.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "ftnet_test_20260918"
    assert summary["steps"] >= 1


def test_train_entrypoint_validate_only_checks_normalization(tmp_path: Path) -> None:
    data_root = _materialize_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(data_root),
            "--output-root", str(tmp_path / "out"),
            "--run-id", "validate_only",
            "--validate-only",
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout[completed.stdout.index("{"):])
    assert payload["status"] == "VALIDATE_ONLY"
    assert payload["stats_exists"] is True


def test_train_entrypoint_uses_native_dim_16(tmp_path: Path) -> None:
    data_root = _materialize_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(data_root),
            "--output-root", str(tmp_path / "out"),
            "--run-id", "native16",
            "--device", "cpu",
            "--max-epochs", "1",
            "--max-steps", "1",
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "native_dim=16" in completed.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_entrypoint_cuda_smoke(tmp_path: Path) -> None:
    data_root = _materialize_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(data_root),
            "--output-root", str(tmp_path / "out"),
            "--run-id", "cuda_smoke",
            "--device", "cuda",
            "--max-epochs", "1",
            "--max-steps", "2",
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "gpu=" in completed.stdout
