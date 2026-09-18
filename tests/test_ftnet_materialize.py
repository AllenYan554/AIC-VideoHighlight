from __future__ import annotations

import numpy as np
from pathlib import Path
from safetensors.numpy import load_file

from aic_video_highlight.ftnet.materialize import (
    SPLIT_DIRS,
    VideoRef,
    VideoUpstream,
    materialize_split,
    materialize_video,
    verify_materialized,
    write_manifest,
)
from aic_video_highlight.ftnet.native_schema import NATIVE_DIM, NATIVE_FIELDS


def _ref(video_id: str, split: str, category: str = "dog") -> VideoRef:
    return VideoRef(
        canonical_video_id=video_id,
        realized_video_id=video_id,
        category=category,
        stage7_split=split,
        relative_video_path=f"raw/{category}/{video_id}.mp4",
        source_sha256="a" * 64,
    )


def _upstream(ref: VideoRef, length: int = 9) -> VideoUpstream:
    rng = np.random.default_rng(20260917)
    timestamps = np.arange(length, dtype=np.float64) * 0.5
    adjacency = np.ones(length, dtype=bool)
    adjacency[0] = False
    visual = rng.standard_normal((length, 256)).astype(np.float32)
    signals = {}
    for index, name in enumerate(NATIVE_FIELDS):
        if name in ("subject_track_present", "continuity_valid"):
            signals[name] = (rng.random(length) > 0.3).astype(np.float32)
        else:
            signals[name] = rng.random(length).astype(np.float32) * (index + 1)
    target = rng.random(length).astype(np.float32)
    loss_mask = np.ones(length, dtype=bool)
    loss_mask[0] = False
    return VideoUpstream(
        ref=ref,
        timestamps=timestamps,
        source_frame_id=np.arange(length, dtype=np.int64),
        adjacency_mask=adjacency,
        visual=visual,
        native_signals=signals,
        target=target,
        loss_mask=loss_mask,
        candidate_missed_positive=np.zeros(length, dtype=bool),
        metadata={"run_id": "synthetic"},
    )


class _SyntheticProvider:
    def __init__(self) -> None:
        self.refs = {
            "TRAIN": [_ref("train-0", "TRAIN"), _ref("train-1", "TRAIN", "skiing")],
            "VALIDATION": [_ref("val-0", "VALIDATION")],
            "CALIBRATION": [_ref("cal-0", "CALIBRATION")],
        }

    def list_videos(self, stage7_split: str) -> list[VideoRef]:
        return list(self.refs[stage7_split])

    def produce(self, ref: VideoRef) -> VideoUpstream:
        return _upstream(ref)


def test_materialize_writes_frozen_safetensors_contract(tmp_path: Path) -> None:
    provider = _SyntheticProvider()
    records = materialize_split(provider, tmp_path, "TRAIN")

    assert len(records) == 2
    path = tmp_path / SPLIT_DIRS["TRAIN"] / "train-0.safetensors"
    assert path.is_file()
    tensors = load_file(str(path))
    assert tensors["visual"].shape == (9, 256) and tensors["visual"].dtype == np.float16
    assert tensors["native"].shape == (9, NATIVE_DIM)
    assert tensors["native_missing"].shape == (9, NATIVE_DIM)
    assert tensors["target"].shape == (9,)
    assert tensors["loss_mask"].dtype == np.uint8
    assert tensors["adjacency_mask"].dtype == np.uint8
    assert tensors["timestamp"].dtype == np.float64
    assert tensors["source_frame_id"].dtype == np.int64
    assert np.isfinite(tensors["native"]).all()
    assert verify_materialized(path, source_sha256="a" * 64) is True


def test_materialize_is_resumable_and_skips_verified(tmp_path: Path) -> None:
    provider = _SyntheticProvider()
    first = materialize_split(provider, tmp_path, "TRAIN")
    assert all(record["skipped"] is False for record in first)

    second = materialize_split(provider, tmp_path, "TRAIN")
    assert all(record["skipped"] is True for record in second)


def test_materialize_missing_native_is_masked_not_nan(tmp_path: Path) -> None:
    ref = _ref("nan-video", "TRAIN")
    upstream = _upstream(ref)
    upstream.native_signals["bbox_area_ratio"] = np.array(
        [0.1, np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64
    )
    record = materialize_video(upstream, tmp_path)
    tensors = load_file(str(tmp_path / SPLIT_DIRS["TRAIN"] / "nan-video.safetensors"))

    index = NATIVE_FIELDS.index("bbox_area_ratio")
    assert np.isfinite(tensors["native"]).all()
    assert tensors["native"][1, index] == 0.0
    assert tensors["native_missing"][1, index] == 1
    assert record["frame_count"] == 9


def test_write_manifest_records_count_and_paths(tmp_path: Path) -> None:
    provider = _SyntheticProvider()
    records = materialize_split(provider, tmp_path, "VALIDATION")
    manifest = write_manifest(tmp_path, records, extra={"protocol": "test"})
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "materialized-video" in text and "val-0" in text
