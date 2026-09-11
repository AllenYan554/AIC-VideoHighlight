"""Stage 5.2 input-gate mode split (frozen replay integrity vs fresh pipeline)."""

from __future__ import annotations

from scripts.experiments.stage5.run_stage5_2_full_dev import (
    FROZEN_REPLAY_MODE,
    FRESH_PIPELINE_MODE,
    validate_prediction_manifest,
)


def _frozen_manifest() -> dict[str, list[int]]:
    manifest = {f"v{index:03d}": [0] for index in range(165)}
    manifest["v165"] = list(range(51256 - 165))
    return manifest


def test_frozen_replay_accepts_exact_166_51256():
    assert validate_prediction_manifest(_frozen_manifest(), mode=FROZEN_REPLAY_MODE) is None


def test_frozen_replay_rejects_wrong_frame_count():
    manifest = _frozen_manifest()
    manifest["v000"] = [0, 1]
    error = validate_prediction_manifest(manifest, mode=FROZEN_REPLAY_MODE)
    assert error is not None and "FROZEN_REPLAY" in error


def test_frozen_replay_rejects_wrong_video_count():
    manifest = _frozen_manifest()
    manifest.pop("v000")
    error = validate_prediction_manifest(manifest, mode=FROZEN_REPLAY_MODE)
    assert error is not None


def test_fresh_pipeline_accepts_variable_frame_count():
    manifest = {"a": [0, 2, 4], "b": [1]}
    assert validate_prediction_manifest(manifest, mode=FRESH_PIPELINE_MODE) is None


def test_fresh_pipeline_rejects_duplicate_frames():
    manifest = {"a": [0, 0]}
    error = validate_prediction_manifest(manifest, mode=FRESH_PIPELINE_MODE)
    assert error is not None and "duplicate" in error


def test_fresh_pipeline_rejects_unsorted_frames():
    manifest = {"a": [2, 1]}
    error = validate_prediction_manifest(manifest, mode=FRESH_PIPELINE_MODE)
    assert error is not None and "unsorted" in error


def test_fresh_pipeline_accepts_empty_video():
    assert validate_prediction_manifest({"a": [], "b": [1]}, mode=FRESH_PIPELINE_MODE) is None
    assert validate_prediction_manifest({"a": []}, mode=FRESH_PIPELINE_MODE) is None
    assert validate_prediction_manifest({}, mode=FRESH_PIPELINE_MODE) is not None


def test_frozen_replay_still_rejects_empty_video():
    manifest = _frozen_manifest()
    manifest["v000"] = []
    error = validate_prediction_manifest(manifest, mode=FROZEN_REPLAY_MODE)
    assert error is not None and "has no frames" in error


def test_fresh_pipeline_enforces_expected_video_count():
    manifest = {"a": [0], "b": [1]}
    assert validate_prediction_manifest(manifest, mode=FRESH_PIPELINE_MODE) is None
    error = validate_prediction_manifest(
        manifest, mode=FRESH_PIPELINE_MODE, expected_videos=166
    )
    assert error is not None and "expected=166" in error


def test_unknown_mode_is_rejected():
    error = validate_prediction_manifest({"a": [0]}, mode="NOPE")
    assert error is not None and "unknown mode" in error
