"""True-fresh Stage 5.6 orchestration/provenance tests (CPU-only fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from aic_video_highlight.spatial_composition.fresh_pipeline import (
    FreshPipelineError,
    validate_fresh_candidate_cache_binding,
)
from aic_video_highlight.spatial_localization.full_dev import write_shard
from scripts.experiments.stage5 import run as registry
from scripts.experiments.stage5 import run_stage5_6_fresh as fresh
from scripts.experiments.stage5 import run_stage5_6_vhicraft as runner

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "configs" / "experiments" / "stage5"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_fresh_manifest_membership_is_role_ordered_and_deterministic(tmp_path):
    source = tmp_path / "dev.jsonl"
    _jsonl(source, [
        {"video_id": "a", "split": "dev"},
        {"video_id": "b", "split": "dev"},
        {"video_id": "c", "split": "dev"},
    ])
    role = [{"video_id": "c"}, {"video_id": "a"}, {"video_id": "b"}]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert fresh.select_dev_manifest(source, role, first, videos=2) == ["c", "a"]
    assert fresh.select_dev_manifest(source, role, second, videos=2) == ["c", "a"]
    assert first.read_bytes() == second.read_bytes()


def test_stage1_output_is_the_bound_cache_consumer_and_frozen_fallback_fails(tmp_path):
    stage1 = tmp_path / "predictions.jsonl"
    _jsonl(stage1, [{"video_id": "a", "success": True}])
    manifest = {
        "global_semantic_sha256": "1" * 64,
        "source_sets": [{"source_artifact_hashes": {"predictions_jsonl_sha256": file_sha256(stage1)}}],
        "records": [{"video_id": "a"}],
    }
    proof = validate_fresh_candidate_cache_binding(
        manifest,
        stage1_predictions_path=stage1,
        expected_video_ids=["a"],
        forbidden_global_sha256=fresh.FROZEN_CACHE_SHA256,
    )
    assert proof["downstream_consumer"] == "stage4_candidate_cache"
    assert proof["frozen_fallback"] is False
    manifest["global_semantic_sha256"] = fresh.FROZEN_CACHE_SHA256
    with pytest.raises(FreshPipelineError, match="frozen cache"):
        validate_fresh_candidate_cache_binding(
            manifest,
            stage1_predictions_path=stage1,
            expected_video_ids=["a"],
            forbidden_global_sha256=fresh.FROZEN_CACHE_SHA256,
        )


def test_resume_accepts_same_identity_and_rejects_wrong_identity(tmp_path):
    identity = {"schema_version": fresh.RUN_IDENTITY_SCHEMA, "identity_sha256": "a" * 64}
    assert fresh.bind_resume(tmp_path, identity, resume=False) == 0  # noqa: E712
    assert fresh.bind_resume(tmp_path, identity, resume=True) == 1
    with pytest.raises(FreshPipelineError, match="identity mismatch"):
        fresh.bind_resume(
            tmp_path,
            {"schema_version": fresh.RUN_IDENTITY_SCHEMA, "identity_sha256": "b" * 64},
            resume=True,
        )


def test_stage5_3_fresh_binding_uses_current_stage5_1_and_stage5_2(tmp_path):
    predictions = tmp_path / "stage5_1.jsonl"
    index = tmp_path / "index.jsonl"
    metadata = tmp_path / "metadata.json"
    _jsonl(predictions, [{
        "video_id": "v", "targetRatioWH": [9, 16],
        "predictions": [{"frame": 2, "bboxes": [183, 0, 168]}],
    }])
    _jsonl(index, [{"video_id": "v", "targetRatioWH": [9, 16], "video_path": "v.mp4"}])
    metadata.write_text(json.dumps({"records": {"v": {
        "video_id": "v", "video_path": "v.mp4", "width": 534, "height": 300,
        "fps": 30.0, "fps_rational": "30/1", "frame_count": 10,
        "duration_sec": 1.0, "timestamp_mode": "CFR_FPS", "rate_provenance": {},
        "pts_timestamps": None,
    }}}), encoding="utf-8")
    stage52 = tmp_path / "stage5_2"
    policy = [{
        "video_id": "v", "frame": 2, "image_width": 534, "image_height": 300,
        "source_timestamp": 2 / 30, "policy_version": fresh.RTDETR_POLICY,
        "candidate_count": 0, "invalid_candidate_count": 0, "primary": None,
        "status": "CENTER_CROP_FALLBACK", "fallback_reasons": ["NO_DETECTION"],
        "ambiguous": False, "ambiguous_candidate_count": 0, "fallback_box": None,
        "model_error": None,
    }]
    write_shard(stage52, "v", [{"video_id": "v", "frame": 2}], policy, [{"video_id": "v", "frame": 2}])
    inputs, proof = fresh._build_fresh_inputs(predictions, metadata, index, stage52)
    assert proof["expected_frames"] == proof["policy_frames"] == 1
    manifest = fresh._composition_manifest(inputs)
    assert manifest["algorithm"] == fresh.CMP1_METHOD
    assert manifest["frame_count"] == 1


def test_fresh_arms_cannot_call_frozen_chain(tmp_path):
    with pytest.raises(Exception, match="frozen-chain fallback is forbidden"):
        runner.run_arm({}, object(), arm="VC-1", video_ids=[], mode="smoke", execution_head="x")


def test_formal_registry_and_configs_require_true_fresh_gpu():
    for name in ("stage5_6_vhicraft_smoke", "stage5_6_vhicraft_formal", "stage5_6_vhicraft_ablation"):
        config = json.loads((CONF / f"{name}.json").read_text(encoding="utf-8"))
        assert config["arms"] == ["VC-0", "VC-1", "VC-A0"]
        assert config["runtime"]["gpu"] == "REQUIRED"
        assert config["fresh_pipeline"]["schema_version"] == fresh.FRESH_PIPELINE_SCHEMA
        assert registry.LAUNCH[name]["gpu"] == "REQUIRED"

