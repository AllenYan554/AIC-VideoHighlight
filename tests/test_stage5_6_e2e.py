"""Stage 5.6 end-to-end ablation / final-freeze tests (no GPU, no long run)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.spatial_composition.e2e_pipeline import (
    ARM_NAMES,
    ARMS,
    E2E0,
    E2E1,
    E2EA0,
    E2EPipelineError,
    FrameCrop,
    ablation_invariants,
    assemble_prediction_lines,
    build_final_pipeline_manifest,
    compare_replays,
    evaluate_fresh_reproduction,
    fs0_identity_holds,
    load_stage5_4_shard,
    validate_prediction_lines,
)
from aic_video_highlight.spatial_composition.submission import (
    SubmissionValidationError,
    build_submission_record,
    write_predictions_jsonl,
)
from scripts.experiments.stage5 import run as stage5_registry
from scripts.experiments.stage5 import run_stage5_6_e2e as runner
from scripts.validation import validate_official_contract as contract_cli

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "configs" / "experiments" / "stage5"


def _crops(**frames):
    return {int(k[1:]): FrameCrop(int(k[1:]), *v) for k, v in frames.items()}


# ---------------------------------------------------------------------------
# Preregistration identity
# ---------------------------------------------------------------------------

def test_master_and_protocols_close():
    master = runner.validate_master()
    assert master["status"] == runner.MASTER_STATUS
    for name in ("stage5_6_smoke_protocol.json", "stage5_6_formal_protocol.json", "stage5_6_ablation_protocol.json"):
        protocol = json.loads((CONF / name).read_text(encoding="utf-8"))
        runner.validate_protocol(protocol, protocol["status"])
    smoke = json.loads((CONF / "stage5_6_smoke_protocol.json").read_text(encoding="utf-8"))
    formal = json.loads((CONF / "stage5_6_formal_protocol.json").read_text(encoding="utf-8"))
    assert tuple(smoke["arms"]) == ARMS
    assert smoke["protocol_semantic_sha256"] != formal["protocol_semantic_sha256"]
    assert "E2E-A0" in formal["arms"]
    assert "high_recall_retrieval_v0" == formal["frozen_identities"]["prompt"]


def test_execution_configs_bind_protocol_bytes():
    for config_path, protocol_path in (
        (CONF / "stage5_6_e2e_smoke.json", CONF / "stage5_6_smoke_protocol.json"),
        (CONF / "stage5_6_e2e_formal.json", CONF / "stage5_6_formal_protocol.json"),
        (CONF / "stage5_6_e2e_ablation.json", CONF / "stage5_6_ablation_protocol.json"),
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["protocol_sha256"] == file_sha256(protocol_path)
        assert config["execution_config_schema_version"] == runner.EXECUTION_CONFIG_SCHEMA_VERSION
        assert config["stage_identities"]["model"]["revision"] == runner.MODEL_REVISION


# ---------------------------------------------------------------------------
# FS-0 identity + Stage 5.4 ablation fork isolation
# ---------------------------------------------------------------------------

def test_fs0_identity_holds():
    assert fs0_identity_holds([1, 2, 3], [3, 2, 1])
    assert not fs0_identity_holds([1, 2, 3], [1, 2])


def test_ablation_invariants_only_xy_may_differ():
    ts0 = _crops(f0=(10, 0, 168), f1=(20, 0, 168))
    ts5 = _crops(f0=(12, 1, 168), f1=(18, 0, 168))
    result = ablation_invariants(ts0, ts5)
    assert result["frame_ids_identical"] and result["crop_width_identical"]
    assert result["frames_with_changed_xy"] == 2
    with pytest.raises(E2EPipelineError):
        ablation_invariants(ts0, _crops(f0=(10, 0, 168)))
    with pytest.raises(E2EPipelineError):
        ablation_invariants(ts0, _crops(f0=(10, 0, 200), f1=(20, 0, 168)))


# ---------------------------------------------------------------------------
# Official contract
# ---------------------------------------------------------------------------

def _lines():
    crops = {"v1": _crops(f0=(0, 0, 168), f2=(4, 0, 168))}
    return assemble_prediction_lines(crops, target_ratio=(9, 16))


def test_assemble_and_validate_official_contract(tmp_path):
    lines = _lines()
    assert lines[0]["video_id"] == "v1"
    assert lines[0]["targetRatioWH"] == [9, 16]
    assert lines[0]["predictions"][0] == {"frame": 0, "bboxes": [0, 0, 168]}
    path = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(lines, path)
    report = validate_prediction_lines(
        path,
        index={"v1": [9, 16]},
        metadata={"v1": {"width": 534, "height": 300, "frame_count": 100}},
    )
    assert report.is_valid, report.issues
    assert report.stats["prediction_count"] == 2


def test_duplicate_frame_and_bad_bbox_rejected():
    with pytest.raises(SubmissionValidationError):
        build_submission_record(
            video_id="v", target_ratio=(9, 16),
            predictions=[{"frame": 1, "bboxes": [0, 0, 168]}, {"frame": 1, "bboxes": [1, 1, 168]}],
        )
    with pytest.raises(SubmissionValidationError):
        build_submission_record(
            video_id="v", target_ratio=(9, 16), predictions=[{"frame": 1, "bboxes": [0, -1, 168]}]
        )


def test_validator_flags_ratio_height_violation(tmp_path):
    path = tmp_path / "bad.jsonl"
    # w=200 -> derived h = 355 > height 100
    write_predictions_jsonl(
        [{"video_id": "v", "targetRatioWH": [9, 16],
          "predictions": [{"frame": 0, "bboxes": [0, 0, 200]}]}],
        path,
    )
    report = validate_prediction_lines(
        path, index={"v": [9, 16]}, metadata={"v": {"width": 534, "height": 100, "frame_count": 10}}
    )
    assert not report.is_valid
    assert report.stats["ratio_violation_count"] == 1


def test_contract_cli_detects_duplicate_video_and_passes_valid(tmp_path):
    valid = tmp_path / "valid.jsonl"
    write_predictions_jsonl(_lines(), valid)
    report = contract_cli.validate_contract(valid)
    assert report["is_valid"] is True
    assert report["duplicate_video_count"] == 0

    dup = tmp_path / "dup.jsonl"
    write_predictions_jsonl(_lines() + _lines(), dup)
    report = contract_cli.validate_contract(dup)
    assert report["duplicate_video_count"] == 1
    assert report["is_valid"] is False


def test_contract_cli_scopes_index_to_present_unless_full(tmp_path):
    role = tmp_path / "role.json"
    role.write_text(json.dumps({"records": [{"video_id": "v1"}, {"video_id": "v2"}]}), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(_lines(), predictions)
    scoped = contract_cli.validate_contract(predictions, role_manifest=role)
    assert scoped["is_valid"] is True
    full = contract_cli.validate_contract(predictions, role_manifest=role, require_full_index=True)
    assert full["is_valid"] is False
    assert full["base_stats"]["missing_video_count"] == 1


# ---------------------------------------------------------------------------
# Cache / fresh comparator + gate
# ---------------------------------------------------------------------------

def test_compare_replays_and_fresh_gate():
    cached = [{"video_id": "v", "targetRatioWH": [9, 16],
               "predictions": [{"frame": 0, "bboxes": [0, 0, 168]}, {"frame": 1, "bboxes": [1, 0, 168]}]}]
    fresh_identical = json.loads(json.dumps(cached))
    comparison = compare_replays(cached, fresh_identical)
    assert comparison["video_completion_rate"] == 1.0
    assert comparison["bbox_exact_match_rate"] == 1.0
    gate = evaluate_fresh_reproduction(comparison, schema_success_rate=1.0, contract_valid=True)
    assert gate["all_pass"]

    fresh_partial = [{"video_id": "v", "targetRatioWH": [9, 16],
                      "predictions": [{"frame": 0, "bboxes": [0, 0, 168]}, {"frame": 1, "bboxes": [5, 0, 168]}]}]
    comparison2 = compare_replays(cached, fresh_partial)
    assert comparison2["bbox_exact_match_rate"] == 0.5
    gate2 = evaluate_fresh_reproduction(comparison2, schema_success_rate=1.0, contract_valid=True)
    assert not gate2["all_pass"]
    assert "bbox_exact_match_rate" in gate2["failed_checks"]


def test_load_stage5_4_shard(tmp_path):
    shard = tmp_path / "v.json"
    shard.write_text(json.dumps([
        {"frame": 3, "ts0": {"x": 1, "y": 2, "w": 3}, "ts5": {"x": 4, "y": 5, "w": 3}},
    ]), encoding="utf-8")
    parsed = load_stage5_4_shard(shard)
    assert parsed[3]["ts0"] == FrameCrop(3, 1, 2, 3)
    assert parsed[3]["ts5"] == FrameCrop(3, 4, 5, 3)


# ---------------------------------------------------------------------------
# Manifest / report / registry / launcher
# ---------------------------------------------------------------------------

def test_final_pipeline_manifest_binds_identities():
    manifest = build_final_pipeline_manifest(
        execution_head="deadbeef",
        model_name="Qwen/Qwen3.5-4B",
        model_revision=runner.MODEL_REVISION,
        prompt_identity="high_recall_retrieval_v0",
        stage_identities={"stage5_4_method": "projected_state_canonical_center_ema_v1"},
        protocol_sha256="a" * 64,
        config_sha256="b" * 64,
        runner_identity=runner.RUNNER_IDENTITY,
        validator_identity=runner.VALIDATOR_IDENTITY,
        output_schema_version=runner.OUTPUT_SCHEMA_VERSION,
        environment={"name": "autodl"},
        local_model_snapshot={"revision": runner.MODEL_REVISION},
    )
    assert manifest["schema_version"] == "aic.stage5.6-final-pipeline-manifest/v1"
    assert manifest["heldout_access"] == 0
    assert manifest["official_test_access"] == 0


def test_report_generator_sections(tmp_path):
    config = json.loads((CONF / "stage5_6_e2e_formal.json").read_text(encoding="utf-8"))
    output = tmp_path / "experiment_report.md"
    runner.render_stage5_6_report(
        output, config=config,
        validation={"status": "PASS", "final_candidate": {"status": "x"}},
        metrics=None, runtime=None,
    )
    report = output.read_text(encoding="utf-8")
    assert all(f"## {section}" in report for section in runner.REPORT_SECTIONS)


def test_registry_and_launcher():
    for experiment in ("stage5_6_e2e_smoke", "stage5_6_e2e_formal", "stage5_6_e2e_ablation"):
        assert experiment in stage5_registry.RUNNERS
        assert stage5_registry.LAUNCH[experiment]["target"] == "AUTODL"
        assert stage5_registry.LAUNCH[experiment]["strict_git_preflight"] is True
    assert (REPO / "scripts" / "experiments" / "launch_experiment.ps1").is_file()
    described = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "experiments" / "registry.py"),
         "describe", "--experiment", "stage5_6_e2e_formal"],
        capture_output=True, text=True, check=True,
    )
    spec = json.loads(described.stdout)
    assert spec["target"] == "AUTODL"


def test_validate_only_mode_returns_without_execution():
    config = CONF / "stage5_6_e2e_smoke.json"
    environment = REPO / "configs" / "environments" / "windows_local.json"
    assert runner.main(["--config", str(config), "--environment", str(environment), "--validate-only"]) == 0


def test_deterministic_serialization(tmp_path):
    lines = _lines()
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    write_predictions_jsonl(lines, path_a)
    write_predictions_jsonl(lines, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert canonical_sha256(lines) == canonical_sha256(json.loads(json.dumps(lines)))
