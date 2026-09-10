"""Stage 5.4 Amendment 2 Formal preregistration contract tests (no execution)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from scripts.experiments.stage5.run_stage5_4_temporal import (
    build_analysis_set_identity,
    build_analysis_metrics_all_treatments,
    evaluate_amendment2_formal_scientific_gates,
    guard_projection_diagnostics,
    validate_formal_contract,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment2_formal.json"
PROTOCOL_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment2_formal_protocol.json"
TS1_FORMAL_CONFIG_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_formal.json"
SMOKE_MANIFEST_PATH = (
    REPO_ROOT.parent
    / "实验记录/Stage5_逐帧高光预测与空间构图/05_Temporal_Composition_Stabilization"
    / "smoke/00_PROTOCOL/manifest/stage5_4_smoke_manifest_v1.json"
)


def _manifest() -> dict:
    return {
        "manifest_id": "stage5_3_formal_manifest_v1",
        "manifest_sha256": "formal-manifest-sha",
        "video_count": 3,
        "frame_count": 6,
        "videos": [
            {"video_id": "a", "frames": [{"frame": 0}, {"frame": 1}]},
            {"video_id": "b", "frames": [{"frame": 0}]},
            {"video_id": "c", "frames": [{"frame": 4}, {"frame": 5}, {"frame": 6}]},
        ],
    }


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["manifest"]["expected_manifest_sha256"] = "formal-manifest-sha"
    config["manifest"]["expected_video_count"] = 3
    config["manifest"]["expected_frame_count"] = 6
    config["smoke_binding"]["expected_manifest_sha256"] = "smoke"
    config["analysis_sets"]["full_dev166"]["expected_video_count"] = 3
    config["analysis_sets"]["full_dev166"]["expected_frame_count"] = 6
    config["analysis_sets"]["confirmatory_dev142"]["expected_video_count"] = 2
    config["analysis_sets"]["confirmatory_dev142"]["expected_frame_count"] = 5
    config["analysis_sets"]["confirmatory_dev142"]["excluded_smoke_video_ids"] = ["b"]
    full_spec = config["analysis_sets"]["full_dev166"]
    confirm_spec = config["analysis_sets"]["confirmatory_dev142"]
    full = build_analysis_set_identity(
        _manifest(), full_spec["name"], full_spec["definition"], {"a", "b", "c"}
    )
    confirm = build_analysis_set_identity(
        _manifest(), confirm_spec["name"], confirm_spec["definition"], {"a", "c"}
    )
    config["analysis_sets"]["full_dev166"]["identity_sha256"] = full["identity_sha256"]
    config["analysis_sets"]["confirmatory_dev142"]["identity_sha256"] = confirm["identity_sha256"]
    return config


def _protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_amendment2_formal_protocol_is_preregistered_and_sha_bound():
    protocol = _protocol()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "PREREGISTERED_BEFORE_FORMAL"
    assert protocol["protocol_id"] == "stage5_4_amendment2_formal"
    assert config["protocol_sha256"] == file_sha256(PROTOCOL_PATH)
    assert protocol["preregistration"]["ts3_algorithm_head"] == "09443b2f585e2c1c73d2f868c7765b69d03ae7a8"
    assert protocol["final_freeze_criteria"]["no_automatic_freeze"]


def test_amendment2_formal_reuses_frozen_stage5_4_gates_without_relaxation():
    formal_config = json.loads(TS1_FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = _protocol()
    assert config["decision_gates"]["temporal_benefit"] == formal_config["decision_gates"]["temporal_benefit"]
    assert (
        config["decision_gates"]["spatial_regression_guardrails"]
        == formal_config["decision_gates"]["spatial_regression_guardrails"]
    )
    assert protocol["decision_rule"]["frozen_stage5_4_temporal_benefit_gates_applied_to_ts3_vs_ts0"] == (
        config["decision_gates"]["temporal_benefit"]
    )
    assert protocol["decision_rule"]["frozen_stage5_4_spatial_regression_guardrails_applied_to_ts3_vs_ts0"] == (
        config["decision_gates"]["spatial_regression_guardrails"]
    )


def test_amendment2_formal_config_freezes_ts3_and_four_arm_analysis_sets():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    smoothing = config["temporal_smoothing"]
    assert smoothing["alpha"] == 0.5
    assert smoothing["reset_rules"] == ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]
    assert smoothing["ts3"]["method"] == "guarded_constrained_ema_v1"
    assert smoothing["ts3"]["correction_feedback_to_ema_state"] is False
    assert smoothing["ts3"]["extra_hyperparameters"] == []
    assert smoothing["ts2"]["method"] == "motion_adaptive_ema_v1"
    assert config["composition"]["ts3"] == "guarded_constrained_ema_v1"
    assert config["manifest"]["expected_frame_count"] == 51256
    assert config["analysis_sets"]["full_dev166"]["expected_frame_count"] == 51256
    assert config["analysis_sets"]["confirmatory_dev142"]["expected_video_count"] == 142
    assert config["analysis_sets"]["confirmatory_dev142"]["expected_frame_count"] == 43820
    assert len(config["analysis_sets"]["confirmatory_dev142"]["excluded_smoke_video_ids"]) == 24
    assert config["heldout_lock"]["allowed_access"] == 0


def test_dev142_exclusion_matches_local_smoke24_manifest():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    excluded = set(config["analysis_sets"]["confirmatory_dev142"]["excluded_smoke_video_ids"])
    if not SMOKE_MANIFEST_PATH.is_file():
        pytest.skip("local smoke manifest archive copy not available")
    smoke = json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))
    smoke_ids = {str(video["video_id"]) for video in smoke["videos"]}
    assert len(smoke_ids) == 24
    assert smoke_ids == excluded


def test_dev142_canonical_definition_matches_ts1_formal_wording():
    """The pinned Dev142 identity d00f99a0... was computed with the TS-1 Formal wording."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ts1_formal = json.loads(TS1_FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["analysis_sets"]["confirmatory_dev142"]["definition"] == (
        ts1_formal["analysis_sets"]["confirmatory_dev142"]["definition"]
    )
    assert config["analysis_sets"]["confirmatory_dev142"]["definition"] == (
        "Frozen Dev166 minus Stage5.4 Smoke24 whole video_ids"
    )
    assert config["analysis_sets"]["confirmatory_dev142"]["identity_sha256"] == (
        "d00f99a044fc67a7ae484185cab5e8f94928ec3be350e2b27ac26250b72b7434"
    )


def test_validate_formal_contract_accepts_amendment2_formal():
    config = _config()
    smoke = {"manifest_sha256": "smoke", "videos": [{"video_id": "b"}]}
    report = validate_formal_contract(config, _protocol(), _manifest(), smoke)
    assert report["validation"] == "PASS"
    assert report["confirmatory_smoke_overlap"] == 0
    assert report["full_dev166"]["video_count"] == 3
    assert report["confirmatory_dev142"]["video_count"] == 2


def test_validate_formal_contract_rejects_drift():
    config = _config()
    protocol = _protocol()
    smoke = {"manifest_sha256": "smoke", "videos": [{"video_id": "b"}]}

    drifted = json.loads(json.dumps(config))
    drifted["temporal_smoothing"]["ts3"]["correction_feedback_to_ema_state"] = True
    with pytest.raises(ValueError, match="TS-3 guard definition drifted"):
        validate_formal_contract(drifted, protocol, _manifest(), smoke)

    drifted = json.loads(json.dumps(config))
    drifted["temporal_smoothing"]["alpha"] = 0.4
    with pytest.raises(ValueError, match="alpha"):
        validate_formal_contract(drifted, protocol, _manifest(), smoke)

    drifted = json.loads(json.dumps(config))
    drifted["temporal_smoothing"]["ts3"]["extra_hyperparameters"] = ["visible_threshold"]
    with pytest.raises(ValueError, match="parameter-free"):
        validate_formal_contract(drifted, protocol, _manifest(), smoke)

    drifted = json.loads(json.dumps(config))
    drifted["temporal_smoothing"]["ts2"]["full_response_motion_norm"] = 0.3
    with pytest.raises(ValueError, match="TS-2 ablation schedule drifted"):
        validate_formal_contract(drifted, protocol, _manifest(), smoke)

    drifted = json.loads(json.dumps(config))
    drifted["analysis_sets"]["confirmatory_dev142"]["excluded_smoke_video_ids"] = ["c"]
    with pytest.raises(ValueError, match="Smoke24 video identity"):
        validate_formal_contract(drifted, protocol, _manifest(), smoke)

    wrong_protocol = json.loads(json.dumps(protocol))
    wrong_protocol["protocol_id"] = "stage5_4_formal"
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        validate_formal_contract(config, wrong_protocol, _manifest(), smoke)

    draft = json.loads(json.dumps(protocol))
    draft["status"] = "DRAFT_READY_FOR_SMOKE"
    with pytest.raises(ValueError, match="protocol status is invalid"):
        validate_formal_contract(config, draft, _manifest(), smoke)


W = 1600
CROP_W = 506
CROP_H_FLOAT = 899.555555


def _ts3_record(
    video_id: str,
    frame: int,
    stratum: str,
    *,
    guarded: bool = False,
    ts0_x: int | None = None,
    visible: tuple[float, float, float, float] = (0.80, 0.79, 0.795, 0.795),
) -> dict:
    """Full four-arm record: TS-0 jitters, TS-1/TS-3 are stable, TS-2 follows TS-0."""
    xs = {
        "ts0": 400 if frame % 2 == 0 else 700 if ts0_x is None else ts0_x,
        "ts1": 500,
        "ts2": 400 if frame % 2 == 0 else 700,
        "ts3": 520 if guarded else 500,
    }
    record = {
        "video_id": video_id,
        "frame": frame,
        "image_width": W,
        "image_height": 900,
        "stratum": stratum,
        "sanitized": {"status": "OK", "xyxy": [700.0, 400.0, 760.0, 460.0]},
    }
    for index, side in enumerate(("ts0", "ts1", "ts2", "ts3")):
        record[side] = {
            "x": xs[side],
            "y": 0,
            "w": CROP_W,
            "h": CROP_H_FLOAT,
            "reset_reason": None,
            "subject_visible_fraction": visible[index],
            "subject_center_inside": True,
        }
    record["ts0"]["fallback"] = False
    record["ts3"].update(
        {
            "guard_applied": guarded,
            "guard_correction_x": 20.0 if guarded else 0.0,
            "guard_correction_y": 0.0,
        }
    )
    return record


def test_guard_projection_diagnostics_counts_runs_axes_and_strata():
    records = [
        _ts3_record("a", 0, "strongly_off_center", guarded=True),
        _ts3_record("a", 1, "strongly_off_center", guarded=True),
        _ts3_record("a", 2, "strongly_off_center", guarded=False),
        _ts3_record("a", 3, "near_center", guarded=True),
        _ts3_record("a", 9, "near_center", guarded=True),
    ]
    diagnostics = guard_projection_diagnostics(records)
    assert diagnostics["role"] == "DIAGNOSTIC_ONLY"
    assert diagnostics["eligible_nonfallback_frames"] == 5
    assert diagnostics["guard_applied_frames"] == 4
    assert diagnostics["guard_applied_rate"] == 0.8
    assert diagnostics["horizontal_applied_frames"] == 4
    assert diagnostics["vertical_applied_frames"] == 0
    assert diagnostics["both_axes_applied_frames"] == 0
    assert diagnostics["guard_run_count"] == 3
    assert diagnostics["max_guard_run_length"] == 2
    assert diagnostics["strata"]["strongly_off_center"]["guard_applied_frames"] == 2
    assert diagnostics["strata"]["near_center"]["guard_applied_rate"] == 1.0


def test_amendment2_formal_gates_must_pass_on_both_analysis_sets():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    passing_records = [
        _ts3_record("a", 0, "near_center"),
        _ts3_record("a", 1, "near_center"),
        _ts3_record("a", 2, "strongly_off_center", visible=(0.95, 0.93, 0.94, 0.94)),
        _ts3_record("a", 3, "strongly_off_center", visible=(0.95, 0.93, 0.94, 0.94)),
    ]
    result = evaluate_amendment2_formal_scientific_gates(
        {"full_dev166": passing_records, "confirmatory_dev142": passing_records},
        config["decision_gates"],
    )
    assert result["all_pass"] is True

    failing_records = [
        _ts3_record("a", 0, "near_center"),
        _ts3_record("a", 1, "near_center"),
        _ts3_record("a", 2, "strongly_off_center", visible=(0.95, 0.93, 0.94, 0.50)),
        _ts3_record("a", 3, "strongly_off_center", visible=(0.95, 0.93, 0.94, 0.50)),
    ]
    result = evaluate_amendment2_formal_scientific_gates(
        {"full_dev166": passing_records, "confirmatory_dev142": failing_records},
        config["decision_gates"],
    )
    assert result["all_pass"] is False
    assert result["analysis_sets"]["full_dev166"]["all_pass"] is True
    assert result["analysis_sets"]["confirmatory_dev142"]["all_pass"] is False


def test_build_analysis_metrics_all_treatments_reports_guard_block():
    records = [
        _ts3_record("a", 0, "near_center", guarded=True),
        _ts3_record("a", 1, "near_center"),
        _ts3_record("a", 2, "near_center"),
    ]
    metrics = build_analysis_metrics_all_treatments(records)
    assert metrics["frames"] == 3
    assert set(metrics["temporal_stability_pooled"]) == {
        "ts0_displacement",
        "ts0_acceleration",
        "ts1_displacement",
        "ts1_displacement_smoothed_only",
        "ts1_acceleration",
        "ts1_acceleration_smoothed_only",
        "ts2_displacement",
        "ts2_displacement_smoothed_only",
        "ts2_acceleration",
        "ts2_acceleration_smoothed_only",
        "ts3_displacement",
        "ts3_displacement_smoothed_only",
        "ts3_acceleration",
        "ts3_acceleration_smoothed_only",
    }
    assert metrics["guard_projection_diagnostics"]["guard_applied_frames"] == 1
    assert metrics["spatial_guardrails"]["strata"]["near_center"]["ts3_visible"] is not None
