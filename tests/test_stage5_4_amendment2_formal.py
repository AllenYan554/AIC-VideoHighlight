"""Stage 5.4 Amendment 2 Formal preregistration contract tests (no execution)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.experiment_runtime.run_context import RunContext
from scripts.experiments.stage5 import run_stage5_4_temporal as temporal_runner
from scripts.experiments.stage5.run_stage5_4_temporal import (
    aggregate_formal_multi_subject_diagnostics,
    build_analysis_set_identity,
    build_analysis_metrics_all_treatments,
    evaluate_amendment2_formal_scientific_gates,
    guard_projection_diagnostics,
    resolve_experiment_paths,
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


def test_engineering_amendment_uses_fresh_output_identity_without_touching_failed_run(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_dict(
        {
            "name": "synthetic",
            "repo": str(REPO_ROOT),
            "datasets": str(tmp_path / "datasets"),
            "models": str(tmp_path / "models"),
            "hf_cache": str(tmp_path / "hf-cache"),
            "outputs": str(tmp_path / "outputs"),
            "logs": str(tmp_path / "logs"),
            "cache": str(tmp_path / "cache"),
            "tmp": str(tmp_path / "tmp"),
            "archive": str(tmp_path / "archive"),
        }
    )
    failed_paths = environment.for_experiment("stage5", "stage5_4_amendment2_formal")
    failed_paths.create()
    failed_marker = failed_paths.output / "run_manifest.json"
    failed_marker.write_text('{"git_head":"b4059ae"}', encoding="utf-8")

    fresh_paths = resolve_experiment_paths(environment, config)
    assert fresh_paths.output.name == "stage5_4_amendment2_formal_engineering_amendment1"
    assert fresh_paths.output != failed_paths.output
    context = RunContext(
        config["experiment_id"],
        config["stage"],
        config["run_type"],
        REPO_ROOT,
        fresh_paths,
        CONFIG_PATH,
        PROTOCOL_PATH,
    )
    context.start()
    assert failed_marker.read_text(encoding="utf-8") == '{"git_head":"b4059ae"}'
    assert context.manifest_path.is_file()


def test_preregistration_records_postprocessing_crash_as_scientifically_null_amendment():
    protocol = _protocol()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    amendment = protocol["preregistration"]["pre_execution_engineering_amendment"]
    assert amendment["previous_execution_head"] == "b4059ae9e864818e3c0e41d71b1e1e61b05f3528"
    assert amendment["failure_stage"] == "post-processing before machine/summary.json"
    assert amendment["failure_type"] == "KeyError / engineering"
    assert amendment["formal_shards_produced"] == "166/166"
    assert amendment["scientific_aggregates_produced"] == 0
    assert amendment["scientific_results_observed"] == 0
    for key in (
        "algorithm_changed",
        "gates_changed",
        "metrics_changed",
        "dataset_changed",
        "manifest_changed",
        "ts3_changed",
    ):
        assert amendment[key] is False
    assert amendment["old_shard_reuse"] is False
    assert amendment["new_formal_starts_fresh"] is True
    assert config["execution_identity"]["previous_execution_head"] == amendment[
        "previous_execution_head"
    ]
    assert config["execution_identity"]["output_run_id"] == config["output_run_id"]


def test_engineering_amendment_preserves_preregistered_scientific_payload_hashes():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = _protocol()
    config_scientific_keys = (
        "temporal_smoothing", "composition", "manifest", "smoke_binding", "analysis_sets",
        "decision_gates", "metrics", "heldout_lock", "forbidden",
    )
    protocol_scientific_keys = (
        "research_question", "hypothesis", "scope", "scientific_variables",
        "frozen_dependencies", "smoke24_evidence_and_exclusion", "analysis_sets", "metrics",
        "guard_formal_diagnostics", "multi_subject_diagnostics", "fallback_semantics",
        "decision_rule", "engineering_gates", "determinism", "heldout_lock",
        "resources_forbidden", "interpretation_boundary",
    )
    assert canonical_sha256({key: config[key] for key in config_scientific_keys}) == (
        "c2194506953a0407450cac0ca4cda12a864260e58398cc615fd999f11de31145"
    )
    assert canonical_sha256({key: protocol[key] for key in protocol_scientific_keys}) == (
        "80844b531f7d0dcdd93942a0ef2ca553e406749218f61cda6ce1e6d86d1b43fd"
    )


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

    drifted = json.loads(json.dumps(config))
    drifted["output_run_id"] = "stage5_4_amendment2_formal"
    with pytest.raises(ValueError, match="output run identity drifted"):
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
            "crop_w": CROP_W,
            "crop_h": 900,
            "placement_status": "SUBJECT_SHIFTED" if side == "ts0" else f"{side.upper()}_SMOOTHED",
            "subject_visible_fraction": visible[index],
            "subject_center_inside": True,
        }
    record["ts0"].update({"fallback": False, "frozen_regression": False})
    record["geometry_valid"] = {
        f"{side}_{key}": True
        for side in ("ts0", "ts1", "ts2", "ts3")
        for key in ("nonnegative", "width_positive", "x_within_frame", "derived_height_within_frame")
    }
    record["ts3"].update(
        {
            "guard_applied": guarded,
            "guard_correction_x": 20.0 if guarded else 0.0,
            "guard_correction_y": 0.0,
        }
    )
    return record


def _fallback_record(video_id: str, frame: int) -> dict:
    record = _ts3_record(video_id, frame, "no_subject", visible=(None, None, None, None))
    record["sanitized"] = {"status": "INVALID_SANITIZED_SUBJECT", "xyxy": None}
    record["ts0"]["fallback"] = True
    for side in ("ts0", "ts1", "ts2", "ts3"):
        record[side]["x"] = 547
        record[side]["y"] = 0
        record[side]["placement_status"] = "FALLBACK_CENTER_CROP"
    record["ts3"].update(
        {"guard_applied": None, "guard_correction_x": None, "guard_correction_y": None}
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


def _multi_subject_payload(candidate_side: str) -> dict:
    rows = [
        {
            "video_id": "a",
            "frame": 0,
            "primary_center_inside_ts0_crop": True,
            f"primary_center_inside_{candidate_side}_crop": True,
            "secondary_count": 2,
            "secondary_centers_inside_ts0_crop": 2,
            f"secondary_centers_inside_{candidate_side}_crop": 1,
        },
        {
            "video_id": "b",
            "frame": 0,
            "primary_center_inside_ts0_crop": True,
            f"primary_center_inside_{candidate_side}_crop": True,
            "secondary_count": 1,
            "secondary_centers_inside_ts0_crop": 0,
            f"secondary_centers_inside_{candidate_side}_crop": 1,
        },
    ]
    return {"rows": rows}


def test_four_arm_formal_multi_subject_aggregation_uses_explicit_sides():
    analysis_metrics = {"full_dev166": {}, "confirmatory_dev142": {}}
    payloads = {
        f"ts0_vs_{side}": _multi_subject_payload(side) for side in ("ts1", "ts2", "ts3")
    }
    aggregate_formal_multi_subject_diagnostics(
        analysis_metrics,
        payloads,
        confirmatory_ids={"a"},
        four_arm=True,
    )
    for side in ("ts1", "ts2", "ts3"):
        comparison = f"ts0_vs_{side}"
        full = analysis_metrics["full_dev166"]["multi_subject_observation_only"][comparison]
        confirmatory = analysis_metrics["confirmatory_dev142"][
            "multi_subject_observation_only"
        ][comparison]
        assert full[f"all_secondaries_inside_{side}_rate"] == 0.5
        assert full[f"at_least_one_inside_{side}_rate"] == 1.0
        assert confirmatory[f"all_secondaries_inside_{side}_rate"] == 0.0
        assert confirmatory[f"primary_and_all_secondaries_inside_{side}_rate"] == 0.0


def test_two_arm_formal_multi_subject_aggregation_preserves_legacy_output_shape():
    analysis_metrics = {"full_dev166": {}, "confirmatory_dev142": {}}
    aggregate_formal_multi_subject_diagnostics(
        analysis_metrics,
        _multi_subject_payload("ts1"),
        confirmatory_ids={"a"},
        four_arm=False,
    )
    full = analysis_metrics["full_dev166"]["multi_subject_observation_only"]
    confirmatory = analysis_metrics["confirmatory_dev142"][
        "multi_subject_observation_only"
    ]
    assert "ts0_vs_ts1" not in full
    assert full["all_secondaries_inside_ts1_rate"] == 0.5
    assert full["at_least_one_inside_ts1_rate"] == 1.0
    assert confirmatory["all_secondaries_inside_ts1_rate"] == 0.0
    assert confirmatory["primary_and_all_secondaries_inside_ts1_rate"] == 0.0


def test_formal_multi_subject_aggregation_rejects_unknown_comparison():
    analysis_metrics = {"full_dev166": {}, "confirmatory_dev142": {}}
    with pytest.raises(ValueError, match="unknown multi-subject comparison"):
        aggregate_formal_multi_subject_diagnostics(
            analysis_metrics,
            {"ts0_vs_ts4": {"rows": []}},
            confirmatory_ids=set(),
            four_arm=True,
        )


@dataclass(frozen=True)
class _SyntheticInputs:
    policy_records: tuple
    raw_frames: dict
    input_hashes: dict


def test_four_arm_formal_run_reaches_machine_summary_and_validation_with_synthetic_data(
    tmp_path, monkeypatch
):
    strata = (
        "near_center",
        "near_center",
        "moderately_off_center",
        "moderately_off_center",
        "strongly_off_center",
        "strongly_off_center",
    )
    records_by_video = {
        video_id: [
            *[
                _ts3_record(video_id, frame, stratum, guarded=frame == 4)
                for frame, stratum in enumerate(strata)
            ],
            _fallback_record(video_id, 6),
        ]
        for video_id in ("a", "b")
    }
    manifest = {
        "manifest_id": "synthetic_formal",
        "manifest_sha256": "synthetic_manifest_sha",
        "video_count": 2,
        "frame_count": 14,
        "videos": [
            {
                "video_id": video_id,
                "frame_count": 7,
                "frames": [{"frame": frame} for frame in range(7)],
            }
            for video_id in ("a", "b")
        ],
    }
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["protocol"] = str(PROTOCOL_PATH)
    config_path = tmp_path / "formal_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "name": "synthetic",
                "repo": str(REPO_ROOT),
                "datasets": str(tmp_path / "datasets"),
                "models": str(tmp_path / "models"),
                "hf_cache": str(tmp_path / "hf-cache"),
                "outputs": str(tmp_path / "outputs"),
                "logs": str(tmp_path / "logs"),
                "cache": str(tmp_path / "cache"),
                "tmp": str(tmp_path / "tmp"),
                "archive": str(tmp_path / "archive"),
            }
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    bindings = [
        SimpleNamespace(name="stage5_2_raw_detector", format="raw_shard_dir", path=raw_dir)
    ]
    inputs = _SyntheticInputs(policy_records=(), raw_frames={}, input_hashes={})

    monkeypatch.setattr(temporal_runner, "resolve_bindings", lambda *_args: bindings)
    monkeypatch.setattr(temporal_runner, "load_frozen_manifest", lambda *_args: manifest)
    monkeypatch.setattr(temporal_runner, "load_frozen_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(
        temporal_runner,
        "load_manifest_binding",
        lambda *_args: {"manifest_sha256": "smoke", "videos": [{"video_id": "b"}]},
    )
    monkeypatch.setattr(
        temporal_runner,
        "validate_formal_contract",
        lambda *_args: {
            "full_dev166": {"video_ids": ["a", "b"], "identity_sha256": "full"},
            "confirmatory_dev142": {"video_ids": ["a"], "identity_sha256": "confirmatory"},
        },
    )
    monkeypatch.setattr(
        temporal_runner, "validate_formal_frozen_dependencies", lambda *_args: {"validation": "PASS"}
    )
    monkeypatch.setattr(
        temporal_runner,
        "load_frozen_ts0_predictions",
        lambda *_args, **_kwargs: {video_id: {} for video_id in records_by_video},
    )
    monkeypatch.setattr(
        temporal_runner,
        "build_video_records",
        lambda video, *_args, **_kwargs: (records_by_video[video["video_id"]], []),
    )
    monkeypatch.setattr(temporal_runner, "verify_input_bindings", lambda *_args: {})
    monkeypatch.setattr(temporal_runner, "load_raw_shard_dir_subset", lambda *_args: {})

    def synthetic_multi_subject(_inputs, records, _target_ratio, _policy):
        rows = [
            {
                "video_id": video_id,
                "frame": 0,
                "primary_center_inside_ts0_crop": True,
                "primary_center_inside_ts1_crop": True,
                "secondary_count": 1,
                "secondary_centers_inside_ts0_crop": 1,
                "secondary_centers_inside_ts1_crop": 1,
            }
            for video_id in sorted({record["video_id"] for record in records})
        ]
        return temporal_runner.summarize_multi_subject_rows(rows, candidate_side="ts1")

    monkeypatch.setattr(
        temporal_runner, "temporal_multi_subject_diagnostic", synthetic_multi_subject
    )
    args = SimpleNamespace(
        config=config_path,
        environment=environment_path,
        dry_run=False,
        validate_only=False,
        resume=False,
    )
    assert temporal_runner.run(args) == 0

    output = (
        tmp_path
        / "outputs/stage5/stage5_4_amendment2_formal_engineering_amendment1"
    )
    summary = json.loads((output / "machine/summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "machine/metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "machine/validation.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == "stage5_4_amendment2_formal"
    assert summary["frames"] == 14
    assert validation["status"] == "PASS"
    assert validation["engineering_gate"]["fallback_placement_unchanged"] is True
    for set_name in ("full_dev166", "confirmatory_dev142"):
        aggregate = metrics["analysis_sets"][set_name]
        assert set(aggregate["spatial_guardrails"]["strata"]) >= {
            "near_center", "moderately_off_center", "strongly_off_center"
        }
        assert aggregate["guard_projection_diagnostics"]["guard_applied_frames"] > 0
        multi = aggregate["multi_subject_observation_only"]
        assert set(multi) == {"ts0_vs_ts1", "ts0_vs_ts2", "ts0_vs_ts3"}
        assert multi["ts0_vs_ts3"]["all_secondaries_inside_ts3_rate"] == 1.0
    assert (output / "experiment_raw_report.md").is_file()
    assert (output / "AI_REPORT_INPUTS.md").is_file()
