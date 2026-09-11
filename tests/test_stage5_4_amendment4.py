"""TS-5 Amendment 4 development/preregistration tests; no real experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.composition_pipeline import FrozenInputError
from scripts.experiments.stage5 import run as stage5_registry
from scripts.experiments.stage5 import run_stage5_4_temporal as runner


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs/experiments/stage5"
SMOKE_CONFIG = BASE / "stage5_4_amendment4_smoke.json"
FORMAL_CONFIG = BASE / "stage5_4_amendment4_formal.json"
SMOKE_PROTOCOL = BASE / "stage5_4_amendment4_smoke_protocol.json"
FORMAL_PROTOCOL = BASE / "stage5_4_amendment4_formal_protocol.json"
MASTER = BASE / "stage5_4_amendment4_master_preregistration.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic(protocol: dict) -> str:
    return canonical_sha256({
        key: value for key, value in protocol.items()
        if key != "protocol_semantic_sha256"
    })


def _fake_compose(video_id, frame, *_args):
    center = [250.0, 1100.0, 300.0, 1200.0, 400.0, 1000.0][frame]
    return {
        "video_id": video_id,
        "frame": frame,
        "stage5_2_status": "RELIABLE",
        "fallback_reasons": [],
        "ambiguous": frame == 2,
        "ambiguous_candidate_count": 2 if frame == 2 else 0,
        "stratum": "strongly_off_center",
        "horizontal_center_offset": 0.3,
        "sanitized": {
            "status": "OK", "xyxy": [center - 180.0, 350.0, center + 180.0, 550.0],
            "clamp_left": 0.0, "clamp_top": 0.0,
            "clamp_right": 0.0, "clamp_bottom": 0.0,
        },
        "cmp1": {
            "x": max(0, min(int(center - 253), 1094)), "y": 0, "w": 506,
            "h": 506 * 16 / 9, "crop_w": 506, "crop_h": 900,
            "placement_status": "SUBJECT_SHIFTED", "fallback": False,
            "subject_visible_fraction": 1.0, "subject_center_inside": True,
        },
    }


def test_master_binds_both_configs_and_protocol_byte_and_semantic_hashes():
    master = _load(MASTER)
    smoke_config, formal_config = _load(SMOKE_CONFIG), _load(FORMAL_CONFIG)
    smoke_protocol, formal_protocol = _load(SMOKE_PROTOCOL), _load(FORMAL_PROTOCOL)
    assert master["status"] == "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT"
    assert smoke_protocol["status"] == formal_protocol["status"] == master["status"]
    for role, config_path, config, protocol_path, protocol in (
        ("smoke", SMOKE_CONFIG, smoke_config, SMOKE_PROTOCOL, smoke_protocol),
        ("formal", FORMAL_CONFIG, formal_config, FORMAL_PROTOCOL, formal_protocol),
    ):
        binding = master["artifacts"][role]
        assert binding["config_sha256"] == file_sha256(config_path)
        assert binding["protocol_byte_sha256"] == file_sha256(protocol_path)
        assert binding["protocol_semantic_sha256"] == _semantic(protocol)
        assert config["protocol_sha256"] == file_sha256(protocol_path)
        assert config["protocol_semantic_sha256"] == protocol["protocol_semantic_sha256"] == _semantic(protocol)
        assert runner._validate_master_preregistration(config, protocol)
    assert smoke_config["decision_gates"] == smoke_protocol["decision_gates"]
    assert formal_config["decision_gates"] == formal_protocol["decision_gates"]
    assert smoke_config["temporal_smoothing"]["ts5"] == formal_config["temporal_smoothing"]["ts5"] == runner._expected_ts5_definition()
    assert formal_config["promotion_authorization"]["expected_smoke_config_sha256"] == file_sha256(SMOKE_CONFIG)


def test_six_arm_records_preserve_ts4_projection_and_feed_ts5_state(monkeypatch):
    monkeypatch.setattr(runner, "compose_frame", _fake_compose)
    monkeypatch.setattr(runner, "crosscheck_manifest_entry", lambda *_args: [])
    frames = [{"frame": frame} for frame in range(6)]
    video = {"video_id": "v", "image_width": 1600, "image_height": 900, "frames": frames}
    inputs = SimpleNamespace(stage5_1_predictions={
        "v": {"predictions": [{"frame": frame, "bboxes": []} for frame in range(6)]}
    })
    records, mismatches = runner.build_video_records(
        video, inputs, [9, 16],
        {"near_center_lt": 0.1, "strongly_off_center_gte": 0.25},
        0.5, 9, 16, include_ts2=True, include_ts3=True, include_ts4=True,
        include_ts5=True,
    )
    assert mismatches == []
    assert all({"ts0", "ts1", "ts2", "ts3", "ts4", "ts5"} <= record.keys() for record in records)
    assert all(all(record["geometry_valid"].values()) for record in records)
    assert all(record["ts5"]["state_output_residual_l1"] == 0.0 for record in records)
    assert all(record["ts5"]["subject_visible_fraction"] == record["ts4"]["subject_visible_fraction"] for record in records)
    assert any(record["ts5"]["guard_applied"] for record in records)
    diagnostic = runner.projected_state_attribution_diagnostics(records)
    assert diagnostic["role"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["decision_inputs"] is False
    assert diagnostic["state_output_residual_l1_px"]["max"] == 0.0


@pytest.mark.parametrize("identity_ok,engineering,scientific,authorized", [
    (True, True, True, True),
    (False, True, True, False),
    (True, False, True, False),
    (True, True, False, False),
])
def test_promotion_marker_is_fail_closed(identity_ok, engineering, scientific, authorized):
    validation = {
        "status": "PASS" if engineering and scientific else "FAIL",
        "gates": {"engineering": engineering},
        "scientific_gates": {"status": "PASS" if scientific else "FAIL", "all_pass": scientific},
    }
    marker = runner.build_promotion_marker(
        validation, identity_ok=identity_ok, method="projected_state_constrained_ema_v1",
        smoke_manifest_identity="manifest", execution_head="head",
        protocol_byte_sha256="byte", protocol_semantic_sha256="semantic",
        config_sha256="config", validation_sha256="validation",
    )
    assert marker["all_pass"] is authorized
    assert marker["decision"] == ("AUTHORIZE_FORMAL" if authorized else "STOP_BEFORE_FORMAL")
    assert marker["override_allowed"] is False


def test_exact_threshold_and_terminal_adjudication_forbid_near_pass():
    assert runner._gate(0.2999, 0.3)["pass"] is False
    passed = {
        "status": "PASS", "gates": {"engineering": True},
        "scientific_gates": {"status": "PASS", "all_pass": True},
        "deterministic_replay": True,
    }
    assert runner.adjudicate_ts5_formal(passed)["status"] == "TS5_FINAL_FROZEN"
    failed = {**passed, "scientific_gates": {"status": "FAIL", "all_pass": False}}
    decision = runner.adjudicate_ts5_formal(failed)
    assert decision["status"] == "TS5_NOT_READY_FOR_FREEZE"
    assert decision["stage5_4_terminal"] == "STAGE5_4_CLOSE_NO_AMENDMENT5"
    assert decision["near_pass_allowed"] is False


def test_formal_is_blocked_before_execution_without_exact_smoke_evidence(tmp_path):
    config = _load(FORMAL_CONFIG)
    env_path = tmp_path / "environment.json"
    env_path.write_text(json.dumps({
        "name": "test", "repo": str(REPO), "datasets": str(tmp_path / "datasets"),
        "models": str(tmp_path / "models"), "hf_cache": str(tmp_path / "hf"),
        "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs"),
        "cache": str(tmp_path / "cache"), "tmp": str(tmp_path / "tmp"),
        "archive": str(tmp_path / "archive"),
    }), encoding="utf-8")
    with pytest.raises(FrozenInputError, match="BLOCKED_BEFORE_EXECUTION"):
        runner.load_amendment4_promotion_evidence(config, EnvironmentPaths.from_json(env_path))


def test_smoke_preflight_emits_complete_a_to_y_audit(tmp_path, monkeypatch):
    config = _load(SMOKE_CONFIG)
    head = "a" * 40
    monkeypatch.setenv("AIC_EXPECTED_GIT_HEAD", head)
    monkeypatch.setenv("AIC_WINDOWS_ORIGIN_MASTER_HEAD", head)
    monkeypatch.setenv("AIC_WINDOWS_GIT_CLEAN", "1")

    def fake_check_output(command, **_kwargs):
        if command[0] == "git":
            key = tuple(command[-2:])
            return {
                ("rev-parse", "HEAD"): head,
                ("rev-parse", "origin/master"): head,
                ("branch", "--show-current"): "master",
                ("status", "--porcelain"): "",
            }[key]
        raise AssertionError(command)

    manifest = {
        "manifest_id": "stage5_4_smoke_manifest_v1",
        "manifest_sha256": config["manifest"]["expected_manifest_sha256"],
        "video_count": 24, "frame_count": 1080,
    }
    frozen_ts0 = {
        f"v{video:02d}": {frame: [0, 0, 506] for frame in range(45)}
        for video in range(24)
    }
    monkeypatch.setattr(runner.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(runner, "load_frozen_manifest", lambda *_args: manifest)
    monkeypatch.setattr(runner, "load_frozen_inputs", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(runner, "load_frozen_ts0_predictions", lambda *_args, **_kwargs: frozen_ts0)
    environment = EnvironmentPaths.from_dict({
        "name": "test", "repo": str(REPO), "datasets": str(tmp_path / "datasets"),
        "models": str(tmp_path / "models"), "hf_cache": str(tmp_path / "hf"),
        "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs"),
        "cache": str(tmp_path / "cache"), "tmp": str(tmp_path / "tmp"),
        "archive": str(tmp_path / "archive"),
    })
    result = runner.amendment4_execution_preflight(
        config, environment, [], runner.resolve_experiment_paths(environment, config),
        resume=False,
    )
    keys = {
        key[0] for section in ("git", "run_lifecycle", "identity")
        for key in result[section] if key[0].isalpha() and key[1:2] == "_"
    }
    assert keys == set("ABCDEFGHIJKLMNOPQRSTUVWXY")
    assert result["validation"] == "PASS"
    assert result["heldout_access"] == result["official_test_access"] == 0


def test_report_registry_and_launcher_contracts_are_ready(tmp_path):
    assert {"stage5_4_amendment4_smoke", "stage5_4_amendment4_formal"} <= stage5_registry.RUNNERS.keys()
    for experiment in ("stage5_4_amendment4_smoke", "stage5_4_amendment4_formal"):
        assert stage5_registry.LAUNCH[experiment] == {
            "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
            "forbid_active_processes": ["vllm", "qwen"],
        }
        progress_fields = _load(BASE / Path(stage5_registry.CONFIGS[experiment]).name)["runtime"]["progress_fields"]
        assert {"videos", "frames", "percent", "current_video", "current_sequence", "elapsed", "ETA", "errors", "invalid", "resume_state"} <= set(progress_fields)
    config = _load(FORMAL_CONFIG)
    output = tmp_path / "experiment_report.md"
    validation = {
        "status": "FAIL", "gates": {"engineering": True},
        "scientific_gates": {"status": "FAIL", "all_pass": False},
        "deterministic_replay": True,
    }
    runner.render_amendment4_experiment_report(
        output, config=config, summary={"ts5": runner._expected_ts5_definition()},
        metrics={"ts5_projected_state_attribution": {"role": "DIAGNOSTIC_ONLY"}},
        validation=validation, runtime={"heldout_access": 0, "official_test_access": 0},
    )
    report = output.read_text(encoding="utf-8")
    assert all(f"## {section}" in report for section in runner.AMENDMENT4_REPORT_SECTIONS)
    assert "TS5_NOT_READY_FOR_FREEZE" in report
    assert "STAGE5_4_CLOSE_NO_AMENDMENT5" in report
