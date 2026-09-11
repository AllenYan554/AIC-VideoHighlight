"""TS-5 Revised canonical-center projected-state tests; no real experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.center_crop import derived_height
from aic_video_highlight.spatial_composition.composition_pipeline import FrozenInputError
from aic_video_highlight.spatial_composition.subject_shifted_crop import stage5_1_crop_height
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    PLACEMENT_TS5_CANONICAL_STATE_SMOOTHED,
    RESET_FALLBACK,
    RESET_FRAME_GAP,
    TemporalObservation,
    canonical_center_from_placement,
    place_crop_from_center,
    project_bbox_maximum_visibility,
    SmoothedFrame,
    smooth_video_sequence_bbox_guarded,
    smooth_video_sequence_canonical_center_projected_state_bbox_guarded,
    smooth_video_sequence_projected_state_bbox_guarded,
)
from scripts.experiments.stage5 import run as stage5_registry
from scripts.experiments.stage5 import run_stage5_4_temporal as runner


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs/experiments/stage5"
SMOKE_CONFIG = BASE / "stage5_4_amendment4_revised_smoke.json"
FORMAL_CONFIG = BASE / "stage5_4_amendment4_revised_formal.json"
SMOKE_PROTOCOL = BASE / "stage5_4_amendment4_revised_smoke_protocol.json"
FORMAL_PROTOCOL = BASE / "stage5_4_amendment4_revised_formal_protocol.json"
MASTER = BASE / "stage5_4_amendment4_revised_master_preregistration.json"

TW, TH = 9.0, 16.0
W, H = 1600, 900

# Real original-TS-5 definition-failure geometry (qvh_000194_9x16, frame 27).
FAIL_IMAGE = (534, 300)
FAIL_BBOX = (222.13531529903412, 3.081884980201721, 533.992711186409, 299.32301938533783)
FAIL_STATE = (361.1223478449658, 145.41726868748614)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic(protocol: dict) -> str:
    return canonical_sha256({
        key: value for key, value in protocol.items()
        if key != "protocol_semantic_sha256"
    })


def obs(frame: int, cx: float, cy: float = 450.0, fallback: bool = False) -> TemporalObservation:
    return TemporalObservation(frame=frame, fallback=fallback, ideal_center_x=cx, ideal_center_y=cy)


def _assert_revised_invariants(frames: list[SmoothedFrame], width: int, height: int) -> None:
    for item in frames:
        if item.ema_center_x is None:
            continue
        assert item.placement_status == PLACEMENT_TS5_CANONICAL_STATE_SMOOTHED
        assert item.ema_center_x == canonical_center_from_placement(
            item.x, item.y, item.w, item.crop_h
        )[0]
        assert item.ema_center_y == canonical_center_from_placement(
            item.x, item.y, item.w, item.crop_h
        )[1]
        reconstructed = place_crop_from_center(
            width, height, TW, TH, (item.ema_center_x, item.ema_center_y)
        )[:2]
        assert reconstructed == (item.x, item.y)
        assert item.state_output_residual_l1 == 0.0


# ---------------------------------------------------------------------------
# Frozen preregistration identity
# ---------------------------------------------------------------------------

def test_revised_master_binds_both_configs_and_protocol_hashes():
    master = _load(MASTER)
    smoke_config, formal_config = _load(SMOKE_CONFIG), _load(FORMAL_CONFIG)
    smoke_protocol, formal_protocol = _load(SMOKE_PROTOCOL), _load(FORMAL_PROTOCOL)
    assert master["status"] == runner.TS5_REVISED_MASTER_STATUS
    for role, config_path, protocol_path, protocol in (
        ("smoke", SMOKE_CONFIG, SMOKE_PROTOCOL, smoke_protocol),
        ("formal", FORMAL_CONFIG, FORMAL_PROTOCOL, formal_protocol),
    ):
        binding = master["artifacts"][role]
        assert binding["config_sha256"] == file_sha256(config_path)
        assert binding["protocol_byte_sha256"] == file_sha256(protocol_path)
        assert binding["protocol_semantic_sha256"] == _semantic(protocol)
        config = _load(config_path)
        assert config["protocol_sha256"] == file_sha256(protocol_path)
        assert config["protocol_semantic_sha256"] == protocol["protocol_semantic_sha256"]
        assert runner._validate_master_preregistration(config, protocol)
        assert config["temporal_smoothing"]["ts5"] == runner._expected_ts5_revised_definition()
    assert smoke_config["decision_gates"] == smoke_protocol["decision_gates"]
    assert formal_config["decision_gates"] == formal_protocol["decision_gates"]
    assert formal_config["promotion_authorization"]["expected_smoke_config_sha256"] == file_sha256(
        SMOKE_CONFIG
    )
    assert formal_config["promotion_authorization"]["expected_smoke_protocol_sha256"] == file_sha256(
        SMOKE_PROTOCOL
    )
    assert formal_config["promotion_authorization"]["expected_smoke_protocol_semantic_sha256"] == _semantic(
        smoke_protocol
    )


def test_revised_registry_and_revised_smoke_protocol_are_distinct_from_history():
    for experiment in ("stage5_4_amendment4_revised_smoke", "stage5_4_amendment4_revised_formal"):
        assert experiment in stage5_registry.RUNNERS
        assert stage5_registry.LAUNCH[experiment] == {
            "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
            "forbid_active_processes": ["vllm", "qwen"],
        }
    assert _load(SMOKE_PROTOCOL)["protocol_semantic_sha256"] != (
        "cdfc8724daf045e3265f2030a1cdbbf0ba06430692efbef1590170fd6194fabf"
    )
    assert _load(FORMAL_PROTOCOL)["protocol_semantic_sha256"] != (
        "290e212bc8307763484911fba0efa49b51bdb05e6f0dcbd570e6b15d9bbd58c2"
    )
    assert file_sha256(SMOKE_CONFIG) != "b4cdc9b0ba8509c3c71dd9c91dc11260c37ad20165e489cfe94b2ba01291b894"
    assert file_sha256(FORMAL_CONFIG) != "81317d49f9e31d5c04f311d75c28fec52e0bb650d5d523147731fe28b4118d4c"


def test_revised_formal_blocked_before_execution_without_exact_smoke_evidence(tmp_path):
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


def test_revised_adjudication_and_marker_are_fail_closed_and_exact():
    passed = {
        "status": "PASS", "gates": {"engineering": True},
        "scientific_gates": {"status": "PASS", "all_pass": True},
        "deterministic_replay": True,
    }
    decision = runner.adjudicate_ts5_revised_formal(passed)
    assert decision["status"] == "TS5_REVISED_FINAL_FROZEN"
    assert decision["stage5_4_terminal"] == "STAGE5_4_CLOSED"
    failed = {**passed, "scientific_gates": {"status": "FAIL", "all_pass": False}}
    decision = runner.adjudicate_ts5_revised_formal(failed)
    assert decision["status"] == "TS5_REVISED_NOT_READY_FOR_FREEZE"
    assert decision["near_pass_allowed"] is False
    marker = runner.build_promotion_marker(
        passed, identity_ok=True,
        method=runner.TS5_REVISED_METHOD,
        smoke_manifest_identity="manifest", execution_head="head",
        protocol_byte_sha256="byte", protocol_semantic_sha256="semantic",
        config_sha256="config", validation_sha256="validation",
    )
    assert marker["all_pass"] is True
    assert marker["method"] == "projected_state_canonical_center_ema_v1"
    assert marker["override_allowed"] is False
    denied = runner.build_promotion_marker(
        failed, identity_ok=True,
        method=runner.TS5_REVISED_METHOD,
        smoke_manifest_identity="manifest", execution_head="head",
        protocol_byte_sha256="byte", protocol_semantic_sha256="semantic",
        config_sha256="config", validation_sha256="validation",
    )
    assert denied["all_pass"] is False
    assert denied["decision"] == "STOP_BEFORE_FORMAL"


def test_revised_report_generator_contract_ready(tmp_path):
    config = _load(FORMAL_CONFIG)
    output = tmp_path / "experiment_report.md"
    validation = {
        "status": "FAIL", "gates": {"engineering": True},
        "scientific_gates": {"status": "FAIL", "all_pass": False},
        "deterministic_replay": True,
    }
    runner.render_amendment4_experiment_report(
        output, config=config, summary={"ts5": runner._expected_ts5_revised_definition()},
        metrics={"ts5_projected_state_attribution": {"role": "DIAGNOSTIC_ONLY"}},
        validation=validation, runtime={"heldout_access": 0, "official_test_access": 0},
    )
    report = output.read_text(encoding="utf-8")
    assert "TS5_REVISED_NOT_READY_FOR_FREEZE" in report
    assert "STAGE5_4_CLOSED" in report
    assert all(f"## {section}" in report for section in runner.AMENDMENT4_REPORT_SECTIONS)


# ---------------------------------------------------------------------------
# Canonical lift: exact inverse of the frozen placement map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height", [(534, 300), (1600, 900), (500, 500), (1080, 1920)])
def test_canonical_center_exactly_reconstructs_every_legal_boundary_placement(width, height):
    x0, y0, cw, dh, _, _ = place_crop_from_center(width, height, TW, TH, (width / 2, height / 2))
    crop_h = stage5_1_crop_height(width, height, TW, TH)
    max_x = width - cw
    max_y = int(math.floor(height - float(derived_height(cw, TW, TH))))
    assert max_y >= 0
    for x in sorted({0, max_x // 2, max_x}):
        for y in sorted({0, max_y}):
            center = canonical_center_from_placement(x, y, cw, crop_h)
            assert place_crop_from_center(width, height, TW, TH, center)[:2] == (x, y)


def test_canonical_center_uses_frozen_internal_crop_height_and_is_unique():
    x0, y0, cw, dh, _, _ = place_crop_from_center(534, 300, TW, TH, (361.1223478449658, 145.41726868748614))
    crop_h = stage5_1_crop_height(534, 300, TW, TH)
    # The frozen internal crop height (300) is the placement-map scalar.
    assert (x0, y0, cw, crop_h) == (277, 0, 168, 300)
    emitted_x, emitted_y = 277, 1
    canonical = canonical_center_from_placement(emitted_x, emitted_y, cw, crop_h)
    assert canonical == (361.0, 151.0)
    assert place_crop_from_center(534, 300, TW, TH, canonical)[:2] == (277, 1)
    # The geometric center built from the official derived height (298.667) does
    # NOT invert the frozen map; this is exactly why the canonical lift uses the
    # placement scalar crop_h, with no tolerance or rounding parameter added.
    derived_center = (
        361.0,
        1.0 + float(derived_height(cw, TW, TH)) / 2.0,
    )
    assert place_crop_from_center(534, 300, TW, TH, derived_center)[:2] == (277, 0)


# ---------------------------------------------------------------------------
# Real frame-27 definition-failure regression
# ---------------------------------------------------------------------------

def test_original_ts5_crashes_on_real_frame27_geometry():
    with pytest.raises(RuntimeError, match="projected EMA state does not reproduce"):
        smooth_video_sequence_projected_state_bbox_guarded(
            *FAIL_IMAGE, TW, TH,
            [obs(27, FAIL_STATE[0], FAIL_STATE[1])],
            {27: FAIL_BBOX},
        )


def test_revised_ts5_reconstructs_real_frame27_geometry_exactly():
    frames = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        *FAIL_IMAGE, TW, TH,
        [obs(27, FAIL_STATE[0], FAIL_STATE[1])],
        {27: FAIL_BBOX},
    )
    item = frames[0]
    assert (item.x, item.y) == (277, 1)
    assert (item.ema_center_x, item.ema_center_y) == (361.0, 151.0)
    assert item.state_output_residual_l1 == 0.0
    assert place_crop_from_center(*FAIL_IMAGE, TW, TH, (item.ema_center_x, item.ema_center_y))[:2] == (277, 1)


# ---------------------------------------------------------------------------
# Synthetic behaviour coverage
# ---------------------------------------------------------------------------

def test_revised_static_and_linear_sequences_keep_canonical_state():
    static = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, [obs(frame, 800.0) for frame in range(6)],
        {frame: (200.0, 300.0, 1400.0, 600.0) for frame in range(6)},
    )
    _assert_revised_invariants(static, W, H)
    linear = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, [obs(frame, 500.0 + 60.0 * frame) for frame in range(8)],
        {frame: (500.0 + 60.0 * frame - 180.0, 350.0, 500.0 + 60.0 * frame + 180.0, 550.0) for frame in range(8)},
    )
    _assert_revised_invariants(linear, W, H)


def test_revised_sudden_jump_and_oversized_bbox_feed_back_canonical_center():
    observations = [obs(0, 200.0, 300.0), obs(1, 850.0, 300.0), obs(2, 850.0, 300.0)]
    bboxes = {frame: (100.0, 50.0, 900.0, 550.0) for frame in range(3)}
    projected = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    _assert_revised_invariants(projected, W, H)
    assert all(item.bbox_larger_than_crop for item in projected)
    assert projected[1].ema_center_x == projected[1].x + projected[1].w / 2
    assert projected[1].ema_center_y == projected[1].y + projected[1].crop_h / 2
    assert projected[2].proposal_center_x == pytest.approx(
        0.5 * observations[2].ideal_center_x + 0.5 * projected[1].ema_center_x
    )


def test_revised_moving_feasible_boundary_stays_feasible_and_canonical():
    observations = [obs(frame, 1200.0 + 5.0 * frame) for frame in range(30)]
    bboxes = {
        frame: (1050.0 + 5.0 * frame, 300.0, 1350.0 + 5.0 * frame, 600.0)
        for frame in range(30)
    }
    projected = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    _assert_revised_invariants(projected, W, H)
    assert all(item.safe_x_min <= item.x <= item.safe_x_max for item in projected)
    assert all(item.safe_y_min <= item.y <= item.safe_y_max for item in projected)


def test_revised_reset_and_fallback_semantics_equal_ts4():
    observations = [
        obs(0, 300.0), obs(1, 1300.0), obs(4, 1200.0),
        obs(5, 800.0, fallback=True), obs(6, 400.0),
    ]
    bboxes = {
        0: (200.0, 350.0, 400.0, 550.0),
        1: (1100.0, 350.0, 1400.0, 550.0),
        4: (1000.0, 350.0, 1300.0, 550.0),
        6: (300.0, 350.0, 500.0, 550.0),
    }
    ts4 = smooth_video_sequence_bbox_guarded(W, H, TW, TH, observations, bboxes)
    revised = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    assert (revised[0].x, revised[0].y) == (ts4[0].x, ts4[0].y)
    assert revised[2].reset_reason == ts4[2].reset_reason == RESET_FRAME_GAP
    assert revised[3].placement_status == ts4[3].placement_status
    assert revised[4].reset_reason == ts4[4].reset_reason == RESET_FALLBACK
    for item in revised:
        if item.reset_reason == RESET_FALLBACK or item.ema_center_x is None:
            continue
        assert place_crop_from_center(W, H, TW, TH, (item.ema_center_x, item.ema_center_y))[:2] == (
            item.x, item.y
        )


def test_revised_deterministic_replay_and_long_guard_active_run():
    observations = [obs(frame, 700.0, 450.0) for frame in range(40)]
    bboxes = {frame: (1300.0, 100.0, 1500.0, 800.0) for frame in range(40)}
    first = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    replay = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    assert first == replay
    _assert_revised_invariants(first, W, H)
    assert sum(1 for item in first if item.guard_applied) > 10


def test_revised_uses_the_identical_ts4_projection_operator():
    observations = [obs(0, 300.0), obs(1, 300.0)]
    bboxes = {0: (100.0, 50.0, 900.0, 550.0), 1: (100.0, 50.0, 900.0, 550.0)}
    ts4 = smooth_video_sequence_bbox_guarded(W, H, TW, TH, observations, bboxes)
    revised = smooth_video_sequence_canonical_center_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    # On the first frame both methods share the same proposal state (the
    # observation), so the reused TS-4 P_t must emit the identical placement.
    assert (revised[0].x, revised[0].y) == (ts4[0].x, ts4[0].y)
    assert revised[0].guard_applied == ts4[0].guard_applied
    assert revised[0].guard_correction_x == ts4[0].guard_correction_x
    assert revised[0].guard_correction_y == ts4[0].guard_correction_y


def test_revised_runner_plumbing_and_mechanism_evaluator(monkeypatch):
    def fake_compose(video_id, frame, *_args):
        center = [250.0, 1100.0, 300.0, 1200.0, 400.0, 1000.0][frame]
        return {
            "video_id": video_id,
            "frame": frame,
            "stage5_2_status": "RELIABLE",
            "fallback_reasons": [],
            "ambiguous": False,
            "ambiguous_candidate_count": 0,
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

    monkeypatch.setattr(runner, "compose_frame", fake_compose)
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
        include_ts5=True, ts5_revised=True,
    )
    assert mismatches == []
    assert all(record["ts5"]["state_output_residual_l1"] == 0.0 for record in records)
    for record in records:
        item = record["ts5"]
        if item["reset_reason"] is not None:
            continue
        assert item["ema_center_x"] == item["x"] + item["w"] / 2
        assert item["ema_center_y"] == item["y"] + item["crop_h"] / 2
        assert place_crop_from_center(1600, 900, 9, 16, (item["ema_center_x"], item["ema_center_y"]))[:2] == (
            item["x"], item["y"]
        )
    result = runner.evaluate_ts5_revised_scientific_gates(
        records,
        runner.pooled_distributions(records),
        runner.spatial_metrics_for_all_treatments(records),
        _load(SMOKE_CONFIG)["decision_gates"],
    )
    checks = result["ts5_revised_canonical_state_checks"]
    assert all(check["pass"] for check in checks.values())
    assert result["amendment4_mechanism_checks"]
    diagnostic = runner.projected_state_attribution_diagnostics(records)
    assert diagnostic["role"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["decision_inputs"] is False
    assert diagnostic["state_output_residual_l1_px"]["max"] == 0.0
