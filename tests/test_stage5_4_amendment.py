"""Stage 5.4 Amendment 1 preregistered-development infrastructure tests."""

from __future__ import annotations

import json
from pathlib import Path

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from scripts.experiments.stage5.run_stage5_4_temporal import validate_amendment_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment_smoke.json"
PROTOCOL_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment_smoke_protocol.json"


def load_documents() -> tuple[dict, dict]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
    )


def test_amendment_protocol_is_sha_bound_draft_not_formal_preregistration():
    config, protocol = load_documents()
    assert protocol["status"] == "DRAFT_READY_FOR_SMOKE"
    assert protocol["status"] != "PREREGISTERED_BEFORE_FORMAL"
    assert config["protocol_sha256"] == file_sha256(PROTOCOL_PATH)
    assert config["experiment_id"] == protocol["protocol_id"]


def test_amendment_schedule_has_one_candidate_and_reuses_frozen_motion_scale():
    config, protocol = load_documents()
    schedule = protocol["alpha_schedule"]
    ts2 = config["temporal_smoothing"]["ts2"]
    assert protocol["parameter_candidate_set"]["enabled"] is False
    assert ts2["parameter_candidate_set"] is False
    assert schedule["alpha_min"] == ts2["alpha_min"] == 0.5
    assert schedule["smoothing_ceiling_motion_norm"] == ts2["smoothing_ceiling_motion_norm"] == 0.1
    assert schedule["full_response_motion_norm"] == ts2["full_response_motion_norm"] == 0.2
    assert protocol["motion_definition"]["equation"] == "m_t = abs(X_t - X_(t-1)) / W"


def test_amendment_reuses_exact_smoke24_and_covers_motion_and_position_strata():
    config, protocol = load_documents()
    smoke = protocol["smoke"]
    assert smoke["manifest_sha256"] == config["manifest"]["expected_manifest_sha256"]
    assert (smoke["videos"], smoke["frames"]) == (24, 1080)
    transition_counts = smoke["coverage_audit"]["motion_transitions_by_frozen_schedule"]
    assert all(transition_counts[key] > 0 for key in ("m_le_0_10", "m_gt_0_10_lt_0_20", "m_gte_0_20"))
    strata = smoke["coverage_audit"]["spatial_strata_frames"]
    assert all(strata[key] > 0 for key in ("near_center", "moderately_off_center", "strongly_off_center"))


def test_amendment_gates_are_identical_to_frozen_stage5_4_formal_gates():
    config, protocol = load_documents()
    formal = json.loads(
        (REPO_ROOT / "configs/experiments/stage5/stage5_4_formal.json").read_text(encoding="utf-8")
    )
    assert config["decision_gates"]["temporal_benefit"] == formal["decision_gates"]["temporal_benefit"]
    assert config["decision_gates"]["spatial_regression_guardrails"] == formal["decision_gates"]["spatial_regression_guardrails"]
    frozen = protocol["decision_rule"]["frozen_stage5_4_gates"]
    assert frozen["temporal_benefit"] == formal["decision_gates"]["temporal_benefit"]
    assert frozen["spatial_guardrails"] == formal["decision_gates"]["spatial_regression_guardrails"]


def test_amendment_contract_static_validator_and_heldout_lock():
    config, protocol = load_documents()
    manifest = {
        "manifest_id": "stage5_4_smoke_manifest_v1",
        "manifest_sha256": "2909f7a831a1f90fb149f0b07033586ab2678ead398180d3715c2ee077e2d766",
        "video_count": 24,
        "frame_count": 1080,
    }
    result = validate_amendment_contract(config, protocol, manifest)
    assert result["validation"] == "PASS"
    assert result["heldout_access"] == 0
    assert config["heldout_lock"]["allowed_access"] == 0
    assert protocol["heldout_lock"]["allowed_access"] == 0
    forbidden_tokens = ("heldout", "hard", "official_test", "official-test")
    for name, entry in config["inputs"].items():
        candidate = f"{name} {entry.get('path', '')}".lower()
        assert not any(token in candidate for token in forbidden_tokens)


def test_amendment_runtime_and_output_collision_contract():
    config, protocol = load_documents()
    assert config["runtime"]["gpu"] == "NONE"
    assert config["runtime"]["resume"] is True
    assert config["runtime"]["validate_only"] is True
    assert protocol["runtime"]["output_root"].endswith("/stage5_4_amendment_smoke/")
    assert "refuses" in protocol["runtime"]["output_collision"]
    assert set(("videos", "frames", "current_video", "current_sequence", "elapsed", "ETA", "errors", "invalid")) <= set(config["runtime"]["progress_fields"])
