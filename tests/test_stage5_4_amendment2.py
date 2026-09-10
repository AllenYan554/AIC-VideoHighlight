"""Stage 5.4 Amendment 2 constrained-EMA Smoke infrastructure tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from scripts.experiments.stage5.run_stage5_4_temporal import validate_amendment2_contract
from scripts.experiments.stage5 import run_stage5_4_temporal as temporal_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment2_smoke.json"
PROTOCOL_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment2_smoke_protocol.json"


def load_documents() -> tuple[dict, dict]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
    )


def test_amendment2_is_sha_bound_smoke_draft_not_formal_preregistration():
    config, protocol = load_documents()
    assert protocol["status"] == "DRAFT_READY_FOR_SMOKE"
    assert protocol["status"] != "PREREGISTERED_BEFORE_FORMAL"
    assert config["protocol_sha256"] == file_sha256(PROTOCOL_PATH)
    assert config["experiment_id"] == protocol["protocol_id"]


def test_ts3_has_fixed_alpha_and_only_parameter_free_center_constraint():
    config, protocol = load_documents()
    smoothing = config["temporal_smoothing"]
    ts3 = smoothing["ts3"]
    assert smoothing["alpha"] == 0.5
    assert ts3 == {
        "method": "guarded_constrained_ema_v1",
        "proposal": "ema_crop_center_v1 fixed alpha=0.5",
        "constraint": "current_frozen_primary_subject_center_inside_final_crop",
        "projection": "nearest_point_on_safe_integer_top_left_rectangle",
        "correction_feedback_to_ema_state": False,
        "extra_hyperparameters": [],
    }
    assert protocol["guard_constraint"]["extra_hyperparameters"] == []
    assert protocol["scientific_variables"]["only_changed_variable"].startswith(
        "Post-EMA placement projection"
    )


def test_amendment2_reuses_exact_smoke24_and_reports_all_four_treatments():
    config, protocol = load_documents()
    smoke = protocol["smoke"]
    assert smoke["manifest_sha256"] == config["manifest"]["expected_manifest_sha256"]
    assert (smoke["videos"], smoke["frames"]) == (24, 1080)
    assert protocol["metrics"]["treatments"] == ["TS-0", "TS-1", "TS-2", "TS-3"]
    strata = smoke["coverage_audit"]["spatial_strata_frames"]
    assert all(strata[key] > 0 for key in ("near_center", "moderately_off_center", "strongly_off_center"))


def test_amendment2_reuses_frozen_stage5_4_gates_without_relaxation():
    config, protocol = load_documents()
    formal = json.loads(
        (REPO_ROOT / "configs/experiments/stage5/stage5_4_formal.json").read_text(encoding="utf-8")
    )
    assert config["decision_gates"]["temporal_benefit"] == formal["decision_gates"]["temporal_benefit"]
    assert config["decision_gates"]["spatial_regression_guardrails"] == formal["decision_gates"]["spatial_regression_guardrails"]
    frozen = protocol["decision_rule"]["frozen_stage5_4_gates"]
    assert frozen["temporal_benefit"] == formal["decision_gates"]["temporal_benefit"]
    assert frozen["spatial_guardrails"] == formal["decision_gates"]["spatial_regression_guardrails"]


def test_amendment2_static_contract_and_heldout_lock():
    config, protocol = load_documents()
    manifest = {
        "manifest_id": "stage5_4_smoke_manifest_v1",
        "manifest_sha256": "2909f7a831a1f90fb149f0b07033586ab2678ead398180d3715c2ee077e2d766",
        "video_count": 24,
        "frame_count": 1080,
    }
    result = validate_amendment2_contract(config, protocol, manifest)
    assert result["validation"] == "PASS"
    assert result["heldout_access"] == 0
    assert result["gpu_requirement"] == "NONE"
    assert config["heldout_lock"]["allowed_access"] == 0
    forbidden_tokens = ("heldout", "hard", "official_test", "official-test")
    for name, entry in config["inputs"].items():
        candidate = f"{name} {entry.get('path', '')}".lower()
        assert not any(token in candidate for token in forbidden_tokens)


def test_amendment2_runtime_progress_resume_and_collision_contract():
    config, protocol = load_documents()
    assert config["runtime"]["gpu"] == "NONE"
    assert config["runtime"]["resume"] is True
    assert config["runtime"]["validate_only"] is True
    assert protocol["runtime"]["output_root"].endswith("/stage5_4_amendment2_smoke/")
    assert "refuses" in protocol["runtime"]["output_collision"]
    required = {
        "videos", "frames", "current_video", "current_sequence", "elapsed", "ETA",
        "errors", "invalid", "resume state",
    }
    assert required <= set(config["runtime"]["progress_fields"])


def test_amendment2_video_records_keep_four_matched_sides_and_valid_geometry(monkeypatch):
    centers = {0: 200.0, 1: 1400.0}

    def fake_compose(video_id, frame, *_args):
        center = centers[frame]
        return {
            "video_id": video_id,
            "frame": frame,
            "stage5_2_status": "RELIABLE",
            "fallback_reasons": [],
            "ambiguous": False,
            "ambiguous_candidate_count": 0,
            "stratum": "strongly_off_center",
            "horizontal_center_offset": 0.375,
            "sanitized": {
                "status": "OK",
                "xyxy": [center - 20.0, 400.0, center + 20.0, 500.0],
                "clamp_left": 0.0,
                "clamp_top": 0.0,
                "clamp_right": 0.0,
                "clamp_bottom": 0.0,
            },
            "cmp1": {
                "x": max(0, min(int(center - 253), 1094)),
                "y": 0,
                "w": 506,
                "h": 506 * 16 / 9,
                "crop_w": 506,
                "crop_h": 900,
                "placement_status": "SUBJECT_SHIFTED",
                "fallback": False,
                "subject_visible_fraction": 1.0,
                "subject_center_inside": True,
            },
        }

    monkeypatch.setattr(temporal_runner, "compose_frame", fake_compose)
    monkeypatch.setattr(temporal_runner, "crosscheck_manifest_entry", lambda *_args: [])
    video = {
        "video_id": "v",
        "image_width": 1600,
        "image_height": 900,
        "frames": [{"frame": 0}, {"frame": 1}],
    }
    inputs = SimpleNamespace(
        stage5_1_predictions={
            "v": {"predictions": [{"frame": 0, "bboxes": []}, {"frame": 1, "bboxes": []}]}
        }
    )
    records, mismatches = temporal_runner.build_video_records(
        video,
        inputs,
        [9, 16],
        {"near_center_lt": 0.1, "strongly_off_center_gte": 0.25},
        0.5,
        9,
        16,
        include_ts2=True,
        include_ts3=True,
    )
    assert mismatches == []
    assert all(set(("ts0", "ts1", "ts2", "ts3")) <= set(record) for record in records)
    assert all(all(record["geometry_valid"].values()) for record in records)
    assert all(record["ts3"]["subject_center_inside"] for record in records)
    assert [(record["video_id"], record["frame"]) for record in records] == [("v", 0), ("v", 1)]
