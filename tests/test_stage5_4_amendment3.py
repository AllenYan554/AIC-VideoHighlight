"""Stage 5.4 Amendment 3 five-arm and promotion-contract tests (no real experiment)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from scripts.experiments.stage5 import run_stage5_4_temporal as temporal_runner
from scripts.experiments.stage5.run_stage5_4_amendment3_pipeline import (
    _resume_flags,
    validate_pipeline_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment3_smoke.json"
SMOKE_PROTOCOL = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment3_smoke_protocol.json"
FORMAL_CONFIG = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment3_formal.json"
FORMAL_PROTOCOL = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment3_formal_protocol.json"
PIPELINE_CONFIG = REPO_ROOT / "configs/experiments/stage5/stage5_4_amendment3_pipeline.json"


@dataclass(frozen=True)
class _Inputs:
    policy_records: tuple
    raw_frames: dict
    input_hashes: dict
    stage5_1_predictions: dict


def _fake_compose(video_id, frame, *_args):
    center = {0: 300.0, 1: 1300.0}[frame]
    x1, x2 = (center - 180.0, center + 180.0)
    return {
        "video_id": video_id,
        "frame": frame,
        "stage5_2_status": "RELIABLE",
        "fallback_reasons": [],
        "ambiguous": False,
        "ambiguous_candidate_count": 0,
        "stratum": "strongly_off_center",
        "horizontal_center_offset": 0.3125,
        "sanitized": {
            "status": "OK", "xyxy": [x1, 350.0, x2, 550.0],
            "clamp_left": 0.0, "clamp_top": 0.0, "clamp_right": 0.0, "clamp_bottom": 0.0,
        },
        "cmp1": {
            "x": max(0, min(int(center - 253), 1094)), "y": 0, "w": 506,
            "h": 506 * 16 / 9, "crop_w": 506, "crop_h": 900,
            "placement_status": "SUBJECT_SHIFTED", "fallback": False,
            "subject_visible_fraction": 1.0, "subject_center_inside": True,
        },
    }


def test_five_arm_video_records_include_bbox_guard_and_preserve_prior_arms(monkeypatch):
    monkeypatch.setattr(temporal_runner, "compose_frame", _fake_compose)
    monkeypatch.setattr(temporal_runner, "crosscheck_manifest_entry", lambda *_args: [])
    video = {
        "video_id": "v", "image_width": 1600, "image_height": 900,
        "frames": [{"frame": 0}, {"frame": 1}],
    }
    inputs = SimpleNamespace(
        stage5_1_predictions={
            "v": {"predictions": [{"frame": 0, "bboxes": []}, {"frame": 1, "bboxes": []}]}
        }
    )
    records, mismatches = temporal_runner.build_video_records(
        video, inputs, [9, 16],
        {"near_center_lt": 0.1, "strongly_off_center_gte": 0.25},
        0.5, 9, 16, include_ts2=True, include_ts3=True, include_ts4=True,
    )
    assert mismatches == []
    assert all(set(("ts0", "ts1", "ts2", "ts3", "ts4")) <= set(record) for record in records)
    assert all(all(record["geometry_valid"].values()) for record in records)
    assert all(record["ts4"]["subject_visible_fraction"] >= record["ts3"]["subject_visible_fraction"] for record in records)
    assert records[1]["ts4"]["guard_applied"]
    assert records[1]["ts4"]["visible_gain"] > 0.0


def test_amendment3_protocols_are_jointly_preregistered_and_sha_bound():
    smoke_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    smoke_protocol = json.loads(SMOKE_PROTOCOL.read_text(encoding="utf-8"))
    formal_config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    formal_protocol = json.loads(FORMAL_PROTOCOL.read_text(encoding="utf-8"))
    expected_status = "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT"
    assert smoke_protocol["status"] == formal_protocol["status"] == expected_status
    assert smoke_config["protocol_sha256"] == file_sha256(SMOKE_PROTOCOL)
    assert formal_config["protocol_sha256"] == file_sha256(FORMAL_PROTOCOL)
    assert smoke_config["temporal_smoothing"]["ts4"] == formal_config["temporal_smoothing"]["ts4"]
    assert smoke_config["temporal_smoothing"]["ts4"]["extra_hyperparameters"] == []
    assert formal_protocol["execution_authorization"] == "SMOKE_PASS_TO_FORMAL_ONLY"


@pytest.mark.parametrize(
    ("engineering", "scientific_status", "scientific_pass", "identity_ok", "authorized"),
    [
        (True, "PASS", True, True, True),
        (True, "PASS", False, True, False),
        (False, "PASS", True, True, False),
        (True, "EXIT_2", False, True, False),
        (True, "PASS", True, False, False),
    ],
)
def test_smoke_to_formal_promotion_scenarios(
    engineering, scientific_status, scientific_pass, identity_ok, authorized
):
    validation = {
        "status": "PASS" if engineering and scientific_pass else "FAIL",
        "gates": {"engineering": engineering},
        "scientific_gates": {"status": scientific_status, "all_pass": scientific_pass},
    }
    decision = temporal_runner.evaluate_smoke_promotion(
        validation, identity_ok=identity_ok
    )
    assert decision["formal_authorized"] is authorized
    assert decision["override_allowed"] is False


def test_formal_report_contract_declares_all_required_sections():
    contract = temporal_runner.AMENDMENT3_FORMAL_REPORT_SECTIONS
    assert {
        "History", "Method", "TS-0~TS-4", "Smoke result", "Full Dev166", "Dev142",
        "Temporal", "Spatial", "Strata", "BBox diagnostics", "Guard diagnostics",
        "Multi-subject", "Fallback", "Engineering gates", "Scientific gates",
        "Mechanism checks", "Limitations", "Conclusion", "Artifact paths",
        "Git/Protocol/Config identities",
    } <= set(contract)


def test_pipeline_contract_freezes_both_stages_and_disables_override():
    config = json.loads(PIPELINE_CONFIG.read_text(encoding="utf-8"))
    result = validate_pipeline_contract(config)
    assert result["validation"] == "PASS"
    assert all(result["checks"].values())
    assert config["override_allowed"] is False


def test_pipeline_only_forwards_resume_for_an_existing_run(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    assert _resume_flags(False, output) == ()
    assert _resume_flags(True, output) == ()
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert _resume_flags(True, output) == ("--resume",)


def test_five_arm_formal_synthetic_run_reaches_reports_and_machine_contract(tmp_path, monkeypatch):
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config["protocol"] = str(FORMAL_PROTOCOL)
    config_path = tmp_path / "formal.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps({
        "name": "synthetic", "repo": str(REPO_ROOT),
        "datasets": str(tmp_path / "datasets"), "models": str(tmp_path / "models"),
        "hf_cache": str(tmp_path / "hf-cache"), "outputs": str(tmp_path / "outputs"),
        "logs": str(tmp_path / "logs"), "cache": str(tmp_path / "cache"),
        "tmp": str(tmp_path / "tmp"), "archive": str(tmp_path / "archive"),
    }), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = {
        "manifest_id": "synthetic", "manifest_sha256": "synthetic", "video_count": 2,
        "frame_count": 14,
        "videos": [{
            "video_id": video_id, "image_width": 1600, "image_height": 900,
            "frame_count": 7, "frames": [{"frame": frame} for frame in range(7)],
        } for video_id in ("a", "b")],
    }
    predictions = {
        video_id: {"predictions": [{"frame": frame, "bboxes": []} for frame in range(7)]}
        for video_id in ("a", "b")
    }
    inputs = _Inputs(policy_records=(), raw_frames={}, input_hashes={}, stage5_1_predictions=predictions)
    centers = [500.0, 540.0, 650.0, 700.0, 300.0, 1300.0]
    strata = ["near_center", "near_center", "moderately_off_center", "moderately_off_center", "strongly_off_center", "strongly_off_center"]

    def compose(video_id, frame, *_args):
        if frame == 6:
            return {
                "video_id": video_id, "frame": frame, "stage5_2_status": "FALLBACK",
                "fallback_reasons": ["PRIMARY_ABSENT"], "ambiguous": False,
                "ambiguous_candidate_count": 0, "stratum": "no_subject",
                "horizontal_center_offset": None,
                "sanitized": {"status": "PRIMARY_ABSENT", "xyxy": None},
                "cmp1": {"x": 547, "y": 0, "w": 506, "h": 506 * 16 / 9,
                         "crop_w": 506, "crop_h": 900, "placement_status": "FALLBACK_CENTER_CROP",
                         "fallback": True, "subject_visible_fraction": None, "subject_center_inside": False},
            }
        center = centers[frame]
        return {
            "video_id": video_id, "frame": frame, "stage5_2_status": "RELIABLE",
            "fallback_reasons": [], "ambiguous": frame == 4, "ambiguous_candidate_count": 2 if frame == 4 else 0,
            "stratum": strata[frame], "horizontal_center_offset": abs(center / 1600 - 0.5),
            "sanitized": {"status": "OK", "xyxy": [center - 180, 350.0, center + 180, 550.0],
                          "clamp_left": 0.0, "clamp_top": 0.0, "clamp_right": 0.0, "clamp_bottom": 0.0},
            "cmp1": {"x": max(0, min(int(center - 253), 1094)), "y": 0, "w": 506,
                     "h": 506 * 16 / 9, "crop_w": 506, "crop_h": 900,
                     "placement_status": "SUBJECT_SHIFTED", "fallback": False,
                     "subject_visible_fraction": 1.0, "subject_center_inside": True},
        }

    frozen = {
        video_id: {
            frame: ([547, 0, 506] if frame == 6 else [max(0, min(int(centers[frame] - 253), 1094)), 0, 506])
            for frame in range(7)
        } for video_id in ("a", "b")
    }
    bindings = [SimpleNamespace(name="stage5_2_raw_detector", format="raw_shard_dir", path=raw_dir)]
    monkeypatch.setattr(temporal_runner, "resolve_bindings", lambda *_args: bindings)
    monkeypatch.setattr(temporal_runner, "load_frozen_manifest", lambda *_args: manifest)
    monkeypatch.setattr(temporal_runner, "load_frozen_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(temporal_runner, "load_manifest_binding", lambda *_args: {"manifest_sha256": "smoke", "videos": [{"video_id": "b"}]})
    monkeypatch.setattr(temporal_runner, "validate_formal_contract", lambda *_args: {"full_dev166": {"video_ids": ["a", "b"], "identity_sha256": "full"}, "confirmatory_dev142": {"video_ids": ["a"], "identity_sha256": "confirm"}})
    monkeypatch.setattr(temporal_runner, "load_amendment3_promotion_evidence", lambda *_args: {"decision": {"formal_authorized": True}, "validation": {"status": "PASS"}})
    monkeypatch.setattr(temporal_runner, "validate_formal_frozen_dependencies", lambda *_args: {"validation": "PASS"})
    monkeypatch.setattr(temporal_runner, "load_frozen_ts0_predictions", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(temporal_runner, "compose_frame", compose)
    monkeypatch.setattr(temporal_runner, "crosscheck_manifest_entry", lambda *_args: [])
    monkeypatch.setattr(temporal_runner, "verify_input_bindings", lambda *_args: {})
    monkeypatch.setattr(temporal_runner, "load_raw_shard_dir_subset", lambda *_args: {})

    def multi(_inputs, records, _ratio, _policy):
        rows = [{"video_id": video_id, "frame": 4, "primary_center_inside_ts0_crop": True,
                 "primary_center_inside_ts1_crop": True, "secondary_count": 1,
                 "secondary_centers_inside_ts0_crop": 1, "secondary_centers_inside_ts1_crop": 1}
                for video_id in sorted({record["video_id"] for record in records})]
        return temporal_runner.summarize_multi_subject_rows(rows, candidate_side="ts1")

    monkeypatch.setattr(temporal_runner, "temporal_multi_subject_diagnostic", multi)
    monkeypatch.setattr(temporal_runner, "evaluate_amendment3_formal_scientific_gates", lambda *_args: {"status": "PASS", "analysis_sets": {}, "all_pass": True})
    args = SimpleNamespace(config=config_path, environment=environment_path, dry_run=False, validate_only=False, resume=False)
    assert temporal_runner.run(args) == 0
    output = tmp_path / "outputs/stage5/stage5_4_amendment3_formal"
    summary = json.loads((output / "machine/summary.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "machine/validation.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "machine/metrics.json").read_text(encoding="utf-8"))
    assert summary["ts4"]["method"] == "bbox_aware_constrained_ema_v1"
    assert validation["status"] == "PASS"
    assert metrics["bbox_guard_diagnostics"]["eligible"] == 12
    for path in ("experiment_report.md", "experiment_raw_report.md", "AI_REPORT_INPUTS.md", "machine/artifact_manifest.json"):
        assert (output / path).is_file()
    report = (output / "experiment_report.md").read_text(encoding="utf-8")
    assert all(f"## {section}" in report for section in temporal_runner.AMENDMENT3_FORMAL_REPORT_SECTIONS)
