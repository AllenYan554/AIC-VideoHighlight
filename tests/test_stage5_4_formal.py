"""Stage 5.4 Formal preregistration and decision-gate contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from scripts.experiments.stage5.run_stage5_4_temporal import (
    build_analysis_set_identity,
    crosscheck_manifest_entry,
    evaluate_formal_scientific_gates,
    validate_formal_contract,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_formal.json"
PROTOCOL_PATH = REPO_ROOT / "configs/experiments/stage5/stage5_4_formal_protocol.json"


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


def _analysis_metrics(
    *,
    displacement=(0.10, 0.07, 0.20, 0.19),
    acceleration=(0.10, 0.06),
    jumps=((0.20, 0.15), (0.10, 0.05), (0.04, 0.02)),
    spatial=(0.80, 0.78, 0.50, 0.47, 1.0, 0.99, 0.90, 0.87),
) -> dict:
    d0, d1, p950, p951 = displacement
    a0, a1 = acceleration
    mean0, mean1, v900, v901, inside0, inside1, strong0, strong1 = spatial
    return {
        "temporal_stability_pooled": {
            "ts0_displacement": {
                "mean": d0,
                "p95": p950,
                "large_jump_ratios_descriptive_only": {
                    ">0.10": jumps[0][0], ">0.20": jumps[1][0], ">0.30": jumps[2][0]
                },
            },
            "ts1_displacement": {
                "mean": d1,
                "p95": p951,
                "large_jump_ratios_descriptive_only": {
                    ">0.10": jumps[0][1], ">0.20": jumps[1][1], ">0.30": jumps[2][1]
                },
            },
            "ts0_acceleration": {"mean": a0},
            "ts1_acceleration": {"mean": a1},
        },
        "spatial_guardrails": {
            "ts0": {"mean": mean0, "thresholds": {">=0.90": v900}, "subject_center_inside_rate": inside0},
            "ts1": {"mean": mean1, "thresholds": {">=0.90": v901}, "subject_center_inside_rate": inside1},
            "strata": {
                "strongly_off_center": {
                    "ts0_visible": {"mean": strong0},
                    "ts1_visible": {"mean": strong1},
                }
            },
        },
    }


def _gates() -> dict:
    return {
        "temporal_benefit": {
            "minimum_mean_displacement_relative_reduction": 0.20,
            "minimum_mean_acceleration_relative_reduction": 0.30,
            "maximum_p95_displacement_relative_regression": 0.0,
            "large_jump_nonincrease_thresholds": [0.10, 0.20, 0.30],
            "minimum_gt_0_20_relative_reduction": 0.25,
        },
        "spatial_regression_guardrails": {
            "minimum_mean_visible_fraction_delta": -0.03,
            "minimum_visible_ge_0_90_delta": -0.05,
            "minimum_subject_center_containment_delta": -0.02,
            "minimum_strong_off_center_mean_visible_delta": -0.05,
        },
    }


def test_confirmatory_identity_is_whole_video_set_subtraction():
    manifest = _manifest()
    full = build_analysis_set_identity(manifest, "Frozen Dev166", "full", {"a", "b", "c"})
    confirmatory = build_analysis_set_identity(
        manifest, "Confirmatory Dev142", "Frozen Dev166 minus Smoke24 whole video_ids", {"a", "c"}
    )
    assert full["video_count"] == 3
    assert full["frame_count"] == 6
    assert confirmatory["video_count"] == 2
    assert confirmatory["frame_count"] == 5
    assert confirmatory["video_ids"] == ["a", "c"]
    assert len(confirmatory["identity_sha256"]) == 64


def test_formal_scientific_gates_apply_to_full_and_confirmatory_sets():
    metrics = {"full_dev166": _analysis_metrics(), "confirmatory_dev142": _analysis_metrics()}
    result = evaluate_formal_scientific_gates(metrics, _gates())
    assert result["all_pass"] is True
    assert all(item["pass"] for item in result["analysis_sets"].values())

    metrics["confirmatory_dev142"] = _analysis_metrics(displacement=(0.10, 0.081, 0.20, 0.19))
    result = evaluate_formal_scientific_gates(metrics, _gates())
    assert result["all_pass"] is False
    assert result["analysis_sets"]["confirmatory_dev142"]["temporal_benefit"][
        "mean_displacement_relative_reduction"
    ]["pass"] is False


def test_formal_manifest_crosscheck_uses_frozen_stage53_fields_without_smoke_only_keys():
    entry = {
        "frame": 0,
        "stratum": "near_center",
        "stage5_2_status": "PRIMARY",
        "horizontal_center_offset": 0.05,
    }
    composed = {
        "video_id": "v",
        "frame": 0,
        "stratum": "near_center",
        "stage5_2_status": "PRIMARY",
        "horizontal_center_offset": 0.05,
        "ambiguous": False,
        "cmp1": {"fallback": False},
        "sanitized": {"xyxy": [0.0, 0.0, 2.0, 2.0]},
    }
    assert crosscheck_manifest_entry(entry, composed) == []
    entry["stage5_2_status"] = "CENTER_CROP_FALLBACK"
    assert crosscheck_manifest_entry(entry, composed) == ["v:0 stage5_2_status"]


def test_formal_config_and_protocol_freeze_required_identities():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "stage5_4_formal"
    assert config["temporal_smoothing"]["alpha"] == 0.5
    assert config["temporal_smoothing"]["reset_rules"] == ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]
    assert config["manifest"]["expected_manifest_sha256"] == (
        "2af7d66468fc8d5546f2fabf20aba2f799a17d76ce7c8f4c281fb930539cb24d"
    )
    assert config["inputs"]["stage5_3_frozen_cmp1"]["sha256"] == (
        "dce61c4ce1e83358a02a0c4a0ef38666eae2726fa116826980506997005580b1"
    )
    assert config["analysis_sets"]["full_dev166"]["expected_video_count"] == 166
    assert config["analysis_sets"]["full_dev166"]["expected_frame_count"] == 51256
    assert config["analysis_sets"]["confirmatory_dev142"]["expected_video_count"] == 142
    assert config["analysis_sets"]["confirmatory_dev142"]["expected_frame_count"] == 43820
    assert protocol["status"] in {"DRAFT", "PREREGISTERED_BEFORE_FORMAL"}
    if config["protocol_sha256"] is not None:
        assert config["protocol_sha256"] == file_sha256(PROTOCOL_PATH)


def test_validate_formal_contract_rejects_smoke_overlap_or_identity_drift():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest = _manifest()
    smoke = {"manifest_sha256": "smoke", "videos": [{"video_id": "b"}]}
    config = json.loads(json.dumps(config))
    config["manifest"]["expected_manifest_sha256"] = "formal-manifest-sha"
    config["manifest"]["expected_video_count"] = 3
    config["manifest"]["expected_frame_count"] = 6
    config["smoke_binding"]["expected_manifest_sha256"] = "smoke"
    full_spec = config["analysis_sets"]["full_dev166"]
    confirm_spec = config["analysis_sets"]["confirmatory_dev142"]
    full = build_analysis_set_identity(
        manifest, full_spec["name"], full_spec["definition"], {"a", "b", "c"}
    )
    confirm = build_analysis_set_identity(
        manifest, confirm_spec["name"], confirm_spec["definition"], {"a", "c"}
    )
    config["analysis_sets"]["full_dev166"].update(
        expected_video_count=3, expected_frame_count=6, identity_sha256=full["identity_sha256"]
    )
    config["analysis_sets"]["confirmatory_dev142"].update(
        expected_video_count=2,
        expected_frame_count=5,
        excluded_smoke_video_ids=["b"],
        identity_sha256=confirm["identity_sha256"],
    )
    report = validate_formal_contract(config, protocol, manifest, smoke)
    assert report["validation"] == "PASS"
    assert report["confirmatory_smoke_overlap"] == 0

    config["analysis_sets"]["confirmatory_dev142"]["excluded_smoke_video_ids"] = ["c"]
    with pytest.raises(ValueError, match="Smoke24 video identity"):
        validate_formal_contract(config, protocol, manifest, smoke)
