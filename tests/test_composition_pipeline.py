"""Stage 5.3 composition pipeline end-to-end tests on synthetic frozen inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.experiments.stage5.run_stage5_3_composition import load_deferred_raw_inputs
from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputError,
    InputBinding,
    aggregate_metrics,
    box_too_large_diagnostic,
    build_manifest,
    compose_all_frames,
    compose_frame,
    crosscheck_mirror_against_artifact,
    engineering_gate,
    fallback_reason_diagnostic,
    load_frozen_inputs,
    mirror_reliable_candidate,
    multi_subject_diagnostic,
)
from aic_video_highlight.spatial_composition.official_emission import write_and_validate_official
from aic_video_highlight.spatial_localization.subject_localization import SubjectPolicyConfig

TARGET_RATIO = [9, 16]
POLICY_CONFIG = SubjectPolicyConfig(person_priority=True, target_ratio=(9, 16))

VIDEO_A = "qvh_testa_9x16"
VIDEO_B = "qvh_testb_9x16"


def candidate(box, score, label, label_id=0):
    return {"box_xyxy": list(box), "score": score, "label_id": label_id, "label": label}


def policy_record(video_id, frame, status, primary=None, reasons=(), ambiguous=False, ambiguous_count=0):
    return {
        "video_id": video_id,
        "frame": frame,
        "status": status,
        "primary": primary,
        "fallback_reasons": list(reasons),
        "ambiguous": ambiguous,
        "ambiguous_candidate_count": ambiguous_count,
    }


def synthetic_inputs(tmp_path: Path):
    width_a, height_a = 534, 300
    width_b, height_b = 270, 480
    metadata = {
        "records": {
            VIDEO_A: {"video_id": VIDEO_A, "width": width_a, "height": height_a, "frame_count": 4496},
            VIDEO_B: {"video_id": VIDEO_B, "width": width_b, "height": height_b, "frame_count": 2000},
        },
        "schema": "aic.spatial-video-metadata/v1",
    }
    predictions = [
        {
            "video_id": VIDEO_A,
            "targetRatioWH": TARGET_RATIO,
            "predictions": [
                {"frame": frame, "bboxes": [183, 0, 168]} for frame in range(6)
            ],
        },
        {
            "video_id": VIDEO_B,
            "targetRatioWH": TARGET_RATIO,
            "predictions": [
                {"frame": frame, "bboxes": [0, 0, 270]} for frame in range(2)
            ],
        },
    ]
    index = [
        {"video_id": VIDEO_A, "video_path": "videos/dev/a.mp4", "targetRatioWH": TARGET_RATIO, "source_group": "a"},
        {"video_id": VIDEO_B, "video_path": "videos/dev/b.mp4", "targetRatioWH": TARGET_RATIO, "source_group": "b"},
    ]
    crop_w_a = (height_a * 9) // 16
    center_x_a = (width_a - crop_w_a) // 2
    policy = [
        policy_record(
            VIDEO_A, 0, "PRIMARY",
            primary={"xyxy": [200.0, 100.0, 260.0, 200.0], "score": 0.9, "label": "person"},
        ),
        policy_record(
            VIDEO_A, 1, "PRIMARY",
            primary={"xyxy": [20.0, 100.0, 80.0, 200.0], "score": 0.8, "label": "person"},
        ),
        policy_record(
            VIDEO_A, 2, "PRIMARY",
            primary={"xyxy": [300.0, 120.0, 420.0, 240.0], "score": 0.7, "label": "car"},
        ),
        policy_record(VIDEO_A, 3, "CENTER_CROP_FALLBACK", reasons=["NO_DETECTION"]),
        policy_record(
            VIDEO_A, 4, "PRIMARY",
            primary={"xyxy": [100.0, 100.0, 200.0, 220.0], "score": 0.9, "label": "person"},
            ambiguous=True, ambiguous_count=1,
        ),
        policy_record(
            VIDEO_A, 5, "PRIMARY",
            primary={"xyxy": [600.0, 100.0, 650.0, 200.0], "score": 0.6, "label": "person"},
        ),
        policy_record(VIDEO_B, 0, "CENTER_CROP_FALLBACK", reasons=["BOX_TOO_LARGE"]),
        policy_record(
            VIDEO_B, 1, "PRIMARY",
            primary={"xyxy": [90.0, 200.0, 150.0, 300.0], "score": 0.85, "label": "person"},
        ),
    ]
    raw_frames = [
        {
            "video_id": VIDEO_A, "frame": 0, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [
                candidate((200.0, 100.0, 260.0, 200.0), 0.9, "person"),
                candidate((10.0, 10.0, 40.0, 40.0), 0.2, "chair"),
            ],
        },
        {
            "video_id": VIDEO_A, "frame": 1, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [candidate((20.0, 100.0, 80.0, 200.0), 0.8, "person")],
        },
        {
            "video_id": VIDEO_A, "frame": 2, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [candidate((300.0, 120.0, 420.0, 240.0), 0.7, "car", 2)],
        },
        {
            "video_id": VIDEO_A, "frame": 3, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [],
        },
        {
            "video_id": VIDEO_A, "frame": 4, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [
                candidate((100.0, 100.0, 200.0, 220.0), 0.9, "person"),
                candidate((380.0, 100.0, 480.0, 220.0), 0.87, "person"),
            ],
        },
        {
            "video_id": VIDEO_A, "frame": 5, "image_width": width_a, "image_height": height_a, "model_error": None,
            "candidates": [candidate((600.0, 100.0, 650.0, 200.0), 0.6, "person")],
        },
        {
            "video_id": VIDEO_B, "frame": 0, "image_width": width_b, "image_height": height_b, "model_error": None,
            "candidates": [candidate((0.0, 0.0, 270.0, 480.0), 0.95, "person")],
        },
        {
            "video_id": VIDEO_B, "frame": 1, "image_width": width_b, "image_height": height_b, "model_error": None,
            "candidates": [candidate((90.0, 200.0, 150.0, 300.0), 0.85, "person")],
        },
    ]
    weak_reference = [
        {"video_id": VIDEO_A, "rois": {"0": [183.0, 0.0, 168.0, 300.0], "1": [60.0, 0.0, 168.0, 300.0]}, "targetRatioWH": TARGET_RATIO},
    ]
    return write_inputs(tmp_path, metadata, predictions, index, policy, raw_frames, weak_reference)


def write_inputs(tmp_path, metadata, predictions, index, policy, raw_frames, weak_reference):
    paths = {
        "video_metadata_cache": tmp_path / "metadata_cache.json",
        "stage5_1_predictions": tmp_path / "predictions.jsonl",
        "dev166_index": tmp_path / "index.jsonl",
        "stage5_2_policy_artifact": tmp_path / "policy_v1.json",
        "stage5_2_raw_detector": tmp_path / "raw_detector.json",
        "weak_spatial_reference": tmp_path / "weak.jsonl",
    }
    paths["video_metadata_cache"].write_text(json.dumps(metadata), encoding="utf-8")
    paths["stage5_1_predictions"].write_text(
        "\n".join(json.dumps(record) for record in predictions) + "\n", encoding="utf-8"
    )
    paths["dev166_index"].write_text(
        "\n".join(json.dumps(record) for record in index) + "\n", encoding="utf-8"
    )
    paths["stage5_2_policy_artifact"].write_text(json.dumps(policy), encoding="utf-8")
    paths["stage5_2_raw_detector"].write_text(json.dumps({"frames": raw_frames}), encoding="utf-8")
    paths["weak_spatial_reference"].write_text(
        "\n".join(json.dumps(record) for record in weak_reference) + "\n", encoding="utf-8"
    )
    bindings = [
        InputBinding(name, path, file_sha256(path), "jsonl" if name in ("stage5_1_predictions", "dev166_index", "weak_spatial_reference") else "json")
        for name, path in paths.items()
    ]
    return bindings


def test_load_verifies_hashes_and_cross_checks(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    assert set(inputs.input_hashes) == {binding.name for binding in bindings}
    tampered = [InputBinding(b.name, b.path, "0" * 64, b.format) for b in bindings]
    with pytest.raises(FrozenInputError):
        load_frozen_inputs(tampered)


def test_deferred_raw_loading_rehydrates_inputs_for_diagnostics(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    raw_binding = next(b for b in bindings if b.name == "stage5_2_raw_detector")
    raw_rows = json.loads(raw_binding.path.read_text(encoding="utf-8"))["frames"]
    raw_dir = tmp_path / "raw_detector"
    raw_dir.mkdir()
    (raw_dir / "synthetic.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n",
        encoding="utf-8",
    )
    deferred_bindings = [
        InputBinding(binding.name, raw_dir, "", "raw_shard_dir")
        if binding.name == "stage5_2_raw_detector"
        else binding
        for binding in bindings
    ]

    inputs = load_deferred_raw_inputs(deferred_bindings)

    assert len(inputs.raw_frames) == len(raw_rows)
    assert crosscheck_mirror_against_artifact(inputs, POLICY_CONFIG) == {
        "compared": 6,
        "mismatches": 0,
    }
    assert multi_subject_diagnostic(inputs, TARGET_RATIO, POLICY_CONFIG)["ambiguous_frames"] == 1
    assert fallback_reason_diagnostic(inputs, TARGET_RATIO, POLICY_CONFIG)["fallback_frames"] == 2


def test_manifest_is_deterministic_and_model_blind(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    kwargs = {
        "manifest_id": "stage5_3_smoke_manifest_v1",
        "strata_thresholds": {"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        "target_ratio": TARGET_RATIO,
    }
    first = build_manifest(inputs, **kwargs)
    second = build_manifest(inputs, **kwargs)
    assert first == second
    assert first["frame_count"] == 8
    assert first["video_count"] == 2
    assert first["strata_counts"]["near_center"] == 2
    assert first["strata_counts"]["strongly_off_center"] == 1
    assert first["strata_counts"]["moderately_off_center"] == 2
    assert first["strata_counts"]["no_subject"] == 3


def test_composition_preserves_frame_identity_and_cmp0(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    manifest = build_manifest(
        inputs,
        manifest_id="m",
        strata_thresholds={"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        target_ratio=TARGET_RATIO,
    )
    records = compose_all_frames(manifest, inputs, TARGET_RATIO)
    gate = engineering_gate(manifest, records)
    assert gate["frame_identity_complete"]
    assert gate["missing"] == 0 and gate["extra"] == 0 and gate["duplicates"] == 0
    assert gate["invalid_crop"] == 0 and gate["out_of_bounds"] == 0 and gate["ratio_violations"] == 0
    assert gate["cmp0_frozen_regression"] == 0
    for record in records:
        assert record["cmp0"]["x"] == (183 if record["video_id"] == VIDEO_A else 0)
    assert all(record["cmp0"]["w"] == (168 if record["video_id"] == VIDEO_A else 270) for record in records)


def test_cmp1_shifts_toward_subject_and_falls_back_correctly(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    manifest = build_manifest(
        inputs,
        manifest_id="m",
        strata_thresholds={"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        target_ratio=TARGET_RATIO,
    )
    records = {(r["video_id"], r["frame"]): r for r in compose_all_frames(manifest, inputs, TARGET_RATIO)}

    near = records[(VIDEO_A, 0)]
    assert near["cmp1"]["placement_status"] == "SUBJECT_SHIFTED"
    assert near["cmp1"]["x"] == 146
    assert near["cmp1"]["subject_visible_fraction"] == pytest.approx(1.0)
    assert near["cmp0"]["subject_visible_fraction"] == pytest.approx(1.0)
    assert near["cmp0"]["subject_center_inside"] is True
    assert near["cmp1"]["subject_center_inside"] is True

    off = records[(VIDEO_A, 1)]
    assert off["cmp1"]["x"] == 0
    assert off["cmp1"]["placement_status"] == "SUBJECT_SHIFTED"
    assert off["cmp1"]["subject_visible_fraction"] == pytest.approx(1.0)
    assert off["cmp1"]["subject_visible_fraction"] > off["cmp0"]["subject_visible_fraction"]

    non_person = records[(VIDEO_A, 2)]
    assert non_person["primary_label"] == "car"
    assert non_person["cmp1"]["placement_status"] == "SUBJECT_SHIFTED"

    fallback = records[(VIDEO_A, 3)]
    assert fallback["cmp1"]["placement_status"] == "FALLBACK_CENTER_CROP"
    assert fallback["cmp1"]["fallback"] is True
    assert fallback["sanitized"]["status"] == "PRIMARY_ABSENT"

    invalid = records[(VIDEO_A, 5)]
    assert invalid["sanitized"]["status"] == "INVALID_SANITIZED_SUBJECT"
    assert invalid["cmp1"]["placement_status"] == "FALLBACK_CENTER_CROP"

    assert records[(VIDEO_A, 0)]["weak_reference"] is not None
    assert records[(VIDEO_A, 2)]["weak_reference"] is None


def test_deterministic_composition_output(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    manifest = build_manifest(
        inputs,
        manifest_id="m",
        strata_thresholds={"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        target_ratio=TARGET_RATIO,
    )
    first = compose_all_frames(manifest, inputs, TARGET_RATIO)
    second = compose_all_frames(manifest, inputs, TARGET_RATIO)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_metrics_report_visibility_and_strata(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    thresholds = {"near_center_lt": 0.10, "strongly_off_center_gte": 0.25}
    manifest = build_manifest(inputs, manifest_id="m", strata_thresholds=thresholds, target_ratio=TARGET_RATIO)
    records = compose_all_frames(manifest, inputs, TARGET_RATIO, thresholds)
    metrics = aggregate_metrics(manifest, records, thresholds)
    assert metrics["engineering_gate"]["frame_identity_complete"]
    assert metrics["composition_counts"]["cmp1_fallback"] == 3
    assert metrics["strata"]["strongly_off_center"]["mean_improvement"] > 0
    assert metrics["strata"]["moderately_off_center"]["mean_improvement"] > 0
    assert metrics["subject_center_inside"]["cmp1_rate"] > metrics["subject_center_inside"]["cmp0_rate"]
    assert metrics["selected_primary_overflow_audit"]["frames_with_primary_bbox"] == 6
    assert metrics["weak_reference_diagnostic"]["frames_with_reference"] == 2


def test_official_emission_passes_independent_validator(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    manifest = build_manifest(
        inputs,
        manifest_id="m",
        strata_thresholds={"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        target_ratio=TARGET_RATIO,
    )
    records = compose_all_frames(manifest, inputs, TARGET_RATIO)
    temporal = {
        video_id: {int(pred["frame"]) for pred in record["predictions"]}
        for video_id, record in inputs.stage5_1_predictions.items()
    }
    result = write_and_validate_official(
        records, "cmp1", TARGET_RATIO, tmp_path / "cmp1.jsonl",
        inputs.metadata, inputs.index, temporal,
    )
    assert result["is_valid"] and result["issue_count"] == 0
    assert result["predictions"] == 8
    assert (tmp_path / "cmp1.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_mirror_crosscheck_and_diagnostics(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    crosscheck = crosscheck_mirror_against_artifact(inputs, POLICY_CONFIG)
    assert crosscheck == {"compared": 6, "mismatches": 0}

    too_large = box_too_large_diagnostic(inputs, TARGET_RATIO, POLICY_CONFIG)
    assert too_large["box_too_large_frames"] == 1
    row = too_large["rows"][0]
    assert row["trigger_candidate_found"] and row["class"] == "person"
    assert row["hypothetical_cmp1_visible_fraction"] == pytest.approx(1.0)

    multi = multi_subject_diagnostic(inputs, TARGET_RATIO, POLICY_CONFIG)
    assert multi["ambiguous_frames"] == 1
    row = multi["rows"][0]
    assert row["secondary_count"] == 1
    assert row["composition"] == "person+person"
    assert row["secondary_centers"][0]["inside_cmp0_crop"] is False
    assert row["secondary_centers"][0]["inside_cmp1_crop"] is False
    assert "person+person" in multi["composition_summary"]

    mirrored, invalid_count = mirror_reliable_candidate(
        inputs.raw_frames[(VIDEO_A, 0)]["candidates"], POLICY_CONFIG
    )
    assert mirrored.label == "person" and invalid_count == 0


def test_fallback_reason_diagnostic_reports_counts_and_classes(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    diagnostic = fallback_reason_diagnostic(inputs, TARGET_RATIO, POLICY_CONFIG)
    assert diagnostic["fallback_frames"] == 2
    no_detection = diagnostic["reasons"]["NO_DETECTION"]
    assert no_detection["count"] == 1 and no_detection["trigger_candidate_found"] == 0
    too_large = diagnostic["reasons"]["BOX_TOO_LARGE"]
    assert too_large["count"] == 1 and too_large["trigger_candidate_found"] == 1
    assert too_large["class_distribution"] == {"person": 1}
    assert too_large["bbox_area_ratio"]["mean"] == pytest.approx(1.0)


def test_gate_detects_tampered_frozen_predictions(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    manifest = build_manifest(
        inputs,
        manifest_id="m",
        strata_thresholds={"near_center_lt": 0.10, "strongly_off_center_gte": 0.25},
        target_ratio=TARGET_RATIO,
    )
    records = compose_all_frames(manifest, inputs, TARGET_RATIO)
    tampered = json.loads(json.dumps(records[0]))
    tampered["cmp0"]["x"] = 999
    gate = engineering_gate(manifest, [*records, tampered])
    assert gate["duplicates"] == 1 and gate["extra"] == 0


def test_compose_frame_accepts_precomputed_frozen_boxes(tmp_path):
    bindings = synthetic_inputs(tmp_path)
    inputs = load_frozen_inputs(bindings)
    frozen = {0: [183, 0, 168], 1: [183, 0, 168], 2: [183, 0, 168], 3: [183, 0, 168], 4: [183, 0, 168], 5: [183, 0, 168]}
    record = compose_frame(VIDEO_A, 4, inputs, TARGET_RATIO, {"near_center_lt": 0.10, "strongly_off_center_gte": 0.25}, frozen)
    assert record["frame"] == 4
    assert math.isfinite(record["cmp1"]["h"])
