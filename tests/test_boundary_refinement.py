"""Stage 4.4 boundary refinement tests: identity, guard, fallback, determinism."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.boundary_refinement import (
    BR0_DECISION,
    BoundaryRefinementError,
    BoundaryResponseError,
    REFINER_VERSION,
    assess_event_identity,
    assess_refinement_files,
    build_boundary_prompt,
    build_protocol_binding,
    compute_local_context_window,
    create_boundary_refinement_result as _create_boundary_refinement_result,
    load_boundary_protocol,
    parse_boundary_response,
    refine_candidates,
    refiner_config_from_protocol,
    replay_refined_predictions,
    validate_boundary_refinement_payload as _validate_boundary_refinement_payload,
)
from aic_video_highlight.highlight_retrieval.candidate_cache import (
    canonical_json_bytes,
    semantic_sha256,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (
    EXPECTED_CACHE_GLOBAL_HASH,
    compare_evaluations,
    create_role_manifest,
    evaluate_replayed_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "stage4_4_boundary_protocol.json"

HEX_A = "a" * 64
HEX_B = "b" * 64

BR1_CONFIG = {
    "refiner_name": "BR-1",
    "refiner_version": REFINER_VERSION,
    "parameters": {
        "context_padding_sec": 20.0,
        "max_context_duration_sec": 120.0,
        "min_parent_temporal_iou": 0.5,
    },
}
BR0_CONFIG = {
    "refiner_name": "BR-0",
    "refiner_version": REFINER_VERSION,
    "parameters": {},
}
TEST_GIT_HEAD = "c" * 40


def _protocol_and_bytes_hash():
    protocol = load_boundary_protocol(PROTOCOL_PATH)
    return protocol, hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def create_boundary_refinement_result(
    manifest, role_manifest, records, config, *, model_fn=None, prepare_context_fn=None
):
    protocol, bytes_hash = _protocol_and_bytes_hash()
    binding = build_protocol_binding(
        protocol,
        protocol_bytes_sha256=bytes_hash,
        refiner_name=config["refiner_name"],
        role_manifest_hash=role_manifest["semantic_sha256"],
        git_head=TEST_GIT_HEAD,
    )
    return _create_boundary_refinement_result(
        manifest,
        role_manifest,
        records,
        config,
        model_fn=model_fn,
        prepare_context_fn=prepare_context_fn,
        protocol_binding=binding,
    )


def validate_boundary_refinement_payload(result, manifest, role_manifest, records):
    protocol, bytes_hash = _protocol_and_bytes_hash()
    return _validate_boundary_refinement_payload(
        result,
        manifest,
        role_manifest,
        records,
        protocol=protocol,
        protocol_bytes_sha256=bytes_hash,
        expected_git_head=TEST_GIT_HEAD,
    )


def make_candidate(video_id: str, start: float, end: float, score: float, reason: str, chunk: int):
    return {
        "merged_candidate_id": f"cand-{video_id}-{start}-{end}",
        "start_sec": start,
        "end_sec": end,
        "score": score,
        "reason": reason,
        "source_chunk": chunk,
        "contributor_raw_candidate_ids": [f"raw-{video_id}-{start}-{end}"],
    }


def make_cache_record(video_id: str, candidates: list[dict], duration: float = 60.0):
    record = {
        "cache_schema_version": "aic.frozen-candidate-cache/v1",
        "video_id": video_id,
        "split": "dev",
        "duration_sec": duration,
        "merged_candidates": candidates,
    }
    record["semantic_sha256"] = semantic_sha256(record)
    return record


ROLE_ITEMS = [
    ("qvh_smoke_a", [make_candidate("qvh_smoke_a", 10.0, 20.0, 0.9, "关键进球瞬间", 0)]),
    (
        "qvh_smoke_b",
        [
            make_candidate("qvh_smoke_b", 5.0, 12.0, 0.7, "演讲关键结论", 0),
            make_candidate("qvh_smoke_b", 30.0, 45.0, 0.8, "实验成功演示", 1),
        ],
    ),
]


def make_role_and_cache():
    records = {video_id: make_cache_record(video_id, candidates) for video_id, candidates in ROLE_ITEMS}
    manifest = {"global_semantic_sha256": EXPECTED_CACHE_GLOBAL_HASH}
    role_manifest = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": video_id, "split": "dev"} for video_id, _ in ROLE_ITEMS],
        source_cache_global_hash=EXPECTED_CACHE_GLOBAL_HASH,
        source_cache_manifest_sha256=HEX_A,
        audit_membership_source_sha256=HEX_B,
    )
    return manifest, role_manifest, records


def trim_start_model_fn(prompt, *, video_id, candidate_id, window):
    """Deterministic fake refiner: move start 2s later, keep end (same event)."""
    original_start, original_end = _LOCAL_ORIGINAL[candidate_id]
    local_start = (original_start + 2.0) - float(window["start_sec"])
    local_end = original_end - float(window["start_sec"])
    return (
        json.dumps(
            {
                "refined_start_sec": local_start,
                "refined_end_sec": local_end,
                "decision": "REFINE",
                "confidence": 0.8,
                "boundary_reason": "trim setup",
            }
        ),
        "stop",
    )


_LOCAL_ORIGINAL: dict[str, tuple[float, float]] = {}


def build_local_original_map(records):
    _LOCAL_ORIGINAL.clear()
    for record in records.values():
        for candidate in record["merged_candidates"]:
            _LOCAL_ORIGINAL[candidate["merged_candidate_id"]] = (
                float(candidate["start_sec"]),
                float(candidate["end_sec"]),
            )


def response(payload: dict, finish_reason: str | None = "stop"):
    return json.dumps(payload, ensure_ascii=False), finish_reason


def test_br0_is_exact_identity():
    manifest, role_manifest, records = make_role_and_cache()
    result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)
    summary = validate_boundary_refinement_payload(result, manifest, role_manifest, records)
    assert summary["input_candidate_count"] == 3
    assert summary["decision_counts"] == {BR0_DECISION: 3, "REFINE": 0, "IDENTITY_FALLBACK": 0}
    for record in result["records"]:
        for refinement in record["candidate_refinements"]:
            assert refinement["refined_start_sec"] == refinement["original_start_sec"]
            assert refinement["refined_end_sec"] == refinement["original_end_sec"]
            assert refinement["model_response"] is None
            assert refinement["decision_rule"] == "br0.identity"


@pytest.mark.parametrize("excess", [8e-8, 5e-7, 1e-6])
def test_br0_accepts_stage42_frozen_endpoint_tolerance(excess):
    """Stage 4.2 accepts frozen endpoints up to duration + 1e-6."""
    duration = 60.0
    frozen_end = duration + excess
    candidate = make_candidate("qvh_tolerance", 59.0, frozen_end, 0.9, "末帧事件", 0)
    records = {"qvh_tolerance": make_cache_record("qvh_tolerance", [candidate], duration)}
    manifest = {"global_semantic_sha256": EXPECTED_CACHE_GLOBAL_HASH}
    role_manifest = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "qvh_tolerance", "split": "dev"}],
        source_cache_global_hash=EXPECTED_CACHE_GLOBAL_HASH,
        source_cache_manifest_sha256=HEX_A,
        audit_membership_source_sha256=HEX_B,
    )

    result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)

    refinement = result["records"][0]["candidate_refinements"][0]
    assert refinement["original_end_sec"] == frozen_end
    assert refinement["refined_end_sec"] == frozen_end


def test_br0_rejects_endpoint_beyond_stage42_frozen_tolerance():
    duration = 60.0
    candidate = make_candidate(
        "qvh_tolerance_invalid", 59.0, duration + 1.0001e-6, 0.9, "末帧事件", 0
    )
    records = {
        "qvh_tolerance_invalid": make_cache_record(
            "qvh_tolerance_invalid", [candidate], duration
        )
    }
    manifest = {"global_semantic_sha256": EXPECTED_CACHE_GLOBAL_HASH}
    role_manifest = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "qvh_tolerance_invalid", "split": "dev"}],
        source_cache_global_hash=EXPECTED_CACHE_GLOBAL_HASH,
        source_cache_manifest_sha256=HEX_A,
        audit_membership_source_sha256=HEX_B,
    )

    with pytest.raises(BoundaryRefinementError):
        create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)


def test_br0_determinism():
    manifest, role_manifest, records = make_role_and_cache()
    first = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)
    second = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_br0_replay_equals_frozen_projection():
    manifest, role_manifest, records = make_role_and_cache()
    result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)
    replay = replay_refined_predictions(result, records)
    assert [item["video_id"] for item in replay] == [video_id for video_id, _ in ROLE_ITEMS]
    for item in replay:
        candidates = records[item["video_id"]]["merged_candidates"]
        assert item["merged_prediction_segments"] == [
            {
                "start_sec": candidate["start_sec"],
                "end_sec": candidate["end_sec"],
                "score": candidate["score"],
                "reason": candidate["reason"],
                "source_chunk": candidate["source_chunk"],
            }
            for candidate in candidates
        ]


def test_br0_accepts_frozen_record_with_zero_candidates():
    records = {"qvh_empty": make_cache_record("qvh_empty", [])}
    manifest = {"global_semantic_sha256": EXPECTED_CACHE_GLOBAL_HASH}
    role_manifest = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "qvh_empty", "split": "dev"}],
        source_cache_global_hash=EXPECTED_CACHE_GLOBAL_HASH,
        source_cache_manifest_sha256=HEX_A,
        audit_membership_source_sha256=HEX_B,
    )

    result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)

    assert result["record_count"] == 1
    assert result["input_candidate_count"] == 0
    assert result["records"][0]["candidate_refinements"] == []


def test_local_window_clamps_and_caps():
    window = compute_local_context_window(
        start_sec=10.0,
        end_sec=20.0,
        duration_sec=60.0,
        context_padding_sec=20.0,
        max_context_duration_sec=120.0,
    )
    assert window == {"start_sec": 0.0, "end_sec": 40.0}
    capped = compute_local_context_window(
        start_sec=5.0,
        end_sec=30.0,
        duration_sec=60.0,
        context_padding_sec=20.0,
        max_context_duration_sec=40.0,
    )
    assert capped == {"start_sec": 0.0, "end_sec": 40.0}
    long_candidate = compute_local_context_window(
        start_sec=0.0,
        end_sec=50.0,
        duration_sec=60.0,
        context_padding_sec=20.0,
        max_context_duration_sec=40.0,
    )
    assert long_candidate == {"start_sec": 0.0, "end_sec": 50.0}


def test_parse_boundary_response_maps_to_absolute_time():
    parsed = parse_boundary_response(
        json.dumps(
            {
                "refined_start_sec": 2.0,
                "refined_end_sec": 7.5,
                "decision": "REFINE",
                "confidence": 0.77,
                "boundary_reason": "trim setup",
            }
        ),
        context_start_sec=10.0,
        context_end_sec=50.0,
    )
    assert parsed["refined_start_sec"] == 12.0
    assert parsed["refined_end_sec"] == 17.5
    assert parsed["decision"] == "REFINE"


def test_parse_boundary_response_accepts_markdown_fence():
    fenced = (
        "```json\n"
        '{"refined_start_sec": 1.0, "refined_end_sec": 3.5, "decision": "REFINE",'
        ' "confidence": 0.9, "boundary_reason": "trim"}\n'
        "```"
    )
    parsed = parse_boundary_response(fenced, context_start_sec=0.0, context_end_sec=10.0)
    assert parsed["refined_start_sec"] == 1.0
    assert parsed["refined_end_sec"] == 3.5


@pytest.mark.parametrize(
    "payload",
    [
        '{"refined_start_sec":1.0,"refined_end_sec":2.0,"decision":"REFINE","confidence":0.5}',
        '{"refined_start_sec":1.0,"refined_end_sec":2.0,"decision":"REFINE","confidence":0.5,"boundary_reason":"x","extra":1}',
        '{"refined_start_sec":1.0,"refined_end_sec":2.0,"decision":"DROP","confidence":0.5,"boundary_reason":"x"}',
        '{"refined_start_sec":2.0,"refined_end_sec":1.0,"decision":"REFINE","confidence":0.5,"boundary_reason":"x"}',
        '{"refined_start_sec":-1.0,"refined_end_sec":2.0,"decision":"REFINE","confidence":0.5,"boundary_reason":"x"}',
        '{"refined_start_sec":1.0,"refined_end_sec":200.0,"decision":"REFINE","confidence":0.5,"boundary_reason":"x"}',
        '{"refined_start_sec":1.0,"refined_end_sec":2.0,"decision":"REFINE","confidence":2.0,"boundary_reason":"x"}',
        '{"refined_start_sec":1.0,"refined_end_sec":2.0,"decision":"REFINE","confidence":0.5,"boundary_reason":1}',
        "not json at all",
    ],
)
def test_parse_boundary_response_rejects_invalid(payload):
    with pytest.raises(BoundaryResponseError):
        parse_boundary_response(payload, context_start_sec=0.0, context_end_sec=50.0)


def test_event_identity_guard_rejects_drift():
    window = {"start_sec": 0.0, "end_sec": 40.0}
    ok = assess_event_identity(
        original_start_sec=10.0,
        original_end_sec=20.0,
        refined_start_sec=11.0,
        refined_end_sec=19.0,
        window=window,
        duration_sec=60.0,
        min_parent_temporal_iou=0.5,
    )
    assert ok["pass"]
    drift = assess_event_identity(
        original_start_sec=10.0,
        original_end_sec=20.0,
        refined_start_sec=38.0,
        refined_end_sec=40.0,
        window=window,
        duration_sec=60.0,
        min_parent_temporal_iou=0.5,
    )
    assert not drift["pass"]
    assert "parent_tiou" in drift["failed_checks"]
    outside = assess_event_identity(
        original_start_sec=10.0,
        original_end_sec=20.0,
        refined_start_sec=5.0,
        refined_end_sec=45.0,
        window=window,
        duration_sec=60.0,
        min_parent_temporal_iou=0.5,
    )
    assert not outside["pass"]
    assert "within_local_window" in outside["failed_checks"]
    invalid = assess_event_identity(
        original_start_sec=10.0,
        original_end_sec=20.0,
        refined_start_sec=20.0,
        refined_end_sec=20.0,
        window=window,
        duration_sec=60.0,
        min_parent_temporal_iou=0.5,
    )
    assert not invalid["pass"]


def test_br1_accepts_valid_refine():
    manifest, role_manifest, records = make_role_and_cache()
    build_local_original_map(records)
    decisions = refine_candidates(
        records["qvh_smoke_a"]["merged_candidates"],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=trim_start_model_fn,
    )
    assert decisions[0]["decision"] == "REFINE"
    assert decisions[0]["refined_start_sec"] == 12.0
    assert decisions[0]["refined_end_sec"] == 20.0
    assert decisions[0]["parent_consistency_guard"]["pass"] is True
    assert decisions[0]["model_response"]["finish_reason"] == "stop"
    assert decisions[0]["local_context_window"] == {"start_sec": 0.0, "end_sec": 40.0}


def test_br1_maps_local_boundaries_from_actual_clip_origin_not_requested_window():
    candidate = make_candidate("qvh_origin", 30.0, 35.0, 0.9, "局部事件", 0)

    def prepare_context(*, video_id, candidate_id, window):
        assert window == {"start_sec": 10.0, "end_sec": 55.0}
        return {
            "coordinate_contract": "source_sec = actual_source_origin_sec + local_sec",
            "extraction_backend": "ffmpeg_reencode",
            "requested_source_start_sec": 10.0,
            "requested_source_end_sec": 55.0,
            "actual_source_origin_sec": 10.033333,
            "clip_duration_sec": 44.966667,
            "actual_source_frame_index": None,
            "source_fps": None,
            "first_source_frame_checksum": "ABCD1234",
        }

    def model_fn(prompt, *, video_id, candidate_id, window):
        return response(
            {
                "refined_start_sec": 20.466667,
                "refined_end_sec": 24.966667,
                "decision": "REFINE",
                "confidence": 0.9,
                "boundary_reason": "frame-quantized origin",
            }
        )

    decision = refine_candidates(
        [candidate],
        BR1_CONFIG,
        duration_sec=60.0,
        video_id="qvh_origin",
        model_fn=model_fn,
        prepare_context_fn=prepare_context,
    )[0]

    assert decision["clip_timing"]["actual_source_origin_sec"] == 10.033333
    assert decision["refined_start_sec"] == pytest.approx(30.5)
    assert decision["refined_end_sec"] == pytest.approx(35.0)


def test_br1_falls_back_on_parse_error():
    manifest, role_manifest, records = make_role_and_cache()

    def model_fn(prompt, *, video_id, candidate_id, window):
        return "The event probably starts a bit later, around 11 to 19 seconds.", "stop"

    decisions = refine_candidates(
        records["qvh_smoke_a"]["merged_candidates"],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )
    assert decisions[0]["decision"] == "IDENTITY_FALLBACK"
    assert decisions[0]["decision_rule"] == "br1.fallback_parse_error"


def test_br1_falls_back_on_expected_transport_connection_failure():
    _, _, records = make_role_and_cache()
    candidate = records["qvh_smoke_a"]["merged_candidates"][0]

    def model_fn(prompt, *, video_id, candidate_id, window):
        raise ConnectionError("synthetic connection failure")

    decisions = refine_candidates(
        [candidate],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )

    assert decisions[0]["decision_rule"] == "br1.fallback_transport_error"
    assert decisions[0]["refined_start_sec"] == candidate["start_sec"]
    assert decisions[0]["refined_end_sec"] == candidate["end_sec"]


def test_br1_empty_response_has_distinct_fallback_rule():
    _, _, records = make_role_and_cache()
    candidate = records["qvh_smoke_a"]["merged_candidates"][0]

    def model_fn(prompt, *, video_id, candidate_id, window):
        return "  ", "stop"

    decisions = refine_candidates(
        [candidate],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )

    assert decisions[0]["decision_rule"] == "br1.fallback_empty_response"


def test_br1_falls_back_on_expected_transport_timeout():
    _, _, records = make_role_and_cache()
    candidate = records["qvh_smoke_a"]["merged_candidates"][0]

    def model_fn(prompt, *, video_id, candidate_id, window):
        raise TimeoutError("synthetic timeout")

    decisions = refine_candidates(
        [candidate],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )

    assert decisions[0]["decision_rule"] == "br1.fallback_transport_error"
    assert decisions[0]["model_response"]["transport_error_type"] == "TimeoutError"
    assert decisions[0]["refined_start_sec"] == 10.0
    assert decisions[0]["refined_end_sec"] == 20.0


def test_br1_does_not_hide_programming_errors_as_transport_fallback():
    _, _, records = make_role_and_cache()
    candidate = records["qvh_smoke_a"]["merged_candidates"][0]

    def model_fn(prompt, *, video_id, candidate_id, window):
        raise RuntimeError("synthetic programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        refine_candidates(
            [candidate],
            BR1_CONFIG,
            duration_sec=records["qvh_smoke_a"]["duration_sec"],
            video_id="qvh_smoke_a",
            model_fn=model_fn,
        )


def test_br1_honours_model_identity_fallback():
    manifest, role_manifest, records = make_role_and_cache()

    def model_fn(prompt, *, video_id, candidate_id, window):
        return response(
            {
                "refined_start_sec": 10.0,
                "refined_end_sec": 20.0,
                "decision": "IDENTITY_FALLBACK",
                "confidence": 0.3,
                "boundary_reason": "uncertain",
            }
        )

    decisions = refine_candidates(
        records["qvh_smoke_a"]["merged_candidates"],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )
    assert decisions[0]["decision"] == "IDENTITY_FALLBACK"
    assert decisions[0]["decision_rule"] == "br1.model_identity"


def test_br1_guard_rejects_drifted_event():
    manifest, role_manifest, records = make_role_and_cache()

    def model_fn(prompt, *, video_id, candidate_id, window):
        local_start = 38.0 - float(window["start_sec"])
        local_end = 40.0 - float(window["start_sec"])
        return response(
            {
                "refined_start_sec": local_start,
                "refined_end_sec": local_end,
                "decision": "REFINE",
                "confidence": 0.9,
                "boundary_reason": "moved to another event",
            }
        )

    decisions = refine_candidates(
        records["qvh_smoke_a"]["merged_candidates"],
        BR1_CONFIG,
        duration_sec=records["qvh_smoke_a"]["duration_sec"],
        video_id="qvh_smoke_a",
        model_fn=model_fn,
    )
    assert decisions[0]["decision"] == "IDENTITY_FALLBACK"
    assert decisions[0]["decision_rule"] == "br1.fallback_parent_consistency_guard"
    assert decisions[0]["refined_start_sec"] == 10.0
    assert decisions[0]["refined_end_sec"] == 20.0
    assert decisions[0]["parent_consistency_guard"]["failed_checks"]


def test_br1_requires_model_fn():
    manifest, role_manifest, records = make_role_and_cache()
    with pytest.raises(BoundaryRefinementError):
        create_boundary_refinement_result(manifest, role_manifest, records, BR1_CONFIG)


def _refined_result():
    manifest, role_manifest, records = make_role_and_cache()
    build_local_original_map(records)
    result = create_boundary_refinement_result(
        manifest, role_manifest, records, BR1_CONFIG, model_fn=trim_start_model_fn
    )
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)
    return manifest, role_manifest, records, result


def _rehash(result: dict) -> dict:
    payload = json.loads(json.dumps(result, ensure_ascii=False))
    for record in payload["records"]:
        record.pop("video_refinement_semantic_sha256")
        record["video_refinement_semantic_sha256"] = semantic_sha256(record)
    payload.pop("boundary_refinement_semantic_hash")
    payload["boundary_refinement_semantic_hash"] = semantic_sha256(payload)
    return payload


def test_validator_negative_controls():
    manifest, role_manifest, records, result = _refined_result()

    drifted = json.loads(json.dumps(result, ensure_ascii=False))
    refinement = drifted["records"][0]["candidate_refinements"][0]
    refinement["refined_start_sec"] = 38.0
    refinement["refined_end_sec"] = 40.0
    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(drifted), manifest, role_manifest, records
        )

    reidentified = json.loads(json.dumps(result, ensure_ascii=False))
    reidentified["records"][0]["candidate_refinements"][0]["merged_candidate_id"] = "cand-other"
    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(reidentified), manifest, role_manifest, records
        )

    shrunk = json.loads(json.dumps(result, ensure_ascii=False))
    shrunk["records"][1]["candidate_refinements"].pop()
    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(shrunk), manifest, role_manifest, records
        )

    lying_fallback = json.loads(json.dumps(result, ensure_ascii=False))
    fallback = lying_fallback["records"][0]["candidate_refinements"][0]
    fallback["decision"] = "IDENTITY_FALLBACK"
    fallback["decision_rule"] = "br1.fallback_parse_error"
    fallback["refined_start_sec"] = 15.0
    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(lying_fallback), manifest, role_manifest, records
        )

    forbidden = json.loads(json.dumps(result, ensure_ascii=False))
    forbidden["human_hint"] = "some hint"
    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(forbidden), manifest, role_manifest, records
        )


def test_result_records_protocol_runtime_binding():
    manifest, role_manifest, records = make_role_and_cache()

    result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)

    assert result["protocol_binding"]["protocol_status"] == "DRAFT_FOR_INDEPENDENT_REVIEW"
    assert result["protocol_binding"]["model_identifier"] is None
    assert result["protocol_binding"]["git_head"]


def test_br1_machine_summary_exposes_all_mutually_exclusive_rule_counters():
    manifest, role_manifest, records = make_role_and_cache()
    build_local_original_map(records)
    result = create_boundary_refinement_result(
        manifest, role_manifest, records, BR1_CONFIG, model_fn=trim_start_model_fn
    )

    assert set(result["decision_rule_counts"]) == {
        "br1.model_refine",
        "br1.model_identity",
        "br1.fallback_transport_error",
        "br1.fallback_empty_response",
        "br1.fallback_parse_error",
        "br1.fallback_parent_consistency_guard",
    }
    assert sum(result["decision_rule_counts"].values()) == result["input_candidate_count"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_bytes_sha256",), "d" * 64),
        (("protocol_status",), "PREREGISTERED_BEFORE_FORMAL"),
        (("model_identifier",), "wrong/model"),
        (("prompt_semantic_sha256",), "e" * 64),
        (("refiner_runtime_config", "parameters", "context_padding_sec"), 21.0),
        (("role_manifest_hash",), "f" * 64),
    ],
)
def test_validator_rejects_protocol_runtime_binding_negative_controls(path, value):
    manifest, role_manifest, records = make_role_and_cache()
    build_local_original_map(records)
    result = create_boundary_refinement_result(
        manifest, role_manifest, records, BR1_CONFIG, model_fn=trim_start_model_fn
    )
    mutated = json.loads(json.dumps(result, ensure_ascii=False))
    target = mutated["protocol_binding"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(BoundaryRefinementError):
        validate_boundary_refinement_payload(
            _rehash(mutated), manifest, role_manifest, records
        )


def test_protocol_loader_rejects_wrong_status_with_valid_semantic_hash(tmp_path: Path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_status"] = "UNREVIEWED"
    protocol["protocol_semantic_sha256"] = semantic_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_semantic_sha256"}
    )
    path = tmp_path / "wrong-status.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(BoundaryRefinementError, match="status"):
        load_boundary_protocol(path, allow_draft=True)


@pytest.mark.parametrize(
    ("model", "timeout", "message"),
    [
        ("wrong/model", None, "--model"),
        ("Qwen/Qwen3.5-4B", 121.0, "--vllm-timeout-sec"),
    ],
)
def test_cli_runtime_overrides_are_assertions_not_method_changes(model, timeout, message):
    from scripts.run_boundary_refinement import _build_br1_runtime

    protocol = load_boundary_protocol(PROTOCOL_PATH)
    args = argparse.Namespace(
        dataset_manifest=Path("unused-manifest.jsonl"),
        video_root=Path("unused-video-root"),
        model=model,
        vllm_timeout_sec=timeout,
    )

    with pytest.raises(SystemExit, match=message):
        _build_br1_runtime(args, protocol)


def test_replay_keeps_frozen_score_reason_source():
    manifest, role_manifest, records, result = _refined_result()
    replay = replay_refined_predictions(result, records)
    target = replay[0]["merged_prediction_segments"][0]
    cache_candidate = records["qvh_smoke_a"]["merged_candidates"][0]
    assert target["score"] == cache_candidate["score"]
    assert target["reason"] == cache_candidate["reason"]
    assert target["source_chunk"] == cache_candidate["source_chunk"]
    assert target["start_sec"] == 12.0
    assert target["end_sec"] == 20.0


def test_protocol_loader_and_config(tmp_path: Path):
    protocol = load_boundary_protocol(PROTOCOL_PATH, allow_draft=True)
    assert protocol["protocol_status"] == "DRAFT_FOR_INDEPENDENT_REVIEW"
    with pytest.raises(BoundaryRefinementError):
        load_boundary_protocol(PROTOCOL_PATH, allow_draft=False)
    config = refiner_config_from_protocol(protocol, "BR-1")
    assert config["parameters"] == BR1_CONFIG["parameters"]
    br0 = refiner_config_from_protocol(protocol, "BR-0")
    assert br0["parameters"] == {}

    original = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tampered = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tampered["dev_recall_guardrail"]["min_mean_recall"] = 0.5
    bad_path = tmp_path / "bad_protocol.json"
    bad_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(BoundaryRefinementError):
        load_boundary_protocol(bad_path, allow_draft=True)
    assert original["protocol_semantic_sha256"] == protocol["protocol_semantic_sha256"]


def test_full_chain_evaluate_and_assess(tmp_path: Path):
    manifest, role_manifest, records = make_role_and_cache()
    build_local_original_map(records)
    br0_result = create_boundary_refinement_result(manifest, role_manifest, records, BR0_CONFIG)
    br1_result = create_boundary_refinement_result(
        manifest, role_manifest, records, BR1_CONFIG, model_fn=trim_start_model_fn
    )
    br0_replay = replay_refined_predictions(br0_result, records)
    br1_replay = replay_refined_predictions(br1_result, records)

    references = {
        "qvh_smoke_a": [{"start_sec": 12.0, "end_sec": 20.0}],
        "qvh_smoke_b": [
            {"start_sec": 7.0, "end_sec": 12.0},
            {"start_sec": 32.0, "end_sec": 45.0},
        ],
    }
    frozen = [
        {
            "video_id": video_id,
            "split": "dev",
            "weak_reference_segments": references[video_id],
        }
        for video_id, _ in ROLE_ITEMS
    ]
    durations = {video_id: records[video_id]["duration_sec"] for video_id, _ in ROLE_ITEMS}
    common = dict(
        durations_by_id=durations,
        role="dev_tune",
        role_manifest_hash=role_manifest["semantic_sha256"],
    )
    br0_evaluation = evaluate_replayed_predictions(
        br0_replay,
        frozen,
        selection_result_hash=br0_result["boundary_refinement_semantic_hash"],
        selection_metadata={"refiner_name": "BR-0"},
        **common,
    )
    br1_evaluation = evaluate_replayed_predictions(
        br1_replay,
        frozen,
        selection_result_hash=br1_result["boundary_refinement_semantic_hash"],
        selection_metadata={"refiner_name": "BR-1"},
        **common,
    )
    comparison = compare_evaluations(br0_evaluation, br1_evaluation)
    assert comparison["record_count"] == 2
    assert comparison["per_video"][0]["delta_temporal_iou"] > 0.0
    assert comparison["aggregate_delta"]["delta_mean_weak_ref_recall"] == 0.0

    br0_eval_path = tmp_path / "br0_evaluation.json"
    br1_eval_path = tmp_path / "br1_evaluation.json"
    br0_eval_path.write_bytes(canonical_json_bytes(br0_evaluation))
    br1_eval_path.write_bytes(canonical_json_bytes(br1_evaluation))
    assessment_path = tmp_path / "assessment.json"
    report = assess_refinement_files(
        br0_eval_path,
        br1_eval_path,
        PROTOCOL_PATH,
        "dev",
        assessment_path,
        allow_draft_protocol=True,
    )
    assert set(report["assessments"]) == {"dev_recall_guardrail", "dev_promotion_gate", "pass"}
    assert report["assessments"]["dev_recall_guardrail"]["pass"] is True


def test_prompt_contains_no_audit_content():
    prompt = build_boundary_prompt(
        context_duration_sec=40.0,
        candidate_start_sec=10.0,
        candidate_end_sec=20.0,
        candidate_reason="关键进球瞬间",
    )
    assert "IDENTITY_FALLBACK" in prompt
    assert "REFINE" in prompt
    assert "qvh_" not in prompt
    for forbidden in ("audit", "adjudicat", "recall", "precision", "ground truth", "label"):
        assert forbidden not in prompt.lower()
