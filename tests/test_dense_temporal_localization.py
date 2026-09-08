from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from aic_video_highlight.highlight_retrieval.dense_temporal_localization import (
    DenseTemporalError,
    DenseTemporalResponseError,
    build_dense_prompt,
    build_temporal_bins,
    create_dense_localization_result,
    decode_dense_labels,
    parse_dense_response,
    replay_dense_predictions,
    validate_dense_localization_payload,
)
from aic_video_highlight.highlight_retrieval.candidate_cache import semantic_sha256


def clip_timing(duration: float = 10.0, origin: float = 3.25) -> dict:
    return {
        "coordinate_contract": "source_sec = actual_source_origin_sec + local_sec",
        "extraction_backend": "ffmpeg_reencode",
        "requested_source_start_sec": 3.2,
        "requested_source_end_sec": 3.2 + duration,
        "actual_source_origin_sec": origin,
        "clip_duration_sec": duration,
        "actual_source_frame_index": None,
        "source_fps": None,
        "first_source_frame_checksum": "ABCD1234",
    }


def test_build_temporal_bins_exact_duration_and_source_mapping():
    bins = build_temporal_bins(clip_timing(), bin_size_sec=2.0, max_bin_count=64)

    assert [item["bin_id"] for item in bins] == [f"bin_{i:03d}" for i in range(5)]
    assert bins[0] == {
        "bin_id": "bin_000",
        "local_start_sec": 0.0,
        "local_end_sec": 2.0,
        "source_start_sec": 3.25,
        "source_end_sec": 5.25,
    }
    assert bins[-1]["local_end_sec"] == 10.0
    assert bins[-1]["source_end_sec"] == 13.25


def test_build_temporal_bins_final_short_bin_and_non_requested_origin():
    timing = clip_timing(duration=5.25, origin=3.266667)
    bins = build_temporal_bins(timing, bin_size_sec=2.0, max_bin_count=64)
    assert len(bins) == 3
    assert bins[-1]["local_start_sec"] == 4.0
    assert bins[-1]["local_end_sec"] == 5.25
    assert bins[-1]["source_start_sec"] == pytest.approx(7.266667)
    assert bins[-1]["source_end_sec"] == pytest.approx(8.516667)


def test_build_temporal_bins_enforces_max_64():
    assert len(build_temporal_bins(clip_timing(128.0), bin_size_sec=2.0, max_bin_count=64)) == 64
    with pytest.raises(DenseTemporalError, match="exceeding"):
        build_temporal_bins(clip_timing(128.01), bin_size_sec=2.0, max_bin_count=64)


def labels_for(bins: list[dict], labels: list[str]) -> str:
    return json.dumps(
        {
            "bins": [
                {"bin_id": item["bin_id"], "label": label, "confidence": 0.8}
                for item, label in zip(bins, labels, strict=True)
            ]
        }
    )


def test_parse_dense_response_accepts_full_bins_fence_and_canonical_sort():
    bins = build_temporal_bins(clip_timing(6.0), bin_size_sec=2.0, max_bin_count=64)
    raw = json.dumps(
        {"bins": [
            {"bin_id": "bin_002", "label": "CORE", "confidence": 1.0},
            {"bin_id": "bin_000", "label": "OUTSIDE", "confidence": 0.1},
            {"bin_id": "bin_001", "label": "CONTEXT", "confidence": 0.5},
        ]}
    )
    parsed = parse_dense_response(f"```json\n{raw}\n```", bins)
    assert [item["bin_id"] for item in parsed] == ["bin_000", "bin_001", "bin_002"]


@pytest.mark.parametrize(
    "render",
    [
        lambda array: json.dumps(array),
        lambda array: f"```json\n{json.dumps(array)}\n```",
        lambda array: f"Brief analysis omitted.\n```json\n{json.dumps({'bins': array})}\n```",
        lambda array: f"Brief analysis omitted.\n```json\n{json.dumps(array)}\n```",
    ],
)
def test_parse_dense_response_is_envelope_tolerant_but_content_strict(render):
    bins = build_temporal_bins(clip_timing(6.0), bin_size_sec=2.0, max_bin_count=64)
    array = [
        {"bin_id": item["bin_id"], "label": "CORE", "confidence": 0.8}
        for item in bins
    ]
    parsed = parse_dense_response(render(array), bins)
    assert [item["bin_id"] for item in parsed] == ["bin_000", "bin_001", "bin_002"]


@pytest.mark.parametrize(
    "raw",
    [
        'prose before {"bins": []}',
        '```json\n{"bins": []}\n```\n```json\n{"bins": []}\n```',
        'prose before ```json\n{"bins": []}',
    ],
)
def test_parse_dense_response_rejects_ambiguous_or_unclosed_envelopes(raw):
    bins = build_temporal_bins(clip_timing(6.0), bin_size_sec=2.0, max_bin_count=64)
    with pytest.raises(DenseTemporalResponseError):
        parse_dense_response(raw, bins)


@pytest.mark.parametrize(
    "raw_builder",
    [
        lambda bins: labels_for(bins[:-1], ["CORE"] * (len(bins) - 1)),
        lambda bins: json.dumps({"bins": [
            {"bin_id": bins[0]["bin_id"], "label": "CORE", "confidence": 0.5},
            {"bin_id": bins[0]["bin_id"], "label": "CORE", "confidence": 0.5},
            {"bin_id": bins[2]["bin_id"], "label": "CORE", "confidence": 0.5},
        ]}),
        lambda bins: json.dumps({"bins": [
            {"bin_id": "bin_999", "label": "CORE", "confidence": 0.5},
            *[{"bin_id": x["bin_id"], "label": "CORE", "confidence": 0.5} for x in bins[1:]],
        ]}),
        lambda bins: labels_for(bins, ["HIGHLIGHT", "CORE", "CORE"]),
        lambda bins: json.dumps({"bins": [
            {"bin_id": x["bin_id"], "label": "CORE", "confidence": value}
            for x, value in zip(bins, [float("nan"), 0.5, 0.5], strict=True)
        ]}),
        lambda bins: labels_for(bins, ["CORE"] * len(bins)).replace("0.8", "1.1", 1),
        lambda bins: "{not json",
    ],
)
def test_parse_dense_response_rejects_schema_failures(raw_builder):
    bins = build_temporal_bins(clip_timing(6.0), bin_size_sec=2.0, max_bin_count=64)
    with pytest.raises(DenseTemporalResponseError):
        parse_dense_response(raw_builder(bins), bins)


def make_bins(labels: list[str], origin: float = 0.0) -> tuple[list[dict], list[dict]]:
    bins = build_temporal_bins(clip_timing(len(labels) * 2.0, origin), bin_size_sec=2.0, max_bin_count=64)
    classified = [
        {"bin_id": item["bin_id"], "label": label, "confidence": 0.5}
        for item, label in zip(bins, labels, strict=True)
    ]
    return bins, classified


def test_decoder_selects_context_core_component_span():
    bins, classified = make_bins(["OUTSIDE", "CONTEXT", "CORE", "CORE", "CONTEXT", "OUTSIDE"])
    decoded = decode_dense_labels(bins, classified, original_start_sec=4.0, original_end_sec=8.0)
    assert decoded["decision_rule"] == "dtl1.dense_refine"
    assert (decoded["refined_start_sec"], decoded["refined_end_sec"]) == (2.0, 10.0)


@pytest.mark.parametrize(
    ("labels", "original", "expected"),
    [
        (["CONTEXT", "CORE", "OUTSIDE"], (2.0, 4.0), (0.0, 4.0)),
        (["OUTSIDE", "CORE", "CONTEXT"], (2.0, 4.0), (2.0, 6.0)),
    ],
)
def test_decoder_allows_left_or_right_expansion(labels, original, expected):
    bins, classified = make_bins(labels)
    decoded = decode_dense_labels(bins, classified, original_start_sec=original[0], original_end_sec=original[1])
    assert (decoded["refined_start_sec"], decoded["refined_end_sec"]) == expected


@pytest.mark.parametrize("labels", [["OUTSIDE", "OUTSIDE"], ["CONTEXT", "CONTEXT"]])
def test_decoder_no_core_is_identity_fallback(labels):
    bins, classified = make_bins(labels)
    decoded = decode_dense_labels(bins, classified, original_start_sec=1.0, original_end_sec=3.0)
    assert decoded["decision_rule"] == "dtl1.fallback_no_core"
    assert (decoded["refined_start_sec"], decoded["refined_end_sec"]) == (1.0, 3.0)


def test_decoder_chooses_largest_parent_overlap_and_rejects_adjacent_event():
    bins, classified = make_bins(["CORE", "OUTSIDE", "CORE", "CORE"])
    decoded = decode_dense_labels(bins, classified, original_start_sec=5.0, original_end_sec=7.0)
    assert decoded["selected_component"]["bin_ids"] == ["bin_002", "bin_003"]
    assert decoded["refined_start_sec"] == 4.0

    adjacent = decode_dense_labels(bins, classified, original_start_sec=2.0, original_end_sec=4.0)
    assert adjacent["decision_rule"] == "dtl1.fallback_no_parent_overlap"


def test_decoder_ties_break_by_core_duration_count_then_earliest_start():
    bins, classified = make_bins(["CORE", "OUTSIDE", "CORE", "OUTSIDE", "CORE"])
    decoded = decode_dense_labels(bins, classified, original_start_sec=0.0, original_end_sec=10.0)
    assert decoded["selected_component"]["bin_ids"] == ["bin_000"]


def test_decoder_exact_same_interval_is_dense_identity():
    bins, classified = make_bins(["OUTSIDE", "CORE", "OUTSIDE"])
    decoded = decode_dense_labels(bins, classified, original_start_sec=2.0, original_end_sec=4.0)
    assert decoded["decision_rule"] == "dtl1.dense_identity"


def candidate(candidate_id="merged-v1-test", start=2.0, end=6.0) -> dict:
    return {
        "merged_candidate_id": candidate_id,
        "start_sec": start,
        "end_sec": end,
        "score": 0.9,
        "reason": "goal",
        "source_chunk": 0,
    }


def role_and_cache() -> tuple[dict, dict, dict]:
    item = candidate()
    record = {
        "video_id": "qvh_test",
        "split": "dev",
        "duration_sec": 10.0,
        "merged_candidates": [item],
    }
    record["semantic_sha256"] = semantic_sha256(record)
    manifest = {
        "global_semantic_sha256": "4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246"
    }
    role = {
        "role": "dev_tune",
        "source_cache_global_hash": manifest["global_semantic_sha256"],
        "records": [{"video_id": "qvh_test", "split": "dev"}],
        "semantic_sha256": "b" * 64,
    }
    return manifest, role, {"qvh_test": record}


DTL0 = {"localizer_name": "DTL-0", "localizer_version": "aic.dense-temporal-localization/v1", "parameters": {}}
DTL1 = {
    "localizer_name": "DTL-1",
    "localizer_version": "aic.dense-temporal-localization/v1",
    "parameters": {"context_padding_sec": 20.0, "max_context_duration_sec": 120.0, "bin_size_sec": 2.0, "max_bin_count": 64},
}


def fake_prepare(**kwargs):
    timing = clip_timing(10.0, 0.0)
    timing["requested_source_start_sec"] = float(kwargs["window"]["start_sec"])
    timing["requested_source_end_sec"] = float(kwargs["window"]["end_sec"])
    return timing


def model_response(labels):
    def model_fn(prompt, **kwargs):
        bins = build_temporal_bins(clip_timing(10.0, 0.0), bin_size_sec=2.0, max_bin_count=64)
        return labels_for(bins, labels), "stop"
    return model_fn


@pytest.mark.parametrize(
    ("labels", "rule"),
    [
        (["OUTSIDE", "CORE", "CONTEXT", "OUTSIDE", "OUTSIDE"], "dtl1.dense_identity"),
        (["CORE", "CORE", "OUTSIDE", "OUTSIDE", "OUTSIDE"], "dtl1.dense_refine"),
        (["OUTSIDE"] * 5, "dtl1.fallback_no_core"),
    ],
)
def test_fake_model_dense_outcomes(labels, rule):
    manifest, role, records = role_and_cache()
    result = create_dense_localization_result(manifest, role, records, DTL1, model_fn=model_response(labels), prepare_context_fn=fake_prepare)
    assert result["records"][0]["candidate_localizations"][0]["decision_rule"] == rule


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["OUTSIDE", "OUTSIDE", "CORE", "OUTSIDE", "OUTSIDE"], (4.0, 6.0)),
        (["CORE", "CORE", "CORE", "OUTSIDE", "OUTSIDE"], (0.0, 6.0)),
        (["OUTSIDE", "CORE", "CORE", "CORE", "OUTSIDE"], (2.0, 8.0)),
        (["CORE", "OUTSIDE", "CORE", "OUTSIDE", "OUTSIDE"], (4.0, 6.0)),
    ],
)
def test_fake_model_full_chain_shrink_expand_and_multi_component(labels, expected):
    manifest, role, records = role_and_cache()
    result = create_dense_localization_result(
        manifest, role, records, DTL1,
        model_fn=model_response(labels), prepare_context_fn=fake_prepare,
    )
    summary = validate_dense_localization_payload(result, manifest, role, records)
    replay = replay_dense_predictions(result, records)
    segment = replay[0]["merged_prediction_segments"][0]
    assert summary["candidate_count"] == 1
    assert (segment["start_sec"], segment["end_sec"]) == expected
    assert (segment["score"], segment["reason"], segment["source_chunk"]) == (0.9, "goal", 0)


@pytest.mark.parametrize(
    ("model_fn", "rule"),
    [
        (lambda *args, **kwargs: ("not-json", "stop"), "dtl1.fallback_parse_error"),
        (lambda *args, **kwargs: (" ", "stop"), "dtl1.fallback_empty_response"),
    ],
)
def test_fake_model_response_fallbacks(model_fn, rule):
    manifest, role, records = role_and_cache()
    result = create_dense_localization_result(manifest, role, records, DTL1, model_fn=model_fn, prepare_context_fn=fake_prepare)
    assert result["records"][0]["candidate_localizations"][0]["decision_rule"] == rule


def test_fake_model_transport_fallback_and_runtime_fail_closed():
    manifest, role, records = role_and_cache()
    def transport(*args, **kwargs):
        raise TimeoutError("timeout")
    result = create_dense_localization_result(manifest, role, records, DTL1, model_fn=transport, prepare_context_fn=fake_prepare)
    assert result["records"][0]["candidate_localizations"][0]["decision_rule"] == "dtl1.fallback_transport_error"
    def bug(*args, **kwargs):
        raise RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        create_dense_localization_result(manifest, role, records, DTL1, model_fn=bug, prepare_context_fn=fake_prepare)


def test_dtl0_exact_identity_and_replay_frozen_fields():
    manifest, role, records = role_and_cache()
    result = create_dense_localization_result(manifest, role, records, DTL0)
    localization = result["records"][0]["candidate_localizations"][0]
    assert localization["decision_rule"] == "dtl0.identity"
    replay = replay_dense_predictions(result, records)
    assert replay[0]["merged_prediction_segments"][0] == {
        "start_sec": 2.0, "end_sec": 6.0, "score": 0.9, "reason": "goal", "source_chunk": 0
    }


def test_prompt_is_short_structured_and_forbids_timestamps():
    bins, _ = make_bins(["OUTSIDE", "CORE"])
    prompt = build_dense_prompt("goal", bins, original_start_sec=2.0, original_end_sec=4.0)
    assert "Label every bin exactly once" in prompt
    assert "Do not output timestamps" in prompt
    assert "CORE" in prompt and "CONTEXT" in prompt and "OUTSIDE" in prompt
    assert "overlap=true" in prompt


def rehash_result(result):
    result = deepcopy(result)
    for record in result["records"]:
        record.pop("video_localization_semantic_sha256", None)
        record["video_localization_semantic_sha256"] = semantic_sha256(record)
    result.pop("dense_localization_semantic_sha256", None)
    result["dense_localization_semantic_sha256"] = semantic_sha256(result)
    return result


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["records"][0]["candidate_localizations"][0]["classified_bins"][1].__setitem__("label", "CONTEXT"),
        lambda r: r["records"][0]["candidate_localizations"][0]["temporal_bins"][1].__setitem__("source_end_sec", 9.0),
        lambda r: r["records"][0]["candidate_localizations"][0]["selected_component"].__setitem__("bin_ids", ["bin_000"]),
        lambda r: r["records"][0]["candidate_localizations"][0].__setitem__("refined_end_sec", 9.0),
        lambda r: r["records"][0]["candidate_localizations"][0].__setitem__("merged_candidate_id", "other"),
        lambda r: r["protocol_binding"].__setitem__("protocol_semantic_sha256", "c" * 64),
        lambda r: r.__setitem__("source_cache_global_hash", "d" * 64),
        lambda r: r.__setitem__("role_manifest_hash", "e" * 64),
        lambda r: r["protocol_binding"].__setitem__("prompt_semantic_sha256", "f" * 64),
        lambda r: r.__setitem__("input_candidate_count", 2),
    ],
)
def test_validator_negative_controls(mutate):
    manifest, role, records = role_and_cache()
    binding = {
        "protocol_semantic_sha256": "a" * 64,
        "protocol_bytes_sha256": "1" * 64,
        "source_cache_global_hash": manifest["global_semantic_sha256"],
        "role_manifest_hash": role["semantic_sha256"],
        "prompt_semantic_sha256": None,
        "localizer_name": "DTL-1",
    }
    result = create_dense_localization_result(
        manifest, role, records, DTL1,
        model_fn=model_response(["OUTSIDE", "CORE", "CONTEXT", "OUTSIDE", "OUTSIDE"]),
        prepare_context_fn=fake_prepare, protocol_binding=binding,
    )
    validate_dense_localization_payload(result, manifest, role, records, expected_protocol_binding=binding)
    mutated = deepcopy(result)
    mutate(mutated)
    with pytest.raises(DenseTemporalError):
        validate_dense_localization_payload(rehash_result(mutated), manifest, role, records, expected_protocol_binding=binding)
