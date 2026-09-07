from __future__ import annotations

from copy import deepcopy

from aic_video_highlight.highlight_retrieval.candidate_cache import (
    canonical_json_bytes,
    semantic_sha256,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (
    CandidateSelectionError,
    SELECTOR_VERSION,
    apply_selector,
    assess_evaluation_comparison,
    compare_evaluations,
    choose_dev_evaluation,
    create_role_manifest,
    create_selection_result,
    evaluate_replayed_predictions,
    partition_cache_records,
    replay_selection_payload,
    selector_config_from_protocol,
    validate_role_manifest,
    validate_selection_result_payload,
)


def _formal_manifest_records() -> list[dict[str, object]]:
    return [
        {
            "video_id": f"dev_{index:03d}",
            "split": "dev",
            "source_record_index": index,
        }
        for index in range(183)
    ] + [
        {
            "video_id": f"hard_{index:03d}",
            "split": "hard",
            "source_record_index": index,
        }
        for index in range(248)
    ]


def _formal_audit_membership() -> list[dict[str, str]]:
    return [
        {"video_id": f"dev_{index:03d}", "split": "dev"}
        for index in range(17)
    ] + [
        {"video_id": f"hard_{index:03d}", "split": "hard"}
        for index in range(19)
    ]


def test_role_partition_is_exact_disjoint_and_complete() -> None:
    roles = partition_cache_records(
        _formal_manifest_records(), _formal_audit_membership()
    )

    assert len(roles["dev_tune"]) == 166
    assert len(roles["hard_stress"]) == 229
    assert len(roles["audit_diagnostic"]) == 36
    sets = {
        role: {item["video_id"] for item in records}
        for role, records in roles.items()
    }
    assert sets["dev_tune"].isdisjoint(sets["hard_stress"])
    assert sets["dev_tune"].isdisjoint(sets["audit_diagnostic"])
    assert sets["hard_stress"].isdisjoint(sets["audit_diagnostic"])
    assert set().union(*sets.values()) == {
        item["video_id"] for item in _formal_manifest_records()
    }


def test_role_partition_rejects_wrong_split_and_unknown_membership() -> None:
    records = _formal_manifest_records()
    membership = _formal_audit_membership()
    membership[0] = {**membership[0], "split": "hard"}

    try:
        partition_cache_records(records, membership)
    except CandidateSelectionError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("wrong-split audit membership must fail")

    membership = _formal_audit_membership()
    membership[0] = {"video_id": "not_in_cache", "split": "dev"}
    try:
        partition_cache_records(records, membership)
    except CandidateSelectionError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown audit membership must fail")


def _candidate(candidate_id: str, score: float, reason: str = "明确事件") -> dict:
    return {
        "merged_candidate_id": candidate_id,
        "start_sec": 1.0,
        "end_sec": 2.0,
        "score": score,
        "reason": reason,
        "source_chunk": 0,
        "contributor_raw_candidate_ids": [f"raw-{candidate_id}"],
    }


def _selector(name: str, parameters: dict) -> dict:
    return {
        "selector_name": name,
        "selector_version": SELECTOR_VERSION,
        "parameters": parameters,
    }


def test_sel0_and_sel1_threshold_edges_are_deterministic() -> None:
    candidates = [_candidate("a", 0.79), _candidate("b", 0.80), _candidate("c", 0.81)]

    sel0 = apply_selector(candidates, _selector("SEL-0", {}))
    assert [item["decision"] for item in sel0] == ["KEEP", "KEEP", "KEEP"]

    config = _selector("SEL-1", {"threshold": 0.80})
    first = apply_selector(candidates, config)
    second = apply_selector(candidates, config)
    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert [item["decision"] for item in first] == ["DROP", "KEEP", "KEEP"]


def test_sel2_handles_ties_and_single_candidate_without_forced_top1() -> None:
    tied = [_candidate("a", 0.90), _candidate("b", 0.90), _candidate("c", 0.84)]
    decisions = apply_selector(tied, _selector("SEL-2", {"delta": 0.05}))
    assert [item["decision"] for item in decisions] == ["KEEP", "KEEP", "DROP"]
    assert apply_selector(
        [_candidate("only", 0.60)], _selector("SEL-2", {"delta": 0.0})
    )[0]["decision"] == "KEEP"


def test_sel3_drops_only_clear_contextual_negative_and_keeps_uncertain() -> None:
    rules = {
        "rules": [
            {
                "rule_id": "setup_only",
                "pattern": "为后续.{0,8}铺垫",
                "positive_override_patterns": ["高潮", "关键结果"],
            }
        ]
    }
    candidates = [
        _candidate("drop", 0.9, "普通环境展示，为后续人物出现做铺垫"),
        _candidate("override", 0.9, "为后续高潮做铺垫"),
        _candidate("uncertain", 0.9, "人物准备起跳，可能出现精彩动作"),
        _candidate("unrelated", 0.9, "人物完成跳跃并庆祝"),
    ]

    decisions = apply_selector(candidates, _selector("SEL-3", rules))
    assert [item["decision"] for item in decisions] == [
        "DROP",
        "KEEP",
        "KEEP",
        "KEEP",
    ]
    assert all("video_id" not in item["decision_rule"] for item in decisions)


def _cache_record(video_id: str = "dev_017", split: str = "dev") -> dict:
    record = {
        "video_id": video_id,
        "split": split,
        "duration_sec": 10.0,
        "merged_candidates": [
            _candidate("candidate-a", 0.75, "人物准备起跳"),
            _candidate("candidate-b", 0.90, "人物完成跳跃"),
        ],
    }
    record["semantic_sha256"] = semantic_sha256(record)
    return record


def test_role_and_selection_hashes_are_deterministic_and_sel0_replays_order() -> None:
    role = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "dev_017", "split": "dev"}],
        source_cache_global_hash="a" * 64,
        source_cache_manifest_sha256="b" * 64,
        audit_membership_source_sha256="c" * 64,
    )
    validate_role_manifest(role, expected_count=1)
    cache_manifest = {"global_semantic_sha256": "a" * 64}
    records = {"dev_017": _cache_record()}
    config = _selector("SEL-0", {})

    first = create_selection_result(cache_manifest, role, records, config)
    second = create_selection_result(cache_manifest, role, records, config)
    assert first == second
    assert first["records"][0]["selected_candidate_ids"] == [
        "candidate-a",
        "candidate-b",
    ]
    assert validate_selection_result_payload(first, cache_manifest, role, records)[
        "selected_candidate_count"
    ] == 2


def test_selection_validator_rejects_mutation_leakage_and_decision_corruption() -> None:
    role = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "dev_017", "split": "dev"}],
        source_cache_global_hash="a" * 64,
        source_cache_manifest_sha256="b" * 64,
        audit_membership_source_sha256="c" * 64,
    )
    cache_manifest = {"global_semantic_sha256": "a" * 64}
    records = {"dev_017": _cache_record()}
    result = create_selection_result(
        cache_manifest, role, records, _selector("SEL-0", {})
    )

    corruptions = []
    modified_boundary = deepcopy(result)
    modified_boundary["records"][0]["candidate_decisions"][0]["end_sec"] = 99.0
    corruptions.append(modified_boundary)
    reference_leak = deepcopy(result)
    reference_leak["weak_reference_segments"] = []
    corruptions.append(reference_leak)
    metric_leak = deepcopy(result)
    metric_leak["metrics"] = {"f1": 1.0}
    corruptions.append(metric_leak)
    audit_label_leak = deepcopy(result)
    audit_label_leak["S1"] = "forbidden"
    corruptions.append(audit_label_leak)
    modified_score = deepcopy(result)
    modified_score["records"][0]["candidate_decisions"][0]["score"] = 0.01
    corruptions.append(modified_score)
    unknown = deepcopy(result)
    unknown["records"][0]["candidate_decisions"][0]["merged_candidate_id"] = "unknown"
    corruptions.append(unknown)
    duplicate = deepcopy(result)
    duplicate["records"][0]["candidate_decisions"][1] = deepcopy(
        duplicate["records"][0]["candidate_decisions"][0]
    )
    corruptions.append(duplicate)
    missing = deepcopy(result)
    missing["records"][0]["candidate_decisions"].pop()
    corruptions.append(missing)

    for corrupted in corruptions:
        try:
            validate_selection_result_payload(corrupted, cache_manifest, role, records)
        except CandidateSelectionError:
            pass
        else:
            raise AssertionError("corrupted selection result must fail")


def test_role_validator_rejects_audit_in_dev_and_heldout() -> None:
    bad_dev = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "audit_member", "split": "dev"}],
        source_cache_global_hash="a" * 64,
        source_cache_manifest_sha256="b" * 64,
        audit_membership_source_sha256="c" * 64,
    )
    try:
        validate_role_manifest(
            bad_dev,
            expected_count=1,
            forbidden_video_ids={"audit_member"},
        )
    except CandidateSelectionError:
        pass
    else:
        raise AssertionError("Audit member in Dev-Tune must fail")

    hard_in_dev = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "hard_member", "split": "hard"}],
        source_cache_global_hash="a" * 64,
        source_cache_manifest_sha256="b" * 64,
        audit_membership_source_sha256="c" * 64,
    )
    try:
        validate_role_manifest(hard_in_dev, expected_count=1)
    except CandidateSelectionError:
        pass
    else:
        raise AssertionError("Hard member in Dev-Tune must fail")

    heldout = deepcopy(bad_dev)
    heldout["records"][0]["split"] = "heldout"
    try:
        validate_role_manifest(heldout, expected_count=1)
    except CandidateSelectionError:
        pass
    else:
        raise AssertionError("Heldout role record must fail")


def test_sel0_replay_uses_exact_frozen_candidate_values() -> None:
    role = create_role_manifest(
        role="dev_tune",
        records=[{"video_id": "dev_017", "split": "dev"}],
        source_cache_global_hash="a" * 64,
        source_cache_manifest_sha256="b" * 64,
        audit_membership_source_sha256="c" * 64,
    )
    cache_manifest = {"global_semantic_sha256": "a" * 64}
    records = {"dev_017": _cache_record()}
    result = create_selection_result(
        cache_manifest, role, records, _selector("SEL-0", {})
    )

    replay = replay_selection_payload(result, records)
    expected = [
        {
            key: candidate[key]
            for key in ("start_sec", "end_sec", "score", "reason", "source_chunk")
        }
        for candidate in records["dev_017"]["merged_candidates"]
    ]
    assert replay == [
        {
            "video_id": "dev_017",
            "split": "dev",
            "merged_prediction_segments": expected,
        }
    ]


def test_evaluation_is_post_selection_and_sel0_matches_frozen_metric_semantics() -> None:
    selected = [
        {
            "video_id": "dev_017",
            "split": "dev",
            "merged_prediction_segments": [
                {"start_sec": 1.0, "end_sec": 3.0, "score": 0.9, "reason": "x", "source_chunk": 0}
            ],
        }
    ]
    frozen = [
        {
            "video_id": "dev_017",
            "weak_reference_segments": [{"start_sec": 2.0, "end_sec": 4.0}],
        }
    ]
    result = evaluate_replayed_predictions(
        selected,
        frozen,
        durations_by_id={"dev_017": 10.0},
        role="dev_tune",
        role_manifest_hash="d" * 64,
        selection_result_hash="e" * 64,
    )

    metrics = result["per_video"][0]
    assert metrics["weak_ref_precision"] == 0.5
    assert metrics["weak_ref_recall"] == 0.5
    assert metrics["weak_ref_f1"] == 0.5
    assert metrics["temporal_iou"] == 1.0 / 3.0
    assert metrics["prediction_coverage"] == 0.2
    assert metrics["over_prediction_sec"] == 1.0
    assert metrics["missed_reference_sec"] == 1.0
    assert metrics["final_segment_count"] == 1
    assert result["aggregate"]["empty_prediction_count"] == 0


def test_paired_comparison_and_guardrail_are_machine_decidable() -> None:
    selected = [{"video_id": "dev_017", "split": "dev", "merged_prediction_segments": []}]
    frozen = [
        {
            "video_id": "dev_017",
            "weak_reference_segments": [{"start_sec": 2.0, "end_sec": 4.0}],
        }
    ]
    baseline_selected = [
        {
            "video_id": "dev_017",
            "split": "dev",
            "merged_prediction_segments": [
                {"start_sec": 1.0, "end_sec": 4.0, "score": 0.9, "reason": "x", "source_chunk": 0}
            ],
        }
    ]
    baseline = evaluate_replayed_predictions(
        baseline_selected,
        frozen,
        durations_by_id={"dev_017": 10.0},
        role="dev_tune",
        role_manifest_hash="d" * 64,
        selection_result_hash="e" * 64,
    )
    candidate = evaluate_replayed_predictions(
        selected,
        frozen,
        durations_by_id={"dev_017": 10.0},
        role="dev_tune",
        role_manifest_hash="d" * 64,
        selection_result_hash="f" * 64,
    )
    comparison = compare_evaluations(baseline, candidate)
    assert comparison["per_video"][0]["delta_weak_ref_recall"] == -1.0
    assessment = assess_evaluation_comparison(
        comparison,
        {
            "min_mean_recall": 0.94,
            "min_delta_mean_recall": -0.02,
            "max_delta_missed_reference_fraction": 0.02,
            "max_delta_empty_prediction_count": 0,
            "max_delta_recall_eq_0_count": 0,
            "max_delta_recall_lt_0_5_count": 1,
            "max_delta_recall_lt_0_8_count": 3,
        },
    )
    assert assessment["pass"] is False
    assert "min_mean_recall" in assessment["failed_checks"]


def test_protocol_grid_and_no_video_specific_reason_rule() -> None:
    protocol = {
        "selectors": {
            "SEL-1": {
                "selector_version": SELECTOR_VERSION,
                "parameter_name": "threshold",
                "grid": [0.7, 0.8],
            }
        }
    }
    assert selector_config_from_protocol(protocol, "SEL-1", 0.8)["parameters"] == {
        "threshold": 0.8
    }
    try:
        selector_config_from_protocol(protocol, "SEL-1", 0.81)
    except CandidateSelectionError:
        pass
    else:
        raise AssertionError("out-of-grid parameter must fail")

    video_rule = _selector(
        "SEL-3",
        {
            "rules": [
                {
                    "rule_id": "qvh_000001_special",
                    "pattern": "anything",
                    "positive_override_patterns": [],
                }
            ]
        },
    )
    try:
        apply_selector([_candidate("a", 0.8)], video_rule)
    except CandidateSelectionError:
        pass
    else:
        raise AssertionError("video-specific lexical rule must fail")


def test_dev_parameter_freeze_uses_preregistered_conservative_tie_break() -> None:
    selected = [
        {
            "video_id": "dev_017",
            "split": "dev",
            "merged_prediction_segments": [
                {"start_sec": 1.0, "end_sec": 3.0, "score": 0.9, "reason": "x", "source_chunk": 0}
            ],
        }
    ]
    frozen = [
        {
            "video_id": "dev_017",
            "weak_reference_segments": [{"start_sec": 1.0, "end_sec": 3.0}],
        }
    ]
    baseline = evaluate_replayed_predictions(
        selected,
        frozen,
        durations_by_id={"dev_017": 10.0},
        role="dev_tune",
        role_manifest_hash="d" * 64,
        selection_result_hash="e" * 64,
    )
    candidates = []
    for index, threshold in enumerate((0.8, 0.75)):
        config = _selector("SEL-1", {"threshold": threshold})
        candidates.append(
            evaluate_replayed_predictions(
                selected,
                frozen,
                durations_by_id={"dev_017": 10.0},
                role="dev_tune",
                role_manifest_hash="d" * 64,
                selection_result_hash=("f" if index == 0 else "a") * 64,
                selection_metadata={
                    "selector_name": "SEL-1",
                    "selector_config": config,
                    "selector_config_hash": semantic_sha256(config),
                },
            )
        )
    protocol = {
        "protocol_semantic_sha256": "1" * 64,
        "selectors": {
            "SEL-1": {
                "selector_version": SELECTOR_VERSION,
                "parameter_name": "threshold",
                "grid": [0.75, 0.8],
            }
        },
        "dev_recall_guardrail": {
            "min_mean_recall": 0.94,
            "min_delta_mean_recall": -0.02,
            "max_delta_missed_reference_fraction": 0.02,
            "max_delta_empty_prediction_count": 0,
            "max_delta_recall_eq_0_count": 0,
            "max_delta_recall_lt_0_5_count": 1,
            "max_delta_recall_lt_0_8_count": 3,
        },
    }

    freeze = choose_dev_evaluation(baseline, candidates, protocol, "SEL-1")
    assert freeze["status"] == "FROZEN"
    assert freeze["chosen_selector_config"]["parameters"]["threshold"] == 0.75
