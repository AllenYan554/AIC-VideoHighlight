"""Deterministic Stage 4.3 filtering over the frozen candidate cache.

Selection and evaluation are deliberately separated: selector functions only
receive frozen merged candidates and candidate-internal statistics.  Weak
references are accepted solely by the post-selection evaluation functions.
"""

from __future__ import annotations

import math
import re
import json
import statistics
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .candidate_cache import canonical_json_bytes, semantic_sha256, validate_cache
from .baseline_experiment import evaluate_weak_references


ROLE_MANIFEST_SCHEMA_VERSION = "aic.stage4-role-manifest/v1"
ROLE_MANIFEST_SUMMARY_SCHEMA_VERSION = "aic.stage4-role-manifest-summary/v1"
SELECTION_RESULT_SCHEMA_VERSION = "aic.candidate-selection-result/v1"
SELECTION_EVALUATION_SCHEMA_VERSION = "aic.candidate-selection-evaluation/v1"
SELECTOR_VERSION = "aic.lightweight-candidate-selection/v1"
EXPECTED_CACHE_GLOBAL_HASH = (
    "4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246"
)
EXPECTED_ROLE_COUNTS = {
    "dev_tune": 166,
    "hard_stress": 229,
    "audit_diagnostic": 36,
}


class CandidateSelectionError(ValueError):
    """Raised when a Stage 4.3 isolation or selection invariant is broken."""


def partition_cache_records(
    cache_records: Iterable[Mapping[str, Any]],
    audit_membership: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Partition Dev/Hard cache identities without inspecting audit labels."""
    records = list(cache_records)
    cache_by_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    for item in records:
        video_id = item.get("video_id")
        split = item.get("split")
        if not isinstance(video_id, str) or not video_id or video_id in cache_by_id:
            raise CandidateSelectionError("cache contains duplicate or invalid video_id")
        if split not in {"dev", "hard"}:
            raise CandidateSelectionError("cache contains Heldout or an unknown split")
        cache_by_id[video_id] = str(split)
        ordered_ids.append(video_id)

    audit_by_id: dict[str, str] = {}
    for item in audit_membership:
        # Intentionally access only identity fields.  Other source fields are
        # outside the Stage 4.3 data-isolation contract.
        video_id = item.get("video_id")
        split = item.get("split")
        if not isinstance(video_id, str) or not video_id:
            raise CandidateSelectionError("audit membership has an invalid video_id")
        if video_id in audit_by_id:
            raise CandidateSelectionError("audit membership contains a duplicate video_id")
        if video_id not in cache_by_id:
            raise CandidateSelectionError(f"audit membership references unknown video: {video_id}")
        if split != cache_by_id[video_id]:
            raise CandidateSelectionError(f"audit membership split mismatch for {video_id}")
        audit_by_id[video_id] = str(split)

    roles = {name: [] for name in EXPECTED_ROLE_COUNTS}
    for video_id in ordered_ids:
        split = cache_by_id[video_id]
        record = {"video_id": video_id, "split": split}
        if video_id in audit_by_id:
            roles["audit_diagnostic"].append(record)
        elif split == "dev":
            roles["dev_tune"].append(record)
        else:
            roles["hard_stress"].append(record)
    return roles


def _finite_unit_interval(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateSelectionError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CandidateSelectionError(f"{field} must be finite and in [0, 1]")
    return result


def _validate_selector_config(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if set(config) != {"selector_name", "selector_version", "parameters"}:
        raise CandidateSelectionError("selector config has unexpected or missing fields")
    name = config.get("selector_name")
    if name not in {"SEL-0", "SEL-1", "SEL-2", "SEL-3"}:
        raise CandidateSelectionError("unknown selector_name")
    if config.get("selector_version") != SELECTOR_VERSION:
        raise CandidateSelectionError("unsupported selector_version")
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CandidateSelectionError("selector parameters must be an object")
    if name == "SEL-0" and parameters:
        raise CandidateSelectionError("SEL-0 does not accept parameters")
    if name == "SEL-1" and set(parameters) != {"threshold"}:
        raise CandidateSelectionError("SEL-1 requires only threshold")
    if name == "SEL-2" and set(parameters) != {"delta"}:
        raise CandidateSelectionError("SEL-2 requires only delta")
    if name == "SEL-3" and set(parameters) != {"rules"}:
        raise CandidateSelectionError("SEL-3 requires only frozen rules")
    return str(name), parameters


def apply_selector(
    candidates: Iterable[Mapping[str, Any]],
    selector_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return one deterministic KEEP/DROP decision per input candidate."""
    name, parameters = _validate_selector_config(selector_config)
    items = list(candidates)
    for candidate in items:
        if not isinstance(candidate.get("merged_candidate_id"), str):
            raise CandidateSelectionError("candidate is missing merged_candidate_id")
        _finite_unit_interval(candidate.get("score"), "candidate score")

    maximum = max((float(item["score"]) for item in items), default=None)
    decisions: list[dict[str, Any]] = []
    for candidate in items:
        candidate_id = str(candidate["merged_candidate_id"])
        score = float(candidate["score"])
        if name == "SEL-0":
            decision = "KEEP"
            rule = "sel0.identity_keep"
            value: Any = None
        elif name == "SEL-1":
            threshold = _finite_unit_interval(parameters["threshold"], "threshold")
            decision = "KEEP" if score >= threshold else "DROP"
            rule = "sel1.score_gte_threshold"
            value = {"score": score, "threshold": threshold}
        elif name == "SEL-2":
            delta = _finite_unit_interval(parameters["delta"], "delta")
            if maximum is None:  # pragma: no cover - loop is empty in this case
                raise AssertionError("unreachable maximum state")
            gap = maximum - score
            decision = "KEEP" if score >= maximum - delta else "DROP"
            rule = "sel2.score_within_video_max_delta"
            value = {"score": score, "video_max_score": maximum, "score_gap": gap, "delta": delta}
        else:
            rules = parameters["rules"]
            if not isinstance(rules, list) or not rules:
                raise CandidateSelectionError("SEL-3 rules must be a non-empty list")
            reason = candidate.get("reason")
            if not isinstance(reason, str):
                raise CandidateSelectionError("candidate reason must be a string")
            matched_rule_ids: list[str] = []
            for rule_config in rules:
                if not isinstance(rule_config, Mapping) or set(rule_config) != {
                    "rule_id",
                    "pattern",
                    "positive_override_patterns",
                }:
                    raise CandidateSelectionError("invalid SEL-3 rule schema")
                rule_id = rule_config["rule_id"]
                pattern = rule_config["pattern"]
                overrides = rule_config["positive_override_patterns"]
                if (
                    not isinstance(rule_id, str)
                    or not rule_id
                    or not isinstance(pattern, str)
                    or not isinstance(overrides, list)
                    or not all(isinstance(item, str) for item in overrides)
                ):
                    raise CandidateSelectionError("invalid SEL-3 rule value")
                serialized_rule = f"{rule_id} {pattern} {' '.join(overrides)}"
                if "video_id" in serialized_rule.lower() or re.search(
                    r"qvh_\d+", serialized_rule, flags=re.IGNORECASE
                ):
                    raise CandidateSelectionError("video-specific SEL-3 rule is forbidden")
                try:
                    negative_match = re.search(pattern, reason) is not None
                    positive_override = any(
                        re.search(item, reason) is not None for item in overrides
                    )
                except re.error as exc:
                    raise CandidateSelectionError(f"invalid SEL-3 regex: {rule_id}") from exc
                if negative_match and not positive_override:
                    matched_rule_ids.append(rule_id)
            decision = "DROP" if matched_rule_ids else "KEEP"
            rule = (
                "sel3." + "+".join(matched_rule_ids)
                if matched_rule_ids
                else "sel3.uncertain_or_no_clear_negative_keep"
            )
            value = {"matched_rule_ids": matched_rule_ids}
        decisions.append(
            {
                "merged_candidate_id": candidate_id,
                "decision": decision,
                "decision_rule": rule,
                "selector_value": value,
            }
        )
    return decisions


def _canonical_copy(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CandidateSelectionError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CandidateSelectionError(f"{field} must be a SHA-256 hex digest") from exc
    return value


def _semantic_payload(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != hash_field}


def create_role_manifest(
    *,
    role: str,
    records: Iterable[Mapping[str, Any]],
    source_cache_global_hash: str,
    source_cache_manifest_sha256: str,
    audit_membership_source_sha256: str,
) -> dict[str, Any]:
    """Create one canonical role manifest from identity-only records."""
    if role not in EXPECTED_ROLE_COUNTS:
        raise CandidateSelectionError("unknown Stage 4.3 role")
    identities = [
        {"video_id": str(item["video_id"]), "split": str(item["split"])}
        for item in records
    ]
    split_counts = {
        split: sum(item["split"] == split for item in identities)
        for split in ("dev", "hard")
        if any(item["split"] == split for item in identities)
    }
    manifest: dict[str, Any] = {
        "role_manifest_schema_version": ROLE_MANIFEST_SCHEMA_VERSION,
        "role": role,
        "source_cache_global_hash": _require_sha256(
            source_cache_global_hash, "source_cache_global_hash"
        ),
        "record_count": len(identities),
        "split_counts": split_counts,
        "records": identities,
        "generation_provenance": {
            "generator": "candidate_selection.build_role_manifests",
            "generator_version": SELECTOR_VERSION,
            "source_cache_manifest_sha256": _require_sha256(
                source_cache_manifest_sha256, "source_cache_manifest_sha256"
            ),
            "audit_membership_source_sha256": _require_sha256(
                audit_membership_source_sha256, "audit_membership_source_sha256"
            ),
            "identity_projection_fields": ["video_id", "split"],
            "heldout_accessed": False,
        },
    }
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    return manifest


def validate_role_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_count: int | None = None,
    forbidden_video_ids: set[str] | None = None,
) -> dict[str, Any]:
    if manifest.get("role_manifest_schema_version") != ROLE_MANIFEST_SCHEMA_VERSION:
        raise CandidateSelectionError("unsupported role manifest schema")
    role = manifest.get("role")
    if role not in EXPECTED_ROLE_COUNTS:
        raise CandidateSelectionError("unknown role manifest role")
    _require_sha256(manifest.get("source_cache_global_hash"), "source cache hash")
    claimed_hash = _require_sha256(manifest.get("semantic_sha256"), "role manifest hash")
    if claimed_hash != semantic_sha256(_semantic_payload(manifest, "semantic_sha256")):
        raise CandidateSelectionError("role manifest semantic hash mismatch")
    provenance = manifest.get("generation_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("heldout_accessed") is not False:
        raise CandidateSelectionError("role manifest provenance is invalid")
    _require_sha256(provenance.get("source_cache_manifest_sha256"), "cache manifest hash")
    _require_sha256(
        provenance.get("audit_membership_source_sha256"), "audit membership source hash"
    )
    if provenance.get("identity_projection_fields") != ["video_id", "split"]:
        raise CandidateSelectionError("role manifest used fields outside identity projection")
    records = manifest.get("records")
    if not isinstance(records, list) or manifest.get("record_count") != len(records):
        raise CandidateSelectionError("role manifest record_count mismatch")
    if expected_count is not None and len(records) != expected_count:
        raise CandidateSelectionError("role manifest has the wrong expected count")
    seen: set[str] = set()
    actual_split_counts: dict[str, int] = {}
    for item in records:
        if not isinstance(item, Mapping) or set(item) != {"video_id", "split"}:
            raise CandidateSelectionError("role record must contain only video_id and split")
        video_id = item.get("video_id")
        split = item.get("split")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise CandidateSelectionError("role manifest has duplicate or invalid video_id")
        if split not in {"dev", "hard"}:
            raise CandidateSelectionError("Heldout or unknown split in role manifest")
        if role == "dev_tune" and split != "dev":
            raise CandidateSelectionError("Hard video is forbidden in Dev-Tune")
        if role == "hard_stress" and split != "hard":
            raise CandidateSelectionError("Dev video is forbidden in Hard-Stress")
        if forbidden_video_ids and video_id in forbidden_video_ids:
            raise CandidateSelectionError("forbidden Audit member appears in tuning role")
        seen.add(video_id)
        actual_split_counts[str(split)] = actual_split_counts.get(str(split), 0) + 1
    if manifest.get("split_counts") != actual_split_counts:
        raise CandidateSelectionError("role manifest split_counts mismatch")
    return {"role": role, "record_count": len(records), "semantic_sha256": claimed_hash}


_SELECTION_FORBIDDEN_KEY_FRAGMENTS = (
    "reference",
    "metric",
    "precision",
    "recall",
    "temporal_iou",
    "f1",
    "audit_id",
    "adjudicat",
    "reviewer",
    "human",
    "label",
)
_SELECTION_FORBIDDEN_VALUE_FRAGMENTS = (
    "weak_reference",
    "reference segment",
    "precision",
    "recall",
    "temporal_iou",
    "adjudicat",
    "reviewer rationale",
    "human rationale",
    "heldout",
)


def _check_selection_leakage(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).lower()
            if normalized in {"s1", "s2", "s3", "s4", "s5"} or any(
                fragment in normalized for fragment in _SELECTION_FORBIDDEN_KEY_FRAGMENTS
            ):
                raise CandidateSelectionError(f"forbidden selector field at {path}.{key}")
            _check_selection_leakage(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _check_selection_leakage(value, f"{path}[{index}]")
    elif isinstance(payload, str):
        normalized = payload.lower()
        if normalized in {"s1", "s2", "s3", "s4", "s5"} or any(
            fragment in normalized for fragment in _SELECTION_FORBIDDEN_VALUE_FRAGMENTS
        ):
            raise CandidateSelectionError(f"forbidden selector value at {path}")


def create_selection_result(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic selection result without accepting references."""
    validate_role_manifest(role_manifest, expected_count=len(role_manifest.get("records", [])))
    cache_hash = cache_manifest.get("global_semantic_sha256")
    if cache_hash != role_manifest.get("source_cache_global_hash"):
        raise CandidateSelectionError("role manifest does not match source cache")
    name, _ = _validate_selector_config(selector_config)
    config_payload = _canonical_copy(selector_config)
    config_hash = semantic_sha256(config_payload)
    output_records: list[dict[str, Any]] = []
    input_count = 0
    selected_count = 0
    for identity in role_manifest["records"]:
        video_id = identity["video_id"]
        record = cache_records_by_id.get(video_id)
        if record is None:
            raise CandidateSelectionError(f"role video is missing from cache: {video_id}")
        if record.get("split") != identity["split"]:
            raise CandidateSelectionError(f"role/cache split mismatch for {video_id}")
        candidates = record.get("merged_candidates")
        if not isinstance(candidates, list):
            raise CandidateSelectionError(f"cache merged_candidates missing for {video_id}")
        base_decisions = apply_selector(candidates, selector_config)
        decisions = []
        for candidate, decision in zip(candidates, base_decisions, strict=True):
            decisions.append(
                {
                    **decision,
                    "original_candidate_semantic_sha256": semantic_sha256(candidate),
                }
            )
        selected_ids = [
            item["merged_candidate_id"]
            for item in decisions
            if item["decision"] == "KEEP"
        ]
        output_record: dict[str, Any] = {
            "video_id": video_id,
            "split": identity["split"],
            "source_record_semantic_sha256": _require_sha256(
                record.get("semantic_sha256"), f"{video_id} source record hash"
            ),
            "candidate_decisions": decisions,
            "selected_candidate_ids": selected_ids,
        }
        output_record["video_selection_semantic_sha256"] = semantic_sha256(output_record)
        output_records.append(output_record)
        input_count += len(candidates)
        selected_count += len(selected_ids)
    result: dict[str, Any] = {
        "selector_schema_version": SELECTION_RESULT_SCHEMA_VERSION,
        "source_cache_global_hash": cache_hash,
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "selector_name": name,
        "selector_version": SELECTOR_VERSION,
        "selector_config": config_payload,
        "selector_config_hash": config_hash,
        "record_count": len(output_records),
        "input_candidate_count": input_count,
        "selected_candidate_count": selected_count,
        "records": output_records,
    }
    _check_selection_leakage(result)
    result["selection_result_semantic_hash"] = semantic_sha256(result)
    return result


def validate_selection_result_payload(
    result: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate schema, leakage, hashes, completeness, and exact decisions."""
    _check_selection_leakage(result)
    if result.get("selector_schema_version") != SELECTION_RESULT_SCHEMA_VERSION:
        raise CandidateSelectionError("unsupported selection result schema")
    claimed_hash = _require_sha256(
        result.get("selection_result_semantic_hash"), "selection result hash"
    )
    if claimed_hash != semantic_sha256(
        _semantic_payload(result, "selection_result_semantic_hash")
    ):
        raise CandidateSelectionError("selection result semantic hash mismatch")
    selector_config = result.get("selector_config")
    if not isinstance(selector_config, Mapping):
        raise CandidateSelectionError("selection result selector_config is missing")
    expected = create_selection_result(
        cache_manifest, role_manifest, cache_records_by_id, selector_config
    )
    if _canonical_copy(result) != expected:
        raise CandidateSelectionError(
            "selection result is not the exact deterministic KEEP/DROP projection"
        )
    return {
        "record_count": result["record_count"],
        "input_candidate_count": result["input_candidate_count"],
        "selected_candidate_count": result["selected_candidate_count"],
        "selection_result_semantic_hash": claimed_hash,
    }


def replay_selection_payload(
    result: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay selected predictions by retrieving immutable cache candidates."""
    replayed: list[dict[str, Any]] = []
    result_records = result.get("records")
    if not isinstance(result_records, list):
        raise CandidateSelectionError("selection result records are missing")
    for item in result_records:
        video_id = item.get("video_id")
        record = cache_records_by_id.get(str(video_id))
        if record is None:
            raise CandidateSelectionError(f"selection video is missing from cache: {video_id}")
        candidates = record.get("merged_candidates")
        if not isinstance(candidates, list):
            raise CandidateSelectionError(f"cache merged_candidates missing for {video_id}")
        by_id = {candidate["merged_candidate_id"]: candidate for candidate in candidates}
        selected_ids = item.get("selected_candidate_ids")
        if not isinstance(selected_ids, list) or any(key not in by_id for key in selected_ids):
            raise CandidateSelectionError(f"unknown selected candidate for {video_id}")
        segments = [
            {
                "start_sec": float(by_id[key]["start_sec"]),
                "end_sec": float(by_id[key]["end_sec"]),
                "score": float(by_id[key]["score"]),
                "reason": str(by_id[key].get("reason", "")),
                "source_chunk": by_id[key].get("source_chunk"),
            }
            for key in selected_ids
        ]
        replayed.append(
            {
                "video_id": video_id,
                "split": item.get("split"),
                "merged_prediction_segments": segments,
            }
        )
    return replayed


def evaluate_replayed_predictions(
    selected_predictions: Iterable[Mapping[str, Any]],
    frozen_predictions: Iterable[Mapping[str, Any]],
    *,
    durations_by_id: Mapping[str, float],
    role: str,
    role_manifest_hash: str,
    selection_result_hash: str,
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate after selection using the unchanged frozen metric function."""
    if role not in EXPECTED_ROLE_COUNTS:
        raise CandidateSelectionError("unknown evaluation role")
    _require_sha256(role_manifest_hash, "role_manifest_hash")
    _require_sha256(selection_result_hash, "selection_result_hash")
    frozen_by_id: dict[str, Mapping[str, Any]] = {}
    for item in frozen_predictions:
        video_id = item.get("video_id")
        if not isinstance(video_id, str) or video_id in frozen_by_id:
            raise CandidateSelectionError("frozen predictions contain invalid video IDs")
        frozen_by_id[video_id] = item

    per_video: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in selected_predictions:
        video_id = item.get("video_id")
        split = item.get("split")
        if not isinstance(video_id, str) or video_id in seen:
            raise CandidateSelectionError("selected predictions contain invalid video IDs")
        if split not in {"dev", "hard"}:
            raise CandidateSelectionError("Heldout or unknown split in selected predictions")
        frozen = frozen_by_id.get(video_id)
        if frozen is None:
            raise CandidateSelectionError(f"missing frozen prediction for {video_id}")
        segments = item.get("merged_prediction_segments")
        references = frozen.get("weak_reference_segments")
        if not isinstance(segments, list) or not isinstance(references, list):
            raise CandidateSelectionError(f"evaluation inputs are incomplete for {video_id}")
        duration = float(durations_by_id.get(video_id, 0.0))
        if not math.isfinite(duration) or duration <= 0:
            raise CandidateSelectionError(f"invalid duration for {video_id}")
        frozen_metrics = evaluate_weak_references(segments, references)
        predicted_sec = float(frozen_metrics["prediction_duration_sec"])
        reference_sec = float(frozen_metrics["reference_duration_sec"])
        intersection_sec = float(frozen_metrics["intersection_sec"])
        per_video.append(
            {
                "video_id": video_id,
                "split": split,
                "weak_ref_precision": float(frozen_metrics["weak_ref_precision"]),
                "weak_ref_recall": float(frozen_metrics["weak_ref_recall"]),
                "weak_ref_f1": float(frozen_metrics["weak_ref_f1"]),
                "temporal_iou": float(frozen_metrics["temporal_iou"]),
                "prediction_coverage": predicted_sec / duration,
                "over_prediction_sec": predicted_sec - intersection_sec,
                "missed_reference_sec": reference_sec - intersection_sec,
                "prediction_duration_sec": predicted_sec,
                "reference_duration_sec": reference_sec,
                "intersection_sec": intersection_sec,
                "final_segment_count": len(segments),
                "empty_prediction": len(segments) == 0,
                "zero_duration_reference_count": int(
                    frozen_metrics["zero_duration_reference_count"]
                ),
            }
        )
        seen.add(video_id)

    mean_fields = (
        "weak_ref_precision",
        "weak_ref_recall",
        "weak_ref_f1",
        "temporal_iou",
        "prediction_coverage",
        "over_prediction_sec",
        "missed_reference_sec",
        "final_segment_count",
    )
    aggregate: dict[str, Any] = {
        f"mean_{field}": statistics.fmean(float(item[field]) for item in per_video)
        if per_video
        else 0.0
        for field in mean_fields
    }
    total_reference = sum(float(item["reference_duration_sec"]) for item in per_video)
    total_missed = sum(float(item["missed_reference_sec"]) for item in per_video)
    aggregate.update(
        {
            "record_count": len(per_video),
            "empty_prediction_count": sum(bool(item["empty_prediction"]) for item in per_video),
            "recall_lt_0_8_count": sum(item["weak_ref_recall"] < 0.8 for item in per_video),
            "recall_lt_0_5_count": sum(item["weak_ref_recall"] < 0.5 for item in per_video),
            "recall_eq_0_count": sum(item["weak_ref_recall"] == 0.0 for item in per_video),
            "total_prediction_duration_sec": sum(
                float(item["prediction_duration_sec"]) for item in per_video
            ),
            "total_reference_duration_sec": total_reference,
            "total_over_prediction_sec": sum(
                float(item["over_prediction_sec"]) for item in per_video
            ),
            "total_missed_reference_sec": total_missed,
            "missed_reference_fraction": (
                total_missed / total_reference if total_reference else float(total_missed > 0)
            ),
            "final_segment_count": sum(int(item["final_segment_count"]) for item in per_video),
        }
    )
    evaluation: dict[str, Any] = {
        "evaluation_schema_version": SELECTION_EVALUATION_SCHEMA_VERSION,
        "role": role,
        "role_manifest_hash": role_manifest_hash,
        "selection_result_hash": selection_result_hash,
        "metric_implementation": (
            "aic_video_highlight.highlight_retrieval.baseline_experiment."
            "evaluate_weak_references"
        ),
        "record_count": len(per_video),
        "per_video": per_video,
        "aggregate": aggregate,
    }
    if selection_metadata is not None:
        evaluation["selection_metadata"] = _canonical_copy(selection_metadata)
    evaluation["evaluation_semantic_hash"] = semantic_sha256(evaluation)
    return evaluation


_PAIRED_METRIC_FIELDS = (
    "weak_ref_precision",
    "weak_ref_recall",
    "weak_ref_f1",
    "temporal_iou",
    "prediction_coverage",
    "over_prediction_sec",
    "missed_reference_sec",
    "final_segment_count",
)


def _validate_evaluation_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("evaluation_schema_version") != SELECTION_EVALUATION_SCHEMA_VERSION:
        raise CandidateSelectionError("unsupported evaluation schema")
    claimed = _require_sha256(payload.get("evaluation_semantic_hash"), "evaluation hash")
    if claimed != semantic_sha256(_semantic_payload(payload, "evaluation_semantic_hash")):
        raise CandidateSelectionError("evaluation semantic hash mismatch")


def compare_evaluations(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Create exact paired deltas against the role-matched SEL-0 baseline."""
    _validate_evaluation_payload(baseline)
    _validate_evaluation_payload(candidate)
    if baseline.get("role") != candidate.get("role") or baseline.get(
        "role_manifest_hash"
    ) != candidate.get("role_manifest_hash"):
        raise CandidateSelectionError("evaluations are not from the same frozen role")
    baseline_rows = baseline.get("per_video")
    candidate_rows = candidate.get("per_video")
    if not isinstance(baseline_rows, list) or not isinstance(candidate_rows, list):
        raise CandidateSelectionError("evaluation per_video rows are missing")
    if [item.get("video_id") for item in baseline_rows] != [
        item.get("video_id") for item in candidate_rows
    ]:
        raise CandidateSelectionError("evaluation video order/identity mismatch")
    paired: list[dict[str, Any]] = []
    for base, current in zip(baseline_rows, candidate_rows, strict=True):
        row = {"video_id": base["video_id"], "split": base["split"]}
        for field in _PAIRED_METRIC_FIELDS:
            row[f"baseline_{field}"] = base[field]
            row[f"candidate_{field}"] = current[field]
            row[f"delta_{field}"] = float(current[field]) - float(base[field])
        paired.append(row)
    base_aggregate = baseline.get("aggregate")
    current_aggregate = candidate.get("aggregate")
    if not isinstance(base_aggregate, Mapping) or not isinstance(current_aggregate, Mapping):
        raise CandidateSelectionError("evaluation aggregate is missing")
    aggregate_delta = {
        f"delta_{key}": float(current_aggregate[key]) - float(value)
        for key, value in base_aggregate.items()
        if key in current_aggregate and isinstance(value, (int, float))
    }
    comparison: dict[str, Any] = {
        "comparison_schema_version": "aic.candidate-selection-comparison/v1",
        "role": baseline["role"],
        "role_manifest_hash": baseline["role_manifest_hash"],
        "baseline_evaluation_hash": baseline["evaluation_semantic_hash"],
        "candidate_evaluation_hash": candidate["evaluation_semantic_hash"],
        "record_count": len(paired),
        "baseline_aggregate": _canonical_copy(base_aggregate),
        "candidate_aggregate": _canonical_copy(current_aggregate),
        "aggregate_delta": aggregate_delta,
        "per_video": paired,
    }
    if "selection_metadata" in candidate:
        comparison["candidate_selection_metadata"] = _canonical_copy(
            candidate["selection_metadata"]
        )
    comparison["comparison_semantic_hash"] = semantic_sha256(comparison)
    return comparison


_GATE_VALUE_PATHS = {
    "min_mean_recall": ("candidate_aggregate", "mean_weak_ref_recall", "min"),
    "min_delta_mean_recall": ("aggregate_delta", "delta_mean_weak_ref_recall", "min"),
    "max_delta_missed_reference_fraction": (
        "aggregate_delta",
        "delta_missed_reference_fraction",
        "max",
    ),
    "max_delta_empty_prediction_count": (
        "aggregate_delta",
        "delta_empty_prediction_count",
        "max",
    ),
    "max_delta_recall_eq_0_count": (
        "aggregate_delta",
        "delta_recall_eq_0_count",
        "max",
    ),
    "max_delta_recall_lt_0_5_count": (
        "aggregate_delta",
        "delta_recall_lt_0_5_count",
        "max",
    ),
    "max_delta_recall_lt_0_8_count": (
        "aggregate_delta",
        "delta_recall_lt_0_8_count",
        "max",
    ),
    "min_delta_mean_f1": ("aggregate_delta", "delta_mean_weak_ref_f1", "min"),
    "min_delta_mean_precision": (
        "aggregate_delta",
        "delta_mean_weak_ref_precision",
        "min",
    ),
    "min_delta_mean_temporal_iou": (
        "aggregate_delta",
        "delta_mean_temporal_iou",
        "min",
    ),
    "max_delta_mean_prediction_coverage": (
        "aggregate_delta",
        "delta_mean_prediction_coverage",
        "max",
    ),
    "max_delta_total_over_prediction_sec": (
        "aggregate_delta",
        "delta_total_over_prediction_sec",
        "max",
    ),
}


def assess_evaluation_comparison(
    comparison: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a fully machine-readable pre-registered gate."""
    unknown = set(gate) - set(_GATE_VALUE_PATHS)
    if unknown:
        raise CandidateSelectionError(f"unknown gate checks: {sorted(unknown)}")
    checks: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for name, threshold_value in gate.items():
        section, key, direction = _GATE_VALUE_PATHS[name]
        values = comparison.get(section)
        if not isinstance(values, Mapping) or key not in values:
            raise CandidateSelectionError(f"comparison lacks gate value: {section}.{key}")
        observed = float(values[key])
        threshold = float(threshold_value)
        passed = observed >= threshold if direction == "min" else observed <= threshold
        checks[name] = {
            "observed": observed,
            "operator": ">=" if direction == "min" else "<=",
            "threshold": threshold,
            "pass": passed,
        }
        if not passed:
            failed.append(name)
    return {"pass": not failed, "failed_checks": failed, "checks": checks}


def choose_dev_evaluation(
    baseline: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    selector_name: str,
) -> dict[str, Any]:
    """Freeze one SEL-1/SEL-2 config using only the registered Dev ordering."""
    if selector_name not in {"SEL-1", "SEL-2"}:
        raise CandidateSelectionError("parameter freeze supports only SEL-1 and SEL-2")
    _validate_evaluation_payload(baseline)
    if baseline.get("role") != "dev_tune":
        raise CandidateSelectionError("parameter selection is allowed only on Dev-Tune")
    selectors = protocol.get("selectors")
    if not isinstance(selectors, Mapping) or selector_name not in selectors:
        raise CandidateSelectionError("selector is absent from frozen protocol")
    definition = selectors[selector_name]
    parameter_name = definition.get("parameter_name")
    if parameter_name not in {"threshold", "delta"}:
        raise CandidateSelectionError("invalid selector parameter in protocol")
    rows: list[dict[str, Any]] = []
    seen_parameters: set[float] = set()
    for evaluation in candidates:
        _validate_evaluation_payload(evaluation)
        metadata = evaluation.get("selection_metadata")
        if not isinstance(metadata, Mapping) or metadata.get("selector_name") != selector_name:
            raise CandidateSelectionError("candidate evaluation selector metadata mismatch")
        config = metadata.get("selector_config")
        if not isinstance(config, Mapping):
            raise CandidateSelectionError("candidate evaluation lacks selector config")
        parameters = config.get("parameters")
        if not isinstance(parameters, Mapping) or set(parameters) != {parameter_name}:
            raise CandidateSelectionError("candidate selector parameter schema mismatch")
        parameter = float(parameters[parameter_name])
        expected_config = selector_config_from_protocol(protocol, selector_name, parameter)
        if config != expected_config:
            raise CandidateSelectionError("candidate config differs from frozen protocol")
        if metadata.get("selector_config_hash") != semantic_sha256(config):
            raise CandidateSelectionError("candidate selector config hash mismatch")
        if parameter in seen_parameters:
            raise CandidateSelectionError("duplicate parameter evaluation")
        seen_parameters.add(parameter)
        comparison = compare_evaluations(baseline, evaluation)
        guardrail = assess_evaluation_comparison(
            comparison, protocol["dev_recall_guardrail"]
        )
        aggregate = evaluation["aggregate"]
        conservative = parameter if selector_name == "SEL-1" else -parameter
        sort_key = (
            -float(aggregate["mean_weak_ref_f1"]),
            -float(aggregate["mean_weak_ref_recall"]),
            int(aggregate["recall_lt_0_8_count"]),
            float(aggregate["missed_reference_fraction"]),
            -float(aggregate["mean_temporal_iou"]),
            float(aggregate["mean_prediction_coverage"]),
            conservative,
            parameter,
        )
        rows.append(
            {
                "parameter": parameter,
                "selector_config": _canonical_copy(config),
                "selector_config_hash": metadata["selector_config_hash"],
                "evaluation_semantic_hash": evaluation["evaluation_semantic_hash"],
                "recall_guardrail": guardrail,
                "rank_values": {
                    "mean_f1": aggregate["mean_weak_ref_f1"],
                    "mean_recall": aggregate["mean_weak_ref_recall"],
                    "recall_lt_0_8_count": aggregate["recall_lt_0_8_count"],
                    "missed_reference_fraction": aggregate[
                        "missed_reference_fraction"
                    ],
                    "mean_temporal_iou": aggregate["mean_temporal_iou"],
                    "mean_prediction_coverage": aggregate[
                        "mean_prediction_coverage"
                    ],
                },
                "_sort_key": sort_key,
            }
        )
    expected_parameters = {float(item) for item in definition.get("grid", [])}
    if seen_parameters != expected_parameters:
        raise CandidateSelectionError(
            "parameter freeze requires every frozen grid value exactly once"
        )
    eligible = sorted(
        (row for row in rows if row["recall_guardrail"]["pass"]),
        key=lambda row: row["_sort_key"],
    )
    public_rows = [
        {key: value for key, value in row.items() if key != "_sort_key"}
        for row in sorted(rows, key=lambda row: row["parameter"])
    ]
    freeze: dict[str, Any] = {
        "parameter_freeze_schema_version": "aic.candidate-selection-parameter-freeze/v1",
        "protocol_semantic_sha256": _require_sha256(
            protocol.get("protocol_semantic_sha256"), "protocol hash"
        ),
        "role": "dev_tune",
        "role_manifest_hash": baseline["role_manifest_hash"],
        "baseline_evaluation_hash": baseline["evaluation_semantic_hash"],
        "selector_name": selector_name,
        "candidate_evaluations": public_rows,
        "status": "FROZEN" if eligible else "NO_ELIGIBLE_CONFIGURATION",
        "chosen_selector_config": (
            _canonical_copy(eligible[0]["selector_config"]) if eligible else None
        ),
        "chosen_selector_config_hash": (
            eligible[0]["selector_config_hash"] if eligible else None
        ),
        "chosen_evaluation_hash": (
            eligible[0]["evaluation_semantic_hash"] if eligible else None
        ),
        "selection_rule": _canonical_copy(protocol.get("parameter_selection_rule")),
        "hard_results_observed": False,
        "audit_results_observed": False,
        "heldout_accessed": False,
    }
    freeze["parameter_freeze_semantic_sha256"] = semantic_sha256(freeze)
    return freeze


_ROLE_FILENAMES = {
    "dev_tune": "dev_tune_166.json",
    "hard_stress": "hard_stress_229.json",
    "audit_diagnostic": "audit36_diagnostic.json",
}
ROLE_SUMMARY_FILENAME = "stage4_3_role_manifest_summary.json"
PROTOCOL_SCHEMA_VERSION = "aic.stage4-selection-experiment-protocol/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateSelectionError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateSelectionError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CandidateSelectionError(f"cannot read JSONL: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateSelectionError(f"invalid JSONL line {index}: {path}") from exc
        if not isinstance(item, dict):
            raise CandidateSelectionError(f"JSONL line {index} is not an object: {path}")
        rows.append(item)
    return rows


def load_experiment_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json_object(path.expanduser().resolve())
    if protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise CandidateSelectionError("unsupported Stage 4.3 protocol schema")
    claimed = _require_sha256(
        protocol.get("protocol_semantic_sha256"), "protocol semantic hash"
    )
    if claimed != semantic_sha256(_semantic_payload(protocol, "protocol_semantic_sha256")):
        raise CandidateSelectionError("protocol semantic hash mismatch")
    if protocol.get("protocol_status") != "PREREGISTERED_BEFORE_FORMAL":
        raise CandidateSelectionError("protocol is not frozen before Formal")
    if protocol.get("source_cache_global_hash") != EXPECTED_CACHE_GLOBAL_HASH:
        raise CandidateSelectionError("protocol references the wrong Frozen Cache")
    selectors = protocol.get("selectors")
    if not isinstance(selectors, Mapping) or set(selectors) != {
        "SEL-0",
        "SEL-1",
        "SEL-2",
        "SEL-3",
    }:
        raise CandidateSelectionError("protocol selector registry is incomplete")
    return protocol


def selector_config_from_protocol(
    protocol: Mapping[str, Any], selector_name: str, parameter: float | None = None
) -> dict[str, Any]:
    selectors = protocol.get("selectors")
    if not isinstance(selectors, Mapping) or selector_name not in selectors:
        raise CandidateSelectionError("selector is absent from protocol")
    definition = selectors[selector_name]
    if not isinstance(definition, Mapping) or definition.get("selector_version") != SELECTOR_VERSION:
        raise CandidateSelectionError("selector protocol version mismatch")
    if selector_name in {"SEL-0", "SEL-3"}:
        if parameter is not None:
            raise CandidateSelectionError(f"{selector_name} does not accept a grid parameter")
        parameters = definition.get("parameters")
    else:
        if parameter is None:
            raise CandidateSelectionError(f"{selector_name} requires a grid parameter")
        grid = definition.get("grid")
        parameter_name = definition.get("parameter_name")
        if not isinstance(grid, list) or parameter not in [float(item) for item in grid]:
            raise CandidateSelectionError("parameter is outside the frozen protocol grid")
        if parameter_name not in {"threshold", "delta"}:
            raise CandidateSelectionError("invalid protocol parameter name")
        parameters = {str(parameter_name): float(parameter)}
    config = {
        "selector_name": selector_name,
        "selector_version": SELECTOR_VERSION,
        "parameters": _canonical_copy(parameters),
    }
    _validate_selector_config(config)
    return config


def _write_new_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(payload))


def _load_cache_payloads(
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = cache_dir.expanduser().resolve()
    summary = validate_cache(root)
    if summary["global_semantic_sha256"] != EXPECTED_CACHE_GLOBAL_HASH:
        raise CandidateSelectionError("Frozen Cache global hash is not the Stage 4.2 frozen hash")
    manifest = _read_json_object(root / "cache_manifest.json")
    records = {
        entry["video_id"]: _read_json_object(root / entry["path"])
        for entry in manifest["records"]
    }
    return manifest, records


def _create_role_summary(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    source_cache_manifest_sha256: str,
    audit_membership_source_sha256: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "role_manifest_summary_schema_version": ROLE_MANIFEST_SUMMARY_SCHEMA_VERSION,
        "source_cache_global_hash": EXPECTED_CACHE_GLOBAL_HASH,
        "source_cache_manifest_sha256": source_cache_manifest_sha256,
        "audit_membership_source_sha256": audit_membership_source_sha256,
        "role_manifests": {
            role: {
                "path": _ROLE_FILENAMES[role],
                "record_count": manifest["record_count"],
                "semantic_sha256": manifest["semantic_sha256"],
            }
            for role, manifest in manifests.items()
        },
        "pairwise_disjoint": True,
        "union_record_count": sum(
            int(manifest["record_count"]) for manifest in manifests.values()
        ),
        "heldout_accessed": False,
    }
    summary["semantic_sha256"] = semantic_sha256(summary)
    return summary


def build_role_manifests(
    cache_dir: Path,
    audit_membership_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the formal 166/229/36 identity-only role manifests."""
    cache_root = cache_dir.expanduser().resolve()
    manifest, _ = _load_cache_payloads(cache_root)
    if manifest.get("record_count") != 431:
        raise CandidateSelectionError("formal role isolation requires exactly 431 cache records")
    source_path = audit_membership_path.expanduser().resolve()
    source = _read_json_object(source_path)
    if source.get("heldout_accessed") is not False or source.get("heldout_count") != 0:
        raise CandidateSelectionError("Audit membership source indicates Heldout access")
    membership = source.get("mapping")
    if not isinstance(membership, list):
        raise CandidateSelectionError("Audit membership mapping is missing")
    identity_membership = [
        {"video_id": item.get("video_id"), "split": item.get("split")}
        for item in membership
        if isinstance(item, Mapping)
    ]
    if len(identity_membership) != len(membership):
        raise CandidateSelectionError("Audit membership contains a non-object entry")
    roles = partition_cache_records(manifest["records"], identity_membership)
    actual_counts = {role: len(records) for role, records in roles.items()}
    if actual_counts != EXPECTED_ROLE_COUNTS:
        raise CandidateSelectionError(
            f"formal role counts differ from 166/229/36: {actual_counts}"
        )
    audit_splits = {
        split: sum(item["split"] == split for item in roles["audit_diagnostic"])
        for split in ("dev", "hard")
    }
    if audit_splits != {"dev": 17, "hard": 19}:
        raise CandidateSelectionError("Audit split counts differ from Dev17/Hard19")

    cache_manifest_sha = _file_sha256(cache_root / "cache_manifest.json")
    audit_source_sha = _file_sha256(source_path)
    manifests = {
        role: create_role_manifest(
            role=role,
            records=records,
            source_cache_global_hash=EXPECTED_CACHE_GLOBAL_HASH,
            source_cache_manifest_sha256=cache_manifest_sha,
            audit_membership_source_sha256=audit_source_sha,
        )
        for role, records in roles.items()
    }
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"role manifest output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for role, role_manifest in manifests.items():
        _write_new_canonical_json(output / _ROLE_FILENAMES[role], role_manifest)
    summary = _create_role_summary(
        manifests,
        source_cache_manifest_sha256=cache_manifest_sha,
        audit_membership_source_sha256=audit_source_sha,
    )
    _write_new_canonical_json(output / ROLE_SUMMARY_FILENAME, summary)
    validate_role_manifest_directory(output, cache_manifest=manifest)
    return summary


def validate_role_manifest_directory(
    directory: Path,
    *,
    cache_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    manifests = {
        role: _read_json_object(root / filename)
        for role, filename in _ROLE_FILENAMES.items()
    }
    for role, manifest in manifests.items():
        validate_role_manifest(manifest, expected_count=EXPECTED_ROLE_COUNTS[role])
        if manifest.get("role") != role:
            raise CandidateSelectionError("role manifest filename/role mismatch")
        if manifest.get("source_cache_global_hash") != cache_manifest.get(
            "global_semantic_sha256"
        ):
            raise CandidateSelectionError("role manifest cache hash mismatch")
    sets = {
        role: {item["video_id"] for item in manifest["records"]}
        for role, manifest in manifests.items()
    }
    if not (
        sets["dev_tune"].isdisjoint(sets["hard_stress"])
        and sets["dev_tune"].isdisjoint(sets["audit_diagnostic"])
        and sets["hard_stress"].isdisjoint(sets["audit_diagnostic"])
    ):
        raise CandidateSelectionError("role manifests overlap")
    cache_ids = {item["video_id"] for item in cache_manifest.get("records", [])}
    if set().union(*sets.values()) != cache_ids:
        raise CandidateSelectionError("role manifest union does not equal the cache")
    summary = _read_json_object(root / ROLE_SUMMARY_FILENAME)
    cache_manifest_hashes = {
        manifest["generation_provenance"]["source_cache_manifest_sha256"]
        for manifest in manifests.values()
    }
    audit_source_hashes = {
        manifest["generation_provenance"]["audit_membership_source_sha256"]
        for manifest in manifests.values()
    }
    if len(cache_manifest_hashes) != 1 or len(audit_source_hashes) != 1:
        raise CandidateSelectionError("role manifest provenance hashes disagree")
    if summary.get("source_cache_manifest_sha256") not in cache_manifest_hashes or summary.get(
        "audit_membership_source_sha256"
    ) not in audit_source_hashes:
        raise CandidateSelectionError("role summary provenance hashes disagree")
    claimed_hash = _require_sha256(summary.get("semantic_sha256"), "role summary hash")
    if claimed_hash != semantic_sha256(_semantic_payload(summary, "semantic_sha256")):
        raise CandidateSelectionError("role summary semantic hash mismatch")
    expected_summary = _create_role_summary(
        manifests,
        source_cache_manifest_sha256=summary.get("source_cache_manifest_sha256"),
        audit_membership_source_sha256=summary.get("audit_membership_source_sha256"),
    )
    if summary != expected_summary:
        raise CandidateSelectionError("role summary is inconsistent with role manifests")
    return {
        "counts": {role: len(values) for role, values in sets.items()},
        "union_record_count": len(cache_ids),
        "pairwise_disjoint": True,
        "semantic_sha256": claimed_hash,
    }


def validate_role_manifests(cache_dir: Path, role_dir: Path) -> dict[str, Any]:
    manifest, _ = _load_cache_payloads(cache_dir)
    return validate_role_manifest_directory(role_dir, cache_manifest=manifest)


def run_selection_to_file(
    cache_dir: Path,
    role_manifest_path: Path,
    selector_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest, records = _load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    selector_config = _read_json_object(selector_config_path.expanduser().resolve())
    result = create_selection_result(manifest, role_manifest, records, selector_config)
    validate_selection_result_payload(result, manifest, role_manifest, records)
    _write_new_canonical_json(output_path, result)
    return result


def run_protocol_selection_to_file(
    cache_dir: Path,
    role_manifest_path: Path,
    protocol_path: Path,
    selector_name: str,
    parameter: float | None,
    output_path: Path,
) -> dict[str, Any]:
    manifest, records = _load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    role_summary = validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    protocol = load_experiment_protocol(protocol_path)
    if protocol.get("role_manifest_summary_hash") != role_summary["semantic_sha256"]:
        raise CandidateSelectionError("protocol role summary hash mismatch")
    role_hashes = protocol.get("role_manifest_hashes")
    if not isinstance(role_hashes, Mapping) or role_hashes.get(
        role_manifest["role"]
    ) != role_manifest["semantic_sha256"]:
        raise CandidateSelectionError("protocol role manifest hash mismatch")
    config = selector_config_from_protocol(protocol, selector_name, parameter)
    result = create_selection_result(manifest, role_manifest, records, config)
    validate_selection_result_payload(result, manifest, role_manifest, records)
    _write_new_canonical_json(output_path, result)
    return result


def validate_selection_result_file(
    cache_dir: Path,
    role_manifest_path: Path,
    selection_result_path: Path,
) -> dict[str, Any]:
    manifest, records = _load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result_path = selection_result_path.expanduser().resolve()
    result = _read_json_object(result_path)
    if result_path.read_bytes() != canonical_json_bytes(result):
        raise CandidateSelectionError("selection result is not canonical JSON")
    return validate_selection_result_payload(result, manifest, role_manifest, records)


def replay_selection_to_jsonl(
    cache_dir: Path,
    role_manifest_path: Path,
    selection_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest, records = _load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result = _read_json_object(selection_result_path.expanduser().resolve())
    validate_selection_result_payload(result, manifest, role_manifest, records)
    replayed = replay_selection_payload(result, records)
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in replayed))
    return {
        "record_count": len(replayed),
        "segment_count": sum(len(item["merged_prediction_segments"]) for item in replayed),
        "selection_result_semantic_hash": result["selection_result_semantic_hash"],
    }


def evaluate_selection_to_file(
    cache_dir: Path,
    role_manifest_path: Path,
    selection_result_path: Path,
    selected_predictions_path: Path,
    frozen_predictions_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    manifest, records = _load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result = _read_json_object(selection_result_path.expanduser().resolve())
    validate_selection_result_payload(result, manifest, role_manifest, records)
    selected = _read_jsonl(selected_predictions_path.expanduser().resolve())
    expected_replay = replay_selection_payload(result, records)
    if selected != expected_replay:
        raise CandidateSelectionError("selected predictions do not match validated replay")
    frozen: list[dict[str, Any]] = []
    frozen_ids: set[str] = set()
    for source_path in frozen_predictions_paths:
        for item in _read_jsonl(source_path.expanduser().resolve()):
            video_id = item.get("video_id")
            if not isinstance(video_id, str) or video_id in frozen_ids:
                raise CandidateSelectionError(
                    "frozen prediction sources contain duplicate or invalid video IDs"
                )
            frozen_ids.add(video_id)
            frozen.append(item)
    if not frozen:
        raise CandidateSelectionError("at least one frozen prediction source is required")
    role_ids = [item["video_id"] for item in role_manifest["records"]]
    frozen_by_id = {item.get("video_id"): item for item in frozen}
    if any(video_id not in frozen_by_id for video_id in role_ids):
        raise CandidateSelectionError("frozen predictions do not cover the role manifest")
    role_frozen = [frozen_by_id[video_id] for video_id in role_ids]
    evaluation = evaluate_replayed_predictions(
        selected,
        role_frozen,
        durations_by_id={video_id: float(records[video_id]["duration_sec"]) for video_id in role_ids},
        role=role_manifest["role"],
        role_manifest_hash=role_manifest["semantic_sha256"],
        selection_result_hash=result["selection_result_semantic_hash"],
        selection_metadata={
            "source_cache_global_hash": manifest["global_semantic_sha256"],
            "selector_name": result["selector_name"],
            "selector_version": result["selector_version"],
            "selector_config": result["selector_config"],
            "selector_config_hash": result["selector_config_hash"],
        },
    )
    _write_new_canonical_json(output_path, evaluation)
    return evaluation


def assess_evaluation_files(
    baseline_evaluation_path: Path,
    candidate_evaluation_path: Path,
    protocol_path: Path,
    phase: str,
    output_path: Path,
) -> dict[str, Any]:
    baseline = _read_json_object(baseline_evaluation_path.expanduser().resolve())
    candidate = _read_json_object(candidate_evaluation_path.expanduser().resolve())
    protocol = load_experiment_protocol(protocol_path)
    comparison = compare_evaluations(baseline, candidate)
    if phase == "dev":
        recall = assess_evaluation_comparison(
            comparison, protocol["dev_recall_guardrail"]
        )
        promotion = assess_evaluation_comparison(
            comparison, protocol["dev_promotion_gate"]
        )
        assessments = {
            "dev_recall_guardrail": recall,
            "dev_promotion_gate": promotion,
            "pass": recall["pass"] and promotion["pass"],
        }
    elif phase == "hard":
        hard = assess_evaluation_comparison(comparison, protocol["hard_stress_gate"])
        assessments = {"hard_stress_gate": hard, "pass": hard["pass"]}
    else:
        raise CandidateSelectionError("assessment phase must be dev or hard")
    report: dict[str, Any] = {
        "assessment_schema_version": "aic.candidate-selection-assessment/v1",
        "phase": phase,
        "protocol_semantic_sha256": protocol["protocol_semantic_sha256"],
        "comparison": comparison,
        "assessments": assessments,
    }
    report["assessment_semantic_sha256"] = semantic_sha256(report)
    _write_new_canonical_json(output_path, report)
    return report


def freeze_dev_parameter_to_file(
    baseline_evaluation_path: Path,
    candidate_evaluation_paths: Iterable[Path],
    protocol_path: Path,
    selector_name: str,
    output_path: Path,
) -> dict[str, Any]:
    baseline = _read_json_object(baseline_evaluation_path.expanduser().resolve())
    candidates = [
        _read_json_object(path.expanduser().resolve())
        for path in candidate_evaluation_paths
    ]
    if not candidates:
        raise CandidateSelectionError("at least one candidate evaluation is required")
    protocol = load_experiment_protocol(protocol_path)
    freeze = choose_dev_evaluation(baseline, candidates, protocol, selector_name)
    _write_new_canonical_json(output_path, freeze)
    return freeze
