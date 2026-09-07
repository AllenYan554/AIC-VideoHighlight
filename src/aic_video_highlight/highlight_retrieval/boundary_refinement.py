"""Stage 4.4 temporal boundary refinement over the frozen candidate cache.

Only ``start_sec``/``end_sec`` of an existing frozen merged candidate may
change.  Candidate count, candidate identity, score, reason, and source fields
are structurally immutable: the replay layer always projects those fields from
the frozen cache, and the refinement result never carries replacements for
them.  BR-0 is a pure identity control; BR-1 asks an injected model callable
for the same event's refined boundaries inside a frozen local context window
and falls back to identity on any parse, schema, or event-identity-guard
failure.  This module never reads references, metrics, audit labels, or
Heldout data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .candidate_cache import canonical_json_bytes, semantic_sha256
from .candidate_selection import (
    CandidateSelectionError,
    EXPECTED_CACHE_GLOBAL_HASH,
    EXPECTED_ROLE_COUNTS,
    _read_json_object,
    _require_sha256,
    _semantic_payload,
    _write_new_canonical_json,
    validate_role_manifest,
    validate_role_manifest_directory,
)
from .temporal_metrics import temporal_iou


BOUNDARY_RESULT_SCHEMA_VERSION = "aic.boundary-refinement-result/v1"
REFINER_VERSION = "aic.temporal-boundary-refinement/v1"
BOUNDARY_PROTOCOL_SCHEMA_VERSION = "aic.stage4-boundary-experiment-protocol/v1"
DRAFT_PROTOCOL_STATUSES = ("DRAFT_FOR_INDEPENDENT_REVIEW",)
FORMAL_PROTOCOL_STATUS = "PREREGISTERED_BEFORE_FORMAL"
BR1_PARAMETERS = ("context_padding_sec", "max_context_duration_sec", "min_parent_temporal_iou")
BR0_DECISION = "IDENTITY"
BR1_DECISIONS = ("REFINE", "IDENTITY_FALLBACK")
RESPONSE_DECISIONS = BR1_DECISIONS
_WINDOW_TOLERANCE_SEC = 1e-9
_IDENTITY_TOLERANCE_SEC = 1e-9
_BOUNDARY_PROMPT_VERSION = "boundary_local_context_v0_draft"
_MAX_BOUNDARY_REASON_CHARS = 400
_MAX_RAW_RESPONSE_CHARS = 2000


class BoundaryRefinementError(ValueError):
    """Raised when a Stage 4.4 isolation, guard, or refinement invariant breaks."""


class BoundaryResponseError(ValueError):
    """Raised when a model response does not satisfy the boundary schema."""


ModelFn = Callable[..., tuple[str, str | None]]


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundaryResponseError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise BoundaryResponseError(f"{field} must be finite")
    return number


def _positive_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number <= 0:
        raise BoundaryRefinementError(f"{field} must be greater than zero")
    return number


def _unit_interval(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if not 0.0 <= number <= 1.0:
        raise BoundaryRefinementError(f"{field} must be in [0, 1]")
    return number


def _validate_refiner_config(config: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    if set(config) != {"refiner_name", "refiner_version", "parameters"}:
        raise BoundaryRefinementError("refiner config has unexpected or missing fields")
    name = config.get("refiner_name")
    if name not in {"BR-0", "BR-1"}:
        raise BoundaryRefinementError("unknown refiner_name")
    if config.get("refiner_version") != REFINER_VERSION:
        raise BoundaryRefinementError("unsupported refiner_version")
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise BoundaryRefinementError("refiner parameters must be an object")
    if name == "BR-0":
        if parameters:
            raise BoundaryRefinementError("BR-0 does not accept parameters")
        return str(name), {}
    if set(parameters) != set(BR1_PARAMETERS):
        raise BoundaryRefinementError("BR-1 requires exactly the frozen parameter set")
    values = {
        "context_padding_sec": _positive_number(
            parameters["context_padding_sec"], "context_padding_sec"
        ),
        "max_context_duration_sec": _positive_number(
            parameters["max_context_duration_sec"], "max_context_duration_sec"
        ),
        "min_parent_temporal_iou": _unit_interval(
            parameters["min_parent_temporal_iou"], "min_parent_temporal_iou"
        ),
    }
    return str(name), values


def refiner_config_from_protocol(
    protocol: Mapping[str, Any], refiner_name: str
) -> dict[str, Any]:
    refiners = protocol.get("refiners")
    if not isinstance(refiners, Mapping) or refiner_name not in refiners:
        raise BoundaryRefinementError("refiner is absent from protocol")
    definition = refiners[refiner_name]
    if not isinstance(definition, Mapping) or definition.get("refiner_version") != REFINER_VERSION:
        raise BoundaryRefinementError("refiner protocol version mismatch")
    config = {
        "refiner_name": refiner_name,
        "refiner_version": REFINER_VERSION,
        "parameters": _canonical_copy(definition.get("parameters")),
    }
    _validate_refiner_config(config)
    return config


def load_boundary_protocol(
    path: Path, *, allow_draft: bool = True
) -> dict[str, Any]:
    protocol = _read_json_object(path.expanduser().resolve())
    if protocol.get("protocol_schema_version") != BOUNDARY_PROTOCOL_SCHEMA_VERSION:
        raise BoundaryRefinementError("unsupported Stage 4.4 protocol schema")
    claimed = _require_sha256(
        protocol.get("protocol_semantic_sha256"), "protocol semantic hash"
    )
    if claimed != semantic_sha256(_semantic_payload(protocol, "protocol_semantic_sha256")):
        raise BoundaryRefinementError("protocol semantic hash mismatch")
    allowed = (FORMAL_PROTOCOL_STATUS,) + DRAFT_PROTOCOL_STATUSES if allow_draft else (
        FORMAL_PROTOCOL_STATUS,
    )
    if protocol.get("protocol_status") not in allowed:
        raise BoundaryRefinementError("protocol status is not accepted for this run")
    if protocol.get("source_cache_global_hash") != EXPECTED_CACHE_GLOBAL_HASH:
        raise BoundaryRefinementError("protocol references the wrong Frozen Cache")
    refiners = protocol.get("refiners")
    if not isinstance(refiners, Mapping) or set(refiners) != {"BR-0", "BR-1"}:
        raise BoundaryRefinementError("protocol refiner registry is incomplete")
    return protocol


def build_boundary_prompt(
    *,
    context_duration_sec: float,
    candidate_start_sec: float,
    candidate_end_sec: float,
    candidate_reason: str,
) -> str:
    """Build the frozen BR-1 prompt; the model sees only the local clip."""
    if context_duration_sec <= 0:
        raise BoundaryRefinementError("context_duration_sec must be greater than zero")
    if not 0.0 <= candidate_start_sec < candidate_end_sec <= context_duration_sec:
        raise BoundaryRefinementError("candidate interval must lie inside the local clip")
    reason = str(candidate_reason).strip()
    if not reason:
        reason = "（无附加说明）"
    return f"""你是视频高光系统中的时间边界精修模块。你不是在重新寻找高光。
系统已经给你一个候选事件区间，你的职责只有一个：判断同一事件的有效起止边界，并做局部修正。

给你观看的是一个局部视频片段，时长 {context_duration_sec:.3f} 秒。
候选事件区间（相对本片段开头的秒数）：start_sec = {candidate_start_sec:.3f}，end_sec = {candidate_end_sec:.3f}。
该候选的检索理由：{reason}

规则：
1. 保持事件身份：修正后的区间必须仍然是这同一个事件，不许移到旁边其他事件上。
2. 去掉明显的前置铺垫（setup）和事件结束后的无关拖尾（tail）。
3. 保留核心动作、事件高潮或结果，以及理解该事件所必需的最少上下文。
4. 不要为了追求更短而过度裁剪；宁可略微放宽，也不要切断事件核心。
5. 如果边界轻微不确定，或你无法确认事件边界，返回原边界并把 decision 设为 IDENTITY_FALLBACK。
6. 时间数值都是相对本片段开头的秒数，范围 0 到 {context_duration_sec:.3f}，不要输出帧号或原视频全局时间。

仅输出一个 JSON 对象，不要输出 Markdown 或解释，格式为：
{{"refined_start_sec": 4.2, "refined_end_sec": 8.6, "decision": "REFINE", "confidence": 0.8, "boundary_reason": "简短理由"}}

decision 只允许 "REFINE"（已按同一事件修正边界）或 "IDENTITY_FALLBACK"（不确定，保持原边界）。
不允许出现 DROP、CREATE、NEW_EVENT 或第二个事件。

Prompt 版本：{_BOUNDARY_PROMPT_VERSION}
"""


def compute_local_context_window(
    *,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    context_padding_sec: float,
    max_context_duration_sec: float,
) -> dict[str, float]:
    """Deterministic clamped local window that always contains the candidate."""
    start = _finite_number(start_sec, "start_sec")
    end = _finite_number(end_sec, "end_sec")
    duration = _finite_number(duration_sec, "duration_sec")
    padding = _positive_number(context_padding_sec, "context_padding_sec")
    max_context = _positive_number(max_context_duration_sec, "max_context_duration_sec")
    if duration <= 0:
        raise BoundaryRefinementError("duration_sec must be greater than zero")
    if not 0.0 <= start < end <= duration:
        raise BoundaryRefinementError("candidate interval must satisfy 0 <= start < end <= duration")
    window_start = max(0.0, start - padding)
    window_end = min(duration, end + padding)
    excess = (window_end - window_start) - max_context
    if excess > 0:
        available_before = start - window_start
        available_after = window_end - end
        if available_before >= available_after:
            trim_before = min(excess, available_before)
            excess -= trim_before
            trim_after = min(excess, available_after)
        else:
            trim_after = min(excess, available_after)
            excess -= trim_after
            trim_before = min(excess, available_before)
        window_start += trim_before
        window_end -= trim_after
        if not (
            window_start <= start < end <= window_end
        ):  # pragma: no cover - arithmetic safeguard
            raise BoundaryRefinementError("window trimming broke candidate containment")
    return {"start_sec": window_start, "end_sec": window_end}


def assess_event_identity(
    *,
    original_start_sec: float,
    original_end_sec: float,
    refined_start_sec: float,
    refined_end_sec: float,
    window: Mapping[str, float],
    duration_sec: float,
    min_parent_temporal_iou: float,
) -> dict[str, Any]:
    """Re-checkable event identity guard for one refinement proposal."""
    refined_start = _finite_number(refined_start_sec, "refined_start_sec")
    refined_end = _finite_number(refined_end_sec, "refined_end_sec")
    interval_valid = (
        0.0 <= refined_start
        and refined_start < refined_end
        and refined_end <= float(duration_sec) + _WINDOW_TOLERANCE_SEC
    )
    within_window = (
        refined_start >= float(window["start_sec"]) - _WINDOW_TOLERANCE_SEC
        and refined_end <= float(window["end_sec"]) + _WINDOW_TOLERANCE_SEC
    )
    if interval_valid:
        parent_iou = temporal_iou(
            (float(original_start_sec), float(original_end_sec)),
            (refined_start, refined_end),
        )
    else:
        parent_iou = 0.0
    parent_ok = interval_valid and parent_iou >= float(min_parent_temporal_iou)
    checks = {
        "interval_valid": {
            "pass": interval_valid,
            "observed": [refined_start, refined_end],
        },
        "within_local_window": {
            "pass": within_window,
            "observed": [refined_start, refined_end],
            "window": [float(window["start_sec"]), float(window["end_sec"])],
        },
        "parent_tiou": {
            "pass": parent_ok,
            "observed": parent_iou,
            "threshold": float(min_parent_temporal_iou),
        },
    }
    failed = sorted(name for name, item in checks.items() if not item["pass"])
    return {"pass": not failed, "failed_checks": failed, "checks": checks}


def parse_boundary_response(
    text: str, *, context_start_sec: float, context_end_sec: float
) -> dict[str, Any]:
    """Parse the strict BR-1 response schema; clip-local times map to absolute."""
    context_start = _finite_number(context_start_sec, "context_start_sec")
    context_end = _finite_number(context_end_sec, "context_end_sec")
    try:
        payload = json.loads(str(text))
    except json.JSONDecodeError as exc:
        raise BoundaryResponseError("response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise BoundaryResponseError("response must be a JSON object")
    required = {
        "refined_start_sec",
        "refined_end_sec",
        "decision",
        "confidence",
        "boundary_reason",
    }
    if set(payload) != required:
        raise BoundaryResponseError("response schema has unexpected or missing fields")
    local_start = _finite_number(payload["refined_start_sec"], "refined_start_sec")
    local_end = _finite_number(payload["refined_end_sec"], "refined_end_sec")
    clip_length = context_end - context_start
    if local_start < 0 or local_end > clip_length + _WINDOW_TOLERANCE_SEC:
        raise BoundaryResponseError("refined times must lie inside the local clip")
    if local_end <= local_start:
        raise BoundaryResponseError("refined_end_sec must be greater than refined_start_sec")
    decision = payload["decision"]
    if decision not in RESPONSE_DECISIONS:
        raise BoundaryResponseError("decision must be REFINE or IDENTITY_FALLBACK")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise BoundaryResponseError("confidence must be in [0, 1]")
    reason = payload["boundary_reason"]
    if not isinstance(reason, str):
        raise BoundaryResponseError("boundary_reason must be a string")
    if len(reason) > _MAX_BOUNDARY_REASON_CHARS:
        raise BoundaryResponseError("boundary_reason exceeds the length limit")
    return {
        "refined_start_sec": context_start + local_start,
        "refined_end_sec": context_start + local_end,
        "decision": str(decision),
        "confidence": confidence,
        "boundary_reason": reason,
    }


def _canonical_copy(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


_FORBIDDEN_KEY_FRAGMENTS = (
    "reference",
    "metric",
    "precision",
    "recall",
    "temporal_iou",
    "f1",
    "audit",
    "adjudicat",
    "reviewer",
    "human",
    "label",
)


def _check_forbidden_keys(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise BoundaryRefinementError(f"forbidden boundary field at {path}.{key}")
            _check_forbidden_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _check_forbidden_keys(value, f"{path}[{index}]")


def _check_result_leakage(result: Mapping[str, Any]) -> None:
    """Leakage scan over generated content; the frozen refiner_config echo is
    covered by its own protocol hash and may contain metric-like parameter
    names such as ``min_parent_temporal_iou``."""
    for key, value in result.items():
        if key == "refiner_config":
            continue
        normalized = str(key).lower()
        if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
            raise BoundaryRefinementError(f"forbidden boundary field at $.{key}")
        _check_forbidden_keys(value, f"$.{key}")


def _identity_decision(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "merged_candidate_id": str(candidate["merged_candidate_id"]),
        "decision": BR0_DECISION,
        "decision_rule": "br0.identity",
        "original_start_sec": float(candidate["start_sec"]),
        "original_end_sec": float(candidate["end_sec"]),
        "refined_start_sec": float(candidate["start_sec"]),
        "refined_end_sec": float(candidate["end_sec"]),
        "confidence": 1.0,
        "boundary_reason": "",
        "local_context_window": None,
        "identity_guard": None,
        "model_response": None,
    }


def refine_candidates(
    candidates: Iterable[Mapping[str, Any]],
    refiner_config: Mapping[str, Any],
    *,
    duration_sec: float,
    video_id: str,
    model_fn: ModelFn | None = None,
) -> list[dict[str, Any]]:
    """Return one deterministic boundary decision per input candidate."""
    name, parameters = _validate_refiner_config(refiner_config)
    items = list(candidates)
    decisions: list[dict[str, Any]] = []
    for candidate in items:
        if not isinstance(candidate.get("merged_candidate_id"), str):
            raise BoundaryRefinementError("candidate is missing merged_candidate_id")
        original_start = float(candidate["start_sec"])
        original_end = float(candidate["end_sec"])
        if not 0.0 <= original_start < original_end <= float(duration_sec):
            raise BoundaryRefinementError(
                f"candidate interval violates duration for {candidate['merged_candidate_id']}"
            )
        if name == "BR-0":
            decisions.append(_identity_decision(candidate))
            continue
        if model_fn is None:
            raise BoundaryRefinementError("BR-1 requires an injected model callable")
        candidate_id = str(candidate["merged_candidate_id"])
        window = compute_local_context_window(
            start_sec=original_start,
            end_sec=original_end,
            duration_sec=float(duration_sec),
            context_padding_sec=parameters["context_padding_sec"],
            max_context_duration_sec=parameters["max_context_duration_sec"],
        )
        clip_length = float(window["end_sec"]) - float(window["start_sec"])
        local_start = original_start - float(window["start_sec"])
        local_end = original_end - float(window["start_sec"])
        prompt = build_boundary_prompt(
            context_duration_sec=clip_length,
            candidate_start_sec=local_start,
            candidate_end_sec=local_end,
            candidate_reason=str(candidate.get("reason", "")),
        )
        raw_text, finish_reason = model_fn(
            prompt,
            video_id=video_id,
            candidate_id=candidate_id,
            window=window,
        )
        raw_record = {
            "text_sha256": semantic_sha256(str(raw_text)),
            "text_preview": str(raw_text)[:_MAX_RAW_RESPONSE_CHARS],
            "finish_reason": None if finish_reason is None else str(finish_reason),
        }
        base = {
            "merged_candidate_id": candidate_id,
            "original_start_sec": original_start,
            "original_end_sec": original_end,
            "local_context_window": window,
            "model_response": raw_record,
        }
        try:
            parsed = parse_boundary_response(
                raw_text,
                context_start_sec=float(window["start_sec"]),
                context_end_sec=float(window["end_sec"]),
            )
        except BoundaryResponseError as exc:
            decisions.append(
                {
                    **base,
                    "decision": "IDENTITY_FALLBACK",
                    "decision_rule": "br1.fallback_parse_error",
                    "refined_start_sec": original_start,
                    "refined_end_sec": original_end,
                    "confidence": 0.0,
                    "boundary_reason": f"parse error: {exc}",
                    "identity_guard": None,
                }
            )
            continue
        if parsed["decision"] == "IDENTITY_FALLBACK":
            decisions.append(
                {
                    **base,
                    "decision": "IDENTITY_FALLBACK",
                    "decision_rule": "br1.model_identity_fallback",
                    "refined_start_sec": original_start,
                    "refined_end_sec": original_end,
                    "confidence": float(parsed["confidence"]),
                    "boundary_reason": parsed["boundary_reason"],
                    "identity_guard": None,
                }
            )
            continue
        guard = assess_event_identity(
            original_start_sec=original_start,
            original_end_sec=original_end,
            refined_start_sec=parsed["refined_start_sec"],
            refined_end_sec=parsed["refined_end_sec"],
            window=window,
            duration_sec=float(duration_sec),
            min_parent_temporal_iou=parameters["min_parent_temporal_iou"],
        )
        if not guard["pass"]:
            decisions.append(
                {
                    **base,
                    "decision": "IDENTITY_FALLBACK",
                    "decision_rule": "br1.fallback_event_identity_guard",
                    "refined_start_sec": original_start,
                    "refined_end_sec": original_end,
                    "confidence": float(parsed["confidence"]),
                    "boundary_reason": parsed["boundary_reason"],
                    "identity_guard": guard,
                }
            )
            continue
        decisions.append(
            {
                **base,
                "decision": "REFINE",
                "decision_rule": "br1.model_refine",
                "refined_start_sec": float(parsed["refined_start_sec"]),
                "refined_end_sec": float(parsed["refined_end_sec"]),
                "confidence": float(parsed["confidence"]),
                "boundary_reason": parsed["boundary_reason"],
                "identity_guard": guard,
            }
        )
    return decisions


def create_boundary_refinement_result(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    refiner_config: Mapping[str, Any],
    *,
    model_fn: ModelFn | None = None,
) -> dict[str, Any]:
    """Create a deterministic boundary refinement result without references."""
    validate_role_manifest(
        role_manifest, expected_count=len(role_manifest.get("records", []))
    )
    cache_hash = cache_manifest.get("global_semantic_sha256")
    if cache_hash != role_manifest.get("source_cache_global_hash"):
        raise BoundaryRefinementError("role manifest does not match source cache")
    name, _ = _validate_refiner_config(refiner_config)
    config_payload = _canonical_copy(refiner_config)
    config_hash = semantic_sha256(config_payload)
    records: list[dict[str, Any]] = []
    candidate_total = 0
    decision_counts = {BR0_DECISION: 0, "REFINE": 0, "IDENTITY_FALLBACK": 0}
    for identity in role_manifest["records"]:
        video_id = identity["video_id"]
        record = cache_records_by_id.get(video_id)
        if record is None:
            raise BoundaryRefinementError(f"role video is missing from cache: {video_id}")
        if record.get("split") != identity["split"]:
            raise BoundaryRefinementError(f"role/cache split mismatch for {video_id}")
        candidates = record.get("merged_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise BoundaryRefinementError(f"cache merged_candidates missing for {video_id}")
        refinements = refine_candidates(
            candidates,
            refiner_config,
            duration_sec=float(record["duration_sec"]),
            video_id=video_id,
            model_fn=model_fn,
        )
        for candidate, refinement in zip(candidates, refinements, strict=True):
            refinement["parent_candidate_semantic_sha256"] = semantic_sha256(
                {"video_id": video_id, **candidate}
            )
        decision_counts = {
            key: value + sum(item["decision"] == key for item in refinements)
            for key, value in decision_counts.items()
        }
        candidate_total += len(candidates)
        video_record = {
            "video_id": video_id,
            "split": identity["split"],
            "source_record_semantic_sha256": _require_sha256(
                record.get("semantic_sha256"), f"{video_id} source record hash"
            ),
            "candidate_refinements": refinements,
        }
        video_record["video_refinement_semantic_sha256"] = semantic_sha256(video_record)
        records.append(video_record)
    result: dict[str, Any] = {
        "boundary_result_schema_version": BOUNDARY_RESULT_SCHEMA_VERSION,
        "source_cache_global_hash": cache_hash,
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "refiner_name": name,
        "refiner_version": REFINER_VERSION,
        "refiner_config": config_payload,
        "refiner_config_hash": config_hash,
        "record_count": len(records),
        "input_candidate_count": candidate_total,
        "decision_counts": decision_counts,
        "records": records,
    }
    _check_result_leakage(result)
    result["boundary_refinement_semantic_hash"] = semantic_sha256(result)
    return result


def validate_boundary_refinement_payload(
    result: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate schema, hashes, candidate identity, and every guard decision."""
    _check_result_leakage(result)
    if result.get("boundary_result_schema_version") != BOUNDARY_RESULT_SCHEMA_VERSION:
        raise BoundaryRefinementError("unsupported boundary refinement result schema")
    claimed = _require_sha256(
        result.get("boundary_refinement_semantic_hash"), "boundary result hash"
    )
    if claimed != semantic_sha256(
        _semantic_payload(result, "boundary_refinement_semantic_hash")
    ):
        raise BoundaryRefinementError("boundary refinement semantic hash mismatch")
    if result.get("source_cache_global_hash") != cache_manifest.get(
        "global_semantic_sha256"
    ):
        raise BoundaryRefinementError("boundary result cache hash mismatch")
    if result.get("role") != role_manifest.get("role") or result.get(
        "role_manifest_hash"
    ) != role_manifest.get("semantic_sha256"):
        raise BoundaryRefinementError("boundary result role mismatch")
    name, parameters = _validate_refiner_config(result.get("refiner_config"))
    if result.get("refiner_config_hash") != semantic_sha256(result.get("refiner_config")):
        raise BoundaryRefinementError("boundary result config hash mismatch")
    if result.get("refiner_version") != REFINER_VERSION:
        raise BoundaryRefinementError("boundary result refiner version mismatch")
    result_records = result.get("records")
    if not isinstance(result_records, list) or result.get("record_count") != len(
        result_records
    ):
        raise BoundaryRefinementError("boundary result record_count mismatch")
    if result.get("role") in EXPECTED_ROLE_COUNTS and result_records and len(
        result_records
    ) != len(role_manifest.get("records", [])):
        raise BoundaryRefinementError("boundary result role coverage mismatch")
    decision_counts = {BR0_DECISION: 0, "REFINE": 0, "IDENTITY_FALLBACK": 0}
    candidate_total = 0
    for result_record, role_identity in zip(
        result_records, role_manifest.get("records", []), strict=True
    ):
        video_id = result_record.get("video_id")
        if video_id != role_identity.get("video_id"):
            raise BoundaryRefinementError("boundary result video order mismatch")
        record = cache_records_by_id.get(str(video_id))
        if record is None:
            raise BoundaryRefinementError(f"boundary video missing from cache: {video_id}")
        if not (
            record.get("split") == result_record.get("split") == role_identity.get("split")
        ):
            raise BoundaryRefinementError(f"boundary split mismatch for {video_id}")
        candidates = record.get("merged_candidates")
        refinements = result_record.get("candidate_refinements")
        if not isinstance(candidates, list) or not isinstance(refinements, list):
            raise BoundaryRefinementError(f"boundary refinements missing for {video_id}")
        if len(candidates) != len(refinements):
            raise BoundaryRefinementError(f"candidate count changed for {video_id}")
        duration = float(record["duration_sec"])
        claimed_video_hash = _require_sha256(
            result_record.get("video_refinement_semantic_sha256"),
            f"{video_id} video hash",
        )
        semantic_video = {
            key: value
            for key, value in result_record.items()
            if key != "video_refinement_semantic_sha256"
        }
        if claimed_video_hash != semantic_sha256(semantic_video):
            raise BoundaryRefinementError(f"video refinement hash mismatch for {video_id}")
        if _require_sha256(
            result_record.get("source_record_semantic_sha256"), "source record hash"
        ) != record.get("semantic_sha256"):
            raise BoundaryRefinementError(f"cache record hash mismatch for {video_id}")
        for candidate, refinement in zip(candidates, refinements, strict=True):
            candidate_id = str(candidate["merged_candidate_id"])
            if not isinstance(refinement, Mapping):
                raise BoundaryRefinementError(f"invalid refinement for {video_id}")
            if str(refinement.get("merged_candidate_id")) != candidate_id:
                raise BoundaryRefinementError(f"candidate identity changed for {video_id}")
            expected_parent_hash = semantic_sha256({"video_id": str(video_id), **candidate})
            if refinement.get("parent_candidate_semantic_sha256") != expected_parent_hash:
                raise BoundaryRefinementError(f"parent candidate hash mismatch for {video_id}")
            original_start = float(candidate["start_sec"])
            original_end = float(candidate["end_sec"])
            if float(refinement.get("original_start_sec", -1.0)) != original_start or float(
                refinement.get("original_end_sec", -1.0)
            ) != original_end:
                raise BoundaryRefinementError(
                    f"original interval mismatch for {video_id}/{candidate_id}"
                )
            decision = refinement.get("decision")
            if decision not in decision_counts:
                raise BoundaryRefinementError(f"unknown boundary decision for {video_id}")
            decision_counts[decision] += 1
            refined_start = _finite_number(refinement.get("refined_start_sec"), "refined_start")
            refined_end = _finite_number(refinement.get("refined_end_sec"), "refined_end")
            if decision == BR0_DECISION or decision == "IDENTITY_FALLBACK":
                if (
                    abs(refined_start - original_start) > _IDENTITY_TOLERANCE_SEC
                    or abs(refined_end - original_end) > _IDENTITY_TOLERANCE_SEC
                ):
                    raise BoundaryRefinementError(
                        f"{decision} must preserve the original interval for {video_id}/{candidate_id}"
                    )
                if decision == BR0_DECISION and refinement.get("decision_rule") != "br0.identity":
                    raise BoundaryRefinementError(f"BR-0 rule mismatch for {video_id}")
            else:
                window = compute_local_context_window(
                    start_sec=original_start,
                    end_sec=original_end,
                    duration_sec=duration,
                    context_padding_sec=parameters["context_padding_sec"],
                    max_context_duration_sec=parameters["max_context_duration_sec"],
                )
                emitted_window = refinement.get("local_context_window")
                if not isinstance(emitted_window, Mapping) or any(
                    abs(float(emitted_window[key]) - float(window[key])) > _WINDOW_TOLERANCE_SEC
                    for key in ("start_sec", "end_sec")
                ):
                    raise BoundaryRefinementError(
                        f"local window mismatch for {video_id}/{candidate_id}"
                    )
                guard = assess_event_identity(
                    original_start_sec=original_start,
                    original_end_sec=original_end,
                    refined_start_sec=refined_start,
                    refined_end_sec=refined_end,
                    window=window,
                    duration_sec=duration,
                    min_parent_temporal_iou=parameters["min_parent_temporal_iou"],
                )
                if not guard["pass"]:
                    raise BoundaryRefinementError(
                        f"REFINE violates the event identity guard for {video_id}/{candidate_id}"
                    )
                if refinement.get("decision_rule") != "br1.model_refine":
                    raise BoundaryRefinementError(f"REFINE rule mismatch for {video_id}")
            if decision == "IDENTITY_FALLBACK":
                rule = str(refinement.get("decision_rule", ""))
                if not rule.startswith("br1.fallback"):
                    raise BoundaryRefinementError(
                        f"fallback rule mismatch for {video_id}/{candidate_id}"
                    )
            if not isinstance(refinement.get("boundary_reason"), str):
                raise BoundaryRefinementError(f"boundary_reason missing for {video_id}")
            confidence = _finite_number(refinement.get("confidence"), "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise BoundaryRefinementError(f"confidence out of range for {video_id}")
            candidate_total += 1
    if result.get("input_candidate_count") != candidate_total:
        raise BoundaryRefinementError("boundary result candidate count mismatch")
    if result.get("decision_counts") != decision_counts:
        raise BoundaryRefinementError("boundary decision counts mismatch")
    return {
        "record_count": len(result_records),
        "input_candidate_count": candidate_total,
        "decision_counts": decision_counts,
        "boundary_refinement_semantic_hash": claimed,
    }


def replay_refined_predictions(
    result: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project refined predictions; score/reason/source come from the frozen cache."""
    replayed: list[dict[str, Any]] = []
    result_records = result.get("records")
    if not isinstance(result_records, list):
        raise BoundaryRefinementError("boundary result records are missing")
    for item in result_records:
        video_id = str(item.get("video_id"))
        record = cache_records_by_id.get(video_id)
        if record is None:
            raise BoundaryRefinementError(f"boundary video is missing from cache: {video_id}")
        candidates = record.get("merged_candidates")
        refinements = item.get("candidate_refinements")
        if not isinstance(candidates, list) or not isinstance(refinements, list):
            raise BoundaryRefinementError(f"boundary replay inputs missing for {video_id}")
        if len(candidates) != len(refinements):
            raise BoundaryRefinementError(f"candidate count changed for {video_id}")
        segments = [
            {
                "start_sec": float(refinement["refined_start_sec"]),
                "end_sec": float(refinement["refined_end_sec"]),
                "score": float(candidate["score"]),
                "reason": str(candidate.get("reason", "")),
                "source_chunk": candidate.get("source_chunk"),
            }
            for candidate, refinement in zip(candidates, refinements, strict=True)
        ]
        replayed.append(
            {
                "video_id": video_id,
                "split": item.get("split"),
                "merged_prediction_segments": segments,
            }
        )
    return replayed


def load_cache_payloads(cache_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the frozen cache and load its manifest and records."""
    from .candidate_selection import _load_cache_payloads

    manifest, records = _load_cache_payloads(cache_dir)
    if manifest.get("global_semantic_sha256") != EXPECTED_CACHE_GLOBAL_HASH:
        raise BoundaryRefinementError("cache is not the Stage 4.2 frozen cache")
    return manifest, records


def run_refinement_to_file(
    cache_dir: Path,
    role_manifest_path: Path,
    protocol_path: Path,
    refiner_name: str,
    output_path: Path,
    *,
    model_fn: ModelFn | None = None,
    allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    """Protocol-bound refinement chain: load → refine → validate → write."""
    manifest, records = load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    role_summary = validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    protocol = load_boundary_protocol(protocol_path, allow_draft=allow_draft_protocol)
    if protocol.get("role_manifest_summary_hash") != role_summary["semantic_sha256"]:
        raise BoundaryRefinementError("protocol role summary hash mismatch")
    role_hashes = protocol.get("role_manifest_hashes")
    if not isinstance(role_hashes, Mapping) or role_hashes.get(
        role_manifest["role"]
    ) != role_manifest["semantic_sha256"]:
        raise BoundaryRefinementError("protocol role manifest hash mismatch")
    config = refiner_config_from_protocol(protocol, refiner_name)
    result = create_boundary_refinement_result(
        manifest, role_manifest, records, config, model_fn=model_fn
    )
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)
    _write_new_canonical_json(output_path, result)
    return result


def validate_refinement_file(
    cache_dir: Path,
    role_manifest_path: Path,
    refinement_result_path: Path,
) -> dict[str, Any]:
    manifest, records = load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result_path = refinement_result_path.expanduser().resolve()
    result = _read_json_object(result_path)
    if result_path.read_bytes() != canonical_json_bytes(result):
        raise BoundaryRefinementError("boundary refinement result is not canonical JSON")
    return validate_boundary_refinement_payload(result, manifest, role_manifest, records)


def replay_refinement_to_jsonl(
    cache_dir: Path,
    role_manifest_path: Path,
    refinement_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest, records = load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result = _read_json_object(refinement_result_path.expanduser().resolve())
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)
    replayed = replay_refined_predictions(result, records)
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in replayed))
    return {
        "record_count": len(replayed),
        "segment_count": sum(len(item["merged_prediction_segments"]) for item in replayed),
        "boundary_refinement_semantic_hash": result["boundary_refinement_semantic_hash"],
    }


def evaluate_refinement_to_file(
    cache_dir: Path,
    role_manifest_path: Path,
    refinement_result_path: Path,
    refined_predictions_path: Path,
    frozen_predictions_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate refined predictions with the unchanged frozen metric function."""
    from .candidate_selection import evaluate_replayed_predictions

    manifest, records = load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    result = _read_json_object(refinement_result_path.expanduser().resolve())
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)
    refined = _read_jsonl(refined_predictions_path.expanduser().resolve())
    expected_replay = replay_refined_predictions(result, records)
    if refined != expected_replay:
        raise BoundaryRefinementError("refined predictions do not match validated replay")
    frozen: list[dict[str, Any]] = []
    frozen_ids: set[str] = set()
    for source_path in frozen_predictions_paths:
        for item in _read_jsonl(source_path.expanduser().resolve()):
            video_id = item.get("video_id")
            if not isinstance(video_id, str) or video_id in frozen_ids:
                raise BoundaryRefinementError(
                    "frozen prediction sources contain duplicate or invalid video IDs"
                )
            frozen_ids.add(video_id)
            frozen.append(item)
    if not frozen:
        raise BoundaryRefinementError("at least one frozen prediction source is required")
    role_ids = [item["video_id"] for item in role_manifest["records"]]
    frozen_by_id = {item.get("video_id"): item for item in frozen}
    if any(video_id not in frozen_by_id for video_id in role_ids):
        raise BoundaryRefinementError("frozen predictions do not cover the role manifest")
    role_frozen = [frozen_by_id[video_id] for video_id in role_ids]
    evaluation = evaluate_replayed_predictions(
        refined,
        role_frozen,
        durations_by_id={
            video_id: float(records[video_id]["duration_sec"]) for video_id in role_ids
        },
        role=role_manifest["role"],
        role_manifest_hash=role_manifest["semantic_sha256"],
        selection_result_hash=result["boundary_refinement_semantic_hash"],
        selection_metadata={
            "source_cache_global_hash": manifest["global_semantic_sha256"],
            "refiner_name": result["refiner_name"],
            "refiner_version": result["refiner_version"],
            "refiner_config": result["refiner_config"],
            "refiner_config_hash": result["refiner_config_hash"],
        },
    )
    _write_new_canonical_json(output_path, evaluation)
    return evaluation


def assess_refinement_files(
    baseline_evaluation_path: Path,
    candidate_evaluation_path: Path,
    protocol_path: Path,
    phase: str,
    output_path: Path,
    *,
    allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    from .candidate_selection import (
        assess_evaluation_comparison,
        compare_evaluations,
    )

    baseline = _read_json_object(baseline_evaluation_path.expanduser().resolve())
    candidate = _read_json_object(candidate_evaluation_path.expanduser().resolve())
    protocol = load_boundary_protocol(protocol_path, allow_draft=allow_draft_protocol)
    comparison = compare_evaluations(baseline, candidate)
    if phase == "dev":
        recall = assess_evaluation_comparison(comparison, protocol["dev_recall_guardrail"])
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
        raise BoundaryRefinementError("assessment phase must be dev or hard")
    report: dict[str, Any] = {
        "assessment_schema_version": "aic.boundary-refinement-assessment/v1",
        "phase": phase,
        "protocol_semantic_sha256": protocol["protocol_semantic_sha256"],
        "comparison": comparison,
        "assessments": assessments,
    }
    report["assessment_semantic_sha256"] = semantic_sha256(report)
    _write_new_canonical_json(output_path, report)
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BoundaryRefinementError(f"cannot read JSONL: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BoundaryRefinementError(f"invalid JSONL line {index}: {path}") from exc
        if not isinstance(item, dict):
            raise BoundaryRefinementError(f"JSONL line {index} is not an object: {path}")
        rows.append(item)
    return rows
