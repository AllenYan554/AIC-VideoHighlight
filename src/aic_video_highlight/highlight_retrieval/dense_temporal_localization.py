"""Stage 4.6 dense temporal localization over frozen merged candidates.

DTL-0 is an exact identity control. DTL-1 asks an injected model callable to
classify fixed temporal bins as CORE, CONTEXT, or OUTSIDE; only deterministic
program code converts those labels to an interval. This module never reads
references, metrics, Audit adjudication, or Heldout data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .candidate_cache import canonical_json_bytes, semantic_sha256
from .candidate_selection import (
    EXPECTED_CACHE_GLOBAL_HASH,
    _read_json_object,
    _semantic_payload,
    _write_new_canonical_json,
    validate_role_manifest,
    validate_role_manifest_directory,
)
from .video_clip import COORDINATE_CONTRACT


DTL_VERSION = "aic.dense-temporal-localization/v1"
RESULT_SCHEMA_VERSION = "aic.dense-temporal-localization-result/v1"
LABELS = ("CORE", "CONTEXT", "OUTSIDE")
PROMPT_ID = "dense_temporal_bins_v1_1"
DECODER_VERSION = "dtl.component-anchor/v1"
_CLIP_FIELDS = {
    "coordinate_contract",
    "extraction_backend",
    "requested_source_start_sec",
    "requested_source_end_sec",
    "actual_source_origin_sec",
    "clip_duration_sec",
    "actual_source_frame_index",
    "source_fps",
    "first_source_frame_checksum",
}
_RULES = (
    "dtl0.identity",
    "dtl1.dense_refine",
    "dtl1.dense_identity",
    "dtl1.fallback_no_core",
    "dtl1.fallback_no_parent_overlap",
    "dtl1.fallback_parse_error",
    "dtl1.fallback_empty_response",
    "dtl1.fallback_transport_error",
)

ModelFn = Callable[..., tuple[str, str | None] | str]
PrepareContextFn = Callable[..., Mapping[str, Any]]


class DenseTemporalError(ValueError):
    """Raised when a Stage 4.6 invariant is violated."""


class DenseTemporalResponseError(ValueError):
    """Raised when a model response violates the dense-label schema."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DenseTemporalError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DenseTemporalError(f"{field} must be finite")
    return result


def build_temporal_bins(
    clip_timing: Mapping[str, Any],
    *,
    bin_size_sec: float,
    max_bin_count: int,
) -> list[dict[str, Any]]:
    """Build deterministic fixed-width local bins using the verified clip origin."""
    if clip_timing.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise DenseTemporalError("clip coordinate contract mismatch")
    origin = _finite_number(
        clip_timing.get("actual_source_origin_sec"), "actual_source_origin_sec"
    )
    duration = _finite_number(clip_timing.get("clip_duration_sec"), "clip_duration_sec")
    size = _finite_number(bin_size_sec, "bin_size_sec")
    if origin < 0 or duration <= 0 or size <= 0:
        raise DenseTemporalError("clip origin/duration and bin size must be positive")
    if isinstance(max_bin_count, bool) or not isinstance(max_bin_count, int) or max_bin_count <= 0:
        raise DenseTemporalError("max_bin_count must be a positive integer")
    count = math.ceil(duration / size)
    if count > max_bin_count:
        raise DenseTemporalError(
            f"clip requires {count} bins, exceeding max_bin_count={max_bin_count}"
        )
    bins: list[dict[str, Any]] = []
    for index in range(count):
        local_start = index * size
        local_end = min(duration, (index + 1) * size)
        bins.append(
            {
                "bin_id": f"bin_{index:03d}",
                "local_start_sec": local_start,
                "local_end_sec": local_end,
                "source_start_sec": origin + local_start,
                "source_end_sec": origin + local_end,
            }
        )
    return bins


def _canonical_copy(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _positive_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _validate_clip_timing(
    timing: Mapping[str, Any], window: Mapping[str, Any]
) -> dict[str, Any]:
    if set(timing) != _CLIP_FIELDS:
        raise DenseTemporalError("clip timing has unexpected or missing fields")
    if timing.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise DenseTemporalError("unsupported clip coordinate contract")
    backend = timing.get("extraction_backend")
    if backend not in {"ffmpeg_reencode", "opencv_reencode", "injected_model_context"}:
        raise DenseTemporalError("unsupported clip extraction backend")
    requested_start = _finite_number(timing.get("requested_source_start_sec"), "requested start")
    requested_end = _finite_number(timing.get("requested_source_end_sec"), "requested end")
    origin = _finite_number(timing.get("actual_source_origin_sec"), "actual origin")
    duration = _finite_number(timing.get("clip_duration_sec"), "clip duration")
    if duration <= 0:
        raise DenseTemporalError("clip duration must be positive")
    if abs(requested_start - float(window["start_sec"])) > 1e-6 or abs(
        requested_end - float(window["end_sec"])
    ) > 1e-6:
        raise DenseTemporalError("clip timing does not match requested local window")
    if origin < requested_start - 1e-6 or origin >= requested_end:
        raise DenseTemporalError("clip origin is outside the requested window")
    allowed_overrun = 0.0
    checksum = timing.get("first_source_frame_checksum")
    if backend == "opencv_reencode":
        fps = _finite_number(timing.get("source_fps"), "source_fps")
        frame_index = timing.get("actual_source_frame_index")
        if fps <= 0 or not isinstance(frame_index, int) or frame_index < 0:
            raise DenseTemporalError("OpenCV clip frame identity is invalid")
        if abs(origin - frame_index / fps) > 1e-6:
            raise DenseTemporalError("OpenCV clip origin/frame identity mismatch")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise DenseTemporalError("OpenCV first-frame checksum is invalid")
        allowed_overrun = 1.0 / fps
    elif backend == "ffmpeg_reencode":
        if timing.get("actual_source_frame_index") is not None or timing.get("source_fps") is not None:
            raise DenseTemporalError("ffmpeg clip must not invent frame identity")
        if not isinstance(checksum, str) or not checksum:
            raise DenseTemporalError("ffmpeg first-frame checksum is invalid")
    if origin + duration > requested_end + allowed_overrun + 1e-6:
        raise DenseTemporalError("clip duration extends beyond requested window")
    return _canonical_copy(timing)


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _load_dense_envelope(raw_response: str) -> dict[str, Any]:
    """Accept a narrow set of envelopes while leaving bin content strict."""
    text = raw_response.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as whole_error:
        fenced = list(_FENCED_JSON.finditer(text))
        if len(fenced) != 1:
            raise DenseTemporalResponseError(
                "response must be JSON or contain exactly one complete fenced JSON payload"
            ) from whole_error
        outside = text[: fenced[0].start()] + text[fenced[0].end() :]
        if "```" in outside:
            raise DenseTemporalResponseError("response contains an ambiguous or unclosed fence")
        try:
            payload = json.loads(fenced[0].group(1))
        except json.JSONDecodeError as fenced_error:
            raise DenseTemporalResponseError("fenced payload is not strict JSON") from fenced_error
    if isinstance(payload, list):
        return {"bins": payload}
    if not isinstance(payload, dict):
        raise DenseTemporalResponseError("JSON payload must be an object or bins array")
    return payload


def parse_dense_response(
    raw_response: str, expected_bins: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Strictly parse exactly one label and diagnostic confidence per expected bin."""
    payload = _load_dense_envelope(raw_response)
    if set(payload) != {"bins"} or not isinstance(payload.get("bins"), list):
        raise DenseTemporalResponseError("top-level schema must contain only bins array")
    expected_ids = [str(item["bin_id"]) for item in expected_bins]
    if len(payload["bins"]) != len(expected_ids):
        raise DenseTemporalResponseError("bin count mismatch")
    parsed_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload["bins"]):
        if not isinstance(item, Mapping) or set(item) != {"bin_id", "label", "confidence"}:
            raise DenseTemporalResponseError(f"bins[{index}] has invalid schema")
        bin_id = item.get("bin_id")
        if not isinstance(bin_id, str) or bin_id not in expected_ids:
            raise DenseTemporalResponseError(f"bins[{index}] has unknown bin_id")
        if bin_id in parsed_by_id:
            raise DenseTemporalResponseError(f"duplicate bin_id: {bin_id}")
        label = item.get("label")
        if label not in LABELS:
            raise DenseTemporalResponseError(f"invalid label for {bin_id}")
        confidence = item.get("confidence")
        if isinstance(confidence, bool):
            raise DenseTemporalResponseError(f"confidence must be numeric for {bin_id}")
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError as error:
                raise DenseTemporalResponseError(
                    f"confidence must be numeric for {bin_id}"
                ) from error
        elif isinstance(confidence, (int, float)):
            confidence = float(confidence)
        else:
            raise DenseTemporalResponseError(f"confidence must be numeric for {bin_id}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise DenseTemporalResponseError(f"confidence out of range for {bin_id}")
        parsed_by_id[bin_id] = {
            "bin_id": bin_id,
            "label": str(label),
            "confidence": confidence,
        }
    if set(parsed_by_id) != set(expected_ids):
        raise DenseTemporalResponseError("missing expected bin")
    return [parsed_by_id[bin_id] for bin_id in expected_ids]


def _component_payload(
    component_bins: Sequence[Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    original_start: float,
    original_end: float,
) -> dict[str, Any]:
    core_bins = [item for item in component_bins if labels_by_id[str(item["bin_id"])]["label"] == "CORE"]
    start = float(component_bins[0]["source_start_sec"])
    end = float(component_bins[-1]["source_end_sec"])
    return {
        "bin_ids": [str(item["bin_id"]) for item in component_bins],
        "source_start_sec": start,
        "source_end_sec": end,
        "parent_overlap_duration_sec": _positive_overlap(start, end, original_start, original_end),
        "core_duration_sec": sum(
            float(item["source_end_sec"]) - float(item["source_start_sec"])
            for item in core_bins
        ),
        "core_bin_count": len(core_bins),
    }


def decode_dense_labels(
    temporal_bins: Sequence[Mapping[str, Any]],
    classified_bins: Sequence[Mapping[str, Any]],
    *,
    original_start_sec: float,
    original_end_sec: float,
) -> dict[str, Any]:
    """Apply the preregistered component/CORE/positive-parent-overlap decoder."""
    original_start = _finite_number(original_start_sec, "original_start_sec")
    original_end = _finite_number(original_end_sec, "original_end_sec")
    if original_end <= original_start:
        raise DenseTemporalError("original candidate interval is invalid")
    expected_ids = [str(item["bin_id"]) for item in temporal_bins]
    observed_ids = [str(item.get("bin_id")) for item in classified_bins]
    if observed_ids != expected_ids:
        raise DenseTemporalError("classified bins must be in canonical expected order")
    labels_by_id = {str(item["bin_id"]): item for item in classified_bins}
    if any(item.get("label") not in LABELS for item in classified_bins):
        raise DenseTemporalError("classified bins contain an invalid label")

    components: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for bin_payload in temporal_bins:
        if labels_by_id[str(bin_payload["bin_id"])]["label"] == "OUTSIDE":
            if current:
                components.append(current)
                current = []
        else:
            current.append(bin_payload)
    if current:
        components.append(current)

    with_core = [
        component
        for component in components
        if any(labels_by_id[str(item["bin_id"])]["label"] == "CORE" for item in component)
    ]
    if not with_core:
        return {
            "decision_rule": "dtl1.fallback_no_core",
            "refined_start_sec": original_start,
            "refined_end_sec": original_end,
            "selected_component": None,
        }
    payloads = [
        _component_payload(component, labels_by_id, original_start, original_end)
        for component in with_core
    ]
    valid = [item for item in payloads if item["parent_overlap_duration_sec"] > 0.0]
    if not valid:
        return {
            "decision_rule": "dtl1.fallback_no_parent_overlap",
            "refined_start_sec": original_start,
            "refined_end_sec": original_end,
            "selected_component": None,
        }
    selected = min(
        valid,
        key=lambda item: (
            -float(item["parent_overlap_duration_sec"]),
            -float(item["core_duration_sec"]),
            -int(item["core_bin_count"]),
            float(item["source_start_sec"]),
        ),
    )
    refined_start = float(selected["source_start_sec"])
    refined_end = float(selected["source_end_sec"])
    rule = (
        "dtl1.dense_identity"
        if refined_start == original_start and refined_end == original_end
        else "dtl1.dense_refine"
    )
    return {
        "decision_rule": rule,
        "refined_start_sec": refined_start,
        "refined_end_sec": refined_end,
        "selected_component": selected,
    }


def build_dense_prompt(
    highlight_query: str,
    temporal_bins: Sequence[Mapping[str, Any]],
    *,
    original_start_sec: float,
    original_end_sec: float,
) -> str:
    """Build the short JSON-only DTL-1 classification prompt."""
    rows = []
    for item in temporal_bins:
        overlap = _positive_overlap(
            float(item["source_start_sec"]),
            float(item["source_end_sec"]),
            float(original_start_sec),
            float(original_end_sec),
        ) > 0.0
        rows.append(
            f'{item["bin_id"]} local=[{float(item["local_start_sec"]):.3f},'
            f'{float(item["local_end_sec"]):.3f}) overlap={str(overlap).lower()}'
        )
    query = highlight_query.strip() or "current frozen highlight candidate"
    expected_ids = ",".join(str(item["bin_id"]) for item in temporal_bins)
    return (
        f"Task: label fixed bins for the current anchored highlight.\n"
        f"Highlight query: {query}\n"
        f"Original candidate source interval: [{float(original_start_sec):.3f}, {float(original_end_sec):.3f})\n"
        "Labels: CORE=essential action/result; CONTEXT=necessary adjacent lead-in/aftermath/reaction; "
        "OUTSIDE=unrelated/idle/ordinary setup/transition.\n"
        "If uncertain between CONTEXT and OUTSIDE, choose CONTEXT. If uncertain between CORE and CONTEXT, "
        "choose CORE or CONTEXT, never aggressive OUTSIDE.\n"
        "The candidate is the anchor. Do not find another highlight. Do not drop/create candidates, change score, "
        "use references/metrics/human labels, or search beyond this local window. Do not output timestamps.\n"
        "OUTPUT CONTRACT: ONLY return one JSON object with top-level key bins. DO NOT return a bare JSON array. "
        "DO NOT add prose, reasoning, explanation, or Markdown commentary. DO NOT omit any bin or stop early. "
        f"Return all {len(temporal_bins)} bins in one response. Required IDs exactly once: {expected_ids}.\n"
        "Label every bin exactly once. JSON only: {\"bins\":[{\"bin_id\":\"bin_000\",\"label\":\"CORE\",\"confidence\":0.8}]}\n"
        + "\n".join(rows)
    )


PROMPT_SEMANTIC_SHA256 = semantic_sha256(
    {"prompt_id": PROMPT_ID, "template": build_dense_prompt("<query>", [
        {"bin_id": "bin_000", "local_start_sec": 0.0, "local_end_sec": 2.0,
         "source_start_sec": 10.0, "source_end_sec": 12.0}
    ], original_start_sec=10.0, original_end_sec=12.0)}
)


def _validate_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if set(config) != {"localizer_name", "localizer_version", "parameters"}:
        raise DenseTemporalError("localizer config has unexpected or missing fields")
    name = config.get("localizer_name")
    if name not in {"DTL-0", "DTL-1"} or config.get("localizer_version") != DTL_VERSION:
        raise DenseTemporalError("unsupported localizer config")
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise DenseTemporalError("localizer parameters must be an object")
    expected = set() if name == "DTL-0" else {
        "context_padding_sec", "max_context_duration_sec", "bin_size_sec", "max_bin_count"
    }
    if set(parameters) != expected:
        raise DenseTemporalError("localizer parameters do not match the fixed design")
    if name == "DTL-1":
        for key in ("context_padding_sec", "max_context_duration_sec", "bin_size_sec"):
            if _finite_number(parameters[key], key) <= 0:
                raise DenseTemporalError(f"{key} must be positive")
        if not isinstance(parameters["max_bin_count"], int) or isinstance(parameters["max_bin_count"], bool):
            raise DenseTemporalError("max_bin_count must be an integer")
    return str(name), _canonical_copy(parameters)


def _identity_localization(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "merged_candidate_id": str(candidate["merged_candidate_id"]),
        "parent_candidate_semantic_sha256": semantic_sha256(candidate),
        "original_start_sec": float(candidate["start_sec"]),
        "original_end_sec": float(candidate["end_sec"]),
        "refined_start_sec": float(candidate["start_sec"]),
        "refined_end_sec": float(candidate["end_sec"]),
        "decision_rule": "dtl0.identity",
        "local_context_window": None,
        "clip_timing": None,
        "temporal_bins": [],
        "classified_bins": [],
        "selected_component": None,
        "model_response": None,
    }


def _fallback_localization(
    base: Mapping[str, Any], rule: str, model_response: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(base)
    result.update({
        "refined_start_sec": float(base["original_start_sec"]),
        "refined_end_sec": float(base["original_end_sec"]),
        "decision_rule": rule,
        "classified_bins": [],
        "selected_component": None,
        "model_response": _canonical_copy(model_response),
    })
    return result


def _dense_localization(
    candidate: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    duration_sec: float,
    video_id: str,
    model_fn: ModelFn,
    prepare_context_fn: PrepareContextFn,
) -> dict[str, Any]:
    from .boundary_refinement import compute_local_context_window

    original_start = float(candidate["start_sec"])
    original_end = float(candidate["end_sec"])
    window = compute_local_context_window(
        start_sec=original_start,
        end_sec=original_end,
        duration_sec=duration_sec,
        context_padding_sec=float(parameters["context_padding_sec"]),
        max_context_duration_sec=float(parameters["max_context_duration_sec"]),
    )
    candidate_id = str(candidate["merged_candidate_id"])
    timing = _validate_clip_timing(
        prepare_context_fn(video_id=video_id, candidate_id=candidate_id, window=window), window
    )
    bins = build_temporal_bins(
        timing,
        bin_size_sec=float(parameters["bin_size_sec"]),
        max_bin_count=int(parameters["max_bin_count"]),
    )
    base = {
        "merged_candidate_id": candidate_id,
        "parent_candidate_semantic_sha256": semantic_sha256(candidate),
        "original_start_sec": original_start,
        "original_end_sec": original_end,
        "local_context_window": window,
        "clip_timing": timing,
        "temporal_bins": bins,
    }
    prompt = build_dense_prompt(
        str(candidate.get("reason", "")), bins,
        original_start_sec=original_start, original_end_sec=original_end,
    )
    try:
        response = model_fn(prompt, video_id=video_id, candidate_id=candidate_id, window=window)
    except (ConnectionError, TimeoutError) as exc:
        return _fallback_localization(base, "dtl1.fallback_transport_error", {
            "transport_error_type": type(exc).__name__, "finish_reason": None,
            "raw_response_preview": "", "text_sha256": semantic_sha256("")
        })
    if isinstance(response, tuple):
        raw_text, finish_reason = response
    else:
        raw_text, finish_reason = response, None
    if not isinstance(raw_text, str):
        raise DenseTemporalError("model callable must return text or (text, finish_reason)")
    provenance = {
        "transport_error_type": None,
        "finish_reason": finish_reason,
        "raw_response_preview": raw_text,
        "text_sha256": semantic_sha256(raw_text),
    }
    if not raw_text.strip():
        return _fallback_localization(base, "dtl1.fallback_empty_response", provenance)
    try:
        classified = parse_dense_response(raw_text, bins)
    except DenseTemporalResponseError:
        return _fallback_localization(base, "dtl1.fallback_parse_error", provenance)
    decoded = decode_dense_labels(
        bins, classified, original_start_sec=original_start, original_end_sec=original_end
    )
    return {
        **base,
        **decoded,
        "classified_bins": classified,
        "model_response": provenance,
    }


def create_dense_localization_result(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    localizer_config: Mapping[str, Any],
    *,
    model_fn: ModelFn | None = None,
    prepare_context_fn: PrepareContextFn | None = None,
    protocol_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one protocol-bindable localization result for an identity-only role."""
    name, parameters = _validate_config(localizer_config)
    if cache_manifest.get("global_semantic_sha256") != role_manifest.get("source_cache_global_hash"):
        raise DenseTemporalError("role manifest does not match source cache")
    if name == "DTL-1" and (model_fn is None or prepare_context_fn is None):
        raise DenseTemporalError("DTL-1 requires model and context preparation callables")
    role_records = role_manifest.get("records")
    if not isinstance(role_records, list):
        raise DenseTemporalError("role records are missing")
    output_records = []
    decision_counts = {rule: 0 for rule in _RULES if rule.startswith(name.lower().replace("-", ""))}
    candidate_count = 0
    seen: set[str] = set()
    for identity in role_records:
        video_id = identity.get("video_id") if isinstance(identity, Mapping) else None
        if not isinstance(video_id, str) or video_id in seen:
            raise DenseTemporalError("role contains duplicate or invalid video ID")
        seen.add(video_id)
        record = cache_records_by_id.get(video_id)
        if record is None or record.get("split") != identity.get("split"):
            raise DenseTemporalError(f"role/cache mismatch for {video_id}")
        candidates = record.get("merged_candidates")
        if not isinstance(candidates, list):
            raise DenseTemporalError(f"merged candidates missing for {video_id}")
        localizations = []
        for item in candidates:
            if name == "DTL-0":
                localization = _identity_localization(item)
            else:
                assert model_fn is not None and prepare_context_fn is not None
                localization = _dense_localization(
                    item, parameters, duration_sec=float(record["duration_sec"]),
                    video_id=video_id, model_fn=model_fn, prepare_context_fn=prepare_context_fn,
                )
            decision_counts.setdefault(localization["decision_rule"], 0)
            decision_counts[localization["decision_rule"]] += 1
            localizations.append(localization)
            candidate_count += 1
        output_record = {
            "video_id": video_id,
            "split": record["split"],
            "duration_sec": float(record["duration_sec"]),
            "source_record_semantic_sha256": record.get("semantic_sha256"),
            "candidate_localizations": localizations,
        }
        output_record["video_localization_semantic_sha256"] = semantic_sha256(output_record)
        output_records.append(output_record)
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "localizer_name": name,
        "localizer_version": DTL_VERSION,
        "localizer_config": _canonical_copy(localizer_config),
        "localizer_config_hash": semantic_sha256(localizer_config),
        "source_cache_global_hash": cache_manifest.get("global_semantic_sha256"),
        "role": role_manifest.get("role"),
        "role_manifest_hash": role_manifest.get("semantic_sha256"),
        "protocol_binding": _canonical_copy(protocol_binding) if protocol_binding is not None else None,
        "input_record_count": len(output_records),
        "input_candidate_count": candidate_count,
        "decision_rule_counts": decision_counts,
        "records": output_records,
    }
    result["dense_localization_semantic_sha256"] = semantic_sha256(result)
    return result


def validate_dense_localization_payload(
    result: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    expected_protocol_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct bins/components/output and reject artifact drift."""
    claimed = result.get("dense_localization_semantic_sha256")
    if claimed != semantic_sha256(_semantic_payload(result, "dense_localization_semantic_sha256")):
        raise DenseTemporalError("dense localization semantic hash mismatch")
    if result.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise DenseTemporalError("unsupported result schema")
    if result.get("source_cache_global_hash") != cache_manifest.get("global_semantic_sha256"):
        raise DenseTemporalError("cache hash mismatch")
    if result.get("role") != role_manifest.get("role") or result.get("role_manifest_hash") != role_manifest.get("semantic_sha256"):
        raise DenseTemporalError("role binding mismatch")
    if result.get("protocol_binding") != expected_protocol_binding:
        raise DenseTemporalError("protocol binding mismatch")
    name, parameters = _validate_config(result.get("localizer_config", {}))
    if result.get("localizer_name") != name or result.get("localizer_config_hash") != semantic_sha256(result.get("localizer_config")):
        raise DenseTemporalError("localizer config binding mismatch")
    result_records = result.get("records")
    role_records = role_manifest.get("records")
    if not isinstance(result_records, list) or not isinstance(role_records, list) or len(result_records) != len(role_records):
        raise DenseTemporalError("record count changed")
    counts: dict[str, int] = {
        rule: 0 for rule in _RULES if rule.startswith(name.lower().replace("-", ""))
    }
    candidate_total = 0
    for output_record, identity in zip(result_records, role_records, strict=True):
        video_id = identity["video_id"]
        record = cache_records_by_id.get(video_id)
        if record is None or output_record.get("video_id") != video_id or output_record.get("split") != record.get("split"):
            raise DenseTemporalError("record identity mismatch")
        if output_record.get("source_record_semantic_sha256") != record.get("semantic_sha256"):
            raise DenseTemporalError("source record hash mismatch")
        if output_record.get("duration_sec") != float(record["duration_sec"]):
            raise DenseTemporalError("duration mismatch")
        video_hash = output_record.get("video_localization_semantic_sha256")
        if video_hash != semantic_sha256(_semantic_payload(output_record, "video_localization_semantic_sha256")):
            raise DenseTemporalError("video localization hash mismatch")
        localizations = output_record.get("candidate_localizations")
        candidates = record.get("merged_candidates")
        if not isinstance(localizations, list) or not isinstance(candidates, list) or len(localizations) != len(candidates):
            raise DenseTemporalError("candidate count changed")
        for localization, candidate in zip(localizations, candidates, strict=True):
            original_start = float(candidate["start_sec"])
            original_end = float(candidate["end_sec"])
            if localization.get("merged_candidate_id") != candidate.get("merged_candidate_id"):
                raise DenseTemporalError("candidate ID mismatch")
            if localization.get("parent_candidate_semantic_sha256") != semantic_sha256(candidate):
                raise DenseTemporalError("parent candidate hash mismatch")
            if localization.get("original_start_sec") != original_start or localization.get("original_end_sec") != original_end:
                raise DenseTemporalError("original interval mismatch")
            rule = localization.get("decision_rule")
            if rule not in _RULES or not str(rule).startswith(name.lower().replace("-", "")):
                raise DenseTemporalError("invalid decision rule")
            if name == "DTL-0":
                expected = _identity_localization(candidate)
                if localization != expected:
                    raise DenseTemporalError("DTL-0 is not exact identity")
            else:
                window = localization.get("local_context_window")
                timing = localization.get("clip_timing")
                if not isinstance(window, Mapping) or not isinstance(timing, Mapping):
                    raise DenseTemporalError("DTL-1 clip provenance missing")
                from .boundary_refinement import compute_local_context_window
                expected_window = compute_local_context_window(
                    start_sec=original_start, end_sec=original_end,
                    duration_sec=float(record["duration_sec"]),
                    context_padding_sec=float(parameters["context_padding_sec"]),
                    max_context_duration_sec=float(parameters["max_context_duration_sec"]),
                )
                if window != expected_window:
                    raise DenseTemporalError("local context window mismatch")
                checked_timing = _validate_clip_timing(timing, expected_window)
                expected_bins = build_temporal_bins(
                    checked_timing, bin_size_sec=float(parameters["bin_size_sec"]),
                    max_bin_count=int(parameters["max_bin_count"]),
                )
                if localization.get("temporal_bins") != expected_bins:
                    raise DenseTemporalError("temporal bin construction mismatch")
                classified = localization.get("classified_bins")
                response = localization.get("model_response")
                if not isinstance(response, Mapping):
                    raise DenseTemporalError("model response provenance missing")
                preview = response.get("raw_response_preview")
                if not isinstance(preview, str) or response.get("text_sha256") != semantic_sha256(preview):
                    raise DenseTemporalError("model response hash mismatch")
                if rule in {"dtl1.fallback_transport_error", "dtl1.fallback_empty_response", "dtl1.fallback_parse_error"}:
                    if classified != [] or localization.get("selected_component") is not None:
                        raise DenseTemporalError("response fallback contains decoded labels")
                    if rule == "dtl1.fallback_empty_response" and preview.strip():
                        raise DenseTemporalError("empty fallback provenance mismatch")
                    if rule == "dtl1.fallback_parse_error":
                        try:
                            parse_dense_response(preview, expected_bins)
                        except DenseTemporalResponseError:
                            pass
                        else:
                            raise DenseTemporalError("parse fallback provenance mismatch")
                    if rule == "dtl1.fallback_transport_error" and response.get("transport_error_type") not in {"ConnectionError", "TimeoutError"}:
                        raise DenseTemporalError("transport fallback provenance mismatch")
                    decoded = {
                        "decision_rule": rule, "refined_start_sec": original_start,
                        "refined_end_sec": original_end, "selected_component": None,
                    }
                else:
                    if not isinstance(classified, list):
                        raise DenseTemporalError("classified bins missing")
                    canonical_classified = parse_dense_response(preview, expected_bins)
                    if classified != canonical_classified:
                        raise DenseTemporalError("classified bins do not match model response")
                    decoded = decode_dense_labels(
                        expected_bins, canonical_classified,
                        original_start_sec=original_start, original_end_sec=original_end,
                    )
                    if decoded["decision_rule"] != rule:
                        raise DenseTemporalError("decision rule is not reproducible")
                for field in ("refined_start_sec", "refined_end_sec", "selected_component"):
                    if localization.get(field) != decoded[field]:
                        raise DenseTemporalError(f"decoder output mismatch: {field}")
            counts[str(rule)] = counts.get(str(rule), 0) + 1
            candidate_total += 1
    if result.get("input_record_count") != len(result_records) or result.get("input_candidate_count") != candidate_total:
        raise DenseTemporalError("aggregate input count mismatch")
    if result.get("decision_rule_counts") != counts:
        raise DenseTemporalError("decision rule counts mismatch")
    return {
        "record_count": len(result_records),
        "candidate_count": candidate_total,
        "decision_rule_counts": counts,
        "dense_localization_semantic_sha256": claimed,
    }


def replay_dense_predictions(
    result: Mapping[str, Any], cache_records_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Replay boundaries while re-projecting score/reason/source from frozen cache."""
    replayed = []
    records = result.get("records")
    if not isinstance(records, list):
        raise DenseTemporalError("result records missing")
    for output_record in records:
        video_id = str(output_record.get("video_id"))
        source = cache_records_by_id.get(video_id)
        if source is None:
            raise DenseTemporalError(f"replay video missing from cache: {video_id}")
        candidates = source.get("merged_candidates")
        localizations = output_record.get("candidate_localizations")
        if not isinstance(candidates, list) or not isinstance(localizations, list) or len(candidates) != len(localizations):
            raise DenseTemporalError("candidate count changed during replay")
        segments = []
        for candidate, localization in zip(candidates, localizations, strict=True):
            if candidate.get("merged_candidate_id") != localization.get("merged_candidate_id"):
                raise DenseTemporalError("candidate ID changed during replay")
            segment = {
                "start_sec": float(localization["refined_start_sec"]),
                "end_sec": float(localization["refined_end_sec"]),
                "score": float(candidate["score"]),
                "reason": str(candidate.get("reason", "")),
                "source_chunk": candidate.get("source_chunk"),
            }
            if "source" in candidate:
                segment["source"] = _canonical_copy(candidate["source"])
            segments.append(segment)
        replayed.append({"video_id": video_id, "split": output_record.get("split"), "merged_prediction_segments": segments})
    return replayed


def full_cache_dtl0_identity(
    cache_manifest: Mapping[str, Any], cache_records_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    """CPU-only exact identity regression over every manifest record/candidate."""
    records = cache_manifest.get("records")
    if not isinstance(records, list):
        raise DenseTemporalError("cache manifest records missing")
    mismatches = 0
    candidates = 0
    for identity in records:
        video_id = identity.get("video_id")
        record = cache_records_by_id.get(video_id)
        if record is None:
            raise DenseTemporalError(f"cache record missing: {video_id}")
        for candidate in record.get("merged_candidates", []):
            decision = _identity_localization(candidate)
            candidates += 1
            if (
                decision["merged_candidate_id"] != candidate.get("merged_candidate_id")
                or decision["refined_start_sec"] != candidate.get("start_sec")
                or decision["refined_end_sec"] != candidate.get("end_sec")
            ):
                mismatches += 1
    return {"records": len(records), "candidates": candidates, "mismatches": mismatches}


def load_dense_protocol(path: Path, *, allow_draft: bool = True) -> dict[str, Any]:
    protocol = _read_json_object(path.expanduser().resolve())
    claimed = protocol.get("protocol_semantic_sha256")
    if claimed != semantic_sha256(_semantic_payload(protocol, "protocol_semantic_sha256")):
        raise DenseTemporalError("protocol semantic hash mismatch")
    accepted = {"PREREGISTERED_BEFORE_FORMAL"}
    if allow_draft:
        accepted.add("DRAFT_FOR_INDEPENDENT_REVIEW")
    if protocol.get("protocol_status") not in accepted:
        raise DenseTemporalError("protocol status is not accepted for this run")
    if protocol.get("source_cache_global_hash") != EXPECTED_CACHE_GLOBAL_HASH:
        raise DenseTemporalError("protocol is not bound to the frozen Stage 4.2 cache")
    dtl1 = protocol.get("localizers", {}).get("DTL-1", {})
    if dtl1.get("prompt", {}).get("prompt_semantic_sha256") != PROMPT_SEMANTIC_SHA256:
        raise DenseTemporalError("protocol prompt hash mismatch")
    if protocol.get("decoder", {}).get("decoder_version") != DECODER_VERSION:
        raise DenseTemporalError("protocol decoder version mismatch")
    if protocol.get("label_schema", {}).get("labels") != list(LABELS):
        raise DenseTemporalError("protocol label enum mismatch")
    config_from_protocol(protocol, "DTL-0")
    config_from_protocol(protocol, "DTL-1")
    return protocol


def config_from_protocol(protocol: Mapping[str, Any], name: str) -> dict[str, Any]:
    entry = protocol.get("localizers", {}).get(name)
    if not isinstance(entry, Mapping):
        raise DenseTemporalError(f"protocol localizer missing: {name}")
    config = {
        "localizer_name": name,
        "localizer_version": entry.get("localizer_version"),
        "parameters": entry.get("parameters"),
    }
    _validate_config(config)
    return config


def resolve_git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path.expanduser().resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise DenseTemporalError("cannot resolve Git HEAD") from exc
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise DenseTemporalError("cannot resolve Git HEAD")
    return head


def build_protocol_binding(
    protocol: Mapping[str, Any], *, protocol_bytes_sha256: str,
    localizer_name: str, role_manifest_hash: str, git_head: str,
) -> dict[str, Any]:
    entry = protocol["localizers"][localizer_name]
    model = entry.get("model") if localizer_name == "DTL-1" else None
    prompt = entry.get("prompt") if localizer_name == "DTL-1" else None
    return {
        "protocol_status": protocol["protocol_status"],
        "protocol_semantic_sha256": protocol["protocol_semantic_sha256"],
        "protocol_bytes_sha256": protocol_bytes_sha256,
        "source_cache_global_hash": protocol["source_cache_global_hash"],
        "role_manifest_hash": role_manifest_hash,
        "localizer_name": localizer_name,
        "localizer_version": entry["localizer_version"],
        "parameters": _canonical_copy(entry["parameters"]),
        "model_identifier": None if model is None else model["model_name"],
        "model_revision": None if model is None else model["model_revision"],
        "prompt_id": None if prompt is None else prompt["prompt_id"],
        "prompt_version": None if prompt is None else prompt["prompt_version"],
        "prompt_semantic_sha256": None if prompt is None else prompt["prompt_semantic_sha256"],
        "decoder_version": None if localizer_name == "DTL-0" else DECODER_VERSION,
        "clip_coordinate_contract": COORDINATE_CONTRACT,
        "git_head": git_head,
    }


def load_cache_payloads(cache_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .boundary_refinement import load_cache_payloads as load_frozen_cache
    manifest, records = load_frozen_cache(cache_dir)
    return manifest, records


def _load_protocol_role(
    cache_dir: Path, role_manifest_path: Path, protocol_path: Path, *, allow_draft: bool
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, records = load_cache_payloads(cache_dir)
    role_path = role_manifest_path.expanduser().resolve()
    role_summary = validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role = _read_json_object(role_path)
    validate_role_manifest(role, expected_count=len(role.get("records", [])))
    resolved_protocol = protocol_path.expanduser().resolve()
    protocol = load_dense_protocol(resolved_protocol, allow_draft=allow_draft)
    if protocol.get("role_manifest_summary_hash") != role_summary.get("semantic_sha256"):
        raise DenseTemporalError("protocol role summary hash mismatch")
    if protocol.get("role_manifest_hashes", {}).get(role["role"]) != role["semantic_sha256"]:
        raise DenseTemporalError("protocol role hash mismatch")
    binding = build_protocol_binding(
        protocol,
        protocol_bytes_sha256=hashlib.sha256(resolved_protocol.read_bytes()).hexdigest(),
        localizer_name="DTL-0",
        role_manifest_hash=role["semantic_sha256"],
        git_head=resolve_git_head(resolved_protocol),
    )
    return manifest, records, role, protocol, binding


def run_localization_to_file(
    cache_dir: Path, role_manifest_path: Path, protocol_path: Path,
    localizer_name: str, output_path: Path, *, model_fn: ModelFn | None = None,
    prepare_context_fn: PrepareContextFn | None = None, allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    manifest, records, role, protocol, binding = _load_protocol_role(
        cache_dir, role_manifest_path, protocol_path, allow_draft=allow_draft_protocol
    )
    binding = build_protocol_binding(
        protocol,
        protocol_bytes_sha256=hashlib.sha256(protocol_path.expanduser().resolve().read_bytes()).hexdigest(),
        localizer_name=localizer_name,
        role_manifest_hash=role["semantic_sha256"],
        git_head=resolve_git_head(protocol_path),
    )
    result = create_dense_localization_result(
        manifest, role, records, config_from_protocol(protocol, localizer_name),
        model_fn=model_fn, prepare_context_fn=prepare_context_fn, protocol_binding=binding,
    )
    validate_dense_localization_payload(
        result, manifest, role, records, expected_protocol_binding=binding
    )
    _write_new_canonical_json(output_path, result)
    return result


def validate_localization_file(
    cache_dir: Path, role_manifest_path: Path, result_path: Path, protocol_path: Path,
    *, allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    manifest, records, role, protocol, _ = _load_protocol_role(
        cache_dir, role_manifest_path, protocol_path, allow_draft=allow_draft_protocol
    )
    payload_path = result_path.expanduser().resolve()
    result = _read_json_object(payload_path)
    if payload_path.read_bytes() != canonical_json_bytes(result):
        raise DenseTemporalError("localization result is not canonical JSON")
    binding = build_protocol_binding(
        protocol,
        protocol_bytes_sha256=hashlib.sha256(protocol_path.expanduser().resolve().read_bytes()).hexdigest(),
        localizer_name=result.get("localizer_name"),
        role_manifest_hash=role["semantic_sha256"],
        git_head=resolve_git_head(protocol_path),
    )
    return validate_dense_localization_payload(
        result, manifest, role, records, expected_protocol_binding=binding
    )


def replay_localization_to_jsonl(
    cache_dir: Path, role_manifest_path: Path, result_path: Path, output_path: Path,
    protocol_path: Path, *, allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    validation = validate_localization_file(
        cache_dir, role_manifest_path, result_path, protocol_path,
        allow_draft_protocol=allow_draft_protocol,
    )
    _, records = load_cache_payloads(cache_dir)
    result = _read_json_object(result_path.expanduser().resolve())
    replayed = replay_dense_predictions(result, records)
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(canonical_json_bytes(item) for item in replayed))
    return {**validation, "segment_count": sum(len(x["merged_prediction_segments"]) for x in replayed)}


def evaluate_localization_to_file(
    cache_dir: Path, role_manifest_path: Path, result_path: Path,
    replay_path: Path, frozen_prediction_paths: Iterable[Path], output_path: Path,
    protocol_path: Path, *, allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    from .candidate_selection import evaluate_replayed_predictions
    validate_localization_file(
        cache_dir, role_manifest_path, result_path, protocol_path,
        allow_draft_protocol=allow_draft_protocol,
    )
    manifest, records, role, _, _ = _load_protocol_role(
        cache_dir, role_manifest_path, protocol_path, allow_draft=allow_draft_protocol
    )
    result = _read_json_object(result_path.expanduser().resolve())
    replayed = _read_jsonl(replay_path.expanduser().resolve())
    if replayed != replay_dense_predictions(result, records):
        raise DenseTemporalError("replay does not match validated localization")
    frozen: dict[str, dict[str, Any]] = {}
    for path in frozen_prediction_paths:
        for item in _read_jsonl(path.expanduser().resolve()):
            video_id = item.get("video_id")
            if not isinstance(video_id, str) or video_id in frozen:
                raise DenseTemporalError("frozen predictions contain invalid/duplicate video ID")
            frozen[video_id] = item
    role_ids = [item["video_id"] for item in role["records"]]
    if any(video_id not in frozen for video_id in role_ids):
        raise DenseTemporalError("frozen predictions do not cover role")
    evaluation = evaluate_replayed_predictions(
        replayed, [frozen[x] for x in role_ids],
        durations_by_id={x: float(records[x]["duration_sec"]) for x in role_ids},
        role=role["role"], role_manifest_hash=role["semantic_sha256"],
        selection_result_hash=result["dense_localization_semantic_sha256"],
        selection_metadata={
            "source_cache_global_hash": manifest["global_semantic_sha256"],
            "localizer_name": result["localizer_name"],
            "localizer_version": result["localizer_version"],
            "localizer_config": result["localizer_config"],
            "localizer_config_hash": result["localizer_config_hash"],
        },
    )
    _write_new_canonical_json(output_path, evaluation)
    return evaluation


def summarize_evaluations_to_file(
    baseline_evaluation_path: Path, candidate_evaluation_path: Path,
    protocol_path: Path, output_path: Path, *, allow_draft_protocol: bool = True,
) -> dict[str, Any]:
    from .candidate_selection import compare_evaluations
    protocol = load_dense_protocol(protocol_path, allow_draft=allow_draft_protocol)
    comparison = compare_evaluations(
        _read_json_object(baseline_evaluation_path.expanduser().resolve()),
        _read_json_object(candidate_evaluation_path.expanduser().resolve()),
    )
    summary = {
        "summary_schema_version": "aic.dense-temporal-summary/v1",
        "protocol_semantic_sha256": protocol["protocol_semantic_sha256"],
        "promotion_gate": None,
        "comparison": comparison,
    }
    summary["summary_semantic_sha256"] = semantic_sha256(summary)
    _write_new_canonical_json(output_path, summary)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DenseTemporalError(f"invalid JSONL line {index}: {path}") from exc
        if not isinstance(item, dict):
            raise DenseTemporalError(f"JSONL line {index} is not an object: {path}")
        rows.append(item)
    return rows
