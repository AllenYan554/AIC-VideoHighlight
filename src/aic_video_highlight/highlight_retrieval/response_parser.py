"""Strict parsing for model-produced highlight retrieval JSON."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .schemas import HighlightSegment


class ResponseParseError(ValueError):
    """Raised when a model response violates the highlight retrieval contract."""


class TruncatedResponseError(ResponseParseError):
    """Raised when a model response appears cut off before JSON completion.

    Truncated responses must never be treated as successful formal results.
    """


_JSON_FENCE = re.compile(r"\A```(?:json)?\s*(\{.*\})\s*```\Z", re.DOTALL | re.IGNORECASE)

_TRUNCATION_HINTS = (
    "Unterminated string",
    "Expecting value",
    "Expecting property name",
    "Expecting ',' delimiter",
    "Expecting ':' delimiter",
)


def _load_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    fence_match = _JSON_FENCE.fullmatch(text)
    if fence_match:
        text = fence_match.group(1)
    elif text.startswith("```") and not text.endswith("```"):
        raise TruncatedResponseError(
            "response is fenced with ``` but the closing fence is missing; "
            "the response was likely truncated by the token limit"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith(_TRUNCATION_HINTS) and exc.pos >= len(text.rstrip()) - 1:
            raise TruncatedResponseError(
                f"response appears truncated before valid JSON completion: "
                f"{exc.msg} at position {exc.pos}"
            ) from exc
        if text.endswith("```"):
            # A complete JSON object followed by a stray closing fence is a
            # formatting drift, not truncation: strip the fence and retry once.
            try:
                payload = json.loads(text[:-3].rstrip())
            except json.JSONDecodeError:
                raise ResponseParseError(f"response is not strict JSON: {exc.msg}") from exc
        else:
            raise ResponseParseError(f"response is not strict JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ResponseParseError("top-level JSON value must be an object")
    return payload


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseParseError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ResponseParseError(f"{field_name} must be finite")
    return number


def parse_highlight_response(
    raw_response: str,
    *,
    chunk_duration_sec: float,
    source_chunk: int,
) -> list[HighlightSegment]:
    """Parse chunk-local seconds, clipping only an end beyond the chunk boundary."""
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be greater than zero")

    payload = _load_json_object(raw_response)
    has_highlight = payload.get("has_highlight")
    raw_segments = payload.get("segments")
    if not isinstance(has_highlight, bool):
        raise ResponseParseError("has_highlight must be a boolean")
    if not isinstance(raw_segments, list):
        raise ResponseParseError("segments must be an array")
    if not has_highlight and raw_segments:
        raise ResponseParseError("segments must be empty when has_highlight is false")

    parsed: list[HighlightSegment] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ResponseParseError(f"segments[{index}] must be an object")
        start_sec = _finite_number(item.get("start_sec"), f"segments[{index}].start_sec")
        end_sec = _finite_number(item.get("end_sec"), f"segments[{index}].end_sec")
        score = _finite_number(item.get("score"), f"segments[{index}].score")
        reason = item.get("reason", "")
        if not isinstance(reason, str):
            raise ResponseParseError(f"segments[{index}].reason must be a string")
        if start_sec < 0:
            raise ResponseParseError(f"segments[{index}].start_sec must be >= 0")
        if end_sec <= start_sec:
            raise ResponseParseError(f"segments[{index}].end_sec must be > start_sec")
        if not 0 <= score <= 1:
            raise ResponseParseError(f"segments[{index}].score must be in [0, 1]")
        if start_sec >= chunk_duration_sec:
            raise ResponseParseError(f"segments[{index}] starts outside the chunk")
        if end_sec > chunk_duration_sec:
            original_end = end_sec
            end_sec = chunk_duration_sec
            note = f"[parser: clipped end_sec from {original_end:g} to {end_sec:g}]"
            reason = f"{reason} {note}".strip()

        parsed.append(
            HighlightSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                score=score,
                reason=reason,
                source_chunk=source_chunk,
            )
        )

    if has_highlight and not parsed:
        raise ResponseParseError("segments must not be empty when has_highlight is true")
    return parsed
