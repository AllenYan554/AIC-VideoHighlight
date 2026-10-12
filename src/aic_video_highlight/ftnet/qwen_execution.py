"""Qwen retrieval execution instrumentation (Stage 7 PHD2 engineering only).

This module adds two purely-execution capabilities on top of the frozen
retrieval building blocks:

``overlap_render``
    Render chunk *N+1* (ffmpeg, CPU) while the request for chunk *N* is in
    flight.  Requests are still issued strictly one at a time and in chunk
    order, so the model observes exactly the same single-request batch stream
    as the promoted ``concurrency = 1`` execution.  The frozen chunker, prompt
    builder, parser and merger are called unchanged.

audit sidecar
    Per-chunk raw response, usage counters, finish reason and phase timings.
    It is written next to the retrieval artifact (never inside it), so the
    frozen artifact schema is untouched.

Nothing in this module changes model identity, prompt, sampling fps, chunking,
overlap, temperature, thinking mode, token limits, parser or merge semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .index import IndexEntry, resolve_video_path
from .real_provider import (
    RECOVERY_MAX_NEW_TOKENS,
    RECOVERY_TEMPERATURE,
    RETRIEVAL_SCHEMA,
    QWEN_MODEL,
    QWEN_REVISION,
    RealProviderError,
    RetrievalSettings,
    _sha256_file,
)

CHUNK_AUDIT_SCHEMA = "aic.stage7.ftnet.qwen-chunk-audit/v1"
ARM_SUMMARY_SCHEMA = "aic.stage7.ftnet.qwen-audit-arm/v1"
COMPARE_SCHEMA = "aic.stage7.ftnet.qwen-audit-compare/v1"

AUDIT_SUFFIX = ".chunks.json"


def chunk_audit_path(root: str | Path, video_id: str) -> Path:
    return Path(root) / "retrieval_audit" / f"{video_id}{AUDIT_SUFFIX}"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return target


def _probe_codec(video_path: Path, *, ffprobe_bin: str = "ffprobe") -> str:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else ""


def _usage_fields(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(value, int)
        }
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            result[name] = value
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if isinstance(cached, int):
        result["cached_tokens"] = cached
    return result


def _analyze_chunk(
    client: Any,
    media: Path,
    chunk: Any,
    config: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
    request_hook: Callable[[str, Path, int], None] | None = None,
) -> tuple[list[Any], dict[str, Any], float, list[dict[str, Any]]]:
    """Request one chunk with the frozen retry plan; return parsed segments.

    Mirrors ``HighlightRetrievalPipeline._analyze_chunk`` plus the recovery plan
    implemented in ``real_provider._run_retrieval_with_recovery`` so that a
    successful primary request is byte-identical to the promoted execution.
    """

    from aic_video_highlight.retrieval.prompt_builder import build_prompt
    from aic_video_highlight.retrieval.response_parser import (
        ResponseParseError,
        TruncatedResponseError,
        parse_highlight_response,
    )

    prompt = build_prompt(
        config.prompt_version, chunk.duration_sec, config.max_segments_per_chunk
    )
    attempts: list[dict[str, Any]] = []

    def _request(max_new_tokens: int, temperature: float, label: str):
        if request_hook is not None:
            request_hook(label, media, chunk.index)
        request_started = clock()
        response = client.analyze_video(
            media,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            coarse_fps=config.coarse_fps,
            enable_thinking=config.enable_thinking,
        )
        latency = clock() - request_started
        record = {
            "chunk_index": chunk.index,
            "attempt": label,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "finish_reason": response.finish_reason,
            "latency_sec": round(latency, 6),
            **_usage_fields(response),
        }
        attempts.append(record)
        return response, latency

    plan: list[tuple[int, float, str]] = [
        (config.max_new_tokens, config.temperature, "primary")
    ]
    last_error: Exception | None = None
    response = None
    parse_time = 0.0
    local_segments = None
    while plan:
        max_new_tokens, temperature, label = plan.pop(0)
        response, _ = _request(max_new_tokens, temperature, label)
        parse_started = clock()
        try:
            if response.finish_reason == "length":
                raise TruncatedResponseError(
                    "model output was truncated (finish_reason=length); "
                    "refusing to treat partial JSON as a formal result"
                )
            local_segments = parse_highlight_response(
                response.content,
                chunk_duration_sec=chunk.duration_sec,
                source_chunk=chunk.index,
            )
            parse_time += clock() - parse_started
            last_error = None
            break
        except TruncatedResponseError as exc:
            parse_time += clock() - parse_started
            last_error = exc
            if label == "primary":
                plan = [
                    (RECOVERY_MAX_NEW_TOKENS, config.temperature, "truncation_retry_512"),
                    (2 * RECOVERY_MAX_NEW_TOKENS, config.temperature, "truncation_retry_1024"),
                ] + plan
            elif label != "truncation_retry_1024":
                plan = [
                    (2 * RECOVERY_MAX_NEW_TOKENS, config.temperature, "truncation_retry_1024")
                ] + plan
        except ResponseParseError as exc:
            parse_time += clock() - parse_started
            last_error = exc
            if label == "primary":
                plan = [
                    (config.max_new_tokens, RECOVERY_TEMPERATURE, "parse_retry_temp"),
                    (RECOVERY_MAX_NEW_TOKENS, RECOVERY_TEMPERATURE, "parse_retry_512"),
                ] + plan
    if local_segments is None or response is None:
        assert last_error is not None
        raise last_error

    record = {
        "chunk_index": chunk.index,
        "chunk_start_sec": chunk.start_sec,
        "chunk_end_sec": chunk.end_sec,
        "raw_response": response.content,
        "finish_reason": response.finish_reason,
        "request_latency_sec": sum(float(item["latency_sec"]) for item in attempts),
        "final_request_latency_sec": float(attempts[-1]["latency_sec"]),
        "parse_success": True,
        "parse_error": None,
        "parsed_segments": [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "score": item.score,
                "reason": item.reason,
                "source_chunk": item.source_chunk,
            }
            for item in local_segments
        ],
        "recovery_attempts": [
            attempt for attempt in attempts if attempt["attempt"] != "primary"
        ],
    }
    return local_segments, record, parse_time, attempts


def _render_job(
    source: Path,
    chunk: Any,
    output: Path,
    *,
    ffmpeg_bin: str,
    renderer: Callable[..., None],
    clock: Callable[[], float],
) -> tuple[Path, float]:
    started = clock()
    renderer(source, chunk, output, ffmpeg_bin=ffmpeg_bin)
    return output, clock() - started


def execute_chunks(
    chunks: Sequence[Any],
    *,
    render: Callable[[int, Any], tuple[Path, float]],
    analyze: Callable[[int, Any, Path], Any],
    overlap_render: bool,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    """Run chunk rendering plus analysis with at most one look-ahead render.

    ``render``/``analyze`` are injected so the scheduling contract is unit
    testable without a GPU or ffmpeg.  Requests are still issued strictly in
    chunk order, one at a time; only the CPU-side render of chunk *N+1* overlaps
    the in-flight request for chunk *N*.
    """

    records: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-render")
    pending: Future | None = None
    try:
        for position, chunk in enumerate(chunks):
            if pending is not None:
                wait_started = clock()
                media, render_sec = pending.result()
                render_wait = clock() - wait_started
                pending = None
            else:
                media, render_sec = render(position, chunk)
                render_wait = 0.0
            if overlap_render and position + 1 < len(chunks):
                pending = executor.submit(render, position + 1, chunks[position + 1])
            value = analyze(position, chunk, media)
            records.append(
                {
                    "position": position,
                    "render_sec": float(render_sec),
                    "render_wait_sec": float(render_wait),
                    "value": value,
                }
            )
    finally:
        executor.shutdown(wait=True)
    return records


def run_retrieval_audited(
    entry: IndexEntry,
    *,
    dataset_root: str | Path,
    client: Any,
    settings: RetrievalSettings,
    overlap_render: bool = False,
    renderer: Callable[..., None] | None = None,
    audit_path: str | Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
    request_hook: Callable[[str, Path, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run frozen-semantics retrieval with per-chunk audit and optional overlap."""

    from aic_video_highlight.retrieval.pipeline import _render_chunk
    from aic_video_highlight.retrieval.video_chunker import build_chunks
    from aic_video_highlight.retrieval.video_metadata import probe_video

    if renderer is None:
        renderer = _render_chunk
    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        raise RealProviderError(f"video file missing: {video_path}")
    config = settings.config()
    ffprobe_bin = "ffprobe"
    ffmpeg_bin = "ffmpeg"

    started = clock()
    recovery_log: list[dict[str, Any]] = []
    compatibility_transcode_s = 0.0
    compatibility_transcode = False

    codec_name = _probe_codec(video_path, ffprobe_bin=ffprobe_bin)
    duration_sec = float(entry.duration_sec or 0.0)
    media_path = video_path
    compat_dir: tempfile.TemporaryDirectory | None = None
    if codec_name == "av1" and duration_sec <= config.chunk_seconds + 1e-6:
        from aic_video_highlight.retrieval.video_clip import extract_video_clip

        compat_dir = tempfile.TemporaryDirectory(
            prefix=".aic-highlight-retrieval-compat-", dir=video_path.parent
        )
        media_path = Path(compat_dir.name) / "chunk-00000.mp4"
        transcode_started = clock()
        extract_video_clip(video_path, media_path, start_sec=0.0, end_sec=duration_sec)
        compatibility_transcode_s = clock() - transcode_started
        compatibility_transcode = True

    try:
        probe_started = clock()
        meta = probe_video(media_path, ffprobe_bin=ffprobe_bin)
        probe_sec = clock() - probe_started
        chunks = build_chunks(
            meta.duration_sec,
            chunk_seconds=config.chunk_seconds,
            overlap_seconds=config.overlap_seconds,
        )

        candidates: list[Any] = []
        raw_chunk_outputs: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        chunk_audits: list[dict[str, Any]] = []
        extraction_time = 0.0
        inference_time = 0.0
        parsing_time = 0.0
        render_wait_total = 0.0

        if len(chunks) == 1:
            segments, record, parse_time, attempts = _analyze_chunk(
                client, meta.path, chunks[0], config, clock=clock, request_hook=request_hook
            )
            candidates.extend(segments)
            raw_chunk_outputs.append(record)
            raw_responses.append(record["raw_response"])
            inference_time += record["request_latency_sec"]
            parsing_time += parse_time
            chunk_audits.append(
                {
                    "chunk_index": chunks[0].index,
                    "chunk_start_sec": chunks[0].start_sec,
                    "chunk_end_sec": chunks[0].end_sec,
                    "render_sec": 0.0,
                    "render_wait_sec": 0.0,
                    "attempts": attempts,
                    "finish_reason": record["finish_reason"],
                    "parse_success": record["parse_success"],
                    "parse_error": record["parse_error"],
                    "raw_response": record["raw_response"],
                    "parsed_segments": record["parsed_segments"],
                }
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=".aic-highlight-retrieval-", dir=meta.path.parent
            ) as temp_dir:
                work = Path(temp_dir)

                def _render(position: int, chunk: Any) -> tuple[Path, float]:
                    return _render_job(
                        meta.path,
                        chunk,
                        work / f"chunk-{chunk.index:05d}.mp4",
                        ffmpeg_bin=ffmpeg_bin,
                        renderer=renderer,
                        clock=clock,
                    )

                def _analyze(position: int, chunk: Any, media: Path):
                    return _analyze_chunk(
                        client,
                        media,
                        chunk,
                        config,
                        clock=clock,
                        request_hook=request_hook,
                    )

                records = execute_chunks(
                    chunks,
                    render=_render,
                    analyze=_analyze,
                    overlap_render=overlap_render,
                    clock=clock,
                )
                for record in records:
                    chunk = chunks[record["position"]]
                    local_segments, raw_record, parse_time, attempts = record["value"]
                    from aic_video_highlight.retrieval.schemas import HighlightSegment

                    candidates.extend(
                        HighlightSegment(
                            start_sec=chunk.start_sec + item.start_sec,
                            end_sec=chunk.start_sec + item.end_sec,
                            score=item.score,
                            reason=item.reason,
                            source_chunk=item.source_chunk,
                        )
                        for item in local_segments
                    )
                    raw_chunk_outputs.append(raw_record)
                    raw_responses.append(raw_record["raw_response"])
                    inference_time += raw_record["request_latency_sec"]
                    parsing_time += parse_time
                    extraction_time += record["render_sec"]
                    render_wait_total += record["render_wait_sec"]
                    chunk_audits.append(
                        {
                            "chunk_index": chunk.index,
                            "chunk_start_sec": chunk.start_sec,
                            "chunk_end_sec": chunk.end_sec,
                            "render_sec": record["render_sec"],
                            "render_wait_sec": record["render_wait_sec"],
                            "attempts": attempts,
                            "finish_reason": raw_record["finish_reason"],
                            "parse_success": raw_record["parse_success"],
                            "parse_error": raw_record["parse_error"],
                            "raw_response": raw_record["raw_response"],
                            "parsed_segments": raw_record["parsed_segments"],
                        }
                    )
        for row in chunk_audits:
            for attempt in row["attempts"]:
                if attempt["attempt"] != "primary":
                    recovery_log.append({"trigger": "chunk_level_recovery", **attempt})

        merge_started = clock()
        from aic_video_highlight.retrieval.candidate_merger import merge_segments

        merged = merge_segments(candidates, tiou_threshold=config.merge_tiou_threshold)
        merging_time = clock() - merge_started
        total_time = clock() - started
        from aic_video_highlight.retrieval.schemas import HighlightRetrievalResult

        result = HighlightRetrievalResult(
            video_id=meta.video_id,
            duration_sec=meta.duration_sec,
            segments=merged,
            raw_responses=raw_responses,
            inference_time_sec=inference_time,
            candidate_segments=candidates,
            raw_chunk_outputs=raw_chunk_outputs,
            timing={
                "probe_sec": probe_sec,
                "chunk_extraction_sec": extraction_time,
                "model_inference_sec": inference_time,
                "parsing_sec": parsing_time,
                "merging_sec": merging_time,
                "total_sec": total_time,
            },
        )
        if compatibility_transcode_s:
            # The production path folds the AV1 compatibility transcode into the
            # chunk extraction bucket; keep the audit accounting identical.
            result.timing["chunk_extraction_sec"] = float(
                result.timing.get("chunk_extraction_sec", 0.0)
            ) + compatibility_transcode_s
            extraction_time = float(result.timing["chunk_extraction_sec"])
    finally:
        if compat_dir is not None:
            compat_dir.cleanup()

    wall_sec = clock() - started
    chunks_payload = [
        {
            "chunk_index": int(raw["chunk_index"]),
            "chunk_start_sec": float(raw["chunk_start_sec"]),
            "chunk_end_sec": float(raw["chunk_end_sec"]),
            "finish_reason": raw.get("finish_reason"),
            "parsed_segment_count": len(raw.get("parsed_segments", [])),
            "request_latency_sec": float(raw.get("request_latency_sec", 0.0)),
            "parse_success": bool(raw.get("parse_success", False)),
        }
        for raw in result.raw_chunk_outputs
    ]
    raw_candidates = [
        {
            "chunk_index": int(segment.source_chunk) if segment.source_chunk is not None else -1,
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "score": float(segment.score),
            "reason": str(segment.reason),
        }
        for segment in result.candidate_segments
    ]
    merged_payload = [
        {
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "score": float(segment.score),
            "reason": str(segment.reason),
        }
        for segment in result.segments
    ]
    from aic_video_highlight.retrieval.prompt_builder import build_prompt

    first_chunk = chunks_payload[0] if chunks_payload else {
        "chunk_start_sec": 0.0,
        "chunk_end_sec": 0.0,
    }
    prompt_text = build_prompt(
        config.prompt_version,
        max(0.0, float(first_chunk["chunk_end_sec"]) - float(first_chunk["chunk_start_sec"])),
        config.max_segments_per_chunk,
    )
    payload = {
        "schema": RETRIEVAL_SCHEMA,
        "video_id": entry.video_id,
        "relative_video_path": entry.relative_video_path,
        "source_sha256": entry.source_sha256,
        "duration_sec": float(result.duration_sec),
        "chunks": chunks_payload,
        "raw_candidates": raw_candidates,
        "merged_candidates": merged_payload,
        "timing": {**result.timing, "artifact_wall_sec": wall_sec},
        "provenance": {
            "qwen_model_id": QWEN_MODEL,
            "qwen_revision": QWEN_REVISION,
            "prompt_id": config.prompt_version,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "retrieval_config_sha256": _sha256_file(settings.config_path),
            "chunk_seconds": config.chunk_seconds,
            "overlap_seconds": config.overlap_seconds,
            "coarse_fps": config.coarse_fps,
            "merge_tiou_threshold": config.merge_tiou_threshold,
            "recovery_applied": bool(recovery_log),
            "recovery_log": recovery_log,
            "media_compatibility_transcode": compatibility_transcode,
            "media_compatibility_codec": codec_name or None,
            "media_compatibility_transcode_sec": compatibility_transcode_s,
        },
    }
    audit = {
        "schema": CHUNK_AUDIT_SCHEMA,
        "video_id": entry.video_id,
        "relative_video_path": entry.relative_video_path,
        "execution": {
            "overlap_render": bool(overlap_render),
            "model_request_concurrency": 1,
            "request_order": "chunk_index_ascending",
            "qwen_model_id": QWEN_MODEL,
            "qwen_revision": QWEN_REVISION,
            "prompt_id": config.prompt_version,
            "coarse_fps": config.coarse_fps,
            "chunk_seconds": config.chunk_seconds,
            "overlap_seconds": config.overlap_seconds,
            "temperature": config.temperature,
            "enable_thinking": config.enable_thinking,
            "max_new_tokens": config.max_new_tokens,
        },
        "video": {
            "duration_sec": float(result.duration_sec),
            "codec_name": codec_name or None,
            "chunk_count": len(chunk_audits),
            "single_chunk": len(chunks_payload) == 1,
            "artifact_wall_sec": wall_sec,
            "timing": dict(result.timing),
            "render_sec_total": extraction_time,
            "render_wait_sec_total": render_wait_total,
            "request_sec_total": inference_time,
            "compatibility_transcode_sec": compatibility_transcode_s,
        },
        "chunks": chunk_audits,
    }
    if audit_path is not None:
        atomic_write_json(audit_path, audit)
    return payload, audit


# ---------------------------------------------------------------------------
# comparison + metric helpers
# ---------------------------------------------------------------------------


def _canonical_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": float(item["start_sec"]),
            "end_sec": float(item["end_sec"]),
            "score": float(item["score"]),
            "reason": str(item.get("reason", "")),
            "source_chunk": item.get("source_chunk"),
        }
        for item in segments
    ]


def _chunk_map(audit: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(row["chunk_index"]): row for row in audit.get("chunks", [])}


def compare_audit_arms(
    left_root: str | Path,
    right_root: str | Path,
    video_ids: Sequence[str],
) -> dict[str, Any]:
    """Chunk-level and video-level equivalence between two audit arms."""

    left = Path(left_root)
    right = Path(right_root)
    chunk_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        left_audit = json.loads(chunk_audit_path(left, video_id).read_text(encoding="utf-8"))
        right_audit = json.loads(chunk_audit_path(right, video_id).read_text(encoding="utf-8"))
        left_chunks = _chunk_map(left_audit)
        right_chunks = _chunk_map(right_audit)
        if set(left_chunks) != set(right_chunks):
            raise RuntimeError(f"chunk membership mismatch for {video_id}")
        for index in sorted(left_chunks):
            a = left_chunks[index]
            b = right_chunks[index]
            exact = str(a["raw_response"]) == str(b["raw_response"])
            semantic = _canonical_segments(a["parsed_segments"]) == _canonical_segments(
                b["parsed_segments"]
            )
            chunk_rows.append(
                {
                    "video_id": video_id,
                    "chunk_index": index,
                    "raw_exact_match": bool(exact),
                    "semantic_match": bool(exact and semantic),
                    "parse_success_left": bool(a["parse_success"]),
                    "parse_success_right": bool(b["parse_success"]),
                }
            )
        left_artifact = json.loads(
            (left / "retrieval" / f"{video_id}.json").read_text(encoding="utf-8")
        )
        right_artifact = json.loads(
            (right / "retrieval" / f"{video_id}.json").read_text(encoding="utf-8")
        )
        merged_equal = _canonical_segments(
            left_artifact["merged_candidates"]
        ) == _canonical_segments(right_artifact["merged_candidates"])
        raw_equal = _canonical_segments(
            left_artifact["raw_candidates"]
        ) == _canonical_segments(right_artifact["raw_candidates"])
        video_rows.append(
            {
                "video_id": video_id,
                "chunk_count": len(left_chunks),
                "candidate_count_left": len(left_artifact["raw_candidates"]),
                "candidate_count_right": len(right_artifact["raw_candidates"]),
                "raw_candidates_equal": bool(raw_equal),
                "merged_candidates_equal": bool(merged_equal),
                "merged_count_left": len(left_artifact["merged_candidates"]),
                "merged_count_right": len(right_artifact["merged_candidates"]),
            }
        )
    total_chunks = len(chunk_rows)
    exact = sum(1 for row in chunk_rows if row["raw_exact_match"])
    semantic = sum(1 for row in chunk_rows if row["semantic_match"])
    videos_equal = sum(1 for row in video_rows if row["merged_candidates_equal"])
    return {
        "schema": COMPARE_SCHEMA,
        "left_root": str(left),
        "right_root": str(right),
        "video_count": len(video_rows),
        "chunk_count": total_chunks,
        "chunk_raw_exact_match_count": exact,
        "chunk_semantic_match_count": semantic,
        "chunk_raw_exact_match_rate": round(exact / total_chunks, 6) if total_chunks else None,
        "chunk_semantic_match_rate": round(semantic / total_chunks, 6) if total_chunks else None,
        "video_merged_match_count": videos_equal,
        "video_candidate_semantic_match_rate": (
            round(videos_equal / len(video_rows), 6) if video_rows else None
        ),
        "all_chunks_exact": exact == total_chunks,
        "all_videos_equal": videos_equal == len(video_rows),
        "chunks": chunk_rows,
        "videos": video_rows,
    }


def fetch_prometheus_metrics(url: str, *, timeout_sec: float = 10.0) -> dict[str, Any]:
    """Read ``/metrics`` and aggregate vLLM families into sums and maxima."""

    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        text = response.read().decode("utf-8", errors="replace")
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    maxima: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if not head or not value:
            continue
        name = head.split("{", 1)[0].strip()
        if name.endswith("_bucket"):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        sums[name] = sums.get(name, 0.0) + number
        counts[name] = counts.get(name, 0.0) + 1.0
        maxima[name] = max(maxima.get(name, number), number)
    return {
        "metric_sum": sums,
        "metric_sample_count": counts,
        "metric_max": maxima,
    }


def metrics_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left = before.get("metric_sum", {})
    right = after.get("metric_sum", {})
    delta: dict[str, float] = {}
    for key, value in right.items():
        delta[key] = float(value) - float(left.get(key, 0.0))
    return delta


def summarize_request_metrics(delta: Mapping[str, float]) -> dict[str, Any]:
    """Extract the per-family request timing decomposition from a metric delta."""

    def value(name: str) -> float | None:
        raw = delta.get(name)
        return None if raw is None else float(raw)

    def count(name: str) -> float | None:
        return value(name)

    result: dict[str, Any] = {}
    for label, name in (
        ("queue_time_sec", "vllm:request_queue_time_seconds_sum"),
        ("prefill_time_sec", "vllm:request_prefill_time_seconds_sum"),
        ("decode_time_sec", "vllm:request_decode_time_seconds_sum"),
        ("inference_time_sec", "vllm:request_inference_time_seconds_sum"),
        ("ttft_sec", "vllm:time_to_first_token_seconds_sum"),
        ("inter_token_sec", "vllm:inter_token_latency_seconds_sum"),
        ("time_per_output_token_sec", "vllm:request_time_per_output_token_seconds_sum"),
        ("e2e_request_latency_sec", "vllm:e2e_request_latency_seconds_sum"),
    ):
        total = value(name)
        finished = value(
            name.replace("_sum", "_count")
            if name.endswith("_sum")
            else name + "_count"
        )
        result[f"{label}_total"] = total
        result[f"{label}_count"] = finished
        result[f"{label}_mean"] = (
            total / finished if total is not None and finished else None
        )
    for label, name in (
        ("prompt_tokens", "vllm:prompt_tokens_total"),
        ("generation_tokens", "vllm:generation_tokens_total"),
        ("request_success", "vllm:request_success_total"),
        ("prefix_cache_hits", "vllm:prefix_cache_hits_total"),
        ("prefix_cache_queries", "vllm:prefix_cache_queries_total"),
        ("mm_cache_hits", "vllm:mm_cache_hits_total"),
        ("mm_cache_queries", "vllm:mm_cache_queries_total"),
        ("num_preemptions", "vllm:num_preemptions_total"),
        ("prompt_tokens_cached", "vllm:prompt_tokens_cached_total"),
    ):
        result[label] = value(name)
    result["request_prompt_tokens_mean"] = None
    prompt_total = value("vllm:request_prompt_tokens_sum")
    prompt_count = value("vllm:request_prompt_tokens_count")
    if prompt_total is not None and prompt_count:
        result["request_prompt_tokens_mean"] = prompt_total / prompt_count
    result["request_generation_tokens_mean"] = None
    generation_total = value("vllm:request_generation_tokens_sum")
    generation_count = value("vllm:request_generation_tokens_count")
    if generation_total is not None and generation_count:
        result["request_generation_tokens_mean"] = generation_total / generation_count
    result["request_prompt_tokens_total"] = prompt_total
    result["request_generation_tokens_total"] = generation_total
    result["raw_metric_keys"] = sorted(delta)
    return result


__all__ = [
    "ARM_SUMMARY_SCHEMA",
    "CHUNK_AUDIT_SCHEMA",
    "COMPARE_SCHEMA",
    "atomic_write_json",
    "chunk_audit_path",
    "compare_audit_arms",
    "execute_chunks",
    "fetch_prometheus_metrics",
    "metrics_delta",
    "run_retrieval_audited",
    "summarize_request_metrics",
]
