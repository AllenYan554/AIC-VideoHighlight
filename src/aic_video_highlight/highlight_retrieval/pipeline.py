"""End-to-end highlight candidate retrieval orchestration."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .candidate_merger import merge_segments
from .prompt_builder import build_high_recall_prompt
from .qwen_vllm_client import QwenVLLMClient
from .response_parser import TruncatedResponseError, parse_highlight_response
from .schemas import HighlightRetrievalResult, HighlightSegment, VideoChunk
from .video_chunker import build_chunks
from .video_clip import extract_video_clip
from .video_metadata import probe_video


class HighlightRetrievalPipelineError(RuntimeError):
    pass


def parse_saved_raw_outputs(
    raw_chunk_outputs: list[dict],
    *,
    tiou_threshold: float,
) -> tuple[list[HighlightSegment], list[HighlightSegment], list[dict], dict[str, float]]:
    """Re-run parser and merge stages from persisted responses without model calls."""
    candidates: list[HighlightSegment] = []
    normalized_outputs: list[dict] = []
    parsing_started_at = time.perf_counter()
    for saved in sorted(raw_chunk_outputs, key=lambda item: item["chunk_index"]):
        record = dict(saved)
        chunk_start = float(record["chunk_start_sec"])
        chunk_end = float(record["chunk_end_sec"])
        if record.get("finish_reason") == "length":
            record["parse_success"] = False
            record["parse_error"] = (
                "TruncatedResponseError: model output was truncated "
                "(finish_reason=length); refusing to treat partial JSON as a result"
            )
            normalized_outputs.append(record)
            raise TruncatedResponseError(record["parse_error"])
        try:
            local_segments = parse_highlight_response(
                str(record["raw_response"]),
                chunk_duration_sec=chunk_end - chunk_start,
                source_chunk=int(record["chunk_index"]),
            )
        except Exception as exc:
            record["parse_success"] = False
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
            normalized_outputs.append(record)
            raise
        record["parse_success"] = True
        record["parse_error"] = None
        record["parsed_segments"] = [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "score": item.score,
                "reason": item.reason,
                "source_chunk": item.source_chunk,
            }
            for item in local_segments
        ]
        normalized_outputs.append(record)
        candidates.extend(
            HighlightSegment(
                start_sec=chunk_start + item.start_sec,
                end_sec=chunk_start + item.end_sec,
                score=item.score,
                reason=item.reason,
                source_chunk=item.source_chunk,
            )
            for item in local_segments
        )
    parsing_time_sec = time.perf_counter() - parsing_started_at
    merging_started_at = time.perf_counter()
    merged = merge_segments(candidates, tiou_threshold=tiou_threshold)
    merging_time_sec = time.perf_counter() - merging_started_at
    return (
        candidates,
        merged,
        normalized_outputs,
        {"parsing_sec": parsing_time_sec, "merging_sec": merging_time_sec},
    )


@dataclass(frozen=True, slots=True)
class HighlightRetrievalConfig:
    model: str = "Qwen/Qwen3.5-4B"
    chunk_seconds: float = 30.0
    overlap_seconds: float = 5.0
    coarse_fps: float = 2.0
    max_segments_per_chunk: int = 5
    max_new_tokens: int = 256
    temperature: float = 0.0
    merge_tiou_threshold: float = 0.5
    request_timeout_sec: float = 120.0
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than zero")
        if self.overlap_seconds < 0 or self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be in [0, chunk_seconds)")
        if self.coarse_fps <= 0:
            raise ValueError("coarse_fps must be greater than zero")
        if self.max_segments_per_chunk <= 0 or self.max_new_tokens <= 0:
            raise ValueError("segment and token limits must be greater than zero")
        if not 0 <= self.merge_tiou_threshold <= 1:
            raise ValueError("merge_tiou_threshold must be in [0, 1]")
        if self.request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be greater than zero")


def load_highlight_retrieval_config(path: str | Path) -> HighlightRetrievalConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("highlight retrieval config must be a YAML mapping")
    allowed = {item.name for item in fields(HighlightRetrievalConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown highlight retrieval config keys: {', '.join(unknown)}")
    return HighlightRetrievalConfig(**payload)


def _render_chunk(
    source: Path,
    chunk: VideoChunk,
    output: Path,
    *,
    ffmpeg_bin: str,
) -> None:
    """Create one temporary MP4 clip; no persistent frame images are generated."""
    try:
        extract_video_clip(
            source,
            output,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            ffmpeg_bin=ffmpeg_bin,
        )
    except Exception as exc:
        raise HighlightRetrievalPipelineError(
            f"failed to create temporary chunk {chunk.index}: {exc}"
        ) from exc


class HighlightRetrievalPipeline:
    def __init__(
        self,
        client: QwenVLLMClient,
        config: HighlightRetrievalConfig,
        *,
        ffprobe_bin: str = "ffprobe",
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.client = client
        self.config = config
        self.ffprobe_bin = ffprobe_bin
        self.ffmpeg_bin = ffmpeg_bin

    def _analyze_chunk(
        self,
        media: Path,
        chunk: VideoChunk,
        raw_output_sink: Callable[[dict], None] | None = None,
    ) -> tuple[list[HighlightSegment], dict, float]:
        prompt = build_high_recall_prompt(chunk.duration_sec, self.config.max_segments_per_chunk)
        request_started_at = time.perf_counter()
        model_response = self.client.analyze_video(
            media,
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            coarse_fps=self.config.coarse_fps,
            enable_thinking=self.config.enable_thinking,
        )
        request_latency_sec = time.perf_counter() - request_started_at
        raw_response = model_response.content
        finish_reason = model_response.finish_reason
        raw_record = {
            "chunk_index": chunk.index,
            "chunk_start_sec": chunk.start_sec,
            "chunk_end_sec": chunk.end_sec,
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "request_latency_sec": request_latency_sec,
            "parse_success": None,
            "parse_error": None,
            "parsed_segments": [],
        }
        if raw_output_sink is not None:
            raw_output_sink(raw_record)

        if finish_reason == "length":
            truncation_error = TruncatedResponseError(
                "model output was truncated (finish_reason=length); "
                "refusing to treat partial JSON as a formal result"
            )
            raw_record["parse_success"] = False
            raw_record["parse_error"] = f"{type(truncation_error).__name__}: {truncation_error}"
            if raw_output_sink is not None:
                raw_output_sink(raw_record)
            raise truncation_error

        parse_started_at = time.perf_counter()
        try:
            local_segments = parse_highlight_response(
                raw_response,
                chunk_duration_sec=chunk.duration_sec,
                source_chunk=chunk.index,
            )
        except Exception as exc:
            raw_record["parse_success"] = False
            raw_record["parse_error"] = f"{type(exc).__name__}: {exc}"
            if raw_output_sink is not None:
                raw_output_sink(raw_record)
            raise
        parse_time_sec = time.perf_counter() - parse_started_at
        raw_record["parse_success"] = True
        raw_record["parsed_segments"] = [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "score": item.score,
                "reason": item.reason,
                "source_chunk": item.source_chunk,
            }
            for item in local_segments
        ]
        if raw_output_sink is not None:
            raw_output_sink(raw_record)
        global_segments = [
            HighlightSegment(
                start_sec=chunk.start_sec + item.start_sec,
                end_sec=chunk.start_sec + item.end_sec,
                score=item.score,
                reason=item.reason,
                source_chunk=item.source_chunk,
            )
            for item in local_segments
        ]
        return global_segments, raw_record, parse_time_sec

    def run(
        self,
        video_path: str | Path,
        *,
        raw_output_sink: Callable[[dict], None] | None = None,
    ) -> HighlightRetrievalResult:
        started_at = time.perf_counter()
        probe_started_at = time.perf_counter()
        meta = probe_video(video_path, ffprobe_bin=self.ffprobe_bin)
        probe_time_sec = time.perf_counter() - probe_started_at
        chunks = build_chunks(
            meta.duration_sec,
            chunk_seconds=self.config.chunk_seconds,
            overlap_seconds=self.config.overlap_seconds,
        )
        candidates: list[HighlightSegment] = []
        raw_responses: list[str] = []
        raw_chunk_outputs: list[dict] = []
        parsing_time_sec = 0.0
        model_inference_time_sec = 0.0
        chunk_extraction_time_sec = 0.0

        if len(chunks) == 1:
            segments, raw_record, parse_time_sec = self._analyze_chunk(
                meta.path, chunks[0], raw_output_sink
            )
            candidates.extend(segments)
            raw_responses.append(raw_record["raw_response"])
            raw_chunk_outputs.append(raw_record)
            parsing_time_sec += parse_time_sec
            model_inference_time_sec += raw_record["request_latency_sec"]
        else:
            # The temporary directory is adjacent to the source video so it stays
            # inside vLLM's configured --allowed-local-media-path tree.
            with tempfile.TemporaryDirectory(
                prefix=".aic-highlight-retrieval-", dir=meta.path.parent
            ) as temp_dir:
                for chunk in chunks:
                    chunk_path = Path(temp_dir) / f"chunk-{chunk.index:05d}.mp4"
                    extraction_started_at = time.perf_counter()
                    _render_chunk(meta.path, chunk, chunk_path, ffmpeg_bin=self.ffmpeg_bin)
                    chunk_extraction_time_sec += time.perf_counter() - extraction_started_at
                    segments, raw_record, parse_time_sec = self._analyze_chunk(
                        chunk_path, chunk, raw_output_sink
                    )
                    candidates.extend(segments)
                    raw_responses.append(raw_record["raw_response"])
                    raw_chunk_outputs.append(raw_record)
                    parsing_time_sec += parse_time_sec
                    model_inference_time_sec += raw_record["request_latency_sec"]

        merge_started_at = time.perf_counter()
        merged = merge_segments(candidates, tiou_threshold=self.config.merge_tiou_threshold)
        merging_time_sec = time.perf_counter() - merge_started_at
        total_time_sec = time.perf_counter() - started_at
        return HighlightRetrievalResult(
            video_id=meta.video_id,
            duration_sec=meta.duration_sec,
            segments=merged,
            raw_responses=raw_responses,
            inference_time_sec=model_inference_time_sec,
            candidate_segments=candidates,
            raw_chunk_outputs=raw_chunk_outputs,
            timing={
                "probe_sec": probe_time_sec,
                "chunk_extraction_sec": chunk_extraction_time_sec,
                "model_inference_sec": model_inference_time_sec,
                "parsing_sec": parsing_time_sec,
                "merging_sec": merging_time_sec,
                "total_sec": total_time_sec,
            },
        )
