"""End-to-end highlight candidate retrieval orchestration."""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .candidate_merger import merge_segments
from .prompt_builder import build_high_recall_prompt
from .qwen_vllm_client import QwenVLLMClient
from .response_parser import parse_highlight_response
from .schemas import HighlightRetrievalResult, HighlightSegment, VideoChunk
from .video_chunker import build_chunks
from .video_metadata import probe_video


class HighlightRetrievalPipelineError(RuntimeError):
    pass


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
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-y",
        "-ss",
        f"{chunk.start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{chunk.duration_sec:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise HighlightRetrievalPipelineError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise HighlightRetrievalPipelineError(
            f"failed to create temporary chunk {chunk.index}: {detail}"
        )


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

    def _analyze_chunk(self, media: Path, chunk: VideoChunk) -> tuple[list[HighlightSegment], str]:
        prompt = build_high_recall_prompt(chunk.duration_sec, self.config.max_segments_per_chunk)
        raw_response = self.client.analyze_video(
            media,
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            coarse_fps=self.config.coarse_fps,
        )
        local_segments = parse_highlight_response(
            raw_response,
            chunk_duration_sec=chunk.duration_sec,
            source_chunk=chunk.index,
        )
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
        return global_segments, raw_response

    def run(self, video_path: str | Path) -> HighlightRetrievalResult:
        started_at = time.perf_counter()
        meta = probe_video(video_path, ffprobe_bin=self.ffprobe_bin)
        chunks = build_chunks(
            meta.duration_sec,
            chunk_seconds=self.config.chunk_seconds,
            overlap_seconds=self.config.overlap_seconds,
        )
        candidates: list[HighlightSegment] = []
        raw_responses: list[str] = []

        if len(chunks) == 1:
            segments, response = self._analyze_chunk(meta.path, chunks[0])
            candidates.extend(segments)
            raw_responses.append(response)
        else:
            # The temporary directory is adjacent to the source video so it stays
            # inside vLLM's configured --allowed-local-media-path tree.
            with tempfile.TemporaryDirectory(
                prefix=".aic-highlight-retrieval-", dir=meta.path.parent
            ) as temp_dir:
                for chunk in chunks:
                    chunk_path = Path(temp_dir) / f"chunk-{chunk.index:05d}.mp4"
                    _render_chunk(meta.path, chunk, chunk_path, ffmpeg_bin=self.ffmpeg_bin)
                    segments, response = self._analyze_chunk(chunk_path, chunk)
                    candidates.extend(segments)
                    raw_responses.append(response)

        merged = merge_segments(candidates, tiou_threshold=self.config.merge_tiou_threshold)
        return HighlightRetrievalResult(
            video_id=meta.video_id,
            duration_sec=meta.duration_sec,
            segments=merged,
            raw_responses=raw_responses,
            inference_time_sec=time.perf_counter() - started_at,
        )
