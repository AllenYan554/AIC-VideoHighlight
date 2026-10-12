"""Real upstream production stages for Stage 7 FTNet (GPU host only).

Three stages produce the artifacts consumed by ``provider_core``:

    retrieval : Qwen3.5-4B (vLLM) high-recall candidates per video
    detection : 2.0 fps whole-video grid decode + frozen RT-DETR r50vd
                detection candidates and encoder level-0 GAP features
    assemble  : pure CPU native/target assembly and safetensors materialization
                (implemented in ``pipeline.py`` on top of ``provider_core``)

Every stage writes one artifact per video and is resumable at video
granularity.  Nothing here modifies frozen component code; it calls the same
library entry points the frozen Stage 5/6 pipelines use.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aic_video_highlight.ftnet.materialize import VideoRef
from aic_video_highlight.retrieval.pipeline import (
    HighlightRetrievalConfig,
    HighlightRetrievalPipeline,
    _render_chunk,
    load_highlight_retrieval_config,
)
from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient

from .index import IndexEntry, resolve_annotation_dir, resolve_video_path
from .performance_decode import (
    WINDOWED_PREFIX,
    DecodeCounters,
    iter_decoded_batches,
)
from .provider_core import (
    ChunkSpan,
    MergedCandidate,
    RawCandidate,
    RetrievalContext,
)
from .sampling import GridSample, build_uniform_grid
from .youtube_highlights import load_mturk_clips

RETRIEVAL_SCHEMA = "aic.stage7.ftnet.retrieval-artifact/v1"
DETECTION_SCHEMA = "aic.stage7.ftnet.detection-artifact/v1"
GRID_SCHEMA = "aic.stage7.ftnet.grid/v1"

QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RTDETR_MODEL = "PekingU/rtdetr_r50vd"
RTDETR_REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"

_DEFAULT_RETRIEVAL_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "highlight_retrieval.yaml"
)


class RealProviderError(RuntimeError):
    """Raised when a production stage cannot complete for a video."""


# Operational recovery for deterministic Qwen output failures.  These values are
# used only to re-request a failed chunk; every attempt is recorded in the
# retrieval artifact provenance.  They never change the frozen prompt, chunking,
# merge threshold or the frozen parser.
RECOVERY_TEMPERATURE = 0.3
RECOVERY_MAX_NEW_TOKENS = 512


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RetrievalSettings:
    base_url: str = "http://127.0.0.1:8000/v1"
    config_path: Path = _DEFAULT_RETRIEVAL_CONFIG
    timeout_sec: float = 120.0

    def config(self) -> HighlightRetrievalConfig:
        return load_highlight_retrieval_config(self.config_path)


def retrieval_artifact_path(work_root: str | Path, video_id: str) -> Path:
    return Path(work_root) / "retrieval" / f"{video_id}.json"


def detection_artifact_path(work_root: str | Path, video_id: str) -> Path:
    return Path(work_root) / "detections" / f"{video_id}.npz"


def grid_artifact_path(work_root: str | Path, video_id: str) -> Path:
    return Path(work_root) / "grid" / f"{video_id}.npz"


def load_retrieval_context(path: str | Path) -> tuple[RetrievalContext, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != RETRIEVAL_SCHEMA:
        raise RealProviderError(f"unexpected retrieval artifact schema: {path}")
    retrieval = RetrievalContext(
        chunks=tuple(
            ChunkSpan(
                chunk_index=int(row["chunk_index"]),
                chunk_start_sec=float(row["chunk_start_sec"]),
                chunk_end_sec=float(row["chunk_end_sec"]),
            )
            for row in payload["chunks"]
        ),
        raw_candidates=tuple(
            RawCandidate(
                chunk_index=int(row["chunk_index"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                score=float(row["score"]),
            )
            for row in payload["raw_candidates"]
        ),
        merged_candidates=tuple(
            MergedCandidate(
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                score=float(row.get("score", 0.0)),
            )
            for row in payload["merged_candidates"]
        ),
    )
    return retrieval, payload


def load_grid(entry: IndexEntry, path: str | Path) -> GridSample:
    import numpy as np

    payload = np.load(Path(path), allow_pickle=False)
    sample = GridSample(
        frames=np.asarray(payload["frames"], dtype=np.int64),
        timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
        adjacency_mask=np.asarray(payload["adjacency"], dtype=bool),
        target_timestamps=np.asarray(payload["target_timestamps"], dtype=np.float64),
    )
    expected = entry.frame_count
    if sample.frames.shape[0] == 0:
        raise RealProviderError(f"empty sampling grid for {entry.video_id}")
    if sample.frames.max() >= int(expected or 0):
        raise RealProviderError(f"grid frame id exceeds frame_count for {entry.video_id}")
    return sample


def write_grid(path: str | Path, sample: GridSample) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        frames=sample.frames.astype(np.int64),
        timestamps=sample.timestamps.astype(np.float64),
        adjacency=sample.adjacency_mask.astype(bool),
        target_timestamps=sample.target_timestamps.astype(np.float64),
    )
    os.replace(temporary, target)


def build_grid_for_entry(entry: IndexEntry) -> GridSample:
    from aic_video_highlight.composition.frame_projection import CFR_FPS, VideoTiming

    if entry.frame_count is None or entry.fps is None or entry.timestamp_mode is None:
        raise RealProviderError(f"index entry is not probed: {entry.video_id}")
    pts = None
    if entry.timestamp_mode != CFR_FPS:
        if not entry.pts_timestamps:
            raise RealProviderError(f"PTS_TABLE entry lacks pts timestamps: {entry.video_id}")
        from fractions import Fraction

        pts = tuple(Fraction(value) for value in entry.pts_timestamps)
    timing = VideoTiming(
        video_id=entry.video_id,
        fps=float(entry.fps),
        frame_count=int(entry.frame_count),
        timestamp_mode=str(entry.timestamp_mode),
        pts_timestamps=pts,
    )
    return build_uniform_grid(timing)


def _run_retrieval_with_recovery(
    entry: IndexEntry,
    video_path: Path,
    client: QwenVLLMClient,
    config: HighlightRetrievalConfig,
) -> tuple[Any, list[dict[str, Any]]]:
    """Chunk-level recovery loop around the frozen retrieval pipeline.

    Used only after the frozen pipeline raises on a deterministic model-output
    failure.  Frozen building blocks (probe, chunker, chunk renderer, prompt
    builder, parser, merger) are reused unchanged; a failed chunk is re-requested
    once with a recovery parameter set and every attempt is journaled.
    """

    import tempfile

    from aic_video_highlight.retrieval.candidate_merger import merge_segments
    from aic_video_highlight.retrieval.prompt_builder import build_prompt
    from aic_video_highlight.retrieval.response_parser import (
        ResponseParseError,
        TruncatedResponseError,
        parse_highlight_response,
    )
    from aic_video_highlight.retrieval.schemas import (
        HighlightRetrievalResult,
        HighlightSegment,
    )
    from aic_video_highlight.retrieval.video_chunker import build_chunks
    from aic_video_highlight.retrieval.video_metadata import probe_video

    pipeline = HighlightRetrievalPipeline(client, config)
    started = time.perf_counter()
    meta = probe_video(video_path, ffprobe_bin=pipeline.ffprobe_bin)
    chunks = build_chunks(
        meta.duration_sec,
        chunk_seconds=config.chunk_seconds,
        overlap_seconds=config.overlap_seconds,
    )
    candidates: list[Any] = []
    raw_chunk_outputs: list[dict[str, Any]] = []
    raw_responses: list[str] = []
    inference_time = 0.0
    parsing_time = 0.0
    extraction_time = 0.0
    attempts_log: list[dict[str, Any]] = []

    def _analyze(media: Path, chunk: Any) -> tuple[list[Any], dict[str, Any], float]:
        from aic_video_highlight.retrieval.response_parser import (
            ResponseParseError,
            TruncatedResponseError,
            parse_highlight_response,
        )

        prompt = build_prompt(
            config.prompt_version, chunk.duration_sec, config.max_segments_per_chunk
        )
        chunk_attempts: list[dict[str, Any]] = []

        def _request(max_new_tokens: int, temperature: float, label: str):
            request_started = time.perf_counter()
            response = client.analyze_video(
                media,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                coarse_fps=config.coarse_fps,
                enable_thinking=config.enable_thinking,
            )
            latency = time.perf_counter() - request_started
            entry = {
                "chunk_index": chunk.index,
                "attempt": label,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "finish_reason": response.finish_reason,
                "latency_sec": round(latency, 3),
            }
            chunk_attempts.append(entry)
            attempts_log.append(entry)
            return response, latency

        plan: list[tuple[int, float, str]] = [
            (config.max_new_tokens, config.temperature, "primary")
        ]
        last_error: Exception | None = None
        response = None
        latency = 0.0
        local_segments = None
        parse_time = 0.0
        while plan:
            max_new_tokens, temperature, label = plan.pop(0)
            response, latency = _request(max_new_tokens, temperature, label)
            parse_started = time.perf_counter()
            try:
                local_segments = parse_highlight_response(
                    response.content,
                    chunk_duration_sec=chunk.duration_sec,
                    source_chunk=chunk.index,
                )
                parse_time += time.perf_counter() - parse_started
                last_error = None
                break
            except TruncatedResponseError as exc:
                parse_time += time.perf_counter() - parse_started
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
                parse_time += time.perf_counter() - parse_started
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
            "request_latency_sec": sum(
                float(attempt["latency_sec"]) for attempt in chunk_attempts
            ),
            "final_request_latency_sec": latency,
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
                attempt for attempt in chunk_attempts if attempt["attempt"] != "primary"
            ],
        }
        return local_segments, record, parse_time

    if len(chunks) == 1:
        local_segments, record, parse_time = _analyze(meta.path, chunks[0])
        candidates.extend(
            HighlightSegment(
                start_sec=chunks[0].start_sec + item.start_sec,
                end_sec=chunks[0].start_sec + item.end_sec,
                score=item.score,
                reason=item.reason,
                source_chunk=item.source_chunk,
            )
            for item in local_segments
        )
        raw_chunk_outputs.append(record)
        raw_responses.append(record["raw_response"])
        inference_time += record["request_latency_sec"]
        parsing_time += parse_time
    else:
        with tempfile.TemporaryDirectory(
            prefix=".aic-highlight-recovery-", dir=meta.path.parent
        ) as temp_dir:
            for chunk in chunks:
                chunk_path = Path(temp_dir) / f"chunk-{chunk.index:05d}.mp4"
                extraction_started = time.perf_counter()
                _render_chunk(meta.path, chunk, chunk_path, ffmpeg_bin=pipeline.ffmpeg_bin)
                extraction_time += time.perf_counter() - extraction_started
                local_segments, record, parse_time = _analyze(chunk_path, chunk)
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
                raw_chunk_outputs.append(record)
                raw_responses.append(record["raw_response"])
                inference_time += record["request_latency_sec"]
                parsing_time += parse_time
    merged = merge_segments(candidates, tiou_threshold=config.merge_tiou_threshold)
    result = HighlightRetrievalResult(
        video_id=meta.video_id,
        duration_sec=meta.duration_sec,
        segments=merged,
        raw_responses=raw_responses,
        inference_time_sec=inference_time,
        candidate_segments=candidates,
        raw_chunk_outputs=raw_chunk_outputs,
        timing={
            "probe_sec": 0.0,
            "chunk_extraction_sec": extraction_time,
            "model_inference_sec": inference_time,
            "parsing_sec": parsing_time,
            "merging_sec": 0.0,
            "total_sec": time.perf_counter() - started,
        },
    )
    return result, attempts_log


def run_retrieval_for_entry(
    entry: IndexEntry,
    *,
    dataset_root: str | Path,
    client: QwenVLLMClient,
    settings: RetrievalSettings,
    overlap_render: bool = False,
    audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen high-recall retrieval and serialize the candidate context.

    ``overlap_render`` enables the promoted execution optimization: the CPU-side
    ffmpeg render of chunk N+1 runs while the request for chunk N is in flight.
    Requests remain strictly serial, in chunk order, with concurrency 1, so the
    model input stream is unchanged.  The flag defaults to ``False`` and the
    default code path below is the untouched frozen implementation.

    When enabled (or when ``audit_path`` requests a per-chunk sidecar) the
    already-validated implementation in ``ftnet.qwen_execution`` is reused
    verbatim; this module does not carry a second scheduling algorithm.
    """

    if overlap_render or audit_path is not None:
        from .qwen_execution import run_retrieval_audited

        payload, _audit = run_retrieval_audited(
            entry,
            dataset_root=dataset_root,
            client=client,
            settings=settings,
            overlap_render=overlap_render,
            audit_path=audit_path,
        )
        return payload

    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        raise RealProviderError(f"video file missing: {video_path}")
    config = settings.config()
    pipeline = HighlightRetrievalPipeline(client, config)
    started = time.perf_counter()
    recovery_log: list[dict[str, Any]] = []
    compatibility_transcode_s = 0.0
    compatibility_transcode = False

    def _execute(media_path: Path):
        try:
            return pipeline.run(media_path)
        except Exception as exc:  # noqa: BLE001 - frozen parse recovery only
            from aic_video_highlight.retrieval.response_parser import ResponseParseError

            if not isinstance(exc, ResponseParseError):
                raise
            recovery_log.append(
                {
                    "initial_error": f"{type(exc).__name__}: {exc}",
                    "trigger": "frozen_pipeline_failed",
                }
            )
            recovered, attempts = _run_retrieval_with_recovery(
                entry, media_path, client, config
            )
            recovery_log.extend(attempts)
            return recovered

    codec_probe = subprocess.run(
        [
            "ffprobe",
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
    codec_name = codec_probe.stdout.strip().lower() if codec_probe.returncode == 0 else ""
    duration_sec = float(entry.duration_sec or 0.0)
    # vLLM/OpenCV on the benchmark host cannot decode AV1.  Multi-chunk videos
    # already pass through the frozen libx264 chunk renderer; use that exact
    # renderer for a single short AV1 clip as a compatibility adapter too.
    if codec_name == "av1" and duration_sec <= config.chunk_seconds + 1e-6:
        from aic_video_highlight.retrieval.video_clip import extract_video_clip

        with tempfile.TemporaryDirectory(
            prefix=".aic-highlight-retrieval-compat-", dir=video_path.parent
        ) as temp_dir:
            proxy = Path(temp_dir) / "chunk-00000.mp4"
            transcode_started = time.perf_counter()
            extract_video_clip(
                video_path,
                proxy,
                start_sec=0.0,
                end_sec=duration_sec,
            )
            compatibility_transcode_s = time.perf_counter() - transcode_started
            compatibility_transcode = True
            result = _execute(proxy)
    else:
        result = _execute(video_path)
    wall_sec = time.perf_counter() - started
    if compatibility_transcode_s:
        result.timing["chunk_extraction_sec"] = float(
            result.timing.get("chunk_extraction_sec", 0.0)
        ) + compatibility_transcode_s

    chunks = []
    for raw in result.raw_chunk_outputs:
        chunks.append(
            {
                "chunk_index": int(raw["chunk_index"]),
                "chunk_start_sec": float(raw["chunk_start_sec"]),
                "chunk_end_sec": float(raw["chunk_end_sec"]),
                "finish_reason": raw.get("finish_reason"),
                "parsed_segment_count": len(raw.get("parsed_segments", [])),
                "request_latency_sec": float(raw.get("request_latency_sec", 0.0)),
                "parse_success": bool(raw.get("parse_success", False)),
            }
        )
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
    merged = [
        {
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "score": float(segment.score),
            "reason": str(segment.reason),
        }
        for segment in result.segments
    ]
    config_sha256 = _sha256_file(settings.config_path)
    from aic_video_highlight.retrieval.prompt_builder import build_prompt

    first_chunk = chunks[0] if chunks else {"chunk_start_sec": 0.0, "chunk_end_sec": 0.0}
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
        "chunks": chunks,
        "raw_candidates": raw_candidates,
        "merged_candidates": merged,
        "timing": {**result.timing, "artifact_wall_sec": wall_sec},
        "provenance": {
            "qwen_model_id": QWEN_MODEL,
            "qwen_revision": QWEN_REVISION,
            "prompt_id": config.prompt_version,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "retrieval_config_sha256": config_sha256,
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
    return payload


def _decode_window(
    video_path: Path, frame_ids: Sequence[int]
) -> dict[int, np.ndarray]:
    from aic_video_highlight.localization.localization_runner import decode_needed_frames

    return decode_needed_frames(video_path, list(frame_ids))


def create_rtdetr_localizer(
    *,
    rtdetr_snapshot: str | Path,
    rtdetr_model_id: str = RTDETR_MODEL,
    device: str = "cuda",
):
    """Load the frozen RT-DETR runtime once for a detection stage.

    Keeping construction at this explicit seam prevents the per-video
    ``from_pretrained`` reload that dominated the original Full154 run while
    leaving the processor, weights, dtype and inference path unchanged.
    """

    from aic_video_highlight.localization.rt_detr_localizer import RTDetrLocalizer

    return RTDetrLocalizer(
        model_id=rtdetr_model_id,
        local_path=rtdetr_snapshot,
        device=device,
        torch_dtype="float32",
    )


def save_detection(
    path: str | Path,
    *,
    visual: np.ndarray,
    offsets: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    sample: GridSample,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(DETECTION_SCHEMA),
        visual=np.asarray(visual, dtype=np.float16),
        offsets=np.asarray(offsets, dtype=np.int32),
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        class_names=np.asarray(tuple(class_names)),
        frames=np.asarray(sample.frames, dtype=np.int64),
        timestamps=np.asarray(sample.timestamps, dtype=np.float64),
        adjacency=np.asarray(sample.adjacency_mask, dtype=bool),
        target_timestamps=np.asarray(sample.target_timestamps, dtype=np.float64),
    )
    os.replace(temporary, target)
    return target


def run_detection_for_entry(
    entry: IndexEntry,
    *,
    dataset_root: str | Path,
    work_root: str | Path,
    rtdetr_snapshot: str | Path,
    rtdetr_model_id: str = RTDETR_MODEL,
    device: str = "cuda",
    batch_size: int = 8,
    top_k: int = 100,
    decode_window: int = 64,
    decode_mode: str = WINDOWED_PREFIX,
    localizer: Any | None = None,
    model_load_s: float = 0.0,
    model_load_count: int = 0,
) -> dict[str, Any]:
    """Decode the 2.0 fps grid and run frozen RT-DETR detection + level-0 GAP."""

    import torch

    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        raise RealProviderError(f"video file missing: {video_path}")
    grid_started = time.perf_counter()
    sample = build_grid_for_entry(entry)
    grid_creation_s = time.perf_counter() - grid_started
    grid_write_started = time.perf_counter()
    write_grid(grid_artifact_path(work_root, entry.video_id), sample)
    grid_serialization_s = time.perf_counter() - grid_write_started
    frame_ids = sample.frames
    frame_count = int(frame_ids.shape[0])
    visual = np.empty((frame_count, 256), dtype=np.float16)
    candidate_boxes: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    candidate_labels: list[np.ndarray] = []
    offsets = np.zeros(frame_count + 1, dtype=np.int32)

    total_started = time.perf_counter()
    if localizer is None:
        load_started = time.perf_counter()
        localizer = create_rtdetr_localizer(
            rtdetr_snapshot=rtdetr_snapshot,
            rtdetr_model_id=rtdetr_model_id,
            device=device,
        )
        model_load_s = time.perf_counter() - load_started
        model_load_count = 1
    label_count = max(int(key) for key in localizer.id2label) + 1
    class_names = tuple(
        str(localizer.id2label.get(index, index)) for index in range(label_count)
    )
    width = int(entry.width or 0)
    height = int(entry.height or 0)
    if width <= 0 or height <= 0:
        raise RealProviderError(f"index entry lacks frame dimensions: {entry.video_id}")
    decode_counters = DecodeCounters()
    inference_s = 0.0
    processor_s = 0.0
    h2d_s = 0.0
    pure_forward_wall_s = 0.0
    pure_forward_cuda_s = 0.0
    gap_s = 0.0
    postprocess_s = 0.0
    d2h_s = 0.0
    batch_count = 0
    processed = 0
    total_candidates = 0
    batches = iter_decoded_batches(
        video_path,
        frame_ids,
        batch_size=batch_size,
        decode_window=decode_window,
        mode=decode_mode,
        counters=decode_counters,
    )
    for batch, images in batches:
        processor_started = time.perf_counter()
        inputs = localizer.processor(images=images, return_tensors="pt")
        processor_s += time.perf_counter() - processor_started
        h2d_started = time.perf_counter()
        pixel_values = inputs["pixel_values"].to(
            localizer.device, dtype=localizer.torch_dtype
        )
        if pixel_values.is_cuda:
            torch.cuda.synchronize(pixel_values.device)
        h2d_s += time.perf_counter() - h2d_started
        inference_started = time.perf_counter()
        with torch.inference_mode():
                    cuda_start = cuda_end = None
                    if pixel_values.is_cuda:
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                    outputs = localizer.model(
                        pixel_values=pixel_values, output_hidden_states=True
                    )
                    if cuda_end is not None:
                        cuda_end.record()
                        torch.cuda.synchronize(pixel_values.device)
                        pure_forward_cuda_s += float(cuda_start.elapsed_time(cuda_end)) / 1000.0
                    pure_forward_wall_s += time.perf_counter() - inference_started
                    encoder = outputs.encoder_last_hidden_state
                    level0 = encoder[0] if isinstance(encoder, (tuple, list)) else encoder
                    if tuple(level0.shape) != (len(batch), 256, 80, 80):
                        raise RealProviderError(
                            f"unexpected encoder level-0 shape {tuple(level0.shape)} for "
                            f"{entry.video_id}"
                        )
                    gap_started = time.perf_counter()
                    gap_tensor = level0.mean(dim=(2, 3)).to(torch.float16)
                    if gap_tensor.is_cuda:
                        torch.cuda.synchronize(gap_tensor.device)
                    gap_s += time.perf_counter() - gap_started
                    d2h_started = time.perf_counter()
                    gap = gap_tensor.cpu().numpy()
                    d2h_s += time.perf_counter() - d2h_started
                    postprocess_started = time.perf_counter()
                    logits = outputs.logits
                    pred_boxes = outputs.pred_boxes
                    scores_all = logits.sigmoid()
                    best_scores, best_labels = scores_all.max(dim=-1)
                    keep = min(top_k, best_scores.shape[-1])
                    top_scores, top_indices = best_scores.topk(keep, dim=-1)
                    if pixel_values.is_cuda:
                        torch.cuda.synchronize(pixel_values.device)
                    postprocess_s += time.perf_counter() - postprocess_started
                    for batch_index, frame in enumerate(batch):
                        frame_postprocess_started = time.perf_counter()
                        visual[processed] = gap[batch_index]
                        boxes_normalized = pred_boxes[batch_index][top_indices[batch_index]]
                        cx = boxes_normalized[:, 0]
                        cy = boxes_normalized[:, 1]
                        bw = boxes_normalized[:, 2]
                        bh = boxes_normalized[:, 3]
                        x1 = (cx - bw / 2.0) * width
                        y1 = (cy - bh / 2.0) * height
                        x2 = (cx + bw / 2.0) * width
                        y2 = (cy + bh / 2.0) * height
                        boxes_tensor = torch.stack([x1, y1, x2, y2], dim=-1)
                        if boxes_tensor.is_cuda:
                            torch.cuda.synchronize(boxes_tensor.device)
                        postprocess_s += time.perf_counter() - frame_postprocess_started
                        d2h_started = time.perf_counter()
                        boxes = boxes_tensor.cpu().numpy()
                        candidate_boxes.append(boxes.astype(np.float32))
                        candidate_scores.append(
                            top_scores[batch_index].cpu().numpy().astype(np.float32)
                        )
                        candidate_labels.append(
                            best_labels[batch_index][top_indices[batch_index]]
                            .cpu()
                            .numpy()
                            .astype(np.int32)
                        )
                        d2h_s += time.perf_counter() - d2h_started
                        total_candidates += len(top_scores[batch_index])
                        processed += 1
                        offsets[processed] = total_candidates
        inference_s += time.perf_counter() - inference_started
        batch_count += 1
    if processed != frame_count:
        raise RealProviderError(f"detection processed {processed}/{frame_count} frames")
    if not np.isfinite(visual).all():
        raise RealProviderError(f"non-finite GAP features for {entry.video_id}")

    boxes_array = (
        np.concatenate(candidate_boxes, axis=0)
        if candidate_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )
    scores_array = (
        np.concatenate(candidate_scores, axis=0)
        if candidate_scores
        else np.zeros((0,), dtype=np.float32)
    )
    labels_array = (
        np.concatenate(candidate_labels, axis=0)
        if candidate_labels
        else np.zeros((0,), dtype=np.int32)
    )
    if int(offsets[-1]) != len(scores_array):
        raise RealProviderError("candidate offsets do not match candidate rows")
    path = detection_artifact_path(work_root, entry.video_id)
    serialize_started = time.perf_counter()
    save_detection(
        path,
        visual=visual,
        offsets=offsets,
        boxes=boxes_array,
        scores=scores_array,
        labels=labels_array,
        class_names=class_names,
        sample=sample,
    )
    serialize_s = time.perf_counter() - serialize_started
    return {
        "video_id": entry.video_id,
        "frames": frame_count,
        "candidates": int(len(scores_array)),
        "wall_sec": time.perf_counter() - total_started,
        "decode_s": decode_counters.decode_wall_seconds,
        "rtdetr_load_s": float(model_load_s),
        "rtdetr_inference_s": inference_s,
        "serialize_s": serialize_s,
        "model_load_count": int(model_load_count),
        "batch_count": batch_count,
        "decode_mode": decode_mode,
        "profile": {
            "grid_creation_seconds": grid_creation_s,
            "grid_serialization_seconds": grid_serialization_s,
            **decode_counters.to_dict(),
            "image_processor_seconds": processor_s,
            "h2d_seconds": h2d_s,
            "rtdetr_pure_forward_wall_seconds": pure_forward_wall_s,
            "rtdetr_pure_forward_cuda_seconds": pure_forward_cuda_s,
            "gap_seconds": gap_s,
            "postprocess_seconds": postprocess_s,
            "d2h_seconds": d2h_s,
            "rtdetr_forward_count": batch_count,
            "processed_frame_count": processed,
            "npz_serialization_compression_write_seconds": serialize_s,
        },
        "artifact": str(path),
    }


def load_detection(path: str | Path):
    payload = np.load(Path(path), allow_pickle=False)
    from .provider_core import DetectionArtifacts

    detections = DetectionArtifacts(
        offsets=np.asarray(payload["offsets"], dtype=np.int32),
        boxes=np.asarray(payload["boxes"], dtype=np.float32),
        scores=np.asarray(payload["scores"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int32),
        class_names=tuple(str(name) for name in payload["class_names"]),
    )
    sample = GridSample(
        frames=np.asarray(payload["frames"], dtype=np.int64),
        timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
        adjacency_mask=np.asarray(payload["adjacency"], dtype=bool),
        target_timestamps=np.asarray(payload["target_timestamps"], dtype=np.float64),
    )
    visual = np.asarray(payload["visual"], dtype=np.float16)
    return detections, sample, visual


def video_ref_from_entry(entry: IndexEntry) -> VideoRef:
    return VideoRef(
        canonical_video_id=entry.video_id,
        realized_video_id=entry.realized_video_id,
        category=entry.category,
        stage7_split=entry.split,
        relative_video_path=entry.relative_video_path,
        source_sha256=entry.source_sha256,
    )


class RealUpstreamProvider:
    """UpstreamProvider backed by staged retrieval/detection artifacts.

    The provider is read-only over the work root; production stages are driven
    by ``pipeline.py``.  ``produce`` fails closed if a video's artifacts are
    missing or inconsistent.
    """

    def __init__(
        self,
        entries: Iterable[IndexEntry],
        *,
        dataset_root: str | Path,
        work_root: str | Path,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.work_root = Path(work_root).expanduser().resolve()
        self.entries = {entry.video_id: entry for entry in entries}
        self._order = {split: [] for split in ("TRAIN", "VALIDATION", "CALIBRATION")}
        for entry in entries:
            self._order[entry.split].append(entry.video_id)

    def list_videos(self, stage7_split: str) -> list[VideoRef]:
        return [video_ref_from_entry(self.entries[vid]) for vid in self._order[stage7_split]]

    def has_artifacts(self, video_id: str) -> bool:
        return (
            retrieval_artifact_path(self.work_root, video_id).is_file()
            and detection_artifact_path(self.work_root, video_id).is_file()
        )

    def produce(self, ref: VideoRef):
        from .provider_core import VideoArtifacts, build_video_upstream

        entry = self.entries.get(ref.canonical_video_id)
        if entry is None:
            raise RealProviderError(f"unknown video: {ref.canonical_video_id}")
        retrieval_path = retrieval_artifact_path(self.work_root, entry.video_id)
        detection_path = detection_artifact_path(self.work_root, entry.video_id)
        if not retrieval_path.is_file() or not detection_path.is_file():
            raise RealProviderError(
                f"staged artifacts missing for {entry.video_id}; run the production stages first"
            )
        retrieval, retrieval_payload = load_retrieval_context(retrieval_path)
        detections, sample, visual = load_detection(detection_path)
        clips = load_mturk_clips(resolve_annotation_dir(self.dataset_root, entry))
        if entry.frame_count is not None and sample.frames.shape[0] == 0:
            raise RealProviderError(f"empty grid for {entry.video_id}")
        artifacts = VideoArtifacts(
            ref=video_ref_from_entry(entry),
            width=int(entry.width or 0),
            height=int(entry.height or 0),
            timestamps=sample.timestamps,
            source_frame_ids=sample.frames,
            adjacency_mask=sample.adjacency_mask,
            visual=visual,
            retrieval=retrieval,
            detections=detections,
            mturk_clips=clips,
            retrieval_metadata={
                "retrieval_provenance": retrieval_payload.get("provenance", {}),
                "retrieval_timing": retrieval_payload.get("timing", {}),
            },
        )
        return build_video_upstream(artifacts)


__all__ = [
    "DETECTION_SCHEMA",
    "GRID_SCHEMA",
    "QWEN_MODEL",
    "QWEN_REVISION",
    "RTDETR_MODEL",
    "RTDETR_REVISION",
    "RETRIEVAL_SCHEMA",
    "RealProviderError",
    "RealUpstreamProvider",
    "RetrievalSettings",
    "build_grid_for_entry",
    "create_rtdetr_localizer",
    "detection_artifact_path",
    "grid_artifact_path",
    "load_detection",
    "load_grid",
    "load_retrieval_context",
    "retrieval_artifact_path",
    "run_detection_for_entry",
    "run_retrieval_for_entry",
    "save_detection",
    "video_ref_from_entry",
    "write_grid",
]
