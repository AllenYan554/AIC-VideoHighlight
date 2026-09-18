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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aic_video_highlight.ftnet.materialize import VideoRef
from aic_video_highlight.retrieval.pipeline import (
    HighlightRetrievalConfig,
    HighlightRetrievalPipeline,
    load_highlight_retrieval_config,
)
from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient

from .index import IndexEntry, resolve_annotation_dir, resolve_video_path
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


def run_retrieval_for_entry(
    entry: IndexEntry,
    *,
    dataset_root: str | Path,
    client: QwenVLLMClient,
    settings: RetrievalSettings,
) -> dict[str, Any]:
    """Run the frozen high-recall retrieval and serialize the candidate context."""

    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        raise RealProviderError(f"video file missing: {video_path}")
    config = settings.config()
    pipeline = HighlightRetrievalPipeline(client, config)
    started = time.perf_counter()
    result = pipeline.run(video_path)
    wall_sec = time.perf_counter() - started

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
            }
        )
    raw_candidates = [
        {
            "chunk_index": int(segment.source_chunk) if segment.source_chunk is not None else -1,
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "score": float(segment.score),
        }
        for segment in result.candidate_segments
    ]
    merged = [
        {
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "score": float(segment.score),
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
        },
    }
    return payload


def _decode_window(
    video_path: Path, frame_ids: Sequence[int]
) -> dict[int, np.ndarray]:
    from aic_video_highlight.localization.localization_runner import decode_needed_frames

    return decode_needed_frames(video_path, list(frame_ids))


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
) -> dict[str, Any]:
    """Decode the 2.0 fps grid and run frozen RT-DETR detection + level-0 GAP."""

    import torch

    from aic_video_highlight.localization.rt_detr_localizer import RTDetrLocalizer

    video_path = resolve_video_path(dataset_root, entry)
    if not video_path.is_file():
        raise RealProviderError(f"video file missing: {video_path}")
    sample = build_grid_for_entry(entry)
    write_grid(grid_artifact_path(work_root, entry.video_id), sample)
    frame_ids = sample.frames
    frame_count = int(frame_ids.shape[0])
    visual = np.empty((frame_count, 256), dtype=np.float16)
    candidate_boxes: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    candidate_labels: list[np.ndarray] = []
    offsets = np.zeros(frame_count + 1, dtype=np.int32)

    localizer = RTDetrLocalizer(
        model_id=rtdetr_model_id,
        local_path=rtdetr_snapshot,
        device=device,
        torch_dtype="float32",
    )
    label_count = max(int(key) for key in localizer.id2label) + 1
    class_names = tuple(
        str(localizer.id2label.get(index, index)) for index in range(label_count)
    )
    width = int(entry.width or 0)
    height = int(entry.height or 0)
    if width <= 0 or height <= 0:
        raise RealProviderError(f"index entry lacks frame dimensions: {entry.video_id}")
    started = time.perf_counter()
    processed = 0
    total_candidates = 0
    position = 0
    while position < frame_count:
        window = frame_ids[position : position + decode_window]
        decoded = _decode_window(video_path, window)
        try:
            for batch_start in range(0, len(window), batch_size):
                batch = window[batch_start : batch_start + batch_size]
                images = [decoded[int(frame)] for frame in batch]
                inputs = localizer.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    localizer.device, dtype=localizer.torch_dtype
                )
                with torch.inference_mode():
                    outputs = localizer.model(
                        pixel_values=pixel_values, output_hidden_states=True
                    )
                    encoder = outputs.encoder_last_hidden_state
                    level0 = encoder[0] if isinstance(encoder, (tuple, list)) else encoder
                    if tuple(level0.shape) != (len(batch), 256, 80, 80):
                        raise RealProviderError(
                            f"unexpected encoder level-0 shape {tuple(level0.shape)} for "
                            f"{entry.video_id}"
                        )
                    gap = level0.mean(dim=(2, 3)).to(torch.float16).cpu().numpy()
                    logits = outputs.logits
                    pred_boxes = outputs.pred_boxes
                    scores_all = logits.sigmoid()
                    best_scores, best_labels = scores_all.max(dim=-1)
                    keep = min(top_k, best_scores.shape[-1])
                    top_scores, top_indices = best_scores.topk(keep, dim=-1)
                    for batch_index, frame in enumerate(batch):
                        visual[position + batch_start + batch_index] = gap[batch_index]
                        boxes_normalized = pred_boxes[batch_index][top_indices[batch_index]]
                        cx = boxes_normalized[:, 0]
                        cy = boxes_normalized[:, 1]
                        bw = boxes_normalized[:, 2]
                        bh = boxes_normalized[:, 3]
                        x1 = (cx - bw / 2.0) * width
                        y1 = (cy - bh / 2.0) * height
                        x2 = (cx + bw / 2.0) * width
                        y2 = (cy + bh / 2.0) * height
                        boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
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
                        total_candidates += len(top_scores[batch_index])
                        processed += 1
                        offsets[position + batch_start + batch_index + 1] = total_candidates
        finally:
            del decoded
        position += len(window)
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
    return {
        "video_id": entry.video_id,
        "frames": frame_count,
        "candidates": int(len(scores_array)),
        "wall_sec": time.perf_counter() - started,
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
