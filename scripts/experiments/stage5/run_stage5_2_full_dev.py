"""Stage 5.2 Full Dev inference: resumable per-video RT-DETR localization over Dev166."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from aic_video_highlight.spatial_composition.center_crop import compute_center_crop, derived_height
from aic_video_highlight.spatial_localization import (
    DEFAULT_MODEL_ID,
    RTDetrLocalizer,
    SubjectCandidate,
    SubjectPolicyConfig,
    SubjectDecision,
    merge_and_verify,
    select_primary_subject,
    shard_is_complete,
    subject_centered_crop,
    write_shard,
)

FROZEN_REPLAY_MODE = "FROZEN_REPLAY"
FRESH_PIPELINE_MODE = "FRESH_PIPELINE"
FROZEN_REPLAY_VIDEO_COUNT = 166
FROZEN_REPLAY_FRAME_COUNT = 51256


def validate_prediction_manifest(
    predictions: dict[str, list[int]],
    *,
    mode: str,
    expected_videos: int | None = None,
) -> str | None:
    """Validate the Stage 5.1 prediction manifest before inference.

    ``FROZEN_REPLAY`` preserves the historical frozen integrity gate exactly
    (166 videos / 51256 frames) so VC-0 replay behaviour is unchanged.

    ``FRESH_PIPELINE`` validates only internal consistency: a fresh Qwen run may
    legitimately emit a different frame set, and the preregistered fresh
    reproduction gate is structural (frame-set Jaccard / bbox match), not exact
    equality with the frozen frame count.
    """
    if mode not in (FROZEN_REPLAY_MODE, FRESH_PIPELINE_MODE):
        return f"INPUT GATE FAILED: unknown mode {mode!r}"
    if not predictions:
        return "INPUT GATE FAILED: no videos"

    total = 0
    for video_id in sorted(predictions):
        frames = [int(frame) for frame in predictions[video_id]]
        if not frames:
            return f"INPUT GATE FAILED: video {video_id} has no frames"
        if any(frame < 0 for frame in frames):
            return f"INPUT GATE FAILED: negative frame for {video_id}"
        if mode == FRESH_PIPELINE_MODE:
            if len(set(frames)) != len(frames):
                return f"INPUT GATE FAILED: duplicate frames for {video_id}"
            if frames != sorted(frames):
                return f"INPUT GATE FAILED: unsorted frames for {video_id}"
        total += len(frames)

    if mode == FROZEN_REPLAY_MODE:
        if (
            len(predictions) != FROZEN_REPLAY_VIDEO_COUNT
            or total != FROZEN_REPLAY_FRAME_COUNT
        ):
            return (
                "INPUT GATE FAILED: "
                f"videos={len(predictions)} frames={total} "
                f"(FROZEN_REPLAY requires {FROZEN_REPLAY_VIDEO_COUNT}/"
                f"{FROZEN_REPLAY_FRAME_COUNT})"
            )
    elif expected_videos is not None and len(predictions) != expected_videos:
        return (
            "INPUT GATE FAILED: "
            f"videos={len(predictions)} expected={expected_videos} (FRESH_PIPELINE)"
        )
    return None


def decode_needed_frames(video_path: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    """Sequential OpenCV grab; only needed frames retrieved and kept in memory."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    needed = set(frame_ids)
    max_needed = max(needed)
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while index <= max_needed:
            ok = capture.grab()
            if not ok:
                break
            if index in needed:
                retrieved, image = capture.retrieve()
                if not retrieved:
                    raise RuntimeError(f"frame {index} could not be retrieved from {video_path}")
                frames[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    missing = needed - set(frames)
    if missing:
        raise RuntimeError(f"frames missing after decode of {video_path}: {sorted(missing)[:5]}")
    return frames


def main_frame_record(video_id, frame_id, width, height, candidates, model_error=None):
    return {
        "video_id": video_id,
        "frame": frame_id,
        "image_width": width,
        "image_height": height,
        "model_error": model_error,
        "candidates": [
            {"box_xyxy": list(c.box), "score": c.score, "label_id": c.label_id, "label": c.label}
            for c in candidates
        ],
    }


def run(args: argparse.Namespace) -> int:
    import torch

    torch.cuda.reset_peak_memory_stats()
    index = {
        json.loads(line)["video_id"]: json.loads(line)
        for line in Path(args.index).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    predictions = {}
    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            predictions[rec["video_id"]] = [p["frame"] for p in rec["predictions"]]
    meta_cache = json.loads(Path(args.metadata_cache).read_text(encoding="utf-8"))["records"]
    expected_keys = {(vid, f) for vid, frames in predictions.items() for f in frames}
    gate_error = validate_prediction_manifest(
        predictions,
        mode=args.mode,
        expected_videos=args.expected_videos,
    )
    if gate_error is not None:
        print(gate_error)
        return 2
    video_base = Path(args.video_base)
    output_dir = Path(args.output_dir)

    config = SubjectPolicyConfig(person_priority=True, target_ratio=(9, 16))
    localizer = RTDetrLocalizer(
        model_id=args.model_id,
        local_path=args.local_path,
        device=args.device,
        torch_dtype=args.dtype,
    )
    provenance = {
        "model_id": args.model_id,
        "snapshot": args.local_path or "",
        "dtype": args.dtype,
        "device": localizer.device,
        "policy_version": config.policy_version,
    }
    processed = 0
    skipped = 0
    retry_frames = 0
    model_error_frames = 0
    inference_ms = []
    decode_ms = []
    started = time.perf_counter()
    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir()) / "stage52_fulldev"

    for video_id in sorted(predictions):
        frame_ids = sorted(predictions[video_id])
        if shard_is_complete(output_dir, video_id, frame_ids):
            skipped += 1
            continue
        video_path = video_base / index[video_id]["video_path"]
        meta = meta_cache[video_id]
        width, height = int(meta["width"]), int(meta["height"])
        fps_r = Fraction(str(meta["fps"]))
        t0 = time.perf_counter()
        frames = decode_needed_frames(video_path, frame_ids)
        t1 = time.perf_counter()
        decode_ms.append((t1 - t0) * 1000.0 / max(1, len(frame_ids)))

        raw_records = []
        policy_records = []
        crop_records = []
        with torch.inference_mode():
            for frame_id in frame_ids:
                image = frames[frame_id]
                timestamp = float(Fraction(frame_id) / fps_r)
                attempt = 0
                candidates = None
                error_text = None
                while attempt < 2:
                    try:
                        ta = time.perf_counter()
                        inputs = localizer.processor(images=image, return_tensors="pt")
                        pixel_values = inputs["pixel_values"].to(localizer.device, dtype=localizer.torch_dtype)
                        outputs = localizer.model(pixel_values=pixel_values)
                        candidates = localizer._decode(outputs, width, height, top_k=100)
                        tb = time.perf_counter()
                        inference_ms.append((tb - ta) * 1000.0)
                        break
                    except Exception as exc:
                        attempt += 1
                        retry_frames += 1
                        error_text = f"{type(exc).__name__}: {exc}"
                if candidates is None:
                    model_error_frames += 1
                    candidates = ()
                    decision = SubjectDecision(
                        video_id=video_id,
                        frame=frame_id,
                        image_width=width,
                        image_height=height,
                        candidates=(),
                        invalid_candidate_count=0,
                        primary=None,
                        status="CENTER_CROP_FALLBACK",
                        fallback_reasons=("MODEL_ERROR",),
                        ambiguous=False,
                        ambiguous_candidate_count=0,
                        fallback_box=None,
                        provenance=dict(provenance),
                    )
                    raw_records.append(main_frame_record(video_id, frame_id, width, height, (), model_error=error_text))
                else:
                    decision = select_primary_subject(video_id, frame_id, width, height, candidates, config, provenance)
                    raw_records.append(main_frame_record(video_id, frame_id, width, height, candidates))
                decision_record = {
                    "video_id": video_id,
                    "frame": frame_id,
                    "image_width": width,
                    "image_height": height,
                    "source_timestamp": timestamp,
                    "policy_version": config.policy_version,
                    "candidate_count": len(decision.candidates),
                    "invalid_candidate_count": decision.invalid_candidate_count,
                    "primary": None if decision.primary is None else {
                        "xyxy": list(decision.primary.box),
                        "score": decision.primary.score,
                        "label": decision.primary.label,
                        "label_id": decision.primary.label_id,
                    },
                    "status": decision.status,
                    "fallback_reasons": list(decision.fallback_reasons),
                    "ambiguous": decision.ambiguous,
                    "ambiguous_candidate_count": decision.ambiguous_candidate_count,
                    "fallback_box": list(decision.fallback_box) if decision.fallback_box else None,
                    "model_error": error_text if error_text else None,
                }
                policy_records.append(decision_record)
                subject_box = decision.primary.box if decision.primary is not None else None
                if subject_box is not None:
                    crop = subject_centered_crop(subject_box, width, height, 9, 16)
                    crop_records.append({
                        "video_id": video_id,
                        "frame": frame_id,
                        "policy": "v1",
                        "subject_xyxy": list(subject_box),
                        "crop_xywh": [crop.x, crop.y, crop.w, crop.h],
                        "status": crop.status,
                        "contains_subject": crop.contains_subject,
                        "degraded": crop.degraded,
                    })
                else:
                    fallback = decision.fallback_box or compute_center_crop(width, height, 9, 16)
                    h_val = derived_height(fallback[2], 9, 16)
                    crop_records.append({
                        "video_id": video_id,
                        "frame": frame_id,
                        "policy": "v1",
                        "subject_xyxy": None,
                        "crop_xywh": [fallback[0], fallback[1], fallback[2], float(h_val)],
                        "status": "CENTER_CROP_FALLBACK",
                        "contains_subject": False,
                        "degraded": False,
                    })
        del frames
        write_shard(output_dir, video_id, raw_records, policy_records, crop_records)
        processed += 1
        if processed % 20 == 0:
            elapsed = time.perf_counter() - started
            print(f"progress: {processed} processed / {skipped} skipped, {elapsed:.0f}s", flush=True)

    report = merge_and_verify(output_dir, expected_keys)
    total_wall = time.perf_counter() - started

    def latency_stats(values):
        ordered = sorted(values)
        n = len(ordered)
        if not n:
            return {}
        return {
            "n": n,
            "mean_ms": round(sum(ordered) / n, 2),
            "median_ms": round(ordered[n // 2], 2),
            "p90_ms": round(ordered[int(n * 0.90)], 2),
            "p95_ms": round(ordered[int(n * 0.95)], 2),
            "p99_ms": round(ordered[min(int(n * 0.99), n - 1)], 2),
            "max_ms": round(ordered[-1], 2),
        }

    summary = {
        "videos": len(predictions),
        "frames": report["frame_count"],
        "missing": report["missing_keys"],
        "extra": report["extra_keys"],
        "duplicates": report["duplicate_keys"],
        "processed_shards": processed,
        "skipped_shards": skipped,
        "model_error_frames": model_error_frames,
        "retry_frames": retry_frames,
        "total_wall_sec": round(total_wall, 1),
        "fps_overall": round(report["frame_count"] / total_wall, 2),
        "latency_inference_ms": latency_stats(inference_ms),
        "latency_decode_per_video_ms": latency_stats(decode_ms),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
        "semantic_hashes": report["semantic_hashes"],
    }
    (output_dir / "full_dev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--video-base", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument(
        "--mode",
        default=FROZEN_REPLAY_MODE,
        choices=[FROZEN_REPLAY_MODE, FRESH_PIPELINE_MODE],
        help=(
            "FROZEN_REPLAY keeps the exact 166/51256 integrity gate; "
            "FRESH_PIPELINE validates only fresh-manifest internal consistency"
        ),
    )
    parser.add_argument(
        "--expected-videos",
        type=int,
        default=None,
        help="FRESH_PIPELINE only: require exactly this many videos (e.g. 166)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
