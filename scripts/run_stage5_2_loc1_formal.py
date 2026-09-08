"""Stage 5.2 Formal runner: shared raw RT-DETR candidates + policy v0/v1 + diagnostic crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from aic_video_highlight.spatial_composition.center_crop import compute_center_crop, derived_height
from aic_video_highlight.spatial_localization import (
    DEFAULT_MODEL_ID,
    RTDetrLocalizer,
    SubjectCandidate,
    SubjectPolicyConfig,
    select_primary_subject,
    subject_centered_crop,
)


def decision_payload(decision):
    primary_box = None
    if decision.primary is not None:
        x1, y1, x2, y2 = decision.primary.box
        primary_box = {
            "xyxy": [x1, y1, x2, y2],
            "score": decision.primary.score,
            "label": decision.primary.label,
        }
    return {
        "video_id": decision.video_id,
        "frame": decision.frame,
        "primary": primary_box,
        "status": decision.status,
        "fallback_reasons": list(decision.fallback_reasons),
        "ambiguous": decision.ambiguous,
        "ambiguous_candidate_count": decision.ambiguous_candidate_count,
        "fallback_box": list(decision.fallback_box) if decision.fallback_box else None,
    }


def candidates_payload(candidates):
    return [
        {
            "box_xyxy": list(c.box),
            "score": c.score,
            "label_id": c.label_id,
            "label": c.label,
        }
        for c in candidates
    ]


def run(args: argparse.Namespace) -> int:
    import torch

    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames_dir = Path(args.frames_dir)
    config_v0 = SubjectPolicyConfig(person_priority=False, target_ratio=(9, 16))
    config_v1 = SubjectPolicyConfig(person_priority=True, target_ratio=(9, 16))
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
        "manifest_id": manifest["manifest_id"],
    }

    raw_records = []
    v0_records = []
    v1_records = []
    crop_records = []
    inference_ms = []
    decode_ms = []
    model_error_frames = 0

    with torch.inference_mode():
        for entry in manifest["entries"]:
            video_id = entry["video_id"]
            for frame_id in entry["frames"]:
                t0 = time.perf_counter()
                png = frames_dir / f"{video_id}__frame_{frame_id}.png"
                image = np.asarray(Image.open(png).convert("RGB")).copy()
                height, width = image.shape[0], image.shape[1]
                t1 = time.perf_counter()
                try:
                    inputs = localizer.processor(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(localizer.device, dtype=localizer.torch_dtype)
                    outputs = localizer.model(pixel_values=pixel_values)
                    candidates = localizer._decode(outputs, width, height, top_k=100)
                except Exception as exc:
                    model_error_frames += 1
                    raw_records.append(
                        {
                            "video_id": video_id,
                            "frame": frame_id,
                            "image_width": width,
                            "image_height": height,
                            "model_error": str(exc)[:200],
                            "candidates": [],
                        }
                    )
                    continue
                t2 = time.perf_counter()
                decode_ms.append((t1 - t0) * 1000.0)
                inference_ms.append((t2 - t1) * 1000.0)
                raw_records.append(
                    {
                        "video_id": video_id,
                        "frame": frame_id,
                        "image_width": width,
                        "image_height": height,
                        "model_error": None,
                        "candidates": candidates_payload(candidates),
                    }
                )
                d0 = select_primary_subject(video_id, frame_id, width, height, candidates, config_v0, provenance)
                d1 = select_primary_subject(video_id, frame_id, width, height, candidates, config_v1, provenance)
                v0_records.append(decision_payload(d0))
                v1_records.append(decision_payload(d1))
                for tag, decision in (("v0", d0), ("v1", d1)):
                    subject_box = decision.primary.box if decision.primary is not None else None
                    if subject_box is not None:
                        crop = subject_centered_crop(subject_box, width, height, 9, 16)
                        crop_payload = {
                            "video_id": video_id,
                            "frame": frame_id,
                            "policy": tag,
                            "subject_xyxy": list(subject_box),
                            "crop_xywh": [crop.x, crop.y, crop.w, crop.h],
                            "status": crop.status,
                            "contains_subject": crop.contains_subject,
                            "degraded": crop.degraded,
                        }
                    else:
                        fallback = decision.fallback_box
                        h_val = derived_height(fallback[2], 9, 16)
                        crop_payload = {
                            "video_id": video_id,
                            "frame": frame_id,
                            "policy": tag,
                            "subject_xyxy": None,
                            "crop_xywh": [fallback[0], fallback[1], fallback[2], float(h_val)],
                            "status": "CENTER_CROP_FALLBACK",
                            "contains_subject": False,
                            "degraded": False,
                        }
                    crop_records.append(crop_payload)

    def semantic_sha(records):
        payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def latency_stats(values):
        ordered = sorted(values)
        n = len(ordered)
        return {
            "n": n,
            "mean_ms": round(sum(ordered) / n, 2) if n else None,
            "median_ms": round(ordered[n // 2], 2) if n else None,
            "p90_ms": round(ordered[int(n * 0.90)], 2) if n else None,
            "p95_ms": round(ordered[int(n * 0.95)], 2) if n else None,
            "max_ms": round(ordered[-1], 2) if n else None,
        }

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    result = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.manifest_sha,
        "model_id": args.model_id,
        "snapshot": args.local_path,
        "dtype": args.dtype,
        "device": localizer.device,
        "gpu_name": torch.cuda.get_device_name(0) if localizer.device.startswith("cuda") else None,
        "peak_allocated_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_reserved, 1),
        "batch_size": 1,
        "frame_count": len(raw_records),
        "model_error_frames": model_error_frames,
        "latency_inference_ms": latency_stats(inference_ms),
        "latency_decode_ms": latency_stats(decode_ms),
        "semantic_hashes": {
            "raw_candidates": semantic_sha(raw_records),
            "policy_v0_decisions": semantic_sha(v0_records),
            "policy_v1_decisions": semantic_sha(v1_records),
            "crop_diagnostics": semantic_sha(crop_records),
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_detector.json").write_text(
        json.dumps({"meta": {k: v for k, v in result.items() if k not in ("semantic_hashes",)}, "frames": raw_records}, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    (out_dir / "policy_v0.json").write_text(json.dumps(v0_records, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    (out_dir / "policy_v1.json").write_text(json.dumps(v1_records, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    (out_dir / "crop_diagnostic.json").write_text(json.dumps(crop_records, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    (out_dir / "run_meta.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    print(json.dumps({k: v for k, v in result.items() if k != "semantic_hashes"}, ensure_ascii=False, indent=2))
    print("semantic_hashes:", json.dumps(result["semantic_hashes"], indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha", default="")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys

    sys.exit(run(parse_args()))
