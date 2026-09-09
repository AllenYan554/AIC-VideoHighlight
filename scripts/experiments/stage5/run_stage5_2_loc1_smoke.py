"""Stage 5.2 LOC-1 smoke runner: RT-DETR localization on the frozen smoke manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from aic_video_highlight.spatial_localization import (
    DEFAULT_MODEL_ID,
    RTDetrLocalizer,
    SubjectPolicyConfig,
    decision_to_record,
)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    frames_dir = Path(args.frames_dir)
    config = SubjectPolicyConfig(target_ratio=(9, 16))
    localizer = RTDetrLocalizer(
        model_id=args.model_id,
        local_path=args.local_path,
        device=args.device,
        torch_dtype=args.dtype,
    )
    provenance = {
        "model_id": args.model_id,
        "local_path": str(args.local_path) if args.local_path else "",
        "dtype": args.dtype,
        "device": localizer.device,
        "policy_version": "primary_subject_policy_v0",
    }

    records = []
    started = time.perf_counter()
    for entry in manifest["entries"]:
        video_id = entry["video_id"]
        frames = []
        for frame_id in entry["frames"]:
            png = frames_dir / f"{video_id}__frame_{frame_id}.png"
            image = np.asarray(Image.open(png).convert("RGB"))
            frames.append((frame_id, image))
        decisions = localizer.localize_frames(video_id, frames, config, provenance)
        records.extend(decision_to_record(d) for d in decisions)
    elapsed = time.perf_counter() - started

    result = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.manifest_sha,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": localizer.device,
        "policy_version": "primary_subject_policy_v0",
        "frame_count": len(records),
        "wall_time_sec": round(elapsed, 3),
        "decisions": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    print(
        json.dumps(
            {
                "frames": len(records),
                "wall_time_sec": round(elapsed, 3),
                "primary": sum(1 for r in records if r["status"] == "PRIMARY"),
                "fallback": sum(1 for r in records if r["status"] == "CENTER_CROP_FALLBACK"),
                "output": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha", default="")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys

    sys.exit(run(parse_args()))
