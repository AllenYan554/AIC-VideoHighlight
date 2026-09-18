"""FTNet dataset materialization: upstream signals -> per-video safetensors.

The heavy upstream computation (Qwen retrieval, RT-DETR features, LOC, CMP, TS)
is provided through an :class:`UpstreamProvider`.  This module owns only the
frozen output contract, native assembly, Y target and resume/verify logic, so
that local synthetic runs and AutoDL production runs share identical code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from safetensors.numpy import load_file, save_file

from .native_schema import (
    NATIVE_DIM,
    NATIVE_FIELDS,
    NATIVE_SCHEMA_VERSION,
    NativeSchemaError,
    build_native_matrix,
)


MATERIALIZE_SCHEMA = "aic.stage7.ftnet.materialized-video/v1"
SPLIT_DIRS = {"TRAIN": "train", "VALIDATION": "validation", "CALIBRATION": "calibration"}


class MaterializeError(RuntimeError):
    """Raised when a video cannot be materialized into the frozen contract."""


@dataclass(frozen=True)
class VideoRef:
    canonical_video_id: str
    realized_video_id: str
    category: str
    stage7_split: str
    relative_video_path: str
    source_sha256: str


@dataclass
class VideoUpstream:
    """Raw upstream signals for one video before the frozen transforms."""

    ref: VideoRef
    timestamps: np.ndarray
    source_frame_id: np.ndarray
    adjacency_mask: np.ndarray
    visual: np.ndarray
    native_signals: dict[str, np.ndarray]
    target: np.ndarray
    loss_mask: np.ndarray
    candidate_missed_positive: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class UpstreamProvider(Protocol):
    def list_videos(self, stage7_split: str) -> list[VideoRef]: ...

    def produce(self, ref: VideoRef) -> VideoUpstream: ...


def _as_bool(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=bool)


def validate_upstream(upstream: VideoUpstream) -> int:
    timestamps = np.asarray(upstream.timestamps, dtype=np.float64)
    frame_count = timestamps.shape[0]
    if frame_count == 0:
        raise MaterializeError(f"empty sequence for {upstream.ref.canonical_video_id}")
    if np.asarray(upstream.source_frame_id).shape != (frame_count,):
        raise MaterializeError("source_frame_id must have shape [T]")
    if np.asarray(upstream.adjacency_mask).shape != (frame_count,):
        raise MaterializeError("adjacency_mask must have shape [T]")
    visual = np.asarray(upstream.visual)
    if visual.shape != (frame_count, 256):
        raise MaterializeError(f"visual must have shape [T,256], got {visual.shape}")
    if not np.isfinite(visual).all():
        raise MaterializeError("visual must be finite")
    target = np.asarray(upstream.target, dtype=np.float32)
    if target.shape != (frame_count,):
        raise MaterializeError("target must have shape [T]")
    if not np.isfinite(target).all() or target.min() < 0.0 or target.max() > 1.0:
        raise MaterializeError("target must be finite in [0,1]")
    loss_mask = _as_bool(upstream.loss_mask)
    if loss_mask.shape != (frame_count,):
        raise MaterializeError("loss_mask must have shape [T]")
    missing_names = set(NATIVE_FIELDS) - set(upstream.native_signals)
    if missing_names:
        raise MaterializeError(f"missing native signals: {sorted(missing_names)}")
    for name, value in upstream.native_signals.items():
        if np.asarray(value).shape != (frame_count,):
            raise MaterializeError(f"native signal {name} must have shape [T]")
    return frame_count


def assemble_native(upstream: VideoUpstream) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``[T,16]`` native and ``[T,16]`` structural-missing mask."""

    validate_upstream(upstream)
    missing = np.zeros((upstream.timestamps.shape[0], NATIVE_DIM), dtype=np.float32)
    raw_signals: dict[str, np.ndarray] = {}
    for name in NATIVE_FIELDS:
        values = np.asarray(upstream.native_signals[name], dtype=np.float64)
        finite = np.isfinite(values)
        missing[~finite, NATIVE_FIELDS.index(name)] = 1.0
        raw_signals[name] = np.where(finite, values, 0.0).astype(np.float32)
    raw = build_native_matrix({name: torch_from_numpy(raw_signals[name]) for name in NATIVE_FIELDS})
    native = raw.numpy()
    if native.shape[-1] != NATIVE_DIM:
        raise NativeSchemaError("native assembly produced wrong dimension")
    return native, missing


def torch_from_numpy(array: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def video_safetensors_payload(upstream: VideoUpstream) -> dict[str, np.ndarray]:
    native, native_missing = assemble_native(upstream)
    payload = {
        "visual": np.asarray(upstream.visual, dtype=np.float16),
        "native": native.astype(np.float32),
        "native_missing": native_missing.astype(np.uint8),
        "target": np.asarray(upstream.target, dtype=np.float32),
        "loss_mask": _as_bool(upstream.loss_mask).astype(np.uint8),
        "adjacency_mask": _as_bool(upstream.adjacency_mask).astype(np.uint8),
        "timestamp": np.asarray(upstream.timestamps, dtype=np.float64),
        "source_frame_id": np.asarray(upstream.source_frame_id, dtype=np.int64),
    }
    if not np.isfinite(payload["native"]).all():
        raise MaterializeError("raw native must be finite (missing must be masked, not NaN)")
    return payload


def _video_path(output_root: Path, ref: VideoRef) -> Path:
    return output_root / SPLIT_DIRS[ref.stage7_split] / f"{ref.canonical_video_id}.safetensors"


def verify_materialized(path: Path, *, source_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        tensors = load_file(str(path))
    except Exception:
        return False
    if tensors.get("visual", np.zeros((0, 0))).shape[1:] != (256,):
        return False
    if tensors.get("native", np.zeros((0, 0))).shape[1:] != (NATIVE_DIM,):
        return False
    required = {
        "visual", "native", "native_missing", "target",
        "loss_mask", "adjacency_mask", "timestamp", "source_frame_id",
    }
    if not required.issubset(tensors):
        return False
    return bool(np.isfinite(tensors["native"]).all()) and bool(np.isfinite(tensors["visual"]).all())


def materialize_video(
    upstream: VideoUpstream,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    output_root = Path(output_root)
    ref = upstream.ref
    path = _video_path(output_root, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    skipped = False
    if path.exists() and not overwrite and verify_materialized(path, source_sha256=ref.source_sha256):
        skipped = True
        frame_count = int(np.asarray(upstream.timestamps).shape[0])
    else:
        payload = video_safetensors_payload(upstream)
        frame_count = int(payload["timestamp"].shape[0])
        metadata = {
            "schema": MATERIALIZE_SCHEMA,
            "video_id": ref.canonical_video_id,
            "realized_video_id": ref.realized_video_id,
            "source_id": ref.canonical_video_id,
            "source_sha256": ref.source_sha256,
            "category": ref.category,
            "split": ref.stage7_split,
            "native_schema_version": NATIVE_SCHEMA_VERSION,
            **{key: str(value) for key, value in upstream.metadata.items()},
        }
        save_file(payload, str(path), metadata=metadata)
    return {
        "canonical_video_id": ref.canonical_video_id,
        "category": ref.category,
        "stage7_split": ref.stage7_split,
        "relative_path": str(path.relative_to(output_root)).replace("\\", "/"),
        "frame_count": frame_count,
        "skipped": skipped,
        "missed_positive_frames": int(
            np.asarray(
                upstream.candidate_missed_positive
                if upstream.candidate_missed_positive is not None
                else np.zeros(frame_count, dtype=bool)
            ).sum()
        ),
    }


def materialize_split(
    provider: UpstreamProvider,
    output_root: str | Path,
    stage7_split: str,
    *,
    overwrite: bool = False,
) -> list[dict]:
    refs = provider.list_videos(stage7_split)
    records = []
    for ref in refs:
        upstream = provider.produce(ref)
        if upstream.ref.stage7_split != stage7_split:
            raise MaterializeError("provider returned mismatched split")
        records.append(materialize_video(upstream, output_root, overwrite=overwrite))
    return records


def write_manifest(output_root: str | Path, records: list[dict], extra: dict | None = None) -> Path:
    output_root = Path(output_root)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MATERIALIZE_SCHEMA,
        "native_schema_version": NATIVE_SCHEMA_VERSION,
        "count": len(records),
        "records": records,
    }
    if extra:
        payload.update(extra)
    path = manifest_dir / "materialized_videos.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
