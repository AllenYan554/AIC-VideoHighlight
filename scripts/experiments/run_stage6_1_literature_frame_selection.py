#!/usr/bin/env python3
"""Canonical Stage 6.1 literature frame-selection Dev166 Formal runner.

What this script is
-------------------
The *orchestration* entry point for the preregistered Stage 6.1 comparison:

* Arm A -- ``FS0_CONTROL``                 (identity KEEP-ALL over frozen FS0)
* Arm B -- ``PGL_SUM_TABLE_IV_10_MEAN``    (PGL-SUM, Zenodo Table IV x10 mean)
* Arm C -- ``VASNET_XAI_SUM_10_MEAN``      (VASNet, XAI-SUM x10 mean)

It owns **no** scientific mathematics.  Every selector, feature extractor,
KTS/knapsack step, timing map and evaluator lives in the frozen library modules:

* :mod:`aic_video_highlight.composition.literature_features`
* :mod:`aic_video_highlight.composition.literature_frame_selection`
* :mod:`aic_video_highlight.composition.pgl_sum_selector`
* :mod:`aic_video_highlight.composition.vasnet_selector`
* :mod:`aic_video_highlight.evaluation.official_like`
* :mod:`aic_video_highlight.evaluation.contract`

The runner starts from the byte-frozen Stage 6.0 TS5/FS0 upstream
(``vhicraft_dev166_202609141233``) and may only KEEP/DROP existing FS0
predictions.  It never re-runs Qwen, frame projection, RT-DETR, LOC v1, CMP-1 or
TS-5 and never regenerates a bbox.

Modes
-----
``--validate-only``
    Read-only identity + model + hardware preflight.  Extracts no features and
    runs no arm.
``--arm fs0|pgl|vasnet|all``
    Execute the requested arm(s).  ``all`` also enforces the FS0 reproduction
    gate before running the two literature arms.
``--resume``
    Reuse an already-written arm/predictions artefact and the per-video shared
    feature cache when their bound identities match.

The external ``formal_protocol.json`` / ``model_manifest.json`` are frozen
inputs; their SHA256 is re-verified on every invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _path in (str(_REPO / "src"), str(_REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aic_video_highlight.composition.frame_projection import (  # noqa: E402
    CFR_FPS,
    PTS_TABLE,
    VideoTiming,
)
from aic_video_highlight.composition.literature_features import (  # noqa: E402
    FEATURE_DIM,
    GoogleNetPool5Extractor,
    read_frames_bgr,
)
from aic_video_highlight.composition.literature_frame_selection import (  # noqa: E402
    KTSConfig,
    LiteratureFrameSelection,
    apply_keep_drop,
    literature_select,
    sample_frame_mapping,
)
from aic_video_highlight.composition.pgl_sum_selector import (  # noqa: E402
    load_pgl_sum_model,
    pgl_sum_frame_scores,
)
from aic_video_highlight.composition.vasnet_selector import (  # noqa: E402
    load_vasnet_model,
    vasnet_frame_scores,
)
from aic_video_highlight.evaluation.contract import validate_contract  # noqa: E402
from aic_video_highlight.evaluation.official_like import evaluate_files  # noqa: E402
from aic_video_highlight.experiment_runtime.hashing import (  # noqa: E402
    canonical_sha256,
    file_sha256,
)
from aic_video_highlight.experiment_runtime.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
)

RUNNER_SCHEMA = "aic.stage6_1.formal-runner/v1"
ARM_SCHEMA = "aic.stage6_1.arm-output/v1"
CACHE_SCHEMA = "aic.stage6_1.feature-cache/v1"

SAMPLE_FPS = 2.0
WORST_K = 20
TIE_EPSILON = 1e-12
FLOAT_TOLERANCE = 1e-9

PGL = "pgl"
VASNET = "vasnet"
FS0 = "fs0"
METHOD_OF_ARM = {PGL: "pgl_sum", VASNET: "vasnet"}
ARM_DIRECTORY = {FS0: "01_FS0_Control", PGL: "02_PGL_SUM", VASNET: "03_VASNet"}

# --- frozen identities (must match the preregistered external manifests) -----
FROZEN_PROTOCOL_SHA = "1ed35045ead49848f220d5d1a8d63404055da2de919b2261ac523c23610526ff"
FROZEN_MODEL_MANIFEST_SHA = "5d1e16341a11697c6a449733a2542f565aefa3766010d51d399766d2ae5182a5"
FROZEN_CONTROL_PREDICTIONS_SHA = "3cf4566b9cda13ca6dd34f78b3a05558d4dfe3b6b57910499982e47d6f9a31c1"
FROZEN_REFERENCE_VALID_SHA = "fe6ba426dd0b17948a88718a86e871f242dc59a40e38534037cc9659c446d515"
FROZEN_EMPTY_REFERENCE_SHA = "eb433ecada9ed6e13f83a8c6ce8454fc57c31b9cfd9693b08c89bdfa7d2c2453"
FROZEN_GOOGLENET_SHA = "1378be20a8e875cf1568b8a71654e704449655e34711a959a38b04fb34905cef"

#: Frozen Stage 6.0 control metrics the FS0 arm must reproduce.
CONTROL_OFFICIAL_LIKE = {
    "macro_precision": 0.169459591940892,
    "macro_recall": 0.2974219914513993,
    "macro_f": 0.20260061017000433,
}
CONTROL_FRAME = {
    "n_pred": 52019,
    "n_ref": 18635,
    "tp": 17620,
    "fp": 34399,
    "fn": 1015,
    "precision": 0.33872238989599956,
    "recall": 0.9455325999463375,
    "f1": 0.498768647210349,
}

FIGURE_NAMES = (
    "overall_score_comparison.png",
    "valid109_official_like_comparison.png",
    "frame_precision_recall_comparison.png",
    "prediction_count_drop.png",
    "per_video_delta_f_pgl.png",
    "per_video_delta_f_vasnet.png",
    "recall_regression_distribution.png",
    "empty_reference_behavior.png",
)


class Stage61Error(RuntimeError):
    """Raised for any fatal Stage 6.1 runner/identity/protocol failure."""


# ---------------------------------------------------------------------------
# Small IO / hashing utilities
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_video_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise Stage61Error(f"duplicate video ids in identity file: {path}")
    return ids


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_predictions_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(Path(path), text)


def verify_sha256(name: str, path: Path, expected: str) -> str:
    actual = file_sha256(Path(path))
    if actual.lower() != expected.lower():
        raise Stage61Error(
            f"frozen identity mismatch for {name}: expected {expected}, got {actual} ({path})"
        )
    return actual


def current_git_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def science_tree_changed(repo: Path, frozen_head: str | None) -> bool | None:
    """True if ``src/`` differs from ``frozen_head``; None if git cannot tell."""
    if not frozen_head:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "diff", "--quiet", frozen_head, "--", "src"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0:
        return False
    if completed.returncode == 1:
        return True
    return None


# ---------------------------------------------------------------------------
# Timing / sampling identity
# ---------------------------------------------------------------------------

def build_timing(record: Mapping[str, Any], video_id: str) -> VideoTiming:
    """Reconstruct the frozen per-video timing from the Stage 6.0 metadata cache."""
    mode = str(record.get("timestamp_mode") or CFR_FPS)
    pts_raw = record.get("pts_timestamps")
    pts: tuple[Fraction, ...] | None = None
    if mode == PTS_TABLE:
        if not isinstance(pts_raw, list):
            raise Stage61Error(f"{video_id}: PTS_TABLE requires pts_timestamps")
        pts = tuple(Fraction(str(value)) for value in pts_raw)
    return VideoTiming(
        video_id=video_id,
        fps=float(record["fps"]),
        frame_count=int(record["frame_count"]),
        timestamp_mode=mode,
        pts_timestamps=pts,
    )


def sample_trace(mapping: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": int(item.sample_index),
            "source_frame": int(item.source_frame),
            "target_timestamp": str(item.target_timestamp),
            "source_timestamp": str(item.source_timestamp),
        }
        for item in mapping
    ]


def sample_trace_identity(mapping: Sequence[Any]) -> str:
    return canonical_sha256(sample_trace(mapping))


def kts_config_for(n_samples: int) -> KTSConfig:
    """``ncp_max = min(ceil(T / 2), T - 1)`` with the frozen vmax/lmin/lmax."""
    if n_samples < 1:
        raise Stage61Error("cannot build KTS config for an empty sample grid")
    ncp_max = min(int(math.ceil(n_samples / SAMPLE_FPS)), n_samples - 1)
    return KTSConfig(ncp_max=max(1, ncp_max), vmax=1.0, lmin=1)


# ---------------------------------------------------------------------------
# Frozen upstream loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    timing: VideoTiming
    fs0_frames: tuple[int, ...]
    control_record: Mapping[str, Any]
    video_path: str | None
    video_sha256: str | None
    sample_frames: tuple[int, ...]
    mapping_identity: str


def fs0_frames_of(record: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(sorted({int(item["frame"]) for item in record.get("predictions", [])}))


def build_video_specs(
    metadata_records: Mapping[str, Any],
    control_records: Sequence[Mapping[str, Any]],
    video_shas: Mapping[str, str],
    *,
    sample_fps: float = SAMPLE_FPS,
) -> list[VideoSpec]:
    control_by_id = {str(record["video_id"]): record for record in control_records}
    specs: list[VideoSpec] = []
    for video_id in sorted(control_by_id):
        if video_id not in metadata_records:
            raise Stage61Error(f"{video_id}: missing from the frozen metadata cache")
        record = control_by_id[video_id]
        timing = build_timing(metadata_records[video_id], video_id)
        mapping = sample_frame_mapping(timing, sample_fps)
        specs.append(
            VideoSpec(
                video_id=video_id,
                timing=timing,
                fs0_frames=fs0_frames_of(record),
                control_record=record,
                video_path=metadata_records[video_id].get("video_path"),
                video_sha256=video_shas.get(video_id),
                sample_frames=tuple(item.source_frame for item in mapping),
                mapping_identity=sample_trace_identity(mapping),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Feature cache (shared by PGL-SUM and VASNet; engineering reuse only)
# ---------------------------------------------------------------------------

def feature_cache_identity(
    spec: VideoSpec,
    *,
    aic_git_head: str | None,
    sample_fps: float = SAMPLE_FPS,
) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "video_id": spec.video_id,
        "video_path": spec.video_path,
        "video_sha256": spec.video_sha256,
        "feature_identity": "FEATURE_EXTRACTOR_COMPATIBLE_REPRODUCTION",
        "extractor": "torchvision.models.googlenet",
        "extractor_weights": "GoogLeNet_Weights.IMAGENET1K_V1",
        "extractor_weights_sha256": FROZEN_GOOGLENET_SHA,
        "feature_dim": FEATURE_DIM,
        "sample_fps": sample_fps,
        "sample_count": len(spec.sample_frames),
        "mapping_sha256": spec.mapping_identity,
        "aic_git_head": aic_git_head,
    }


def feature_cache_is_valid(identity: Mapping[str, Any], existing: Mapping[str, Any] | None) -> bool:
    if not existing:
        return False
    return all(existing.get(key) == identity.get(key) for key in identity)


class SharedFeatureCache:
    """One GoogLeNet extraction per video, reused by both literature arms."""

    def __init__(
        self,
        root: Path,
        extractor: GoogleNetPool5Extractor,
        *,
        aic_git_head: str | None,
        sample_fps: float = SAMPLE_FPS,
        resume: bool = True,
    ) -> None:
        self.root = Path(root)
        self.extractor = extractor
        self.aic_git_head = aic_git_head
        self.sample_fps = sample_fps
        self.resume = resume
        self.hits = 0
        self.misses = 0
        self.extract_seconds = 0.0
        self.total_frames = 0

    def _paths(self, video_id: str) -> tuple[Path, Path]:
        directory = self.root / video_id
        return directory / "features.npy", directory / "identity.json"

    def features(self, spec: VideoSpec) -> np.ndarray:
        feature_path, identity_path = self._paths(spec.video_id)
        identity = feature_cache_identity(spec, aic_git_head=self.aic_git_head, sample_fps=self.sample_fps)
        existing = load_json(identity_path) if identity_path.is_file() else None
        if self.resume and feature_path.is_file() and feature_cache_is_valid(identity, existing):
            array = np.load(feature_path)
            if array.shape == (len(spec.sample_frames), FEATURE_DIM):
                self.hits += 1
                return array
        if spec.video_path is None:
            raise Stage61Error(f"{spec.video_id}: no video_path in the frozen metadata cache")
        started = time.perf_counter()
        frames = read_frames_bgr(spec.video_path, spec.sample_frames)
        array = self.extractor.extract(frames).astype(np.float32)
        self.extract_seconds += time.perf_counter() - started
        self.total_frames += len(spec.sample_frames)
        self.misses += 1
        if array.shape != (len(spec.sample_frames), FEATURE_DIM):
            raise Stage61Error(f"{spec.video_id}: unexpected feature shape {array.shape}")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, array)
        stored = dict(identity)
        stored["features_npy_sha256"] = file_sha256(feature_path)
        atomic_write_json(identity_path, stored)
        return array


# ---------------------------------------------------------------------------
# Model ensembles
# ---------------------------------------------------------------------------

def checkpoint_entries(protocol: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    policy = protocol["checkpoint_policy"]
    key = "pgl_sum" if arm == PGL else "vasnet"
    entries = list(policy[key])
    if len(entries) != 10:
        raise Stage61Error(f"{arm}: expected 10 preregistered checkpoints, found {len(entries)}")
    return entries


def verify_checkpoints(arm: str, entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        verify_sha256(f"{arm} checkpoint {entry['path']}", Path(entry["path"]), str(entry["sha256"]))


def load_models(arm: str, entries: Sequence[Mapping[str, Any]], device: str) -> list[Any]:
    models: list[Any] = []
    for entry in entries:
        if arm == PGL:
            models.append(load_pgl_sum_model(entry["path"], device=device))
        else:
            models.append(load_vasnet_model(entry["path"], device=device))
    return models


def ensemble_scores(arm: str, models: Sequence[Any], features: np.ndarray, device: str) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(features.astype(np.float32)))
    score_fn = pgl_sum_frame_scores if arm == PGL else vasnet_frame_scores
    stacked = np.stack([np.asarray(score_fn(model, tensor, device=device)) for model in models], axis=0)
    return stacked.mean(axis=0)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def fs0_reconstruct(control_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """KEEP-ALL identity: every frozen FS0 row survives byte-identical."""
    return [dict(record) for record in control_records]


def select_with_scores(
    method: str,
    spec: VideoSpec,
    features: np.ndarray,
    scores: np.ndarray,
) -> LiteratureFrameSelection:
    """Delegate the KEEP/DROP decision to the frozen literature adapter."""
    if len(spec.sample_frames) < 2:
        return literature_select(
            method=method,
            sample_frames=spec.sample_frames,
            frame_scores=list(np.asarray(scores, dtype=np.float64)),
            n_frames=spec.timing.frame_count,
            fs0_frames=spec.fs0_frames,
            shot_bounds=((0, len(spec.sample_frames)),),
        )
    return literature_select(
        method=method,
        sample_frames=spec.sample_frames,
        frame_scores=list(np.asarray(scores, dtype=np.float64)),
        n_frames=spec.timing.frame_count,
        fs0_frames=spec.fs0_frames,
        features=features,
        kts=kts_config_for(len(spec.sample_frames)),
    )


def run_literature_arm(
    arm: str,
    specs: Sequence[VideoSpec],
    feature_fn: Callable[[VideoSpec], np.ndarray],
    score_fn: Callable[[str, np.ndarray], np.ndarray],
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, LiteratureFrameSelection]]:
    method = METHOD_OF_ARM[arm]
    predictions: list[dict[str, Any]] = []
    selections: dict[str, LiteratureFrameSelection] = {}
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        features = np.asarray(feature_fn(spec), dtype=np.float32)
        scores = np.asarray(score_fn(arm, features), dtype=np.float64)
        if scores.shape[0] != len(spec.sample_frames):
            raise Stage61Error(f"{spec.video_id}: ensemble score length mismatch")
        selection = select_with_scores(method, spec, features, scores)
        selections[spec.video_id] = selection
        rows = list(spec.control_record.get("predictions", []))
        kept = apply_keep_drop({spec.video_id: rows}, {spec.video_id: selection})[spec.video_id]
        record = dict(spec.control_record)
        record["predictions"] = kept
        predictions.append(record)
        if progress is not None:
            progress(index, total, spec.video_id)
    return predictions, selections


def audit_subset_invariants(
    specs: Sequence[VideoSpec],
    predictions: Sequence[Mapping[str, Any]],
    selections: Mapping[str, LiteratureFrameSelection],
) -> dict[str, Any]:
    """Assert KEEP subset FS0, exact KEEP/DROP partition, and byte-frozen bboxes."""
    frozen = {spec.video_id: spec.control_record for spec in specs}
    kept_total = dropped_total = 0
    violations: list[str] = []
    for record in predictions:
        video_id = str(record["video_id"])
        spec_frozen = frozen[video_id]
        fs0 = set(fs0_frames_of(spec_frozen))
        kept_rows = list(record.get("predictions", []))
        kept_frames = [int(row["frame"]) for row in kept_rows]
        selection = selections[video_id]
        if set(kept_frames) != set(selection.kept_frames):
            violations.append(f"{video_id}: KEEP mask disagrees with adapter selection")
        if not set(kept_frames) <= fs0:
            violations.append(f"{video_id}: KEEP contains frames outside FS0")
        if set(kept_frames) | set(selection.dropped_frames) != fs0:
            violations.append(f"{video_id}: KEEP+DROP does not cover FS0")
        if set(kept_frames) & set(selection.dropped_frames):
            violations.append(f"{video_id}: KEEP and DROP overlap")
        frozen_rows = {int(row["frame"]): row for row in spec_frozen.get("predictions", [])}
        for row in kept_rows:
            frame = int(row["frame"])
            if frame not in frozen_rows:
                violations.append(f"{video_id}: added frame {frame}")
            elif dict(row) != dict(frozen_rows[frame]):
                violations.append(f"{video_id}: modified prediction row at frame {frame}")
        kept_total += len(kept_frames)
        dropped_total += len(selection.dropped_frames)
    if violations:
        raise Stage61Error("FS0 subset invariant failure: " + "; ".join(violations[:10]))
    return {
        "video_count": len(predictions),
        "kept_frames": kept_total,
        "dropped_frames": dropped_total,
        "violations": 0,
        "selected_subset_of_fs0": True,
        "kept_bbox_unchanged": True,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_group(
    predictions_path: Path,
    reference_path: Path,
    *,
    video_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    report = evaluate_files(predictions_path, reference_path, video_ids=video_ids)
    per_video = report["per_video"]
    n_pred = sum(row["prediction_frames"] for row in per_video)
    n_ref = sum(row["reference_frames"] for row in per_video)
    tp = sum(row["exact_common_frames"] for row in per_video)
    fp = n_pred - tp
    fn = n_ref - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "video_count": len(per_video),
        "macro_precision": report["Official-like Weak-Reference Precision"],
        "macro_recall": report["Official-like Weak-Reference Recall"],
        "macro_f": report["Official-like Weak-Reference F-score"],
        "frame_n_pred": n_pred,
        "frame_n_ref": n_ref,
        "frame_tp": tp,
        "frame_fp": fp,
        "frame_fn": fn,
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
        "per_video": {
            str(row["video_id"]): {
                "n_pred": int(row["prediction_frames"]),
                "n_ref": int(row["reference_frames"]),
                "matched": int(row["exact_common_frames"]),
                "sum_iou": float(row["sum_spatial_iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f": float(row["f_score"]),
            }
            for row in per_video
        },
    }


def compute_paired(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    tie_epsilon: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Per-video Win/Tie/Loss and ΔF over the shared video id set."""
    ids = sorted(set(baseline["per_video"]) & set(challenger["per_video"]))
    deltas: list[tuple[str, float]] = []
    wins = ties = losses = 0
    for video_id in ids:
        delta = challenger["per_video"][video_id]["f"] - baseline["per_video"][video_id]["f"]
        deltas.append((video_id, delta))
        if abs(delta) <= tie_epsilon:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    ordered = sorted(deltas, key=lambda item: (item[1], item[0]))
    return {
        "video_count": len(ids),
        "win": wins,
        "tie": ties,
        "loss": losses,
        "max_gain": {"video_id": ordered[-1][0], "delta_f": ordered[-1][1]} if ordered else None,
        "max_regression": {"video_id": ordered[0][0], "delta_f": ordered[0][1]} if ordered else None,
        "worst_k": [{"video_id": vid, "delta_f": delta} for vid, delta in ordered[:WORST_K]],
        "per_video_delta_f": {vid: delta for vid, delta in deltas},
    }


def recall_guard(metrics: Mapping[str, Any]) -> dict[str, int]:
    """Count per-video frame-membership recall below guard thresholds."""
    below_08 = below_05 = zero = 0
    for row in metrics["per_video"].values():
        if row["n_ref"] <= 0:
            continue
        recall = row["matched"] / row["n_ref"]
        if recall == 0.0:
            zero += 1
        if recall < 0.5:
            below_05 += 1
        if recall < 0.8:
            below_08 += 1
    return {"recall_lt_0_8": below_08, "recall_lt_0_5": below_05, "recall_eq_0": zero}


def empty_reference_audit(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    empty_ids: Sequence[str],
) -> dict[str, Any]:
    ids = [video_id for video_id in empty_ids if video_id in baseline["per_video"]]
    double_empty = 0
    rows: list[dict[str, Any]] = []
    base_pred = chal_pred = base_empty = chal_empty = 0
    for video_id in ids:
        before = baseline["per_video"][video_id]
        after = challenger["per_video"][video_id]
        base_pred += before["n_pred"]
        chal_pred += after["n_pred"]
        base_empty += 1 if before["n_pred"] == 0 else 0
        chal_empty += 1 if after["n_pred"] == 0 else 0
        transition = before["n_pred"] > 0 and after["n_pred"] == 0
        double_empty += 1 if transition else 0
        rows.append(
            {
                "video_id": video_id,
                "n_ref": before["n_ref"],
                "n_pred_before": before["n_pred"],
                "n_pred_after": after["n_pred"],
                "double_empty_transition": transition,
            }
        )
    return {
        "video_count": len(ids),
        "n_pred_before": base_pred,
        "n_pred_after": chal_pred,
        "drop_ratio": (base_pred - chal_pred) / base_pred if base_pred else 0.0,
        "empty_before": base_empty,
        "empty_after": chal_empty,
        "double_empty_transitions": double_empty,
        "not_scientific_gain": "DOUBLE_EMPTY_NOT_SCIENTIFIC_GAIN",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Outputs / runtime
# ---------------------------------------------------------------------------

@dataclass
class RuntimeTracker:
    phases: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    reused: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started)

    def add_count(self, name: str, value: int) -> None:
        self.counts[name] = self.counts.get(name, 0) + value

    @property
    def total(self) -> float:
        return sum(self.phases.values())

    def as_dict(self, *, device: str, batch_size: int, peak_vram_mib: float | None) -> dict[str, Any]:
        return {
            "schema": "aic.stage6_1.runtime/v1",
            "phases_seconds": dict(sorted(self.phases.items())),
            "operations": dict(sorted(self.counts.items())),
            "reused": dict(sorted(self.reused.items())),
            "total_seconds": self.total,
            "scope_note": "selector-only runtime; Stage 6.0 upstream runtime excluded",
            "environment": {
                "device": device,
                "batch_size": batch_size,
                "peak_vram_mib": peak_vram_mib,
                "peak_vram_note": "torch allocator peak when available; not upstream",
            },
        }


def arm_dir(output_root: Path, arm: str) -> Path:
    return output_root / ARM_DIRECTORY[arm]


def write_arm_output(
    output_root: Path,
    arm: str,
    predictions: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    directory = arm_dir(output_root, arm)
    predictions_path = directory / "predictions" / "predictions.jsonl"
    write_predictions_jsonl(predictions_path, predictions)
    contract = validate_contract(predictions_path)
    payload = dict(identity)
    payload["predictions_file_sha256"] = file_sha256(predictions_path)
    payload["predictions_scientific_sha256"] = canonical_sha256(list(predictions))
    payload["contract_is_valid"] = bool(contract["is_valid"])
    payload["contract_issue_count"] = int(contract["issue_count"])
    payload["predicted_frame_count"] = sum(len(record.get("predictions", [])) for record in predictions)
    atomic_write_json(directory / "identity.json", payload)
    if not contract["is_valid"]:
        raise Stage61Error(f"{arm}: predictions contract invalid ({contract['issue_count']} issues)")
    return predictions_path, payload


def arm_is_resumable(directory: Path, expected: Mapping[str, Any]) -> bool:
    identity_path = directory / "identity.json"
    predictions_path = directory / "predictions" / "predictions.jsonl"
    if not identity_path.is_file() or not predictions_path.is_file():
        return False
    existing = load_json(identity_path)
    return all(existing.get(key) == expected.get(key) for key in expected)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_arms(value: str) -> tuple[str, ...]:
    if value == "all":
        return (FS0, PGL, VASNET)
    if value in (FS0, PGL, VASNET):
        return (value,)
    raise Stage61Error(f"unknown arm: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", default="all", choices=["fs0", "pgl", "vasnet", "all"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--valid-ids", type=Path, default=None)
    parser.add_argument("--empty-ids", type=Path, default=None)
    return parser.parse_args(argv)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise Stage61Error(message)


@dataclass
class RunContext:
    protocol_path: Path
    model_manifest_path: Path
    protocol: dict[str, Any]
    control_run: Path
    control_predictions_path: Path
    metadata_cache_path: Path
    dev_selected_path: Path
    reference_path: Path
    valid_ids_path: Path
    empty_ids_path: Path
    output_root: Path
    run_id: str
    device: str
    batch_size: int
    valid_ids: list[str]
    empty_ids: list[str]
    metadata_records: Mapping[str, Any]
    specs: list[VideoSpec]
    aic_git_head: str | None
    science_tree_changed: bool | None

    @property
    def group_id(self) -> str:
        return f"stage6_1_dev166_formal_{self.run_id}"

    def arm_run_id(self, arm: str) -> str:
        return f"stage6_1_{arm}_dev166_{self.run_id}"


def build_context(args: argparse.Namespace) -> RunContext:
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    verify_sha256("formal_protocol.json", protocol_path, FROZEN_PROTOCOL_SHA)
    model_manifest_path = args.model_manifest.resolve()
    load_json(model_manifest_path)
    verify_sha256("model_manifest.json", model_manifest_path, FROZEN_MODEL_MANIFEST_SHA)

    control_run = args.control_run.resolve()
    control_predictions_path = control_run / "predictions" / "predictions.jsonl"
    verify_sha256("control predictions", control_predictions_path, FROZEN_CONTROL_PREDICTIONS_SHA)
    metadata_cache_path = control_run / "frame_projection" / "metadata_cache.json"
    dev_selected_path = control_run / "input" / "dev_selected.jsonl"

    manifest_dir = protocol_path.parent
    valid_ids_path = args.valid_ids or (
        manifest_dir / protocol["evaluation_protocol"]["reference_valid_primary"]["identity_file"]
    )
    empty_ids_path = args.empty_ids or (
        manifest_dir / protocol["evaluation_protocol"]["empty_reference_diagnostic"]["identity_file"]
    )
    reference_path = args.reference or Path(
        protocol["frozen_stage6_0_control"]["weak_spatial_reference_path"]
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d%H%M")
    aic_git_head = current_git_head(_REPO)
    changed = science_tree_changed(_REPO, protocol["aic_code"]["git_head"])

    metadata_records = load_json(metadata_cache_path)["records"]
    control_records = read_jsonl(control_predictions_path)
    video_shas = {
        str(row["video_id"]): str(row.get("sha256", "")) for row in read_jsonl(dev_selected_path)
    }
    specs = build_video_specs(metadata_records, control_records, video_shas)

    return RunContext(
        protocol_path=protocol_path,
        model_manifest_path=model_manifest_path,
        protocol=protocol,
        control_run=control_run,
        control_predictions_path=control_predictions_path,
        metadata_cache_path=metadata_cache_path,
        dev_selected_path=dev_selected_path,
        reference_path=reference_path.resolve(),
        valid_ids_path=Path(valid_ids_path).resolve(),
        empty_ids_path=Path(empty_ids_path).resolve(),
        output_root=args.output_root.resolve(),
        run_id=run_id,
        device=args.device,
        batch_size=args.batch_size,
        valid_ids=read_video_ids(Path(valid_ids_path)),
        empty_ids=read_video_ids(Path(empty_ids_path)),
        metadata_records=metadata_records,
        specs=specs,
        aic_git_head=aic_git_head,
        science_tree_changed=changed,
    )


def run_validate_only(context: RunContext) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["formal_protocol_sha256"] = verify_sha256(
        "formal_protocol.json", context.protocol_path, FROZEN_PROTOCOL_SHA
    )
    checks["model_manifest_sha256"] = verify_sha256(
        "model_manifest.json", context.model_manifest_path, FROZEN_MODEL_MANIFEST_SHA
    )
    checks["control_predictions_sha256"] = verify_sha256(
        "control predictions", context.control_predictions_path, FROZEN_CONTROL_PREDICTIONS_SHA
    )
    checks["reference_valid_109_sha256"] = verify_sha256(
        "reference_valid_109.txt", context.valid_ids_path, FROZEN_REFERENCE_VALID_SHA
    )
    checks["empty_reference_57_sha256"] = verify_sha256(
        "empty_reference_57.txt", context.empty_ids_path, FROZEN_EMPTY_REFERENCE_SHA
    )
    checks["reference_sha256"] = file_sha256(context.reference_path)
    checks["dev_selected_sha256"] = file_sha256(context.dev_selected_path)
    checks["valid_ids_count"] = len(context.valid_ids)
    checks["empty_ids_count"] = len(context.empty_ids)
    checks["video_count"] = len(context.specs)
    checks["metadata_cache_records"] = len(context.metadata_records)
    _expect(len(context.specs) == 166, "expected 166 control videos")
    _expect(len(context.valid_ids) == 109, "expected 109 reference-valid ids")
    _expect(len(context.empty_ids) == 57, "expected 57 empty-reference ids")

    weights_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "googlenet-1378be20.pth"
    checks["googlenet_weights_present"] = weights_path.is_file()
    if weights_path.is_file():
        checks["googlenet_weights_sha256"] = verify_sha256(
            "googlenet weights", weights_path, FROZEN_GOOGLENET_SHA
        )

    for arm in (PGL, VASNET):
        entries = checkpoint_entries(context.protocol, arm)
        verify_checkpoints(arm, entries)
        checks[f"{arm}_checkpoint_count"] = len(entries)
        checks[f"{arm}_checkpoint_sha256"] = [entry["sha256"] for entry in entries]
        models = load_models(arm, entries, "cpu")
        checks[f"{arm}_models_loaded"] = len(models)
        del models

    device_ok = False
    vram_mib = None
    try:
        import torch

        device_ok = torch.cuda.is_available() if context.device.startswith("cuda") else True
        if device_ok and context.device.startswith("cuda"):
            vram_mib = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["torch_error"] = str(exc)
    checks["device"] = context.device
    checks["device_available"] = device_ok
    checks["device_total_vram_mib"] = vram_mib
    _expect(device_ok, f"device not available: {context.device}")

    context.output_root.mkdir(parents=True, exist_ok=True)
    probe = context.output_root / ".write_probe"
    atomic_write_text(probe, "ok")
    probe.unlink()
    checks["output_root_writable"] = True
    checks["control_run_exists"] = context.control_run.is_dir()
    checks["metadata_cache_exists"] = context.metadata_cache_path.is_file()
    checks["science_tree_changed_vs_frozen_head"] = context.science_tree_changed
    checks["aic_git_head"] = context.aic_git_head
    checks["frozen_science_head"] = context.protocol["aic_code"]["git_head"]
    checks["status"] = "VALIDATE_ONLY_PASS"
    return checks


def _arm_identity(context: RunContext, arm: str) -> dict[str, Any]:
    entries = checkpoint_entries(context.protocol, arm) if arm != FS0 else []
    return {
        "schema": ARM_SCHEMA,
        "arm": arm,
        "run_id": context.arm_run_id(arm),
        "run_group": context.group_id,
        "method": "FS0_KEEP_ALL" if arm == FS0 else METHOD_OF_ARM[arm],
        "protocol_sha256": FROZEN_PROTOCOL_SHA,
        "model_manifest_sha256": FROZEN_MODEL_MANIFEST_SHA,
        "control_predictions_sha256": FROZEN_CONTROL_PREDICTIONS_SHA,
        "aic_git_head": context.aic_git_head,
        "video_count": len(context.specs),
        "checkpoint_count": len(entries),
        "checkpoint_sha256": [entry["sha256"] for entry in entries],
        "sample_fps": SAMPLE_FPS,
        "feature_identity": "FEATURE_EXTRACTOR_COMPATIBLE_REPRODUCTION",
    }


def _progress(index: int, total: int, video_id: str) -> None:
    if index == 1 or index % 20 == 0 or index == total:
        print(f"[stage6.1] {index}/{total} {video_id}", flush=True)


def _peak_vram_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None
    return None


def _fs0_gate(full: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for key, expected in CONTROL_OFFICIAL_LIKE.items():
        actual = float(full[key])
        if abs(actual - expected) > FLOAT_TOLERANCE:
            mismatches.append(f"{key}: expected {expected}, got {actual}")
    frame_map = {
        "frame_n_pred": "n_pred",
        "frame_n_ref": "n_ref",
        "frame_tp": "tp",
        "frame_fp": "fp",
        "frame_fn": "fn",
        "frame_precision": "precision",
        "frame_recall": "recall",
        "frame_f1": "f1",
    }
    for key, expected_key in frame_map.items():
        actual = full[key]
        expected = CONTROL_FRAME[expected_key]
        if isinstance(expected, int):
            if int(actual) != expected:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        elif abs(float(actual) - expected) > FLOAT_TOLERANCE:
            mismatches.append(f"{key}: expected {expected}, got {actual}")
    return {
        "pass": not mismatches,
        "reproduced_official_like": {
            "macro_precision": full["macro_precision"],
            "macro_recall": full["macro_recall"],
            "macro_f": full["macro_f"],
        },
        "reproduced_frame": {key: full[key] for key in frame_map},
        "expected_official_like": CONTROL_OFFICIAL_LIKE,
        "expected_frame": CONTROL_FRAME,
        "mismatches": mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args(argv)
    try:
        context = build_context(args)
    except Stage61Error as exc:
        print(f"[stage6.1] PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        checks = run_validate_only(context)
        atomic_write_json(context.output_root / "_state" / "validate_only.json", checks)
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0

    try:
        return _run_formal(context, args)
    except Stage61Error as exc:
        print(f"[stage6.1] STOP: {exc}", file=sys.stderr)
        return 2


def _run_formal(context: RunContext, args: argparse.Namespace) -> int:
    arms = resolve_arms(args.arm)
    runtime = RuntimeTracker()
    context.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        context.output_root / "_state" / "run_identity.json",
        {
            "schema": RUNNER_SCHEMA,
            "run_group": context.group_id,
            "run_id": context.run_id,
            "arms": list(arms),
            "protocol_sha256": FROZEN_PROTOCOL_SHA,
            "model_manifest_sha256": FROZEN_MODEL_MANIFEST_SHA,
            "control_predictions_sha256": FROZEN_CONTROL_PREDICTIONS_SHA,
            "aic_git_head": context.aic_git_head,
            "frozen_science_head": context.protocol["aic_code"]["git_head"],
            "science_tree_changed_vs_frozen_head": context.science_tree_changed,
        },
    )

    results: dict[str, dict[str, Any]] = {}

    if FS0 in arms:
        with runtime.phase("fs0_reconstruction"):
            fs0_records = fs0_reconstruct([spec.control_record for spec in context.specs])
        fs0_identity = _arm_identity(context, FS0)
        if args.resume and arm_is_resumable(arm_dir(context.output_root, FS0), fs0_identity):
            print("[stage6.1] FS0 arm reused from resume", flush=True)
            runtime.reused["fs0_videos"] = len(context.specs)
        else:
            write_arm_output(context.output_root, FS0, fs0_records, fs0_identity)
            runtime.add_count("fs0_frames_written", sum(len(r["predictions"]) for r in fs0_records))
        fs0_path = arm_dir(context.output_root, FS0) / "predictions" / "predictions.jsonl"
        with runtime.phase("fs0_evaluation"):
            fs0_full = evaluate_group(fs0_path, context.reference_path)
        gate = _fs0_gate(fs0_full)
        results["fs0"] = {"full166": fs0_full, "gate": gate}
        atomic_write_json(arm_dir(context.output_root, FS0) / "audit" / "fs0_gate.json", gate)
        if not gate["pass"]:
            raise Stage61Error(f"FS0 reproduction gate FAILED: {gate['mismatches']}")
        print(f"[stage6.1] FS0 gate PASS (Full166 macro F={fs0_full['macro_f']:.12f})", flush=True)

    need_features = any(arm in arms for arm in (PGL, VASNET))
    cache: SharedFeatureCache | None = None
    if need_features:
        extractor = GoogleNetPool5Extractor(device=context.device, batch_size=context.batch_size)
        cache = SharedFeatureCache(
            context.output_root / "_feature_cache",
            extractor,
            aic_git_head=context.aic_git_head,
            resume=args.resume,
        )

        def feature_fn(spec: VideoSpec) -> np.ndarray:
            assert cache is not None
            return cache.features(spec)
    else:
        feature_fn = lambda spec: np.zeros((len(spec.sample_frames), FEATURE_DIM), dtype=np.float32)  # noqa: E731

    for arm in (PGL, VASNET):
        if arm not in arms:
            continue
        entries = checkpoint_entries(context.protocol, arm)
        expected_identity = _arm_identity(context, arm)
        directory = arm_dir(context.output_root, arm)
        if args.resume and arm_is_resumable(directory, expected_identity):
            print(f"[stage6.1] {arm} arm reused from resume", flush=True)
            runtime.reused[f"{arm}_videos"] = len(context.specs)
            continue
        with runtime.phase(f"{arm}_model_load"):
            models = load_models(arm, entries, context.device)

        def score_fn(method_arm: str, features: np.ndarray, _models: list[Any] = models, _arm: str = arm) -> np.ndarray:
            with runtime.phase(f"{_arm}_inference"):
                return ensemble_scores(_arm, _models, features, context.device)

        with runtime.phase(f"{arm}_kts_knapsack_fs0"):
            predictions, selections = run_literature_arm(
                arm, context.specs, feature_fn, score_fn, progress=_progress
            )
        audit = audit_subset_invariants(context.specs, predictions, selections)
        predictions_path, _payload = write_arm_output(context.output_root, arm, predictions, expected_identity)
        atomic_write_json(directory / "audit" / "subset_invariants.json", audit)
        runtime.add_count(f"{arm}_frames_written", sum(len(r["predictions"]) for r in predictions))
        runtime.add_count(f"{arm}_videos", len(context.specs))
        del models
        with runtime.phase(f"{arm}_evaluation"):
            full = evaluate_group(predictions_path, context.reference_path)
            valid = evaluate_group(predictions_path, context.reference_path, video_ids=context.valid_ids)
            empty = evaluate_group(predictions_path, context.reference_path, video_ids=context.empty_ids)
        results[arm] = {"full166": full, "valid109": valid, "empty57": empty, "audit": audit}

    if cache is not None:
        runtime.counts["feature_cache_hits"] = cache.hits
        runtime.counts["feature_cache_misses"] = cache.misses
        runtime.phases["feature_extraction"] = cache.extract_seconds
        runtime.counts["feature_frames_extracted"] = cache.total_frames

    if FS0 not in arms:
        print("[stage6.1] literature arm run without FS0; comparison skipped", flush=True)
        atomic_write_json(context.output_root / "_state" / "results.json", {"arms": arms, "results": results})
        return 0

    with runtime.phase("comparison"):
        comparison = build_comparison(context, results)
        write_comparison_outputs(context, comparison, runtime)

    atomic_write_json(
        context.output_root / "_state" / "results.json", {"arms": arms, "results": results}
    )
    print("[stage6.1] STAGE6_1_FORMAL_COMPLETE", flush=True)
    return 0


def build_comparison(context: RunContext, results: Mapping[str, Any]) -> dict[str, Any]:
    fs0_path = arm_dir(context.output_root, FS0) / "predictions" / "predictions.jsonl"
    fs0_valid = evaluate_group(fs0_path, context.reference_path, video_ids=context.valid_ids)
    fs0_empty = evaluate_group(fs0_path, context.reference_path, video_ids=context.empty_ids)
    fs0 = results["fs0"]
    comparison: dict[str, Any] = {
        "fs0": {
            "full166": fs0["full166"],
            "valid109": fs0_valid,
            "empty57": fs0_empty,
            "gate": fs0["gate"],
        },
        "challengers": {},
    }
    for arm in (PGL, VASNET):
        if arm not in results:
            comparison["challengers"][arm] = {"status": "NOT_EVALUATED"}
            continue
        entry = results[arm]
        comparison["challengers"][arm] = {
            "status": "EVALUATED",
            "full166": entry["full166"],
            "valid109": entry["valid109"],
            "empty57": entry["empty57"],
            "paired_full166": compute_paired(fs0["full166"], entry["full166"]),
            "paired_valid109": compute_paired(fs0_valid, entry["valid109"]),
            "recall_guard_valid109": recall_guard(entry["valid109"]),
            "empty_reference_audit": empty_reference_audit(fs0_empty, entry["empty57"], context.empty_ids),
        }
    return comparison


def _metric_row(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm": name,
        "full166_macro_P": metrics["full166"]["macro_precision"],
        "full166_macro_R": metrics["full166"]["macro_recall"],
        "full166_macro_F": metrics["full166"]["macro_f"],
        "valid109_macro_P": metrics["valid109"]["macro_precision"],
        "valid109_macro_R": metrics["valid109"]["macro_recall"],
        "valid109_macro_F": metrics["valid109"]["macro_f"],
        "frame_n_pred": metrics["full166"]["frame_n_pred"],
        "frame_n_ref": metrics["full166"]["frame_n_ref"],
        "frame_tp": metrics["full166"]["frame_tp"],
        "frame_fp": metrics["full166"]["frame_fp"],
        "frame_fn": metrics["full166"]["frame_fn"],
        "frame_precision": metrics["full166"]["frame_precision"],
        "frame_recall": metrics["full166"]["frame_recall"],
        "frame_f1": metrics["full166"]["frame_f1"],
    }


def write_comparison_outputs(
    context: RunContext,
    comparison: Mapping[str, Any],
    runtime: RuntimeTracker,
) -> None:
    comparison_dir = context.output_root / "04_Comparison"
    tables = comparison_dir / "tables"
    figures = comparison_dir / "figures"
    runtime_dir = comparison_dir / "runtime"
    for directory in (tables, figures, runtime_dir, comparison_dir / "audit", comparison_dir / "report"):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [_metric_row("FS0", comparison["fs0"])]
    for arm in (PGL, VASNET):
        entry = comparison["challengers"][arm]
        if entry["status"] == "EVALUATED":
            rows.append(_metric_row(arm.upper(), entry))
    write_csv(
        tables / "arm_comparison.csv",
        ["arm", "full166_macro_P", "full166_macro_R", "full166_macro_F",
         "valid109_macro_P", "valid109_macro_R", "valid109_macro_F",
         "frame_n_pred", "frame_n_ref", "frame_tp", "frame_fp", "frame_fn",
         "frame_precision", "frame_recall", "frame_f1"],
        rows,
    )

    for arm in (PGL, VASNET):
        entry = comparison["challengers"][arm]
        if entry["status"] != "EVALUATED":
            continue
        write_csv(
            tables / f"paired_valid109_{arm}.csv",
            ["video_id", "delta_f"],
            [
                {"video_id": vid, "delta_f": delta}
                for vid, delta in sorted(entry["paired_valid109"]["per_video_delta_f"].items())
            ],
        )
        write_csv(tables / f"worst_k_regression_{arm}.csv", ["video_id", "delta_f"], entry["paired_valid109"]["worst_k"])
        write_csv(
            tables / f"empty57_{arm}.csv",
            ["video_id", "n_ref", "n_pred_before", "n_pred_after", "double_empty_transition"],
            entry["empty_reference_audit"]["rows"],
        )

    atomic_write_json(
        comparison_dir / "audit" / "comparison.json",
        {
            "run_group": context.group_id,
            "run_id": context.run_id,
            "fs0_gate": comparison["fs0"]["gate"],
            "challengers": {
                arm: {
                    key: value
                    for key, value in entry.items()
                    if key not in ("full166", "valid109", "empty57")
                }
                for arm, entry in comparison["challengers"].items()
            },
        },
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generate_figures(comparison, figures, plt)
    except Exception as exc:  # pragma: no cover - plotting environment dependent
        atomic_write_json(comparison_dir / "audit" / "figure_error.json", {"error": str(exc)})

    runtime_payload = runtime.as_dict(
        device=context.device,
        batch_size=context.batch_size,
        peak_vram_mib=_peak_vram_mib(),
    )
    atomic_write_json(runtime_dir / "runtime.json", runtime_payload)
    atomic_write_text(
        comparison_dir / "report" / "Stage6_1_Literature_Frame_Selection_Formal_Report.md",
        render_report(context, comparison, runtime_payload),
    )


def generate_figures(comparison: Mapping[str, Any], figures: Path, plt: Any) -> None:
    evaluated = [arm for arm in (PGL, VASNET) if comparison["challengers"][arm]["status"] == "EVALUATED"]
    arms = ["FS0"] + [arm.upper() for arm in evaluated]
    full = [comparison["fs0"]["full166"]] + [comparison["challengers"][arm]["full166"] for arm in evaluated]
    valid = [comparison["fs0"]["valid109"]] + [comparison["challengers"][arm]["valid109"] for arm in evaluated]
    xs = list(range(len(arms)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, (key, label) in enumerate((("macro_precision", "P"), ("macro_recall", "R"), ("macro_f", "F"))):
        ax.bar([x + (offset - 1) * width for x in xs], [m[key] for m in full], width=width, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.legend(); ax.set_ylim(0, 0.6)
    ax.set_title("Full166 official_like macro P/R/F"); ax.set_ylabel("score")
    fig.tight_layout(); fig.savefig(figures / "overall_score_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, (key, label) in enumerate((("macro_precision", "P"), ("macro_recall", "R"), ("macro_f", "F"))):
        ax.bar([x + (offset - 1) * width for x in xs], [m[key] for m in valid], width=width, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.legend(); ax.set_ylim(0, 0.6)
    ax.set_title("Reference-valid109 official_like macro P/R/F (primary)"); ax.set_ylabel("score")
    fig.tight_layout(); fig.savefig(figures / "valid109_official_like_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, (key, label) in enumerate((("frame_precision", "frame P"), ("frame_recall", "frame R"), ("frame_f1", "frame F1"))):
        ax.bar([x + (offset - 1) * width for x in xs], [m[key] for m in full], width=width, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.legend(); ax.set_ylim(0, 1)
    ax.set_title("Full166 frame-membership P/R/F1"); ax.set_ylabel("score")
    fig.tight_layout(); fig.savefig(figures / "frame_precision_recall_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    base = full[0]["frame_n_pred"]
    ax.bar(xs, [m["frame_n_pred"] for m in full], color="#4c72b0")
    for x, m in zip(xs, full):
        ax.text(x, m["frame_n_pred"], str(m["frame_n_pred"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.set_ylabel("N_pred")
    ax2 = ax.twinx()
    ax2.plot(xs, [1 - m["frame_n_pred"] / base for m in full], "r-o", label="drop ratio vs FS0")
    ax2.set_ylabel("drop ratio"); ax2.legend(loc="upper right")
    ax.set_title("Prediction count and drop ratio (Full166)")
    fig.tight_layout(); fig.savefig(figures / "prediction_count_drop.png", dpi=140); plt.close(fig)

    for arm in evaluated:
        deltas = comparison["challengers"][arm]["paired_valid109"]["per_video_delta_f"]
        ordered = sorted(deltas.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(ordered)), [value for _, value in ordered], color="#c44e52")
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("video (sorted by delta F)"); ax.set_ylabel("delta F (challenger - FS0)")
        ax.set_title(f"Reference-valid109 per-video delta F: {arm.upper()} vs FS0")
        fig.tight_layout(); fig.savefig(figures / f"per_video_delta_f_{arm}.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in evaluated:
        recalls = [
            row["matched"] / row["n_ref"]
            for row in comparison["challengers"][arm]["valid109"]["per_video"].values()
            if row["n_ref"] > 0
        ]
        ax.hist(recalls, bins=20, alpha=0.5, label=f"{arm.upper()} frame recall")
    ax.set_xlabel("per-video frame-membership recall"); ax.set_ylabel("video count")
    ax.set_title("Reference-valid109 recall regression distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "recall_regression_distribution.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if evaluated:
        names = [arm.upper() for arm in evaluated]
        for offset, arm in enumerate(evaluated):
            audit = comparison["challengers"][arm]["empty_reference_audit"]
            ax.bar([offset - 0.2, offset + 0.2], [audit["n_pred_before"], audit["n_pred_after"]], width=0.4, label=names[offset])
        ax.set_xticks(range(len(evaluated))); ax.set_xticklabels(names)
        ax.set_ylabel("total N_pred (empty-reference 57)"); ax.legend()
        ax.set_title("Empty-reference57 behavior (double-empty is NOT scientific gain)")
    fig.tight_layout(); fig.savefig(figures / "empty_reference_behavior.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def decision_for(entry: Mapping[str, Any], fs0_valid: Mapping[str, Any]) -> str:
    if entry["status"] != "EVALUATED":
        return "NOT_EVALUATED / BLOCKED"
    paired = entry["paired_valid109"]
    guard = entry["recall_guard_valid109"]
    full = entry["full166"]
    if paired["win"] > paired["loss"] and full["frame_precision"] > CONTROL_FRAME["precision"] and guard["recall_eq_0"] == 0:
        return "PROMOTION_CANDIDATE"
    return "NO_PROMOTION"


def render_report(context: RunContext, comparison: Mapping[str, Any], runtime_payload: Mapping[str, Any]) -> str:
    fs0 = comparison["fs0"]
    lines: list[str] = []
    add = lines.append
    add("# Stage 6.1 Literature Frame Selection - Dev166 Formal Report")
    add("")
    add("## 1. Protocol identity")
    add("")
    add(f"- run group: `{context.group_id}`")
    add(f"- formal_protocol.json SHA256: `{FROZEN_PROTOCOL_SHA}`")
    add(f"- model_manifest.json SHA256: `{FROZEN_MODEL_MANIFEST_SHA}`")
    add(f"- control predictions SHA256: `{FROZEN_CONTROL_PREDICTIONS_SHA}`")
    add(f"- GoogLeNet weights SHA256: `{FROZEN_GOOGLENET_SHA}`")
    add("")
    add("## 2. Runner / HEAD identity")
    add("")
    add("- runner: `scripts/experiments/run_stage6_1_literature_frame_selection.py`")
    add(f"- AIC git HEAD at run: `{context.aic_git_head}`")
    add(f"- frozen science HEAD: `{context.protocol['aic_code']['git_head']}`")
    add(f"- src/ changed vs frozen head: `{context.science_tree_changed}`")
    add("")
    add("## 3. Frozen control")
    add("")
    add("Stage 6.0 run `vhicraft_dev166_202609141233`: 166 videos, 52,019 FS0 frames, reused byte-for-byte.")
    add("")
    add("## 4. FS0 reproduction gate")
    add("")
    gate = fs0["gate"]
    add(f"- result: {'PASS' if gate['pass'] else 'FAIL'}")
    add(f"- Full166 macro P/R/F: {_fmt(fs0['full166']['macro_precision'],12)} / {_fmt(fs0['full166']['macro_recall'],12)} / {_fmt(fs0['full166']['macro_f'],12)}")
    add(f"- frame N_pred/N_ref/TP/FP/FN: {fs0['full166']['frame_n_pred']} / {fs0['full166']['frame_n_ref']} / {fs0['full166']['frame_tp']} / {fs0['full166']['frame_fp']} / {fs0['full166']['frame_fn']}")
    add("")
    add("## 5. PGL results")
    add("")
    add(_challenger_section(comparison, PGL, fs0))
    add("## 6. VAS results")
    add("")
    add(_challenger_section(comparison, VASNET, fs0))
    add("## 7. Full166 metrics")
    add("")
    add(_comparison_table(comparison))
    add("## 8. Valid109 metrics (primary diagnostic)")
    add("")
    add(_valid_table(comparison))
    add("## 9. Empty57 metrics")
    add("")
    add(_empty_table(comparison))
    add("## 10. Frame metrics")
    add("")
    add("See Full166 frame columns in section 7; per-arm tables under `tables/`.")
    add("")
    add("## 11. Paired analysis")
    add("")
    for arm in (PGL, VASNET):
        entry = comparison["challengers"][arm]
        if entry["status"] != "EVALUATED":
            add(f"- {arm.upper()}: NOT_EVALUATED")
            continue
        p = entry["paired_valid109"]
        add(f"- {arm.upper()} valid109: Win={p['win']} Tie={p['tie']} Loss={p['loss']}; "
            f"max gain={_fmt(p['max_gain']['delta_f']) if p['max_gain'] else 'n/a'}, "
            f"max regression={_fmt(p['max_regression']['delta_f']) if p['max_regression'] else 'n/a'}")
    add("")
    add("## 12. Recall guard")
    add("")
    for arm in (PGL, VASNET):
        entry = comparison["challengers"][arm]
        if entry["status"] != "EVALUATED":
            add(f"- {arm.upper()}: NOT_EVALUATED")
            continue
        g = entry["recall_guard_valid109"]
        add(f"- {arm.upper()} valid109 frame recall: <0.8={g['recall_lt_0_8']}, <0.5={g['recall_lt_0_5']}, =0={g['recall_eq_0']}")
    add("")
    add("## 13. Runtime")
    add("")
    for phase, seconds in runtime_payload["phases_seconds"].items():
        add(f"- {phase}: {seconds:.2f}s")
    add(f"- total: {runtime_payload['total_seconds']:.2f}s")
    add("")
    add("## 14. Figures")
    add("")
    for name in FIGURE_NAMES:
        add(f"- `figures/{name}`")
    add("")
    add("## 15. Limitations")
    add("")
    add("- Dev166 is a weak-reference development proxy; official_like is NOT_OFFICIAL_SCORE.")
    add("- PGL-SUM uses official released pretrained weights; VASNet uses SECONDARY XAI-SUM weights (NOT_PROVEN_ORIGINAL_BOX).")
    add("- Features are FEATURE_EXTRACTOR_COMPATIBLE_REPRODUCTION, not exact original benchmark features.")
    add("- KTS uses AIC_KTS_COMPATIBILITY_PREREGISTRATION, not the authors' unpublished H5 parameters.")
    add("- Empty-reference57 double-empty outcomes are not scientific gains (DOUBLE_EMPTY_NOT_SCIENTIFIC_GAIN).")
    add("")
    add("## 16. Scientific decision")
    add("")
    for arm in (PGL, VASNET):
        add(f"- {arm.upper()}: {decision_for(comparison['challengers'][arm], comparison['fs0']['valid109'])}")
    add("")
    add("## 17. Integrity audit")
    add("")
    add("- protocol changed = NO")
    add("- Dev checkpoint selection = NO")
    add("- Qwen rerun = NO")
    add("- RT-DETR rerun = NO")
    add("- bbox changed = NO")
    add("- Hard229 / Heldout / Official Test accessed = NO")
    add("")
    add("## 18. Next step")
    add("")
    add("Await user decision on whether any challenger enters the Hard229 gate. Stage 6.1 Dev166 Formal stops here.")
    add("")
    return "\n".join(lines)


def _challenger_section(comparison: Mapping[str, Any], arm: str, fs0: Mapping[str, Any]) -> str:
    entry = comparison["challengers"][arm]
    if entry["status"] != "EVALUATED":
        return "NOT_EVALUATED\n"
    full = entry["full166"]
    valid = entry["valid109"]
    empty = entry["empty57"]
    lines = [
        f"- Full166: P={_fmt(full['macro_precision'])}, R={_fmt(full['macro_recall'])}, F={_fmt(full['macro_f'])}",
        f"- Valid109: P={_fmt(valid['macro_precision'])}, R={_fmt(valid['macro_recall'])}, F={_fmt(valid['macro_f'])}",
        f"- Frame: N_pred={full['frame_n_pred']}, N_ref={full['frame_n_ref']}, TP={full['frame_tp']}, FP={full['frame_fp']}, FN={full['frame_fn']}, "
        f"P={_fmt(full['frame_precision'])}, R={_fmt(full['frame_recall'])}, F1={_fmt(full['frame_f1'])}",
        f"- Empty57 N_pred before/after: {entry['empty_reference_audit']['n_pred_before']}/{entry['empty_reference_audit']['n_pred_after']} "
        f"(double-empty transitions={entry['empty_reference_audit']['double_empty_transitions']})",
        "",
    ]
    return "\n".join(lines)


def _row(comparison: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    if arm == "FS0":
        return comparison["fs0"]
    return comparison["challengers"][arm]


def _comparison_table(comparison: Mapping[str, Any]) -> str:
    lines = ["| Metric | FS0 | PGL | VAS |", "| --- | --- | --- | --- |"]
    arms = ["FS0", PGL, VASNET]
    for key, label in (("macro_f", "Full166 macro F"), ("frame_precision", "Frame P"),
                       ("frame_recall", "Frame R"), ("frame_f1", "Frame F1"),
                       ("frame_n_pred", "N_pred"), ("frame_tp", "TP"), ("frame_fp", "FP"), ("frame_fn", "FN")):
        cells = []
        for arm in arms:
            entry = _row(comparison, arm)
            if entry.get("status") == "NOT_EVALUATED":
                cells.append("n/a")
            else:
                cells.append(_fmt(entry["full166"][key]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _empty_count(metrics: Mapping[str, Any]) -> int:
    return sum(1 for row in metrics["per_video"].values() if row["n_pred"] == 0)


def _valid_table(comparison: Mapping[str, Any]) -> str:
    lines = ["| Metric | FS0 | PGL | VAS |", "| --- | --- | --- | --- |"]

    def cell(arm: str, key: str) -> str:
        entry = _row(comparison, arm)
        return "n/a" if entry.get("status") == "NOT_EVALUATED" else _fmt(entry["valid109"][key])

    for key, label in (
        ("macro_precision", "Valid109 macro P"),
        ("macro_recall", "Valid109 macro R"),
        ("macro_f", "Valid109 macro F"),
    ):
        lines.append(f"| {label} | " + " | ".join(cell(arm, key) for arm in ("FS0", PGL, VASNET)) + " |")

    wtl = []
    for arm in ("FS0", PGL, VASNET):
        entry = _row(comparison, arm)
        if arm == "FS0" or entry.get("status") != "EVALUATED":
            wtl.append("-")
        else:
            p = entry["paired_valid109"]
            wtl.append(f"{p['win']}/{p['tie']}/{p['loss']}")
    lines.append("| Valid109 W/T/L vs FS0 | - | " + " | ".join(wtl[1:]) + " |")
    lines.append("")
    return "\n".join(lines)


def _empty_table(comparison: Mapping[str, Any]) -> str:
    lines = ["| Empty57 | FS0 | PGL | VAS |", "| --- | --- | --- | --- |"]
    fs0_empty = comparison["fs0"]["empty57"]
    rows = [
        ("N_pred (before -> after)", _fmt(fs0_empty["frame_n_pred"]), "n_pred_before", "n_pred_after"),
    ]
    for label, base_value, before_key, after_key in rows:
        cells = [base_value]
        for arm in (PGL, VASNET):
            entry = _row(comparison, arm)
            if entry.get("status") != "EVALUATED":
                cells.append("n/a")
            else:
                cells.append(f"{entry['empty_reference_audit'][before_key]} -> {entry['empty_reference_audit'][after_key]}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    for label, base_value, key in (
        ("drop ratio", "0.000000", "drop_ratio"),
        ("empty outputs", str(_empty_count(fs0_empty)), "empty_after"),
        ("double-empty transitions", "0", "double_empty_transitions"),
    ):
        cells = [base_value]
        for arm in (PGL, VASNET):
            entry = _row(comparison, arm)
            cells.append("n/a" if entry.get("status") != "EVALUATED" else _fmt(entry["empty_reference_audit"][key]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
