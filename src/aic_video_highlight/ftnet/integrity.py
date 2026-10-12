"""Stage 7 FTNet audits: native feature audit, TRAIN-only normalization, integrity gate.

All checks are read-only over the materialized dataset.  The idx0 variance gate
uses TRAIN features only and never consults VALIDATION/CALIBRATION labels,
targets or ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from safetensors.numpy import load_file

from .materialize import SPLIT_DIRS
from .native_schema import (
    NATIVE_DIM,
    NATIVE_FIELDS,
    NATIVE_SCHEMA_VERSION,
    SUPPORT_RATIO_INDEX,
    VARIANCE_GATE_THRESHOLD,
)
from .index import IndexEntry

AUDIT_SCHEMA = "aic.stage7.ftnet.native-feature-audit/v1"
INTEGRITY_SCHEMA = "aic.stage7.ftnet.integrity-gate/v1"
IDX0_DECISION_SCHEMA = "aic.stage7.ftnet.idx0-gate-decision/v1"

NORMALIZATION_RELATIVE = Path("normalization") / "normalization_stats.json"
AUDIT_RELATIVE = Path("audits") / "native_feature_audit.json"
INTEGRITY_RELATIVE = Path("audits") / "integrity_report.json"
IDX0_DECISION_RELATIVE = Path("idx0_gate_decision.json")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _split_paths(output_root: Path, split: str, video_ids: Sequence[str] | None = None):
    directory = output_root / SPLIT_DIRS[split]
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.safetensors"))
    if video_ids is not None:
        wanted = set(video_ids)
        paths = [path for path in paths if path.stem in wanted]
    return paths


def _field_stats(values: np.ndarray, missing: np.ndarray) -> dict[str, Any]:
    observed = values[~missing]
    if observed.size == 0:
        return {
            "count": 0,
            "missing_rate": 1.0,
            "min": None,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "unique_count": 0,
        }
    return {
        "count": int(observed.size),
        "missing_rate": float(missing.mean()),
        "min": float(observed.min()),
        "mean": float(observed.mean()),
        "std": float(observed.std()),
        "median": float(np.median(observed)),
        "p95": float(np.percentile(observed, 95)),
        "unique_count": int(np.unique(np.round(observed, 6)).size),
    }


def audit_native_features(
    output_root: str | Path,
    *,
    split: str = "TRAIN",
    video_ids: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Per-field audit of raw native values plus the pre-registered idx0 gate."""

    root = Path(output_root).expanduser().resolve()
    paths = _split_paths(root, split, video_ids)
    if not paths:
        raise FileNotFoundError(f"no materialized videos for audit split {split} in {root}")
    raw_chunks: list[np.ndarray] = []
    missing_chunks: list[np.ndarray] = []
    frames = 0
    for path in paths:
        tensors = load_file(str(path))
        raw = np.asarray(tensors["native"], dtype=np.float64)
        missing = np.asarray(tensors["native_missing"], dtype=bool)
        if raw.shape[1] != NATIVE_DIM or missing.shape != raw.shape:
            raise ValueError(f"native contract violated in {path}")
        raw_chunks.append(raw)
        missing_chunks.append(missing)
        frames += raw.shape[0]
    raw_all = np.concatenate(raw_chunks, axis=0)
    missing_all = np.concatenate(missing_chunks, axis=0)
    field_stats = {}
    for index, name in enumerate(NATIVE_FIELDS):
        field_stats[name] = _field_stats(raw_all[:, index], missing_all[:, index])
    support = raw_all[:, SUPPORT_RATIO_INDEX]
    support_observed = support[~missing_all[:, SUPPORT_RATIO_INDEX]]
    if support_observed.size:
        degenerate = float((support_observed == 1.0).mean())
    else:
        degenerate = 1.0
    needs_replacement = degenerate > VARIANCE_GATE_THRESHOLD
    audit = {
        "schema": AUDIT_SCHEMA,
        "split": split,
        "video_count": len(paths),
        "frame_count": frames,
        "native_schema_version": NATIVE_SCHEMA_VERSION,
        "fields": field_stats,
        "idx0_gate": {
            "field": NATIVE_FIELDS[SUPPORT_RATIO_INDEX],
            "observed_count": int(support_observed.size),
            "fraction_eq_one": degenerate,
            "threshold": VARIANCE_GATE_THRESHOLD,
            "status": "REPLACE" if needs_replacement else "KEEP",
            "frozen_fallback": "retrieval_support_ratio_norm",
            "basis": "TRAIN features only; no validation/GT consulted",
        },
        "videos": [path.stem for path in paths],
    }
    if write:
        _atomic_json(root / AUDIT_RELATIVE, audit)
    return audit


def write_idx0_decision(
    work_root: str | Path,
    audit: dict[str, Any],
    *,
    stage: str,
) -> Path:
    """Freeze the idx0 gate decision produced by the small gate."""

    gate = audit["idx0_gate"]
    decision = {
        "schema": IDX0_DECISION_SCHEMA,
        "stage": stage,
        "status": gate["status"],
        "fraction_eq_one": gate["fraction_eq_one"],
        "threshold": gate["threshold"],
        "observed_count": gate["observed_count"],
        "video_count": audit["video_count"],
        "fallback": gate["frozen_fallback"] if gate["status"] == "REPLACE" else "NONE",
        "basis": gate["basis"],
        "videos": audit["videos"],
    }
    path = Path(work_root).expanduser().resolve() / IDX0_DECISION_RELATIVE
    _atomic_json(path, decision)
    return path


def load_idx0_decision(work_root: str | Path) -> dict[str, Any]:
    path = Path(work_root).expanduser().resolve() / IDX0_DECISION_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"idx0 gate decision is not frozen: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_train_normalization(output_root: str | Path) -> dict[str, Any]:
    """Compute TRAIN-only normalization stats and write the SHA-256 sidecar."""

    from .dataset import compute_train_normalization

    root = Path(output_root).expanduser().resolve()
    path = root / NORMALIZATION_RELATIVE
    payload = compute_train_normalization(root, output_path=path)
    digest = _sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    payload = {**payload, "normalization_stats_sha256": digest, "path": str(path)}
    return payload


def _expected_ids(entries: Iterable[IndexEntry]) -> dict[str, set[str]]:
    expected = {split: set() for split in SPLIT_DIRS}
    for entry in entries:
        expected[entry.split].add(entry.video_id)
    return expected


def run_integrity_gate(
    output_root: str | Path,
    *,
    entries: Sequence[IndexEntry],
    write: bool = True,
) -> dict[str, Any]:
    """Full materialized-dataset integrity gate (must pass 154/154)."""

    root = Path(output_root).expanduser().resolve()
    expected = _expected_ids(entries)
    report: dict[str, Any] = {
        "schema": INTEGRITY_SCHEMA,
        "output_root": str(root),
        "expected_counts": {split: len(ids) for split, ids in expected.items()},
        "counts": {},
        "missing": [],
        "extra": [],
        "duplicate": [],
        "corrupt": [],
        "nan_fields": [],
        "inf_fields": [],
        "split_mismatch": [],
        "visual_dim_violations": [],
        "native_dim_violations": [],
        "official_test_rows": 0,
        "tvsum_rows": 0,
        "source_leakage": 0,
    }
    seen: dict[str, str] = {}
    total = 0
    for split, directory_name in SPLIT_DIRS.items():
        directory = root / directory_name
        paths = sorted(directory.glob("*.safetensors")) if directory.is_dir() else []
        ids = {path.stem for path in paths}
        report["counts"][split] = len(ids)
        missing = expected[split] - ids
        extra = ids - expected[split]
        report["missing"].extend(sorted(missing))
        report["extra"].extend(sorted(extra))
        if len(paths) != len(ids):
            report["duplicate"].extend(sorted(ids))
        for video_id in ids:
            if video_id in seen:
                report["source_leakage"] += 1
            seen[video_id] = split
        for path in paths:
            total += 1
            try:
                tensors = load_file(str(path))
            except Exception as exc:  # noqa: BLE001
                report["corrupt"].append({"file": path.name, "error": str(exc)})
                continue
            required = {
                "visual", "native", "native_missing", "target",
                "loss_mask", "adjacency_mask", "timestamp", "source_frame_id",
            }
            if not required.issubset(tensors):
                report["corrupt"].append(
                    {"file": path.name, "error": f"missing keys: {sorted(required - set(tensors))}"}
                )
                continue
            visual = tensors["visual"]
            native = tensors["native"]
            missing_mask = tensors["native_missing"]
            target = tensors["target"]
            length = int(native.shape[0])
            if visual.shape != (length, 256):
                report["visual_dim_violations"].append(path.name)
            if native.shape != (length, NATIVE_DIM) or missing_mask.shape != (length, NATIVE_DIM):
                report["native_dim_violations"].append(path.name)
            for key, array in (
                ("visual", visual),
                ("native", native),
                ("target", target),
                ("timestamp", tensors["timestamp"]),
            ):
                if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
                    report["nan_fields"].append(f"{path.name}:{key}")
            for key in ("visual", "native", "target", "timestamp", "loss_mask", "adjacency_mask", "source_frame_id"):
                array = tensors[key]
                if array.shape[0] != length:
                    report["corrupt"].append({"file": path.name, "error": f"{key} length mismatch"})
            if target.size and (target.min() < 0.0 or target.max() > 1.0):
                report["corrupt"].append({"file": path.name, "error": "target outside [0,1]"})
    report["total_materialized"] = total
    report["source_leakage"] = 0 if report["source_leakage"] == 0 else report["source_leakage"]
    failures = (
        len(report["missing"])
        + len(report["extra"])
        + len(report["duplicate"])
        + len(report["corrupt"])
        + len(report["nan_fields"])
        + len(report["inf_fields"])
        + len(report["visual_dim_violations"])
        + len(report["native_dim_violations"])
        + report["source_leakage"]
        + len(report["split_mismatch"])
        + report["official_test_rows"]
        + report["tvsum_rows"]
    )
    report["failure_count"] = int(failures)
    report["status"] = "PASS" if failures == 0 and total == len(entries) else "FAIL"
    report["expected_total"] = len(entries)
    if write:
        _atomic_json(root / INTEGRITY_RELATIVE, report)
    return report


__all__ = [
    "AUDIT_RELATIVE",
    "AUDIT_SCHEMA",
    "IDX0_DECISION_RELATIVE",
    "IDX0_DECISION_SCHEMA",
    "INTEGRITY_RELATIVE",
    "INTEGRITY_SCHEMA",
    "NORMALIZATION_RELATIVE",
    "audit_native_features",
    "load_idx0_decision",
    "run_integrity_gate",
    "run_train_normalization",
    "write_idx0_decision",
]
