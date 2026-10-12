"""Policy-enforcing dataset adapter skeletons for FTNet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .highlight_targets import HighlightTargetAdapter, HighlightTargetStrategy


class DatasetNotReadyError(RuntimeError):
    """Raised when an external dataset or sample is not materially ready."""


class DatasetPolicyError(RuntimeError):
    """Raised when an attempted dataset use violates the experiment policy."""


def _read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetNotReadyError(f"required dataset file is missing: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetNotReadyError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise DatasetNotReadyError(
                f"missing sample_id at {path}:{line_number}"
            )
        if sample_id in rows:
            raise DatasetNotReadyError(f"duplicate sample_id {sample_id} in {path}")
        rows[sample_id] = row
    return rows


def _require_row(
    rows: dict[str, dict[str, Any]],
    sample_id: str,
    manifest_name: str,
) -> dict[str, Any]:
    try:
        return rows[sample_id]
    except KeyError as exc:
        raise DatasetNotReadyError(
            f"sample {sample_id} is absent from {manifest_name}"
        ) from exc


@dataclass(frozen=True)
class YouTubeHighlightSample:
    sample_id: str
    split: str
    targets: tuple[float, ...]
    target_strategy: HighlightTargetStrategy
    raw_record: dict[str, Any]
    alignment_record: dict[str, Any]


class YouTubeHighlightsAdapter:
    REQUIRED_FILES = (
        Path("manifests/raw_manifest.jsonl"),
        Path("manifests/alignment_manifest.jsonl"),
        Path("manifests/split_manifest.jsonl"),
        Path("annotations/mturk_labels.jsonl"),
    )

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def validate_ready(self) -> None:
        for relative_path in self.REQUIRED_FILES:
            path = self.dataset_root / relative_path
            if not path.is_file():
                raise DatasetNotReadyError(
                    f"required {relative_path.stem} file is missing: {path}"
                )

    def load_sample(
        self,
        sample_id: str,
        *,
        split: str,
        target_adapter: HighlightTargetAdapter,
        for_training: bool = True,
    ) -> YouTubeHighlightSample:
        self.validate_ready()
        raw = _require_row(
            _read_jsonl_index(self.dataset_root / self.REQUIRED_FILES[0]),
            sample_id,
            "raw_manifest",
        )
        status = str(raw.get("availability_status", "")).strip().lower()
        if status != "available":
            raise DatasetNotReadyError(
                f"sample {sample_id} is not available (status={status or '<missing>'})"
            )

        alignment = _require_row(
            _read_jsonl_index(self.dataset_root / self.REQUIRED_FILES[1]),
            sample_id,
            "alignment_manifest",
        )
        if for_training and alignment.get("alignment_pass") is not True:
            raise DatasetPolicyError(
                f"sample {sample_id} has alignment_pass=false and cannot enter training"
            )

        split_row = _require_row(
            _read_jsonl_index(self.dataset_root / self.REQUIRED_FILES[2]),
            sample_id,
            "split_manifest",
        )
        actual_split = str(split_row.get("split", ""))
        if actual_split != split:
            raise DatasetPolicyError(
                f"sample {sample_id} belongs to split {actual_split}, not {split}"
            )

        label = _require_row(
            _read_jsonl_index(self.dataset_root / self.REQUIRED_FILES[3]),
            sample_id,
            "mturk_labels",
        )
        targets = target_adapter.adapt(label)
        return YouTubeHighlightSample(
            sample_id=sample_id,
            split=actual_split,
            targets=targets,
            target_strategy=target_adapter.strategy,
            raw_record=raw,
            alignment_record=alignment,
        )


class Stage6BootstrapAdapter:
    """Read-only bridge to an already materialized Stage 6 bootstrap manifest."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def load_manifest(self) -> dict[str, Any]:
        path = self.dataset_root / "manifests" / "dataset_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetNotReadyError(
                f"required dataset_manifest is missing: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DatasetNotReadyError(f"dataset_manifest is invalid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise DatasetNotReadyError(f"dataset_manifest must be an object: {path}")
        return payload


class TVSumExternalTestAdapter:
    """Sealed external-test boundary; never a training adapter."""

    def __init__(self, dataset_root: str | Path, *, allow_gt: bool = False) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.allow_gt = allow_gt

    def validate_usage(
        self,
        *,
        training: bool,
        access_ground_truth: bool = False,
    ) -> None:
        if training:
            raise DatasetPolicyError(
                "TVSum is sealed for external test and cannot enter a training split"
            )
        if access_ground_truth and not self.allow_gt:
            raise DatasetPolicyError("TVSum ground truth is sealed (allow_gt=false)")
