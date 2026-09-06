"""Frozen candidate cache export, validation, and identity replay.

This module consumes persisted Stage 3 model responses.  It never reads video
frames, references, metrics, or adjudication labels and never calls a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .candidate_merger import merge_segments
from .response_parser import parse_highlight_response
from .schemas import HighlightSegment
from .temporal_metrics import temporal_iou


CACHE_SCHEMA_VERSION = "aic.frozen-candidate-cache/v1"
ID_SCHEME_VERSION = "aic.candidate-id/v1"
CANONICALIZATION_VERSION = "aic.canonical-json/v1"
FROZEN_DATASET_VERSION = "aic_highlight_dev_v1.1"
FROZEN_MODEL_NAME = "Qwen/Qwen3.5-4B"
FROZEN_PROMPT_VERSION = "high_recall_retrieval_v0"
ALLOWED_SPLITS = ("dev", "hard")
_SPLIT_ORDER = {split: index for index, split in enumerate(ALLOWED_SPLITS)}


class CandidateCacheError(ValueError):
    """Raised when frozen inputs or a cache violate the Stage 4.2 contract."""


@dataclass(frozen=True, slots=True)
class FrozenSource:
    split: str
    experiment_dir: Path
    run_config_path: Path
    predictions_path: Path
    raw_dir: Path


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateCacheError("canonical JSON rejects non-finite floats")
        return 0.0 if value == 0.0 else value
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize semantic JSON as UTF-8, sorted keys, compact separators, LF."""
    normalized = _canonical_value(payload)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _id_payload(fields: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    common = {
        "id_scheme_version": ID_SCHEME_VERSION,
        "kind": kind,
        "video_id": fields["video_id"],
        "start_sec": float(fields["start_sec"]),
        "end_sec": float(fields["end_sec"]),
        "score": float(fields["score"]),
        "reason": str(fields.get("reason", "")),
        "source_chunk": fields.get("source_chunk"),
    }
    if kind == "raw":
        common.update(
            {
                "chunk_index": int(fields["chunk_index"]),
                "candidate_index": int(fields["candidate_index"]),
                "chunk_start_sec": float(fields["chunk_start_sec"]),
                "chunk_end_sec": float(fields["chunk_end_sec"]),
                "local_start_sec": float(fields["local_start_sec"]),
                "local_end_sec": float(fields["local_end_sec"]),
            }
        )
    else:
        common["contributor_raw_candidate_ids"] = sorted(
            str(item) for item in fields["contributor_raw_candidate_ids"]
        )
    return common


def raw_candidate_id(fields: Mapping[str, Any]) -> str:
    return f"raw-v1-{semantic_sha256(_id_payload(fields, kind='raw'))}"


def merged_candidate_id(fields: Mapping[str, Any]) -> str:
    return f"merged-v1-{semantic_sha256(_id_payload(fields, kind='merged'))}"


def _segment_projection(segment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start_sec": float(segment["start_sec"]),
        "end_sec": float(segment["end_sec"]),
        "score": float(segment["score"]),
        "reason": str(segment.get("reason", "")),
        "source_chunk": segment.get("source_chunk"),
    }


def _segments_equal(left: list[Mapping[str, Any]], right: list[Mapping[str, Any]]) -> bool:
    return [_segment_projection(item) for item in left] == [
        _segment_projection(item) for item in right
    ]


def _tracked_merge(
    raw_candidates: list[dict[str, Any]], *, tiou_threshold: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        raw_candidates,
        key=lambda item: (item["start_sec"], item["end_sec"], -item["score"]),
    )
    merged: list[dict[str, Any]] = []
    for candidate in ordered:
        projected = _segment_projection(candidate)
        projected["contributor_raw_candidate_ids"] = [candidate["raw_candidate_id"]]
        if not merged or temporal_iou(
            (merged[-1]["start_sec"], merged[-1]["end_sec"]),
            (projected["start_sec"], projected["end_sec"]),
        ) < tiou_threshold:
            merged.append(projected)
            continue

        previous = merged[-1]
        reasons = list(
            dict.fromkeys(
                reason
                for reason in (previous["reason"], projected["reason"])
                if reason
            )
        )
        merged[-1] = {
            "start_sec": min(previous["start_sec"], projected["start_sec"]),
            "end_sec": max(previous["end_sec"], projected["end_sec"]),
            "score": max(previous["score"], projected["score"]),
            "reason": " | ".join(reasons),
            "source_chunk": (
                previous["source_chunk"]
                if previous["source_chunk"] == projected["source_chunk"]
                else None
            ),
            "contributor_raw_candidate_ids": [
                *previous["contributor_raw_candidate_ids"],
                *projected["contributor_raw_candidate_ids"],
            ],
        }
    return merged


def _safe_provenance(run_config: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "experiment_name",
        "dataset_name",
        "dataset_version",
        "model_name",
        "model_revision",
        "prompt_version",
        "sampling_fps",
        "chunk_seconds",
        "chunk_overlap_seconds",
        "merge_threshold",
        "merge_strategy",
        "git_commit_head",
    )
    missing = [key for key in required if key not in run_config]
    if missing:
        raise CandidateCacheError(f"run_config provenance is missing: {', '.join(missing)}")
    request_parameters = run_config.get("request_parameters")
    if not isinstance(request_parameters, dict):
        raise CandidateCacheError("run_config request_parameters is missing")
    return {
        "experiment_name": str(run_config["experiment_name"]),
        "dataset_name": str(run_config["dataset_name"]),
        "dataset_version": str(run_config["dataset_version"]),
        "model_name": str(run_config["model_name"]),
        "model_revision": str(run_config["model_revision"]),
        "prompt_version": str(run_config["prompt_version"]),
        "sampling_fps": float(run_config["sampling_fps"]),
        "chunk_seconds": float(run_config["chunk_seconds"]),
        "chunk_overlap_seconds": float(run_config["chunk_overlap_seconds"]),
        "merge_threshold": float(run_config["merge_threshold"]),
        "merge_strategy": str(run_config["merge_strategy"]),
        "git_commit_head": str(run_config["git_commit_head"]),
        "runtime_versions": {
            "python": str(run_config.get("python_version", "unavailable")),
            "torch": str(run_config.get("torch_version", "unavailable")),
            "vllm": str(run_config.get("vllm_version", "unavailable")),
        },
        "request_parameters": _canonical_value(request_parameters),
    }


def _validate_frozen_run_config(run_config: Mapping[str, Any], split: str) -> None:
    expected = {
        "split": split,
        "dataset_version": FROZEN_DATASET_VERSION,
        "model_name": FROZEN_MODEL_NAME,
        "prompt_version": FROZEN_PROMPT_VERSION,
    }
    for key, value in expected.items():
        if run_config.get(key) != value:
            raise CandidateCacheError(f"frozen run_config requires {key}={value}")
    if not run_config.get("model_revision") or not run_config.get("git_commit_head"):
        raise CandidateCacheError("frozen run_config lacks model or Git revision provenance")


def build_video_cache_record(
    *,
    raw_payload: Mapping[str, Any],
    frozen_prediction: Mapping[str, Any],
    run_config: Mapping[str, Any],
    split: str,
    raw_journal_sha256: str,
    frozen_prediction_record_sha256: str,
    run_config_sha256: str,
) -> dict[str, Any]:
    """Build one cache record while proving parser, merge, and prediction identity."""
    if split not in ALLOWED_SPLITS:
        raise CandidateCacheError("candidate cache accepts only dev and hard; Heldout is forbidden")
    _validate_frozen_run_config(run_config, split)
    video_id = raw_payload.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise CandidateCacheError("raw journal has an invalid video_id")
    if frozen_prediction.get("video_id") != video_id:
        raise CandidateCacheError(f"raw/prediction video_id mismatch for {video_id}")
    chunks = raw_payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise CandidateCacheError(f"raw journal has no chunks for {video_id}")
    chunk_indices = [int(item["chunk_index"]) for item in chunks]
    if chunk_indices != sorted(chunk_indices) or len(set(chunk_indices)) != len(chunk_indices):
        raise CandidateCacheError(f"raw chunk order/identity is invalid for {video_id}")
    duration_sec = float(raw_payload["experiment_clip_duration_sec"])
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise CandidateCacheError(f"invalid experiment duration for {video_id}")

    raw_candidates: list[dict[str, Any]] = []
    persisted_global_candidates: list[dict[str, Any]] = []
    source_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_index = int(chunk["chunk_index"])
        chunk_start = float(chunk["chunk_start_sec"])
        chunk_end = float(chunk["chunk_end_sec"])
        if chunk_start < 0 or chunk_end <= chunk_start:
            raise CandidateCacheError(f"invalid chunk interval for {video_id} chunk {chunk_index}")
        if chunk.get("finish_reason") == "length":
            raise CandidateCacheError(
                f"truncated frozen response for {video_id} chunk {chunk_index}"
            )
        local_segments = parse_highlight_response(
            str(chunk["raw_response"]),
            chunk_duration_sec=chunk_end - chunk_start,
            source_chunk=chunk_index,
        )
        persisted_local = chunk.get("parsed_segments")
        if not isinstance(persisted_local, list) or not _segments_equal(
            [asdict(item) for item in local_segments], persisted_local
        ):
            raise CandidateCacheError(
                f"persisted parsed candidate identity mismatch for {video_id} chunk {chunk_index}"
            )
        source_chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_start_sec": chunk_start,
                "chunk_end_sec": chunk_end,
                "finish_reason": (
                    None
                    if chunk.get("finish_reason") is None
                    else str(chunk["finish_reason"])
                ),
                "candidate_count": len(local_segments),
                "raw_response_sha256": hashlib.sha256(
                    str(chunk["raw_response"]).encode("utf-8")
                ).hexdigest(),
            }
        )
        for candidate_index, local in enumerate(local_segments):
            fields = {
                "video_id": video_id,
                "chunk_index": chunk_index,
                "candidate_index": candidate_index,
                "chunk_start_sec": chunk_start,
                "chunk_end_sec": chunk_end,
                "local_start_sec": float(local.start_sec),
                "local_end_sec": float(local.end_sec),
                "start_sec": chunk_start + float(local.start_sec),
                "end_sec": chunk_start + float(local.end_sec),
                "score": float(local.score),
                "reason": local.reason,
                "source_chunk": local.source_chunk,
            }
            fields["raw_candidate_id"] = raw_candidate_id(fields)
            if fields["end_sec"] > duration_sec + 1e-6:
                raise CandidateCacheError(f"candidate exceeds experiment duration for {video_id}")
            raw_candidates.append(fields)
            persisted_global_candidates.append(_segment_projection(fields))

    frozen_candidates = frozen_prediction.get("parsed_chunk_segments")
    if not isinstance(frozen_candidates, list) or not _segments_equal(
        persisted_global_candidates, frozen_candidates
    ):
        raise CandidateCacheError(f"parsed candidate identity mismatch for {video_id}")

    merge_threshold = float(run_config["merge_threshold"])
    tracked_merged = _tracked_merge(raw_candidates, tiou_threshold=merge_threshold)
    original_merger_output = [
        asdict(item)
        for item in merge_segments(
            [HighlightSegment(**_segment_projection(item)) for item in raw_candidates],
            tiou_threshold=merge_threshold,
        )
    ]
    if not _segments_equal(tracked_merged, original_merger_output):
        raise CandidateCacheError(f"lineage replay diverges from frozen merger for {video_id}")
    frozen_merged = frozen_prediction.get("merged_prediction_segments")
    if not isinstance(frozen_merged, list) or not _segments_equal(
        tracked_merged, frozen_merged
    ):
        raise CandidateCacheError(f"merged prediction identity mismatch for {video_id}")

    merged_candidates: list[dict[str, Any]] = []
    for item in tracked_merged:
        fields = {"video_id": video_id, **item}
        fields["merged_candidate_id"] = merged_candidate_id(fields)
        fields.pop("video_id")
        merged_candidates.append(fields)

    record: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "id_scheme_version": ID_SCHEME_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "video_id": video_id,
        "split": split,
        "coordinate_system": "clip-local seconds",
        "duration_sec": duration_sec,
        "source_provenance": {
            **_safe_provenance(run_config),
            "source_artifact_hashes": {
                "raw_journal_sha256": raw_journal_sha256,
                "frozen_prediction_record_sha256": frozen_prediction_record_sha256,
                "run_config_sha256": run_config_sha256,
            },
        },
        "source_chunks": source_chunks,
        "raw_candidates": raw_candidates,
        "merged_candidates": merged_candidates,
    }
    record["semantic_sha256"] = semantic_sha256(record)
    return record


def _resolve_source(split: str, experiment_dir: Path) -> FrozenSource:
    if split not in ALLOWED_SPLITS:
        raise CandidateCacheError("candidate cache accepts only dev and hard; Heldout is forbidden")
    root = experiment_dir.expanduser().resolve()
    layouts = (
        (root / "run_config.json", root / "predictions.jsonl", root / "raw"),
        (
            root / "config" / "run_config.json",
            root / "results" / "predictions.jsonl",
            root / "results" / "raw",
        ),
    )
    for run_config, predictions, raw_dir in layouts:
        if run_config.is_file() and predictions.is_file() and raw_dir.is_dir():
            return FrozenSource(split, root, run_config, predictions, raw_dir)
    raise CandidateCacheError(
        f"cannot resolve frozen run_config/predictions/raw artifacts under {root}"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateCacheError(f"expected a JSON object: {path}")
    return payload


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("video_id"), str):
            raise CandidateCacheError(f"invalid prediction record at {path}:{line_number}")
        video_id = payload["video_id"]
        if video_id in seen:
            raise CandidateCacheError(f"duplicate frozen prediction video_id: {video_id}")
        if payload.get("success") is not True:
            raise CandidateCacheError(f"frozen prediction is not successful: {video_id}")
        seen.add(video_id)
        records.append(payload)
    if not records:
        raise CandidateCacheError(f"predictions file is empty: {path}")
    return records


def export_candidate_cache(
    sources: Mapping[str, Path],
    *,
    output_dir: Path,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Export an immutable cache from explicit Dev/Hard frozen experiment roots."""
    if not sources:
        raise CandidateCacheError("at least one dev/hard source is required")
    unknown = set(sources) - set(ALLOWED_SPLITS)
    if unknown:
        raise CandidateCacheError("candidate cache accepts only dev and hard; Heldout is forbidden")
    if limit_per_split is not None and limit_per_split <= 0:
        raise CandidateCacheError("limit_per_split must be greater than zero")
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable cache output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    source_sets: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for split in sorted(sources, key=_SPLIT_ORDER.__getitem__):
        source = _resolve_source(split, Path(sources[split]))
        run_config = _read_json_object(source.run_config_path)
        _validate_frozen_run_config(run_config, split)
        predictions = _read_predictions(source.predictions_path)
        selected = predictions[:limit_per_split] if limit_per_split else predictions
        run_config_hash = _file_sha256(source.run_config_path)
        source_sets.append(
            {
                "split": split,
                "record_count": len(selected),
                "full_source_record_count": len(predictions),
                "source_provenance": _safe_provenance(run_config),
                "source_artifact_hashes": {
                    "run_config_sha256": run_config_hash,
                    "predictions_jsonl_sha256": _file_sha256(source.predictions_path),
                },
            }
        )
        for source_record_index, prediction in enumerate(selected):
            video_id = prediction["video_id"]
            if video_id in seen_video_ids:
                raise CandidateCacheError(f"video_id appears in multiple sources: {video_id}")
            seen_video_ids.add(video_id)
            raw_path = source.raw_dir / f"{video_id}.json"
            if not raw_path.is_file():
                raise CandidateCacheError(f"missing frozen raw journal for {video_id}")
            raw_payload = _read_json_object(raw_path)
            record = build_video_cache_record(
                raw_payload=raw_payload,
                frozen_prediction=prediction,
                run_config=run_config,
                split=split,
                raw_journal_sha256=_file_sha256(raw_path),
                frozen_prediction_record_sha256=semantic_sha256(prediction),
                run_config_sha256=run_config_hash,
            )
            relative_path = f"records/{video_id}.json"
            _write_canonical_json(output / relative_path, record)
            manifest_records.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "source_record_index": source_record_index,
                    "path": relative_path,
                    "semantic_sha256": record["semantic_sha256"],
                }
            )

    manifest: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "id_scheme_version": ID_SCHEME_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "dataset_version": FROZEN_DATASET_VERSION,
        "record_count": len(manifest_records),
        "complete_source_export": all(
            item["record_count"] == item["full_source_record_count"]
            for item in source_sets
        ),
        "source_sets": source_sets,
        "records": manifest_records,
    }
    manifest["global_semantic_sha256"] = semantic_sha256(manifest)
    _write_canonical_json(output / "cache_manifest.json", manifest)
    validate_cache(output)
    return manifest


_FORBIDDEN_KEY_FRAGMENTS = (
    "reference",
    "metric",
    "precision",
    "recall",
    "temporal_iou",
    "f1",
    "audit",
    "adjudicat",
    "reviewer",
    "human",
    "judg",
    "label",
)


def _check_forbidden_fields(payload: Any, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise CandidateCacheError(
                    f"forbidden development-label field at {path}.{key}"
                )
            _check_forbidden_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _check_forbidden_fields(value, f"{path}[{index}]")


def _require_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise CandidateCacheError(f"missing or invalid source hash: {field}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CandidateCacheError(f"missing or invalid source hash: {field}") from exc


def _validate_interval(item: Mapping[str, Any], *, field: str) -> None:
    try:
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        score = float(item["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateCacheError(f"invalid interval fields in {field}") from exc
    if not all(math.isfinite(value) for value in (start, end, score)):
        raise CandidateCacheError(f"invalid interval in {field}")
    if start < 0 or end <= start or not 0 <= score <= 1:
        raise CandidateCacheError(f"invalid interval in {field}")


def _validate_record(record: dict[str, Any]) -> None:
    _check_forbidden_fields(record)
    if record.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise CandidateCacheError("unsupported cache schema version")
    if record.get("id_scheme_version") != ID_SCHEME_VERSION:
        raise CandidateCacheError("unsupported candidate ID scheme version")
    if record.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise CandidateCacheError("unsupported canonicalization version")
    if record.get("split") not in ALLOWED_SPLITS:
        raise CandidateCacheError("Heldout contamination or unknown split in cache record")
    video_id = record.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise CandidateCacheError("cache record has an invalid video_id")
    provenance = record.get("source_provenance")
    if not isinstance(provenance, dict):
        raise CandidateCacheError(f"provenance is missing for {video_id}")
    if provenance.get("dataset_version") != FROZEN_DATASET_VERSION:
        raise CandidateCacheError(f"wrong frozen dataset version for {video_id}")
    if provenance.get("model_name") != FROZEN_MODEL_NAME:
        raise CandidateCacheError(f"wrong frozen model for {video_id}")
    if provenance.get("prompt_version") != FROZEN_PROMPT_VERSION:
        raise CandidateCacheError(f"wrong frozen prompt for {video_id}")
    for required in (
        "experiment_name",
        "dataset_name",
        "model_revision",
        "git_commit_head",
        "merge_threshold",
        "merge_strategy",
        "request_parameters",
        "runtime_versions",
    ):
        if required not in provenance:
            raise CandidateCacheError(f"provenance field {required} is missing for {video_id}")
    source_hashes = provenance.get("source_artifact_hashes")
    if not isinstance(source_hashes, dict):
        raise CandidateCacheError(f"source hashes are missing for {video_id}")
    for key in (
        "raw_journal_sha256",
        "frozen_prediction_record_sha256",
        "run_config_sha256",
    ):
        _require_hash(source_hashes.get(key), f"{video_id}.{key}")

    source_chunks = record.get("source_chunks")
    raw_candidates = record.get("raw_candidates")
    merged_candidates = record.get("merged_candidates")
    if (
        not isinstance(source_chunks, list)
        or not source_chunks
        or not isinstance(raw_candidates, list)
        or not isinstance(merged_candidates, list)
    ):
        raise CandidateCacheError(f"candidate arrays are missing for {video_id}")
    chunk_map: dict[int, dict[str, Any]] = {}
    for chunk in source_chunks:
        if not isinstance(chunk, dict):
            raise CandidateCacheError(f"invalid source chunk for {video_id}")
        chunk_index = int(chunk["chunk_index"])
        if chunk_index in chunk_map:
            raise CandidateCacheError(f"duplicate source chunk for {video_id}")
        chunk_start = float(chunk["chunk_start_sec"])
        chunk_end = float(chunk["chunk_end_sec"])
        if chunk_start < 0 or chunk_end <= chunk_start:
            raise CandidateCacheError(f"invalid source chunk interval for {video_id}")
        if not isinstance(chunk.get("candidate_count"), int) or chunk["candidate_count"] < 0:
            raise CandidateCacheError(f"invalid source chunk candidate_count for {video_id}")
        _require_hash(chunk.get("raw_response_sha256"), f"{video_id}.raw_response_sha256")
        chunk_map[chunk_index] = chunk
    if list(chunk_map) != sorted(chunk_map):
        raise CandidateCacheError(f"source chunk ordering is invalid for {video_id}")
    raw_ids: list[str] = []
    raw_order: list[tuple[int, int]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise CandidateCacheError(f"invalid raw candidate for {video_id}")
        _validate_interval(item, field=f"{video_id}.raw_candidates")
        raw_id = item.get("raw_candidate_id")
        raw_ids.append(str(raw_id))
        raw_order.append((int(item["chunk_index"]), int(item["candidate_index"])))
        chunk = chunk_map.get(int(item["chunk_index"]))
        if chunk is None:
            raise CandidateCacheError(f"raw candidate references unknown chunk for {video_id}")
        if item.get("source_chunk") != item.get("chunk_index"):
            raise CandidateCacheError(f"raw source_chunk mismatch for {video_id}")
        if (
            float(item["local_start_sec"]) + float(item["chunk_start_sec"])
            != float(item["start_sec"])
            or float(item["local_end_sec"]) + float(item["chunk_start_sec"])
            != float(item["end_sec"])
            or float(item["local_start_sec"]) < 0
            or float(item["local_end_sec"]) > (
                float(item["chunk_end_sec"]) - float(item["chunk_start_sec"])
            )
            or float(item["chunk_start_sec"]) != float(chunk["chunk_start_sec"])
            or float(item["chunk_end_sec"]) != float(chunk["chunk_end_sec"])
        ):
            raise CandidateCacheError(f"raw local/global time lineage mismatch for {video_id}")
        if raw_id != raw_candidate_id({"video_id": video_id, **item}):
            raise CandidateCacheError(f"raw candidate ID does not match content for {video_id}")
    if len(raw_ids) != len(set(raw_ids)):
        raise CandidateCacheError(f"duplicate raw candidate ID for {video_id}")
    if raw_order != sorted(raw_order):
        raise CandidateCacheError(f"raw candidate ordering is invalid for {video_id}")
    candidates_by_chunk: dict[int, list[int]] = {}
    for chunk_index, candidate_index in raw_order:
        candidates_by_chunk.setdefault(chunk_index, []).append(candidate_index)
    if any(indices != list(range(len(indices))) for indices in candidates_by_chunk.values()):
        raise CandidateCacheError(f"raw candidate indices are not contiguous for {video_id}")
    for chunk_index, chunk in chunk_map.items():
        if chunk["candidate_count"] != len(candidates_by_chunk.get(chunk_index, [])):
            raise CandidateCacheError(f"source chunk candidate_count mismatch for {video_id}")

    known_raw_ids = set(raw_ids)
    merged_ids: list[str] = []
    for item in merged_candidates:
        if not isinstance(item, dict):
            raise CandidateCacheError(f"invalid merged candidate for {video_id}")
        _validate_interval(item, field=f"{video_id}.merged_candidates")
        contributors = item.get("contributor_raw_candidate_ids")
        if not isinstance(contributors, list) or not contributors:
            raise CandidateCacheError(f"merged candidate has no contributor for {video_id}")
        if len(contributors) != len(set(contributors)):
            raise CandidateCacheError(f"duplicate contributor in merged candidate for {video_id}")
        unknown = set(contributors) - known_raw_ids
        if unknown:
            raise CandidateCacheError(f"unknown contributor for {video_id}: {sorted(unknown)}")
        merged_id = item.get("merged_candidate_id")
        merged_ids.append(str(merged_id))
        if merged_id != merged_candidate_id({"video_id": video_id, **item}):
            raise CandidateCacheError(f"merged candidate ID does not match content for {video_id}")
    if len(merged_ids) != len(set(merged_ids)):
        raise CandidateCacheError(f"duplicate merged candidate ID for {video_id}")
    contributed = {
        contributor
        for item in merged_candidates
        for contributor in item["contributor_raw_candidate_ids"]
    }
    if contributed != known_raw_ids:
        raise CandidateCacheError(f"orphan contributor lineage for {video_id}")

    claimed_hash = record.get("semantic_sha256")
    semantic_record = {key: value for key, value in record.items() if key != "semantic_sha256"}
    if claimed_hash != semantic_sha256(semantic_record):
        raise CandidateCacheError(f"semantic hash mismatch for {video_id}")

    expected_without_ids = _tracked_merge(
        raw_candidates, tiou_threshold=float(provenance["merge_threshold"])
    )
    expected_merged: list[dict[str, Any]] = []
    for item in expected_without_ids:
        fields = {"video_id": video_id, **item}
        fields["merged_candidate_id"] = merged_candidate_id(fields)
        fields.pop("video_id")
        expected_merged.append(fields)
    if merged_candidates != expected_merged:
        raise CandidateCacheError(f"contributor lineage mismatch for {video_id}")


def validate_cache(cache_dir: Path) -> dict[str, Any]:
    root = cache_dir.expanduser().resolve()
    manifest_path = root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise CandidateCacheError(f"cache manifest is missing: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise CandidateCacheError("cache manifest is not canonical JSON")
    _check_forbidden_fields(manifest)
    if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise CandidateCacheError("unsupported cache schema version")
    if manifest.get("dataset_version") != FROZEN_DATASET_VERSION:
        raise CandidateCacheError("cache manifest has the wrong frozen dataset version")
    records = manifest.get("records")
    if not isinstance(records, list) or manifest.get("record_count") != len(records):
        raise CandidateCacheError("cache manifest record_count is inconsistent")
    seen_video_ids: set[str] = set()
    seen_paths: set[str] = set()
    split_counts = {split: 0 for split in ALLOWED_SPLITS}
    raw_count = 0
    merged_count = 0
    for entry in records:
        if not isinstance(entry, dict):
            raise CandidateCacheError("invalid record entry in cache manifest")
        video_id = entry.get("video_id")
        split = entry.get("split")
        if not isinstance(video_id, str) or video_id in seen_video_ids:
            raise CandidateCacheError(f"duplicate or invalid manifest video_id: {video_id}")
        if split not in ALLOWED_SPLITS:
            raise CandidateCacheError("Heldout contamination or unknown manifest split")
        seen_video_ids.add(video_id)
        split_counts[split] += 1
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise CandidateCacheError(f"unsafe cache record path for {video_id}")
        record_path = root / relative
        relative_posix = relative.as_posix()
        if relative_posix in seen_paths:
            raise CandidateCacheError(f"duplicate cache record path: {relative_posix}")
        seen_paths.add(relative_posix)
        record = _read_json_object(record_path)
        if record_path.read_bytes() != canonical_json_bytes(record):
            raise CandidateCacheError(f"cache record is not canonical JSON for {video_id}")
        if record.get("video_id") != video_id or record.get("split") != split:
            raise CandidateCacheError(f"manifest/record identity mismatch for {video_id}")
        _validate_record(record)
        if entry.get("semantic_sha256") != record.get("semantic_sha256"):
            raise CandidateCacheError(f"manifest semantic hash mismatch for {video_id}")
        raw_count += len(record["raw_candidates"])
        merged_count += len(record["merged_candidates"])

    source_sets = manifest.get("source_sets")
    if not isinstance(source_sets, list) or not source_sets:
        raise CandidateCacheError("cache manifest source provenance is missing")
    source_splits: list[str] = []
    for source_set in source_sets:
        if source_set.get("split") not in ALLOWED_SPLITS:
            raise CandidateCacheError("Heldout contamination in source provenance")
        source_splits.append(source_set["split"])
        if source_set.get("record_count") != split_counts[source_set["split"]]:
            raise CandidateCacheError("source set record_count is inconsistent")
        if (
            not isinstance(source_set.get("full_source_record_count"), int)
            or source_set["full_source_record_count"] < source_set["record_count"]
        ):
            raise CandidateCacheError("source set full_source_record_count is inconsistent")
        hashes = source_set.get("source_artifact_hashes")
        if not isinstance(hashes, dict):
            raise CandidateCacheError("source set hashes are missing")
        _require_hash(hashes.get("run_config_sha256"), "source_set.run_config_sha256")
        _require_hash(
            hashes.get("predictions_jsonl_sha256"),
            "source_set.predictions_jsonl_sha256",
        )
    if source_splits != sorted(set(source_splits), key=_SPLIT_ORDER.__getitem__):
        raise CandidateCacheError("source set ordering or uniqueness is invalid")
    expected_complete = all(
        item["record_count"] == item["full_source_record_count"] for item in source_sets
    )
    if manifest.get("complete_source_export") is not expected_complete:
        raise CandidateCacheError("complete_source_export flag is inconsistent")

    expected_order = sorted(
        records,
        key=lambda item: (_SPLIT_ORDER[item["split"]], item["source_record_index"]),
    )
    if records != expected_order:
        raise CandidateCacheError("manifest record ordering is invalid")
    for split in source_splits:
        indices = [
            int(item["source_record_index"])
            for item in records
            if item["split"] == split
        ]
        if indices != list(range(len(indices))):
            raise CandidateCacheError(f"source record indices are not contiguous for {split}")
    actual_record_files = {
        path.relative_to(root).as_posix()
        for path in (root / "records").rglob("*")
        if path.is_file()
    }
    if actual_record_files != seen_paths:
        raise CandidateCacheError("cache contains missing or unlisted record files")

    claimed_global_hash = manifest.get("global_semantic_sha256")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "global_semantic_sha256"
    }
    if claimed_global_hash != semantic_sha256(semantic_manifest):
        raise CandidateCacheError("global manifest semantic hash mismatch")
    return {
        "record_count": len(records),
        "raw_candidate_count": raw_count,
        "merged_candidate_count": merged_count,
        "split_counts": split_counts,
        "complete_source_export": expected_complete,
        "global_semantic_sha256": claimed_global_hash,
    }


def _replayed_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "video_id": record["video_id"],
        "split": record["split"],
        "merged_prediction_segments": [
            _segment_projection(item) for item in record["merged_candidates"]
        ],
    }


def replay_cache(cache_dir: Path, output_path: Path) -> dict[str, int]:
    root = cache_dir.expanduser().resolve()
    validate_cache(root)
    output = output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"replay output already exists: {output}")
    manifest = _read_json_object(root / "cache_manifest.json")
    records: list[dict[str, Any]] = []
    for entry in manifest["records"]:
        record = _read_json_object(root / entry["path"])
        records.append(_replayed_record(record))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical_json_bytes(item) for item in records))
    temporary.replace(output)
    return {
        "record_count": len(records),
        "segment_count": sum(len(item["merged_prediction_segments"]) for item in records),
    }


def compare_identity(
    cache_dir: Path,
    sources: Mapping[str, Path],
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    root = cache_dir.expanduser().resolve()
    validate_cache(root)
    if set(sources) - set(ALLOWED_SPLITS):
        raise CandidateCacheError("identity comparison accepts only dev and hard sources")
    manifest = _read_json_object(root / "cache_manifest.json")
    source_sets = {item["split"]: item for item in manifest["source_sets"]}
    cache_entries_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in ALLOWED_SPLITS
    }
    for entry in manifest["records"]:
        cache_entries_by_split[entry["split"]].append(entry)

    mismatches: list[dict[str, Any]] = []
    matched = 0
    compared = 0
    for split in sorted(sources, key=_SPLIT_ORDER.__getitem__):
        source = _resolve_source(split, Path(sources[split]))
        run_config = _read_json_object(source.run_config_path)
        _validate_frozen_run_config(run_config, split)
        source_hashes = source_sets.get(split, {}).get("source_artifact_hashes", {})
        if source_hashes.get("run_config_sha256") != _file_sha256(source.run_config_path):
            mismatches.append({"split": split, "kind": "run_config_source_hash"})
        if source_hashes.get("predictions_jsonl_sha256") != _file_sha256(
            source.predictions_path
        ):
            mismatches.append({"split": split, "kind": "predictions_source_hash"})
        predictions = _read_predictions(source.predictions_path)
        entries = cache_entries_by_split[split]
        predictions = predictions[: len(entries)]
        if [item["video_id"] for item in predictions] != [
            item["video_id"] for item in entries
        ]:
            mismatches.append({"split": split, "kind": "video_order_or_membership"})
            continue
        for prediction, entry in zip(predictions, entries, strict=True):
            compared += 1
            cache_record = _read_json_object(root / entry["path"])
            replayed = _replayed_record(cache_record)
            frozen_segments = prediction.get("merged_prediction_segments")
            if not isinstance(frozen_segments, list) or not _segments_equal(
                replayed["merged_prediction_segments"], frozen_segments
            ):
                mismatches.append(
                    {"video_id": prediction["video_id"], "split": split, "kind": "segments"}
                )
            else:
                matched += 1
    expected_splits = {item["split"] for item in manifest["source_sets"]}
    if set(sources) != expected_splits:
        mismatches.append(
            {
                "kind": "source_split_set",
                "expected": sorted(expected_splits),
                "actual": sorted(sources),
            }
        )
    report = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "record_count": manifest["record_count"],
        "compared_records": compared,
        "matched_records": matched,
        "identity_match": not mismatches and matched == manifest["record_count"],
        "mismatches": mismatches,
    }
    if report_path is not None:
        output = report_path.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"identity report already exists: {output}")
        _write_canonical_json(output, report)
    return report
