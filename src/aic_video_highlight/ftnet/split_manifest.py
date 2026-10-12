"""Materialize the frozen Stage 7.1 FTNet split manifest from dataset manifests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .split_protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    SPLIT_RATIOS,
    SPLIT_ROLES,
    SPLIT_SEED,
    AssignedSplit,
    SplitAudit,
    SplitCandidate,
    SplitProtocolError,
    assign_stage7_splits,
    audit_assignment,
)


MANIFEST_SCHEMA = "aic.stage7.ftnet.split-manifest/v1"
YOUTUBE_HIGHLIGHTS_DIR = "youtube_highlights"
ALIGNMENT_MANIFEST = "alignment_manifest.jsonl"
RAW_MANIFEST = "raw_manifest.jsonl"
AVAILABILITY_MANIFEST = "availability_manifest.jsonl"
SPLIT_MANIFEST_NAME = "stage7_ftnet_split_manifest.json"
DETERMINISTIC_ALGORITHM = (
    "sort canonical source groups per category by "
    'SHA256("VHiCraFTNet-Stage7.1|<seed>|<category>|<canonical_video_id>"); '
    "assign 70/15/15 with remainder priority TRAIN > VALIDATION > CALIBRATION"
)


class SplitManifestError(RuntimeError):
    """Raised when the dataset manifests cannot produce a frozen split."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SplitManifestError(f"required manifest is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SplitManifestError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise SplitManifestError(f"manifest row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _index_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise SplitManifestError(f"manifest row is missing {key}: {row}")
        indexed[value] = row
    return indexed


def load_ftnet_split_candidates(
    dataset_root: str | Path,
    *,
    verify_files: bool = False,
) -> tuple[list[SplitCandidate], dict[str, str]]:
    """Load official-train, alignment-pass candidates and source-file hashes."""

    dataset_dir = Path(dataset_root).expanduser().resolve() / YOUTUBE_HIGHLIGHTS_DIR
    manifests_dir = dataset_dir / "manifests"
    alignment_path = manifests_dir / ALIGNMENT_MANIFEST
    raw_path = manifests_dir / RAW_MANIFEST
    availability_path = manifests_dir / AVAILABILITY_MANIFEST

    alignment_rows = _read_jsonl(alignment_path)
    raw_by_canonical = _index_by(_read_jsonl(raw_path), "canonical_video_id")
    availability_by_canonical = _index_by(
        _read_jsonl(availability_path), "canonical_video_id"
    )

    candidates: list[SplitCandidate] = []
    for row in alignment_rows:
        if str(row.get("official_split", "")).strip().lower() != "train":
            continue
        if row.get("ALIGNMENT_PASS") is not True:
            continue
        canonical_video_id = row.get("canonical_video_id")
        if not isinstance(canonical_video_id, str) or not canonical_video_id:
            raise SplitManifestError(f"alignment row lacks canonical_video_id: {row}")
        raw = raw_by_canonical.get(canonical_video_id)
        if raw is None:
            raise SplitManifestError(
                f"{canonical_video_id} is absent from {RAW_MANIFEST}"
            )
        if str(raw.get("download_status", "")).strip().upper() != "OK":
            raise SplitManifestError(
                f"{canonical_video_id} is not DOWNLOAD_OK in {RAW_MANIFEST}"
            )
        source_file_sha256 = raw.get("sha256")
        if not isinstance(source_file_sha256, str) or len(source_file_sha256) != 64:
            raise SplitManifestError(
                f"{canonical_video_id} lacks a source SHA-256 in {RAW_MANIFEST}"
            )
        relative_path = raw.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise SplitManifestError(
                f"{canonical_video_id} lacks a relative_path in {RAW_MANIFEST}"
            )
        if verify_files and not (dataset_dir / relative_path).is_file():
            raise SplitManifestError(
                f"local video file is missing for {canonical_video_id}: "
                f"{dataset_dir / relative_path}"
            )

        availability = availability_by_canonical.get(canonical_video_id, {})
        original_video_id = availability.get("original_video_id") or canonical_video_id
        replacement_video_id = availability.get("replacement_video_id")
        realized_video_id = row.get("actual_source_id") or canonical_video_id
        category = row.get("category")
        if not isinstance(category, str) or not category:
            raise SplitManifestError(f"alignment row lacks category: {row}")

        candidates.append(
            SplitCandidate(
                canonical_video_id=canonical_video_id,
                realized_video_id=realized_video_id,
                category=category,
                upstream_official_split="train",
                alignment_pass=True,
                local_video_path=relative_path,
                source_file_sha256=source_file_sha256,
                annotation_identity=f"{category}/{canonical_video_id}",
                original_video_id=original_video_id,
                replacement_video_id=(
                    replacement_video_id
                    if isinstance(replacement_video_id, str) and replacement_video_id
                    else None
                ),
                upstream_official_set=raw.get("official_set"),
            )
        )

    dataset_hashes = {
        "alignment_manifest_sha256": _sha256_file(alignment_path),
        "raw_manifest_sha256": _sha256_file(raw_path),
        "availability_manifest_sha256": _sha256_file(availability_path),
    }
    return candidates, dataset_hashes


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _entry_payload(entry: AssignedSplit) -> dict[str, Any]:
    candidate = entry.candidate
    return {
        "canonical_video_id": candidate.canonical_video_id,
        "realized_video_id": candidate.realized_video_id,
        "original_video_id": candidate.original_video_id,
        "replacement_video_id": candidate.replacement_video_id,
        "category": candidate.category,
        "upstream_official_split": candidate.upstream_official_split,
        "upstream_official_set": candidate.upstream_official_set,
        "stage7_split": entry.stage7_split,
        "canonical_source_group": candidate.canonical_video_id,
        "source_group_key": entry.source_group_key,
        "split_hash_key": entry.split_hash_key,
        "alignment_status": (
            "ALIGNMENT_PASS" if candidate.alignment_pass else "ALIGNMENT_FAIL"
        ),
        "replacement_original_id": entry.replacement_original_id,
        "relative_video_path": candidate.local_video_path,
        "annotation_identity": candidate.annotation_identity,
        "source_sha256": candidate.source_file_sha256,
    }


def _ordered_entries(assigned: tuple[AssignedSplit, ...]) -> list[dict[str, Any]]:
    role_index = {role: index for index, role in enumerate(SPLIT_ROLES)}
    entries = [_entry_payload(entry) for entry in assigned]
    entries.sort(
        key=lambda item: (
            role_index[item["stage7_split"]],
            item["category"],
            item["canonical_video_id"],
            item["realized_video_id"],
        )
    )
    return entries


def _deterministic_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload["schema"],
        "protocol_name": payload["protocol_name"],
        "protocol_version": payload["protocol_version"],
        "seed": payload["seed"],
        "ratios": payload["ratios"],
        "deterministic_algorithm": payload["deterministic_algorithm"],
        "upstream_source": payload["upstream_source"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "total": payload["total"],
        "counts": payload["counts"],
        "entries": payload["entries"],
    }


def content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(_deterministic_content(payload)).encode("utf-8")
    ).hexdigest()


def manifest_sha256(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in ("manifest_sha256", "generated_at")
    }
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def determinism_check(
    candidates: list[SplitCandidate], *, seed: int = SPLIT_SEED
) -> bool:
    """Confirm the assignment is invariant to input ordering."""

    first = _ordered_entries(assign_stage7_splits(candidates, seed=seed))
    reversed_order = list(reversed(candidates))
    second = _ordered_entries(assign_stage7_splits(reversed_order, seed=seed))
    return first == second


def build_split_manifest(
    assigned: tuple[AssignedSplit, ...],
    *,
    audit: SplitAudit,
    dataset_hashes: dict[str, str],
    upstream_source: dict[str, Any],
    git_head: str,
    generated_at: str | None = None,
    determinism: bool | None = None,
) -> dict[str, Any]:
    if audit.source_id_leakage != 0:
        raise SplitProtocolError(
            f"refusing to freeze a manifest with source leakage "
            f"({audit.source_id_leakage})"
        )
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "youtube_highlights",
        "seed": SPLIT_SEED,
        "ratios": list(SPLIT_RATIOS),
        "split_roles": list(SPLIT_ROLES),
        "deterministic_algorithm": DETERMINISTIC_ALGORITHM,
        "upstream_source": upstream_source,
        "source_manifest_sha256": dataset_hashes,
        "git_head": git_head,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "total": audit.total,
        "counts": audit.counts,
        "category_counts": audit.category_counts,
        "source_id_leakage": audit.source_id_leakage,
        "official_test_rows": audit.official_test_rows,
        "tvsum_rows": audit.tvsum_rows,
        "non_alignment_pass_rows": audit.non_alignment_pass_rows,
        "determinism": determinism,
        "entries": _ordered_entries(assigned),
    }
    payload["content_sha256"] = content_sha256(payload)
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload


def serialize_manifest(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_split_manifest(
    payload: dict[str, Any], output_path: str | Path
) -> tuple[Path, Path]:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_manifest(payload)
    path.write_text(text, encoding="utf-8")
    file_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{file_sha256}  {path.name}\n", encoding="utf-8")
    return path, sidecar
