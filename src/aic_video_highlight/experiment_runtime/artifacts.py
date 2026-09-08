"""Generate a verifiable manifest for formal files without modifying them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .hashing import file_sha256
from .io import atomic_write_json


def _json_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in ("semantic_sha256", "record_count", "schema_version", "role")
        if key in value
    }


def build_artifact_manifest(root: Path, files: Iterable[Path], output: Path | None = None, *, created_by: str = "experiment_runtime") -> dict[str, Any]:
    root = root.resolve()
    records = []
    for path in sorted((Path(p).resolve() for p in files), key=lambda p: str(p)):
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "relative_path": relative,
                "type": path.suffix.lower().lstrip(".") or "file",
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "created_by": created_by,
                **_json_metadata(path),
            }
        )
    manifest = {"schema_version": "aic.artifact-manifest/v1", "artifact_count": len(records), "artifacts": records}
    if output is not None:
        atomic_write_json(output, manifest)
    return manifest
