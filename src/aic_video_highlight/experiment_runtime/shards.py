"""Atomic shard storage with content and run-identity verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import canonical_sha256, file_sha256
from .io import atomic_write_json


class ShardStore:
    def __init__(self, root: Path, identity_hash: str) -> None:
        self.root = root
        self.identity_hash = identity_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def data_path(self, shard_id: str) -> Path:
        return self.root / f"{shard_id}.json"

    def meta_path(self, shard_id: str) -> Path:
        return self.root / f"{shard_id}.status.json"

    def is_complete(self, shard_id: str) -> bool:
        data, meta = self.data_path(shard_id), self.meta_path(shard_id)
        if not data.exists() or not meta.exists():
            return False
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("status") == "COMPLETED"
            and payload.get("identity_sha256") == self.identity_hash
            and payload.get("bytes_sha256") == file_sha256(data)
        )

    def write(self, shard_id: str, records: Any) -> Path:
        path = self.data_path(shard_id)
        atomic_write_json(path, records)
        atomic_write_json(
            self.meta_path(shard_id),
            {
                "schema_version": "aic.experiment-shard-status/v1",
                "status": "COMPLETED",
                "identity_sha256": self.identity_hash,
                "bytes_sha256": file_sha256(path),
                "semantic_sha256": canonical_sha256(records),
                "record_count": len(records) if hasattr(records, "__len__") else None,
            },
        )
        return path
