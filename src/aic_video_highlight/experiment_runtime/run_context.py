"""Run identity, lifecycle state, and strict resume checks."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import file_sha256
from .io import atomic_write_json
from .paths import ExperimentPaths, validate_experiment_id

STATUSES = {"PENDING", "RUNNING", "INTERRUPTED", "COMPLETED", "FAILED", "VALIDATION_FAILED"}


class RunIdentityMismatch(RuntimeError):
    """Raised when resume inputs differ from the original run identity."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("aic-video-highlight", "PyYAML", "openai"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


@dataclass
class RunContext:
    experiment_id: str
    stage: str
    run_type: str
    repo: Path
    paths: ExperimentPaths
    config_path: Path
    protocol_path: Path
    input_hashes: dict[str, str] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_experiment_id(self.experiment_id)
        if not self.experiment_id.startswith(self.stage + "_"):
            raise ValueError("stage and experiment_id disagree")

    @property
    def manifest_path(self) -> Path:
        return self.paths.output / "run_manifest.json"

    @property
    def status_path(self) -> Path:
        return self.paths.output / "status.json"

    def identity(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "run_type": self.run_type,
            "git_branch": _git(self.repo, "branch", "--show-current"),
            "git_head": _git(self.repo, "rev-parse", "HEAD"),
            "config_path": str(self.config_path.resolve()),
            "config_sha256": file_sha256(self.config_path),
            "protocol_path": str(self.protocol_path.resolve()),
            "protocol_sha256": file_sha256(self.protocol_path),
            "input_artifact_hashes": dict(sorted(self.input_hashes.items())),
            "model": self.model,
        }

    def start(self, *, resume: bool = False) -> dict[str, Any]:
        self.paths.create()
        identity = self.identity()
        if self.manifest_path.exists():
            previous = __import__("json").loads(self.manifest_path.read_text(encoding="utf-8"))
            for key, value in identity.items():
                if previous.get(key) != value:
                    raise RunIdentityMismatch(f"resume identity mismatch: {key}")
            if not resume:
                raise FileExistsError("run already exists; pass --resume after verifying identity")
            manifest = previous
            manifest["resume_mode"] = True
        else:
            manifest = {
                "schema_version": "aic.experiment-run-manifest/v1",
                **identity,
                "started_at": utc_now(),
                "finished_at": None,
                "output_root": str(self.paths.output.resolve()),
                "log_root": str(self.paths.logs.resolve()),
                "tmp_root": str(self.paths.tmp.resolve()),
                "cache_root": str(self.paths.cache.resolve()),
                "resume_mode": resume,
                "machine_environment": {
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "package_versions": _package_versions(),
                    "gpu": self.model.get("gpu"),
                },
            }
        atomic_write_json(self.manifest_path, manifest)
        self.set_status("RUNNING")
        return manifest

    def set_status(self, status: str, **details: Any) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown run status: {status}")
        payload = {"schema_version": "aic.experiment-status/v1", "status": status, "updated_at": utc_now(), **details}
        atomic_write_json(self.status_path, payload)
        if status == "COMPLETED" and self.manifest_path.exists():
            import json

            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["finished_at"] = payload["updated_at"]
            atomic_write_json(self.manifest_path, manifest)
