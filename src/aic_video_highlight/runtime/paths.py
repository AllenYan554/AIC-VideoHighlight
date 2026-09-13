"""Portable environment and per-run path resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id must be lowercase snake_case (letters, digits, '_' or '-'), "
            "for example vhicraft_local_dev166_py312"
        )
    return value


@dataclass(frozen=True)
class RunPaths:
    output: Path
    logs: Path
    cache: Path
    tmp: Path

    def create(self) -> None:
        for path in (self.output, self.logs, self.cache, self.tmp):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EnvironmentPaths:
    name: str
    repo: Path
    datasets: Path
    models: Path
    hf_cache: Path
    outputs: Path
    logs: Path
    cache: Path
    tmp: Path
    archive: Path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentPaths":
        required = {
            "name", "repo", "datasets", "models", "hf_cache", "outputs",
            "logs", "cache", "tmp", "archive",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"environment config missing: {', '.join(missing)}")
        return cls(**{key: Path(payload[key]) if key != "name" else payload[key] for key in required})

    @classmethod
    def from_json(cls, path: Path) -> "EnvironmentPaths":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def for_run(self, run_id: str) -> RunPaths:
        validate_run_id(run_id)
        return RunPaths(
            output=self.outputs / "vhicraft" / run_id,
            logs=self.logs / "vhicraft" / run_id,
            cache=self.cache / "vhicraft" / run_id,
            tmp=self.tmp / run_id,
        )


def path_flavour(path: str) -> str:
    """Classify a configured path without depending on the host OS."""
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return "windows"
    if path.startswith("/"):
        return "posix"
    return "relative"
