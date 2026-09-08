"""Portable environment and per-experiment path resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_ID_RE = re.compile(r"^stage[1-9][0-9]*(?:_[a-z0-9]+)+$")


def validate_experiment_id(value: str) -> str:
    if not EXPERIMENT_ID_RE.fullmatch(value):
        raise ValueError(
            "experiment_id must be lowercase stage-prefixed snake_case, "
            "for example stage5_3_formal"
        )
    return value


@dataclass(frozen=True)
class ExperimentPaths:
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

    def for_experiment(self, stage: str, experiment_id: str) -> ExperimentPaths:
        validate_experiment_id(experiment_id)
        if not re.fullmatch(r"stage[1-9][0-9]*", stage) or not experiment_id.startswith(stage + "_"):
            raise ValueError("stage and experiment_id disagree")
        return ExperimentPaths(
            output=self.outputs / stage / experiment_id,
            logs=self.logs / stage / experiment_id,
            cache=self.cache / stage / experiment_id,
            tmp=self.tmp / experiment_id,
        )


def path_flavour(path: str) -> str:
    """Classify a configured path without depending on the host OS."""
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return "windows"
    if path.startswith("/"):
        return "posix"
    return "relative"
