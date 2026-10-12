"""Relative-only registry for datasets consumed by FTNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    relative_path: Path
    adapter: str
    allow_training: bool
    allow_gt: bool


class DatasetRegistry:
    def __init__(self, specifications: dict[str, DatasetSpec]) -> None:
        if not specifications:
            raise ValueError("dataset registry cannot be empty")
        self._specifications = dict(specifications)

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetRegistry":
        registry_path = Path(path).expanduser().resolve()
        try:
            payload: Any = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"dataset registry does not exist: {registry_path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
            raise ValueError("dataset registry must contain a datasets mapping")

        specifications: dict[str, DatasetSpec] = {}
        for dataset_id, raw in payload["datasets"].items():
            if not isinstance(dataset_id, str) or not isinstance(raw, dict):
                raise ValueError("dataset registry entries must be named mappings")
            relative_text = str(raw.get("relative_path", ""))
            if not relative_text:
                raise ValueError(f"dataset {dataset_id} has no relative_path")
            if (
                PureWindowsPath(relative_text).is_absolute()
                or PurePosixPath(relative_text).is_absolute()
                or PureWindowsPath(relative_text).drive
            ):
                raise ValueError(
                    f"dataset {dataset_id} must use a relative_path, got {relative_text}"
                )
            relative_path = Path(relative_text)
            if ".." in relative_path.parts:
                raise ValueError(f"dataset {dataset_id} relative_path cannot escape its root")
            specifications[dataset_id] = DatasetSpec(
                dataset_id=dataset_id,
                relative_path=relative_path,
                adapter=str(raw.get("adapter", dataset_id)),
                allow_training=bool(raw.get("allow_training", False)),
                allow_gt=bool(raw.get("allow_gt", False)),
            )
        return cls(specifications)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specifications))

    def get(self, dataset_id: str) -> DatasetSpec:
        try:
            return self._specifications[dataset_id]
        except KeyError as exc:
            raise KeyError(f"unknown dataset id: {dataset_id}") from exc

    def resolve(self, dataset_id: str, data_root: str | Path) -> Path:
        root = Path(data_root).expanduser().resolve()
        return (root / self.get(dataset_id).relative_path).resolve()
