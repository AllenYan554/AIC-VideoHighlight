"""Portable resolution of the external dataset root."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


DATA_ROOT_ENV = "VHICRAFT_DATA_ROOT"


class DatasetRootResolutionError(RuntimeError):
    """Raised when an explicitly selected dataset root cannot be resolved."""


def _load_config_root(config_path: Path) -> Path | None:
    try:
        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetRootResolutionError(
            f"Dataset config does not exist: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetRootResolutionError(
            f"Dataset config is not valid YAML/JSON: {config_path}"
        ) from exc

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise DatasetRootResolutionError(
            f"Dataset config must contain a mapping: {config_path}"
        )
    configured = payload.get("dataset_root")
    if configured in (None, ""):
        return None
    root = Path(str(configured)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root


def _require_existing_directory(root: Path, source: str) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.exists():
        raise DatasetRootResolutionError(
            f"Dataset root selected by {source} does not exist: {resolved}"
        )
    if not resolved.is_dir():
        raise DatasetRootResolutionError(
            f"Dataset root selected by {source} is not a directory: {resolved}"
        )
    return resolved


def resolve_dataset_root(
    *,
    cli_data_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    test_fallback: str | Path | None = None,
) -> Path:
    """Resolve a dataset root without any machine-specific implicit default.

    Priority is CLI, ``VHICRAFT_DATA_ROOT``, config ``dataset_root``, then an
    explicitly supplied repo-relative test fallback.  Once a source is chosen,
    an invalid path fails closed instead of silently falling through.
    """

    if cli_data_root not in (None, ""):
        return _require_existing_directory(Path(cli_data_root), "CLI --data-root")

    environment = os.environ if environ is None else environ
    env_value = environment.get(DATA_ROOT_ENV)
    if env_value:
        return _require_existing_directory(Path(env_value), DATA_ROOT_ENV)

    if config_path is not None:
        config = Path(config_path).expanduser().resolve()
        config_root = _load_config_root(config)
        if config_root is not None:
            return _require_existing_directory(config_root, f"config {config}")

    if test_fallback is not None:
        fallback = Path(test_fallback)
        if fallback.is_absolute():
            raise DatasetRootResolutionError(
                "test_fallback must be repo-relative, not an absolute path"
            )
        if repo_root is None:
            raise DatasetRootResolutionError(
                "repo_root is required when test_fallback is used"
            )
        return _require_existing_directory(
            Path(repo_root).expanduser().resolve() / fallback,
            "explicit repo-relative test fallback",
        )

    raise DatasetRootResolutionError(
        "Dataset root is not configured. Pass --data-root, set "
        f"{DATA_ROOT_ENV}, or provide a config containing dataset_root."
    )
