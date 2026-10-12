"""Deterministic FTNet checkpoint save/load for resume verification."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


CHECKPOINT_SCHEMA = "aic.stage7.ftnet.checkpoint/v1"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt or incompatible."""


@dataclass(frozen=True)
class TrainingIdentity:
    git_head: str
    split_manifest_sha256: str
    feature_manifest_sha256: str
    label_adapter: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_payload(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = dict(config)
    else:
        raise CheckpointError("config must be a dataclass instance or a mapping")
    return payload


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    global_step: int,
    identity: TrainingIdentity,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> str:
    if epoch < 0 or global_step < 0:
        raise CheckpointError("epoch and global_step must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "global_step": global_step,
        "identity": identity.as_dict(),
        "config": _config_payload(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
    }
    if extra is not None:
        payload["extra"] = extra
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except OSError as exc:
        raise CheckpointError(f"failed to write checkpoint: {target}") from exc
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return checkpoint_sha256(target)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    expected_identity: TrainingIdentity | None = None,
    expected_config: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"checkpoint is unreadable: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"unexpected checkpoint schema in {source}")

    if expected_identity is not None:
        stored = payload.get("identity", {})
        for key, value in expected_identity.as_dict().items():
            if stored.get(key) != value:
                raise CheckpointError(
                    f"checkpoint identity mismatch for {key}: "
                    f"stored={stored.get(key)!r} expected={value!r}"
                )

    if expected_config is not None:
        stored_config = payload.get("config", {})
        expected_payload = _config_payload(expected_config)
        for key, value in expected_payload.items():
            if stored_config.get(key) != value:
                raise CheckpointError(
                    f"checkpoint config mismatch for {key}: "
                    f"stored={stored_config.get(key)!r} expected={value!r}"
                )

    if model is not None:
        try:
            model.load_state_dict(payload["model_state_dict"])
        except (KeyError, RuntimeError) as exc:
            raise CheckpointError("model state_dict is incompatible") from exc
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is None:
            raise CheckpointError("checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler_state is None:
            raise CheckpointError("checkpoint has no scheduler state")
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, dict):
            raise CheckpointError("checkpoint has no RNG state")
        restore_rng_state(rng_state)

    return {
        "schema": payload["schema"],
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "identity": dict(payload.get("identity", {})),
        "config": dict(payload.get("config", {})),
        "sha256": checkpoint_sha256(source),
    }
