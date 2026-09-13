"""Explicit deployment-only runtime profiles for VHiCraft inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


RUNTIME_PROFILE_SCHEMA = "aic.vhicraft.runtime-profile/v1"
SUPPORTED_RUNTIME_PROFILES = {"default", "local_efficient_sdpa"}


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Numerics-preserving execution policy, separate from scientific config."""

    name: str
    attention_backend: str
    gqa_execution_mode: str
    chunk_heartbeat: bool
    fail_closed: bool
    hardware_scope: Mapping[str, Any]

    @property
    def uses_local_efficient_sdpa(self) -> bool:
        return self.name == "local_efficient_sdpa"

    def identity_fields(self) -> dict[str, Any]:
        return {
            "runtime_profile": self.name,
            "attention_backend": self.attention_backend,
            "gqa_execution_mode": self.gqa_execution_mode,
            "chunk_heartbeat": self.chunk_heartbeat,
            "runtime_fail_closed": self.fail_closed,
            "hardware_scope": dict(self.hardware_scope),
        }


def default_runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        name="default",
        attention_backend="transformers_sdpa_auto",
        gqa_execution_mode="transformers_default",
        chunk_heartbeat=False,
        fail_closed=False,
        hardware_scope={
            "device_type": "portable",
            "compute_dtypes": ["framework_default"],
            "requires_gqa": False,
        },
    )


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"runtime profile does not exist: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime profile must be a JSON object")
    required = {
        "schema_version",
        "runtime_profile",
        "attention_backend",
        "gqa_execution_mode",
        "chunk_heartbeat",
        "fail_closed",
        "hardware_scope",
    }
    unknown = sorted(set(payload) - required)
    missing = sorted(required - set(payload))
    if missing or unknown:
        raise ValueError(
            f"invalid runtime profile keys: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != RUNTIME_PROFILE_SCHEMA:
        raise ValueError(f"unsupported runtime profile schema: {payload['schema_version']}")
    name = str(payload["runtime_profile"])
    if name not in SUPPORTED_RUNTIME_PROFILES:
        raise ValueError(f"unsupported runtime profile: {name}")
    hardware_scope = payload["hardware_scope"]
    if not isinstance(hardware_scope, dict):
        raise ValueError("hardware_scope must be a JSON object")
    profile = RuntimeProfile(
        name=name,
        attention_backend=str(payload["attention_backend"]),
        gqa_execution_mode=str(payload["gqa_execution_mode"]),
        chunk_heartbeat=bool(payload["chunk_heartbeat"]),
        fail_closed=bool(payload["fail_closed"]),
        hardware_scope=hardware_scope,
    )
    expected = {
        "default": (
            "transformers_sdpa_auto",
            "transformers_default",
            False,
            False,
        ),
        "local_efficient_sdpa": (
            "pytorch_efficient_attention",
            "temporary_kv_head_expansion",
            True,
            True,
        ),
    }[name]
    actual = (
        profile.attention_backend,
        profile.gqa_execution_mode,
        profile.chunk_heartbeat,
        profile.fail_closed,
    )
    if actual != expected:
        raise ValueError(f"runtime profile {name} violates its frozen deployment contract")
    return profile


def runtime_machine_identity() -> dict[str, str]:
    """Return auditable execution-library and accelerator identity."""
    try:
        import torch
    except ImportError:
        return {
            "torch_version": "unavailable",
            "cuda_version": "unavailable",
            "gpu_name": "unavailable",
        }
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "unavailable"),
        "gpu_name": (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else "unavailable"
        ),
    }
