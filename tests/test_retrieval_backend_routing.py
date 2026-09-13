"""Deployment routing contracts: AutoDL must use vLLM; Windows local keeps NF4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.composition.fresh_pipeline import FreshPipelineError  # noqa: E402
from aic_video_highlight.runtime.orchestrator import (  # noqa: E402
    build_retrieval_command,
    resolve_retrieval_backend,
)
from aic_video_highlight.runtime.paths import EnvironmentPaths  # noqa: E402


def _environment(name: str) -> EnvironmentPaths:
    return EnvironmentPaths.from_json(REPO / "configs" / "environments" / f"{name}.json")


def _inference(profile: str) -> dict:
    payload = json.loads(
        (REPO / "configs" / "profiles" / f"{profile}.json").read_text(encoding="utf-8")
    )
    return payload["inference"]


def _command(backend: str, tmp_path: Path) -> list[str]:
    return build_retrieval_command(
        python="python",
        manifest=tmp_path / "dev_selected.jsonl",
        video_root=tmp_path / "videos",
        output_dir=tmp_path / "retrieval",
        dataset_name="aic_highlight_dev",
        dataset_version="aic_highlight_dev_v1.1",
        runtime_profile_path=REPO / "configs" / "runtime" / "default.json",
        qwen_backend=backend,
        local_model_path=tmp_path / "models" / "Qwen3.5-4B",
        local_compute_dtype="float16",
        base_url="http://127.0.0.1:8000/v1",
        resume=False,
    )


def _environment_paths(**overrides) -> EnvironmentPaths:
    fields = {
        "name": "test",
        "repo": REPO,
        "datasets": REPO,
        "models": REPO,
        "hf_cache": REPO,
        "outputs": REPO,
        "logs": REPO,
        "cache": REPO,
        "tmp": REPO,
        "archive": REPO,
    }
    fields.update(overrides)
    return EnvironmentPaths(**fields)


def test_autodl_environment_declares_vllm_backend():
    assert _environment("autodl").retrieval_backend == "vllm"


def test_autodl_resolves_to_vllm_even_with_local_profile(tmp_path):
    backend = resolve_retrieval_backend(_inference("smoke"), _environment("autodl"))
    assert backend == "vllm"
    command = _command(backend, tmp_path)
    assert "--base-url" in command
    assert "http://127.0.0.1:8000/v1" in command
    assert "--local-model-path" not in command
    assert "--local-quantization" not in command
    assert "bnb-nf4" not in " ".join(command)


def test_windows_local_environment_keeps_local_quantized_backend(tmp_path):
    environment = _environment("windows_local")
    assert environment.retrieval_backend is None
    backend = resolve_retrieval_backend(_inference("smoke"), environment)
    assert backend == "transformers-bnb-nf4"
    command = _command(backend, tmp_path)
    assert "--local-model-path" in command
    assert "--local-quantization" in command
    assert "bnb-nf4" in command
    assert "--base-url" not in command


def test_explicit_environment_backend_overrides_profile():
    environment = _environment_paths(retrieval_backend="vllm")
    profile = {"qwen_backend": "transformers-bnb-nf4"}
    assert resolve_retrieval_backend(profile, environment) == "vllm"


def test_unknown_backend_is_rejected():
    profile = {"qwen_backend": "bogus"}
    with pytest.raises(FreshPipelineError, match="unsupported qwen_backend"):
        resolve_retrieval_backend(profile, _environment_paths())
