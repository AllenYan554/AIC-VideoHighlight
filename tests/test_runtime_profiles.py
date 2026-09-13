"""Runtime-profile and efficient-SDPA contracts (CPU-only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from aic_video_highlight.runtime.efficient_sdpa import (  # noqa: E402
    RuntimeBackendEvidence,
    RuntimeBackendUnavailable,
    local_efficient_sdpa_forward,
    temporarily_expand_kv,
)
from aic_video_highlight.runtime.orchestrator import build_run_identity  # noqa: E402
from aic_video_highlight.runtime.profiles import load_runtime_profile  # noqa: E402
from scripts import run_vhicraft  # noqa: E402


def test_runtime_profiles_are_explicit_and_default_is_portable():
    default = load_runtime_profile(REPO / "configs" / "runtime" / "default.json")
    local = load_runtime_profile(
        REPO / "configs" / "runtime" / "local_efficient_sdpa.json"
    )

    assert default.name == "default"
    assert default.attention_backend == "transformers_sdpa_auto"
    assert default.gqa_execution_mode == "transformers_default"
    assert default.chunk_heartbeat is False
    assert default.fail_closed is False

    assert local.name == "local_efficient_sdpa"
    assert local.attention_backend == "pytorch_efficient_attention"
    assert local.gqa_execution_mode == "temporary_kv_head_expansion"
    assert local.chunk_heartbeat is True
    assert local.fail_closed is True


def test_cli_omission_resolves_to_default_runtime_profile():
    assert run_vhicraft._resolve_runtime_profile("default") == (
        REPO / "configs" / "runtime" / "default.json"
    ).resolve()


def test_temporary_gqa_expansion_keeps_cache_heads_unchanged():
    key = torch.randn(1, 4, 7, 8)
    value = torch.randn(1, 4, 7, 8)
    key_shape = key.shape
    value_shape = value.shape

    expanded_key, expanded_value = temporarily_expand_kv(
        key, value, query_heads=16
    )

    assert key.shape == key_shape == (1, 4, 7, 8)
    assert value.shape == value_shape == (1, 4, 7, 8)
    assert expanded_key.shape == expanded_value.shape == (1, 16, 7, 8)
    for group in range(4):
        assert torch.equal(expanded_key[:, group * 4], key[:, group])
        assert torch.equal(expanded_value[:, group * 4], value[:, group])


def test_non_gqa_mha_is_delegated_to_stock_sdpa(monkeypatch):
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 4, 3, 8)
    value = torch.randn(1, 4, 3, 8)
    sentinel = torch.randn(1, 3, 4, 8)
    calls = []

    def stock(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel, None

    monkeypatch.setattr(
        "aic_video_highlight.runtime.efficient_sdpa.sdpa_attention_forward", stock
    )
    evidence = RuntimeBackendEvidence()
    module = SimpleNamespace(num_key_value_groups=1, is_causal=False)

    output, weights = local_efficient_sdpa_forward(
        module, query, key, value, None, evidence=evidence
    )

    assert output is sentinel
    assert weights is None
    assert len(calls) == 1
    assert evidence.efficient_attention_calls == 0
    assert evidence.delegated_sdpa_calls == 1


def test_cpu_gqa_is_delegated_and_never_forced(monkeypatch):
    query = torch.randn(1, 16, 3, 8)
    key = torch.randn(1, 4, 3, 8)
    value = torch.randn(1, 4, 3, 8)
    calls = []

    def stock(*args, **kwargs):
        calls.append(True)
        return torch.zeros(1, 3, 16, 8), None

    monkeypatch.setattr(
        "aic_video_highlight.runtime.efficient_sdpa.sdpa_attention_forward", stock
    )
    evidence = RuntimeBackendEvidence()
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    local_efficient_sdpa_forward(
        module, query, key, value, None, evidence=evidence
    )

    assert calls == [True]
    assert evidence.efficient_attention_calls == 0
    assert evidence.delegated_sdpa_calls == 1


def test_efficient_backend_failure_is_fail_closed_without_math_fallback(monkeypatch):
    class FakeCudaTensor:
        device = torch.device("cuda")
        dtype = torch.float16
        shape = (1, 16, 3, 8)

    query = FakeCudaTensor()
    key = SimpleNamespace(shape=(1, 4, 3, 8))
    value = SimpleNamespace(shape=(1, 4, 3, 8))
    evidence = RuntimeBackendEvidence()
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)

    monkeypatch.setattr(
        "aic_video_highlight.runtime.efficient_sdpa.temporarily_expand_kv",
        lambda key, value, query_heads: (key, value),
    )
    monkeypatch.setattr(
        "aic_video_highlight.runtime.efficient_sdpa._forced_efficient_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no viable backend")),
    )

    with pytest.raises(RuntimeBackendUnavailable, match="no viable backend"):
        local_efficient_sdpa_forward(
            module, query, key, value, None, evidence=evidence
        )

    assert evidence.efficient_attention_calls == 0
    assert evidence.backend_failures == 1
    assert evidence.math_attention_calls == 0


def test_run_identity_binds_runtime_profile_and_machine_fields(tmp_path, monkeypatch):
    files = {}
    for name in ("config", "protocol", "manifest", "runtime"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        files[name] = path
    runtime = load_runtime_profile(REPO / "configs" / "runtime" / "default.json")
    monkeypatch.setattr(
        "aic_video_highlight.runtime.orchestrator._git_head", lambda: "a" * 40
    )
    monkeypatch.setattr(
        "aic_video_highlight.runtime.orchestrator.runtime_machine_identity",
        lambda: {
            "torch_version": "test-torch",
            "cuda_version": "test-cuda",
            "gpu_name": "test-gpu",
        },
    )

    identity = build_run_identity(
        config_path=files["config"],
        protocol_path=files["protocol"],
        selected_manifest=files["manifest"],
        profile="smoke",
        runtime_profile=runtime,
        runtime_profile_path=files["runtime"],
    )

    assert identity["runtime_profile"] == "default"
    assert identity["attention_backend"] == "transformers_sdpa_auto"
    assert identity["gqa_execution_mode"] == "transformers_default"
    assert identity["runtime"]["gpu_name"] == "test-gpu"
    assert len(identity["runtime_profile_sha256"]) == 64


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/highlight_retrieval.yaml",
        "configs/profiles/smoke.json",
        "configs/profiles/dev166.json",
        "configs/profiles/official_test.json",
        "configs/vhicraft_v1_protocol.json",
    ],
)
def test_scientific_configuration_bytes_match_starting_head(relative_path):
    completed = subprocess.run(
        ["git", "diff", "--exit-code", "--", relative_path],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout.decode(errors="replace")
