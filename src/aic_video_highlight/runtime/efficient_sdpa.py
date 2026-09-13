"""Fail-closed PyTorch efficient-SDPA adapter for CUDA GQA full attention.

The adapter is registered under its own Transformers AttentionInterface key.
It never replaces the stock ``sdpa`` entry.  K/V expansion is a local tensor
view/materialization used only for the attention call; cache tensors are never
assigned or mutated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import (
    create_position_bias_mask,
    sdpa_attention_forward,
)


ATTENTION_INTERFACE_NAME = "vhicraft_local_efficient_sdpa"


class RuntimeBackendUnavailable(RuntimeError):
    """The requested efficient kernel cannot execute; no MATH fallback occurred."""


@dataclass(slots=True)
class RuntimeBackendEvidence:
    efficient_attention_calls: int = 0
    delegated_sdpa_calls: int = 0
    backend_failures: int = 0
    math_attention_calls: int = 0
    cache_kv_heads: int | None = None
    compute_kv_heads: int | None = None
    max_query_length: int = 0
    max_kv_length: int = 0

    def as_dict(self) -> dict[str, int | None | str]:
        return {
            "attention_backend": "pytorch_efficient_attention",
            "gqa_execution_mode": "temporary_kv_head_expansion",
            "efficient_attention_calls": self.efficient_attention_calls,
            "delegated_sdpa_calls": self.delegated_sdpa_calls,
            "backend_failures": self.backend_failures,
            "math_attention_calls": self.math_attention_calls,
            "cache_kv_heads": self.cache_kv_heads,
            "compute_kv_heads": self.compute_kv_heads,
            "max_query_length": self.max_query_length,
            "max_kv_length": self.max_kv_length,
        }


def temporarily_expand_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat K/V heads without mutating the caller-owned cache tensors."""
    key_heads = int(key.shape[1])
    value_heads = int(value.shape[1])
    if key_heads != value_heads:
        raise ValueError("key and value head counts differ")
    if key_heads <= 0 or query_heads <= key_heads or query_heads % key_heads:
        raise ValueError("temporary expansion requires divisible GQA head counts")
    repeats = query_heads // key_heads

    def _expand(tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, sequence, head_dim = tensor.shape
        return (
            tensor[:, :, None, :, :]
            .expand(batch, heads, repeats, sequence, head_dim)
            .reshape(batch, query_heads, sequence, head_dim)
        )

    return _expand(key), _expand(value)


def _forced_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool,
) -> torch.Tensor:
    # Passing a single backend disables every fallback, including MATH.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )


def _is_eligible_cuda_gqa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    return (
        query.device.type == "cuda"
        and query.dtype in (torch.float16, torch.bfloat16)
        and key_heads == int(value.shape[1])
        and query_heads > key_heads > 0
        and query_heads % key_heads == 0
        and int(getattr(module, "num_key_value_groups", 1))
        == query_heads // key_heads
    )


def local_efficient_sdpa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    *,
    evidence: RuntimeBackendEvidence,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Force efficient SDPA only for eligible CUDA fp16/bf16 GQA calls."""
    if not _is_eligible_cuda_gqa(module, query, key, value):
        evidence.delegated_sdpa_calls += 1
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )
    if kwargs.get("output_attentions", False):
        raise RuntimeBackendUnavailable(
            "local_efficient_sdpa does not support output_attentions=True"
        )

    query_heads = int(query.shape[1])
    cache_heads = int(key.shape[1])
    q_length = int(query.shape[2])
    kv_length = int(key.shape[2])
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    causal = bool(q_length > 1 and attention_mask is None and causal)
    if causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]
        if position_bias is not None:
            position_bias = position_bias[:, :, :, :q_length]
    if position_bias is not None:
        attention_mask = create_position_bias_mask(
            position_bias, attention_mask, causal, query, key
        )
        causal = False

    expanded_key, expanded_value = temporarily_expand_kv(
        key, value, query_heads=query_heads
    )
    try:
        output = _forced_efficient_attention(
            query,
            expanded_key,
            expanded_value,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=causal,
        )
    except Exception as exc:
        evidence.backend_failures += 1
        raise RuntimeBackendUnavailable(
            "local_efficient_sdpa could not execute EFFICIENT_ATTENTION; "
            f"MATH fallback is disabled: {type(exc).__name__}: {exc}"
        ) from exc

    evidence.efficient_attention_calls += 1
    evidence.cache_kv_heads = cache_heads
    evidence.compute_kv_heads = query_heads
    evidence.max_query_length = max(evidence.max_query_length, q_length)
    evidence.max_kv_length = max(evidence.max_kv_length, kv_length)
    return output.transpose(1, 2).contiguous(), None


def registered_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    evidence = getattr(module, "_vhicraft_runtime_evidence", None)
    if not isinstance(evidence, RuntimeBackendEvidence):
        raise RuntimeBackendUnavailable(
            "local_efficient_sdpa module lacks runtime evidence binding"
        )
    return local_efficient_sdpa_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        evidence=evidence,
        **kwargs,
    )


def register_attention_interface() -> str:
    """Add, but never overwrite, the dedicated Transformers attention key."""
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if ATTENTION_INTERFACE_NAME not in ALL_ATTENTION_FUNCTIONS:
        AttentionInterface.register(
            ATTENTION_INTERFACE_NAME, registered_attention_forward
        )
    elif ALL_ATTENTION_FUNCTIONS[ATTENTION_INTERFACE_NAME] is not registered_attention_forward:
        raise RuntimeError(f"attention key collision: {ATTENTION_INTERFACE_NAME}")
    return ATTENTION_INTERFACE_NAME


def bind_runtime_evidence(model: torch.nn.Module, evidence: RuntimeBackendEvidence) -> int:
    """Bind evidence only to Qwen3.5 text full-attention modules."""
    count = 0
    for module in model.modules():
        if (
            module.__class__.__name__ == "Qwen3_5Attention"
            and int(getattr(module, "num_key_value_groups", 1)) > 1
        ):
            module._vhicraft_runtime_evidence = evidence
            count += 1
    if count <= 0:
        raise RuntimeBackendUnavailable(
            "no Qwen3_5Attention GQA modules were found for runtime binding"
        )
    return count


def probe_efficient_backend(device: str, dtype: torch.dtype) -> None:
    """Small fail-fast kernel probe using the real 16Q/4KV/head_dim=256 shape."""
    query = torch.zeros((1, 16, 2, 256), device=device, dtype=dtype)
    key = torch.zeros((1, 4, 2, 256), device=device, dtype=dtype)
    value = torch.zeros_like(key)
    expanded_key, expanded_value = temporarily_expand_kv(key, value, query_heads=16)
    try:
        output = _forced_efficient_attention(
            query,
            expanded_key,
            expanded_value,
            attention_mask=None,
            dropout=0.0,
            scaling=256**-0.5,
            is_causal=True,
        )
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeBackendUnavailable(
            "EFFICIENT_ATTENTION availability probe failed; MATH fallback is disabled: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not bool(torch.isfinite(output).all()):
        raise RuntimeBackendUnavailable("EFFICIENT_ATTENTION probe returned non-finite values")
