"""Device routing contracts: explicit CUDA never silently becomes CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.localization.rt_detr_localizer import (  # noqa: E402
    resolve_device,
)


def test_default_device_follows_cuda_availability():
    assert resolve_device(None, cuda_available=True) == "cuda"
    assert resolve_device(None, cuda_available=False) == "cpu"


def test_explicit_cpu_is_allowed():
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    assert resolve_device("cpu", cuda_available=False) == "cpu"


def test_explicit_cuda_is_preserved():
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    assert resolve_device("cuda:0", cuda_available=True) == "cuda:0"


def test_explicit_cuda_without_cuda_fails_fast():
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        resolve_device("cuda", cuda_available=False)


def test_indexed_cuda_without_cuda_fails_fast():
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        resolve_device("cuda:1", cuda_available=False)


def test_localizer_fails_fast_before_loading_weights(monkeypatch):
    import torch

    from aic_video_highlight.localization.rt_detr_localizer import RTDetrLocalizer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        RTDetrLocalizer(model_id="PekingU/rtdetr_r50vd", device="cuda")
