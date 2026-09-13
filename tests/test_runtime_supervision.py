"""Runtime supervision tests (pure helpers; no GPU, no dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.runtime.orchestrator import (  # noqa: E402
    WATCHDOG_DEFAULT_SEC,
    _child_environment,
    _watchdog_expired,
)


def test_child_environment_is_unbuffered_utf8_and_expandable(monkeypatch):
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = _child_environment()
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_child_environment_preserves_existing_allocator_settings(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    env = _child_environment()
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128,expandable_segments:True"


def test_child_environment_does_not_duplicate_expandable_segments(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env = _child_environment()
    assert env["PYTORCH_CUDA_ALLOC_CONF"].count("expandable_segments") == 1


def test_watchdog_expiry_only_after_limit():
    assert _watchdog_expired(0.0, WATCHDOG_DEFAULT_SEC, WATCHDOG_DEFAULT_SEC) is False
    assert _watchdog_expired(0.0, WATCHDOG_DEFAULT_SEC + 1.0, WATCHDOG_DEFAULT_SEC) is True
    assert _watchdog_expired(0.0, 10_000.0, 0.0) is False
