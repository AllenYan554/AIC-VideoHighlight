"""Container CPU-thread budgeting contracts."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.runtime.cpu_threads import (  # noqa: E402
    cgroup_cpu_quota,
    configure_math_threads,
    cpu_budget,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_cgroup_v2_quota_is_cores(tmp_path):
    cpu_max = _write(tmp_path / "cpu.max", "1500000 100000\n")
    quota = cgroup_cpu_quota(
        cpu_max=cpu_max, cfs_quota=tmp_path / "q", cfs_period=tmp_path / "p"
    )
    assert quota == pytest.approx(15.0)


def test_cgroup_v2_max_falls_back_to_v1(tmp_path):
    cpu_max = _write(tmp_path / "cpu.max", "max 100000\n")
    quota = _write(tmp_path / "quota", "300000\n")
    period = _write(tmp_path / "period", "100000\n")
    assert cgroup_cpu_quota(
        cpu_max=cpu_max, cfs_quota=quota, cfs_period=period
    ) == pytest.approx(3.0)


def test_missing_cgroup_returns_none(tmp_path):
    assert (
        cgroup_cpu_quota(
            cpu_max=tmp_path / "missing",
            cfs_quota=tmp_path / "missing2",
            cfs_period=tmp_path / "missing3",
        )
        is None
    )


def test_cpu_budget_is_min_of_affinity_and_quota():
    assert cpu_budget(affinity=192, quota=15.0) == 15
    assert cpu_budget(affinity=8, quota=15.0) == 8
    assert cpu_budget(affinity=4, quota=None) >= 1


class _FakeTorch(types.SimpleNamespace):
    def __init__(self, threads: int):
        super().__init__()
        self.threads = threads

    def get_num_threads(self) -> int:
        return self.threads

    def set_num_threads(self, value: int) -> None:
        self.threads = value


def test_configure_math_threads_caps_to_budget(monkeypatch):
    fake = _FakeTorch(96)
    monkeypatch.setitem(sys.modules, "torch", fake)
    info = configure_math_threads(max_threads=8)
    assert info == {
        "previous_threads": 96,
        "effective_threads": 8,
        "cpu_budget": 8,
    }
    assert fake.threads == 8


def test_configure_math_threads_never_raises_threads(monkeypatch):
    fake = _FakeTorch(4)
    monkeypatch.setitem(sys.modules, "torch", fake)
    info = configure_math_threads(max_threads=64)
    assert info["effective_threads"] == 4
    assert fake.threads == 4
