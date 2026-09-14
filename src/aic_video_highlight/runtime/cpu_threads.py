"""Container-aware CPU thread budgeting for CPU-heavy host-side work.

The host exposes all of its cores to ``nproc``/``os.cpu_count()``, but the
container CPU quota (cgroup v2 ``cpu.max`` or v1 ``cpu.cfs_quota_us``) is the
real budget.  PyTorch sizes its intra-op pool from the host core count, which
oversubscribes the quota; CPU-bound model host work (for example RT-DETR
anchor generation) then thrashes and becomes order-of-magnitude slower.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
DEFAULT_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
DEFAULT_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cgroup_cpu_quota(
    *,
    cpu_max: Path = DEFAULT_CPU_MAX,
    cfs_quota: Path = DEFAULT_V1_QUOTA,
    cfs_period: Path = DEFAULT_V1_PERIOD,
) -> float | None:
    """Return the container CPU quota in cores, or None when unlimited."""
    try:
        fields = cpu_max.read_text(encoding="utf-8").split()
    except OSError:
        fields = []
    if len(fields) == 2 and fields[0] != "max":
        try:
            quota, period = float(fields[0]), float(fields[1])
        except ValueError:
            quota = period = 0.0
        if quota > 0 and period > 0:
            return quota / period
    try:
        quota = float(cfs_quota.read_text(encoding="utf-8").strip())
        period = float(cfs_period.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if quota > 0 and period > 0:
        return quota / period
    return None


def cpu_budget(*, affinity: int | None = None, quota: float | None = None) -> int:
    """Effective integer CPU budget: min(CPU affinity, cgroup quota)."""
    if affinity is None:
        try:
            affinity = len(os.sched_getaffinity(0))
        except AttributeError:
            affinity = os.cpu_count() or 1
    if quota is None:
        quota = cgroup_cpu_quota()
    if quota is not None and quota >= 1:
        return max(1, min(affinity, int(quota)))
    return max(1, affinity)


def configure_math_threads(max_threads: int | None = None) -> dict[str, int]:
    """Cap torch-visible CPU threads to the container budget."""
    import torch

    previous = int(torch.get_num_threads())
    budget = max(1, int(max_threads) if max_threads else cpu_budget())
    effective = min(previous, budget)
    if effective != previous:
        torch.set_num_threads(effective)
    return {
        "previous_threads": previous,
        "effective_threads": effective,
        "cpu_budget": budget,
    }
