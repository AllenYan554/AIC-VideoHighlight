"""Reset-aware generic temporal stabilization of focus parameters.

Reuses the generic smoothing idea of the frozen TS artifacts while removing
any fixed AIC crop / fixed-ratio binding.  Smoothing never crosses a source
discontinuity, shot boundary or subject-identity reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TS_VERSION = "generic-focus-stabilizer-v1"


class GenericStabilizationError(ValueError):
    """Raised when stabilization inputs violate the contract."""


@dataclass(frozen=True, slots=True)
class StabilizedFocus:
    ts_version: str
    stabilized_focus_center: np.ndarray
    stabilized_focus_scale: np.ndarray
    stabilized_focus_window: np.ndarray
    reset_flag: np.ndarray


def stabilize_focus(
    focus_center: np.ndarray,
    focus_scale: np.ndarray,
    focus_window: np.ndarray,
    *,
    adjacency_mask: np.ndarray | None = None,
    reset_flags: np.ndarray | None = None,
    alpha: float = 0.6,
) -> StabilizedFocus:
    center = np.asarray(focus_center, dtype=np.float64)
    scale = np.asarray(focus_scale, dtype=np.float64)
    window = np.asarray(focus_window, dtype=np.float64)
    if center.ndim != 2 or center.shape[1] != 2:
        raise GenericStabilizationError("focus_center must have shape [T,2]")
    length = center.shape[0]
    if scale.shape != (length,):
        raise GenericStabilizationError("focus_scale must have shape [T]")
    if window.shape != (length, 4):
        raise GenericStabilizationError("focus_window must have shape [T,4]")
    if not 0.0 <= alpha < 1.0:
        raise GenericStabilizationError("alpha must be in [0,1)")

    if adjacency_mask is None:
        adjacency_mask = np.ones(length, dtype=bool)
    else:
        adjacency_mask = np.asarray(adjacency_mask, dtype=bool)
        if adjacency_mask.shape != (length,):
            raise GenericStabilizationError("adjacency_mask must have shape [T]")
    if reset_flags is None:
        reset_flags = np.zeros(length, dtype=bool)
    else:
        reset_flags = np.asarray(reset_flags, dtype=bool)
        if reset_flags.shape != (length,):
            raise GenericStabilizationError("reset_flags must have shape [T]")

    out_center = np.empty_like(center)
    out_scale = np.empty_like(scale)
    out_window = np.empty_like(window)
    out_reset = np.zeros(length, dtype=bool)

    for index in range(length):
        discontinuity = (
            index == 0
            or bool(reset_flags[index])
            or not bool(adjacency_mask[index])
        )
        out_reset[index] = discontinuity
        if discontinuity:
            out_center[index] = center[index]
            out_scale[index] = scale[index]
            out_window[index] = window[index]
        else:
            out_center[index] = alpha * out_center[index - 1] + (1.0 - alpha) * center[index]
            out_scale[index] = alpha * out_scale[index - 1] + (1.0 - alpha) * scale[index]
            out_window[index] = alpha * out_window[index - 1] + (1.0 - alpha) * window[index]

    return StabilizedFocus(
        ts_version=TS_VERSION,
        stabilized_focus_center=out_center,
        stabilized_focus_scale=out_scale,
        stabilized_focus_window=out_window,
        reset_flag=out_reset,
    )


def stabilized_focus_velocity(
    stabilized_focus_center: np.ndarray,
    timestamps: np.ndarray,
    *,
    adjacency_mask: np.ndarray,
) -> np.ndarray:
    """Normalized focus-center displacement per second, raw (log1p applied later)."""

    center = np.asarray(stabilized_focus_center, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    adjacency = np.asarray(adjacency_mask, dtype=bool)
    length = center.shape[0]
    if ts.shape != (length,) or adjacency.shape != (length,):
        raise GenericStabilizationError("timestamps/adjacency_mask must have shape [T]")
    velocity = np.zeros(length, dtype=np.float64)
    for index in range(1, length):
        if not bool(adjacency[index]):
            continue
        dt = ts[index] - ts[index - 1]
        if dt <= 0:
            continue
        displacement = float(np.linalg.norm(center[index] - center[index - 1]))
        velocity[index] = displacement / dt
    return velocity
