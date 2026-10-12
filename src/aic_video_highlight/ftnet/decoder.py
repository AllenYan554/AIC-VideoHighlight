"""Uncalibrated temporal decoder contracts for FTNet probabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


NOT_CALIBRATED = "NOT_CALIBRATED"


@dataclass(frozen=True)
class TemporalSegment:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    mean_probability: float


@dataclass(frozen=True)
class TemporalDecodeResult:
    segments: tuple[TemporalSegment, ...]
    calibration_status: str = NOT_CALIBRATED


class TemporalDecoder(ABC):
    """Interface boundary for future calibrated temporal decoding."""

    calibration_status = NOT_CALIBRATED

    @abstractmethod
    def decode(
        self,
        probabilities: torch.Tensor,
        timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        keep_mask: torch.Tensor,
    ) -> TemporalDecodeResult:
        """Decode a caller-supplied keep decision into continuous segments."""


class MaskDrivenTemporalDecoder(TemporalDecoder):
    """Basic placeholder that deliberately performs no threshold calibration.

    The caller must supply ``keep_mask`` from an external, separately calibrated
    policy.  Segment end times are the inclusive timestamps of their last kept
    frames; duration/gap semantics are intentionally not invented here.
    """

    def decode(
        self,
        probabilities: torch.Tensor,
        timestamps: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        keep_mask: torch.Tensor,
    ) -> TemporalDecodeResult:
        if probabilities.ndim != 1:
            raise ValueError("decoder inputs must be one-dimensional")
        expected = probabilities.shape
        if timestamps.shape != expected:
            raise ValueError("timestamps must match probabilities")
        if sequence_mask.shape != expected or keep_mask.shape != expected:
            raise ValueError("sequence_mask and keep_mask must match probabilities")
        if sequence_mask.dtype is not torch.bool or keep_mask.dtype is not torch.bool:
            raise TypeError("sequence_mask and keep_mask must have dtype torch.bool")
        valid_probabilities = probabilities[sequence_mask]
        valid_timestamps = timestamps[sequence_mask]
        if not bool(torch.isfinite(valid_probabilities).all()):
            raise ValueError("valid probabilities must be finite")
        if not bool(
            ((valid_probabilities >= 0.0) & (valid_probabilities <= 1.0)).all()
        ):
            raise ValueError("valid probabilities must be in [0, 1]")
        if not bool(torch.isfinite(valid_timestamps).all()):
            raise ValueError("valid timestamps must be finite")
        if valid_timestamps.numel() > 1 and not bool(
            (valid_timestamps[1:] >= valid_timestamps[:-1]).all()
        ):
            raise ValueError("valid timestamps must be nondecreasing")

        selected = sequence_mask & keep_mask
        segments: list[TemporalSegment] = []
        start: int | None = None
        for index in range(probabilities.shape[0] + 1):
            is_selected = index < probabilities.shape[0] and bool(selected[index])
            if is_selected and start is None:
                start = index
            if not is_selected and start is not None:
                end = index - 1
                segments.append(
                    TemporalSegment(
                        start_index=start,
                        end_index=end,
                        start_time=float(timestamps[start].detach().cpu()),
                        end_time=float(timestamps[end].detach().cpu()),
                        mean_probability=float(
                            probabilities[start : end + 1].mean().detach().cpu()
                        ),
                    )
                )
                start = None
        return TemporalDecodeResult(segments=tuple(segments))
