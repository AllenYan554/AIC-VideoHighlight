from __future__ import annotations

import inspect

import torch

from aic_video_highlight.ftnet.decoder import (
    NOT_CALIBRATED,
    MaskDrivenTemporalDecoder,
)


def test_decoder_uses_explicit_external_keep_mask_and_builds_segments() -> None:
    decoder = MaskDrivenTemporalDecoder()

    result = decoder.decode(
        probabilities=torch.tensor([0.9, 0.8, 0.2, 0.7]),
        timestamps=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        sequence_mask=torch.tensor([True, True, True, True]),
        keep_mask=torch.tensor([True, True, False, True]),
    )

    assert result.calibration_status == NOT_CALIBRATED
    assert [(segment.start_index, segment.end_index) for segment in result.segments] == [
        (0, 1),
        (3, 3),
    ]
    assert result.segments[0].mean_probability == torch.tensor(0.85).item()


def test_decoder_exposes_no_default_threshold() -> None:
    signature = inspect.signature(MaskDrivenTemporalDecoder.decode)

    assert "threshold" not in signature.parameters
    assert signature.parameters["keep_mask"].default is inspect.Parameter.empty
