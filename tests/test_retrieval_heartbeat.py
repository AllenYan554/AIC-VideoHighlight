"""Chunk heartbeat contracts without model or dataset access."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.retrieval.pipeline import chunk_heartbeat_line  # noqa: E402


@pytest.mark.parametrize("total", [1, 3])
def test_chunk_heartbeat_has_begin_and_end_for_single_and_multi_chunk(total):
    for index in range(total):
        begin = chunk_heartbeat_line("qvh_test", index, total, "begin")
        end = chunk_heartbeat_line("qvh_test", index, total, "end")
        expected = f"chunk={index + 1}/{total}"
        assert "video_id=qvh_test" in begin
        assert expected in begin and "phase=begin" in begin
        assert expected in end and "phase=end" in end


def test_chunk_heartbeat_rejects_invalid_phase():
    with pytest.raises(ValueError, match="phase"):
        chunk_heartbeat_line("qvh_test", 0, 1, "middle")
