from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.index import IndexEntry
from aic_video_highlight.ftnet.real_provider import (
    RECOVERY_MAX_NEW_TOKENS,
    RECOVERY_TEMPERATURE,
    _run_retrieval_with_recovery,
)
from aic_video_highlight.retrieval.pipeline import HighlightRetrievalConfig

VALID_RESPONSE = json.dumps(
    {
        "has_highlight": True,
        "segments": [
            {"start_sec": 0.0, "end_sec": 1.5, "score": 0.9, "reason": "test"}
        ],
    },
    ensure_ascii=False,
)


@dataclass
class _Response:
    content: str
    finish_reason: str | None = "stop"


class _StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def analyze_video(self, video, prompt, *, max_new_tokens, temperature, coarse_fps=None, enable_thinking=None):
        self.calls.append(
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "coarse_fps": coarse_fps,
            }
        )
        return self.responses.pop(0)


def _entry(video_path: Path) -> IndexEntry:
    return IndexEntry(
        dataset_id="youtube_highlights",
        video_id="probe-video",
        realized_video_id="probe-video",
        category="dog",
        split="TRAIN",
        relative_video_path=f"raw/dog/{video_path.name}",
        source_sha256="a" * 64,
        annotation_identity="dog/probe-video",
        width=320,
        height=240,
        frame_count=20,
        duration_sec=2.0,
        fps=10.0,
        fps_rational="10/1",
        timestamp_mode="CFR_FPS",
        decode_status="OK",
    )


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    path = tmp_path / "raw" / "dog" / "probe-video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_recovery_retries_parse_failure_with_temperature(tiny_video: Path) -> None:
    malformed = '{"has_highlight": true, "segments": [{"start_sec": 0.0, "end_sec": 1.0}]}}\n'
    client = _StubClient([_Response(malformed), _Response(VALID_RESPONSE)])
    config = HighlightRetrievalConfig()
    result, attempts = _run_retrieval_with_recovery(_entry(tiny_video), tiny_video, client, config)
    assert len(result.candidate_segments) == 1
    assert result.candidate_segments[0].start_sec == pytest.approx(0.0)
    assert [call["temperature"] for call in client.calls] == [config.temperature, RECOVERY_TEMPERATURE]
    assert attempts[0]["attempt"] == "primary" and attempts[0]["finish_reason"] == "stop"


def test_recovery_retries_truncation_with_token_budget(tiny_video: Path) -> None:
    truncated = _Response('```json\n{"has_highlight": true, "segments": [{"start_sec":', "length")
    client = _StubClient([truncated, _Response(VALID_RESPONSE, "stop")])
    config = HighlightRetrievalConfig()
    result, attempts = _run_retrieval_with_recovery(_entry(tiny_video), tiny_video, client, config)
    assert len(result.candidate_segments) == 1
    assert client.calls[1]["max_new_tokens"] == RECOVERY_MAX_NEW_TOKENS
    assert client.calls[1]["temperature"] == config.temperature


def test_recovery_second_failure_propagates(tiny_video: Path) -> None:
    malformed = '{"has_highlight": true, "segments": [{"start_sec": 0.0, "end_sec": 1.0}]}}\n'
    client = _StubClient([_Response(malformed), _Response(malformed)])
    config = HighlightRetrievalConfig()
    with pytest.raises(Exception):
        _run_retrieval_with_recovery(_entry(tiny_video), tiny_video, client, config)


def test_recovery_handles_multi_chunk_videos(tiny_video: Path) -> None:
    client = _StubClient([_Response(VALID_RESPONSE), _Response(VALID_RESPONSE)])
    config = HighlightRetrievalConfig(chunk_seconds=1.0, overlap_seconds=0.0)
    result, attempts = _run_retrieval_with_recovery(_entry(tiny_video), tiny_video, client, config)
    assert len(result.raw_chunk_outputs) == 2
    assert len(result.candidate_segments) == 2
    assert {row["chunk_index"] for row in attempts} == {0, 1}
