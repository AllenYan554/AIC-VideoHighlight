from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.pipeline import (
    HighlightRetrievalConfig,
    HighlightRetrievalPipeline,
    parse_saved_raw_outputs,
)
from aic_video_highlight.highlight_retrieval.response_parser import ResponseParseError
from aic_video_highlight.highlight_retrieval.schemas import VideoMeta


class MalformedClient:
    def analyze_video(self, *_args, **_kwargs) -> str:
        return "not valid json"


def test_pipeline_persists_raw_output_before_parse_failure(tmp_path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        "aic_video_highlight.highlight_retrieval.pipeline.probe_video",
        lambda *_args, **_kwargs: VideoMeta(
            video_id="clip",
            path=video,
            duration_sec=5.0,
            fps=25.0,
            width=320,
            height=180,
            frame_count=125,
        ),
    )
    persisted: list[dict] = []
    pipeline = HighlightRetrievalPipeline(MalformedClient(), HighlightRetrievalConfig())

    with pytest.raises(ResponseParseError):
        pipeline.run(video, raw_output_sink=lambda record: persisted.append(record.copy()))

    assert persisted[0]["raw_response"] == "not valid json"
    assert persisted[0]["parse_success"] is None
    assert persisted[-1]["parse_success"] is False
    assert "strict JSON" in persisted[-1]["parse_error"]
    assert persisted[-1]["request_latency_sec"] >= 0


def test_saved_raw_outputs_can_be_reparsed_without_model_call() -> None:
    candidates, merged, records, timing = parse_saved_raw_outputs(
        [
            {
                "chunk_index": 0,
                "chunk_start_sec": 0.0,
                "chunk_end_sec": 5.0,
                "raw_response": '{"has_highlight":true,"segments":['
                '{"start_sec":1,"end_sec":3,"score":0.8,"reason":"action"}]}',
                "request_latency_sec": 4.2,
            }
        ],
        tiou_threshold=0.5,
    )

    assert [(item.start_sec, item.end_sec) for item in candidates] == [(1.0, 3.0)]
    assert [(item.start_sec, item.end_sec) for item in merged] == [(1.0, 3.0)]
    assert records[0]["parse_success"] is True
    assert timing["parsing_sec"] >= 0
