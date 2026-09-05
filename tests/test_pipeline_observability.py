from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.pipeline import (
    HighlightRetrievalConfig,
    HighlightRetrievalPipeline,
    parse_saved_raw_outputs,
)
from aic_video_highlight.highlight_retrieval.qwen_vllm_client import ModelResponse
from aic_video_highlight.highlight_retrieval.response_parser import (
    ResponseParseError,
    TruncatedResponseError,
)
from aic_video_highlight.highlight_retrieval.schemas import VideoMeta


class MalformedClient:
    def analyze_video(self, *_args, **_kwargs) -> ModelResponse:
        return ModelResponse(content="not valid json", finish_reason="stop")


class TruncatedClient:
    def analyze_video(self, *_args, **_kwargs) -> ModelResponse:
        return ModelResponse(
            content='```json\n{"has_highlight":true,"segments":[{"start_sec":1,',
            finish_reason="length",
        )


class ValidClient:
    def analyze_video(self, *_args, **_kwargs) -> ModelResponse:
        return ModelResponse(
            content=(
                '{"has_highlight":true,"segments":['
                '{"start_sec":1,"end_sec":3,"score":0.8,"reason":"action"}]}'
            ),
            finish_reason="stop",
        )


def _pipeline_with(
    client, tmp_path, monkeypatch
) -> tuple[HighlightRetrievalPipeline, Path]:
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
    pipeline = HighlightRetrievalPipeline(client, HighlightRetrievalConfig())
    return pipeline, video


def test_pipeline_persists_raw_output_before_parse_failure(tmp_path, monkeypatch) -> None:
    persisted: list[dict] = []
    pipeline, video = _pipeline_with(MalformedClient(), tmp_path, monkeypatch)

    with pytest.raises(ResponseParseError):
        pipeline.run(video, raw_output_sink=lambda record: persisted.append(record.copy()))

    assert persisted[0]["raw_response"] == "not valid json"
    assert persisted[0]["finish_reason"] == "stop"
    assert persisted[0]["parse_success"] is None
    assert persisted[-1]["parse_success"] is False
    assert "strict JSON" in persisted[-1]["parse_error"]
    assert persisted[-1]["request_latency_sec"] >= 0


def test_pipeline_rejects_truncated_output_without_parsing(tmp_path, monkeypatch) -> None:
    persisted: list[dict] = []
    pipeline, video = _pipeline_with(TruncatedClient(), tmp_path, monkeypatch)

    with pytest.raises(TruncatedResponseError):
        pipeline.run(video, raw_output_sink=lambda record: persisted.append(record.copy()))

    assert persisted[0]["finish_reason"] == "length"
    assert persisted[-1]["parse_success"] is False
    assert "TruncatedResponseError" in persisted[-1]["parse_error"]
    assert "finish_reason=length" in persisted[-1]["parse_error"]


def test_pipeline_records_stop_finish_reason_for_valid_output(tmp_path, monkeypatch) -> None:
    persisted: list[dict] = []
    pipeline, video = _pipeline_with(ValidClient(), tmp_path, monkeypatch)

    result = pipeline.run(video, raw_output_sink=lambda record: persisted.append(record.copy()))

    assert persisted[0]["finish_reason"] == "stop"
    assert persisted[-1]["parse_success"] is True
    assert [(item.start_sec, item.end_sec) for item in result.segments] == [(1.0, 3.0)]


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


def test_saved_raw_outputs_reject_finish_reason_length() -> None:
    with pytest.raises(TruncatedResponseError, match="finish_reason=length"):
        parse_saved_raw_outputs(
            [
                {
                    "chunk_index": 0,
                    "chunk_start_sec": 0.0,
                    "chunk_end_sec": 5.0,
                    "raw_response": '{"has_highlight":true,"segments":[{"start_sec":1,',
                    "finish_reason": "length",
                    "request_latency_sec": 4.2,
                }
            ],
            tiou_threshold=0.5,
        )
