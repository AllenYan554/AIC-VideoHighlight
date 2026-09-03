import pytest

from aic_video_highlight.highlight_retrieval.response_parser import (
    ResponseParseError,
    parse_highlight_response,
)


def test_parse_valid_highlight_json() -> None:
    segments = parse_highlight_response(
        '{"has_highlight":true,"segments":['
        '{"start_sec":4.2,"end_sec":8.6,"score":0.86,"reason":"goal"}]}',
        chunk_duration_sec=30.0,
        source_chunk=2,
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 4.2
    assert segments[0].end_sec == 8.6
    assert segments[0].score == 0.86
    assert segments[0].source_chunk == 2


def test_parse_allows_explicit_no_highlight() -> None:
    assert (
        parse_highlight_response(
            '{"has_highlight":false,"segments":[]}',
            chunk_duration_sec=12.0,
            source_chunk=0,
        )
        == []
    )


def test_parse_clips_end_to_chunk_and_records_reason() -> None:
    segments = parse_highlight_response(
        '{"has_highlight":true,"segments":['
        '{"start_sec":9,"end_sec":14,"score":0.7,"reason":"action"}]}',
        chunk_duration_sec=10.0,
        source_chunk=1,
    )

    assert segments[0].end_sec == 10.0
    assert "clipped" in segments[0].reason


@pytest.mark.parametrize(
    "payload",
    [
        "not json 1 2 3",
        '{"has_highlight":true,"segments":[{"start_sec":-1,"end_sec":2,"score":0.5}]}',
        '{"has_highlight":true,"segments":[{"start_sec":3,"end_sec":3,"score":0.5}]}',
        '{"has_highlight":true,"segments":[{"start_sec":1,"end_sec":2,"score":1.1}]}',
        '{"has_highlight":false,"segments":[{"start_sec":1,"end_sec":2,"score":0.5}]}',
    ],
)
def test_parse_rejects_malformed_or_inconsistent_output(payload: str) -> None:
    with pytest.raises(ResponseParseError):
        parse_highlight_response(payload, chunk_duration_sec=10.0, source_chunk=0)
