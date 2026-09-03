import pytest

from aic_video_highlight.stage1.chunker import build_chunks


def test_build_chunks_covers_long_video_with_overlap_without_overflow() -> None:
    chunks = build_chunks(80.0, chunk_seconds=30.0, overlap_seconds=5.0)

    assert [(chunk.start_sec, chunk.end_sec) for chunk in chunks] == [
        (0.0, 30.0),
        (25.0, 55.0),
        (50.0, 80.0),
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_build_chunks_returns_one_chunk_for_short_video() -> None:
    chunks = build_chunks(12.5, chunk_seconds=30.0, overlap_seconds=5.0)

    assert len(chunks) == 1
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 12.5


@pytest.mark.parametrize(
    ("duration", "chunk_seconds", "overlap_seconds"),
    [(0.0, 30.0, 5.0), (30.0, 0.0, 0.0), (30.0, 5.0, 5.0)],
)
def test_build_chunks_rejects_invalid_parameters(
    duration: float, chunk_seconds: float, overlap_seconds: float
) -> None:
    with pytest.raises(ValueError):
        build_chunks(duration, chunk_seconds, overlap_seconds)
