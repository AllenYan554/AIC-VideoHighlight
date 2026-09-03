from aic_video_highlight.highlight_retrieval.candidate_merger import merge_segments
from aic_video_highlight.highlight_retrieval.schemas import HighlightSegment


def test_merge_combines_cross_chunk_duplicates_deterministically() -> None:
    segments = [
        HighlightSegment(4.0, 8.0, 0.70, "action starts", 0),
        HighlightSegment(5.0, 9.0, 0.90, "action resolves", 1),
    ]

    merged = merge_segments(segments, tiou_threshold=0.5)

    assert len(merged) == 1
    assert (merged[0].start_sec, merged[0].end_sec) == (4.0, 9.0)
    assert merged[0].score == 0.90
    assert merged[0].reason == "action starts | action resolves"
    assert merged[0].source_chunk is None


def test_merge_keeps_distinct_candidates_in_time_order() -> None:
    segments = [
        HighlightSegment(20.0, 23.0, 0.8, "later", 1),
        HighlightSegment(1.0, 3.0, 0.7, "earlier", 0),
    ]

    merged = merge_segments(segments, tiou_threshold=0.5)

    assert [(item.start_sec, item.end_sec) for item in merged] == [(1.0, 3.0), (20.0, 23.0)]


def test_merge_does_not_merge_low_tiou_overlap() -> None:
    segments = [
        HighlightSegment(0.0, 10.0, 0.8, "wide", 0),
        HighlightSegment(8.0, 12.0, 0.7, "tail", 1),
    ]

    assert len(merge_segments(segments, tiou_threshold=0.5)) == 2
