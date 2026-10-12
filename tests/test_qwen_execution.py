"""Unit tests for the Qwen execution audit helpers (no GPU, no ffmpeg)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.qwen_execution import (
    CHUNK_AUDIT_SCHEMA,
    chunk_audit_path,
    compare_audit_arms,
    execute_chunks,
    metrics_delta,
    summarize_request_metrics,
)


class _Chunk:
    def __init__(self, index: int) -> None:
        self.index = index
        self.start_sec = float(index * 25)
        self.end_sec = float(index * 25 + 30)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def test_execute_chunks_serial_order_and_lookahead() -> None:
    chunks = [_Chunk(index) for index in range(4)]
    events: list[tuple[str, int, float]] = []
    in_flight = {"count": 0, "max": 0, "request_order": []}

    def render(position: int, chunk: _Chunk):
        events.append(("render", chunk.index, time.perf_counter()))
        time.sleep(0.05)
        return Path(f"chunk-{chunk.index}.mp4"), 0.05

    def analyze(position: int, chunk: _Chunk, media: Path):
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        in_flight["request_order"].append((chunk.index, media.name))
        events.append(("request", chunk.index, time.perf_counter()))
        time.sleep(0.05)
        in_flight["count"] -= 1
        return chunk.index

    for overlap in (False, True):
        in_flight["max"] = 0
        in_flight["request_order"].clear()
        events.clear()
        records = execute_chunks(
            chunks, render=render, analyze=analyze, overlap_render=overlap
        )
        assert [record["value"] for record in records] == [0, 1, 2, 3]
        assert in_flight["max"] == 1
        assert [index for index, _ in in_flight["request_order"]] == [0, 1, 2, 3]
        assert [name for _, name in in_flight["request_order"]] == [
            f"chunk-{index}.mp4" for index in range(4)
        ]
        render_positions = [event[1] for event in events if event[0] == "render"]
        assert sorted(render_positions) == [0, 1, 2, 3]


def test_overlap_render_reduces_wall_clock() -> None:
    chunks = [_Chunk(index) for index in range(5)]

    def render(position: int, chunk: _Chunk):
        time.sleep(0.08)
        return Path(f"chunk-{chunk.index}.mp4"), 0.08

    def analyze(position: int, chunk: _Chunk, media: Path):
        time.sleep(0.08)
        return chunk.index

    started = time.perf_counter()
    execute_chunks(chunks, render=render, analyze=analyze, overlap_render=False)
    serial = time.perf_counter() - started
    started = time.perf_counter()
    records = execute_chunks(chunks, render=render, analyze=analyze, overlap_render=True)
    overlapped = time.perf_counter() - started
    assert [record["value"] for record in records] == [0, 1, 2, 3, 4]
    assert overlapped < serial * 0.8


def test_execute_chunks_propagates_render_failure() -> None:
    chunks = [_Chunk(index) for index in range(3)]

    def render(position: int, chunk: _Chunk):
        if chunk.index == 2:
            raise RuntimeError("ffmpeg exploded")
        return Path(f"chunk-{chunk.index}.mp4"), 0.0

    def analyze(position: int, chunk: _Chunk, media: Path):
        return chunk.index

    with pytest.raises(RuntimeError, match="ffmpeg exploded"):
        execute_chunks(chunks, render=render, analyze=analyze, overlap_render=True)


def _write_audit(root: Path, video_id: str, raw: str, segments: list[dict]) -> None:
    payload = {
        "schema": CHUNK_AUDIT_SCHEMA,
        "video_id": video_id,
        "chunks": [
            {
                "chunk_index": 0,
                "chunk_start_sec": 0.0,
                "chunk_end_sec": 30.0,
                "parse_success": True,
                "parse_error": None,
                "raw_response": raw,
                "parsed_segments": segments,
                "attempts": [{"attempt": "primary"}],
            }
        ],
    }
    path = chunk_audit_path(root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = {
        "schema": "aic.stage7.ftnet.retrieval-artifact/v1",
        "video_id": video_id,
        "chunks": [
            {
                "chunk_index": 0,
                "finish_reason": "stop",
                "parsed_segment_count": len(segments),
                "request_latency_sec": 1.0,
                "parse_success": True,
            }
        ],
        "raw_candidates": segments,
        "merged_candidates": segments,
    }
    artifact_path = root / "retrieval" / f"{video_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


def test_compare_audit_arms_reports_rates(tmp_path: Path) -> None:
    good = [
        {"start_sec": 1.0, "end_sec": 2.0, "score": 0.5, "reason": "a", "source_chunk": 0}
    ]
    other = [
        {"start_sec": 1.0, "end_sec": 2.5, "score": 0.5, "reason": "a", "source_chunk": 0}
    ]
    _write_audit(tmp_path / "left", "vid", '{"has_highlight":true}', good)
    _write_audit(tmp_path / "right", "vid", '{"has_highlight":true}', good)
    report = compare_audit_arms(tmp_path / "left", tmp_path / "right", ["vid"])
    assert report["all_chunks_exact"] and report["all_videos_equal"]
    assert report["chunk_semantic_match_rate"] == 1.0

    _write_audit(tmp_path / "third", "vid", '{"has_highlight":false}', other)
    report = compare_audit_arms(tmp_path / "left", tmp_path / "third", ["vid"])
    assert not report["all_chunks_exact"]
    assert report["chunk_semantic_match_rate"] == 0.0
    assert report["video_candidate_semantic_match_rate"] == 0.0


def test_metrics_delta_and_decomposition() -> None:
    before = {
        "metric_sum": {
            "vllm:request_prefill_time_seconds_sum": 1.0,
            "vllm:request_prefill_time_seconds_count": 1.0,
        }
    }
    after = {
        "metric_sum": {
            "vllm:request_prefill_time_seconds_sum": 4.0,
            "vllm:request_prefill_time_seconds_count": 3.0,
            "vllm:request_decode_time_seconds_sum": 6.0,
            "vllm:request_decode_time_seconds_count": 3.0,
            "vllm:prompt_tokens_total": 900.0,
            "vllm:generation_tokens_total": 150.0,
            "vllm:request_prompt_tokens_sum": 900.0,
            "vllm:request_prompt_tokens_count": 3.0,
            "vllm:request_generation_tokens_sum": 150.0,
            "vllm:request_generation_tokens_count": 3.0,
        }
    }
    delta = metrics_delta(before, after)
    summary = summarize_request_metrics(delta)
    assert summary["prefill_time_sec_total"] == 3.0
    assert summary["prefill_time_sec_count"] == 2.0
    assert summary["prefill_time_sec_mean"] == 1.5
    assert summary["decode_time_sec_total"] == 6.0
    assert summary["prompt_tokens"] == 900.0
    assert summary["request_prompt_tokens_mean"] == 300.0
    assert summary["request_generation_tokens_mean"] == 50.0
