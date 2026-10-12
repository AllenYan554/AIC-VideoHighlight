from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from aic_video_highlight.ftnet import phd2_upstream_benchmark as benchmark_module


def _record():
    return {
        "youtubeId": "video-a",
        "duration_seconds": 12.5,
        "duration_bin": "0-1min",
        "physical_resolution_arm": "v360",
        "relative_path": "raw/360p/video-a.mp4",
        "file_size_bytes": 4,
        "metadata": {"raw_sha256": "1" * 64},
    }


def test_phd2_entry_is_performance_only_and_has_no_gt_adapter():
    entry = benchmark_module._base_entry(_record())
    assert entry.dataset_id == "."
    assert entry.relative_video_path == "raw/360p/video-a.mp4"
    assert entry.annotation_identity == benchmark_module.PERFORMANCE_ONLY
    assert "target" not in entry.__dict__
    assert "annotation" not in entry.__dict__


def test_manifest_resolves_relative_path_from_dataset_root(tmp_path):
    video = tmp_path / "raw" / "360p" / "video-a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"data")
    payload = {
        "schema": benchmark_module.BENCHMARK_SCHEMA,
        "records": [_record()] * 32,
    }
    for index, row in enumerate(payload["records"]):
        row = dict(row)
        row["youtubeId"] = f"video-{index}"
        row["relative_path"] = "raw/360p/video-a.mp4"
        payload["records"][index] = row
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    loaded = benchmark_module.load_benchmark_manifest(
        manifest,
        expected_sha256=benchmark_module.sha256_file(manifest),
        dataset_root=tmp_path,
        verify_raw_sha256=False,
    )
    assert len(loaded["records"]) == 32


def test_prepare_index_uses_authoritative_pts_count(monkeypatch, tmp_path):
    base = benchmark_module._base_entry(_record())
    probed = replace(
        base,
        width=640,
        height=360,
        frame_count=101,
        duration_sec=10.0,
        fps=10.1,
        fps_rational="101/10",
        timestamp_mode="PTS_TABLE",
        decode_status="OK",
        probe_source="ffprobe",
    )
    monkeypatch.setattr(benchmark_module, "probe_index_entry", lambda *args, **kwargs: probed)
    monkeypatch.setattr(
        benchmark_module,
        "extract_pts_with_fallback",
        lambda *args, **kwargs: (("0", "1/10", "1/5"), "frame.pts_time"),
    )
    monkeypatch.setattr(benchmark_module, "resolve_video_path", lambda *args: tmp_path / "video.mp4")
    written = {}
    monkeypatch.setattr(
        benchmark_module,
        "write_index",
        lambda entries, **kwargs: written.setdefault("entries", list(entries)),
    )
    entries = benchmark_module.prepare_index(
        {"records": [_record()]}, dataset_root=tmp_path, output_path=tmp_path / "index.json"
    )
    assert entries[0].frame_count == 3
    assert entries[0].pts_timestamps == ("0", "1/10", "1/5")
    assert entries[0].pts_source == "frame.pts_time"
    assert entries[0].annotation_identity == benchmark_module.PERFORMANCE_ONLY


def test_retrieval_resume_uses_cache_without_request(tmp_path):
    entry = benchmark_module._base_entry(_record())
    cache = tmp_path / "retrieval" / "video-a.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps(
            {
                "video_id": "video-a",
                "chunks": [{"chunk_index": 0}],
                "merged_candidates": [],
                "timing": {
                    "probe_sec": 1.0,
                    "chunk_extraction_sec": 2.0,
                    "model_inference_sec": 3.0,
                    "parsing_sec": 0.1,
                    "merging_sec": 0.1,
                },
                "provenance": {"recovery_log": []},
            }
        ),
        encoding="utf-8",
    )
    summary = benchmark_module.run_retrieval(
        [entry],
        dataset_root=tmp_path,
        output_root=tmp_path,
        base_url="http://127.0.0.1:1/v1",
        concurrency=1,
        timeout_sec=0.01,
        events=benchmark_module.ProfileEvents(),
        phase=benchmark_module.PhaseState(),
        resume=True,
    )
    assert summary["cache_hit_count"] == 1
    assert summary["executed_video_count"] == 0


def test_detection_resume_uses_cache_without_model_load(tmp_path):
    entry = benchmark_module._base_entry(_record())
    artifact = tmp_path / "detections" / "video-a.npz"
    artifact.parent.mkdir(parents=True)
    np.savez(artifact, frames=np.asarray([0], dtype=np.int64))
    profile = {
        "video_id": "video-a",
        "frames": 1,
        "wall_sec": 1.0,
        "profile": {
            "capture_open_count": 1,
            "grabbed_frame_count": 1,
            "retrieved_frame_count": 1,
            "emitted_frame_count": 1,
            "repeated_prefix_decoded_frame_estimate": 0,
            "rtdetr_forward_count": 1,
            "rtdetr_pure_forward_cuda_seconds": 0.1,
        },
    }
    profile_path = tmp_path / "detection_profiles" / "video-a.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    summary = benchmark_module.run_detection(
        [entry],
        dataset_root=tmp_path,
        output_root=tmp_path,
        rtdetr_snapshot=tmp_path,
        batch_size=8,
        decode_window=64,
        decode_mode="one_pass",
        events=benchmark_module.ProfileEvents(),
        phase=benchmark_module.PhaseState(),
        resume=True,
    )
    assert summary["cache_hit_count"] == 1
    assert summary["executed_video_count"] == 0
    assert summary["model_load_count"] == 0
