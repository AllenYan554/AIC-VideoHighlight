from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from aic_video_highlight.ftnet.index import IndexEntry, write_index
from aic_video_highlight.ftnet.integrity import run_integrity_gate, run_train_normalization
from aic_video_highlight.ftnet.pipeline import (
    STAGE_ASSEMBLE,
    STAGE_DETECTION,
    FailureJournal,
    MaterializationSettings,
    StatusStore,
    STATUS_SCHEMA,
    run_detection_stage,
    run_materialization,
)
from aic_video_highlight.ftnet import pipeline as pipeline_module
from aic_video_highlight.ftnet.provider_core import IDX0_FALLBACK_NONE, IDX0_FALLBACK_NORM
from aic_video_highlight.ftnet.real_provider import (
    RETRIEVAL_SCHEMA,
    detection_artifact_path,
    retrieval_artifact_path,
    save_detection,
)
from aic_video_highlight.ftnet.sampling import GridSample

LENGTH = 10


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_video(
    dataset_root: Path,
    work_root: Path,
    *,
    video_id: str,
    category: str = "dog",
    split: str = "TRAIN",
    boxes: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
) -> IndexEntry:
    annotation_dir = dataset_root / "youtube_highlights" / "annotations" / "upstream_repo" / category / video_id
    _write_json(annotation_dir / "mturk_label.json", [[[2, 6], [4, 8]], [5, 3]])
    _write_json(annotation_dir / "clip.json", [[2, 6], [4, 8]])

    timestamps = np.arange(LENGTH, dtype=np.float64) * 0.5
    frames = np.arange(LENGTH, dtype=np.int64)
    adjacency = np.ones(LENGTH, dtype=bool)
    adjacency[0] = False
    sample = GridSample(
        frames=frames,
        timestamps=timestamps,
        adjacency_mask=adjacency,
        target_timestamps=timestamps.copy(),
    )
    rng = np.random.default_rng(11)
    visual = rng.standard_normal((LENGTH, 256)).astype(np.float16)
    offsets = np.arange(0, 2 * (LENGTH + 1), 2, dtype=np.int32)
    detections_boxes = np.tile(np.array([boxes, (40.0, 40.0, 60.0, 60.0)], dtype=np.float32), (LENGTH, 1))
    scores = np.tile(np.array([0.9, 0.6], dtype=np.float32), LENGTH)
    labels = np.tile(np.array([1, 1], dtype=np.int32), LENGTH)
    save_detection(
        detection_artifact_path(work_root, video_id),
        visual=visual,
        offsets=offsets,
        boxes=detections_boxes,
        scores=scores,
        labels=labels,
        class_names=("__background__", "person"),
        sample=sample,
    )
    retrieval = {
        "schema": RETRIEVAL_SCHEMA,
        "video_id": video_id,
        "relative_video_path": f"raw/{category}/{video_id}.mp4",
        "source_sha256": "a" * 64,
        "duration_sec": 5.0,
        "chunks": [
            {
                "chunk_index": 0,
                "chunk_start_sec": 0.0,
                "chunk_end_sec": 2.5,
                "finish_reason": "stop",
                "parsed_segment_count": 1,
                "request_latency_sec": 0.1,
            },
            {
                "chunk_index": 1,
                "chunk_start_sec": 2.5,
                "chunk_end_sec": 5.0,
                "finish_reason": "stop",
                "parsed_segment_count": 1,
                "request_latency_sec": 0.1,
            },
        ],
        "raw_candidates": [
            {"chunk_index": 0, "start_sec": 0.5, "end_sec": 1.5, "score": 0.9},
            {"chunk_index": 1, "start_sec": 3.0, "end_sec": 4.0, "score": 0.8},
        ],
        "merged_candidates": [
            {"start_sec": 0.5, "end_sec": 1.5, "score": 0.9},
            {"start_sec": 3.0, "end_sec": 4.0, "score": 0.8},
        ],
        "timing": {},
        "provenance": {"qwen_model_id": "Qwen/Qwen3.5-4B"},
    }
    _write_json(retrieval_artifact_path(work_root, video_id), retrieval)

    return IndexEntry(
        dataset_id="youtube_highlights",
        video_id=video_id,
        realized_video_id=video_id,
        category=category,
        split=split,
        relative_video_path=f"raw/{category}/{video_id}.mp4",
        source_sha256="a" * 64,
        annotation_identity=f"{category}/{video_id}",
        width=100,
        height=100,
        frame_count=LENGTH,
        duration_sec=5.0,
        fps=2.0,
        fps_rational="2/1",
        timestamp_mode="CFR_FPS",
        decode_status="OK",
    )


def _settings(tmp_path: Path, *, idx0_fallback: str = IDX0_FALLBACK_NONE) -> MaterializationSettings:
    environment = tmp_path / "environment.json"
    _write_json(
        environment,
        {
            "name": "test",
            "repo": str(tmp_path),
            "datasets": str(tmp_path / "datasets"),
            "models": str(tmp_path / "models"),
            "hf_cache": str(tmp_path / "hf"),
            "outputs": str(tmp_path / "outputs"),
            "logs": str(tmp_path / "logs"),
            "cache": str(tmp_path / "cache"),
            "tmp": str(tmp_path / "tmp"),
            "archive": str(tmp_path / "archive"),
            "derived": str(tmp_path / "derived"),
        },
    )
    return MaterializationSettings(
        dataset_root=tmp_path / "datasets",
        work_root=tmp_path / "outputs" / "stage7_ftnet" / "runs" / "test",
        output_root=tmp_path / "derived" / "youtube_highlights_ftnet",
        index_path=tmp_path / "outputs" / "stage7_ftnet" / "index" / "frozen_index.json",
        qwen_snapshot=tmp_path / "models" / "Qwen3.5-4B",
        rtdetr_snapshot=tmp_path / "models" / "rtdetr_r50vd",
        environment_path=environment,
        idx0_fallback=idx0_fallback,
    )


def test_assemble_stage_is_resumable_and_fingerprint_sensitive(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    entry = _prepare_video(settings.dataset_root, settings.work_root, video_id="vid-a")
    write_index([entry], metadata={"dataset_id": "youtube_highlights"}, output_path=settings.index_path)

    summary = run_materialization(
        settings, splits=["TRAIN"], stages=[STAGE_ASSEMBLE], verify_sha256=False
    )
    assert summary["assembled"] == 1 and summary["failed"] == 0
    target = settings.output_root / "train" / "vid-a.safetensors"
    assert target.is_file()
    tensors = load_file(str(target))
    assert tensors["native"].shape == (LENGTH, 16)
    manifest = settings.output_root / "manifests" / "materialized_videos.json"
    assert manifest.is_file()

    again = run_materialization(
        settings, splits=["TRAIN"], stages=[STAGE_ASSEMBLE], verify_sha256=False
    )
    assert again["skipped"] == 1

    settings.idx0_fallback = IDX0_FALLBACK_NORM
    changed = run_materialization(
        settings, splits=["TRAIN"], stages=[STAGE_ASSEMBLE], verify_sha256=False
    )
    assert changed["assembled"] == 1 and changed["skipped"] == 0
    status = json.loads((settings.work_root / "status.json").read_text(encoding="utf-8"))
    assert status["schema"] == STATUS_SCHEMA
    row = status["videos"][0]
    assert row["idx0_fallback"] == IDX0_FALLBACK_NORM


def test_wave_processing_merges_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.wave_size = 1
    entries = [
        _prepare_video(settings.dataset_root, settings.work_root, video_id="wave-a"),
        _prepare_video(
            settings.dataset_root, settings.work_root, video_id="wave-b", category="skating"
        ),
    ]
    write_index(entries, metadata={"dataset_id": "youtube_highlights"}, output_path=settings.index_path)
    summary = run_materialization(
        settings, splits=["TRAIN"], stages=[STAGE_ASSEMBLE], verify_sha256=False
    )
    assert summary["assembled"] == 2 and summary["failed"] == 0
    manifest = json.loads(
        (settings.output_root / "manifests" / "materialized_videos.json").read_text(encoding="utf-8")
    )
    assert [row["canonical_video_id"] for row in manifest["records"]] == ["wave-a", "wave-b"]


def test_normalization_and_integrity_gate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    entries = [
        _prepare_video(settings.dataset_root, settings.work_root, video_id="train-a"),
        _prepare_video(settings.dataset_root, settings.work_root, video_id="train-b", category="skiing"),
        _prepare_video(
            settings.dataset_root, settings.work_root, video_id="val-a", category="surfing", split="VALIDATION"
        ),
        _prepare_video(
            settings.dataset_root,
            settings.work_root,
            video_id="cal-a",
            category="parkour",
            split="CALIBRATION",
        ),
    ]
    write_index(entries, metadata={"dataset_id": "youtube_highlights"}, output_path=settings.index_path)
    summary = run_materialization(settings, stages=[STAGE_ASSEMBLE], verify_sha256=False)
    assert summary["assembled"] == 4 and summary["failed"] == 0

    payload = run_train_normalization(settings.output_root)
    assert payload["train_video_count"] == 2
    stats_path = settings.output_root / "normalization" / "normalization_stats.json"
    assert stats_path.is_file()
    assert (settings.output_root / "normalization" / "normalization_stats.json.sha256").is_file()

    report = run_integrity_gate(settings.output_root, entries=entries)
    assert report["status"] == "PASS"
    assert report["total_materialized"] == 4
    assert report["counts"] == {"TRAIN": 2, "VALIDATION": 1, "CALIBRATION": 1}

    target = settings.output_root / "train" / "train-a.safetensors"
    payload = load_file(str(target))
    payload["native"][0, 0] = np.nan
    from safetensors.numpy import save_file

    save_file(payload, str(target))
    failed = run_integrity_gate(settings.output_root, entries=entries, write=False)
    assert failed["status"] == "FAIL"
    assert any("train-a" in item for item in failed["nan_fields"])


class _ProgressStub:
    stage = ""
    current_video = None
    detection_done = 0
    failed = 0

    def emit(self, **_kwargs) -> None:
        return None


def test_detection_stage_loads_rtdetr_once_for_multiple_videos(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    entries = [
        _prepare_video(settings.dataset_root, settings.work_root, video_id="load-a"),
        _prepare_video(settings.dataset_root, settings.work_root, video_id="load-b"),
    ]
    for entry in entries:
        detection_artifact_path(settings.work_root, entry.video_id).unlink()

    constructed = []
    shared = object()

    def fake_create(**_kwargs):
        constructed.append(shared)
        return shared

    observed = []

    def fake_detection(entry, **kwargs):
        observed.append(kwargs["localizer"])
        return {
            "frames": 10,
            "candidates": 20,
            "wall_sec": 0.5,
            "decode_s": 0.1,
            "rtdetr_load_s": kwargs["model_load_s"],
            "rtdetr_inference_s": 0.3,
            "serialize_s": 0.1,
            "model_load_count": kwargs["model_load_count"],
            "batch_count": 2,
        }

    monkeypatch.setattr(pipeline_module, "create_rtdetr_localizer", fake_create)
    monkeypatch.setattr(pipeline_module, "run_detection_for_entry", fake_detection)
    status = StatusStore(tmp_path / "status.json")
    journal = FailureJournal(tmp_path / "failures.json")
    progress = _ProgressStub()

    failures = run_detection_stage(
        settings, entries, progress=progress, journal=journal, status=status
    )

    assert failures == 0
    assert constructed == [shared]
    assert observed == [shared, shared]
    assert status.get("load-a")["model_load_count"] == 1
    assert status.get("load-b")["model_load_count"] == 0


def test_wave_size_does_not_recreate_heavy_stage_lifecycles(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    settings.wave_size = 1
    entries = [
        _prepare_video(settings.dataset_root, settings.work_root, video_id="wave-life-a"),
        _prepare_video(settings.dataset_root, settings.work_root, video_id="wave-life-b"),
    ]
    write_index(entries, metadata={"dataset_id": "youtube_highlights"}, output_path=settings.index_path)
    calls = []

    def fake_detection_stage(_settings, selected, **_kwargs):
        calls.append([entry.video_id for entry in selected])
        return 0

    monkeypatch.setattr(pipeline_module, "run_detection_stage", fake_detection_stage)
    summary = run_materialization(
        settings,
        stages=[STAGE_DETECTION],
        verify_sha256=False,
    )

    assert calls == [["wave-life-a", "wave-life-b"]]
    assert summary["rtdetr_model_load_count"] == 0
