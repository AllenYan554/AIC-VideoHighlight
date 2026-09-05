import json
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.baseline_experiment import (
    aggregate_results,
    load_baseline_samples,
    persist_run_config,
    resume_decision,
    run_with_failure_isolation,
    write_timeline_figure,
)


def test_manifest_maps_source_video_to_clip_local_reference(tmp_path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    source = video_root / "source_60.0_210.0.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "sample_index": 1,
                    "group": "single_segment",
                    "video_id": "sample-1",
                    "dataset_split": "train",
                    "source_group": "source_60.0_210.0",
                    "filename": source.name,
                    "hf_path": f"s/{source.name}",
                    "clip_start_sec": 10.0,
                    "clip_end_sec": 15.0,
                    "segment_count": 1,
                    "weak_reference_segments": [{"start_sec": 1.0, "end_sec": 3.0}],
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_baseline_samples(manifest, video_root, video_id="sample-1")

    assert len(samples) == 1
    assert samples[0]["source_video_path"] == str(source.resolve())
    assert samples[0]["clip_duration_sec"] == 5.0
    assert samples[0]["weak_reference_segments"] == [{"start_sec": 1.0, "end_sec": 3.0}]


def test_jsonl_manifest_resolves_reference_path_to_clip_local(tmp_path) -> None:
    video_root = tmp_path / "dataset"
    (video_root / "videos" / "dev").mkdir(parents=True)
    source = video_root / "videos" / "dev" / "abc_60.0_210.0.mp4"
    source.write_bytes(b"video")
    references = video_root / "references" / "dev.jsonl"
    references.parent.mkdir(parents=True)
    references.write_text(
        json.dumps(
            {
                "video_id": "qvh_000001_9x16",
                "split": "dev",
                "clip_start_sec": 81.61,
                "clip_end_sec": 87.53,
                "coordinate_system": "clip-local seconds",
                "segments": [{"start_sec": 0.0, "end_sec": 5.03}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "qvh_000001_9x16",
                "split": "dev",
                "relative_video_path": "videos/dev/abc_60.0_210.0.mp4",
                "clip_start_sec": 81.61,
                "clip_end_sec": 87.53,
                "reference_segment_count": 1,
                "reference_coverage": 0.85,
                "reference_path": "references/dev.jsonl#qvh_000001_9x16",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_baseline_samples(
        manifest,
        video_root,
        dataset_name="aic_highlight_dev",
        dataset_version="aic_highlight_dev_v1.1",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample["source_video_path"] == str(source.resolve())
    assert sample["clip_duration_sec"] == pytest.approx(5.92)
    assert sample["clip_start_sec"] == pytest.approx(81.61)
    assert sample["weak_reference_segments"] == [{"start_sec": 0.0, "end_sec": 5.03}]
    assert sample["dataset_name"] == "aic_highlight_dev"
    assert sample["dataset_version"] == "aic_highlight_dev_v1.1"
    assert sample["split"] == "dev"
    assert sample["group"] == "refseg_1"


def test_jsonl_manifest_accepts_embedded_weak_references(tmp_path) -> None:
    video_root = tmp_path / "dataset"
    video_root.mkdir()
    source = video_root / "abc_60.0_210.0.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "qvh_000002_9x16",
                "split": "dev",
                "relative_video_path": "abc_60.0_210.0.mp4",
                "clip_start_sec": 130.194,
                "clip_end_sec": 141.794,
                "reference_segment_count": 2,
                "weak_reference_segments": [
                    {"start_sec": 1.83, "end_sec": 4.26},
                    {"start_sec": 5.46, "end_sec": 11.03},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_baseline_samples(manifest, video_root)

    assert samples[0]["clip_duration_sec"] == pytest.approx(11.6)
    assert len(samples[0]["weak_reference_segments"]) == 2
    assert samples[0]["group"] == "refseg_2"


def test_jsonl_manifest_rejects_reference_outside_clip_bounds(tmp_path) -> None:
    video_root = tmp_path / "dataset"
    video_root.mkdir()
    source = video_root / "abc_60.0_210.0.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "qvh_000003_9x16",
                "split": "dev",
                "relative_video_path": "abc_60.0_210.0.mp4",
                "clip_start_sec": 10.0,
                "clip_end_sec": 15.0,
                "weak_reference_segments": [{"start_sec": 3.0, "end_sec": 6.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside clip-local bounds"):
        load_baseline_samples(manifest, video_root)


def test_jsonl_manifest_rejects_disagreeing_reference_clip_bounds(tmp_path) -> None:
    video_root = tmp_path / "dataset"
    (video_root / "videos").mkdir(parents=True)
    source = video_root / "videos" / "abc.mp4"
    source.write_bytes(b"video")
    references = video_root / "references" / "dev.jsonl"
    references.parent.mkdir(parents=True)
    references.write_text(
        json.dumps(
            {
                "video_id": "qvh_000004_9x16",
                "split": "dev",
                "clip_start_sec": 99.0,
                "clip_end_sec": 105.0,
                "coordinate_system": "clip-local seconds",
                "segments": [{"start_sec": 0.0, "end_sec": 2.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "qvh_000004_9x16",
                "split": "dev",
                "relative_video_path": "videos/abc.mp4",
                "clip_start_sec": 10.0,
                "clip_end_sec": 15.0,
                "reference_path": "references/dev.jsonl#qvh_000004_9x16",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree"):
        load_baseline_samples(manifest, video_root)


def test_resume_skips_complete_success_without_overwrite(tmp_path) -> None:
    result_path = tmp_path / "sample.json"
    result_path.write_text(json.dumps({"video_id": "sample", "success": True}), encoding="utf-8")

    assert resume_decision(result_path, raw_path=tmp_path / "raw.json", resume=True) == "skip"
    assert resume_decision(result_path, raw_path=tmp_path / "raw.json", resume=False) == "blocked"
    assert (
        resume_decision(
            result_path,
            raw_path=tmp_path / "raw.json",
            resume=False,
            overwrite=True,
        )
        == "run_model"
    )

    result_path.write_text(json.dumps({"video_id": "sample", "success": False}), encoding="utf-8")
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps({"chunks": []}), encoding="utf-8")
    assert resume_decision(result_path, raw_path=raw_path, resume=True) == "recover_raw"


def test_resume_preserves_original_run_config(tmp_path) -> None:
    path = tmp_path / "run_config.json"
    original = {"created_at": "first", "command": "model-run"}
    resumed = {"created_at": "later", "command": "model-run --resume"}

    assert persist_run_config(path, original, resume=False) == original
    assert persist_run_config(path, resumed, resume=True) == original
    assert json.loads(path.read_text(encoding="utf-8")) == original

    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        persist_run_config(path, resumed, resume=False)

    assert persist_run_config(path, resumed, resume=False, overwrite=True) == resumed


def test_single_result_aggregation_supports_percentiles() -> None:
    result = {
        "video_id": "sample",
        "success": True,
        "metrics": {
            "weak_ref_precision": 0.5,
            "weak_ref_recall": 0.75,
            "weak_ref_f1": 0.6,
            "temporal_iou": 0.4,
        },
        "timing": {"model_inference_sec": 12.5},
    }

    aggregate = aggregate_results([result])

    assert aggregate["processed"] == 1
    assert aggregate["success"] == 1
    assert aggregate["metrics"]["weak_ref_f1"] == {
        "mean": 0.6,
        "median": 0.6,
        "p50": 0.6,
        "p95": 0.6,
    }
    assert aggregate["metrics"]["inference_time_sec"]["p95"] == 12.5


def test_timeline_visualization_writes_png(tmp_path) -> None:
    output = tmp_path / "timeline.png"
    write_timeline_figure(
        video_id="sample",
        duration_sec=10.0,
        weak_reference_segments=[{"start_sec": 1.0, "end_sec": 3.0}],
        prediction_segments=[{"start_sec": 2.0, "end_sec": 5.0}],
        metrics={"weak_ref_precision": 1 / 3, "weak_ref_recall": 0.5, "weak_ref_f1": 0.4},
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_failure_isolation_continues_after_malformed_output() -> None:
    items = [{"video_id": "bad"}, {"video_id": "good"}]

    def process(item: dict) -> dict:
        if item["video_id"] == "bad":
            raise ValueError("malformed model output")
        return {"video_id": item["video_id"], "success": True}

    results = run_with_failure_isolation(items, process, stage="model_or_parser")

    assert results[0]["success"] is False
    assert results[0]["errors"][0]["exception_type"] == "ValueError"
    assert results[1] == {"video_id": "good", "success": True}
