from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.candidate_cache import (
    CandidateCacheError,
    build_video_cache_record,
    canonical_json_bytes,
    compare_identity,
    export_candidate_cache,
    merged_candidate_id,
    raw_candidate_id,
    replay_cache,
    semantic_sha256,
    validate_cache,
)


def _chunk(
    index: int,
    start: float,
    end: float,
    segments: list[dict],
) -> dict:
    response = {
        "has_highlight": bool(segments),
        "segments": segments,
    }
    return {
        "chunk_index": index,
        "chunk_start_sec": start,
        "chunk_end_sec": end,
        "raw_response": json.dumps(response, ensure_ascii=False),
        "finish_reason": "stop",
        "request_latency_sec": 1.0,
        "parse_success": True,
        "parse_error": None,
        "parsed_segments": [
            {**segment, "source_chunk": index} for segment in segments
        ],
    }


def _run_config(split: str) -> dict:
    return {
        "experiment_name": f"stage3_{split}_full_baseline_v1",
        "model_name": "Qwen/Qwen3.5-4B",
        "model_revision": "frozen-model-revision",
        "vllm_version": "0.28.0",
        "python_version": "3.12.14",
        "torch_version": "2.13.0",
        "prompt_version": "high_recall_retrieval_v0",
        "sampling_fps": 2.0,
        "chunk_seconds": 30,
        "chunk_overlap_seconds": 5,
        "merge_threshold": 0.5,
        "merge_strategy": "sorted adjacent temporal-IoU union; maximum score",
        "dataset_name": "aic_highlight_dev",
        "dataset_version": "aic_highlight_dev_v1.1",
        "split": split,
        "git_commit_head": "frozen-git-head",
        "request_parameters": {
            "max_new_tokens": 512,
            "temperature": 0,
            "enable_thinking": False,
            "timeout_sec": 120,
        },
    }


def _raw_payload(video_id: str = "video_001") -> dict:
    return {
        "video_id": video_id,
        "source_group": "source-window",
        "clip_start_sec": 0.0,
        "clip_end_sec": 35.0,
        "experiment_clip_duration_sec": 35.0,
        "coordinate_system": "clip-local seconds",
        "chunks": [
            _chunk(
                0,
                0.0,
                30.0,
                [
                    {
                        "start_sec": 20.0,
                        "end_sec": 30.0,
                        "score": 0.8,
                        "reason": "动作开始",
                    }
                ],
            ),
            _chunk(
                1,
                24.0,
                35.0,
                [
                    {
                        "start_sec": 0.0,
                        "end_sec": 8.0,
                        "score": 0.9,
                        "reason": "动作延续",
                    }
                ],
            ),
        ],
    }


def _prediction(video_id: str = "video_001") -> dict:
    return {
        "video_id": video_id,
        "raw_chunk_outputs": _raw_payload(video_id)["chunks"],
        "parsed_chunk_segments": [
            {
                "start_sec": 20.0,
                "end_sec": 30.0,
                "score": 0.8,
                "reason": "动作开始",
                "source_chunk": 0,
            },
            {
                "start_sec": 24.0,
                "end_sec": 32.0,
                "score": 0.9,
                "reason": "动作延续",
                "source_chunk": 1,
            },
        ],
        "merged_prediction_segments": [
            {
                "start_sec": 20.0,
                "end_sec": 32.0,
                "score": 0.9,
                "reason": "动作开始 | 动作延续",
                "source_chunk": None,
            }
        ],
        # Export must deliberately ignore these development-only fields.
        "weak_reference_segments": [{"start_sec": 21.0, "end_sec": 24.0}],
        "metrics": {"weak_ref_recall": 1.0, "temporal_iou": 0.25},
        "success": True,
    }


def _write_source(root: Path, split: str, video_ids: tuple[str, ...]) -> Path:
    root.mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "run_config.json").write_text(
        json.dumps(_run_config(split), ensure_ascii=False), encoding="utf-8"
    )
    predictions: list[str] = []
    for video_id in video_ids:
        raw = _raw_payload(video_id)
        (root / "raw" / f"{video_id}.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        predictions.append(json.dumps(_prediction(video_id), ensure_ascii=False))
    (root / "predictions.jsonl").write_text(
        "\n".join(predictions) + "\n", encoding="utf-8"
    )
    return root


def test_deterministic_candidate_ids_and_canonical_serialization() -> None:
    raw_fields = {
        "video_id": "视频_001",
        "chunk_index": 1,
        "candidate_index": 0,
        "chunk_start_sec": 25.0,
        "chunk_end_sec": 35.0,
        "local_start_sec": 0.0,
        "local_end_sec": 7.0,
        "start_sec": 25.0,
        "end_sec": 32.0,
        "score": 0.9,
        "reason": "动作延续",
        "source_chunk": 1,
    }
    reordered = dict(reversed(list(raw_fields.items())))

    assert raw_candidate_id(raw_fields) == raw_candidate_id(reordered)
    assert canonical_json_bytes(raw_fields) == canonical_json_bytes(reordered)
    assert canonical_json_bytes(raw_fields).endswith(b"\n")
    assert b"\\u52a8" not in canonical_json_bytes(raw_fields)
    assert semantic_sha256(raw_fields) == semantic_sha256(reordered)

    merged_fields = {
        "video_id": "视频_001",
        "start_sec": 20.0,
        "end_sec": 32.0,
        "score": 0.9,
        "reason": "动作开始 | 动作延续",
        "source_chunk": None,
        "contributor_raw_candidate_ids": ["raw-b", "raw-a"],
    }
    reordered_merged = {
        **merged_fields,
        "contributor_raw_candidate_ids": ["raw-a", "raw-b"],
    }
    assert merged_candidate_id(merged_fields) == merged_candidate_id(reordered_merged)


def test_build_record_recovers_exact_lineage_and_identity() -> None:
    record = build_video_cache_record(
        raw_payload=_raw_payload(),
        frozen_prediction=_prediction(),
        run_config=_run_config("dev"),
        split="dev",
        raw_journal_sha256="a" * 64,
        frozen_prediction_record_sha256="b" * 64,
        run_config_sha256="c" * 64,
    )

    assert len(record["raw_candidates"]) == 2
    assert [item["candidate_count"] for item in record["source_chunks"]] == [1, 1]
    assert len(record["merged_candidates"]) == 1
    contributors = record["merged_candidates"][0][
        "contributor_raw_candidate_ids"
    ]
    assert contributors == [
        record["raw_candidates"][0]["raw_candidate_id"],
        record["raw_candidates"][1]["raw_candidate_id"],
    ]
    assert record["merged_candidates"][0]["start_sec"] == 20.0
    assert record["merged_candidates"][0]["end_sec"] == 32.0
    assert record["semantic_sha256"] == semantic_sha256(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    )


def test_build_record_rejects_persisted_candidate_or_prediction_drift() -> None:
    prediction = _prediction()
    prediction["parsed_chunk_segments"][0]["end_sec"] = 29.0
    with pytest.raises(CandidateCacheError, match="parsed candidate identity"):
        build_video_cache_record(
            raw_payload=_raw_payload(),
            frozen_prediction=prediction,
            run_config=_run_config("dev"),
            split="dev",
            raw_journal_sha256="a" * 64,
            frozen_prediction_record_sha256="b" * 64,
            run_config_sha256="c" * 64,
        )

    prediction = _prediction()
    prediction["merged_prediction_segments"][0]["end_sec"] = 31.0
    with pytest.raises(CandidateCacheError, match="merged prediction identity"):
        build_video_cache_record(
            raw_payload=_raw_payload(),
            frozen_prediction=prediction,
            run_config=_run_config("dev"),
            split="dev",
            raw_journal_sha256="a" * 64,
            frozen_prediction_record_sha256="b" * 64,
            run_config_sha256="c" * 64,
        )


def test_export_validate_replay_compare_and_rebuild_are_deterministic(
    tmp_path: Path,
) -> None:
    dev = _write_source(tmp_path / "dev-source", "dev", ("video_002", "video_001"))
    hard = _write_source(tmp_path / "hard-source", "hard", ("video_003",))
    build_a = tmp_path / "build-a"
    build_b = tmp_path / "build-b"

    manifest_a = export_candidate_cache(
        {"hard": hard, "dev": dev}, output_dir=build_a
    )
    manifest_b = export_candidate_cache(
        {"dev": dev, "hard": hard}, output_dir=build_b
    )

    assert manifest_a["global_semantic_sha256"] == manifest_b["global_semantic_sha256"]
    assert manifest_a["complete_source_export"] is True
    assert (build_a / "cache_manifest.json").read_bytes() == (
        build_b / "cache_manifest.json"
    ).read_bytes()
    assert validate_cache(build_a)["record_count"] == 3

    replay_path = tmp_path / "replayed.jsonl"
    replay_summary = replay_cache(build_a, replay_path)
    assert replay_summary == {"record_count": 3, "segment_count": 3}
    replayed = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
    assert [item["video_id"] for item in replayed] == [
        "video_002",
        "video_001",
        "video_003",
    ]
    assert set(replayed[0]) == {"video_id", "split", "merged_prediction_segments"}

    comparison = compare_identity(build_a, {"dev": dev, "hard": hard})
    assert comparison["identity_match"] is True
    assert comparison["matched_records"] == 3
    assert comparison["mismatches"] == []


def test_validator_detects_duplicate_ids_invalid_lineage_intervals_and_leakage(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "dev-source", "dev", ("video_001",))
    cache = tmp_path / "cache"
    export_candidate_cache({"dev": source}, output_dir=cache)
    record_path = next((cache / "records").glob("*.json"))
    original = json.loads(record_path.read_text(encoding="utf-8"))

    mutations = []
    duplicate = deepcopy(original)
    duplicate["raw_candidates"].append(deepcopy(duplicate["raw_candidates"][0]))
    mutations.append((duplicate, "duplicate raw candidate ID"))

    bad_contributor = deepcopy(original)
    bad_contributor["merged_candidates"][0]["contributor_raw_candidate_ids"] = ["raw-v1-unknown"]
    mutations.append((bad_contributor, "unknown contributor"))

    bad_interval = deepcopy(original)
    bad_interval["raw_candidates"][0]["end_sec"] = bad_interval["raw_candidates"][0]["start_sec"]
    mutations.append((bad_interval, "invalid interval"))

    leaked = deepcopy(original)
    leaked["weak_reference_segments"] = [{"start_sec": 0.0, "end_sec": 1.0}]
    mutations.append((leaked, "forbidden development-label field"))

    for mutated, message in mutations:
        record_path.write_bytes(canonical_json_bytes(mutated))
        with pytest.raises(CandidateCacheError, match=message):
            validate_cache(cache)
        record_path.write_bytes(canonical_json_bytes(original))


def test_validator_detects_semantic_tampering(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "dev-source", "dev", ("video_001",))
    cache = tmp_path / "cache"
    export_candidate_cache({"dev": source}, output_dir=cache)
    record_path = next((cache / "records").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["merged_candidates"][0]["end_sec"] = 31.5
    record_path.write_bytes(canonical_json_bytes(record))

    with pytest.raises(CandidateCacheError, match="semantic hash|candidate ID"):
        validate_cache(cache)


def test_validator_rejects_noncanonical_record_serialization(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "dev-source", "dev", ("video_001",))
    cache = tmp_path / "cache"
    export_candidate_cache({"dev": source}, output_dir=cache)
    record_path = next((cache / "records").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(CandidateCacheError, match="not canonical JSON"):
        validate_cache(cache)


def test_identity_compare_detects_source_artifact_tampering(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "dev-source", "dev", ("video_001",))
    cache = tmp_path / "cache"
    export_candidate_cache({"dev": source}, output_dir=cache)
    predictions_path = source / "predictions.jsonl"
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    comparison = compare_identity(cache, {"dev": source})

    assert comparison["identity_match"] is False
    assert {item["kind"] for item in comparison["mismatches"]} == {
        "predictions_source_hash"
    }


def test_export_rejects_heldout_and_non_v11_sources(tmp_path: Path) -> None:
    heldout = _write_source(tmp_path / "heldout-source", "heldout", ("video_001",))
    with pytest.raises(CandidateCacheError, match="Heldout|dev and hard"):
        export_candidate_cache({"heldout": heldout}, output_dir=tmp_path / "cache")

    dev = _write_source(tmp_path / "dev-source", "dev", ("video_002",))
    config_path = dev / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_version"] = "aic_highlight_dev_v1"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CandidateCacheError, match="aic_highlight_dev_v1.1"):
        export_candidate_cache({"dev": dev}, output_dir=tmp_path / "cache-v1")


def test_export_refuses_to_overwrite_existing_cache(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "dev-source", "dev", ("video_001",))
    cache = tmp_path / "cache"
    export_candidate_cache({"dev": source}, output_dir=cache)

    with pytest.raises(FileExistsError, match="immutable"):
        export_candidate_cache({"dev": source}, output_dir=cache)
