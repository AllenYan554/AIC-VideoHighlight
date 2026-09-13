"""Official-Test provenance and hidden-GT contracts (synthetic, CPU-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.retrieval.candidate_cache import (  # noqa: E402
    CandidateCacheError,
    export_candidate_cache,
    validate_cache,
)
from aic_video_highlight.composition.fresh_pipeline import (  # noqa: E402
    validate_fresh_candidate_cache_binding,
)
from aic_video_highlight.runtime.orchestrator import (  # noqa: E402
    build_run_identity,
    candidate_cache_split_for_input_mode,
)
from aic_video_highlight.runtime.profiles import load_runtime_profile  # noqa: E402
from aic_video_highlight.retrieval.runner import evaluate_reference_policy  # noqa: E402


def _write_retrieval(root: Path, *, split: str = "test") -> None:
    root.mkdir(parents=True)
    (root / "raw").mkdir()
    segment = {
        "start_sec": 0.0,
        "end_sec": 1.0,
        "score": 0.9,
        "reason": "synthetic highlight",
        "source_chunk": 0,
    }
    raw_chunk = {
        "chunk_index": 0,
        "chunk_start_sec": 0.0,
        "chunk_end_sec": 2.0,
        "raw_response": json.dumps(
            {
                "has_highlight": True,
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "score": 0.9,
                        "reason": "synthetic highlight",
                    }
                ],
            }
        ),
        "finish_reason": "stop",
        "request_latency_sec": 0.1,
        "parse_success": True,
        "parse_error": None,
        "parsed_segments": [segment],
    }
    run_config = {
        "experiment_name": "retrieval",
        "dataset_name": "aic_official_test" if split == "test" else "aic_highlight_dev",
        "dataset_version": "v1" if split == "test" else "aic_highlight_dev_v1.1",
        "split": split,
        "model_name": "Qwen/Qwen3.5-4B",
        "model_revision": "revision",
        "prompt_version": "high_recall_retrieval_v0",
        "sampling_fps": 2.0,
        "chunk_seconds": 30,
        "chunk_overlap_seconds": 5,
        "merge_threshold": 0.5,
        "merge_strategy": "sorted adjacent temporal-IoU union; maximum score",
        "git_commit_head": "a" * 40,
        "python_version": "3.12",
        "torch_version": "test",
        "vllm_version": "unavailable",
        "request_parameters": {"max_new_tokens": 512, "temperature": 0},
    }
    prediction = {
        "video_id": "1",
        "success": True,
        "parsed_chunk_segments": [segment],
        "merged_prediction_segments": [segment],
    }
    raw = {
        "video_id": "1",
        "experiment_clip_duration_sec": 2.0,
        "chunks": [raw_chunk],
    }
    (root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (root / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    (root / "raw" / "1.json").write_text(json.dumps(raw), encoding="utf-8")


def test_candidate_cache_accepts_explicit_official_test_split(tmp_path):
    retrieval = tmp_path / "retrieval"
    cache = tmp_path / "cache"
    _write_retrieval(retrieval)

    manifest = export_candidate_cache({"test": retrieval}, output_dir=cache)
    summary = validate_cache(cache)

    assert manifest["dataset_version"] == "v1"
    assert manifest["source_sets"][0]["split"] == "test"
    assert manifest["records"][0]["split"] == "test"
    assert summary["split_counts"] == {"dev": 0, "hard": 0, "test": 1}


def test_candidate_cache_still_rejects_unknown_split(tmp_path):
    retrieval = tmp_path / "retrieval"
    _write_retrieval(retrieval)

    with pytest.raises(CandidateCacheError, match="dev, hard, and test"):
        export_candidate_cache({"heldout": retrieval}, output_dir=tmp_path / "cache")


def test_candidate_cache_dev_contract_remains_valid(tmp_path):
    retrieval = tmp_path / "retrieval"
    cache = tmp_path / "cache"
    _write_retrieval(retrieval, split="dev")

    manifest = export_candidate_cache({"dev": retrieval}, output_dir=cache)
    summary = validate_cache(cache)

    assert manifest["dataset_version"] == "aic_highlight_dev_v1.1"
    assert summary["split_counts"] == {"dev": 1, "hard": 0, "test": 0}


def test_dev_and_test_use_identical_candidate_construction(tmp_path):
    candidate_payloads = {}
    for split in ("dev", "test"):
        retrieval = tmp_path / f"retrieval-{split}"
        cache = tmp_path / f"cache-{split}"
        _write_retrieval(retrieval, split=split)
        export_candidate_cache({split: retrieval}, output_dir=cache)
        record = json.loads((cache / "records" / "1.json").read_text(encoding="utf-8"))
        candidate_payloads[split] = {
            key: record[key]
            for key in ("source_chunks", "raw_candidates", "merged_candidates")
        }

    assert candidate_payloads["dev"] == candidate_payloads["test"]


def test_numbered_video_route_binds_candidate_cache_as_test(tmp_path):
    retrieval = tmp_path / "retrieval"
    cache = tmp_path / "cache"
    _write_retrieval(retrieval)
    manifest = export_candidate_cache({"test": retrieval}, output_dir=cache)

    binding = validate_fresh_candidate_cache_binding(
        manifest,
        retrieval_predictions_path=retrieval / "predictions.jsonl",
        expected_video_ids=["1"],
        expected_split=candidate_cache_split_for_input_mode("numbered_videos"),
        forbidden_global_sha256="f" * 64,
    )

    assert binding["source_split"] == "test"


def test_official_test_reference_policy_never_accesses_hidden_gt():
    class HiddenGT(dict):
        def __getitem__(self, key):
            if key == "weak_reference_segments":
                raise AssertionError("hidden GT was accessed")
            return super().__getitem__(key)

    policy = evaluate_reference_policy(
        HiddenGT(split="test"),
        [{"start_sec": 0.0, "end_sec": 1.0}],
        evaluator=lambda *_: (_ for _ in ()).throw(AssertionError("evaluator called")),
    )

    assert policy == {
        "status": "UNAVAILABLE",
        "reason": "HIDDEN_GT",
        "metrics": {},
    }


def test_isolated_official_test_identity_records_scope_and_video_ids(tmp_path, monkeypatch):
    paths = {}
    for name in ("config", "protocol", "manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        paths[name] = path
    runtime_path = REPO / "configs" / "runtime" / "local_efficient_sdpa.json"
    monkeypatch.setattr(
        "aic_video_highlight.runtime.orchestrator._git_head", lambda: "a" * 40
    )
    monkeypatch.setattr(
        "aic_video_highlight.runtime.orchestrator.runtime_machine_identity",
        lambda: {"torch_version": "test", "cuda_version": "test", "gpu_name": "test"},
    )

    identity = build_run_identity(
        config_path=paths["config"],
        protocol_path=paths["protocol"],
        selected_manifest=paths["manifest"],
        profile="official_test",
        runtime_profile=load_runtime_profile(runtime_path),
        runtime_profile_path=runtime_path,
        scope="isolated_validation",
        video_ids=["1"],
    )

    assert identity["scope"] == "isolated_validation"
    assert identity["video_ids"] == ["1"]
