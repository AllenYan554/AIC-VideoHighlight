from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.dataset_registry import DatasetRegistry
from aic_video_highlight.ftnet.datasets import (
    DatasetNotReadyError,
    DatasetPolicyError,
    Stage6BootstrapAdapter,
    TVSumExternalTestAdapter,
    YouTubeHighlightsAdapter,
)
from aic_video_highlight.ftnet.highlight_targets import (
    GroundTruthPolicyError,
    HighlightTargetAdapter,
    HighlightTargetStrategy,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "datasets"
    / "ftnet_datasets.yaml"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _youtube_fixture(
    root: Path,
    *,
    availability_status: str = "available",
    alignment_pass: bool = True,
    label_source: str = "mturk_label",
) -> Path:
    dataset = root / "youtube_highlights"
    _write_jsonl(
        dataset / "manifests" / "raw_manifest.jsonl",
        [{"sample_id": "dog-001", "availability_status": availability_status}],
    )
    _write_jsonl(
        dataset / "manifests" / "alignment_manifest.jsonl",
        [{"sample_id": "dog-001", "alignment_pass": alignment_pass}],
    )
    _write_jsonl(
        dataset / "manifests" / "split_manifest.jsonl",
        [{"sample_id": "dog-001", "split": "train"}],
    )
    _write_jsonl(
        dataset / "annotations" / "mturk_labels.jsonl",
        [
            {
                "sample_id": "dog-001",
                "source": label_source,
                "category": "dog",
                "vote_counts": [0, 3, 5],
                "annotator_count": 5,
            }
        ],
    )
    return dataset


def test_registry_uses_relative_paths_and_survives_root_move(tmp_path: Path) -> None:
    registry = DatasetRegistry.from_file(REGISTRY_PATH)
    simulated_d = tmp_path / "D-drive" / "datasets"
    simulated_e = tmp_path / "E-drive" / "datasets"

    before = registry.resolve("youtube_highlights", simulated_d)
    after = registry.resolve("youtube_highlights", simulated_e)

    assert registry.ids() == ("aic_highlight_dev", "tvsum", "youtube_highlights")
    assert before.relative_to(simulated_d) == after.relative_to(simulated_e)


def test_synthetic_youtube_manifest_loads_explicit_soft_vote_target(
    tmp_path: Path,
) -> None:
    dataset = _youtube_fixture(tmp_path)
    targets = HighlightTargetAdapter(HighlightTargetStrategy.SOFT_VOTE_TARGET)

    sample = YouTubeHighlightsAdapter(dataset).load_sample(
        "dog-001",
        split="train",
        target_adapter=targets,
        for_training=True,
    )

    assert sample.sample_id == "dog-001"
    assert sample.targets == (0.0, 0.6, 1.0)
    assert sample.target_strategy is HighlightTargetStrategy.SOFT_VOTE_TARGET


def test_youtube_missing_manifest_reports_dataset_not_ready(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotReadyError, match="raw_manifest"):
        YouTubeHighlightsAdapter(tmp_path).validate_ready()


def test_unavailable_youtube_sample_is_rejected(tmp_path: Path) -> None:
    dataset = _youtube_fixture(tmp_path, availability_status="unavailable")

    with pytest.raises(DatasetNotReadyError, match="not available"):
        YouTubeHighlightsAdapter(dataset).load_sample(
            "dog-001",
            split="train",
            target_adapter=HighlightTargetAdapter(
                HighlightTargetStrategy.SOFT_VOTE_TARGET
            ),
        )


def test_failed_alignment_is_rejected_for_training(tmp_path: Path) -> None:
    dataset = _youtube_fixture(tmp_path, alignment_pass=False)

    with pytest.raises(DatasetPolicyError, match="alignment_pass=false"):
        YouTubeHighlightsAdapter(dataset).load_sample(
            "dog-001",
            split="train",
            target_adapter=HighlightTargetAdapter(
                HighlightTargetStrategy.SOFT_VOTE_TARGET
            ),
            for_training=True,
        )


def test_match_label_cannot_become_ground_truth(tmp_path: Path) -> None:
    dataset = _youtube_fixture(tmp_path, label_source="match_label")

    with pytest.raises(GroundTruthPolicyError, match="match_label"):
        YouTubeHighlightsAdapter(dataset).load_sample(
            "dog-001",
            split="train",
            target_adapter=HighlightTargetAdapter(
                HighlightTargetStrategy.SOFT_VOTE_TARGET
            ),
        )


def test_paper_consensus_strategy_uses_category_specific_rule() -> None:
    adapter = HighlightTargetAdapter(
        HighlightTargetStrategy.PAPER_CONSENSUS_BINARY_TARGET
    )

    ordinary = adapter.adapt(
        {
            "source": "mturk_label",
            "category": "dog",
            "vote_counts": [2, 3],
            "annotator_count": 5,
        }
    )
    strict = adapter.adapt(
        {
            "source": "mturk_label",
            "category": "parkour",
            "vote_counts": [3, 4],
            "annotator_count": 5,
        }
    )

    assert ordinary == (0.0, 1.0)
    assert strict == (0.0, 1.0)


def test_tvsum_cannot_enter_training_even_when_files_exist(tmp_path: Path) -> None:
    adapter = TVSumExternalTestAdapter(tmp_path / "tvsum")

    with pytest.raises(DatasetPolicyError, match="external test"):
        adapter.validate_usage(training=True)


def test_stage6_bootstrap_missing_manifest_is_not_faked(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotReadyError, match="dataset_manifest"):
        Stage6BootstrapAdapter(tmp_path).load_manifest()
