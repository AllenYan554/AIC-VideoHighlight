from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic_video_highlight.ftnet import evaluation as ev
from aic_video_highlight.ftnet.checkpoint import TrainingIdentity, save_checkpoint
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig
from aic_video_highlight.ftnet.native_schema import NATIVE_DIM, NATIVE_FIELDS, NormalizationStats
from aic_video_highlight.ftnet.sampling import SAMPLE_PERIOD_SEC


def _frames(
    *,
    video_id: str = "vid",
    split: str = "VALIDATION",
    target: list[float],
    loss_mask: list[bool],
    adjacency: list[bool] | None = None,
    timestamps: list[float] | None = None,
    source_ids: list[int] | None = None,
) -> ev.VideoFrames:
    length = len(target)
    return ev.VideoFrames(
        video_id=video_id,
        split=split,
        category="dog",
        source_frame_id=np.asarray(source_ids if source_ids is not None else list(range(length)), dtype=np.int64),
        timestamp=np.asarray(timestamps if timestamps is not None else [index * 0.5 for index in range(length)], dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        loss_mask=np.asarray(loss_mask, dtype=bool),
        adjacency_mask=np.asarray(
            adjacency if adjacency is not None else [False] + [True] * (length - 1), dtype=bool
        ),
        native_missing=np.zeros((length, NATIVE_DIM), dtype=bool),
    )


def _scored(frames: ev.VideoFrames, probabilities: list[float]) -> ev.ScoredVideo:
    return ev.ScoredVideo(frames=frames, p_keep=np.asarray(probabilities, dtype=np.float64))


# ---------------------------------------------------------------------------
# 1. confusion matrix / metric formulas
# ---------------------------------------------------------------------------


def test_confusion_and_metric_formulas() -> None:
    y = np.array([True, True, False, False])
    keep = np.array([True, False, True, False])
    counts = ev.confusion_counts(y, keep)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    metrics = ev.metrics_from_counts(counts)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_deletion_rate"] == pytest.approx(0.5)
    empty = ev.metrics_from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] is None


def test_false_deletion_equals_one_minus_recall() -> None:
    y = np.array([True, True, True, False])
    keep = np.array([True, False, False, True])
    metrics = ev.metrics_from_counts(ev.confusion_counts(y, keep))
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["false_deletion_rate"] == pytest.approx(1 - metrics["recall"])


# ---------------------------------------------------------------------------
# 2. PR-AUC
# ---------------------------------------------------------------------------


def test_average_precision_hand_computed() -> None:
    y = np.array([True, False, True])
    p = np.array([0.9, 0.8, 0.7])
    # precision@1 = 1.0 (dR=0.5), precision@2 = 2/3 (dR=0.5) -> 0.8333
    assert ev.average_precision(y, p) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_ties_are_grouped() -> None:
    y = np.array([True, False])
    p = np.array([0.5, 0.5])
    assert ev.average_precision(y, p) == pytest.approx(0.5)


def test_average_precision_no_positives() -> None:
    assert ev.average_precision(np.array([False, False]), np.array([0.2, 0.9])) is None


def test_pr_curve_is_monotone_in_recall() -> None:
    y = np.array([True, True, False, False, True])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    points = ev.pr_curve(y, p)
    recalls = [point["recall"] for point in points]
    assert recalls == sorted(recalls)


def test_concat_universe_restricts_y_and_scores_to_supervised_frames() -> None:
    frames = _frames(target=[1.0, 1.0, 0.0, 0.0], loss_mask=[True, False, True, False])
    scored = _scored(frames, [0.9, 0.8, 0.1, 0.7])
    y, p = ev.concat_universe([scored])
    assert y.tolist() == [True, False]
    assert p.tolist() == [0.9, 0.1]


# ---------------------------------------------------------------------------
# 4. threshold selection
# ---------------------------------------------------------------------------


def test_f1_threshold_unique_optimum() -> None:
    y = np.array([True, True, False, False])
    p = np.array([0.9, 0.6, 0.5, 0.4])
    best = ev.select_tau_f1(y, p)
    assert best["tau"] == pytest.approx(0.6)
    assert best["f1"] == pytest.approx(1.0)
    assert best["recall"] == pytest.approx(1.0)


def test_f1_threshold_tie_break_prefers_higher_recall_then_lower_tau() -> None:
    # F1 == 2/3 is attained by several thresholds; the pre-registered tie-break
    # must keep the highest-recall one (all frames kept) with the lowest tau.
    y = np.array([True, False, True, False])
    p = np.array([0.9, 0.7, 0.5, 0.6])
    best = ev.select_tau_f1(y, p)
    assert best["f1"] == pytest.approx(2 / 3)
    assert best["recall"] == pytest.approx(1.0)
    assert best["tau"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. FS0 baseline / 6. pruning rate / 7. false deletion
# ---------------------------------------------------------------------------


def test_fs0_keeps_everything_and_has_zero_pruning() -> None:
    frames = _frames(target=[1.0, 0.0, 0.6], loss_mask=[True, True, True])
    scored = _scored(frames, [0.1, 0.2, 0.3])
    metrics = ev.operating_metrics([scored], tau=0.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["pruning_rate"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["false_deletion_rate"] == pytest.approx(0.0)


def test_pruning_rate_and_false_deletion_rate_are_machine_computed() -> None:
    frames = _frames(target=[1.0, 1.0, 0.0, 0.0], loss_mask=[True] * 4)
    scored = _scored(frames, [0.9, 0.2, 0.1, 0.05])
    metrics = ev.operating_metrics([scored], tau=0.5)
    assert metrics["tp"] == 1 and metrics["fn"] == 1
    assert metrics["pruning_rate"] == pytest.approx(3 / 4)
    assert metrics["false_deletion_rate"] == pytest.approx(0.5)


def test_universe_excludes_unsupervised_frames() -> None:
    frames = _frames(target=[1.0, 0.0, 1.0], loss_mask=[True, False, True])
    scored = _scored(frames, [0.9, 0.9, 0.1])
    metrics = ev.operating_metrics([scored], tau=0.5)
    assert metrics["universe_frames"] == 2
    assert metrics["tp"] == 1 and metrics["fn"] == 1
    assert metrics["pruning_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 8-10. event proxy: survival / whole-run deletion / short-event filtering
# ---------------------------------------------------------------------------


def test_positive_runs_extract_and_break_on_non_adjacency() -> None:
    frames = _frames(
        target=[0.0] * 6,
        loss_mask=[True] * 6,
        adjacency=[False, True, True, False, True, True],
        timestamps=[0.0, 0.5, 1.0, 4.0, 4.5, 5.0],
    )
    coverage = np.array([False, True, True, True, True, True])
    runs = ev.extract_positive_runs(frames, coverage)
    assert len(runs) == 2
    assert runs[0].frame_indices == (1, 2)
    assert runs[1].frame_indices == (3, 4, 5)
    assert runs[0].duration_sec == pytest.approx(0.5 + SAMPLE_PERIOD_SEC)


def test_run_survival_and_whole_deletion() -> None:
    frames = _frames(target=[1.0] * 6, loss_mask=[True] * 6)
    coverage = np.array([True, True, False, True, True, False])
    runs = ev.extract_positive_runs(frames, coverage)
    assert len(runs) == 2
    scored = _scored(frames, [0.9, 0.9, 0.5, 0.1, 0.1, 0.5])
    metrics = ev.run_metrics(runs, {frames.video_id: scored}, tau=0.5)
    assert metrics["overall"]["runs"] == 2
    assert metrics["overall"]["survival_recall"] == pytest.approx(0.5)
    assert metrics["overall"]["whole_deletion_rate"] == pytest.approx(0.5)
    assert metrics["overall"]["retention_min"] == pytest.approx(0.0)
    assert metrics["status"] == "DERIVED_POSITIVE_RUN_PROXY"
    assert metrics["not_human_event_gt"] is True


def test_short_run_stratum_filtering() -> None:
    short = ev.PositiveRun("v", "VALIDATION", (0, 1), 0.0, 0.5, 1.0)
    medium = ev.PositiveRun("v", "VALIDATION", (2, 3, 4, 5, 6, 7), 2.0, 5.0, 3.5)
    long = ev.PositiveRun("v", "VALIDATION", tuple(range(12)), 8.0, 20.0, 12.5)
    assert short.stratum() == "short"
    assert medium.stratum() == "medium"
    assert long.stratum() == "long"


def test_run_metrics_requires_scored_video() -> None:
    run = ev.PositiveRun("missing", "VALIDATION", (0, 1), 0.0, 0.5, 1.0)
    with pytest.raises(ev.EvaluationError):
        ev.run_metrics([run], {}, tau=0.5)


def test_run_safe_requires_no_whole_run_deletion() -> None:
    frames = _frames(target=[1.0] * 4, loss_mask=[True] * 4)
    coverage = np.array([True, True, False, False])
    runs = ev.extract_positive_runs(frames, coverage)
    # a high threshold would delete the whole run -> must be rejected at recall>=0.95
    scored = _scored(frames, [0.2, 0.2, 0.9, 0.9])
    y = np.array([True, True, True, True])
    p = scored.p_keep
    result = ev.select_tau_run_safe(y, p, runs, {frames.video_id: scored}, min_recall=0.95)
    assert result is not None
    assert result["tau"] <= 0.2
    assert result["pruning_rate"] == pytest.approx(0.0)


def test_select_tau_safe_keeps_all_when_positives_have_min_scores() -> None:
    y = np.array([True, True, True])
    p = np.array([0.1, 0.2, 0.3])
    # tau=0.1 and tau=0 keep every frame; the pre-registered tie-break prefers
    # the higher threshold (same pruning, same recall)
    result = ev.select_tau_safe(y, p, min_recall=0.95)
    assert result is not None
    assert result["tau"] == pytest.approx(0.1)
    assert result["recall"] == pytest.approx(1.0)
    assert result["pruning_rate"] == pytest.approx(0.0)


def test_select_tau_safe_none_without_positives() -> None:
    assert ev.select_tau_safe(np.zeros(4, dtype=bool), np.array([0.1, 0.2, 0.3, 0.4])) is None


def test_audit_event_gt_reports_not_available(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations" / "dog" / "vid"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "mturk_label.json").write_text("[[[0, 4]], [5]]", encoding="utf-8")
    (annotation_dir / "clip.json").write_text("[[0, 4]]", encoding="utf-8")
    frames = _frames(target=[1.0, 0.0, 0.0, 0.0], loss_mask=[True] * 4)
    entry = type("E", (), {"annotation_identity": "dog/vid"})()
    report = ev.audit_event_gt(
        {"VALIDATION": [frames]},
        annotations_root=tmp_path / "annotations",
        entries_by_id={"vid": entry},
    )
    assert report["EVENT_METRICS_AVAILABLE"] == "NO"
    assert "not identify contiguous human highlight events" in report["reason"]
    assert report["run_counts"]["VALIDATION"]["total"] == 1


# ---------------------------------------------------------------------------
# 11-13. split isolation / no TRAIN tuning / deterministic output
# ---------------------------------------------------------------------------


def test_protocol_declares_split_isolation_and_constraints(tmp_path: Path) -> None:
    for name in ("stats.json", "index.json", "manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    protocol = ev.build_protocol(
        checkpoint={"sha256": "a" * 64},
        control_checkpoint={"sha256": "b" * 64},
        data_root="data",
        normalization_path=tmp_path / "stats.json",
        index_path=tmp_path / "index.json",
        split_manifest_path=tmp_path / "manifest.json",
        annotations_root="annotations",
        seed=20260917,
        device="cpu",
        code_git_head="0" * 40,
    )
    assert "never read" in protocol["split_responsibilities"]["TRAIN"]
    assert protocol["threshold_selection"]["cannot_relax_after_results"] is True
    assert protocol["uncertainty"]["bootstrap"] is False
    assert protocol["constraints"]["official_test"] == "NOT ACCESSED"
    assert protocol["constraints"]["tvsum"] == "NOT ACCESSED"
    assert protocol["universe"]["definition"].startswith("loss_mask == 1")


def test_threshold_selection_only_touches_provided_arrays() -> None:
    y_cal = np.array([True, False, True, False])
    p_cal = np.array([0.9, 0.8, 0.2, 0.1])
    y_val = np.array([True, True, False, False])
    p_val = np.array([0.9, 0.1, 0.9, 0.1])
    tau = ev.select_tau_f1(y_cal, p_cal)
    # the returned threshold is derived from calibration arrays only
    assert tau["tau"] in set(p_cal) | {0.0}
    assert ev.select_tau_f1(y_val, p_val)["tau"] != tau["tau"]


def test_metrics_are_deterministic() -> None:
    y = np.array([True, False, True, True, False])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    first = ev.select_tau_f1(y, p)
    second = ev.select_tau_f1(y, p)
    assert first == second
    assert ev.average_precision(y, p) == ev.average_precision(y, p)


# ---------------------------------------------------------------------------
# 14. binary reference semantics
# ---------------------------------------------------------------------------


def test_binary_reference_semantics_and_universe() -> None:
    target = np.array([0.5, 0.49, 0.0, 0.2])
    loss_mask = np.array([True, True, True, False])
    reference = ev.binary_reference(target, loss_mask, 0.5)
    assert reference.tolist() == [True, False, False, False]
    any_positive = ev.binary_reference(target, loss_mask, 0.0)
    assert any_positive.tolist() == [True, True, False, False]


def test_native_missingness_audit_flags_association() -> None:
    length = 4
    frames = _frames(target=[1.0, 1.0, 0.0, 0.0], loss_mask=[True] * length)
    missing = np.zeros((length, NATIVE_DIM), dtype=bool)
    missing[0, 5] = True  # only a positive frame misses field 5
    frames = ev.VideoFrames(
        video_id=frames.video_id,
        split=frames.split,
        category=frames.category,
        source_frame_id=frames.source_frame_id,
        timestamp=frames.timestamp,
        target=frames.target,
        loss_mask=frames.loss_mask,
        adjacency_mask=frames.adjacency_mask,
        native_missing=missing,
    )
    audit = ev.native_missingness_audit([_scored(frames, [0.9, 0.9, 0.1, 0.1])])
    row = audit["rows"][NATIVE_FIELDS.index("subject_track_present")]
    # observed rate Y=1 is 1/2, Y=0 is 2/2 -> delta = -0.5
    assert row["delta"] == pytest.approx(-0.5)
    assert row["risk_flag"] == "POTENTIAL_SHORTCUT_RISK"


def test_native_missingness_audit_ignores_unlabelled_frames() -> None:
    length = 4
    frames = _frames(target=[1.0, 1.0, 0.0, 0.0], loss_mask=[True, False, True, False])
    missing = np.zeros((length, NATIVE_DIM), dtype=bool)
    missing[1, 5] = True  # only an UNLABELLED frame misses field 5 -> must be ignored
    frames = ev.VideoFrames(
        video_id=frames.video_id,
        split=frames.split,
        category=frames.category,
        source_frame_id=frames.source_frame_id,
        timestamp=frames.timestamp,
        target=frames.target,
        loss_mask=frames.loss_mask,
        adjacency_mask=frames.adjacency_mask,
        native_missing=missing,
    )
    audit = ev.native_missingness_audit([_scored(frames, [0.9, 0.9, 0.1, 0.1])])
    assert audit["universe_frames"] == 2
    row = audit["rows"][5]
    assert row["observed_rate_overall"] == pytest.approx(1.0)
    assert row["delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# checkpoint loading + deterministic scoring (CPU, synthetic safetensors)
# ---------------------------------------------------------------------------


def test_checkpoint_loading_and_scoring_is_deterministic(tmp_path: Path) -> None:
    import torch
    from safetensors.numpy import save_file

    torch.manual_seed(11)
    split_dir = tmp_path / "data" / "validation"
    split_dir.mkdir(parents=True)
    length = 8
    save_file(
        {
            "visual": np.random.default_rng(3).standard_normal((length, 256)).astype(np.float16),
            "native": np.zeros((length, NATIVE_DIM), dtype=np.float32),
            "native_missing": np.zeros((length, NATIVE_DIM), dtype=np.uint8),
            "target": np.linspace(0.0, 1.0, length).astype(np.float32),
            "loss_mask": np.ones(length, dtype=np.uint8),
            "adjacency_mask": np.asarray([0] + [1] * (length - 1), dtype=np.uint8),
            "timestamp": np.asarray([0.5 * index for index in range(length)], dtype=np.float64),
            "source_frame_id": np.arange(length, dtype=np.int64),
        },
        str(split_dir / "vid-a.safetensors"),
    )
    config = FTNetConfig(visual_dim=256, branch_dim=16, native_dim=16, native_branch_dim=8, temporal_channels=16)
    model = FTNet(config)
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=7,
        global_step=196,
        identity=TrainingIdentity(
            git_head="0" * 40,
            split_manifest_sha256="UNSET",
            feature_manifest_sha256="UNSET",
            label_adapter="soft_vote_target",
            seed=20260917,
        ),
        config=model.config,
    )
    loaded, info = ev.load_checkpoint_model(checkpoint_path, "cpu")
    assert info["epoch"] == 7 and info["global_step"] == 196
    stats = NormalizationStats(
        mean=tuple([0.0] * NATIVE_DIM),
        std=tuple([1.0] * NATIVE_DIM),
        log1p_fields=(),
    )
    first = ev.score_split(loaded, tmp_path / "data", "VALIDATION", stats, device="cpu")
    second = ev.score_split(loaded, tmp_path / "data", "VALIDATION", stats, device="cpu")
    assert first[0].p_keep.shape == (length,)
    assert np.all((first[0].p_keep >= 0.0) & (first[0].p_keep <= 1.0))
    assert np.array_equal(first[0].p_keep, second[0].p_keep)
    # the checkpoint file must not have been modified by evaluation
    assert ev.sha256_file(checkpoint_path) == info["sha256"]


def test_json_output_is_sorted_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    ev.write_json(path, {"b": 1, "a": 2})
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": 2')
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"a": 2, "b": 1}
