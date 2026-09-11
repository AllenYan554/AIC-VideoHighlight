"""Stage 5.5 frame-level prediction calibration tests (no real experiment)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.spatial_composition.composition_pipeline import FrozenInputError
from aic_video_highlight.spatial_composition.frame_calibration_metrics import (
    adjudicate_stage5_5,
    dev_promotion_objective,
    evaluate_video_set,
    hard_gates,
    macro_metrics,
    recall_guardrails,
    select_dev_winner,
    weak_frame_video_metrics,
)
from aic_video_highlight.spatial_composition.frame_projection import (
    VideoTiming,
    frames_in_segment,
)
from aic_video_highlight.spatial_composition.frame_selection import (
    ARM_NAMES,
    FS0,
    FS1,
    FS2,
    ChunkWindow,
    FinalSegment,
    FrameSupport,
    RawCandidateSpan,
    compute_support_by_frame,
    select_emit_frames,
    select_segment_frames,
)
from scripts.experiments.stage5 import (  # noqa: F401
    run_stage5_5_frame_calibration as runner,
)

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "configs" / "experiments" / "stage5"


def _timing(fps=1.0, frame_count=100, mode="CFR_FPS", pts=None) -> VideoTiming:
    return VideoTiming(
        video_id="v",
        fps=fps,
        frame_count=frame_count,
        timestamp_mode=mode,
        pts_timestamps=pts,
    )


def _support(pairs) -> dict[int, FrameSupport]:
    return {
        frame: FrameSupport(frame=frame, eligible_chunks=e, supporting_chunks=s)
        for frame, (e, s) in pairs.items()
    }


# ---------------------------------------------------------------------------
# E_t / S_t cross-chunk evidence
# ---------------------------------------------------------------------------

def test_support_uses_stage5_1_mapping_exactly():
    timing = _timing(fps=1.0, frame_count=50)
    chunks = (ChunkWindow(0, 0.0, 10.0), ChunkWindow(1, 5.0, 15.0))
    spans = (
        RawCandidateSpan(0, 0.0, 3.0),
        RawCandidateSpan(1, 7.0, 12.0),
    )
    profile = compute_support_by_frame(range(0, 16), timing, chunks, spans)
    # Direct Stage 5.1 membership must reproduce eligibility and support.
    for frame in range(0, 16):
        eligible = [
            c.chunk_index
            for c in chunks
            if frame in frames_in_segment(c.start_sec, c.end_sec, timing)
        ]
        assert profile[frame].eligible_chunks == len(eligible)
        expected_support = 0
        for span in spans:
            if span.chunk_index not in eligible:
                continue
            if frame in frames_in_segment(span.start_sec, span.end_sec, timing):
                expected_support += 1
        assert profile[frame].supporting_chunks == expected_support
    assert profile[7].eligible_chunks == 2
    assert profile[7].supporting_chunks == 1
    assert profile[0].eligible_chunks == 1


def test_eligible_chunk_counting_overlap_region():
    timing = _timing(fps=2.0, frame_count=100)
    chunks = (ChunkWindow(0, 0.0, 10.0), ChunkWindow(1, 8.0, 18.0))
    profile = compute_support_by_frame(range(0, 40), timing, chunks, ())
    # fps=2 -> overlap [8,10) is frames 16..19.
    assert profile[15].eligible_chunks == 1
    assert all(profile[f].eligible_chunks == 2 for f in range(16, 20))
    assert profile[20].eligible_chunks == 1
    assert profile[16].supporting_chunks == 0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def test_fs0_identity_keeps_every_frame():
    frames = tuple(range(12))
    support = _support({f: (3, 1) for f in frames})
    selection = select_segment_frames(FS0, frames, support)
    assert selection.kept == frames
    assert selection.dropped == ()
    assert selection.drop_rule == "fs0.identity_keep"


def test_fs1_left_edge_prune_only():
    frames = tuple(range(8))
    support = _support({0: (2, 1), 1: (2, 1), 2: (2, 2), 3: (2, 2),
                        4: (2, 2), 5: (2, 2), 6: (2, 2), 7: (2, 2)})
    selection = select_segment_frames(FS1, frames, support)
    assert selection.dropped == (0, 1)
    assert selection.kept == (2, 3, 4, 5, 6, 7)


def test_fs1_right_edge_prune_only():
    frames = tuple(range(8))
    support = _support({f: (2, 2) for f in range(6)} | {6: (2, 1), 7: (2, 1)})
    selection = select_segment_frames(FS1, frames, support)
    assert selection.dropped == (6, 7)
    assert selection.kept == (0, 1, 2, 3, 4, 5)


def test_fs1_never_deletes_internal_hole():
    frames = tuple(range(8))
    support = _support({0: (2, 2), 1: (2, 2), 2: (2, 1), 3: (2, 1),
                        4: (2, 2), 5: (2, 2), 6: (2, 2), 7: (2, 2)})
    selection = select_segment_frames(FS1, frames, support)
    assert selection.dropped == ()
    assert selection.kept == frames
    assert selection.drop_rule == "keep_all.no_edge_run"


def test_fs1_all_singleton_segment_keeps_all():
    frames = tuple(range(6))
    support = _support({f: (2, 1) for f in frames})
    selection = select_segment_frames(FS1, frames, support)
    assert selection.core_exists is False
    assert selection.dropped == ()
    assert selection.kept == frames
    assert selection.drop_rule == "keep_all.no_consensus_core"


def test_fs2_prunes_nonunanimous_edge_runs():
    frames = tuple(range(8))
    support = _support({0: (3, 1), 1: (3, 1), 2: (3, 3), 3: (3, 3),
                        4: (3, 3), 5: (3, 3), 6: (3, 2), 7: (3, 2)})
    selection = select_segment_frames(FS2, frames, support)
    assert selection.dropped == (0, 1, 6, 7)
    assert selection.kept == (2, 3, 4, 5)


def test_fs2_no_internal_hole_deletion():
    frames = tuple(range(6))
    support = _support({0: (3, 3), 1: (3, 3), 2: (3, 1), 3: (3, 1),
                        4: (3, 3), 5: (3, 3)})
    selection = select_segment_frames(FS2, frames, support)
    assert selection.dropped == ()


def test_fs2_without_unanimity_core_keeps_all():
    frames = tuple(range(6))
    support = _support({0: (3, 1), 1: (3, 2), 2: (3, 1), 3: (3, 2),
                        4: (3, 1), 5: (3, 2)})
    selection = select_segment_frames(FS2, frames, support)
    assert selection.core_exists is False
    assert selection.dropped == ()


# ---------------------------------------------------------------------------
# Whole-video projection: no new/duplicate/empty frames
# ---------------------------------------------------------------------------

def test_select_emit_frames_reuses_stage5_1_projection_and_is_safe():
    timing = _timing(fps=1.0, frame_count=40)
    segments = (FinalSegment("s0", 0.0, 12.0),)
    chunks = (ChunkWindow(0, 0.0, 12.0), ChunkWindow(1, 0.0, 12.0))
    spans = (
        RawCandidateSpan(0, 0.0, 2.0),
        RawCandidateSpan(0, 5.0, 7.0),
        RawCandidateSpan(1, 2.0, 10.0),
    )
    fs0 = select_emit_frames(FS0, segments, timing, chunks, spans)
    direct = tuple(frames_in_segment(0.0, 12.0, timing))
    assert fs0.candidate_frames == direct
    assert fs0.emitted_frames == direct

    fs1 = select_emit_frames(FS1, segments, timing, chunks, spans)
    assert set(fs1.emitted_frames) <= set(fs1.candidate_frames)
    assert len(set(fs1.emitted_frames)) == len(fs1.emitted_frames)
    assert fs1.emitted_frames  # never emptied
    assert fs0.emitted_frames == select_emit_frames(
        FS0, segments, timing, chunks, spans
    ).emitted_frames


def test_select_emit_frames_no_duplicate_and_deterministic_for_all_policies():
    timing = _timing(fps=2.0, frame_count=200)
    segments = (
        FinalSegment("s0", 0.0, 20.0),
        FinalSegment("s1", 15.0, 30.0),
    )
    chunks = (ChunkWindow(0, 0.0, 20.0), ChunkWindow(1, 15.0, 35.0))
    spans = (
        RawCandidateSpan(0, 0.0, 4.0),
        RawCandidateSpan(1, 18.0, 30.0),
    )
    for policy in (FS0, FS1, FS2):
        first = select_emit_frames(policy, segments, timing, chunks, spans)
        second = select_emit_frames(policy, segments, timing, chunks, spans)
        assert first == second
        assert len(set(first.emitted_frames)) == len(first.emitted_frames)
        assert set(first.emitted_frames) <= set(first.candidate_frames)


# ---------------------------------------------------------------------------
# Metrics / guardrails / decisions
# ---------------------------------------------------------------------------

def test_weak_frame_metrics_exact_formulas():
    metrics = weak_frame_video_metrics({0, 1, 2, 3}, {1, 2, 3, 4})
    assert metrics.true_positive_frames == 3
    assert metrics.predicted_frames == 4
    assert metrics.reference_frames == 4
    assert metrics.precision == 0.75
    assert metrics.recall == 0.75
    assert metrics.f1 == 0.75
    assert metrics.missed_reference_frames == 1
    assert metrics.low_recall is True


def test_macro_and_guardrail_aggregates():
    predictions = {"a": {0, 1, 2, 3}, "b": {0, 1}}
    references = {"a": {1, 2, 3, 4}, "b": {0, 1, 2, 3}}
    evaluation = evaluate_video_set(predictions, references)
    macro = evaluation["macro"]
    assert macro["video_count"] == 2
    assert macro["macro_precision"] == pytest.approx((0.75 + 1.0) / 2)
    assert macro["macro_recall"] == pytest.approx((0.75 + 0.5) / 2)
    assert macro["low_recall_rate"] == pytest.approx(1.0)
    assert macro["missed_reference_frame_rate"] == pytest.approx(3 / 8)
    assert macro["empty_prediction_count"] == 0
    assert "DIAGNOSTIC_ONLY_pooled_f1" in macro


def test_recall_guardrails_and_dev_objective():
    baseline = evaluate_video_set({"v": {0, 1, 2, 3}}, {"v": {0, 1, 2, 3}})
    candidate = evaluate_video_set({"v": {0, 1, 2, 3}}, {"v": {0, 1, 2, 3}})
    guard = recall_guardrails(baseline, candidate)
    assert guard["all_pass"] is True
    objective = dev_promotion_objective(baseline, candidate)
    assert objective["eligible"] is False  # no strict precision gain

    worse = evaluate_video_set({"v": {0}}, {"v": {0, 1, 2, 3}})
    assert recall_guardrails(baseline, worse)["all_pass"] is False


def test_dev_tie_break_prefers_conservative_science_within_equal_metrics():
    reference = {"v": {0, 1, 2, 3}}
    baseline = evaluate_video_set({"v": {0, 1, 2, 3, 4, 5}}, reference)
    fs1 = evaluate_video_set({"v": {0, 1, 2, 3}}, reference)
    fs2 = evaluate_video_set({"v": {0, 1, 2, 3}}, reference)
    fs1["arm"] = ARM_NAMES[FS1]
    fs2["arm"] = ARM_NAMES[FS2]
    baseline["arm"] = ARM_NAMES[FS0]
    # Both arms strictly improve precision and F1 with equal recall.
    assert fs1["macro"]["macro_precision"] > baseline["macro"]["macro_precision"]
    assert fs1["macro"]["macro_f1"] - baseline["macro"]["macro_f1"] >= 0.01
    selection = select_dev_winner(baseline, {FS1: fs1, FS2: fs2})
    assert selection["winner_policy"] == FS1  # equal science -> conservatism FS-1 > FS-2

    no_gain = select_dev_winner(
        baseline,
        {FS1: evaluate_video_set({"v": {0, 1, 2, 3, 4, 5}}, reference)},
    )
    assert no_gain["winner_policy"] is None


def test_hard_gates_and_decision_tree():
    baseline = evaluate_video_set({"v": {0, 1}}, {"v": {0, 1}})
    candidate = evaluate_video_set({"v": {0}}, {"v": {0, 1}})
    result = hard_gates(baseline, candidate)
    assert result["all_pass"] is False
    assert "H3_empty_prediction_count" not in result["failed_checks"]
    decision = adjudicate_stage5_5(FS1, result)
    assert decision["status"] == "FS0_FINAL_FROZEN"
    assert decision["frozen_policy"] == FS0
    assert decision["stage5_5_terminal"] == "STAGE5_5_CLOSED"

    assert adjudicate_stage5_5(None, None)["status"] == "FS0_FINAL_FROZEN"
    assert adjudicate_stage5_5(FS1, None)["status"] == "DEV_WINNER_PENDING_HARD"

    good = hard_gates(baseline, evaluate_video_set({"v": {0, 1}}, {"v": {0, 1}}))
    assert good["all_pass"] is True
    assert adjudicate_stage5_5(FS1, good)["status"] == "FS_CANDIDATE_FINAL_FROZEN"


# ---------------------------------------------------------------------------
# Preregistration identity
# ---------------------------------------------------------------------------

def test_master_protocol_and_configs_hashes_close():
    master = runner.validate_master_preregistration()
    assert master["status"] == runner.MASTER_STATUS
    dev_protocol = json.loads(runner.DEV_PROTOCOL_PATH.read_text(encoding="utf-8"))
    hard_protocol = json.loads(runner.HARD_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert runner.validate_protocol(
        dev_protocol, expected_status="PREREGISTERED_BEFORE_DEV166"
    )
    assert runner.validate_protocol(
        hard_protocol, expected_status="PREREGISTERED_BEFORE_DEV166"
    )
    for policy in (FS0, FS1, FS2):
        runner.validate_arm_config(policy)
    assert dev_protocol["protocol_semantic_sha256"] != hard_protocol["protocol_semantic_sha256"]
    assert "weak-frame temporal proxy" in dev_protocol["proxy_label"]
    # Hard protocol must be frozen before Dev (conditional, not post-hoc).
    assert hard_protocol["status"] == "PREREGISTERED_BEFORE_DEV166"


def test_execution_configs_bind_protocol_bytes():
    for config_path, protocol_path in (
        (runner.DEV_CONFIG_PATH, runner.DEV_PROTOCOL_PATH),
        (runner.HARD_CONFIG_PATH, runner.HARD_PROTOCOL_PATH),
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["protocol_sha256"] == file_sha256(protocol_path)
        assert config["execution_config_schema_version"] == runner.EXECUTION_CONFIG_SCHEMA_VERSION


def test_dev_promotion_marker_binds_and_is_fail_closed(tmp_path):
    marker = runner.build_dev_promotion_marker(
        execution_head="deadbeef",
        dev_protocol_sha256=file_sha256(runner.DEV_PROTOCOL_PATH),
        dev_config_sha256=file_sha256(runner.DEV_CONFIG_PATH),
        evaluation_sha256=canonical_sha256({"x": 1}),
        selected_candidate=FS1,
        guard_results={"all_pass": True},
    )
    assert marker["all_pass"] is True
    assert marker["selected_candidate"] == FS1
    assert marker["override_allowed"] is False
    assert marker["marker_semantic_sha256"] == canonical_sha256(
        {k: v for k, v in marker.items() if k != "marker_semantic_sha256"}
    )


def test_hard_runner_blocked_before_execution_without_marker(tmp_path):
    config = json.loads(runner.HARD_CONFIG_PATH.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_dict(
        {
            "name": "test", "repo": str(REPO), "datasets": str(tmp_path / "datasets"),
            "models": str(tmp_path / "models"), "hf_cache": str(tmp_path / "hf"),
            "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs"),
            "cache": str(tmp_path / "cache"), "tmp": str(tmp_path / "tmp"),
            "archive": str(tmp_path / "archive"),
        }
    )
    with pytest.raises(FrozenInputError, match="BLOCKED_BEFORE_EXECUTION"):
        runner.load_dev_promotion_marker(config, environment)

    marker_path = runner._marker_path(environment)
    marker = runner.build_dev_promotion_marker(
        execution_head="deadbeef",
        dev_protocol_sha256=file_sha256(runner.DEV_PROTOCOL_PATH),
        dev_config_sha256=file_sha256(runner.DEV_CONFIG_PATH),
        evaluation_sha256=canonical_sha256({"x": 1}),
        selected_candidate=FS2,
        guard_results={"all_pass": True},
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    loaded = runner.load_dev_promotion_marker(config, environment)
    assert loaded["selected_candidate"] == FS2


def test_report_generator_emits_all_sections(tmp_path):
    config = json.loads(runner.DEV_CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "experiment_report.md"
    runner.render_stage5_5_report(
        output,
        config=config,
        validation={"status": "PASS", "decision": {"status": "FS0_FINAL_FROZEN"}},
        metrics={"FS-0": {"macro": {"macro_f1": 0.1}}},
        runtime={"heldout_access": 0},
    )
    report = output.read_text(encoding="utf-8")
    assert all(f"## {section}" in report for section in runner.STAGE5_5_REPORT_SECTIONS)
    assert "FS0_FINAL_FROZEN" in report
    assert "weak-frame temporal proxy" in report


def test_registry_and_launcher_register_stage5_5():
    from scripts.experiments.stage5 import run as stage5_registry

    for experiment in ("stage5_5_dev_formal", "stage5_5_hard_confirmation"):
        assert experiment in stage5_registry.RUNNERS
        assert stage5_registry.LAUNCH[experiment] == {
            "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
            "forbid_active_processes": ["vllm", "qwen"],
        }
    launcher = REPO / "scripts" / "experiments" / "launch_experiment.ps1"
    assert launcher.is_file()
    described = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "experiments" / "registry.py"),
         "describe", "--experiment", "stage5_5_hard_confirmation"],
        capture_output=True, text=True, check=True,
    )
    spec = json.loads(described.stdout)
    assert spec["target"] == "AUTODL"
    assert spec["config"] == "configs/experiments/stage5/stage5_5_hard_confirmation.json"


def test_ts5_revised_frozen_baseline_is_bound():
    assert runner.FROZEN_TEMPORAL_BASELINE == "projected_state_canonical_center_ema_v1"
    # Importing the frozen Stage 5.4 TS-5 Revised operator must stay available and
    # unchanged in behaviour (regression sentinel for the Stage 5.5 front input).
    from aic_video_highlight.spatial_composition.temporal_smoothing import (
        canonical_center_from_placement,
        place_crop_from_center,
    )

    center = canonical_center_from_placement(277, 1, 168, 300)
    assert place_crop_from_center(534, 300, 9, 16, center)[:2] == (277, 1)


def test_progress_final_freeze_and_contract_proxy_helpers(tmp_path):
    runner.write_progress(tmp_path / "machine", {"videos": 3, "frames": 30})
    progress = json.loads((tmp_path / "machine" / "progress.json").read_text(encoding="utf-8"))
    assert progress == {"videos": 3, "frames": 30}

    runner.write_final_freeze(
        tmp_path, {"status": "FS0_FINAL_FROZEN", "frozen_policy": FS0}
    )
    freeze = json.loads((tmp_path / "final_freeze.json").read_text(encoding="utf-8"))
    assert freeze["stage5_5_terminal"] == "STAGE5_5_CLOSED"
    with pytest.raises(FileExistsError):
        runner.write_final_freeze(tmp_path, {"status": "FS0_FINAL_FROZEN"})

    proxy_path = tmp_path / "official_contract_proxy.jsonl"
    runner.render_official_contract_proxy(
        proxy_path,
        {"v": [{"frame": 7, "x": 10, "y": 20, "w": 30, "h": 40}]},
    )
    proxy = json.loads(proxy_path.read_text(encoding="utf-8").splitlines()[0])
    assert proxy["proxy"] is True
    assert proxy["official_format"] is False
    assert proxy["frames"] == [{"frame": 7, "x": 10, "y": 20, "w": 30, "h": 40}]


def test_real_frozen_cache_preserves_cross_chunk_provenance_if_available():
    """Provenance audit: E_t/S_t are recoverable from the real Stage 4.2 cache."""
    try:
        environment = EnvironmentPaths.from_json(
            REPO / "configs" / "environments" / "windows_local.json"
        )
    except FileNotFoundError:
        pytest.skip("windows_local.json not available")
    candidates = list(
        environment.archive.glob(
            "Stage4_*/02_Frozen_Candidate_Cache_Identity_Replay/FORMAL_431/cache_build_a"
        )
    )
    if not candidates:
        pytest.skip("local Stage 4.2 formal cache archive not present")
    cache_dir = candidates[0]
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    assert manifest["global_semantic_sha256"] == runner.FROZEN_CANDIDATE_CACHE_GLOBAL_SHA256
    entry = manifest["records"][0]
    record = json.loads((cache_dir / entry["path"]).read_text(encoding="utf-8"))
    windows, spans = runner.parse_chunk_evidence(record)
    assert windows and record["raw_candidates"]
    timing = _timing(fps=2.0, frame_count=100000)
    profile = compute_support_by_frame(range(0, 60), timing, windows, spans)
    assert set(profile) == set(range(0, 60))
    # Every candidate frame has a defined eligible/supporting count.
    assert all(item.eligible_chunks >= 0 for item in profile.values())


def test_frame_only_evaluation_for_analysis_sets_without_ts5():
    timing = _timing(fps=1.0, frame_count=50)
    video = runner.FrozenVideoInputs(
        video_id="v",
        timing=timing,
        segments=(FinalSegment("s0", 0.0, 5.0),),
        chunk_windows=(ChunkWindow(0, 0.0, 5.0), ChunkWindow(1, 0.0, 5.0)),
        raw_spans=(RawCandidateSpan(0, 0.0, 2.0), RawCandidateSpan(1, 3.0, 5.0)),
        reference_frames=(0, 1),
        ts5_frames=(),
        ts5_bboxes=(),
    )
    evaluation = runner.evaluate_arm_on_video_set((video,), FS0)
    assert evaluation["macro"]["total_predicted_frames"] == 5
    assert evaluation["macro"]["empty_prediction_count"] == 0

    selection = runner.run_frame_selection(video, FS1)
    assert set(selection.emitted_frames) <= {0, 1, 2, 3, 4}


def test_load_ts5_frames_reads_stage5_4_shard(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "v.json").write_text(
        json.dumps(
            [
                {"frame": 3, "ts5": {"x": 10, "y": 20, "w": 30, "h": 40}},
                {"frame": 4, "ts5": {"x": 11, "y": 21, "w": 31, "h": 41}},
            ]
        ),
        encoding="utf-8",
    )
    frames, bboxes = runner._load_ts5_frames(tmp_path, "v")
    assert frames == (3, 4)
    assert bboxes == ((3, 10, 20, 30, 40), (4, 11, 21, 31, 41))


def test_execution_configs_bind_real_canonical_input_paths():
    dev = json.loads(runner.DEV_CONFIG_PATH.read_text(encoding="utf-8"))
    hard = json.loads(runner.HARD_CONFIG_PATH.read_text(encoding="utf-8"))
    assert dev["inputs"]["frozen_candidate_cache"]["path"] == (
        "stage4_2_formal_431/FORMAL_431/cache_build_a"
    )
    assert dev["inputs"]["role_manifest"]["path"].endswith("dev_tune_166.json")
    assert dev["inputs"]["stage3_frozen_predictions"]["path"] == (
        "stage3/stage3_dev_full_baseline_v1/predictions.jsonl"
    )
    assert "stage5_4_ts5_revised_output" in dev["inputs"]
    assert hard["inputs"]["role_manifest"]["path"].endswith("hard_stress_229.json")
    assert hard["inputs"]["stage3_frozen_predictions"]["path"] == (
        "stage3/stage3_hard_full_baseline_v1/predictions.jsonl"
    )
    # Hard229 has no Stage 5.4 spatial output or Stage 5.1 metadata cache.
    assert "stage5_4_ts5_revised_output" not in hard["inputs"]
    assert "stage5_1_metadata_cache" not in hard["inputs"]
