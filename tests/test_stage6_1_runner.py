"""Stage 6.1 Formal runner contracts (CPU-only, synthetic frozen layout)."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts" / "experiments" / "run_stage6_1_literature_frame_selection.py"

sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.composition.frame_projection import (  # noqa: E402
    CFR_FPS,
    PTS_TABLE,
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("aic_test_stage61_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- small helpers ---------------------------------------------------------

def _record(video_id: str, frames, ratio=(9, 16), bbox=(1, 1, 10)):
    return {
        "video_id": video_id,
        "targetRatioWH": list(ratio),
        "predictions": [{"frame": int(frame), "bboxes": list(bbox)} for frame in frames],
    }


def _spec(runner, video_id="v0", frame_count=40, fs0=(0, 4, 8, 12), sample_fps=2.0):
    timing = runner.build_timing(
        {"fps": 30.0, "frame_count": frame_count, "timestamp_mode": CFR_FPS, "pts_timestamps": None},
        video_id,
    )
    mapping = runner.sample_frame_mapping(timing, sample_fps)
    return runner.VideoSpec(
        video_id=video_id,
        timing=timing,
        fs0_frames=tuple(fs0),
        control_record=_record(video_id, fs0),
        video_path=f"/tmp/{video_id}.mp4",
        video_sha256="0" * 64,
        sample_frames=tuple(item.source_frame for item in mapping),
        mapping_identity=runner.sample_trace_identity(mapping),
    )


# --- dispatch / config -----------------------------------------------------

def test_resolve_arms(runner):
    assert runner.resolve_arms("all") == (runner.FS0, runner.PGL, runner.VASNET)
    assert runner.resolve_arms("fs0") == (runner.FS0,)
    assert runner.resolve_arms("pgl") == (runner.PGL,)
    assert runner.resolve_arms("vasnet") == (runner.VASNET,)
    with pytest.raises(runner.Stage61Error):
        runner.resolve_arms("nope")


def test_kts_config_matches_protocol_rule(runner):
    for n in (2, 3, 10, 24, 269):
        config = runner.kts_config_for(n)
        assert config.ncp_max == max(1, min(int(np.ceil(n / 2)), n - 1))
        assert config.vmax == 1.0 and config.lmin == 1


def test_build_timing_cfr_and_pts(runner):
    cfr = runner.build_timing({"fps": 29.97, "frame_count": 100, "timestamp_mode": CFR_FPS, "pts_timestamps": None}, "v")
    assert cfr.timestamp_mode == CFR_FPS and cfr.frame_count == 100
    pts = runner.build_timing(
        {"fps": 4.0, "frame_count": 4, "timestamp_mode": PTS_TABLE, "pts_timestamps": ["0", "1/4", "1/2", "3/4"]},
        "v",
    )
    assert pts.pts_timestamps == (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))


def test_sample_trace_identity_is_stable_and_sensitive(runner):
    timing = runner.build_timing({"fps": 30.0, "frame_count": 40, "timestamp_mode": CFR_FPS, "pts_timestamps": None}, "v")
    mapping = runner.sample_frame_mapping(timing, 2.0)
    assert runner.sample_trace_identity(mapping) == runner.sample_trace_identity(list(mapping))
    longer = runner.sample_frame_mapping(
        runner.build_timing({"fps": 30.0, "frame_count": 100, "timestamp_mode": CFR_FPS, "pts_timestamps": None}, "v"),
        2.0,
    )
    assert runner.sample_trace_identity(mapping) != runner.sample_trace_identity(longer)


# --- identity / hashing ----------------------------------------------------

def test_verify_sha256(runner, tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"hello")
    digest = runner.file_sha256(target)
    assert runner.verify_sha256("x", target, digest) == digest
    with pytest.raises(runner.Stage61Error):
        runner.verify_sha256("x", target, "0" * 64)


def test_fs0_reconstruct_preserves_output_identity(runner):
    records = [_record("a", [0, 4, 8]), _record("b", [])]
    rebuilt = runner.fs0_reconstruct(records)
    assert rebuilt == records
    assert runner.canonical_sha256(rebuilt) == runner.canonical_sha256(records)
    assert rebuilt[0] is not records[0]


def test_build_video_specs_and_fs0_frames(runner):
    metadata = {"records": {"a": {"fps": 30.0, "frame_count": 40, "timestamp_mode": CFR_FPS, "pts_timestamps": None}}}
    control = [_record("a", [4, 0, 8, 4])]
    specs = runner.build_video_specs(metadata["records"], control, {"a": "f" * 64})
    assert len(specs) == 1
    assert specs[0].fs0_frames == (0, 4, 8)
    assert specs[0].sample_frames[0] == 0
    assert specs[0].sample_frames == tuple(sorted(set(specs[0].sample_frames)))


# --- feature cache identity (resume) ---------------------------------------

def test_feature_cache_identity_and_validity(runner):
    spec = _spec(runner)
    identity = runner.feature_cache_identity(spec, aic_git_head="deadbeef")
    assert identity["extractor_weights_sha256"] == runner.FROZEN_GOOGLENET_SHA
    assert identity["mapping_sha256"] == spec.mapping_identity
    assert runner.feature_cache_is_valid(identity, dict(identity))
    tampered = dict(identity, aic_git_head="other")
    assert not runner.feature_cache_is_valid(identity, tampered)
    assert not runner.feature_cache_is_valid(identity, None)


def test_shared_feature_cache_writes_and_resumes(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "read_frames_bgr",
        lambda path, frames: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in frames],
    )

    class FakeExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, frames):
            self.calls += 1
            return np.ones((len(frames), 1024), dtype=np.float32)

    spec = _spec(runner)
    extractor = FakeExtractor()
    cache = runner.SharedFeatureCache(tmp_path / "cache", extractor, aic_git_head="head", resume=True)
    first = cache.features(spec)
    assert first.shape == (len(spec.sample_frames), 1024)
    assert extractor.calls == 1
    second = cache.features(spec)
    assert np.array_equal(first, second)
    assert extractor.calls == 1  # served from cache
    assert cache.hits == 1 and cache.misses == 1

    resumed = runner.SharedFeatureCache(tmp_path / "cache", FakeExtractor(), aic_git_head="head", resume=True)
    resumed.features(spec)
    assert resumed.hits == 1


# --- subset adapter / invariants -------------------------------------------

def test_select_with_scores_is_subset_of_fs0(runner):
    spec = _spec(runner, frame_count=200, fs0=tuple(range(0, 200, 6)))
    features = np.random.default_rng(0).standard_normal((len(spec.sample_frames), 1024)).astype(np.float32)
    scores = np.linspace(0.0, 1.0, len(spec.sample_frames))
    selection = runner.select_with_scores("pgl_sum", spec, features, scores)
    assert selection.is_subset_of_fs0
    assert set(selection.kept_frames) | set(selection.dropped_frames) == set(spec.fs0_frames)
    assert set(selection.kept_frames) == set(spec.fs0_frames) & set(selection.selected_source_frames)


def test_run_literature_arm_invariants_and_dispatch(runner):
    specs = [
        _spec(runner, video_id="a", frame_count=80, fs0=tuple(range(0, 80, 3))),
        _spec(runner, video_id="b", frame_count=40, fs0=(0, 8, 16, 24)),
    ]

    def feature_fn(spec):
        return np.random.default_rng(len(spec.video_id)).standard_normal((len(spec.sample_frames), 1024)).astype(np.float32)

    def score_fn(arm, features):
        return np.linspace(0.0, 1.0, features.shape[0])

    for arm, method in ((runner.PGL, "pgl_sum"), (runner.VASNET, "vasnet")):
        predictions, selections = runner.run_literature_arm(arm, specs, feature_fn, score_fn)
        assert len(predictions) == 2
        for record in predictions:
            video_id = record["video_id"]
            assert selections[video_id].method == method
        audit = runner.audit_subset_invariants(specs, predictions, selections)
        assert audit["violations"] == 0
        assert audit["selected_subset_of_fs0"] is True
        assert audit["kept_bbox_unchanged"] is True


def test_audit_subset_invariants_detects_bbox_change(runner):
    from aic_video_highlight.composition.literature_frame_selection import LiteratureFrameSelection

    selection = LiteratureFrameSelection(
        method="pgl_sum",
        sample_frames=(0, 2, 4),
        frame_scores=(1.0, 1.0, 1.0),
        shot_bounds=((0, 2),),
        selected_shots=(0,),
        budget_frames=10,
        selected_sample_indices=(0, 1),
        selected_source_frames=(0, 1, 2, 3),
        fs0_frames=(0, 2),
        kept_frames=(0, 2),
        dropped_frames=(),
    )
    spec = _spec(runner, video_id="a", frame_count=8, fs0=(0, 2))
    record = dict(spec.control_record)
    record["predictions"] = [
        {"frame": 0, "bboxes": [999, 999, 999]},
        {"frame": 2, "bboxes": [1, 1, 10]},
    ]
    with pytest.raises(runner.Stage61Error):
        runner.audit_subset_invariants([spec], [record], {"a": selection})


# --- output identity / resume ----------------------------------------------

def test_write_arm_output_and_resume_identity(runner, tmp_path):
    records = [_record("a", [0, 4, 8]), _record("b", [0])]
    identity = {"schema": "x", "arm": "fs0", "run_group": "g", "control_predictions_sha256": "abc"}
    path, payload = runner.write_arm_output(tmp_path, "fs0", records, identity)
    assert path.is_file()
    assert payload["contract_is_valid"] is True
    assert payload["predictions_scientific_sha256"] == runner.canonical_sha256(records)
    assert runner.arm_is_resumable(runner.arm_dir(tmp_path, "fs0"), identity)
    assert not runner.arm_is_resumable(runner.arm_dir(tmp_path, "fs0"), dict(identity, run_group="other"))


# --- paired / guard analytics ----------------------------------------------

def _metrics(per_video):
    return {"per_video": per_video}


def test_compute_paired_win_tie_loss(runner):
    baseline = _metrics({"a": {"f": 0.2}, "b": {"f": 0.5}, "c": {"f": 0.1}})
    challenger = _metrics({"a": {"f": 0.3}, "b": {"f": 0.5}, "c": {"f": 0.05}})
    paired = runner.compute_paired(baseline, challenger)
    assert (paired["win"], paired["tie"], paired["loss"]) == (1, 1, 1)
    assert paired["max_regression"]["video_id"] == "c"
    assert paired["max_gain"]["video_id"] == "a"


def test_recall_guard_counts(runner):
    metrics = _metrics(
        {
            "a": {"n_ref": 10, "matched": 10},
            "b": {"n_ref": 10, "matched": 4},   # 0.4
            "c": {"n_ref": 10, "matched": 0},   # 0
            "d": {"n_ref": 0, "matched": 0},    # skipped (no reference)
        }
    )
    guard = runner.recall_guard(metrics)
    assert guard["recall_lt_0_8"] == 2
    assert guard["recall_lt_0_5"] == 2
    assert guard["recall_eq_0"] == 1


def test_empty_reference_audit(runner):
    baseline = _metrics({"a": {"n_pred": 5, "n_ref": 0}, "b": {"n_pred": 3, "n_ref": 0}})
    challenger = _metrics({"a": {"n_pred": 0, "n_ref": 0}, "b": {"n_pred": 1, "n_ref": 0}})
    audit = runner.empty_reference_audit(baseline, challenger, ["a", "b"])
    assert audit["n_pred_before"] == 8 and audit["n_pred_after"] == 1
    assert audit["double_empty_transitions"] == 1
    assert audit["not_scientific_gain"] == "DOUBLE_EMPTY_NOT_SCIENTIFIC_GAIN"


# --- integration: frozen binding / validate-only ---------------------------

def _build_frozen_layout(root: Path, monkeypatch, runner, *, valid=109, empty=57):
    total = valid + empty
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    control_run = root / "control"
    (control_run / "predictions").mkdir(parents=True, exist_ok=True)
    (control_run / "frame_projection").mkdir(parents=True, exist_ok=True)
    (control_run / "input").mkdir(parents=True, exist_ok=True)
    reference_path = root / "weak.jsonl"

    checkpoint_dir = root / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pgl_entries, vas_entries = [], []
    for index in range(10):
        pgl_file = checkpoint_dir / f"pgl_{index}.pt"
        pgl_file.write_bytes(f"pgl{index}".encode())
        pgl_entries.append({"dataset": "SumMe", "split": index % 5, "path": str(pgl_file),
                            "sha256": runner.file_sha256(pgl_file)})
        vas_file = checkpoint_dir / f"vas_{index}.pt"
        vas_file.write_bytes(f"vas{index}".encode())
        vas_entries.append({"dataset": "TVSum", "split": index % 5, "path": str(vas_file),
                            "sha256": runner.file_sha256(vas_file)})

    ids = [f"qvh_{i:06d}_9x16" for i in range(total)]
    valid_ids, empty_ids = ids[:valid], ids[valid:]
    (manifest_dir / "reference_valid_109.txt").write_text("\n".join(valid_ids) + "\n", encoding="utf-8")
    (manifest_dir / "empty_reference_57.txt").write_text("\n".join(empty_ids) + "\n", encoding="utf-8")

    protocol = {
        "schema_version": "aic.stage6_1.formal-protocol/v1",
        "aic_code": {"git_head": "f" * 40},
        "checkpoint_policy": {"pgl_sum": pgl_entries, "vasnet": vas_entries},
        "frozen_stage6_0_control": {"weak_spatial_reference_path": str(reference_path)},
        "evaluation_protocol": {
            "reference_valid_primary": {"identity_file": "reference_valid_109.txt"},
            "empty_reference_diagnostic": {"identity_file": "empty_reference_57.txt"},
        },
    }
    protocol_path = manifest_dir / "formal_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    model_manifest_path = manifest_dir / "model_manifest.json"
    model_manifest_path.write_text(json.dumps({"schema_version": "t"}), encoding="utf-8")

    control_records, metadata, reference, dev_selected = [], {}, [], []
    for index, video_id in enumerate(ids):
        frames = [0, 3, 6] if index % 2 == 0 else [0, 3]
        control_records.append(_record(video_id, frames))
        metadata[video_id] = {
            "fps": 30.0, "frame_count": 30, "timestamp_mode": CFR_FPS,
            "pts_timestamps": None, "video_path": str(root / f"{video_id}.mp4"),
        }
        rois = {str(frame): [1.0, 1.0, 10.0, 10.0] for frame in (frames if index < valid else [])}
        reference.append({"video_id": video_id, "rois": rois})
        dev_selected.append({"video_id": video_id, "sha256": "a" * 64})

    control_path = control_run / "predictions" / "predictions.jsonl"
    control_path.write_text("".join(json.dumps(r) + "\n" for r in control_records), encoding="utf-8")
    (control_run / "frame_projection" / "metadata_cache.json").write_text(
        json.dumps({"records": metadata}), encoding="utf-8"
    )
    (control_run / "input" / "dev_selected.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in dev_selected), encoding="utf-8"
    )
    reference_path.write_text("".join(json.dumps(r) + "\n" for r in reference), encoding="utf-8")

    monkeypatch.setattr(runner, "FROZEN_PROTOCOL_SHA", runner.file_sha256(protocol_path))
    monkeypatch.setattr(runner, "FROZEN_MODEL_MANIFEST_SHA", runner.file_sha256(model_manifest_path))
    monkeypatch.setattr(runner, "FROZEN_CONTROL_PREDICTIONS_SHA", runner.file_sha256(control_path))
    monkeypatch.setattr(runner, "FROZEN_REFERENCE_VALID_SHA", runner.file_sha256(manifest_dir / "reference_valid_109.txt"))
    monkeypatch.setattr(runner, "FROZEN_EMPTY_REFERENCE_SHA", runner.file_sha256(manifest_dir / "empty_reference_57.txt"))

    return {
        "protocol": protocol_path,
        "model_manifest": model_manifest_path,
        "control_run": control_run,
        "reference": reference_path,
        "valid_ids": manifest_dir / "reference_valid_109.txt",
        "empty_ids": manifest_dir / "empty_reference_57.txt",
        "output_root": root / "out",
    }


def _args(runner, layout, **overrides):
    base = dict(
        protocol=layout["protocol"], model_manifest=layout["model_manifest"],
        control_run=layout["control_run"], output_root=layout["output_root"],
        reference=layout["reference"], valid_ids=layout["valid_ids"], empty_ids=layout["empty_ids"],
        arm="all", validate_only=False, resume=False, device="cpu", batch_size=64, run_id="202601010000",
    )
    base.update(overrides)
    return type("Args", (), base)


def test_build_context_binds_frozen_identity(runner, tmp_path, monkeypatch):
    layout = _build_frozen_layout(tmp_path, monkeypatch, runner)
    context = runner.build_context(_args(runner, layout))
    assert len(context.specs) == 166
    assert len(context.valid_ids) == 109 and len(context.empty_ids) == 57
    assert context.group_id == "stage6_1_dev166_formal_202601010000"
    assert context.specs[0].sample_frames[0] == 0


def test_build_context_rejects_tampered_protocol(runner, tmp_path, monkeypatch):
    layout = _build_frozen_layout(tmp_path, monkeypatch, runner)
    monkeypatch.setattr(runner, "FROZEN_PROTOCOL_SHA", "0" * 64)
    with pytest.raises(runner.Stage61Error):
        runner.build_context(_args(runner, layout))


def test_validate_only_passes(runner, tmp_path, monkeypatch):
    layout = _build_frozen_layout(tmp_path, monkeypatch, runner)
    monkeypatch.setattr(runner, "load_models", lambda arm, entries, device: list(entries))
    context = runner.build_context(_args(runner, layout))
    checks = runner.run_validate_only(context)
    assert checks["status"] == "VALIDATE_ONLY_PASS"
    assert checks["pgl_checkpoint_count"] == 10
    assert checks["vasnet_checkpoint_count"] == 10
    assert checks["video_count"] == 166


def test_fs0_arm_reproduction_gate_uses_evaluator(runner, tmp_path, monkeypatch):
    layout = _build_frozen_layout(tmp_path, monkeypatch, runner)
    context = runner.build_context(_args(runner, layout))
    fs0_records = runner.fs0_reconstruct([spec.control_record for spec in context.specs])
    path, _ = runner.write_arm_output(context.output_root, runner.FS0, fs0_records, {"arm": "fs0"})
    full = runner.evaluate_group(path, context.reference_path)
    assert full["frame_n_pred"] == sum(len(r["predictions"]) for r in fs0_records)
    assert full["video_count"] == 166
