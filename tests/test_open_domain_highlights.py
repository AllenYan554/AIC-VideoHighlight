"""Synthetic fixtures test contracts; they are never research GT or model results."""
import copy
import hashlib
import json

import pytest

from aic_video_highlight import open_domain_highlights as H


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / "raw"
    proof = tmp_path / "provenance"
    raw.mkdir()
    proof.mkdir()
    video = raw / "v1.bin"
    evidence = proof / "rights.txt"
    video.write_bytes(b"synthetic-test-fixture-not-a-video")
    evidence.write_text("synthetic-explicit-test-permission", encoding="utf-8")
    exclusions = {"schema": H.SCHEMA + ".exclusions/v1", "source_ids": ["YT_RESERVED"], "source_sha256": []}
    row = {"video_id": "v1", "source_ids": ["PUBLIC_SOURCE_1"], "source_url": "https://example.org/video/1",
           "source_sha256": H.file_sha256(video), "video_path": str(video), "duration_s": 4.0,
           "domain": "daily", "content_group": "independent1", "split": "EXTERNAL_TEST_ONLY", "no_query": True,
           "independence_review": "HUMAN_CONFIRMED_NO_KNOWN_REUPLOAD_OR_EXCERPT",
           "license": {"status": "VERIFIED", "identifier": "TEST_PERMISSION", "scope": "RAW_VIDEO_RESEARCH_AND_HUMAN_ANNOTATION",
                       "evidence_url": "https://example.org/license", "evidence_path": str(evidence), "evidence_sha256": H.file_sha256(evidence)}}
    manifest = {"schema": H.SCHEMA + ".manifest/v1", "dataset": "ODH_HUMAN_PILOT_V1", "domain_targets": {"daily": 8},
                "exclusions_sha256": H.digest_json(exclusions), "videos": [row]}
    protocol = {"schema": H.SCHEMA + ".protocol/v1", "label_semantics": "DIRECT_HUMAN_BINARY_HIGHLIGHT", "bin_seconds": 2.0, "n_raters": 3}
    return raw, manifest, exclusions, protocol


def annotations(inputs):
    _, manifest, _, protocol = inputs
    result = []
    for i in range(3):
        a = H.annotation_template(manifest["videos"][0], H.digest_json(protocol), f"human{i}")
        a.update(annotation_method="HUMAN_MANUAL", tool_assistance="NONE", watched_full_video=True)
        for j, b in enumerate(a["bins"]):
            b.update(highlight=j == 0, uncertainty_reason="")
        result.append(a)
    return result


def test_empty_pool_is_explicitly_blocked(inputs):
    raw, m, e, _ = inputs
    m["videos"] = []
    assert H.validate_manifest(m, e, raw_root=raw)["status"] == "BLOCKED_EMPTY_POOL"


def test_valid_admission_checks_real_bytes(inputs):
    raw, m, e, _ = inputs
    assert H.validate_manifest(m, e, raw_root=raw)["admitted_videos"] == 1
    (raw / "v1.bin").write_bytes(b"changed")
    with pytest.raises(H.EligibilityError, match="原视频身份"):
        H.validate_manifest(m, e, raw_root=raw)


@pytest.mark.parametrize("mutation", ["duplicate", "heldout", "license", "query", "split", "time", "path", "proof", "independence"])
def test_reject_unsafe_or_ineligible_source(inputs, mutation):
    raw, m, e, _ = inputs
    r = m["videos"][0]
    if mutation == "duplicate": m["videos"].append(copy.deepcopy(r))
    if mutation == "heldout": r["source_ids"] = ["YT_RESERVED"]
    if mutation == "license": r["license"]["scope"] = "FEATURES_ONLY"
    if mutation == "query": r["no_query"] = False
    if mutation == "split": r["split"] = "TRAIN"
    if mutation == "time": r["duration_s"] = float("nan")
    if mutation == "path": r["video_path"] = str(raw.parent / "protected.bin")
    if mutation == "proof": r["license"]["evidence_sha256"] = "0" * 64
    if mutation == "independence": r["independence_review"] = "UNKNOWN"
    with pytest.raises(H.EligibilityError): H.validate_manifest(m, e, raw_root=raw)


def test_bins_preserve_partial_tail():
    assert H.bins_for_duration(4.1)[-1] == {"index": 2, "start_s": 4.0, "end_s": 4.1}


@pytest.mark.parametrize("field,value", [("annotation_method", "AI"), ("watched_full_video", False), ("tool_assistance", "QWEN")])
def test_ai_or_incomplete_watching_is_not_human_gt(inputs, field, value):
    a = annotations(inputs)
    a[0][field] = value
    with pytest.raises(H.EligibilityError): H.compile_human_gt(inputs[1], inputs[3], a)


@pytest.mark.parametrize("bad_label", [1, 0.5, "true"])
def test_no_rating_binarization(inputs, bad_label):
    a = annotations(inputs)
    a[0]["bins"][0]["highlight"] = bad_label
    with pytest.raises(H.EligibilityError, match="二元"):
        H.compile_human_gt(inputs[1], inputs[3], a)


def test_time_alignment_and_distinct_humans(inputs):
    a = annotations(inputs)
    a[0]["bins"][0]["end_s"] = 1.0
    with pytest.raises(H.EligibilityError, match="时间轴"):
        H.compile_human_gt(inputs[1], inputs[3], a)
    a = annotations(inputs)
    a[1]["rater_id"] = a[0]["rater_id"]
    with pytest.raises(H.EligibilityError, match="三位"):
        H.compile_human_gt(inputs[1], inputs[3], a)


def test_disagreement_needs_independent_human_adjudication(inputs):
    a = annotations(inputs)
    a[0]["bins"][0]["highlight"] = False
    gt = H.compile_human_gt(inputs[1], inputs[3], a)
    assert not gt["human_gt_ready"] and len(gt["unresolved_bins"]) == 1
    verdict = {"video_id": "v1", "index": 0, "source_sha256": inputs[1]["videos"][0]["source_sha256"],
               "protocol_sha256": H.digest_json(inputs[3]), "annotation_method": "HUMAN_MANUAL", "tool_assistance": "NONE",
               "watched_full_video": True, "adjudicator_id": "fourth_human", "highlight": True, "reason": "test adjudication"}
    assert H.compile_human_gt(inputs[1], inputs[3], a, [verdict])["human_gt_ready"]
    verdict["adjudicator_id"] = "human0"
    with pytest.raises(H.EligibilityError, match="独立"):
        H.compile_human_gt(inputs[1], inputs[3], a, [verdict])


def test_tied_scores_have_no_order_advantage():
    assert H.binary_average_precision([True, False], [1., 1.]) == .5
    assert H.binary_average_precision([False, True], [1., 1.]) == .5
    assert H.binary_average_precision([True, False], [2., 1.]) == 1.
    assert H.binary_average_precision([False, True], [2., 1.]) == .5
    assert H.binary_average_precision([False, False], [1., 2.]) is None
    assert H.binary_average_precision([True, True], [1., 2.]) == 1.


def test_pooling_rejects_missing_or_reordered_time():
    assert H.pool_to_bins([0., .5, 1., 1.5, 2., 2.5, 3., 3.5], [1.] * 4 + [0.] * 4, 4.) == [1., 0.]
    for t in ([0., 1., 2., 3.5], [0., .5, .5, 3.5], [0., .5, 1., 1.5]):
        with pytest.raises(H.EligibilityError): H.pool_to_bins(t, [1.] * len(t), 4.)


def test_checkpoint_requires_all_eight_and_exact_hash(tmp_path):
    doc = {"checkpoints": []}
    for seed in H.SEEDS:
        p = tmp_path / f"{seed}.fixture"
        p.write_bytes(str(seed).encode())
        doc["checkpoints"].append({"seed": seed, "path": str(p), "sha256": H.file_sha256(p)})
    assert len(H.verify_checkpoints(doc)) == 8
    doc["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(H.EligibilityError, match="身份"):
        H.verify_checkpoints(doc)


def test_freeze_blocks_missing_pool_before_gt(inputs):
    with pytest.raises(H.EligibilityError, match="50"):
        H.make_freeze(inputs[3], inputs[1], inputs[2], {}, raw_root=inputs[0])


def test_evaluation_requires_complete_eight_seed_predictions(inputs):
    a = annotations(inputs)
    gt = H.compile_human_gt(inputs[1], inputs[3], a)
    ck = {"checkpoints": [{"seed": s, "sha256": hashlib.sha256(str(s).encode()).hexdigest()} for s in H.SEEDS]}
    rows = [{"seed": s, "video_id": "v1", "source_sha256": inputs[1]["videos"][0]["source_sha256"],
             "checkpoint_sha256": ck["checkpoints"][i]["sha256"], "protocol_sha256": H.digest_json(inputs[3]),
             "timestamps": [x / 2 for x in range(8)], "scores": [1.] * 4 + [0.] * 4} for i, s in enumerate(H.SEEDS)]
    result = H.evaluate_predictions(gt, rows, inputs[1], ck, bootstrap_samples=10)
    assert result["primary_macro_ap"] == 1. and result["n_videos"] == 1
    with pytest.raises(H.EligibilityError, match="完整覆盖"):
        H.evaluate_predictions(gt, rows[:-1], inputs[1], ck)
    rows[0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(H.EligibilityError, match="身份"):
        H.evaluate_predictions(gt, rows, inputs[1], ck)


def test_stratified_selection_is_order_independent(inputs):
    raw, m, e, _ = inputs
    second = copy.deepcopy(m["videos"][0])
    p = raw / "v2.bin"
    p.write_bytes(b"another-synthetic-fixture")
    second.update(video_id="v2", source_ids=["PUBLIC_SOURCE_2"], video_path=str(p),
                  source_sha256=H.file_sha256(p), content_group="independent2")
    m["videos"].append(second)
    m["domain_targets"]["daily"] = 1
    selected = H.stratified_selection(m, e, raw_root=raw)
    m["videos"].reverse()
    assert selected["videos"] == H.stratified_selection(m, e, raw_root=raw)["videos"]


def test_constant_annotation_agreement_is_undefined_not_perfect_kappa():
    result = H.annotation_agreement([[True, True, True], [True, True, True]])
    assert result["unanimous_fraction"] == 1. and result["fleiss_kappa"] is None


def test_uncertainty_and_per_video_agreement_keep_denominators(inputs):
    a = annotations(inputs)
    a[0]["bins"][0].update(highlight=None, uncertainty_reason="无法确定")
    result = H.compile_human_gt(inputs[1], inputs[3], a)
    quality = result["annotation_quality"]
    assert quality["n_bins"] == 2 and quality["uncertain_vote_fraction"] == pytest.approx(1 / 6)
    per_video = quality["per_video"][0]
    assert per_video["bins_requiring_adjudication"] == per_video["unresolved_bins"] == 1
    assert per_video["agreement"]["complete_binary_bins"] == 1
    assert result["videos"][0]["rater_ids"] == ["human0", "human1", "human2"]


@pytest.fixture
def freeze_inputs(inputs, tmp_path):
    raw, m, exclusions, protocol = inputs
    m["domain_targets"] = {f"domain{i}": 8 for i in range(8)}
    base_row = m["videos"][0]
    m["videos"] = []
    for i in range(64):
        path = raw / f"video{i}.fixture"
        path.write_bytes(f"synthetic-not-video-{i}".encode())
        row = copy.deepcopy(base_row)
        row.update(video_id=f"v{i}", source_ids=[f"SYNTHETIC_SOURCE_{i}"],
                   video_path=str(path), source_sha256=H.file_sha256(path),
                   content_group=f"independent{i}", domain=f"domain{i % 8}")
        m["videos"].append(row)
    ck = {"checkpoints": []}
    for seed in H.SEEDS:
        path = tmp_path / f"seed{seed}.fixture"
        path.write_bytes(str(seed).encode())
        ck["checkpoints"].append({"seed": seed, "path": str(path), "sha256": H.file_sha256(path)})
    protocol.update(no_query=True, domain_targets=copy.deepcopy(m["domain_targets"]),
                    canonical_checkpoints_sha256=H.digest_json(ck))
    for key, pkey, skey in (("model", "model_code", "model_code_sha256"),
                           ("evaluation_code", "path", "sha256"), ("cli_code", "path", "sha256")):
        path = tmp_path / f"{key}.txt"
        path.write_bytes(f"synthetic-code-identity-{key}".encode())
        protocol[key] = {pkey: str(path), skey: H.file_sha256(path)}
    return raw, m, exclusions, protocol, ck


def test_formal_freeze_binds_sources_protocol_checkpoints_and_code(freeze_inputs):
    raw, m, e, p, ck = freeze_inputs
    frozen = H.make_freeze(p, m, e, ck, raw_root=raw)
    H.verify_freeze(frozen, p, m, e, ck, raw_root=raw)
    assert frozen["labels_read"] is False and len(frozen["code_sha256"]) == 3
    p["bootstrap_seed"] = 999
    with pytest.raises(H.EligibilityError, match="冻结身份变化"):
        H.verify_freeze(frozen, p, m, e, ck, raw_root=raw)


@pytest.mark.parametrize("mutation", ["domain_plan", "model_code", "evaluator_code", "no_query"])
def test_freeze_rejects_changed_design_or_code(freeze_inputs, mutation):
    raw, m, e, p, ck = freeze_inputs
    if mutation == "domain_plan": p["domain_targets"]["domain0"] = 9
    if mutation == "no_query": p["no_query"] = False
    if mutation == "model_code":
        from pathlib import Path
        Path(p["model"]["model_code"]).write_bytes(b"changed")
    if mutation == "evaluator_code":
        from pathlib import Path
        Path(p["evaluation_code"]["path"]).write_bytes(b"changed")
    with pytest.raises(H.EligibilityError):
        H.make_freeze(p, m, e, ck, raw_root=raw)


@pytest.mark.parametrize("positive", [True, False])
def test_evaluation_reports_single_class_and_every_domain(inputs, positive):
    a = annotations(inputs)
    for annotation in a:
        for b in annotation["bins"]: b["highlight"] = positive
    gt = H.compile_human_gt(inputs[1], inputs[3], a)
    ck = {"checkpoints": [{"seed": s, "sha256": hashlib.sha256(str(s).encode()).hexdigest()} for s in H.SEEDS]}
    rows = [{"seed": s, "video_id": "v1", "source_sha256": inputs[1]["videos"][0]["source_sha256"],
             "checkpoint_sha256": ck["checkpoints"][i]["sha256"], "protocol_sha256": H.digest_json(inputs[3]),
             "timestamps": [x / 2 for x in range(8)], "scores": [1.] * 8} for i, s in enumerate(H.SEEDS)]
    result = H.evaluate_predictions(gt, rows, inputs[1], ck, bootstrap_samples=10)
    assert result["n_all_positive"] == int(positive)
    assert result["n_no_positive"] == int(not positive)
    assert result["domain_summary"]["daily"]["mean_ap"] == (1. if positive else None)
    assert result["macro_ap_by_seed"][str(H.SEEDS[0])] == (1. if positive else None)
