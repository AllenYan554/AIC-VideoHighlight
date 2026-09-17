from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aic_video_highlight.ftnet.split_manifest import (
    build_split_manifest,
    content_sha256,
    determinism_check,
    load_ftnet_split_candidates,
    manifest_sha256,
)
from aic_video_highlight.ftnet.split_protocol import (
    SPLIT_ROLES,
    SPLIT_SEED,
    SplitCandidate,
    SplitProtocolError,
    assign_stage7_splits,
    audit_assignment,
)


def _candidate(
    video_id: str,
    category: str = "dog",
    *,
    realized_video_id: str | None = None,
) -> SplitCandidate:
    return SplitCandidate(
        canonical_video_id=video_id,
        realized_video_id=realized_video_id or video_id,
        category=category,
        upstream_official_split="train",
        alignment_pass=True,
        local_video_path=f"raw/{category}/{video_id}.mp4",
        source_file_sha256="a" * 64,
        annotation_identity=f"{category}/{video_id}",
    )


def test_split_assignment_is_deterministic() -> None:
    candidates = [_candidate(f"video-{index:02d}") for index in range(20)]

    first = assign_stage7_splits(candidates, seed=SPLIT_SEED)
    second = assign_stage7_splits(list(reversed(candidates)), seed=SPLIT_SEED)

    assert first == second


def test_same_source_alias_cannot_form_two_split_groups() -> None:
    first = _candidate("canonical-a")
    second = replace(_candidate("canonical-b"), realized_video_id="canonical-a")

    with pytest.raises(SplitProtocolError, match="source alias"):
        assign_stage7_splits([first, second], seed=SPLIT_SEED)


def test_replacement_stays_in_original_canonical_group() -> None:
    replacement = replace(
        _candidate("original-id", "parkour"),
        realized_video_id="replacement-id",
        original_video_id="original-id",
        replacement_video_id="replacement-id",
    )

    assigned = assign_stage7_splits([replacement], seed=SPLIT_SEED)[0]

    assert assigned.replacement_original_id == "original-id"
    assert assigned.source_aliases == ("original-id", "replacement-id")


def test_only_official_train_candidates_are_accepted() -> None:
    official_test = replace(
        _candidate("locked-test-id"),
        upstream_official_split="test",
    )

    with pytest.raises(SplitProtocolError, match="official train"):
        assign_stage7_splits([official_test], seed=SPLIT_SEED)


def test_non_alignment_pass_candidate_is_rejected() -> None:
    failed = replace(_candidate("bad-alignment"), alignment_pass=False)

    with pytest.raises(SplitProtocolError, match="ALIGNMENT_PASS"):
        assign_stage7_splits([failed], seed=SPLIT_SEED)


def test_duplicate_canonical_rows_share_one_group_and_split() -> None:
    first = _candidate("canonical-dup", "skiing")
    second = replace(
        _candidate("canonical-dup", "skiing"),
        realized_video_id="canonical-dup-replacement",
        replacement_video_id="canonical-dup-replacement",
    )

    assigned = assign_stage7_splits([first, second], seed=SPLIT_SEED)

    assert len(assigned) == 2
    assert {entry.source_group_key for entry in assigned}
    assert len({entry.source_group_key for entry in assigned}) == 1
    assert len({entry.stage7_split for entry in assigned}) == 1


def test_canonical_source_cannot_span_two_categories() -> None:
    first = _candidate("canonical-x", "dog")
    second = _candidate("canonical-x", "skating")

    with pytest.raises(SplitProtocolError, match="spans categories"):
        assign_stage7_splits([first, second], seed=SPLIT_SEED)


def test_assignment_is_category_aware() -> None:
    candidates = [
        _candidate(f"dog-{index:02d}", "dog") for index in range(20)
    ] + [_candidate(f"ski-{index:02d}", "skiing") for index in range(20)]

    assigned = assign_stage7_splits(candidates, seed=SPLIT_SEED)

    for category in ("dog", "skiing"):
        roles = [entry.stage7_split for entry in assigned if entry.candidate.category == category]
        assert roles.count("TRAIN") == 14
        assert roles.count("VALIDATION") == 3
        assert roles.count("CALIBRATION") == 3


def test_assignment_counts_are_close_to_target_ratio() -> None:
    candidates = [_candidate(f"video-{index:03d}") for index in range(154)]

    audit = audit_assignment(assign_stage7_splits(candidates, seed=SPLIT_SEED))

    assert audit.total == 154
    assert sum(audit.counts.values()) == 154
    assert audit.counts["TRAIN"] >= audit.counts["VALIDATION"]
    assert audit.counts["TRAIN"] >= audit.counts["CALIBRATION"]
    assert 0.65 <= audit.counts["TRAIN"] / 154 <= 0.75


def test_audit_reports_zero_leakage_for_valid_assignment() -> None:
    candidates = [_candidate(f"video-{index:03d}") for index in range(40)]

    audit = audit_assignment(assign_stage7_splits(candidates, seed=SPLIT_SEED))

    assert audit.source_id_leakage == 0
    assert audit.official_test_rows == 0
    assert audit.tvsum_rows == 0
    assert audit.non_alignment_pass_rows == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _materialize_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "youtube_highlights"
    alignment = []
    raw = []
    availability = []
    for index in range(6):
        video_id = f"video-{index:02d}"
        category = "dog" if index % 2 == 0 else "skiing"
        relative = f"raw/{category}/{video_id}.mp4"
        alignment.append(
            {
                "canonical_video_id": video_id,
                "actual_source_id": video_id,
                "category": category,
                "official_split": "train",
                "ALIGNMENT_PASS": True,
            }
        )
        raw.append(
            {
                "canonical_video_id": video_id,
                "relative_path": relative,
                "download_status": "OK",
                "official_set": "TraiLooseL",
                "sha256": "b" * 64,
            }
        )
        availability.append(
            {
                "canonical_video_id": video_id,
                "original_video_id": video_id,
                "replacement_video_id": None,
            }
        )
    alignment.append(
        {
            "canonical_video_id": "locked-test",
            "actual_source_id": "locked-test",
            "category": "dog",
            "official_split": "test",
            "ALIGNMENT_PASS": True,
        }
    )
    alignment.append(
        {
            "canonical_video_id": "bad-align",
            "actual_source_id": "bad-align",
            "category": "dog",
            "official_split": "train",
            "ALIGNMENT_PASS": False,
        }
    )
    _write_jsonl(dataset / "manifests" / "alignment_manifest.jsonl", alignment)
    _write_jsonl(dataset / "manifests" / "raw_manifest.jsonl", raw)
    _write_jsonl(dataset / "manifests" / "availability_manifest.jsonl", availability)
    return tmp_path


def test_loader_keeps_only_official_train_alignment_pass_rows(tmp_path: Path) -> None:
    root = _materialize_dataset(tmp_path)

    candidates, hashes = load_ftnet_split_candidates(root)

    assert {candidate.canonical_video_id for candidate in candidates} == {
        f"video-{index:02d}" for index in range(6)
    }
    assert all(candidate.upstream_official_split == "train" for candidate in candidates)
    assert all(candidate.alignment_pass for candidate in candidates)
    assert set(hashes) == {
        "alignment_manifest_sha256",
        "raw_manifest_sha256",
        "availability_manifest_sha256",
    }


def test_loader_verify_files_fails_closed_when_video_missing(tmp_path: Path) -> None:
    root = _materialize_dataset(tmp_path)

    with pytest.raises(Exception, match="local video file is missing"):
        load_ftnet_split_candidates(root, verify_files=True)


def _manifest_payload(candidates: list[SplitCandidate]) -> dict:
    assigned = assign_stage7_splits(candidates, seed=SPLIT_SEED)
    audit = audit_assignment(assigned)
    return build_split_manifest(
        assigned,
        audit=audit,
        dataset_hashes={"raw_manifest_sha256": "c" * 64},
        upstream_source={"repo_commit": "deadbeef"},
        git_head="f" * 40,
        generated_at="2026-09-17T00:00:00+00:00",
        determinism=determinism_check(candidates, seed=SPLIT_SEED),
    )


def test_manifest_is_deterministic_and_reproducible() -> None:
    candidates = [_candidate(f"video-{index:03d}") for index in range(30)]

    first = _manifest_payload(candidates)
    second = _manifest_payload(list(reversed(candidates)))

    assert content_sha256(first) == content_sha256(second)
    assert first["entries"] == second["entries"]
    assert first["determinism"] is True


def test_manifest_sha_tracks_content_changes() -> None:
    candidates = [_candidate(f"video-{index:03d}") for index in range(12)]

    baseline = _manifest_payload(candidates)
    changed = _manifest_payload(
        candidates + [_candidate("video-new", "skiing")]
    )

    assert manifest_sha256(baseline) != manifest_sha256(changed)
    assert content_sha256(baseline) != content_sha256(changed)


def test_manifest_refuses_source_leakage() -> None:
    candidates = [_candidate(f"video-{index:03d}") for index in range(10)]
    assigned = assign_stage7_splits(candidates, seed=SPLIT_SEED)
    audit = replace(audit_assignment(assigned), source_id_leakage=1)

    with pytest.raises(SplitProtocolError, match="leakage"):
        build_split_manifest(
            assigned,
            audit=audit,
            dataset_hashes={},
            upstream_source={},
            git_head="f" * 40,
        )


def test_manifest_uses_portable_relative_video_paths() -> None:
    candidates = [_candidate("video-portable", "dog")]

    payload = _manifest_payload(candidates)

    for entry in payload["entries"]:
        assert not Path(entry["relative_video_path"]).is_absolute()
        assert ":" not in entry["relative_video_path"]


def test_manifest_entries_cover_every_split_role() -> None:
    candidates = [
        _candidate(f"{category}-{index:03d}", category)
        for category in ["dog", "skiing", "surfing", "parkour"]
        for index in range(10)
    ]

    payload = _manifest_payload(candidates)

    assert {entry["stage7_split"] for entry in payload["entries"]} == set(SPLIT_ROLES)
    assert payload["total"] == len(candidates)
