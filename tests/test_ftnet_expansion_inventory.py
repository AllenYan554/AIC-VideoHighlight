from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO / "scripts" / "experiments" / "stage7" / "youtubehl_expansion_inventory.py"
PREFLIGHT_PATH = REPO / "scripts" / "experiments" / "stage7" / "youtubehl_expansion_preflight.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _load("aic_test_youtubehl_expansion_inventory", INVENTORY_PATH)
preflight = _load("aic_test_youtubehl_expansion_preflight", PREFLIGHT_PATH)


def _candidate(video_id: str, official_split: str = "train", category: str = "dog") -> dict:
    return {
        "canonical_video_id": video_id,
        "category": category,
        "official_split": official_split,
        "official_set": "TraiLooseL" if official_split == "train" else "TestLooseL",
        "has_mturk_annotation": True,
    }


def test_classify_buckets_cover_all_train_candidates() -> None:
    candidates = [
        _candidate("t-frozen"),
        _candidate("t-aligned"),
        _candidate("t-fail", category="surfing"),
        _candidate("t-avail"),
        _candidate("t-signin"),
        _candidate("t-unavail"),
        _candidate("x-test", official_split="test"),
    ]
    availability = {
        "t-frozen": {"status": "AVAILABLE"},
        "t-aligned": {"status": "AVAILABLE"},
        "t-fail": {"status": "AVAILABLE"},
        "t-avail": {"status": "UNAVAILABLE"},
        "t-signin": {"status": "ERROR"},
        "t-unavail": {"status": "UNAVAILABLE"},
        "x-test": {"status": "AVAILABLE"},
    }
    raw_ids = {"t-frozen", "t-aligned", "t-fail"}
    alignment = {
        "t-frozen": {"ALIGNMENT_PASS": True, "official_split": "train"},
        "t-aligned": {"ALIGNMENT_PASS": True, "official_split": "train"},
        "t-fail": {
            "ALIGNMENT_PASS": False,
            "official_split": "train",
            "frame_range_overshoot_frames": 60.0,
        },
    }
    frozen_ids = {"t-frozen"}
    recheck = {
        "t-avail": {"status": "AVAILABLE"},
        "t-signin": {"status": "SIGN_IN_REQUIRED"},
    }

    buckets = inventory.classify_train_candidates(
        candidates, availability, raw_ids, alignment, frozen_ids, recheck
    )

    assert [row["canonical_video_id"] for row in buckets["used_in_frozen_154"]] == [
        "t-aligned",
        "t-frozen",
    ]
    assert [row["canonical_video_id"] for row in buckets["alignment_failed"]] == ["t-fail"]
    assert [row["canonical_video_id"] for row in buckets["missing_eligible"]] == ["t-avail"]
    assert [row["canonical_video_id"] for row in buckets["sign_in_required"]] == ["t-signin"]
    assert [row["canonical_video_id"] for row in buckets["unavailable"]] == ["t-unavail"]
    flat = {row["canonical_video_id"] for bucket in buckets.values() for row in bucket}
    assert "x-test" not in flat


def test_official_test_never_enters_train_buckets() -> None:
    candidates = [_candidate(f"x-{index}", official_split="test") for index in range(5)]
    buckets = inventory.classify_train_candidates(candidates, {}, set(), {}, set(), {})
    assert all(not bucket for bucket in buckets.values())


def test_alignment_pass_download_without_frozen_membership_is_not_invalid() -> None:
    candidates = [_candidate("t-pass-unused")]
    raw_ids = {"t-pass-unused"}
    alignment = {"t-pass-unused": {"ALIGNMENT_PASS": True, "official_split": "train"}}
    buckets = inventory.classify_train_candidates(candidates, {}, raw_ids, alignment, set(), {})
    assert not buckets["alignment_failed"]
    assert [row["canonical_video_id"] for row in buckets["used_in_frozen_154"]] == [
        "t-pass-unused"
    ]


def test_recheck_scope_selects_only_train_missing_unavailable() -> None:
    candidates = [
        _candidate("t-missing", official_split="train"),
        _candidate("t-picked-err", official_split="train"),
        _candidate("t-present", official_split="train"),
        _candidate("t-available", official_split="train"),
        _candidate("x-test", official_split="test"),
    ]
    availability = {
        "t-missing": {"status": "UNAVAILABLE"},
        "t-picked-err": {"status": "ERROR"},
        "t-present": {"status": "AVAILABLE"},
        "t-available": {"status": "AVAILABLE"},
        "x-test": {"status": "UNAVAILABLE"},
    }
    raw_ids = {"t-present"}
    scope = preflight.select_recheck_scope(candidates, availability, raw_ids)
    assert [row["canonical_video_id"] for row in scope] == ["t-missing", "t-picked-err"]


def test_classify_ytdlp_failure_markers() -> None:
    assert (
        preflight.classify_ytdlp_failure("ERROR: [youtube] abc: Please sign in. Use --cookies")
        == "SIGN_IN_REQUIRED"
    )
    assert (
        preflight.classify_ytdlp_failure("ERROR: [youtube] abc: Private video")
        == "UNAVAILABLE"
    )
    assert (
        preflight.classify_ytdlp_failure("ERROR: [youtube] abc: This video is not available")
        == "UNAVAILABLE"
    )
    assert preflight.classify_ytdlp_failure("ERROR: unable to download webpage") == "ERROR"


def test_source_leakage_detects_group_collision() -> None:
    missing = [
        {"canonical_video_id": "fresh-1", "category": "dog"},
        {"canonical_video_id": "replacement-of-frozen", "category": "parkour"},
    ]
    frozen = [
        {"canonical_video_id": "frozen-1", "canonical_source_group": "frozen-1"},
        {
            "canonical_video_id": "jODvtKkQrNc",
            "canonical_source_group": "jODvtKkQrNc",
            "replacement_video_id": "hiqlZPsR8Mo",
        },
    ]
    missing[1]["canonical_video_id"] = "hiqlZPsR8Mo"
    leakage = inventory.compute_source_leakage(missing, frozen)
    assert [row["canonical_video_id"] for row in leakage] == ["hiqlZPsR8Mo"]


def test_scan_out_of_pool_train_counts_vlist_minus_sel(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream_repo"
    category = upstream / "dog"
    category.mkdir(parents=True)
    (category / "vlist.json").write_text(
        json.dumps(
            [
                [["sel-1", "extra-tight"], "TraiTightL"],
                [["sel-2", "extra-loose"], "TraiLooseL"],
                [["test-1"], "TestLooseL"],
            ]
        ),
        encoding="utf-8",
    )
    (category / "vlist_sel.json").write_text(
        json.dumps([[["sel-1", "sel-2"], "TraiTightL"]]), encoding="utf-8"
    )
    result = inventory.scan_out_of_pool_train(upstream)
    assert result == {"total": 2, "tight": 1, "loose": 1, "with_mturk_label": 0}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def test_inventory_main_end_to_end_and_frozen_valid_cal_guard(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "youtube_highlights"
    manifests = dataset_root / "manifests"
    manifests.mkdir(parents=True)
    (dataset_root / "annotations" / "upstream_repo").mkdir(parents=True)
    derived_root = tmp_path / "derived" / "youtube_highlights_ftnet"
    (derived_root / "manifests").mkdir(parents=True)
    for split, video in (("train", "tr-1"), ("validation", "va-1"), ("calibration", "ca-1")):
        (derived_root / split).mkdir(parents=True)
        (derived_root / split / f"{video}.safetensors").write_bytes(b"stub")

    candidates = [
        _candidate("tr-1"),
        _candidate("tr-missing"),
        _candidate("x-1", official_split="test"),
    ]
    _write_jsonl(manifests / "candidate_manifest.jsonl", candidates)
    _write_jsonl(
        manifests / "availability_manifest.jsonl",
        [
            {"canonical_video_id": "tr-1", "status": "AVAILABLE"},
            {"canonical_video_id": "tr-missing", "status": "UNAVAILABLE"},
            {"canonical_video_id": "x-1", "status": "AVAILABLE"},
        ],
    )
    _write_jsonl(
        manifests / "raw_manifest.jsonl",
        [{"canonical_video_id": "tr-1", "official_split": "train"}],
    )
    _write_jsonl(
        manifests / "alignment_manifest.jsonl",
        [{"canonical_video_id": "tr-1", "official_split": "train", "ALIGNMENT_PASS": True}],
    )
    split_manifest = {
        "entries": [
            {"canonical_video_id": "tr-1", "stage7_split": "TRAIN"},
            {"canonical_video_id": "va-1", "stage7_split": "VALIDATION"},
            {"canonical_video_id": "ca-1", "stage7_split": "CALIBRATION"},
        ]
    }
    (manifests / "stage7_ftnet_split_manifest.json").write_text(
        json.dumps(split_manifest), encoding="utf-8"
    )
    (derived_root / "manifests" / "materialized_videos.json").write_text(
        json.dumps({"count": 3}), encoding="utf-8"
    )
    recheck = tmp_path / "recheck.jsonl"
    recheck.write_text("", encoding="utf-8")
    output_json = tmp_path / "audit.json"
    output_table = tmp_path / "audit.md"
    fingerprint = tmp_path / "fingerprint.json"

    code = inventory.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--derived-root",
            str(derived_root),
            "--recheck-jsonl",
            str(recheck),
            "--output-json",
            str(output_json),
            "--output-table",
            str(output_table),
            "--fingerprint-json",
            str(fingerprint),
        ]
    )
    assert code == 0
    audit = json.loads(output_json.read_text(encoding="utf-8"))
    assert audit["classification"]["missing_eligible"] == 0
    assert audit["excluded"]["official_test"] == 1
    assert audit["eligible_human_train_total"] == 2
    assert audit["local_existing"]["derived_identity_ok"] is False
    assert audit["local_existing"]["derived_split_counts"] == {
        "train": 1,
        "validation": 1,
        "calibration": 1,
    }
    assert (fingerprint).is_file()
    assert output_table.is_file()


def test_frozen_derived_counts_must_match_split_manifest(tmp_path: Path) -> None:
    derived = tmp_path / "derived"
    for split, count in (("train", 2), ("validation", 1), ("calibration", 1)):
        directory = derived / split
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{split}-{index}.safetensors").write_bytes(b"x")
    counts = inventory.count_derived_splits(derived)
    assert counts == {"train": 2, "validation": 1, "calibration": 1}
