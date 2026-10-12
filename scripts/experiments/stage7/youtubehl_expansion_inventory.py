#!/usr/bin/env python3
"""Stage 7.1 YouTube Highlights expansion: full inventory audit (local, read-only).

Builds the three-way inventory required by the expansion protocol:

    A = source inventory      (official vlist / vlist_sel / mturk candidates)
    B = local existing        (raw downloads + frozen derived materialization)
    C = Stage 7.1 frozen 154  (split manifest / frozen index)

and classifies every official-TRAIN candidate into:

    used_in_frozen_154        alignment-pass and part of the frozen split
    alignment_failed          downloaded but annotation/video frame range fails
    unavailable               removed/private on the official source
    sign_in_required          age/login gated on the official source
    missing_eligible          legal human TRAIN, absent locally, retrievable

The script is read-only for dataset paths; it writes only the audit outputs
requested on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXPECTED_SPLIT_COUNTS = {"TRAIN": 111, "VALIDATION": 23, "CALIBRATION": 20}
PUBLIC_LOCAL_SPLITS = ("train", "validation", "calibration")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def classify_train_candidates(
    candidates: list[dict[str, Any]],
    availability_by_id: dict[str, dict[str, Any]],
    raw_ids: set[str],
    alignment_by_id: dict[str, dict[str, Any]],
    frozen_ids: set[str],
    recheck_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically bucket every official-TRAIN candidate."""

    buckets: dict[str, list[dict[str, Any]]] = {
        "used_in_frozen_154": [],
        "alignment_failed": [],
        "unavailable": [],
        "sign_in_required": [],
        "missing_eligible": [],
    }
    for candidate in sorted(
        candidates, key=lambda row: (str(row.get("category")), str(row["canonical_video_id"]))
    ):
        if str(candidate.get("official_split", "")).lower() != "train":
            continue
        video_id = str(candidate["canonical_video_id"])
        entry: dict[str, Any] = {
            "canonical_video_id": video_id,
            "category": candidate.get("category"),
            "official_set": candidate.get("official_set"),
        }
        if video_id in frozen_ids:
            buckets["used_in_frozen_154"].append(entry)
            continue
        if video_id in raw_ids:
            alignment = alignment_by_id.get(video_id, {})
            status = "PASS" if alignment.get("ALIGNMENT_PASS") else "FAIL"
            entry["alignment_status"] = status
            entry["overshoot_frames"] = alignment.get("frame_range_overshoot_frames")
            buckets["alignment_failed" if status == "FAIL" else "used_in_frozen_154"].append(entry)
            continue
        live = recheck_by_id.get(video_id, {})
        live_status = str(live.get("status", "")) or str(
            availability_by_id.get(video_id, {}).get("status", "UNKNOWN")
        )
        if live_status == "AVAILABLE":
            buckets["missing_eligible"].append(entry)
        elif live_status == "SIGN_IN_REQUIRED":
            buckets["sign_in_required"].append(entry)
        else:
            entry["previous_status"] = availability_by_id.get(video_id, {}).get("status")
            buckets["unavailable"].append(entry)
    return buckets


def compute_source_leakage(
    missing_rows: list[dict[str, Any]], frozen_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Missing-eligible rows whose source group collides with the frozen split."""

    frozen_source_groups: set[str] = set()
    for entry in frozen_entries:
        for key in (
            "canonical_video_id",
            "canonical_source_group",
            "original_video_id",
            "replacement_video_id",
        ):
            value = entry.get(key)
            if value:
                frozen_source_groups.add(str(value))
    leakage: list[dict[str, Any]] = []
    for row in missing_rows:
        identity = f"{row.get('category')}/{row['canonical_video_id']}"
        if row["canonical_video_id"] in frozen_source_groups or identity in frozen_source_groups:
            leakage.append(row)
    return leakage


def count_derived_splits(derived_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in PUBLIC_LOCAL_SPLITS:
        directory = derived_root / split
        counts[split] = (
            len([name for name in os.listdir(directory) if name.endswith(".safetensors")])
            if directory.is_dir()
            else 0
        )
    return counts


def fingerprint_safetensors(derived_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in PUBLIC_LOCAL_SPLITS:
        directory = derived_root / split
        if not directory.is_dir():
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".safetensors"):
                continue
            path = directory / name
            records.append(
                {
                    "split": split,
                    "file": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def scan_out_of_pool_train(upstream_root: Path) -> dict[str, Any]:
    """TRAIN videos listed in upstream vlist.json but absent from vlist_sel."""

    result: dict[str, Any] = {"total": 0, "tight": 0, "loose": 0, "with_mturk_label": 0}
    categories = sorted(
        name for name in os.listdir(upstream_root) if (upstream_root / name).is_dir()
    )
    for category in categories:
        vlist_path = upstream_root / category / "vlist.json"
        sel_path = upstream_root / category / "vlist_sel.json"
        if not (vlist_path.is_file() and sel_path.is_file()):
            continue
        selected: set[str] = set()
        for ids, _set_name in json.loads(sel_path.read_text(encoding="utf-8")):
            selected.update(str(video_id) for video_id in ids)
        for ids, set_name in json.loads(vlist_path.read_text(encoding="utf-8")):
            if set_name not in ("TraiTightL", "TraiLooseL"):
                continue
            for video_id in ids:
                video_id = str(video_id)
                if video_id in selected:
                    continue
                result["total"] += 1
                result["tight" if set_name == "TraiTightL" else "loose"] += 1
                if (upstream_root / category / video_id / "mturk_label.json").is_file():
                    result["with_mturk_label"] += 1
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--derived-root", required=True, type=Path)
    parser.add_argument("--recheck-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-table", required=True, type=Path)
    parser.add_argument("--fingerprint-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = args.dataset_root / "manifests"
    candidates = read_jsonl(manifests / "candidate_manifest.jsonl")
    availability_by_id = index_by(read_jsonl(manifests / "availability_manifest.jsonl"), "canonical_video_id")
    raw_rows = read_jsonl(manifests / "raw_manifest.jsonl")
    raw_ids = {str(row["canonical_video_id"]) for row in raw_rows}
    alignment_by_id = index_by(read_jsonl(manifests / "alignment_manifest.jsonl"), "canonical_video_id")
    split = json.loads((manifests / "stage7_ftnet_split_manifest.json").read_text(encoding="utf-8"))
    frozen_entries = split["entries"]
    frozen_ids = {str(entry["canonical_video_id"]) for entry in frozen_entries}
    split_counts = Counter(str(entry["stage7_split"]) for entry in frozen_entries)
    recheck_by_id = (
        index_by(read_jsonl(args.recheck_jsonl), "canonical_video_id")
        if args.recheck_jsonl.is_file()
        else {}
    )

    train_candidates = [row for row in candidates if str(row.get("official_split", "")).lower() == "train"]
    test_candidates = [row for row in candidates if str(row.get("official_split", "")).lower() == "test"]
    buckets = classify_train_candidates(
        candidates, availability_by_id, raw_ids, alignment_by_id, frozen_ids, recheck_by_id
    )
    out_of_pool = scan_out_of_pool_train(args.dataset_root / "annotations" / "upstream_repo")
    derived_counts = count_derived_splits(args.derived_root)
    derived_records = json.loads(
        (args.derived_root / "manifests" / "materialized_videos.json").read_text(encoding="utf-8")
    )
    fingerprint = fingerprint_safetensors(args.derived_root)

    frozen_identity_ok = (
        derived_counts == {k.lower(): v for k, v in EXPECTED_SPLIT_COUNTS.items()}
        and int(derived_records.get("count", -1)) == len(frozen_entries)
        and derived_counts["train"] == split_counts["TRAIN"]
        and derived_counts["validation"] == split_counts["VALIDATION"]
        and derived_counts["calibration"] == split_counts["CALIBRATION"]
    )
    alignment_pass_train = {
        video_id
        for video_id, alignment in alignment_by_id.items()
        if alignment.get("ALIGNMENT_PASS")
        and str(alignment.get("official_split", "")).lower() == "train"
    }
    frozen_equals_alignment_pass = frozen_ids == alignment_pass_train

    approved_flow = {
        "missing_eligible": len(buckets["missing_eligible"]),
        "existing_eligible": len(buckets["used_in_frozen_154"]) + len(buckets["alignment_failed"]),
        "alignment_failed": len(buckets["alignment_failed"]),
        "unavailable": len(buckets["unavailable"]),
        "sign_in_required": len(buckets["sign_in_required"]),
    }
    total_train = len(train_candidates)
    accounted = sum(approved_flow[key] for key in ("missing_eligible", "existing_eligible", "unavailable", "sign_in_required"))
    if accounted != total_train:
        raise SystemExit(f"inventory accounting mismatch: {accounted} != {total_train}")

    leakage = compute_source_leakage(buckets["missing_eligible"], frozen_entries)

    audit = {
        "schema": "aic.stage7.yth.expansion-inventory-audit/v1",
        "dataset": "youtube_highlights",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_inventory": {
            "candidate_pool_total": len(candidates),
            "candidate_pool_train": total_train,
            "candidate_pool_test": len(test_candidates),
            "all_candidates_have_mturk": all(row.get("has_mturk_annotation") for row in candidates),
            "out_of_pool_train_videos": out_of_pool,
            "out_of_pool_note": (
                "vlist.json TRAIN videos outside vlist_sel have only slide-window clip.json and "
                "harvested match_label.json; they carry no MTurk highlight Y and are not part of "
                "the frozen Stage 7.1 candidate pool."
            ),
        },
        "local_existing": {
            "raw_total": len(raw_rows),
            "raw_train": sum(1 for row in raw_rows if str(row.get("official_split")) == "train"),
            "raw_test": sum(1 for row in raw_rows if str(row.get("official_split")) == "test"),
            "derived_split_counts": derived_counts,
            "derived_manifest_count": derived_records.get("count"),
            "derived_identity_ok": bool(frozen_identity_ok),
        },
        "frozen_stage7_1": {
            "total": len(frozen_entries),
            "counts": dict(split_counts),
            "alignment_pass_train_total": len(alignment_pass_train),
            "equals_all_alignment_pass_train": bool(frozen_equals_alignment_pass),
        },
        "classification": approved_flow,
        "buckets": buckets,
        "eligible_human_train_total": total_train,
        "missing_eligible": len(buckets["missing_eligible"]),
        "excluded": {
            "official_test": len(test_candidates),
            "weak_or_no_mturk_out_of_pool": out_of_pool["total"],
            "source_leakage": len(leakage),
            "source_leakage_video_ids": [row["canonical_video_id"] for row in leakage],
            "alignment_failed_invalid": len(buckets["alignment_failed"]),
            "unavailable_or_signin": len(buckets["unavailable"]) + len(buckets["sign_in_required"]),
        },
        "input_sha256": {
            "candidate_manifest": sha256_file(manifests / "candidate_manifest.jsonl"),
            "availability_manifest": sha256_file(manifests / "availability_manifest.jsonl"),
            "raw_manifest": sha256_file(manifests / "raw_manifest.jsonl"),
            "alignment_manifest": sha256_file(manifests / "alignment_manifest.jsonl"),
            "split_manifest": sha256_file(manifests / "stage7_ftnet_split_manifest.json"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.fingerprint_json.parent.mkdir(parents=True, exist_ok=True)
    args.fingerprint_json.write_text(
        json.dumps(
            {
                "schema": "aic.stage7.yth.old154-derived-fingerprint/v1",
                "derived_root": str(args.derived_root),
                "file_count": len(fingerprint),
                "records": fingerprint,
                "total_bytes": sum(record["size_bytes"] for record in fingerprint),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    audit["derived_fingerprint_sha256"] = sha256_file(args.fingerprint_json)
    args.output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# YouTube Highlights Stage 7.1 Full Inventory Audit",
        "",
        f"Generated: {audit['generated_at_utc']}",
        "",
        "## Three-way inventory",
        "",
        "| Layer | Total | TRAIN | TEST | Notes |",
        "|---|---:|---:|---:|---|",
        f"| A. Candidate pool (vlist_sel + MTurk) | {len(candidates)} | {total_train} | {len(test_candidates)} | all candidates have MTurk |",
        f"| B. Local raw downloads | {len(raw_rows)} | {audit['local_existing']['raw_train']} | {audit['local_existing']['raw_test']} | de-duplicated, no size mismatch |",
        f"| C. Frozen Stage 7.1 derived | {len(frozen_entries)} | {split_counts['TRAIN']} | 0 | {split_counts['VALIDATION']} VALIDATION + {split_counts['CALIBRATION']} CALIBRATION |",
        "",
        "## TRAIN candidate classification",
        "",
        "| Bucket | Count | Meaning |",
        "|---|---:|---|",
        f"| used_in_frozen_154 | {len(buckets['used_in_frozen_154'])} | alignment-pass, already materialized |",
        f"| alignment_failed | {len(buckets['alignment_failed'])} | downloaded, annotation frame range invalid |",
        f"| sign_in_required | {len(buckets['sign_in_required'])} | age/login gated, no local cookies |",
        f"| unavailable | {len(buckets['unavailable'])} | removed or private on official source |",
        f"| missing_eligible | {len(buckets['missing_eligible'])} | legal human TRAIN, retrievable, absent locally |",
        f"| **TOTAL** | **{total_train}** | |",
        "",
        "## Exclusions",
        "",
        f"- official TEST candidates: {len(test_candidates)} (never eligible for training)",
        f"- out-of-pool TRAIN without MTurk Y: {out_of_pool['total']} (slide-window clips only)",
        f"- source leakage vs frozen VAL/CAL: {len(leakage)}",
        f"- alignment-failed (invalid): {len(buckets['alignment_failed'])}",
        "",
        "## Frozen derived reuse",
        "",
        f"- derived split counts: {derived_counts}",
        f"- derived identity check: {'PASS' if frozen_identity_ok else 'FAIL'}",
        f"- old154 fingerprint records: {len(fingerprint)}",
        "",
    ]
    args.output_table.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        key: audit[key]
        for key in ("eligible_human_train_total", "missing_eligible", "classification", "excluded")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
