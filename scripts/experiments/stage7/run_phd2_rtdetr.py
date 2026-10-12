#!/usr/bin/env python3
"""PHD2 RT-DETR full-cohort production run (detection + Visual256 only).

Subcommands
-----------
cohort       freeze the derived Qwen-success eligibility cohort (COMPLETE_WITH_EXCLUSIONS)
index        build/probe the frozen 2.0 fps index (+ grid expectations) for the cohort
verify-index audit the probed index (decode status / grid identity)
run          detached detection run (resume-safe, failure-isolated, model loaded once)
verify       audit the produced detection artifact set (missing / duplicate / corrupt)
status       print progress.json

Frozen promoted configuration (no sweep is re-run here):
    PekingU/rtdetr_r50vd, full-video uniform 2.0 fps, one_pass monotonic decode, batch 8.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aic_video_highlight.ftnet.index import DECODE_OK, load_index
from aic_video_highlight.ftnet.phd2_qwen1500 import (
    atomic_json,
    load_raw1500_manifest,
    sha256_file,
)
from aic_video_highlight.ftnet.phd2_rtdetr import (
    COHORT_SCHEMA,
    FAILURE_AUDIT_SCHEMA,
    FROZEN_BATCH_SIZE,
    FROZEN_DECODE_MODE,
    FROZEN_TOP_K,
    Phd2RtdetrError,
    audit_detections,
    build_probed_index,
    build_success_cohort,
    cohort_payload,
    entries_from_cohort,
    run_detection_cohort,
)
from aic_video_highlight.ftnet.real_provider import RTDETR_MODEL, RTDETR_REVISION

DEFAULT_RTDETR_SNAPSHOT = "/root/autodl-tmp/models/rtdetr_r50vd"
SOURCE_FILES = (
    "src/aic_video_highlight/ftnet/phd2_rtdetr.py",
    "src/aic_video_highlight/ftnet/phd2_qwen1500.py",
    "src/aic_video_highlight/ftnet/performance_decode.py",
    "src/aic_video_highlight/ftnet/real_provider.py",
    "src/aic_video_highlight/ftnet/index.py",
)


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def _code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _load_cohort(args):
    records, meta = load_raw1500_manifest(
        args.raw_manifest,
        expected_sha256=args.raw_manifest_sha256,
        dataset_root=args.dataset_root,
        verify_file_sizes=not args.skip_size_check,
        expected_records=args.expected_records,
    )
    eligible, excluded = build_success_cohort(records, retrieval_root=args.retrieval_root)
    return records, meta, eligible, excluded


def command_cohort(args) -> dict:
    records, meta, eligible, excluded = _load_cohort(args)
    stamp = time.strftime("%Y%m%d%H%M")
    cohort_dir = Path(args.run_root) / "cohort"
    payload = cohort_payload(
        run_id=args.run_id,
        raw_manifest_meta=meta,
        total_records=len(records),
        eligible=eligible,
        excluded=excluded,
        retrieval_root=args.retrieval_root,
        dataset_root=args.dataset_root,
    )
    cohort_path = atomic_json(cohort_dir / f"phd2_qwen_success_cohort_{stamp}.json", payload)
    audit_path = atomic_json(
        cohort_dir / f"qwen_failure_audit_{stamp}.json",
        {
            "schema": FAILURE_AUDIT_SCHEMA,
            "run_id": args.run_id,
            "source_manifest_sha256": meta.get("manifest_sha256"),
            "excluded_records": len(excluded),
            "failure_rate": payload["failure_rate"],
            "classification": "QWEN_RETRIEVAL_UNAVAILABLE",
            "excluded": excluded,
        },
    )
    result = {
        "cohort_path": str(cohort_path),
        "failure_audit_path": str(audit_path),
        "original_records": payload["original_records"],
        "eligible_records": payload["eligible_records"],
        "excluded_records": payload["excluded_records"],
        "failure_rate": payload["failure_rate"],
        "status": payload["status"],
        "excluded_ids": payload["excluded_ids"],
        "cohort_schema": COHORT_SCHEMA,
    }
    _print(result)
    return result


def command_index(args) -> dict:
    records, meta, eligible, excluded = _load_cohort(args)
    if not eligible:
        raise Phd2RtdetrError("no eligible videos; refusing to build an empty index")
    entries = entries_from_cohort(eligible)
    run_root = Path(args.run_root)
    index_path = run_root / "index" / "phd2_probed_index.json"
    expectations_path = run_root / "index" / "phd2_grid_expectations.json"

    def log(message: str) -> None:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)

    result = build_probed_index(
        entries,
        dataset_root=args.dataset_root,
        index_path=index_path,
        grid_expectations_path=expectations_path,
        verify_sha256=not args.no_verify_sha,
        extract_pts=True,
        ffprobe_bin=args.ffprobe_bin,
        workers=int(args.workers),
        log=log,
    )
    payload = {
        "index_path": str(result.index_path),
        "index_sha256": result.index_sha256,
        "grid_expectations_path": str(result.expectations_path),
        "audit": result.audit,
        "metadata": result.metadata,
        "eligible_records": len(eligible),
        "excluded_records": len(excluded),
        "workers": int(args.workers),
        "status": "PASS" if result.audit.get("all_ok") and result.audit.get("grid_problems") == 0 else "INCOMPLETE",
        "raw_manifest_sha256": meta.get("manifest_sha256"),
    }
    _print(payload)
    return payload


def command_verify_index(args) -> dict:
    index_path = Path(args.run_root) / "index" / "phd2_probed_index.json"
    entries, metadata = load_index(index_path)
    statuses: dict[str, int] = {}
    for entry in entries:
        statuses[entry.decode_status] = statuses.get(entry.decode_status, 0) + 1
    ok = statuses.get(DECODE_OK, 0)
    result = {
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "total": len(entries),
        "decode_status": statuses,
        "all_ok": ok == len(entries),
        "status": "PASS" if ok == len(entries) else "INCOMPLETE",
        "metadata": metadata,
    }
    _print(result)
    return result


def command_run(args) -> dict:
    index_path = Path(args.run_root) / "index" / "phd2_probed_index.json"
    entries, _metadata = load_index(index_path)
    ok_entries = [entry for entry in entries if entry.decode_status == DECODE_OK]
    non_ok = [entry for entry in entries if entry.decode_status != DECODE_OK]
    if non_ok and not args.allow_probe_exclusions:
        raise Phd2RtdetrError(
            f"index has {len(non_ok)} non-OK entries; fix the index or pass "
            f"--allow-probe-exclusions to run the OK cohort and record the exclusions"
        )
    if non_ok:
        atomic_json(
            Path(args.run_root) / "cohort" / "rtdetr_index_exclusions.json",
            {
                "total": len(entries),
                "detected": len(ok_entries),
                "excluded": len(non_ok),
                "classification": "INDEX_PROBE_UNAVAILABLE",
                "excluded": [
                    {
                        "video_id": entry.video_id,
                        "relative_path": entry.relative_video_path,
                        "decode_status": entry.decode_status,
                        "decode_error": entry.decode_error,
                        "timestamp_mode": entry.timestamp_mode,
                    }
                    for entry in non_ok
                ],
            },
        )
    summary = run_detection_cohort(
        ok_entries,
        dataset_root=args.dataset_root,
        run_root=args.run_root,
        run_id=args.run_id,
        rtdetr_snapshot=args.rtdetr_snapshot,
        rtdetr_model_id=args.rtdetr_model_id,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        decode_mode=FROZEN_DECODE_MODE,
        retry_failed_rounds=args.retry_failed_rounds,
        code_identity=_code_identity(),
    )
    return summary


def command_verify(args) -> dict:
    index_path = Path(args.run_root) / "index" / "phd2_probed_index.json"
    entries, _metadata = load_index(index_path)
    entries = [entry for entry in entries if entry.decode_status == DECODE_OK]
    missing, duplicate, corrupt = audit_detections(args.run_root, entries)
    result = {
        "videos": len(entries),
        "artifacts_present": len(entries) - len(missing),
        "missing": len(missing),
        "missing_ids": missing[:20],
        "duplicate": len(duplicate),
        "duplicate_ids": duplicate[:20],
        "corrupt": len(corrupt),
        "corrupt_ids": corrupt[:20],
        "status": "COMPLETE" if not missing and not duplicate and not corrupt else "INCOMPLETE",
    }
    _print(result)
    return result


def command_status(args) -> dict:
    progress_path = Path(args.run_root) / "progress.json"
    if not progress_path.is_file():
        raise Phd2RtdetrError(f"no progress.json at {progress_path}")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    summary_path = Path(args.run_root) / "summary" / "run_summary.json"
    payload["summary_present"] = summary_path.is_file()
    if payload["summary_present"]:
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    _print(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, *, with_run_root: bool = True) -> None:
        sub.add_argument("--raw-manifest", required=True, type=Path)
        sub.add_argument("--raw-manifest-sha256", required=True)
        sub.add_argument("--dataset-root", required=True, type=Path)
        sub.add_argument("--retrieval-root", required=True, type=Path)
        sub.add_argument("--expected-records", type=int, default=1500)
        sub.add_argument("--skip-size-check", action="store_true")
        if with_run_root:
            sub.add_argument("--run-root", required=True, type=Path)

    cohort = subparsers.add_parser("cohort")
    add_common(cohort)
    cohort.add_argument("--run-id", required=True)

    index = subparsers.add_parser("index")
    add_common(index)
    index.add_argument("--no-verify-sha", action="store_true")
    index.add_argument("--workers", type=int, default=1, help="bounded concurrent ffprobe workers (identical per-entry semantics)")
    index.add_argument("--ffprobe-bin", default="ffprobe")

    verify_index = subparsers.add_parser("verify-index")
    verify_index.add_argument("--run-root", required=True, type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--run-root", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--rtdetr-snapshot", default=DEFAULT_RTDETR_SNAPSHOT)
    run.add_argument("--rtdetr-model-id", default=RTDETR_MODEL)
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    run.add_argument("--top-k", type=int, default=FROZEN_TOP_K)
    run.add_argument("--retry-failed-rounds", type=int, default=2)
    run.add_argument(
        "--allow-probe-exclusions",
        action="store_true",
        help="run the OK cohort and record non-OK index entries as INDEX_PROBE_UNAVAILABLE",
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-root", required=True, type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cohort":
        command_cohort(args)
        return 0
    if args.command == "index":
        result = command_index(args)
        return 0 if result["status"] == "PASS" else 3
    if args.command == "verify-index":
        result = command_verify_index(args)
        return 0 if result["status"] == "PASS" else 3
    if args.command == "run":
        command_run(args)
        return 0
    if args.command == "verify":
        result = command_verify(args)
        return 0 if result["status"] == "COMPLETE" else 3
    command_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
