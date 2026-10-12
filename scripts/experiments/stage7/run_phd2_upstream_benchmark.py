#!/usr/bin/env python3
"""Run the PHD2 Benchmark32 upstream-only engineering benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
    compare_runs,
    load_benchmark_manifest,
    run_benchmark,
    subset_manifest,
)


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--benchmark-manifest", required=True, type=Path)
    run.add_argument("--benchmark-sha256", required=True)
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--environment", required=True, type=Path)
    run.add_argument("--run-root", required=True, type=Path)
    run.add_argument("--run-id", required=True)
    run.add_argument("--index", type=Path, help="reuse a prepared Benchmark32 index")
    run.add_argument("--stages", nargs="+", choices=("retrieval", "detection"), default=("retrieval", "detection"))
    run.add_argument("--qwen-concurrency", type=int, default=1)
    run.add_argument("--rtdetr-batch-size", type=int, default=8)
    run.add_argument("--decode-mode", choices=("windowed_prefix", "one_pass"), default="windowed_prefix")
    run.add_argument("--verify-raw-sha256", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--video-id", action="append", dest="video_ids")

    subset = subparsers.add_parser("subset")
    subset.add_argument("--benchmark-manifest", required=True, type=Path)
    subset.add_argument("--benchmark-sha256", required=True)
    subset.add_argument("--dataset-root", required=True, type=Path)
    subset.add_argument("--purpose", choices=("qwen_sweep", "rtdetr_sweep"), required=True)
    subset.add_argument("--output", required=True, type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-root", required=True, type=Path)
    compare.add_argument("--optimized-root", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        summary = run_benchmark(
            benchmark_manifest=args.benchmark_manifest,
            benchmark_sha256=args.benchmark_sha256,
            dataset_root=args.dataset_root,
            environment_path=args.environment,
            run_root=args.run_root,
            run_id=args.run_id,
            stages=args.stages,
            qwen_concurrency=args.qwen_concurrency,
            rtdetr_batch_size=args.rtdetr_batch_size,
            decode_mode=args.decode_mode,
            verify_raw_sha256=args.verify_raw_sha256,
            entry_ids=args.video_ids,
            index_path_override=args.index,
            resume=args.resume,
        )
        _print(summary)
        return 0
    if args.command == "subset":
        benchmark = load_benchmark_manifest(
            args.benchmark_manifest,
            expected_sha256=args.benchmark_sha256,
            dataset_root=args.dataset_root,
            verify_raw_sha256=False,
        )
        path, sha256, ids = subset_manifest(
            benchmark,
            purpose=args.purpose,
            output_path=args.output,
            benchmark_sha256=args.benchmark_sha256,
        )
        _print({"path": str(path), "sha256": sha256, "video_ids": ids})
        return 0
    report = compare_runs(args.baseline_root, args.optimized_root, args.output)
    _print(report)
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
