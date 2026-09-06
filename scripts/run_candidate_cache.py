#!/usr/bin/env python3
"""Build and validate the Stage 4.2 frozen candidate cache offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_video_highlight.highlight_retrieval.candidate_cache import (
    ALLOWED_SPLITS,
    CandidateCacheError,
    canonical_json_bytes,
    compare_identity,
    export_candidate_cache,
    replay_cache,
    validate_cache,
)


def _source_mapping(values: list[str] | None) -> dict[str, Path]:
    if not values:
        raise CandidateCacheError("at least one --source SPLIT=EXPERIMENT_DIR is required")
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CandidateCacheError("--source must use SPLIT=EXPERIMENT_DIR")
        split, raw_path = value.split("=", 1)
        split = split.strip().lower()
        if split not in ALLOWED_SPLITS:
            raise CandidateCacheError(
                "--source split must be dev or hard; Heldout is forbidden"
            )
        if split in sources:
            raise CandidateCacheError(f"duplicate --source split: {split}")
        if not raw_path.strip():
            raise CandidateCacheError(f"empty experiment path for split {split}")
        sources[split] = Path(raw_path.strip())
    return sources


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="SPLIT=EXPERIMENT_DIR",
        help=(
            "Frozen source root; repeat for dev/hard. Supports the AutoDL output "
            "layout and archived Stage 3 config/results layout."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4.2 frozen candidate cache and identity replay"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Build a new immutable cache")
    _add_sources(export_parser)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Development smoke only; omit for the formal full export",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate cache integrity")
    validate_parser.add_argument("--cache-dir", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path)

    replay_parser = subparsers.add_parser("replay", help="Replay frozen predictions")
    replay_parser.add_argument("--cache-dir", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare replay semantics with frozen predictions"
    )
    compare_parser.add_argument("--cache-dir", type=Path, required=True)
    _add_sources(compare_parser)
    compare_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _write_report(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"report output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(payload))


def main() -> int:
    args = parse_args()
    if args.command == "export":
        result = export_candidate_cache(
            _source_mapping(args.source),
            output_dir=args.output_dir,
            limit_per_split=args.limit_per_split,
        )
        summary = {
            "record_count": result["record_count"],
            "global_semantic_sha256": result["global_semantic_sha256"],
        }
    elif args.command == "validate":
        summary = validate_cache(args.cache_dir)
        if args.report is not None:
            _write_report(args.report, summary)
    elif args.command == "replay":
        summary = replay_cache(args.cache_dir, args.output)
    else:
        summary = compare_identity(
            args.cache_dir,
            _source_mapping(args.source),
            report_path=args.report,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("identity_match", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
