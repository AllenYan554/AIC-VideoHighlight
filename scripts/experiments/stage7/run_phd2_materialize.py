#!/usr/bin/env python3
"""PHD2 materialization runner (PHD2_GENERIC_PRETRAIN_TARGET_V1).

Subcommands
-----------
materialize : build the FTNet PHD2 pretraining dataset (resume-safe, failure-isolated)
verify      : audit the materialized set (missing / corrupt / duplicate)
status      : print progress.json

Inputs are the frozen staged artifacts only:
    Qwen retrieval  <retrieval-root>/retrieval/<video_id>.json
    RT-DETR         <rtdetr-root>/detections/<video_id>.npz + <rtdetr-root>/index
    TRAIN annotation (cloud copy, frozen SHA)
No label is ever fabricated; videos without a positive selection produce zero
pairs (never a fake negative).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aic_video_highlight.ftnet.index import DECODE_OK, load_index
from aic_video_highlight.ftnet.phd2_qwen1500 import sha256_file
from aic_video_highlight.ftnet.phd2_materialize import (
    MATERIALIZED_SCHEMA,
    materialize_cohort,
    materialized_path,
    validate_materialized,
)
from aic_video_highlight.ftnet.phd2_targets import PAIR_SEED_DEFAULT, PROTOCOL_VERSION

SOURCE_FILES = (
    "src/aic_video_highlight/ftnet/phd2_materialize.py",
    "src/aic_video_highlight/ftnet/phd2_targets.py",
    "src/aic_video_highlight/ftnet/provider_core.py",
    "src/aic_video_highlight/ftnet/materialize.py",
    "src/aic_video_highlight/ftnet/real_provider.py",
)


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def _code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _entries(run_root: Path):
    entries, _meta = load_index(run_root / "index" / "phd2_probed_index.json")
    return [entry for entry in entries if entry.decode_status == DECODE_OK]


def command_materialize(args) -> dict:
    entries = _entries(Path(args.run_root))
    summary = materialize_cohort(
        entries,
        dataset_root=args.dataset_root,
        retrieval_root=args.retrieval_root,
        rtdetr_root=args.run_root,
        run_root=args.output_root or args.run_root,
        annotation_path=args.annotation_path,
        pair_seed=args.pair_seed,
        retry_failed_rounds=args.retry_failed_rounds,
        code_identity=_code_identity(),
    )
    _print(summary)
    return summary


def command_verify(args) -> dict:
    entries = _entries(Path(args.run_root))
    root = Path(args.output_root or args.run_root)
    missing: list[str] = []
    corrupt: list[str] = []
    present = {path.stem for path in (root / "materialized").glob("*.safetensors")} if (root / "materialized").is_dir() else set()
    expected = {entry.video_id for entry in entries}
    missing = sorted(expected - present)
    duplicate = sorted(present - expected)
    for entry in entries:
        path = materialized_path(root, entry.video_id)
        if path.is_file():
            ok, _why = validate_materialized(path, entry)
            if not ok:
                corrupt.append(entry.video_id)
    result = {
        "videos": len(entries),
        "present": len(present & expected),
        "missing": len(missing),
        "missing_ids": missing[:20],
        "duplicate": len(duplicate),
        "duplicate_ids": duplicate[:20],
        "corrupt": len(corrupt),
        "corrupt_ids": corrupt[:20],
        "protocol_version": PROTOCOL_VERSION,
        "schema": MATERIALIZED_SCHEMA,
        "status": "COMPLETE" if not missing and not duplicate and not corrupt else "INCOMPLETE",
    }
    _print(result)
    return result


def command_status(args) -> dict:
    root = Path(args.output_root or args.run_root)
    progress_path = root / "progress.json"
    if not progress_path.is_file():
        raise SystemExit(f"no materialize progress.json at {progress_path}")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    summary_path = root / "summary" / "materialize_summary.json"
    payload["summary_present"] = summary_path.is_file()
    if payload["summary_present"]:
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    _print(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub) -> None:
        sub.add_argument("--run-root", required=True, type=Path, help="RT-DETR run root (index + detections)")
        sub.add_argument("--retrieval-root", required=True, type=Path, help="Qwen retrieval run root")
        sub.add_argument("--dataset-root", required=True, type=Path)
        sub.add_argument("--annotation-path", required=True, type=Path, help="cloud TRAIN annotation CSV")
        sub.add_argument("--output-root", type=Path, default=None)
        sub.add_argument("--pair-seed", type=int, default=PAIR_SEED_DEFAULT)

    materialize = subparsers.add_parser("materialize")
    add_common(materialize)
    materialize.add_argument("--retry-failed-rounds", type=int, default=1)

    verify = subparsers.add_parser("verify")
    add_common(verify)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True, type=Path)
    status.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        command_materialize(args)
        return 0
    if args.command == "verify":
        result = command_verify(args)
        return 0 if result["status"] == "COMPLETE" else 3
    command_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
