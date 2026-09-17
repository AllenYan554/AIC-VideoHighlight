"""Freeze the Stage 7.1 FTNet TRAIN/VALIDATION/CALIBRATION split manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.path_resolver import (  # noqa: E402
    resolve_dataset_root,
)
from aic_video_highlight.ftnet.split_manifest import (  # noqa: E402
    SPLIT_MANIFEST_NAME,
    YOUTUBE_HIGHLIGHTS_DIR,
    build_split_manifest,
    determinism_check,
    load_ftnet_split_candidates,
    write_split_manifest,
)
from aic_video_highlight.ftnet.split_protocol import (  # noqa: E402
    SPLIT_SEED,
    assign_stage7_splits,
    audit_assignment,
)


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _upstream_source(dataset_dir: Path) -> dict[str, object]:
    path = dataset_dir / "manifests" / "source_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "canonical_name": payload.get("canonical_name"),
        "repo_url_ssh": payload.get("repo_url_ssh"),
        "repo_commit": payload.get("repo_commit"),
        "freeze_time_utc": payload.get("freeze_time_utc"),
    }


def _summary(payload: dict, *, dry_run: bool, output: Path | None) -> dict:
    return {
        "status": "DRY_RUN" if dry_run else "WRITTEN",
        "formal_training_started": False,
        "gpu_used": False,
        "protocol_name": payload["protocol_name"],
        "seed": payload["seed"],
        "ratios": payload["ratios"],
        "TOTAL_REALIZED_OFFICIAL_TRAIN": payload["total"],
        "TRAIN": payload["counts"]["TRAIN"],
        "VALIDATION": payload["counts"]["VALIDATION"],
        "CALIBRATION": payload["counts"]["CALIBRATION"],
        "SOURCE_ID_LEAKAGE": payload["source_id_leakage"],
        "OFFICIAL_TEST_ROWS": payload["official_test_rows"],
        "TVSUM_ROWS": payload["tvsum_rows"],
        "NON_ALIGNMENT_PASS_ROWS": payload["non_alignment_pass_rows"],
        "DETERMINISM_CHECK": "PASS" if payload["determinism"] else "FAIL",
        "CATEGORY_COUNTS": payload["category_counts"],
        "content_sha256": payload["content_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "manifest_path": str(output) if output is not None else None,
    }


def run(
    *,
    data_root: str | None,
    repo_root: Path,
    dry_run: bool,
    output: Path | None,
) -> tuple[dict, dict]:
    dataset_root = resolve_dataset_root(
        cli_data_root=data_root,
        environ=None,
        config_path=repo_root / "configs" / "datasets" / "ftnet_datasets.yaml",
    )
    dataset_dir = dataset_root / YOUTUBE_HIGHLIGHTS_DIR
    candidates, dataset_hashes = load_ftnet_split_candidates(
        dataset_root, verify_files=True
    )
    assigned = assign_stage7_splits(candidates, seed=SPLIT_SEED)
    audit = audit_assignment(assigned)
    deterministic = determinism_check(candidates, seed=SPLIT_SEED)
    payload = build_split_manifest(
        assigned,
        audit=audit,
        dataset_hashes=dataset_hashes,
        upstream_source=_upstream_source(dataset_dir),
        git_head=_git_head(repo_root),
        determinism=deterministic,
    )
    summary = _summary(payload, dry_run=dry_run, output=output)

    if not dry_run:
        target = output or dataset_dir / "manifests" / SPLIT_MANIFEST_NAME
        path, sidecar = write_split_manifest(payload, target)
        summary["manifest_path"] = str(path)
        summary["sha256_sidecar"] = str(sidecar)

    return payload, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), type=Path
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write and args.dry_run:
        print("choose either --dry-run or --write, not both", file=sys.stderr)
        return 2
    _, summary = run(
        data_root=args.data_root,
        repo_root=args.repo_root.resolve(),
        dry_run=not args.write,
        output=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
