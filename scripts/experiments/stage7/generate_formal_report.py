#!/usr/bin/env python3
"""Generate Stage 7.1 report/audit/figures from existing artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.formal_reporting import generate_formal_deliverables  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=REPO_ROOT / "configs" / "models" / "ftnet_reference.yaml",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    status = generate_formal_deliverables(
        args.formal_dir,
        data_root=args.data_root,
        repo_root=REPO_ROOT,
        index_path=args.index_path,
        training_config_path=args.training_config,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
