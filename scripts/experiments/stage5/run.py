#!/usr/bin/env python3
"""Canonical Stage 5 launcher; Stage 5.3 remains intentionally unregistered."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment", type=Path, default=Path("configs/environments/autodl.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.experiment != "stage5_infra_tiny_fake":
        parser.error(
            f"experiment {args.experiment!r} is not registered; Stage 5.3 has not started and no scientific runner was invented"
        )
    config = args.config or Path("configs/experiments/stage5/stage5_infra_tiny_fake.json")
    command = [sys.executable, "scripts/experiments/run_tiny_fake.py", "--config", str(config), "--environment", str(args.environment)]
    for enabled, flag in ((args.resume, "--resume"), (args.dry_run, "--dry-run"), (args.validate_only, "--validate-only")):
        if enabled:
            command.append(flag)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
