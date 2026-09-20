#!/usr/bin/env python3
"""Canonical Stage 7 launcher (FTNet data production and training).

Registered experiments:
- ftnet_index_probe        : build/verify the frozen YouTube Highlights index (AutoDL)
- ftnet_small_gate         : 5-video stratified TRAIN gate + frozen idx0 decision (AutoDL)
- ftnet_materialize_full   : retrieval -> detection -> assemble for all 154 videos (AutoDL)
- ftnet_retry_failed       : retry only journaled failures (AutoDL)
- ftnet_normalize          : TRAIN-only normalization stats + SHA256 (AutoDL)
- ftnet_integrity          : 154/154 integrity gate
- ftnet_audit              : per-field native feature audit (TRAIN)
- ftnet_train_smoke        : 2-step CUDA smoke through scripts/train_ftnet.py (AutoDL)
- ftnet_train_formal       : 40 epoch formal FTNet baseline (AutoDL)

Environment auto-selection: configs/environments/windows_local.json on Windows,
configs/environments/autodl.json elsewhere.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

RUNNER = "scripts/experiments/stage7/run_stage7_ftnet.py"
EVAL_RUNNER = "scripts/experiments/stage7/run_ftnet_evaluation.py"

RUNNERS = {
    "ftnet_index_probe": RUNNER,
    "ftnet_small_gate": RUNNER,
    "ftnet_materialize_full": RUNNER,
    "ftnet_retry_failed": RUNNER,
    "ftnet_normalize": RUNNER,
    "ftnet_integrity": RUNNER,
    "ftnet_audit": RUNNER,
    "ftnet_train_smoke": RUNNER,
    "ftnet_train_formal": RUNNER,
    "ftnet_eval_posttraining": EVAL_RUNNER,
}

CONFIGS = {
    "ftnet_index_probe": "configs/experiments/stage7/ftnet_index_probe.json",
    "ftnet_small_gate": "configs/experiments/stage7/ftnet_small_gate.json",
    "ftnet_materialize_full": "configs/experiments/stage7/ftnet_materialize_full.json",
    "ftnet_retry_failed": "configs/experiments/stage7/ftnet_retry_failed.json",
    "ftnet_normalize": "configs/experiments/stage7/ftnet_normalize.json",
    "ftnet_integrity": "configs/experiments/stage7/ftnet_integrity.json",
    "ftnet_audit": "configs/experiments/stage7/ftnet_audit.json",
    "ftnet_train_smoke": "configs/experiments/stage7/ftnet_train_smoke.json",
    "ftnet_train_formal": "configs/experiments/stage7/ftnet_train_formal.json",
    "ftnet_eval_posttraining": "configs/experiments/stage7/ftnet_eval_posttraining.json",
}

# Launch metadata consumed by scripts/experiments/registry.py (single source of
# truth for the unified PowerShell launcher). target: WINDOWS | AUTODL.
# gpu: NONE | OPTIONAL | REQUIRED.
LAUNCH = {
    "ftnet_index_probe": {"target": "AUTODL", "gpu": "NONE"},
    "ftnet_small_gate": {"target": "AUTODL", "gpu": "REQUIRED"},
    "ftnet_materialize_full": {"target": "AUTODL", "gpu": "REQUIRED"},
    "ftnet_retry_failed": {"target": "AUTODL", "gpu": "REQUIRED"},
    "ftnet_normalize": {"target": "AUTODL", "gpu": "NONE"},
    "ftnet_integrity": {"target": "WINDOWS", "gpu": "NONE"},
    "ftnet_audit": {"target": "WINDOWS", "gpu": "NONE"},
    "ftnet_train_smoke": {"target": "AUTODL", "gpu": "REQUIRED"},
    "ftnet_train_formal": {"target": "AUTODL", "gpu": "REQUIRED"},
    "ftnet_eval_posttraining": {"target": "WINDOWS", "gpu": "OPTIONAL"},
}


def default_environment() -> Path:
    override = os.environ.get("AIC_EXPERIMENT_ENVIRONMENT")
    if override:
        return Path(override)
    if platform.system() == "Windows":
        return Path("configs/environments/windows_local.json")
    return Path("configs/environments/autodl.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = args.config or Path(CONFIGS[args.experiment])
    environment = args.environment or default_environment()
    command = [
        sys.executable,
        RUNNERS[args.experiment],
        "--config", str(config),
        "--environment", str(environment),
    ]
    for enabled, flag in ((args.resume, "--resume"), (args.dry_run, "--dry-run"), (args.validate_only, "--validate-only")):
        if enabled:
            command.append(flag)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
