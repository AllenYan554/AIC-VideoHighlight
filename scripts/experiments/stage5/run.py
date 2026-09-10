#!/usr/bin/env python3
"""Canonical Stage 5 launcher.

Registered experiments:
- stage5_infra_tiny_fake  : CPU infrastructure acceptance (synthetic)
- stage5_3_smoke          : CPU development smoke for target-ratio composition (CMP-0 vs CMP-1)
- stage5_3_formal         : FORMAL composition baseline (registered, DEFAULT NOT RUN)
- stage5_4_manifest       : deterministic model-blind Stage 5.4 smoke manifest builder (AutoDL, CPU)
- stage5_4_smoke          : Stage 5.4 temporal smoothing development smoke, TS-0 vs TS-1
                            (AutoDL no-card, CPU; requires the frozen manifest SHA pinned
                            in configs/experiments/stage5/stage5_4_smoke.json)
- stage5_4_formal         : preregistered Stage 5.4 Formal over Frozen Dev166
                            plus Confirmatory Dev142 (AutoDL CPU; DEFAULT NOT RUN)
- stage5_4_amendment_smoke: TS-0/TS-1/TS-2 Motion-Adaptive EMA v1 development
                            Smoke24 (AutoDL CPU; explicit future launch only)
- stage5_4_amendment2_smoke: TS-0/TS-1/TS-2/TS-3 constrained-EMA development
                             Smoke24 (AutoDL CPU; explicit future launch only)
- stage5_4_amendment2_formal: PREREGISTERED constrained-EMA Formal, four arms over
                              Frozen Dev166 + Confirmatory Dev142 (AutoDL CPU;
                              DEFAULT NOT RUN)

Environment auto-selection: on Windows the default environment is
configs/environments/windows_local.json, otherwise configs/environments/autodl.json.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

RUNNERS = {
    "stage5_infra_tiny_fake": "scripts/experiments/run_tiny_fake.py",
    "stage5_3_smoke": "scripts/experiments/stage5/run_stage5_3_composition.py",
    "stage5_3_formal": "scripts/experiments/stage5/run_stage5_3_composition.py",
    "stage5_4_manifest": "scripts/experiments/stage5/build_stage5_4_smoke_manifest.py",
    "stage5_4_smoke": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_4_formal": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_4_amendment_smoke": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_4_amendment2_smoke": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_4_amendment2_formal": "scripts/experiments/stage5/run_stage5_4_temporal.py",
}

CONFIGS = {
    "stage5_infra_tiny_fake": "configs/experiments/stage5/stage5_infra_tiny_fake.json",
    "stage5_3_smoke": "configs/experiments/stage5/stage5_3_smoke.json",
    "stage5_3_formal": "configs/experiments/stage5/stage5_3_formal.json",
    "stage5_4_manifest": "configs/experiments/stage5/stage5_4_manifest.json",
    "stage5_4_smoke": "configs/experiments/stage5/stage5_4_smoke.json",
    "stage5_4_formal": "configs/experiments/stage5/stage5_4_formal.json",
    "stage5_4_amendment_smoke": "configs/experiments/stage5/stage5_4_amendment_smoke.json",
    "stage5_4_amendment2_smoke": "configs/experiments/stage5/stage5_4_amendment2_smoke.json",
    "stage5_4_amendment2_formal": "configs/experiments/stage5/stage5_4_amendment2_formal.json",
}

# Launch metadata consumed by scripts/experiments/registry.py (single source of
# truth for the unified PowerShell launcher). target: WINDOWS | AUTODL.
# gpu: NONE | OPTIONAL | REQUIRED. Register new experiments here together
# with RUNNERS/CONFIGS; the PowerShell launcher never needs editing.
LAUNCH = {
    "stage5_infra_tiny_fake": {"target": "WINDOWS", "gpu": "NONE"},
    "stage5_3_smoke": {"target": "WINDOWS", "gpu": "NONE"},
    "stage5_3_formal": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_manifest": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_smoke": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_formal": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_amendment_smoke": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_amendment2_smoke": {"target": "AUTODL", "gpu": "NONE"},
    "stage5_4_amendment2_formal": {"target": "AUTODL", "gpu": "NONE"},
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
