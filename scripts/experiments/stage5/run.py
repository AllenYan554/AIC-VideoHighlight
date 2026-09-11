#!/usr/bin/env python3
"""Canonical Stage 5 launcher.

Registered experiments:
- stage5_infra_tiny_fake  : CPU infrastructure acceptance (synthetic)
- stage5_3_smoke          : CPU development smoke for target-ratio composition (CMP-0 vs CMP-1)
- stage5_3_formal         : FORMAL composition baseline (registered, DEFAULT NOT RUN)
- stage5_4_manifest       : deterministic model-blind Stage 5.4 smoke manifest builder (AutoDL, CPU)
- stage5_4_amendment4_revised_smoke/formal: FINAL_FROZEN TS-5 Revised
  (projected_state_canonical_center_ema_v1); Formal requires the exact immutable
  revised Smoke promotion marker
- stage5_5_dev_formal     : Stage 5.5 frame-level calibration Dev166 (FS-0/FS-1/FS-2)
- stage5_5_hard_confirmation: conditional Hard229 (not run; Dev had no winner)
- stage5_6_vhicraft_smoke / stage5_6_vhicraft_formal / stage5_6_vhicraft_ablation:
  VHiCraft-v1 validation & final freeze (VC-0 cached replay + true-fresh
  VC-1/VC-A0 shared pipeline; Full Formal DEFAULT NOT RUN)

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
    "stage5_4_amendment4_revised_smoke": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_4_amendment4_revised_formal": "scripts/experiments/stage5/run_stage5_4_temporal.py",
    "stage5_5_dev_formal": "scripts/experiments/stage5/run_stage5_5_frame_calibration.py",
    "stage5_5_hard_confirmation": "scripts/experiments/stage5/run_stage5_5_frame_calibration.py",
    "stage5_6_vhicraft_smoke": "scripts/experiments/stage5/run_stage5_6_vhicraft.py",
    "stage5_6_vhicraft_formal": "scripts/experiments/stage5/run_stage5_6_vhicraft.py",
    "stage5_6_vhicraft_ablation": "scripts/experiments/stage5/run_stage5_6_vhicraft.py",
}

CONFIGS = {
    "stage5_infra_tiny_fake": "configs/experiments/stage5/stage5_infra_tiny_fake.json",
    "stage5_3_smoke": "configs/experiments/stage5/stage5_3_smoke.json",
    "stage5_3_formal": "configs/experiments/stage5/stage5_3_formal.json",
    "stage5_4_manifest": "configs/experiments/stage5/stage5_4_manifest.json",
    "stage5_4_amendment4_revised_smoke": "configs/experiments/stage5/stage5_4_amendment4_revised_smoke.json",
    "stage5_4_amendment4_revised_formal": "configs/experiments/stage5/stage5_4_amendment4_revised_formal.json",
    "stage5_5_dev_formal": "configs/experiments/stage5/stage5_5_dev_formal.json",
    "stage5_5_hard_confirmation": "configs/experiments/stage5/stage5_5_hard_confirmation.json",
    "stage5_6_vhicraft_smoke": "configs/experiments/stage5/stage5_6_vhicraft_smoke.json",
    "stage5_6_vhicraft_formal": "configs/experiments/stage5/stage5_6_vhicraft_formal.json",
    "stage5_6_vhicraft_ablation": "configs/experiments/stage5/stage5_6_vhicraft_ablation.json",
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
    "stage5_4_amendment4_revised_smoke": {
        "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
        "forbid_active_processes": ["vllm", "qwen"],
    },
    "stage5_4_amendment4_revised_formal": {
        "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
        "forbid_active_processes": ["vllm", "qwen"],
    },
    "stage5_5_dev_formal": {
        "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
        "forbid_active_processes": ["vllm", "qwen"],
    },
    "stage5_5_hard_confirmation": {
        "target": "AUTODL", "gpu": "NONE", "strict_git_preflight": True,
        "forbid_active_processes": ["vllm", "qwen"],
    },
    "stage5_6_vhicraft_smoke": {
        "target": "AUTODL", "gpu": "REQUIRED", "strict_git_preflight": True,
    },
    "stage5_6_vhicraft_formal": {
        "target": "AUTODL", "gpu": "REQUIRED", "strict_git_preflight": True,
    },
    "stage5_6_vhicraft_ablation": {
        "target": "AUTODL", "gpu": "REQUIRED", "strict_git_preflight": True,
    },
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
