#!/usr/bin/env python3
"""Fail-closed Stage 5.4 Amendment 3 Smoke→Formal executor pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.experiment_runtime.promotion import evaluate_smoke_promotion


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pipeline_contract(config: dict) -> dict:
    """Validate both preregistrations without reading experiment outputs."""
    smoke_config_path = REPO_ROOT / config["smoke"]["config"]
    formal_config_path = REPO_ROOT / config["formal"]["config"]
    smoke = _load(smoke_config_path)
    formal = _load(formal_config_path)
    smoke_protocol_path = REPO_ROOT / smoke["protocol"]
    formal_protocol_path = REPO_ROOT / formal["protocol"]
    smoke_protocol = _load(smoke_protocol_path)
    formal_protocol = _load(formal_protocol_path)
    expected_status = "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT"
    checks = {
        "smoke_protocol_status": smoke_protocol.get("status") == expected_status,
        "formal_protocol_status": formal_protocol.get("status") == expected_status,
        "smoke_protocol_sha": smoke.get("protocol_sha256") == file_sha256(smoke_protocol_path),
        "formal_protocol_sha": formal.get("protocol_sha256") == file_sha256(formal_protocol_path),
        "smoke_config_sha": config["smoke"]["config_sha256"] == file_sha256(smoke_config_path),
        "formal_config_sha": config["formal"]["config_sha256"] == file_sha256(formal_config_path),
        "same_ts4": smoke["temporal_smoothing"]["ts4"] == formal["temporal_smoothing"]["ts4"],
        "same_frozen_gates": smoke["decision_gates"]["temporal_benefit"]
        == formal["decision_gates"]["temporal_benefit"]
        and smoke["decision_gates"]["spatial_regression_guardrails"]
        == formal["decision_gates"]["spatial_regression_guardrails"],
        "promotion_binding": formal["promotion_authorization"]["expected_smoke_config_sha256"]
        == file_sha256(smoke_config_path)
        and formal["promotion_authorization"]["expected_smoke_protocol_sha256"]
        == file_sha256(smoke_protocol_path),
        "override_disabled": formal["promotion_authorization"]["override_allowed"] is False,
    }
    return {"validation": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _stage_command(experiment: str, environment: Path, *flags: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/experiments/stage5/run.py"),
        "--experiment",
        experiment,
        "--environment",
        str(environment),
        *flags,
    ]


def _run_stage(experiment: str, environment: Path, *flags: str) -> int:
    return subprocess.run(
        _stage_command(experiment, environment, *flags), cwd=REPO_ROOT, check=False
    ).returncode


def _promotion_from_output(config: dict, environment: EnvironmentPaths) -> dict:
    smoke_output = environment.outputs / config["smoke"]["output_directory"]
    validation_path = smoke_output / "machine/validation.json"
    config_snapshot = smoke_output / "snapshots/config.json"
    protocol_snapshot = smoke_output / "snapshots/protocol.json"
    if not all(path.is_file() for path in (validation_path, config_snapshot, protocol_snapshot)):
        return evaluate_smoke_promotion({}, identity_ok=False)
    identity_ok = (
        file_sha256(config_snapshot) == config["smoke"]["config_sha256"]
        and file_sha256(protocol_snapshot) == config["smoke"]["protocol_sha256"]
    )
    return evaluate_smoke_promotion(_load(validation_path), identity_ok=identity_ok)


def _resume_flags(requested: bool, output_directory: Path) -> tuple[str, ...]:
    """Forward resume only when there is an actual run to resume."""
    if requested and (output_directory / "run_manifest.json").is_file():
        return ("--resume",)
    return ()


def run(args) -> int:
    config = _load(args.config)
    static = validate_pipeline_contract(config)
    if static["validation"] != "PASS":
        print(json.dumps(static, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    if args.dry_run or args.validate_only:
        payload = {
            **static,
            "action": "NONE",
            "sequence": [
                "validate Smoke", "run/resume Smoke", "evaluate SMOKE_PASS_TO_FORMAL",
                "validate Formal identities", "run/resume Formal", "generate reports", "STOP",
            ],
            "commands": {
                "smoke_validate": _stage_command(config["smoke"]["experiment"], args.environment, "--validate-only"),
                "formal_validate_after_pass": _stage_command(config["formal"]["experiment"], args.environment, "--validate-only"),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    environment = EnvironmentPaths.from_json(args.environment)
    smoke_experiment = config["smoke"]["experiment"]
    formal_experiment = config["formal"]["experiment"]
    existing_promotion = _promotion_from_output(config, environment)
    if not existing_promotion["formal_authorized"]:
        if _run_stage(smoke_experiment, args.environment, "--validate-only") != 0:
            return 2
        smoke_output = environment.outputs / config["smoke"]["output_directory"]
        smoke_flags = _resume_flags(args.resume, smoke_output)
        smoke_exit = _run_stage(smoke_experiment, args.environment, *smoke_flags)
        if smoke_exit != 0:
            print("SMOKE_PASS_TO_FORMAL denied: Smoke did not pass; STOP", file=sys.stderr)
            return smoke_exit
    promotion = _promotion_from_output(config, environment)
    if not promotion["formal_authorized"]:
        print(json.dumps(promotion, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    if _run_stage(formal_experiment, args.environment, "--validate-only") != 0:
        return 2
    formal_output = environment.outputs / config["formal"]["output_directory"]
    formal_flags = _resume_flags(args.resume, formal_output)
    return _run_stage(formal_experiment, args.environment, *formal_flags)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
