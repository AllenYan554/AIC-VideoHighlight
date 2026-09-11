#!/usr/bin/env python3
"""Stable experiment launch-spec interface.

Single source of truth remains the canonical stage launchers
(`scripts/experiments/<stage>/run.py`: RUNNERS / CONFIGS / LAUNCH). This module
aggregates them and emits machine-readable launch specs so the unified Windows
PowerShell launcher (`scripts/experiments/launch_experiment.ps1`) never keeps
its own experiment list. Registering a new experiment means extending the stage
launcher registry only; this file and the PowerShell launcher stay untouched.

CLI:
  python scripts/experiments/registry.py list
  python scripts/experiments/registry.py describe --experiment <name>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "aic.experiment-launch-spec/v1"
SCRIPTS_EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_EXPERIMENTS_DIR.parent.parent

STAGE_LAUNCHERS = {
    "stage5": SCRIPTS_EXPERIMENTS_DIR / "stage5" / "run.py",
}

VALID_TARGETS = {"WINDOWS", "AUTODL"}
VALID_GPU = {"NONE", "OPTIONAL", "REQUIRED"}

ENVIRONMENT_CONFIGS = {
    "WINDOWS": "configs/environments/windows_local.json",
    "AUTODL": "configs/environments/autodl.json",
}


def _load_stage_module(stage: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"aic_stage_launcher_{stage}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_stage_registries():
    for stage, path in sorted(STAGE_LAUNCHERS.items()):
        module = _load_stage_module(stage, path)
        runners = getattr(module, "RUNNERS", None)
        configs = getattr(module, "CONFIGS", None)
        launch = getattr(module, "LAUNCH", None)
        if runners is None:
            raise RuntimeError(f"stage launcher missing RUNNERS registry: {path}")
        if launch is None:
            raise RuntimeError(f"stage launcher missing LAUNCH metadata: {path}")
        missing = sorted(set(runners) - set(launch))
        if missing:
            raise RuntimeError(f"experiments missing LAUNCH metadata: {', '.join(missing)}")
        orphans = sorted(set(launch) - set(runners))
        if orphans:
            raise RuntimeError(f"LAUNCH metadata without registered runner: {', '.join(orphans)}")
        yield stage, path, runners, configs, launch


def _validate_metadata(experiment: str, meta: dict) -> None:
    if meta.get("target") not in VALID_TARGETS:
        raise RuntimeError(f"{experiment}: invalid launch target: {meta.get('target')!r}")
    if meta.get("gpu") not in VALID_GPU:
        raise RuntimeError(f"{experiment}: invalid gpu requirement: {meta.get('gpu')!r}")


def remote_repo() -> str | None:
    environment_path = REPO_ROOT / ENVIRONMENT_CONFIGS["AUTODL"]
    if not environment_path.exists():
        return None
    data = json.loads(environment_path.read_text(encoding="utf-8"))
    return data.get("repo")


def list_experiments() -> list[dict]:
    entries = []
    for stage, _path, runners, _configs, launch in _iter_stage_registries():
        for name in sorted(runners):
            _validate_metadata(name, launch[name])
            entries.append(
                {
                    "experiment": name,
                    "stage": stage,
                    "target": launch[name]["target"],
                    "gpu": launch[name]["gpu"],
                }
            )
    return entries


def describe(experiment: str) -> dict:
    for stage, path, runners, configs, launch in _iter_stage_registries():
        if experiment not in runners:
            continue
        meta = launch[experiment]
        _validate_metadata(experiment, meta)
        spec = {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment,
            "stage": stage,
            "target": meta["target"],
            "gpu": meta["gpu"],
            "stage_launcher": path.relative_to(REPO_ROOT).as_posix(),
            "config": configs.get(experiment) if configs else None,
            "environment_config": ENVIRONMENT_CONFIGS[meta["target"]],
            "remote_repo": remote_repo(),
            "canonical_args": ["--experiment", experiment],
            "supports": {"resume": True, "dry_run": True, "validate_only": True},
        }
        if meta.get("strict_git_preflight"):
            spec["strict_git_preflight"] = True
            spec["forbid_active_processes"] = list(meta.get("forbid_active_processes", []))
        return spec
    known = ", ".join(entry["experiment"] for entry in list_experiments())
    raise KeyError(f"unknown experiment: {experiment} (known experiments: {known})")


def _print_json(payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Experiment launch-spec registry interface.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list all registered experiments with launch metadata")
    describe_parser = commands.add_parser("describe", help="emit the launch spec for one experiment")
    describe_parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            payload = {"schema_version": SCHEMA_VERSION, "experiments": list_experiments()}
        else:
            payload = describe(args.experiment)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 3
    _print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
