#!/usr/bin/env python3
"""CPU-only experiment-runtime acceptance runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from aic_video_highlight.experiment_runtime.artifacts import build_artifact_manifest
from aic_video_highlight.experiment_runtime.hashing import canonical_sha256
from aic_video_highlight.experiment_runtime.io import atomic_write_json
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.experiment_runtime.progress import ProgressReporter
from aic_video_highlight.experiment_runtime.raw_report import render_raw_report, write_ai_report_inputs
from aic_video_highlight.experiment_runtime.run_context import RunContext, RunIdentityMismatch
from aic_video_highlight.experiment_runtime.shards import ShardStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--interrupt-after", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def execute(args) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_json(args.environment)
    paths = environment.for_experiment(config["stage"], config["experiment_id"])
    protocol = Path(config["protocol"])
    context = RunContext(
        config["experiment_id"], config["stage"], config["run_type"], environment.repo,
        paths, args.config, protocol, config.get("input_hashes", {}), config.get("model", {}),
    )
    if args.dry_run:
        print(json.dumps({"experiment_id": config["experiment_id"], "paths": {k: str(v) for k, v in vars(paths).items()}, "action": "NONE"}, indent=2))
        return 0
    identity_hash = canonical_sha256(context.identity())
    store = ShardStore(paths.output / "shards", identity_hash)
    total = int(config.get("total_shards", 3))
    if args.validate_only:
        ok = all(store.is_complete(f"shard_{index:03d}") for index in range(total))
        print(json.dumps({"validation": "PASS" if ok else "FAIL", "complete_shards": sum(store.is_complete(f'shard_{i:03d}') for i in range(total)), "total": total}))
        return 0 if ok else 2
    try:
        context.start(resume=args.resume)
    except (RunIdentityMismatch, FileExistsError) as exc:
        print(f"EXPERIMENT FAILED\n\nReason:\n{exc}\n\nResume available:\nNO", file=sys.stderr)
        return 2
    reporter = ProgressReporter(config["experiment_id"], total, paths.logs)
    started = time.monotonic()
    completed = 0
    try:
        for index in range(total):
            shard_id = f"shard_{index:03d}"
            if not store.is_complete(shard_id):
                store.write(shard_id, [{"shard": index, "value": index * index}])
            completed += 1
            reporter.update(completed, current_shard=shard_id, current_video=f"fake_{index:03d}")
            if args.interrupt_after == completed:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        reporter.interrupt()
        context.set_status("INTERRUPTED", completed_shards=completed, total_shards=total)
        print(f"\nEXPERIMENT FAILED\n\nReason:\nInterrupted safely\n\nResume available:\nYES\n\nResume command:\nSame command + --resume\n\nLog:\n{paths.logs / 'progress.jsonl'}")
        return 130
    machine = paths.output / "machine"
    machine.mkdir(parents=True, exist_ok=True)
    atomic_write_json(machine / "summary.json", {"schema_version": "aic.machine-summary/v1", "experiment_id": config["experiment_id"], "completed_shards": total})
    atomic_write_json(machine / "metrics.json", {"schema_version": "aic.machine-metrics/v1", "synthetic_records": total})
    atomic_write_json(machine / "runtime.json", {"schema_version": "aic.machine-runtime/v1", "wall_sec": round(time.monotonic() - started, 3), "gpu_calls": 0})
    atomic_write_json(machine / "validation.json", {"schema_version": "aic.machine-validation/v1", "status": "PASS", "complete_shards": total, "errors": 0})
    artifacts = list((paths.output / "shards").glob("*.json"))
    build_artifact_manifest(paths.output, artifacts, machine / "artifact_manifest.json", created_by=config["experiment_id"])
    render_raw_report(machine, paths.output / "experiment_raw_report.md", config["experiment_id"])
    write_ai_report_inputs(paths.output)
    context.set_status("COMPLETED", validation="PASS")
    print(f"\n{'=' * 50}\nEXPERIMENT COMPLETE\n{'=' * 50}\nExperiment:\n{config['experiment_id']}\n\nValidation:\nPASS\n\nRaw report:\n{paths.output / 'experiment_raw_report.md'}\n\nAI report inputs:\n{paths.output / 'AI_REPORT_INPUTS.md'}\n\nNext:\nSend experiment_raw_report.md + machine/ to AI for scientific analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
