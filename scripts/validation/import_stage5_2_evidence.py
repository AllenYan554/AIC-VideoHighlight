#!/usr/bin/env python3
"""Read-only compatibility import for existing Stage 5.2 Full Dev evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_video_highlight.experiment_runtime.artifacts import build_artifact_manifest
from aic_video_highlight.experiment_runtime.io import atomic_write_json
from aic_video_highlight.experiment_runtime.raw_report import render_raw_report, write_ai_report_inputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source and output must be separate; frozen source is read-only")
    required = {
        "summary": source / "diagnostics/full_dev_summary.json",
        "metrics": source / "diagnostics/full_dev_audit.json",
        "validation": source / "diagnostics/deterministic_audit_result.json",
        "runtime": source / "diagnostics/full_dev_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        parser.error("missing Stage 5.2 evidence: " + ", ".join(missing))
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        parser.error("output directory must be absent or contain no files")
    machine = output / "machine"
    machine.mkdir(parents=True, exist_ok=True)
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    audit = json.loads(required["metrics"].read_text(encoding="utf-8"))
    deterministic = json.loads(required["validation"].read_text(encoding="utf-8"))
    atomic_write_json(machine / "summary.json", {
        "schema_version": "aic.machine-summary/v1", "source_mode": "READ_ONLY_COMPATIBILITY_IMPORT",
        "videos": summary["videos"], "frames": summary["frames"], "semantic_hashes": summary["semantic_hashes"],
    })
    atomic_write_json(machine / "metrics.json", {"schema_version": "aic.machine-metrics/v1", **audit})
    atomic_write_json(machine / "runtime.json", {
        "schema_version": "aic.machine-runtime/v1", "total_wall_sec": summary["total_wall_sec"],
        "fps_overall": summary["fps_overall"], "latency_inference_ms": summary["latency_inference_ms"],
        "gpu_reexecution": False,
    })
    valid = summary["frames"] == 51256 and not any(summary[key] for key in ("missing", "extra", "duplicates"))
    atomic_write_json(machine / "validation.json", {
        "schema_version": "aic.machine-validation/v1", "status": "PASS" if valid else "FAIL",
        "frames": summary["frames"], "missing": summary["missing"], "extra": summary["extra"],
        "duplicates": summary["duplicates"], "deterministic_audit": deterministic,
    })
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    build_artifact_manifest(source, source_files, machine / "artifact_manifest.json", created_by="stage5_2_read_only_compatibility_import")
    render_raw_report(machine, output / "experiment_raw_report.md", "stage5_2_full_dev_compatibility_import")
    write_ai_report_inputs(output)
    print(json.dumps({"status": "PASS" if valid else "FAIL", "source_files": len(source_files), "gpu_calls": 0, "output": str(output)}, ensure_ascii=False))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
