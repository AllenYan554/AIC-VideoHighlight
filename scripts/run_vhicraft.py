#!/usr/bin/env python3
"""VHiCraft-v1 single runtime entrypoint.

Only this script is an executable entrypoint.  Every runtime variation
(smoke, Dev166, future official-test) is selected through ``infer --profile``
plus ``--run-id``; the pipeline modules live in the ``aic_video_highlight``
package and are never invoked by ad-hoc runner scripts.

Modes
-----
``infer``
    Run the full release pipeline: retrieval -> temporal refinement -> frame
    projection -> subject localization -> spatial composition -> temporal
    stabilization -> prediction selection -> official predictions.jsonl.

``assemble``
    Assemble official predictions from saved stabilization shards without any
    model call (``--stabilization stabilized`` or ``raw`` for the internal
    unstabilized control).

``validate``
    Re-check an existing predictions.jsonl against the official contract.

``evaluate``
    Score Dev predictions with the published exact-frame spatial-IoU formula.
    Results are always labelled weak-reference (``NOT_OFFICIAL_SCORE``).

``report``
    (Re)generate tables, figures and a markdown report from a finished run's
    own artifacts.  ``infer`` already generates these automatically.

``preflight``
    Read-only readiness check for an inference profile (inputs, models, output
    destination, contract and reporting pipelines).  No model is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from aic_video_highlight.composition.submission import write_predictions_jsonl
from aic_video_highlight.composition.vhicraft_pipeline import (
    FrameCrop,
    TARGET_RATIO,
    assemble_prediction_lines,
    load_stabilization_shard,
)
from aic_video_highlight.evaluation.contract import validate_contract
from aic_video_highlight.evaluation.official_like import evaluate_files

DEFAULT_PROFILE_DIR = _REPO / "configs" / "profiles"
DEFAULT_RUNTIME_PROFILE_DIR = _REPO / "configs" / "runtime"
DEFAULT_ENVIRONMENT = _REPO / "configs" / "environments" / "windows_local.json"
ASSEMBLE_STABILIZATION = {"stabilized": "ts5", "raw": "ts0"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_profile(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    return (DEFAULT_PROFILE_DIR / f"{value}.json").resolve()


def _resolve_runtime_profile(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    return (DEFAULT_RUNTIME_PROFILE_DIR / f"{value}.json").resolve()


def assemble(
    *,
    shards_dir: Path,
    role_manifest: Path,
    metadata_cache: Path,
    output: Path,
    stabilization: str,
    target_ratio,
    limit: int | None,
    validate: bool,
    report_path: Path | None,
) -> dict:
    geometry_key = ASSEMBLE_STABILIZATION[stabilization]
    role = _read_json(role_manifest)
    video_ids = [item["video_id"] for item in role["records"]]
    if limit:
        video_ids = video_ids[:limit]
    metadata = _read_json(metadata_cache)["records"]

    crops_by_video: dict[str, dict[int, FrameCrop]] = {}
    for video_id in video_ids:
        shard_path = shards_dir / f"{video_id}.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing stabilization shard: {shard_path}")
        shard = load_stabilization_shard(shard_path)
        crops = {
            frame: entry[geometry_key]
            for frame, entry in shard.items()
            if geometry_key in entry
        }
        if not crops:
            raise ValueError(f"shard has no {geometry_key} geometry: {shard_path}")
        crops_by_video[video_id] = crops

    frame_size = {
        vid: (int(metadata[vid]["width"]), int(metadata[vid]["height"]))
        for vid in crops_by_video
    }
    frame_count = {vid: int(metadata[vid]["frame_count"]) for vid in crops_by_video}
    lines = assemble_prediction_lines(
        crops_by_video,
        target_ratio=target_ratio,
        frame_size_by_video=frame_size,
        frame_count_by_video=frame_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_jsonl(lines, output)

    result = {
        "stabilization": stabilization,
        "geometry_key": geometry_key,
        "video_count": len(lines),
        "prediction_count": sum(len(line["predictions"]) for line in lines),
        "output": str(output),
    }
    if validate:
        scoped_index = {vid: list(target_ratio) for vid in crops_by_video}
        scoped_meta = {
            vid: {
                "width": int(metadata[vid]["width"]),
                "height": int(metadata[vid]["height"]),
                "frame_count": int(metadata[vid]["frame_count"]),
            }
            for vid in crops_by_video
        }
        contract = validate_contract(output)
        from aic_video_highlight.composition.validation import validate_submission_file

        deeper = validate_submission_file(output, index=scoped_index, metadata=scoped_meta)
        result["contract_valid"] = contract["is_valid"] and deeper.is_valid
        result["contract_issues"] = contract["issue_count"] + len(deeper.issues)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(
            (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    infer_parser = sub.add_parser("infer", help="run the full release inference pipeline")
    infer_parser.add_argument("--profile", default="dev166", help="profile name or JSON path")
    infer_parser.add_argument(
        "--runtime-profile",
        default="default",
        help="deployment-only runtime profile name or JSON path (default: default)",
    )
    infer_parser.add_argument(
        "--environment", type=Path, default=DEFAULT_ENVIRONMENT, help="environment JSON"
    )
    infer_parser.add_argument("--run-id", required=True, help="immutable run directory name")
    infer_parser.add_argument(
        "--video-ids", default=None, help="optional comma-separated video id filter"
    )
    infer_parser.add_argument("--resume", action="store_true")

    assemble_parser = sub.add_parser(
        "assemble", help="assemble official predictions from saved stabilization shards"
    )
    assemble_parser.add_argument("--shards-dir", type=Path, required=True)
    assemble_parser.add_argument("--role-manifest", type=Path, required=True)
    assemble_parser.add_argument("--metadata-cache", type=Path, required=True)
    assemble_parser.add_argument("--output", type=Path, required=True)
    assemble_parser.add_argument(
        "--stabilization", choices=sorted(ASSEMBLE_STABILIZATION), default="stabilized"
    )
    assemble_parser.add_argument("--target-ratio", type=int, nargs=2, default=list(TARGET_RATIO))
    assemble_parser.add_argument("--limit", type=int, default=None)
    assemble_parser.add_argument("--validate", action="store_true")
    assemble_parser.add_argument("--report", type=Path, default=None)

    validate_parser = sub.add_parser("validate", help="validate an existing predictions JSONL")
    validate_parser.add_argument("--predictions", type=Path, required=True)
    validate_parser.add_argument("--role-manifest", type=Path, default=None)
    validate_parser.add_argument("--metadata-cache", type=Path, default=None)
    validate_parser.add_argument("--report", type=Path, default=None)

    evaluate_parser = sub.add_parser(
        "evaluate", help="run official-like weak-reference evaluation"
    )
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--references", type=Path, required=True)
    evaluate_parser.add_argument(
        "--video-ids", default=None, help="optional comma-separated evaluation scope"
    )
    evaluate_parser.add_argument("--report", type=Path, default=None)

    report_parser = sub.add_parser(
        "report", help="(re)generate tables, figures and markdown report for a finished run"
    )
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument(
        "--evaluation",
        action="append",
        default=None,
        help="optional weak-reference evaluation as label=path (repeatable)",
    )
    report_parser.add_argument("--summary", type=Path, default=None)

    preflight_parser = sub.add_parser(
        "preflight", help="read-only readiness check for an inference profile"
    )
    preflight_parser.add_argument("--profile", default="official_test")
    preflight_parser.add_argument(
        "--environment", type=Path, default=DEFAULT_ENVIRONMENT, help="environment JSON"
    )
    preflight_parser.add_argument("--run-id", default=None, help="planned run id (must be free)")
    preflight_parser.add_argument("--report", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "infer":
        from aic_video_highlight.runtime.orchestrator import run_inference
        from aic_video_highlight.runtime.paths import EnvironmentPaths

        profile_path = _resolve_profile(args.profile)
        if not profile_path.is_file():
            raise SystemExit(f"profile not found: {profile_path}")
        config = _read_json(profile_path)
        runtime_profile_path = _resolve_runtime_profile(args.runtime_profile)
        if not runtime_profile_path.is_file():
            raise SystemExit(f"runtime profile not found: {runtime_profile_path}")
        from aic_video_highlight.runtime.profiles import load_runtime_profile

        runtime_profile = load_runtime_profile(runtime_profile_path)
        selected = (
            [value.strip() for value in args.video_ids.split(",") if value.strip()]
            if args.video_ids
            else [str(value) for value in config.get("video_ids", [])]
        )
        result = run_inference(
            config,
            EnvironmentPaths.from_json(args.environment.expanduser().resolve()),
            config_path=profile_path,
            protocol_path=_REPO / str(config["protocol"]),
            resume=args.resume,
            only_video_ids=selected or None,
            output_run_id=args.run_id,
            runtime_profile=runtime_profile,
            runtime_profile_path=runtime_profile_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "PASS" else 1

    if args.command == "assemble":
        result = assemble(
            shards_dir=args.shards_dir,
            role_manifest=args.role_manifest,
            metadata_cache=args.metadata_cache,
            output=args.output,
            stabilization=args.stabilization,
            target_ratio=tuple(args.target_ratio),
            limit=args.limit,
            validate=args.validate,
            report_path=args.report,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("contract_valid", True) else 1

    if args.command == "evaluate":
        selected = (
            [value.strip() for value in args.video_ids.split(",") if value.strip()]
            if args.video_ids
            else None
        )
        report = evaluate_files(args.predictions, args.references, video_ids=selected)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_bytes(
                (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report":
        from aic_video_highlight.reporting import render_run_report

        evaluations: dict[str, Path] = {}
        for item in args.evaluation or []:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                raise SystemExit("--evaluation requires label=path")
            evaluations[label] = Path(path)
        rendered = render_run_report(Path(args.run_dir), evaluations=evaluations)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_bytes(
                (json.dumps(rendered, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
        print(json.dumps(rendered, ensure_ascii=False, indent=2))
        return 0

    if args.command == "preflight":
        from aic_video_highlight.runtime.paths import EnvironmentPaths
        from aic_video_highlight.runtime.preflight import run_preflight

        profile_path = _resolve_profile(args.profile)
        if not profile_path.is_file():
            raise SystemExit(f"profile not found: {profile_path}")
        config = _read_json(profile_path)
        environment = EnvironmentPaths.from_json(args.environment.expanduser().resolve())
        result = run_preflight(
            config,
            environment,
            config_path=profile_path,
            protocol_path=_REPO / str(config["protocol"]),
            run_id=args.run_id,
        )
        report_path = args.report or (
            environment.outputs / "vhicraft" / f"preflight_{result['profile']}_{result['timestamp']}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(
            (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "profile",
                        "input_mode",
                        "suggested_run_id",
                        "git_head",
                        "video_count",
                        "official_score_available",
                        "official_score_note",
                        "failed_checks",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"preflight_report={report_path}")
        return 0 if result["status"] == "PASS" else 1

    report = validate_contract(
        args.predictions,
        role_manifest=args.role_manifest,
        metadata_cache=args.metadata_cache,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
    print(json.dumps({k: report[k] for k in (
        "line_count", "unique_video_count", "duplicate_video_count", "non_finite_count",
        "ordering_violations", "issue_count", "is_valid")}, indent=2))
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
