#!/usr/bin/env python3
"""Run deterministic Stage 4.3 selection without video or model access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_video_highlight.highlight_retrieval.candidate_selection import (
    assess_evaluation_files,
    build_role_manifests,
    evaluate_selection_to_file,
    freeze_dev_parameter_to_file,
    replay_selection_to_jsonl,
    run_protocol_selection_to_file,
    validate_role_manifests,
    validate_selection_result_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4.3 lightweight frozen-candidate KEEP/DROP baselines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    roles = commands.add_parser("build-roles", help="Build frozen 166/229/36 role manifests")
    roles.add_argument("--cache-dir", type=Path, required=True)
    roles.add_argument("--audit-membership", type=Path, required=True)
    roles.add_argument("--output-dir", type=Path, required=True)

    validate_roles = commands.add_parser("validate-roles", help="Validate all role manifests")
    validate_roles.add_argument("--cache-dir", type=Path, required=True)
    validate_roles.add_argument("--role-dir", type=Path, required=True)

    select = commands.add_parser("select", help="Create a canonical selection result")
    select.add_argument("--cache-dir", type=Path, required=True)
    select.add_argument("--role-manifest", type=Path, required=True)
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--selector", choices=("SEL-0", "SEL-1", "SEL-2", "SEL-3"), required=True)
    select.add_argument(
        "--parameter",
        type=float,
        help="Frozen grid value; required only for SEL-1/SEL-2",
    )
    select.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-selection", help="Validate selection integrity")
    validate.add_argument("--cache-dir", type=Path, required=True)
    validate.add_argument("--role-manifest", type=Path, required=True)
    validate.add_argument("--selection-result", type=Path, required=True)

    replay = commands.add_parser("replay", help="Replay selected frozen candidates")
    replay.add_argument("--cache-dir", type=Path, required=True)
    replay.add_argument("--role-manifest", type=Path, required=True)
    replay.add_argument("--selection-result", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Post-selection frozen weak-reference evaluation")
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--role-manifest", type=Path, required=True)
    evaluate.add_argument("--selection-result", type=Path, required=True)
    evaluate.add_argument("--selected-predictions", type=Path, required=True)
    evaluate.add_argument(
        "--frozen-predictions",
        action="append",
        type=Path,
        required=True,
        help="Repeat for a mixed Dev+Hard Audit role",
    )
    evaluate.add_argument("--output", type=Path, required=True)

    assess = commands.add_parser("assess", help="Compare with SEL-0 and apply frozen gates")
    assess.add_argument("--baseline-evaluation", type=Path, required=True)
    assess.add_argument("--candidate-evaluation", type=Path, required=True)
    assess.add_argument("--protocol", type=Path, required=True)
    assess.add_argument("--phase", choices=("dev", "hard"), required=True)
    assess.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser(
        "freeze-dev-parameter",
        help="Choose and freeze one SEL-1/SEL-2 config using Dev-only rules",
    )
    freeze.add_argument("--baseline-evaluation", type=Path, required=True)
    freeze.add_argument(
        "--candidate-evaluation",
        action="append",
        type=Path,
        required=True,
        help="Repeat once for every frozen-grid evaluation in the selector family",
    )
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--selector", choices=("SEL-1", "SEL-2"), required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-roles":
        result = build_role_manifests(args.cache_dir, args.audit_membership, args.output_dir)
    elif args.command == "validate-roles":
        result = validate_role_manifests(args.cache_dir, args.role_dir)
    elif args.command == "select":
        payload = run_protocol_selection_to_file(
            args.cache_dir,
            args.role_manifest,
            args.protocol,
            args.selector,
            args.parameter,
            args.output,
        )
        result = {
            key: payload[key]
            for key in (
                "selector_name",
                "record_count",
                "input_candidate_count",
                "selected_candidate_count",
                "selection_result_semantic_hash",
            )
        }
    elif args.command == "validate-selection":
        result = validate_selection_result_file(
            args.cache_dir, args.role_manifest, args.selection_result
        )
    elif args.command == "replay":
        result = replay_selection_to_jsonl(
            args.cache_dir, args.role_manifest, args.selection_result, args.output
        )
    elif args.command == "evaluate":
        payload = evaluate_selection_to_file(
            args.cache_dir,
            args.role_manifest,
            args.selection_result,
            args.selected_predictions,
            args.frozen_predictions,
            args.output,
        )
        result = {
            "record_count": payload["record_count"],
            "evaluation_semantic_hash": payload["evaluation_semantic_hash"],
            "aggregate": payload["aggregate"],
        }
    elif args.command == "assess":
        payload = assess_evaluation_files(
            args.baseline_evaluation,
            args.candidate_evaluation,
            args.protocol,
            args.phase,
            args.output,
        )
        result = {
            "phase": payload["phase"],
            "pass": payload["assessments"]["pass"],
            "assessment_semantic_sha256": payload["assessment_semantic_sha256"],
        }
    else:
        payload = freeze_dev_parameter_to_file(
            args.baseline_evaluation,
            args.candidate_evaluation,
            args.protocol,
            args.selector,
            args.output,
        )
        result = {
            "selector_name": payload["selector_name"],
            "status": payload["status"],
            "chosen_selector_config": payload["chosen_selector_config"],
            "parameter_freeze_semantic_sha256": payload[
                "parameter_freeze_semantic_sha256"
            ],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
