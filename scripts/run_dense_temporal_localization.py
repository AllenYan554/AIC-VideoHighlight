#!/usr/bin/env python3
"""Stage 4.6 DTL-0/DTL-1 development CLI."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from aic_video_highlight.highlight_retrieval.dense_temporal_localization import (
    evaluate_localization_to_file,
    full_cache_dtl0_identity,
    load_cache_payloads,
    load_dense_protocol,
    replay_localization_to_jsonl,
    run_localization_to_file,
    summarize_evaluations_to_file,
    validate_localization_file,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (
    validate_role_manifest_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4.6 dense temporal localization")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("dtl0", "dtl1"):
        run = commands.add_parser(name, help=f"Create a canonical {name.upper()} result")
        run.add_argument("--cache-dir", type=Path, required=True)
        run.add_argument("--role-manifest", type=Path, required=True)
        run.add_argument("--protocol", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--allow-draft-protocol", action="store_true")
        if name == "dtl1":
            run.add_argument("--dataset-manifest", type=Path, required=True)
            run.add_argument("--video-root", type=Path, required=True)
            run.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
            run.add_argument("--model", help="Assertion only; must equal Protocol")
            run.add_argument("--vllm-timeout-sec", type=float, help="Assertion only")
            run.add_argument("--ffmpeg-bin", default="ffmpeg")
            run.add_argument("--work-dir", type=Path)

    validate = commands.add_parser("validate", help="Independently validate a result")
    _common_result_args(validate)

    replay = commands.add_parser("replay", help="Replay frozen fields plus refined edges")
    _common_result_args(replay)
    replay.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Run unchanged weak-reference evaluator")
    _common_result_args(evaluate)
    evaluate.add_argument("--replay", type=Path, required=True)
    evaluate.add_argument("--frozen-predictions", type=Path, action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize", aliases=["assess"], help="Compare DTL-1 with DTL-0 without freezing a gate")
    summarize.add_argument("--baseline-evaluation", type=Path, required=True)
    summarize.add_argument("--candidate-evaluation", type=Path, required=True)
    summarize.add_argument("--protocol", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--allow-draft-protocol", action="store_true")

    identity = commands.add_parser("identity-regression", help="CPU-only full-cache DTL-0 identity check")
    identity.add_argument("--cache-dir", type=Path, required=True)
    identity.add_argument("--role-dir", type=Path, required=True)
    return parser.parse_args()


def _common_result_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--allow-draft-protocol", action="store_true")


def _load_video_paths(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id, relative = row.get("video_id"), row.get("relative_video_path")
        if not isinstance(video_id, str) or not isinstance(relative, str) or video_id in result:
            raise ValueError("dataset manifest contains invalid/duplicate video identity")
        result[video_id] = relative
    return result


def _build_dtl1_runtime(args: argparse.Namespace, protocol: dict):
    """Lazily import video/model dependencies only for an authorized DTL-1 run."""
    from aic_video_highlight.highlight_retrieval.qwen_vllm_client import QwenVLLMClient
    from aic_video_highlight.highlight_retrieval.video_clip import extract_video_clip
    from openai import APIConnectionError, APITimeoutError

    model_config = protocol["localizers"]["DTL-1"]["model"]
    if args.model is not None and args.model != model_config["model_name"]:
        raise SystemExit("--model does not match the Protocol")
    if args.vllm_timeout_sec is not None and args.vllm_timeout_sec != float(model_config["timeout_sec"]):
        raise SystemExit("--vllm-timeout-sec does not match the Protocol")
    client = QwenVLLMClient(
        base_url=args.vllm_base_url,
        model=model_config["model_name"],
        timeout_sec=float(model_config["timeout_sec"]),
    )
    if not client.health_check():
        raise SystemExit(f"vLLM server does not expose model {model_config['model_name']}")
    video_paths = _load_video_paths(args.dataset_manifest)
    work_dir = args.work_dir.expanduser().resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="stage4_6_dtl_clips_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[tuple[str, float, float], tuple[Path, dict]] = {}

    def prepare_context(*, video_id: str, candidate_id: str, window):
        key = (video_id, float(window["start_sec"]), float(window["end_sec"]))
        if key not in clips:
            relative = video_paths.get(video_id)
            if relative is None:
                raise SystemExit(f"dataset manifest lacks video_id: {video_id}")
            clip_path = work_dir / f"{video_id}_{key[1]:.6f}_{key[2]:.6f}.mp4"
            timing = extract_video_clip(
                args.video_root / relative, clip_path,
                start_sec=key[1], end_sec=key[2], ffmpeg_bin=args.ffmpeg_bin,
            )
            clips[key] = (clip_path, timing)
        return clips[key][1]

    def model_fn(prompt: str, *, video_id: str, candidate_id: str, window):
        key = (video_id, float(window["start_sec"]), float(window["end_sec"]))
        if key not in clips:
            raise RuntimeError("context must be prepared before model inference")
        try:
            response = client.analyze_video(
                clips[key][0], prompt,
                max_new_tokens=int(model_config["max_new_tokens"]),
                temperature=float(model_config["temperature"]),
                enable_thinking=bool(model_config["enable_thinking"]),
            )
        except APITimeoutError as exc:
            raise TimeoutError("vLLM inference timed out") from exc
        except APIConnectionError as exc:
            raise ConnectionError("vLLM inference connection failed") from exc
        return response.content, response.finish_reason

    return prepare_context, model_fn


def main() -> int:
    args = parse_args()
    if args.command in {"dtl0", "dtl1"}:
        prepare = model = None
        if args.command == "dtl1":
            protocol = load_dense_protocol(args.protocol, allow_draft=args.allow_draft_protocol)
            prepare, model = _build_dtl1_runtime(args, protocol)
        result = run_localization_to_file(
            args.cache_dir, args.role_manifest, args.protocol,
            args.command.upper(), args.output, model_fn=model, prepare_context_fn=prepare,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        print(json.dumps({"records": result["input_record_count"], "candidates": result["input_candidate_count"], "rules": result["decision_rule_counts"]}, ensure_ascii=False))
    elif args.command == "validate":
        print(json.dumps(validate_localization_file(args.cache_dir, args.role_manifest, args.result, args.protocol, allow_draft_protocol=args.allow_draft_protocol), ensure_ascii=False))
    elif args.command == "replay":
        print(json.dumps(replay_localization_to_jsonl(args.cache_dir, args.role_manifest, args.result, args.output, args.protocol, allow_draft_protocol=args.allow_draft_protocol), ensure_ascii=False))
    elif args.command == "evaluate":
        result = evaluate_localization_to_file(
            args.cache_dir, args.role_manifest, args.result, args.replay,
            args.frozen_predictions, args.output, args.protocol,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        print(json.dumps(result.get("aggregate", result), ensure_ascii=False))
    elif args.command in {"summarize", "assess"}:
        result = summarize_evaluations_to_file(
            args.baseline_evaluation, args.candidate_evaluation, args.protocol, args.output,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        print(json.dumps(result["comparison"]["aggregate_delta"], ensure_ascii=False))
    else:
        manifest, records = load_cache_payloads(args.cache_dir)
        roles = validate_role_manifest_directory(args.role_dir, cache_manifest=manifest)
        if roles["union_record_count"] != manifest["record_count"]:
            raise SystemExit("role union does not cover full cache")
        result = full_cache_dtl0_identity(manifest, records)
        print(json.dumps(result, ensure_ascii=False))
        if result["mismatches"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
