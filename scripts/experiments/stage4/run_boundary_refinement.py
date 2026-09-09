#!/usr/bin/env python3
"""Run Stage 4.4 temporal boundary refinement (BR-0 identity / BR-1 local refiner)."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from aic_video_highlight.highlight_retrieval.boundary_refinement import (
    assess_refinement_files,
    evaluate_refinement_to_file,
    replay_refinement_to_jsonl,
    run_refinement_to_file,
    validate_refinement_file,
    load_boundary_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4.4 temporal boundary refinement baselines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    refine = commands.add_parser("refine", help="Create a canonical boundary refinement result")
    refine.add_argument("--cache-dir", type=Path, required=True)
    refine.add_argument("--role-manifest", type=Path, required=True)
    refine.add_argument("--protocol", type=Path, required=True)
    refine.add_argument("--refiner", choices=("BR-0", "BR-1"), required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument(
        "--allow-draft-protocol",
        action="store_true",
        help="Accept DRAFT_FOR_INDEPENDENT_REVIEW status (development runs only)",
    )
    br1 = refine.add_argument_group(
        "BR-1 model backend", "Required only when --refiner BR-1"
    )
    br1.add_argument("--dataset-manifest", type=Path, help="Dataset manifest JSONL (dev split)")
    br1.add_argument("--video-root", type=Path, help="Dataset root containing relative video paths")
    br1.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    br1.add_argument(
        "--model",
        help="Optional assertion only; must exactly equal the Protocol model",
    )
    br1.add_argument(
        "--vllm-timeout-sec",
        type=float,
        help="Optional assertion only; must exactly equal the Protocol timeout",
    )
    br1.add_argument("--ffmpeg-bin", default="ffmpeg")
    br1.add_argument("--work-dir", type=Path, help="Directory for extracted local clips")

    validate = commands.add_parser("validate-refinement", help="Validate refinement integrity")
    validate.add_argument("--cache-dir", type=Path, required=True)
    validate.add_argument("--role-manifest", type=Path, required=True)
    validate.add_argument("--refinement-result", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--allow-draft-protocol", action="store_true")

    replay = commands.add_parser("replay", help="Replay refined predictions")
    replay.add_argument("--cache-dir", type=Path, required=True)
    replay.add_argument("--role-manifest", type=Path, required=True)
    replay.add_argument("--refinement-result", type=Path, required=True)
    replay.add_argument("--protocol", type=Path, required=True)
    replay.add_argument("--allow-draft-protocol", action="store_true")
    replay.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Frozen weak-reference evaluation")
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--role-manifest", type=Path, required=True)
    evaluate.add_argument("--refinement-result", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--allow-draft-protocol", action="store_true")
    evaluate.add_argument("--refined-predictions", type=Path, required=True)
    evaluate.add_argument(
        "--frozen-predictions",
        action="append",
        type=Path,
        required=True,
        help="Repeat for a mixed Dev+Hard Audit role",
    )
    evaluate.add_argument("--output", type=Path, required=True)

    assess = commands.add_parser("assess", help="Compare with BR-0 and apply frozen gates")
    assess.add_argument("--baseline-evaluation", type=Path, required=True)
    assess.add_argument("--candidate-evaluation", type=Path, required=True)
    assess.add_argument("--protocol", type=Path, required=True)
    assess.add_argument("--phase", choices=("dev", "hard"), required=True)
    assess.add_argument("--output", type=Path, required=True)
    assess.add_argument("--allow-draft-protocol", action="store_true")
    return parser.parse_args()


def _load_video_paths(manifest_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = row.get("video_id")
        relative = row.get("relative_video_path")
        if not isinstance(video_id, str) or not isinstance(relative, str):
            raise ValueError(f"invalid manifest row: {row!r}")
        if video_id in paths:
            raise ValueError(f"duplicate manifest video_id: {video_id}")
        paths[video_id] = relative
    return paths


def _build_br1_runtime(args: argparse.Namespace, protocol):
    """Derive the clip/model runtime from Protocol; CLI values are assertions."""
    if args.dataset_manifest is None or args.video_root is None:
        raise SystemExit("BR-1 requires --dataset-manifest and --video-root")
    from aic_video_highlight.highlight_retrieval.qwen_vllm_client import QwenVLLMClient
    from aic_video_highlight.highlight_retrieval.video_clip import extract_video_clip
    from openai import APIConnectionError, APITimeoutError

    model_config = protocol["refiners"]["BR-1"]["model"]
    protocol_model = model_config["model_name"]
    protocol_timeout = float(model_config["timeout_sec"])
    if args.model is not None and args.model != protocol_model:
        raise SystemExit("--model does not match the frozen Protocol")
    if args.vllm_timeout_sec is not None and args.vllm_timeout_sec != protocol_timeout:
        raise SystemExit("--vllm-timeout-sec does not match the frozen Protocol")

    client = QwenVLLMClient(
        base_url=args.vllm_base_url,
        model=protocol_model,
        timeout_sec=protocol_timeout,
    )
    if not client.health_check():
        raise SystemExit(f"vLLM server does not expose model {protocol_model}")
    video_paths = _load_video_paths(args.dataset_manifest)
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else Path(tempfile.mkdtemp(prefix="stage4_4_clips_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_cache: dict[tuple[str, float, float], tuple[Path, dict]] = {}

    def prepare_context(*, video_id: str, candidate_id: str, window):
        key = (video_id, float(window["start_sec"]), float(window["end_sec"]))
        cached = clip_cache.get(key)
        if cached is None:
            relative = video_paths.get(video_id)
            if relative is None:
                raise SystemExit(f"video manifest lacks video_id: {video_id}")
            source = args.video_root / relative
            clip_path = work_dir / f"{video_id}_{key[1]:.6f}_{key[2]:.6f}.mp4"
            timing = extract_video_clip(
                source,
                clip_path,
                start_sec=key[1],
                end_sec=key[2],
                ffmpeg_bin=args.ffmpeg_bin,
            )
            clip_cache[key] = (clip_path, timing)
            return timing
        return cached[1]

    def refine_one(prompt: str, *, video_id: str, candidate_id: str, window) -> tuple[str, str | None]:
        key = (video_id, float(window["start_sec"]), float(window["end_sec"]))
        if key not in clip_cache:
            raise RuntimeError("context must be prepared before model inference")
        clip_path = clip_cache[key][0]
        try:
            response = client.analyze_video(
                clip_path,
                prompt,
                max_new_tokens=int(model_config["max_new_tokens"]),
                temperature=float(model_config["temperature"]),
                enable_thinking=bool(model_config["enable_thinking"]),
            )
        except APITimeoutError as exc:
            raise TimeoutError("vLLM inference timed out") from exc
        except APIConnectionError as exc:
            raise ConnectionError("vLLM inference connection failed") from exc
        return response.content, response.finish_reason

    return prepare_context, refine_one


def main() -> int:
    args = parse_args()
    if args.command == "refine":
        model_fn = None
        prepare_context_fn = None
        if args.refiner == "BR-1":
            protocol = load_boundary_protocol(
                args.protocol, allow_draft=args.allow_draft_protocol
            )
            prepare_context_fn, model_fn = _build_br1_runtime(args, protocol)
        payload = run_refinement_to_file(
            args.cache_dir,
            args.role_manifest,
            args.protocol,
            args.refiner,
            args.output,
            model_fn=model_fn,
            prepare_context_fn=prepare_context_fn,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        result = {
            key: payload[key]
            for key in (
                "refiner_name",
                "record_count",
                "input_candidate_count",
                "decision_counts",
                "boundary_refinement_semantic_hash",
            )
        }
    elif args.command == "validate-refinement":
        result = validate_refinement_file(
            args.cache_dir,
            args.role_manifest,
            args.refinement_result,
            args.protocol,
            allow_draft_protocol=args.allow_draft_protocol,
        )
    elif args.command == "replay":
        result = replay_refinement_to_jsonl(
            args.cache_dir,
            args.role_manifest,
            args.refinement_result,
            args.output,
            args.protocol,
            allow_draft_protocol=args.allow_draft_protocol,
        )
    elif args.command == "evaluate":
        payload = evaluate_refinement_to_file(
            args.cache_dir,
            args.role_manifest,
            args.refinement_result,
            args.refined_predictions,
            args.frozen_predictions,
            args.output,
            args.protocol,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        result = {
            "record_count": payload["record_count"],
            "evaluation_semantic_hash": payload["evaluation_semantic_hash"],
            "aggregate": payload["aggregate"],
        }
    else:
        payload = assess_refinement_files(
            args.baseline_evaluation,
            args.candidate_evaluation,
            args.protocol,
            args.phase,
            args.output,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        result = {
            "phase": payload["phase"],
            "pass": payload["assessments"]["pass"],
            "assessment_semantic_sha256": payload["assessment_semantic_sha256"],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
