#!/usr/bin/env python3
"""Run reproducible weak-reference development experiments from a manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aic_video_highlight.highlight_retrieval.baseline_experiment import (
    RawOutputJournal,
    atomic_write_json,
    evaluate_weak_references,
    load_baseline_samples,
    persist_run_config,
    resume_decision,
    run_with_failure_isolation,
    write_aggregate_figures,
    write_summary_files,
    write_timeline_figure,
)
from aic_video_highlight.highlight_retrieval.pipeline import (
    HighlightRetrievalPipeline,
    load_highlight_retrieval_config,
    parse_saved_raw_outputs,
)
from aic_video_highlight.highlight_retrieval.prompt_builder import PROMPT_VERSION
from aic_video_highlight.highlight_retrieval.qwen_vllm_client import QwenVLLMClient
from aic_video_highlight.highlight_retrieval.temporal_metrics import (
    ZERO_DURATION_EPSILON_SEC,
)
from aic_video_highlight.highlight_retrieval.video_chunker import build_chunks
from aic_video_highlight.highlight_retrieval.video_clip import extract_video_clip
from aic_video_highlight.highlight_retrieval.video_metadata import probe_video


DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated Baseline 20 weak-reference development experiments"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-id", help="Run only this manifest video_id")
    parser.add_argument("--limit", type=int, help="Process at most this many selected samples")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete successful samples and recover post-processing from saved raw output",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run model calls and replace existing sample/raw results (default: false)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, call the model again for failed samples instead of reparsing saved raw",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser.parse_args()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _segment_dicts(segments) -> list[dict[str, Any]]:
    return [asdict(segment) for segment in segments]


def _build_result(
    sample: dict[str, Any],
    *,
    pipeline_result,
    clip_extraction_sec: float,
    evaluation_sec: float,
    total_sec: float,
) -> dict[str, Any]:
    predictions = _segment_dicts(pipeline_result.segments)
    candidates = _segment_dicts(pipeline_result.candidate_segments)
    metrics = evaluate_weak_references(predictions, sample["weak_reference_segments"])
    warnings: list[str] = []
    if metrics["zero_duration_reference_count"]:
        warnings.append("zero_duration_reference_excluded_from_duration_metrics")
    timing = {
        "clip_extraction_sec": clip_extraction_sec,
        "model_inference_sec": pipeline_result.timing.get("model_inference_sec", 0.0),
        "parsing_sec": pipeline_result.timing.get("parsing_sec", 0.0),
        "merging_sec": pipeline_result.timing.get("merging_sec", 0.0),
        "evaluation_sec": evaluation_sec,
        "total_sec": total_sec,
    }
    return {
        "sample_index": sample["sample_index"],
        "group": sample["group"],
        "video_id": sample["video_id"],
        "source_group": sample["source_group"],
        "source_video_path": sample["source_video_path"],
        "clip_start_sec": sample["clip_start_sec"],
        "clip_end_sec": sample["clip_end_sec"],
        "clip_duration_sec": sample["clip_duration_sec"],
        "experiment_clip_duration_sec": pipeline_result.duration_sec,
        "coordinate_system": "clip-local seconds after source clip extraction",
        "weak_reference_segments": sample["weak_reference_segments"],
        "raw_chunk_outputs": pipeline_result.raw_chunk_outputs,
        "parsed_chunk_segments": candidates,
        "merged_prediction_segments": predictions,
        "warnings": warnings,
        "errors": [],
        "timing": timing,
        "metrics": metrics,
        "success": True,
    }


def _recover_result_from_raw(
    sample: dict[str, Any],
    raw_path: Path,
    *,
    config,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_chunks = raw_payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("saved raw output contains no model responses")
    experiment_duration = float(
        raw_payload.get("experiment_clip_duration_sec", sample["clip_duration_sec"])
    )
    expected_chunks = len(
        build_chunks(experiment_duration, config.chunk_seconds, config.overlap_seconds)
    )
    if len(raw_chunks) != expected_chunks:
        raise ValueError(
            f"saved raw output is incomplete: {len(raw_chunks)} of {expected_chunks} chunks"
        )
    candidates, merged, normalized, post_timing = parse_saved_raw_outputs(
        raw_chunks,
        tiou_threshold=config.merge_tiou_threshold,
    )
    predictions = _segment_dicts(merged)
    evaluation_started_at = time.perf_counter()
    metrics = evaluate_weak_references(predictions, sample["weak_reference_segments"])
    evaluation_sec = time.perf_counter() - evaluation_started_at
    warnings = ["recovered_postprocessing_from_saved_raw_output"]
    if metrics["zero_duration_reference_count"]:
        warnings.append("zero_duration_reference_excluded_from_duration_metrics")
    return {
        "sample_index": sample["sample_index"],
        "group": sample["group"],
        "video_id": sample["video_id"],
        "source_group": sample["source_group"],
        "source_video_path": sample["source_video_path"],
        "clip_start_sec": sample["clip_start_sec"],
        "clip_end_sec": sample["clip_end_sec"],
        "clip_duration_sec": sample["clip_duration_sec"],
        "experiment_clip_duration_sec": experiment_duration,
        "coordinate_system": "clip-local seconds after source clip extraction",
        "weak_reference_segments": sample["weak_reference_segments"],
        "raw_chunk_outputs": normalized,
        "parsed_chunk_segments": _segment_dicts(candidates),
        "merged_prediction_segments": predictions,
        "warnings": warnings,
        "errors": [],
        "timing": {
            "clip_extraction_sec": 0.0,
            "model_inference_sec": sum(
                float(item.get("request_latency_sec", 0.0)) for item in raw_chunks
            ),
            "parsing_sec": post_timing["parsing_sec"],
            "merging_sec": post_timing["merging_sec"],
            "evaluation_sec": evaluation_sec,
            "total_sec": time.perf_counter() - started_at,
        },
        "metrics": metrics,
        "success": True,
    }


def _write_report(
    output_dir: Path,
    *,
    run_config: dict[str, Any],
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    figure_paths: list[Path],
) -> None:
    successful = [item for item in results if item.get("success") is True]
    lines = [
        "# Experiment",
        "",
        "## Configuration",
        "",
        f"- Model: `{run_config['model_name']}`",
        f"- Model revision: `{run_config['model_revision']}`",
        f"- Prompt: `{run_config['prompt_version']}`",
        f"- Sampling FPS: `{run_config['sampling_fps']}`",
        f"- Chunk / overlap: `{run_config['chunk_seconds']}` / `{run_config['chunk_overlap_seconds']}` seconds",
        f"- Reference: `{run_config['reference_type']}`",
        f"- Metric policy: {run_config['metric_policy']}",
        "",
        "## Dataset",
        "",
        f"This dry run contains {len(results)} video from the QVHighlights community mirror. ",
        "The weak references are seed/teacher-derived development references, not official Ground Truth.",
        "",
        "## Results",
        "",
    ]
    for item in results:
        lines.extend([f"### {item['video_id']}", "", f"- Success: `{item.get('success')}`"])
        if item.get("success"):
            lines.extend(
                [
                    f"- Weak reference: `{json.dumps(item['weak_reference_segments'], ensure_ascii=False)}`",
                    f"- Prediction: `{json.dumps(item['merged_prediction_segments'], ensure_ascii=False)}`",
                    f"- Metrics: `{json.dumps(item['metrics'], ensure_ascii=False)}`",
                    f"- Model inference: `{item['timing']['model_inference_sec']:.3f}` seconds",
                    f"- Total: `{item['timing']['total_sec']:.3f}` seconds",
                    f"- Raw response file: `raw/{item['video_id']}.json`",
                ]
            )
        else:
            lines.append(f"- Errors: `{json.dumps(item.get('errors', []), ensure_ascii=False)}`")
        lines.append("")
    lines.extend(["## Warnings", ""])
    warnings = [f"{item['video_id']}: {warning}" for item in results for warning in item.get("warnings", [])]
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    lines.extend(["", "## Visualizations", ""])
    lines.extend(f"- `{path.relative_to(output_dir).as_posix()}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"This run processed {aggregate['success']} successful sample and {aggregate['failed']} failure. ",
            "N=1 validates the experiment path only and does not support a general performance conclusion.",
            "",
            "`temporal_iou` is the intersection duration divided by the union duration after separately merging overlapping prediction and weak-reference intervals. Precision, recall, and F1 are also duration-weighted. Degenerate weak references with duration at or below the configured epsilon remain in raw results but are excluded from duration-based denominators.",
            "",
            "## Reproducibility",
            "",
            f"- Command: `{run_config['command']}`",
            f"- Git HEAD: `{run_config['git_commit_head']}`",
            f"- Python / torch / vLLM: `{run_config['python_version']}` / `{run_config['torch_version']}` / `{run_config['vllm_version']}`",
            f"- Output directory: `{output_dir}`",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.retry_failed and not args.resume:
        raise SystemExit("--retry-failed requires --resume")
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    samples_dir = output_dir / "samples"
    timeline_dir = output_dir / "figures" / "timelines"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)

    config = load_highlight_retrieval_config(args.config)
    samples = load_baseline_samples(
        args.manifest,
        args.video_root,
        video_id=args.video_id,
        limit=args.limit,
    )
    run_config = {
        "experiment_name": output_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": config.model,
        "model_revision": args.model_revision,
        "vllm_version": _package_version("vllm"),
        "python_version": sys.version.split()[0],
        "torch_version": _package_version("torch"),
        "prompt_version": PROMPT_VERSION,
        "sampling_fps": config.coarse_fps,
        "chunk_seconds": config.chunk_seconds,
        "chunk_overlap_seconds": config.overlap_seconds,
        "merge_threshold": config.merge_tiou_threshold,
        "merge_strategy": "sorted adjacent temporal-IoU union; maximum score",
        "manifest_path": str(args.manifest.expanduser().resolve()),
        "video_root": str(args.video_root.expanduser().resolve()),
        "reference_type": "weak_reference_segments",
        "metric_policy": "union-aware duration overlap; not an official competition metric",
        "zero_duration_policy": {
            "epsilon_sec": ZERO_DURATION_EPSILON_SEC,
            "behavior": "preserve and count; exclude from duration denominators",
        },
        "git_commit_head": _git_head(project_root),
        "request_parameters": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "enable_thinking": config.enable_thinking,
            "timeout_sec": config.request_timeout_sec,
        },
        "command": shlex.join(sys.argv),
        "selected_video_ids": [item["video_id"] for item in samples],
    }
    run_config = persist_run_config(
        output_dir / "run_config.json",
        run_config,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    client: QwenVLLMClient | None = None

    def get_pipeline() -> HighlightRetrievalPipeline:
        nonlocal client
        if client is None:
            client = QwenVLLMClient(
                base_url=args.base_url,
                model=config.model,
                timeout_sec=config.request_timeout_sec,
            )
            if not client.health_check():
                raise RuntimeError(f"vLLM does not list configured model: {config.model}")
        return HighlightRetrievalPipeline(
            client,
            config,
            ffprobe_bin=args.ffprobe_bin,
            ffmpeg_bin=args.ffmpeg_bin,
        )

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        result_path = samples_dir / f"{sample['video_id']}.json"
        raw_path = raw_dir / f"{sample['video_id']}.json"
        decision = resume_decision(
            result_path,
            raw_path=raw_path,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        if decision == "skip":
            print(f"RESUME SKIP {sample['video_id']} success result already complete")
            return json.loads(result_path.read_text(encoding="utf-8"))
        if decision == "blocked":
            raise RuntimeError(
                f"result already exists for {sample['video_id']}; use --resume or --overwrite"
            )
        if decision == "recover_raw" and not args.retry_failed:
            print(f"RESUME RAW {sample['video_id']} reparsing without a model call")
            recovered = _recover_result_from_raw(sample, raw_path, config=config)
            atomic_write_json(result_path, recovered)
            return recovered

        source = Path(sample["source_video_path"])
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"source video is missing or empty: {source}")
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=".aic-experiment-clip-", dir=source.parent) as temp:
            experiment_clip = Path(temp) / f"{sample['video_id']}.mp4"
            extraction_started_at = time.perf_counter()
            extract_video_clip(
                source,
                experiment_clip,
                start_sec=float(sample["clip_start_sec"]),
                end_sec=float(sample["clip_end_sec"]),
                ffmpeg_bin=args.ffmpeg_bin,
            )
            clip_extraction_sec = time.perf_counter() - extraction_started_at
            experiment_meta = probe_video(experiment_clip, ffprobe_bin=args.ffprobe_bin)
            journal = RawOutputJournal(
                raw_path,
                {
                    "video_id": sample["video_id"],
                    "source_group": sample["source_group"],
                    "clip_start_sec": sample["clip_start_sec"],
                    "clip_end_sec": sample["clip_end_sec"],
                    "experiment_clip_duration_sec": experiment_meta.duration_sec,
                    "coordinate_system": "clip-local seconds",
                },
            )
            pipeline_result = get_pipeline().run(
                experiment_clip,
                raw_output_sink=journal.record,
            )
        evaluation_started_at = time.perf_counter()
        evaluate_weak_references(
            _segment_dicts(pipeline_result.segments), sample["weak_reference_segments"]
        )
        evaluation_sec = time.perf_counter() - evaluation_started_at
        result = _build_result(
            sample,
            pipeline_result=pipeline_result,
            clip_extraction_sec=clip_extraction_sec,
            evaluation_sec=evaluation_sec,
            total_sec=time.perf_counter() - started_at,
        )
        atomic_write_json(result_path, result)
        print(f"SUCCESS {sample['video_id']}")
        return result

    results = run_with_failure_isolation(samples, process, stage="sample_pipeline")
    for result in results:
        atomic_write_json(samples_dir / f"{result['video_id']}.json", result)

    figure_paths: list[Path] = []
    for result in results:
        if result.get("success") is not True:
            continue
        timeline_path = timeline_dir / f"{result['video_id']}.png"
        write_timeline_figure(
            video_id=result["video_id"],
            duration_sec=result["experiment_clip_duration_sec"],
            weak_reference_segments=result["weak_reference_segments"],
            prediction_segments=result["merged_prediction_segments"],
            metrics=result["metrics"],
            output_path=timeline_path,
        )
        figure_paths.append(timeline_path)
    figure_paths.extend(write_aggregate_figures(results, output_dir / "figures"))
    aggregate = write_summary_files(results, output_dir=output_dir)
    _write_report(
        output_dir,
        run_config=run_config,
        results=results,
        aggregate=aggregate,
        figure_paths=figure_paths,
    )
    print(
        f"SUMMARY processed={aggregate['processed']} success={aggregate['success']} "
        f"failed={aggregate['failed']} output={output_dir}"
    )
    return 0 if aggregate["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
