#!/usr/bin/env python3
"""Qwen throughput / determinism audit for the PHD2 upstream benchmark.

Engineering-only: this script never changes the frozen Qwen scientific
semantics.  It runs the frozen retrieval pipeline through an instrumented
execution wrapper (``aic_video_highlight.ftnet.qwen_execution``) that records
per-chunk raw responses, usage counters and phase timings, and optionally
renders chunk N+1 while chunk N is in flight (requests stay strictly serial).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from aic_video_highlight.ftnet.index import load_index, resolve_video_path
from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
    atomic_json,
    load_benchmark_manifest,
    sha256_file,
)
from aic_video_highlight.ftnet.qwen_execution import (
    ARM_SUMMARY_SCHEMA,
    atomic_write_json,
    chunk_audit_path,
    compare_audit_arms,
    fetch_prometheus_metrics,
    metrics_delta,
    run_retrieval_audited,
    summarize_request_metrics,
)
from aic_video_highlight.ftnet.real_provider import QWEN_MODEL, RetrievalSettings
from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient

SUBSET_SCHEMA = "aic.phd2.qwen-audit-subset/v1"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_METRICS_URL = "http://127.0.0.1:8000/metrics"


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def subset_records(benchmark: dict, video_ids: list[str]) -> dict:
    by_id = {str(record["youtubeId"]): record for record in benchmark["records"]}
    missing = [video_id for video_id in video_ids if video_id not in by_id]
    if missing:
        raise RuntimeError(f"subset ids missing from Benchmark32: {missing}")
    records = [by_id[video_id] for video_id in video_ids]
    payload = {
        "schema": SUBSET_SCHEMA,
        "purpose": "qwen_determinism_audit",
        "benchmark_schema": benchmark.get("schema"),
        "source_manifest_sha256": benchmark.get("source_manifest_sha256"),
        "count": len(records),
        "video_ids": video_ids,
        "records": records,
    }
    payload["subset_sha256"] = _sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return payload


# ---------------------------------------------------------------------------
# vLLM service control (kept byte-comparable with runtime.orchestrator.start_vllm)
# ---------------------------------------------------------------------------


def _service_has_model(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as response:
            payload = json.load(response)
        return any(item.get("id") == QWEN_MODEL for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def start_vllm(
    *,
    python: str,
    model_path: Path,
    media_path: Path,
    log_path: Path,
    gpu_memory_utilization: float,
    max_model_len: int,
    extra_args: list[str] | None = None,
    start_timeout_sec: float = 900.0,
) -> tuple[subprocess.Popen | None, str]:
    if _service_has_model(VLLM_BASE_URL):
        return None, VLLM_BASE_URL
    command = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", QWEN_MODEL,
        "--host", "127.0.0.1", "--port", "8000",
        "--trust-remote-code",
        "--allowed-local-media-path", str(media_path),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    command.extend(extra_args or [])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    child_environment = os.environ.copy()
    child_environment["PATH"] = str(Path(python).parent) + os.pathsep + child_environment.get("PATH", "")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=child_environment,
    )
    deadline = time.monotonic() + start_timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"vLLM exited during startup ({process.returncode}); see {log_path}")
        if _service_has_model(VLLM_BASE_URL):
            log.close()
            return process, VLLM_BASE_URL
        time.sleep(2)
    stop_vllm(process)
    log.close()
    raise RuntimeError(f"vLLM health check timed out; see {log_path}")


def stop_vllm(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def vllm_non_default_args(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "non-default args:" in line:
            return {"line": line.split("non-default args:", 1)[1].strip()[:2000]}
    return {}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def command_subset(args) -> dict:
    benchmark = load_benchmark_manifest(
        args.benchmark_manifest,
        expected_sha256=args.benchmark_sha256,
        dataset_root=args.dataset_root,
        verify_raw_sha256=False,
    )
    video_ids = (
        [str(record["youtubeId"]) for record in benchmark["records"]]
        if args.all_benchmark_videos
        else [str(value) for value in args.video_ids]
    )
    if not video_ids:
        raise RuntimeError("subset requires --video-id or --all-benchmark-videos")
    payload = subset_records(benchmark, video_ids)
    payload["purpose"] = (
        "qwen_full_benchmark32" if args.all_benchmark_videos else "qwen_determinism_audit"
    )
    payload["benchmark_manifest_path"] = str(args.benchmark_manifest.resolve())
    payload["benchmark_manifest_sha256"] = args.benchmark_sha256
    path = atomic_json(args.output, payload)
    digest = sha256_file(path)
    (Path(str(path) + ".sha256")).write_text(f"{digest}  {Path(path).name}\n", encoding="utf-8")
    return {"path": str(path), "sha256": digest, "video_ids": video_ids, "count": len(video_ids)}


def _selected_entries(index_path: Path, ids: list[str]):
    entries, _ = load_index(index_path)
    mapping = {entry.video_id: entry for entry in entries}
    missing = [video_id for video_id in ids if video_id not in mapping]
    if missing:
        raise RuntimeError(f"audit ids missing from index: {missing}")
    return [mapping[video_id] for video_id in ids]


def command_run(args) -> dict:
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    video_ids = [str(value) for value in subset["video_ids"]]
    if args.video_ids:
        video_ids = [str(value) for value in args.video_ids]
        unknown = [value for value in video_ids if value not in subset["video_ids"]]
        if unknown and not args.allow_unlisted_video:
            raise RuntimeError(
                f"--video-id outside the frozen subset: {unknown} "
                "(pass --allow-unlisted-video for a single-video recovery control)"
            )
    entries = _selected_entries(args.index, video_ids)
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        existing = sorted((root / "retrieval").glob("*.json")) if (root / "retrieval").is_dir() else []
        if existing and not args.allow_existing:
            raise RuntimeError(
                f"{root} already contains retrieval artifacts; refusing to masquerade cached "
                "output as a re-run (pass --resume only to continue an interrupted arm)"
            )

    owner = None
    service_load_seconds = 0.0
    metrics_before = metrics_after = None
    started_run = time.perf_counter()
    try:
        if args.start_service:
            service_started = time.perf_counter()
            owner, _ = start_vllm(
                python=os.sys.executable,
                model_path=args.model_path,
                media_path=args.dataset_root,
                log_path=root / "vllm.log",
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                extra_args=args.vllm_arg,
            )
            service_load_seconds = time.perf_counter() - service_started
        metrics_before = fetch_prometheus_metrics(VLLM_METRICS_URL)
        settings = RetrievalSettings(base_url=VLLM_BASE_URL, timeout_sec=args.timeout_sec)
        rows = []
        for entry in entries:
            client = QwenVLLMClient(
                base_url=VLLM_BASE_URL, model=QWEN_MODEL, timeout_sec=args.timeout_sec
            )
            video_started = time.perf_counter()
            payload, audit = run_retrieval_audited(
                entry,
                dataset_root=args.dataset_root,
                client=client,
                settings=settings,
                overlap_render=args.overlap_render,
                audit_path=chunk_audit_path(root, entry.video_id),
            )
            atomic_json(root / "retrieval" / f"{entry.video_id}.json", payload)
            video_wall = time.perf_counter() - video_started
            timing = payload["timing"]
            attempts = [item for row in audit["chunks"] for item in row["attempts"]]
            rows.append(
                {
                    "video_id": entry.video_id,
                    "chunks": len(payload["chunks"]),
                    "video_wall_sec": round(video_wall, 6),
                    "artifact_wall_sec": round(float(timing["artifact_wall_sec"]), 6),
                    "render_sec": round(float(timing["chunk_extraction_sec"]), 6),
                    "request_sec": round(float(timing["model_inference_sec"]), 6),
                    "probe_sec": round(float(timing["probe_sec"]), 6),
                    "parse_sec": round(float(timing["parsing_sec"]), 6),
                    "merge_sec": round(float(timing["merging_sec"]), 6),
                    "render_wait_sec": round(float(audit["video"]["render_wait_sec_total"]), 6),
                    "requests": len(attempts),
                    "non_primary_attempts": sum(
                        1 for item in attempts if item["attempt"] != "primary"
                    ),
                    "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in attempts),
                    "completion_tokens": sum(
                        int(item.get("completion_tokens") or 0) for item in attempts
                    ),
                    "cached_tokens": sum(int(item.get("cached_tokens") or 0) for item in attempts),
                }
            )
            print(
                f"[qwen-audit] {args.arm_id} {entry.video_id} chunks={len(payload['chunks'])} "
                f"wall={video_wall:.1f}s render={timing['chunk_extraction_sec']:.1f}s "
                f"request={timing['model_inference_sec']:.1f}s "
                f"tokens_in={rows[-1]['prompt_tokens']} tokens_out={rows[-1]['completion_tokens']}",
                flush=True,
            )
        metrics_after = fetch_prometheus_metrics(VLLM_METRICS_URL)
    finally:
        if owner is not None:
            stop_vllm(owner)
    run_wall = time.perf_counter() - started_run
    delta = metrics_delta(metrics_before or {}, metrics_after or {})
    summary = {
        "schema": ARM_SUMMARY_SCHEMA,
        "arm_id": args.arm_id,
        "root": str(root),
        "subset_path": str(args.subset),
        "subset_sha256": sha256_file(args.subset),
        "video_ids": video_ids,
        "video_count": len(rows),
        "chunk_count": sum(row["chunks"] for row in rows),
        "run_wall_sec": run_wall,
        "sum_video_wall_sec": round(sum(row["video_wall_sec"] for row in rows), 6),
        "sum_render_sec": round(sum(row["render_sec"] for row in rows), 6),
        "sum_request_sec": round(sum(row["request_sec"] for row in rows), 6),
        "sum_probe_sec": round(sum(row["probe_sec"] for row in rows), 6),
        "sum_parse_sec": round(sum(row["parse_sec"] for row in rows), 6),
        "sum_merge_sec": round(sum(row["merge_sec"] for row in rows), 6),
        "sum_render_wait_sec": round(sum(row["render_wait_sec"] for row in rows), 6),
        "request_count": sum(row["requests"] for row in rows),
        "non_primary_attempt_count": sum(row["non_primary_attempts"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "cached_tokens": sum(row["cached_tokens"] for row in rows),
        "overlap_render": bool(args.overlap_render),
        "qwen_service_start_count": 1 if owner is not None else 0,
        "qwen_service_load_seconds": service_load_seconds,
        "vllm_extra_args": list(args.vllm_arg or []),
        "vllm_non_default_args": vllm_non_default_args(
            args.vllm_log_path or (root / "vllm.log")
        ),
        "server_metrics_delta": delta,
        "server_request_decomposition": summarize_request_metrics(delta),
        "videos": rows,
    }
    atomic_write_json(root / "arm_summary.json", summary)
    return summary


def command_compare(args) -> dict:
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    video_ids = [str(value) for value in subset["video_ids"]]
    if args.video_ids:
        video_ids = [str(value) for value in args.video_ids]
    left = json.loads((args.left_root / "arm_summary.json").read_text(encoding="utf-8"))
    right = json.loads((args.right_root / "arm_summary.json").read_text(encoding="utf-8"))
    if args.left_root == args.right_root:
        raise RuntimeError("left and right arms must differ")
    missing = [row for row in left["video_ids"] if row not in right["video_ids"]]
    if missing:
        raise RuntimeError(f"arm membership mismatch: {missing}")
    report = compare_audit_arms(args.left_root, args.right_root, video_ids)
    report["left_arm"] = left["arm_id"]
    report["right_arm"] = right["arm_id"]
    report["left"] = {
        key: left[key]
        for key in (
            "run_wall_sec",
            "sum_render_sec",
            "sum_request_sec",
            "chunk_count",
            "sum_render_wait_sec",
            "overlap_render",
        )
    }
    report["right"] = {
        key: right[key]
        for key in (
            "run_wall_sec",
            "sum_render_sec",
            "sum_request_sec",
            "chunk_count",
            "sum_render_wait_sec",
            "overlap_render",
        )
    }
    report["wall_ratio"] = (
        round(left["sum_video_wall_sec"] / right["sum_video_wall_sec"], 6)
        if right.get("sum_video_wall_sec")
        else None
    )
    if args.artifact_baseline_root is not None:
        from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
            compare_retrieval_roots,
        )

        artifact = compare_retrieval_roots(
            args.artifact_baseline_root, args.right_root, video_ids
        )
        report["artifact_baseline_root"] = str(args.artifact_baseline_root)
        report["artifact_semantic_equivalence_vs_baseline"] = artifact["pass"]
        report["artifact_rows"] = artifact["rows"]
    if args.output is not None:
        atomic_write_json(args.output, report)
    return report


def command_render_probe(args) -> dict:
    """Measure how much source video ffmpeg decodes per rendered chunk.

    The command mirrors ``retrieval.video_clip.extract_video_clip`` exactly and
    only adds ``-v info`` stderr parsing: this audit never changes the renderer.
    """

    import re
    import subprocess
    import tempfile

    from aic_video_highlight.ftnet.real_provider import RetrievalSettings
    from aic_video_highlight.retrieval.video_chunker import build_chunks
    from aic_video_highlight.retrieval.video_metadata import probe_video

    settings = RetrievalSettings(base_url=VLLM_BASE_URL)
    config = settings.config()
    entries = _selected_entries(args.index, [str(value) for value in args.video_ids])
    frame_pattern = re.compile(r"\bframe=\s*(\d+)")
    rows = []
    for entry in entries:
        video_path = resolve_video_path(args.dataset_root, entry)
        meta = probe_video(video_path)
        chunks = build_chunks(
            meta.duration_sec,
            chunk_seconds=config.chunk_seconds,
            overlap_seconds=config.overlap_seconds,
        )
        fps = float(meta.fps or 0.0)
        with tempfile.TemporaryDirectory(prefix=".render-probe-") as temp_dir:
            for chunk in chunks:
                output = Path(temp_dir) / f"chunk-{chunk.index:05d}.mp4"
                command = [
                    "ffmpeg", "-v", "info", "-y",
                    "-ss", f"{chunk.start_sec:.9f}",
                    "-copyts",
                    "-i", str(video_path),
                    "-map", "0:v:0", "-map", "0:a?",
                    "-vf",
                    f"trim=start={chunk.start_sec:.9f}:end={chunk.end_sec:.9f},"
                    "showinfo,setpts=PTS-STARTPTS",
                    "-af",
                    f"atrim=start={chunk.start_sec:.9f}:end={chunk.end_sec:.9f},"
                    "asetpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac",
                    str(output),
                ]
                started = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                elapsed = time.perf_counter() - started
                matches = frame_pattern.findall(completed.stderr)
                decoded_frames = int(matches[-1]) if matches else None
                rows.append(
                    {
                        "video_id": entry.video_id,
                        "chunk_index": chunk.index,
                        "start_sec": chunk.start_sec,
                        "end_sec": chunk.end_sec,
                        "chunk_seconds": round(chunk.end_sec - chunk.start_sec, 6),
                        "source_fps": fps,
                        "decoded_frames": decoded_frames,
                        "expected_chunk_frames": round((chunk.end_sec - chunk.start_sec) * fps, 3),
                        "decode_amplification": (
                            round(decoded_frames / max((chunk.end_sec - chunk.start_sec) * fps, 1e-9), 4)
                            if decoded_frames is not None
                            else None
                        ),
                        "render_wall_sec": round(elapsed, 6),
                        "output_bytes": output.stat().st_size if output.is_file() else 0,
                        "returncode": completed.returncode,
                    }
                )
                print(
                    f"[render-probe] {entry.video_id} chunk={chunk.index} "
                    f"decoded={decoded_frames} expected={rows[-1]['expected_chunk_frames']} "
                    f"amplification={rows[-1]['decode_amplification']} wall={elapsed:.2f}s",
                    flush=True,
                )
    payload = {
        "schema": "aic.phd2.qwen-render-probe/v1",
        "dataset_root": str(args.dataset_root),
        "index": str(args.index),
        "chunk_seconds": config.chunk_seconds,
        "overlap_seconds": config.overlap_seconds,
        "video_count": len(entries),
        "chunk_count": len(rows),
        "mean_decode_amplification": (
            round(
                sum(row["decode_amplification"] for row in rows if row["decode_amplification"])
                / max(sum(1 for row in rows if row["decode_amplification"]), 1),
                6,
            )
        ),
        "rows": rows,
    }
    atomic_write_json(args.output, payload)
    return {
        "video_count": payload["video_count"],
        "chunk_count": payload["chunk_count"],
        "mean_decode_amplification": payload["mean_decode_amplification"],
        "output": str(args.output),
        "per_video_mean_amplification": {
            video_id: round(
                sum(row["decode_amplification"] for row in rows if row["video_id"] == video_id)
                / max(sum(1 for row in rows if row["video_id"] == video_id), 1),
                4,
            )
            for video_id in sorted({row["video_id"] for row in rows})
        },
    }


def _frozen_chunk_count(duration: float) -> int:
    import math

    return 1 if duration <= 30.0 else int(math.ceil((duration - 30.0) / 25.0)) + 1


def retrieval_stats_simple(root: str | Path) -> dict:
    """Sum the per-video retrieval artifact wall for a run root."""

    total_wall = 0.0
    chunks = 0
    videos = 0
    for path in sorted((Path(root) / "retrieval").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        total_wall += float(payload.get("timing", {}).get("artifact_wall_sec", 0.0))
        chunks += len(payload.get("chunks", []))
        videos += 1
    return {
        "video_count": videos,
        "chunk_count": chunks,
        "summed_wall_seconds": total_wall,
        "seconds_per_chunk": (total_wall / chunks) if chunks else None,
    }


def command_summarize(args) -> dict:
    """Aggregate the audit arms into one machine-readable round summary."""

    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    video_ids = [str(value) for value in subset["video_ids"]]
    arms = {
        key: json.loads((Path(root) / "arm_summary.json").read_text(encoding="utf-8"))
        for key, root in (item.split("=", 1) for item in args.arm)
    }
    required = {"c1_a", "c1_b", "c1_c"}
    missing = required - set(arms)
    if missing:
        raise RuntimeError(f"summary requires the determinism arms: {sorted(missing)}")

    def compare(left: str, right: str) -> dict:
        report = compare_audit_arms(
            Path(arms[left]["root"]), Path(arms[right]["root"]), arms[left]["video_ids"]
        )
        return {
            "left": left,
            "right": right,
            "chunk_count": report["chunk_count"],
            "chunk_raw_exact_match_count": report["chunk_raw_exact_match_count"],
            "chunk_raw_exact_match_rate": report["chunk_raw_exact_match_rate"],
            "chunk_semantic_match_rate": report["chunk_semantic_match_rate"],
            "video_candidate_semantic_match_rate": report["video_candidate_semantic_match_rate"],
            "all_chunks_exact": report["all_chunks_exact"],
            "all_videos_equal": report["all_videos_equal"],
            "mismatch_chunks": [
                {"video_id": row["video_id"], "chunk_index": row["chunk_index"]}
                for row in report["chunks"]
                if not row["raw_exact_match"]
            ][:20],
            "mismatch_videos": [
                row["video_id"] for row in report["videos"] if not row["merged_candidates_equal"]
            ][:20],
        }

    determinism = [compare("c1_a", "c1_b"), compare("c1_a", "c1_c")]
    determinism_pass = all(row["all_chunks_exact"] and row["all_videos_equal"] for row in determinism)

    def artifact_equivalence(arm_id: str) -> dict | None:
        if args.artifact_baseline_root is None:
            return None
        from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
            compare_retrieval_roots,
        )

        report = compare_retrieval_roots(
            args.artifact_baseline_root, Path(arms[arm_id]["root"]), video_ids
        )
        return {
            "arm": arm_id,
            "baseline_root": str(args.artifact_baseline_root),
            "pass": report["pass"],
            "rows": report["rows"],
        }

    challenger = None
    if args.challenger and args.challenger in arms:
        comparisons = []
        for control in ("c1_a", "c1_b", "c1_c"):
            if control not in arms:
                continue
            report = compare_audit_arms(
                Path(arms[args.challenger]["root"]), Path(arms[control]["root"]), video_ids
            )
            comparisons.append(
                {
                    "control": control,
                    "chunk_raw_exact_match_rate": report["chunk_raw_exact_match_rate"],
                    "chunk_semantic_match_rate": report["chunk_semantic_match_rate"],
                    "all_chunks_exact": report["all_chunks_exact"],
                    "all_videos_equal": report["all_videos_equal"],
                    "mismatch_chunks": [
                        {"video_id": row["video_id"], "chunk_index": row["chunk_index"]}
                        for row in report["chunks"]
                        if not row["raw_exact_match"]
                    ][:20],
                }
            )
        serial = arms["c1_a"]
        challenger = {
            "arm": args.challenger,
            "vs_controls": comparisons,
            "all_chunks_exact": all(row["all_chunks_exact"] for row in comparisons),
            "all_videos_equal": all(row["all_videos_equal"] for row in comparisons),
            "chunk_raw_exact_match_rate": min(
                row["chunk_raw_exact_match_rate"] for row in comparisons
            ),
            "summed_video_wall_speedup_vs_c1": (
                float(serial["sum_video_wall_sec"])
                / float(arms[args.challenger]["sum_video_wall_sec"])
            ),
            "per_chunk_wall_speedup_vs_c1": (
                (float(serial["sum_video_wall_sec"]) / max(int(serial["chunk_count"]), 1))
                / (
                    float(arms[args.challenger]["sum_video_wall_sec"])
                    / max(int(arms[args.challenger]["chunk_count"]), 1)
                )
            ),
            "summed_render_reduction_sec": float(serial["sum_render_sec"])
            - float(arms[args.challenger]["sum_render_sec"]),
        }

    def breakdown(arm_id: str) -> dict:
        arm = arms[arm_id]
        chunks = max(int(arm["chunk_count"]), 1)
        server = arm["server_request_decomposition"]
        request_total = float(arm["sum_request_sec"])
        render_total = float(arm["sum_render_sec"])
        inference_total = server.get("inference_time_sec_total")
        e2e_total = server.get("e2e_request_latency_sec_total")
        return {
            "chunk_count": int(arm["chunk_count"]),
            "video_count": int(arm["video_count"]),
            "wall_seconds": float(arm["run_wall_sec"]),
            "summed_video_wall_seconds": float(arm["sum_video_wall_sec"]),
            "chunks_per_second_summed": chunks / float(arm["sum_video_wall_sec"]),
            "per_chunk": {
                "chunk_render_sec": render_total / chunks,
                "client_request_sec": request_total / chunks,
                "probe_sec": float(arm["sum_probe_sec"]) / chunks,
                "parse_sec": float(arm["sum_parse_sec"]) / chunks,
                "merge_sec": float(arm["sum_merge_sec"]) / chunks,
                "prompt_tokens": float(arm["prompt_tokens"]) / chunks,
                "completion_tokens": float(arm["completion_tokens"]) / chunks,
            },
            "totals": {
                "chunk_render_sec": render_total,
                "client_request_sec": request_total,
                "probe_sec": float(arm["sum_probe_sec"]),
                "parse_sec": float(arm["sum_parse_sec"]),
                "merge_sec": float(arm["sum_merge_sec"]),
                "render_wait_sec": float(arm["sum_render_wait_sec"]),
                "non_primary_attempt_count": int(arm["non_primary_attempt_count"]),
            },
            "server": server,
            "derived": {
                "server_inference_sec_total": inference_total,
                "server_e2e_sec_total": e2e_total,
                "client_minus_server_e2e_sec": (
                    request_total - e2e_total if e2e_total is not None else None
                ),
                "server_input_processing_sec_total": (
                    e2e_total - inference_total
                    if (e2e_total is not None and inference_total is not None)
                    else None
                ),
                "gpu_busy_fraction_of_request": (
                    inference_total / request_total if inference_total else None
                ),
            },
        }

    full1500_chunks = None
    if args.full1500_manifest is not None:
        frozen = json.loads(args.full1500_manifest.read_text(encoding="utf-8"))
        full1500_chunks = sum(
            _frozen_chunk_count(float(row["actual_duration_s"])) for row in frozen["records"]
        )

    def scale_to_full1500(arm_id: str) -> dict | None:
        if not full1500_chunks:
            return None
        arm = arms[arm_id]
        chunks = max(int(arm["chunk_count"]), 1)
        row = breakdown(arm_id)
        per_chunk = row["per_chunk"]
        server = row["server"]
        inference = row["derived"]["server_inference_sec_total"]
        e2e = row["derived"]["server_e2e_sec_total"]
        return {
            "full1500_chunks": int(full1500_chunks),
            "qwen_seconds": float(arm["sum_video_wall_sec"]) / chunks * full1500_chunks,
            "qwen_hours": float(arm["sum_video_wall_sec"]) / chunks * full1500_chunks / 3600.0,
            "chunk_render_seconds": per_chunk["chunk_render_sec"] * full1500_chunks,
            "client_request_seconds": per_chunk["client_request_sec"] * full1500_chunks,
            "server_inference_seconds": (
                inference / chunks * full1500_chunks if inference is not None else None
            ),
            "server_prefill_seconds": (
                server.get("prefill_time_sec_total") / chunks * full1500_chunks
                if server.get("prefill_time_sec_total") is not None
                else None
            ),
            "server_decode_seconds": (
                server.get("decode_time_sec_total") / chunks * full1500_chunks
                if server.get("decode_time_sec_total") is not None
                else None
            ),
            "server_input_processing_seconds": (
                (e2e - inference) / chunks * full1500_chunks
                if (e2e is not None and inference is not None)
                else None
            ),
            "parse_merge_seconds": (
                per_chunk["parse_sec"] + per_chunk["merge_sec"]
            ) * full1500_chunks,
        }

    promoted_id = args.challenger if (challenger and challenger["all_chunks_exact"]) else "c1_a"
    promoted = breakdown(promoted_id)
    # c1_b carries the complete request-latency metric set; c1_a was produced
    # before the e2e metric was added to the extraction.
    serial_id = "c1_b"
    serial = breakdown(serial_id)

    benchmark_block = None
    if args.benchmark_arm:
        key, root = args.benchmark_arm
        if key not in arms:
            arm = json.loads((Path(root) / "arm_summary.json").read_text(encoding="utf-8"))
            arms[key] = arm
        arm = arms[key]
        bench_chunks = max(int(arm["chunk_count"]), 1)
        benchmark_block = breakdown(key)
        baseline_summed = None
        if args.artifact_baseline_root is not None:
            from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
                compare_retrieval_roots,
            )

            baseline_stats = retrieval_stats_simple(args.artifact_baseline_root)
            baseline_summed = baseline_stats["summed_wall_seconds"]
            baseline_chunks = baseline_stats["chunk_count"]
            equivalence = compare_retrieval_roots(
                args.artifact_baseline_root, Path(arm["root"]), arm["video_ids"]
            )
            benchmark_block["artifact_equivalence_vs_promoted_baseline"] = {
                "pass": equivalence["pass"],
                "video_count": len(equivalence["rows"]),
                "mismatch_videos": [
                    row["video_id"] for row in equivalence["rows"] if not row["semantic_equal"]
                ],
            }
            benchmark_block["serial_baseline_summed_wall_seconds"] = baseline_summed
            benchmark_block["serial_baseline_chunk_count"] = baseline_chunks
            benchmark_block["qwen_speedup_vs_serial_baseline"] = (
                baseline_summed / float(arm["sum_video_wall_sec"]) if arm["sum_video_wall_sec"] else None
            )
            benchmark_block["serial_seconds_per_chunk"] = baseline_summed / baseline_chunks
            benchmark_block["promoted_seconds_per_chunk"] = (
                float(arm["sum_video_wall_sec"]) / bench_chunks
            )
    summary = {
        "schema": "aic.phd2.qwen-audit-summary/v1",
        "subset": {
            "path": str(args.subset),
            "sha256": sha256_file(args.subset),
            "video_ids": video_ids,
            "count": len(video_ids),
        },
        "arms": {
            key: {
                "arm_id": value["arm_id"],
                "root": value["root"],
                "overlap_render": value["overlap_render"],
                "video_count": value["video_count"],
                "chunk_count": value["chunk_count"],
                "run_wall_sec": value["run_wall_sec"],
                "sum_video_wall_sec": value["sum_video_wall_sec"],
                "sum_render_sec": value["sum_render_sec"],
                "sum_request_sec": value["sum_request_sec"],
                "prompt_tokens": value["prompt_tokens"],
                "completion_tokens": value["completion_tokens"],
                "non_primary_attempt_count": value["non_primary_attempt_count"],
                "qwen_service_load_seconds": value["qwen_service_load_seconds"],
            }
            for key, value in arms.items()
        },
        "self_determinism": {
            "pairs": determinism,
            "verdict": "PASS" if determinism_pass else "FAIL",
            "chunk_exact_match_rate_min": min(
                row["chunk_raw_exact_match_rate"] for row in determinism
            ),
            "chunk_semantic_match_rate_min": min(
                row["chunk_semantic_match_rate"] for row in determinism
            ),
            "video_candidate_semantic_match_rate_min": min(
                row["video_candidate_semantic_match_rate"] for row in determinism
            ),
        },
        "promoted_baseline_artifact_equivalence": {
            "c1_a": artifact_equivalence("c1_a"),
            "c1_p": artifact_equivalence(args.challenger) if args.challenger else None,
        },
        "challenger": challenger,
        "wall_time_breakdown": {
            "c1_serial": {**serial, "reference_arm": serial_id},
            "c1_a": breakdown("c1_a"),
            "c1_b": breakdown("c1_b"),
            "promoted": {**promoted, "reference_arm": promoted_id},
            "benchmark32_promoted": benchmark_block,
            "full1500_c1_serial": scale_to_full1500(serial_id),
            "full1500_promoted": scale_to_full1500("bench32_p") if benchmark_block else scale_to_full1500(promoted_id),
            "full1500_basis_arm": "bench32_p" if benchmark_block else promoted_id,
        },
        "promoted_qwen_chunks_per_second": (
            benchmark_block["chunks_per_second_summed"]
            if benchmark_block
            else promoted["chunks_per_second_summed"]
        ),
        "promoted_qwen_rate_source": (
            f"Qwen audit arm {args.benchmark_arm[0]} over the full Benchmark32 (426 chunks), "
            "summed per-video retrieval wall"
            if benchmark_block
            else f"Qwen audit arm {promoted_id} summed per-video wall over the frozen "
            f"{len(video_ids)}-video determinism subset"
        ),
        "promoted_qwen_execution": (
            "concurrency=1 with one-chunk render look-ahead"
            if promoted_id != "c1_a"
            else "concurrency=1 serial (render overlap not promoted)"
        ),
        "qwen_server_args": arms["c1_a"].get("vllm_non_default_args"),
        "correctness_gates": {
            "self_determinism": "PASS" if determinism_pass else "FAIL",
            "artifact_semantic_equivalence_vs_promoted_baseline": (
                all(
                    row["pass"]
                    for row in (
                        artifact_equivalence("c1_a"),
                        artifact_equivalence(args.challenger) if args.challenger else None,
                    )
                    if row is not None
                )
                if args.artifact_baseline_root is not None
                else None
            ),
            "challenger_equivalence": challenger["all_chunks_exact"] if challenger else None,
        },
    }
    atomic_write_json(args.output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("subset")
    subset.add_argument("--benchmark-manifest", required=True, type=Path)
    subset.add_argument("--benchmark-sha256", required=True)
    subset.add_argument("--dataset-root", required=True, type=Path)
    subset.add_argument("--video-id", action="append", dest="video_ids", default=[])
    subset.add_argument("--all-benchmark-videos", action="store_true")
    subset.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--subset", required=True, type=Path)
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--index", required=True, type=Path)
    run.add_argument("--run-root", required=True, type=Path)
    run.add_argument("--arm-id", required=True)
    run.add_argument("--model-path", required=True, type=Path)
    run.add_argument("--overlap-render", action="store_true")
    run.add_argument("--start-service", action="store_true")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    run.add_argument("--max-model-len", type=int, default=32768)
    run.add_argument("--vllm-arg", action="append", default=None)
    run.add_argument("--vllm-log-path", type=Path, default=None)
    run.add_argument("--timeout-sec", type=float, default=1800.0)
    run.add_argument("--video-id", action="append", dest="video_ids")
    run.add_argument("--allow-unlisted-video", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-existing", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--subset", required=True, type=Path)
    compare.add_argument("--left-root", required=True, type=Path)
    compare.add_argument("--right-root", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument(
        "--artifact-baseline-root",
        type=Path,
        default=None,
        help="optional promoted baseline root for artifact-level semantic comparison",
    )
    compare.add_argument("--video-id", action="append", dest="video_ids")

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--output", required=True, type=Path)
    metrics.add_argument("--label", default="snapshot")

    render_probe = subparsers.add_parser("render-probe")
    render_probe.add_argument("--dataset-root", required=True, type=Path)
    render_probe.add_argument("--index", required=True, type=Path)
    render_probe.add_argument("--video-id", action="append", dest="video_ids", required=True)
    render_probe.add_argument("--output", required=True, type=Path)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--subset", required=True, type=Path)
    summarize.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="ID=ROOT",
        help="arm_id=run_root (arm_summary.json is read from the root)",
    )
    summarize.add_argument("--challenger", default=None)
    summarize.add_argument(
        "--benchmark-arm",
        nargs=2,
        metavar=("ID", "ROOT"),
        default=None,
        help="promoted arm measured over the full Benchmark32 (official ETA basis)",
    )
    summarize.add_argument("--artifact-baseline-root", type=Path, default=None)
    summarize.add_argument("--full1500-manifest", type=Path, default=None)
    summarize.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "subset":
        _print(command_subset(args))
        return 0
    if args.command == "run":
        _print(command_run(args))
        return 0
    if args.command == "compare":
        report = command_compare(args)
        _print(report)
        return 0 if report["all_videos_equal"] else 3
    if args.command == "render-probe":
        _print(command_render_probe(args))
        return 0
    if args.command == "summarize":
        summary = command_summarize(args)
        _print(
            {
                "output": str(args.output),
                "self_determinism": summary["self_determinism"]["verdict"],
                "promoted_arm": summary["challenger"]["arm"] if summary["challenger"] and summary["challenger"]["all_chunks_exact"] else "c1_a",
                "promoted_qwen_chunks_per_second": summary["promoted_qwen_chunks_per_second"],
                "full1500_c1_serial_hours": (
                    summary["wall_time_breakdown"]["full1500_c1_serial"] or {}
                ).get("qwen_hours"),
                "full1500_promoted_hours": (
                    summary["wall_time_breakdown"]["full1500_promoted"] or {}
                ).get("qwen_hours"),
            }
        )
        return 0
    payload = fetch_prometheus_metrics(VLLM_METRICS_URL)
    payload["label"] = args.label
    atomic_write_json(args.output, payload)
    _print({"label": args.label, "families": len(payload["metric_sum"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
