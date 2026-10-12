#!/usr/bin/env python3
"""PHD2 Full1500 Qwen retrieval production run (retrieval only).

Subcommands
-----------
verify-raw : validate the frozen 1500 raw manifest against the uploaded raw tree
plan       : print the planned workload (videos, expected chunks, ETA at the
             measured promoted throughput) without starting a service
run        : detached production run (resume-safe, failure-isolated)
status     : print progress.json
verify     : audit the produced artifact set (missing / duplicate / corrupt)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aic_video_highlight.ftnet.phd2_qwen1500 import (
    PROMPT_VERSION,
    QWEN_MODEL,
    QWEN_REVISION,
    RUN_MANIFEST_SCHEMA,
    RUN_SUMMARY_SCHEMA,
    Qwen1500Error,
    RunCounters,
    append_jsonl,
    artifact_paths,
    atomic_json,
    chunk_count_for_duration,
    entries_from_records,
    is_systemic_error,
    load_raw1500_manifest,
    sha256_file,
    validate_artifact,
)
from aic_video_highlight.ftnet.real_provider import (
    QWEN_MODEL as PROVIDER_QWEN_MODEL,
    RetrievalSettings,
    run_retrieval_for_entry,
)
from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient
from aic_video_highlight.runtime.orchestrator import start_vllm, stop_vllm
from aic_video_highlight.runtime.paths import EnvironmentPaths

FROZEN_SERVER = {
    "vllm_gpu_memory_utilization": 0.80,
    "vllm_max_model_len": 32768,
    "vllm_start_timeout_sec": 900,
}
MEASURED_PROMOTED_CHUNKS_PER_SECOND = 0.187806


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


def _service_has_model(base_url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as response:
            payload = json.load(response)
        return any(item.get("id") == QWEN_MODEL for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def command_verify_raw(args) -> dict:
    records, meta = load_raw1500_manifest(
        args.raw_manifest,
        expected_sha256=args.raw_manifest_sha256,
        dataset_root=args.dataset_root,
        verify_file_sizes=not args.skip_size_check,
        expected_records=args.expected_records,
    )
    expected = sum(
        chunk_count_for_duration(record.duration_sec, args.chunk_seconds, args.overlap_seconds)
        for record in records
    )
    result = {
        **meta,
        "manifest_sha256_pass": meta["manifest_sha256"] == args.raw_manifest_sha256,
        "records": len(records),
        "expected_chunks_30s_5s": expected,
        "total_gib": round(meta["total_bytes"] / 1024**3, 4),
        "arms": {arm: sum(1 for record in records if record.arm == arm) for arm in sorted({r.arm for r in records})},
        "codecs": {codec: sum(1 for record in records if record.vcodec == codec) for codec in sorted({r.vcodec for r in records})},
        "verify_raw": "PASS",
    }
    _print(result)
    return result


def command_plan(args) -> dict:
    records, meta = load_raw1500_manifest(
        args.raw_manifest,
        expected_sha256=args.raw_manifest_sha256,
        dataset_root=args.dataset_root,
        expected_records=args.expected_records,
    )
    expected = sum(chunk_count_for_duration(record.duration_sec) for record in records)
    rate = MEASURED_PROMOTED_CHUNKS_PER_SECOND
    plan = {
        "videos": len(records),
        "expected_chunks": expected,
        "measured_promoted_chunks_per_second": rate,
        "projected_qwen_seconds": expected / rate,
        "projected_qwen_hours": expected / rate / 3600.0,
        "execution": "concurrency=1 + one-chunk render look-ahead",
        "manifest": meta,
    }
    _print(plan)
    return plan


def _progress_payload(counters: RunCounters, expected_chunks_total: int) -> dict:
    payload = counters.to_dict()
    throughput = counters.chunks / counters.elapsed_seconds if counters.elapsed_seconds > 0 else None
    remaining = max(expected_chunks_total - counters.chunks, 0)
    eta = remaining / throughput if throughput else None
    payload["expected_chunks_total"] = expected_chunks_total
    payload["remaining_expected_chunks"] = remaining
    payload["eta_seconds_chunk_basis"] = eta
    payload["eta_hours_chunk_basis"] = eta / 3600.0 if eta else None
    return payload


def _gpu_snapshot() -> dict:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).splitlines()[0]
        utilization, memory, power = [item.strip() for item in raw.split(",")]
        return {"gpu_util_percent": utilization, "vram_used_mib": memory, "gpu_power_w": power}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {}


def command_run(args) -> dict:
    records, meta = load_raw1500_manifest(
        args.raw_manifest,
        expected_sha256=args.raw_manifest_sha256,
        dataset_root=args.dataset_root,
        expected_records=args.expected_records,
    )
    entries = entries_from_records(records, dataset_root=args.dataset_root)
    root = args.run_root.resolve()
    for name in ("retrieval", "raw_responses", "manifest", "logs", "summary", "failures", "resume_state"):
        (root / name).mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    progress_jsonl = root / "progress.jsonl"
    log_path = root / "logs" / "run.log"
    expected_chunks_total = sum(chunk_count_for_duration(record.duration_sec) for record in records)

    code_identity = {
        "real_provider.py": sha256_file(Path(__file__).resolve().parents[3] / "src/aic_video_highlight/ftnet/real_provider.py"),
        "qwen_execution.py": sha256_file(Path(__file__).resolve().parents[3] / "src/aic_video_highlight/ftnet/qwen_execution.py"),
        "phd2_qwen1500.py": sha256_file(Path(__file__).resolve().parents[3] / "src/aic_video_highlight/ftnet/phd2_qwen1500.py"),
        "qwen_vllm_client.py": sha256_file(Path(__file__).resolve().parents[3] / "src/aic_video_highlight/retrieval/qwen_vllm_client.py"),
        "configs/highlight_retrieval.yaml": sha256_file(Path(__file__).resolve().parents[3] / "configs/highlight_retrieval.yaml"),
    }
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": args.run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "QWEN_RETRIEVAL_ONLY",
        "forbidden_stages_not_run": [
            "RT-DETR", "detection", "visual256", "LOC", "CMP", "TS", "native16",
            "pairwise_pu_supervision", "ftnet",
        ],
        "model": {"id": PROVIDER_QWEN_MODEL, "revision": QWEN_REVISION, "prompt": PROMPT_VERSION},
        "execution": {
            "qwen_request_concurrency": 1,
            "overlap_render": bool(args.overlap_render),
            "chunk_level_recovery": True,
            "whole_video_rerun_on_chunk_failure": False,
            "client_reuse": "single_shared_openai_client_for_run",
            "request_order": "manifest_order",
        },
        "frozen_sampling": {
            "chunk_seconds": 30.0,
            "overlap_seconds": 5.0,
            "coarse_fps": 2.0,
            "temperature": 0.0,
            "enable_thinking": False,
            "max_new_tokens_primary": 512,
            "merge_tiou_threshold": 0.5,
        },
        "server": {"base_url": args.base_url, **FROZEN_SERVER},
        "raw_manifest": meta,
        "expected_chunks": expected_chunks_total,
        "code_identity": code_identity,
    }
    atomic_json(root / "manifest" / "run_manifest.json", run_manifest)
    atomic_json(root / "manifest" / "raw_manifest_identity.json", meta)

    log_handle = log_path.open("a", encoding="utf-8", newline="\n")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"=== run start run_id={args.run_id} videos={len(entries)} expected_chunks={expected_chunks_total} ===")
    log(f"execution: concurrency=1 overlap_render={args.overlap_render} client=shared")

    environment = EnvironmentPaths.from_json(args.environment)
    owned = None
    service_started = None
    if not _service_has_model(args.base_url):
        config = {
            "inference": {
                "python": os.sys.executable,
                "qwen_snapshot": {"base": "models", "path": "Qwen3.5-4B"},
                "video_root": {"base": "datasets", "path": str(args.dataset_root)},
                "vllm_base_url": args.base_url,
                **FROZEN_SERVER,
            }
        }
        service_started = time.perf_counter()
        owned, base_url = start_vllm(config, environment, root / "logs" / "vllm.log")
        log(f"vLLM started in {time.perf_counter() - service_started:.1f}s (owned={owned is not None})")
    else:
        base_url = args.base_url
        log("vLLM already healthy; reusing the running service")

    settings = RetrievalSettings(base_url=base_url, timeout_sec=1800.0)
    client = QwenVLLMClient(base_url=base_url, model=QWEN_MODEL, timeout_sec=1800.0)

    counters = RunCounters(
        videos_total=len(entries),
        expected_chunks=expected_chunks_total,
        started_monotonic=time.monotonic(),
    )
    # resume pass: count already-valid artifacts
    for entry in entries:
        artifact, _sidecar = artifact_paths(root, entry.video_id)
        ok, _reason = validate_artifact(artifact, entry.video_id)
        if ok:
            counters.skipped += 1
            counters.completed += 1
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            counters.chunks += len(payload.get("chunks", []))
    log(f"resume pre-scan: completed={counters.completed} chunks={counters.chunks}")

    stop_reason = None
    health_every = 20
    rounds = max(int(args.retry_failed_rounds), 0) + 1
    for round_index in range(rounds):
        pending = [
            entry
            for entry in entries
            if not validate_artifact(artifact_paths(root, entry.video_id)[0], entry.video_id)[0]
        ]
        if round_index > 0:
            log(f"=== retry round {round_index}/{rounds - 1}: {len(pending)} videos pending ===")
            if not pending:
                break
        if not pending:
            log("no pending videos; artifact set already complete")
            break
        for position, entry in enumerate(pending, 1):
            index = entries.index(entry) + 1
            artifact, sidecar = artifact_paths(root, entry.video_id)
            started = time.perf_counter()
            status = "completed"
            error_message = None
            chunks = 0
            retries = 0
            try:
                payload = run_retrieval_for_entry(
                    entry,
                    dataset_root=args.dataset_root,
                    client=client,
                    settings=settings,
                    overlap_render=args.overlap_render,
                    audit_path=sidecar,
                )
                atomic_json(artifact, payload)
                valid, why = validate_artifact(artifact, entry.video_id)
                if not valid:
                    raise Qwen1500Error(f"artifact failed post-write validation: {why}")
                chunks = len(payload.get("chunks", []))
                retries = sum(
                    1
                    for row in payload.get("provenance", {}).get("recovery_log", [])
                    if isinstance(row, dict) and row.get("attempt")
                )
                counters.chunks += chunks
                counters.retries += retries
                counters.completed += 1
                if counters.skipped > 0:
                    counters.skipped = max(counters.skipped - 1, 0)
            except Exception as exc:  # noqa: BLE001 - failure isolation per video
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                counters.failed += 1
                atomic_json(
                    root / "failures" / f"{entry.video_id}.json",
                    {
                        "video_id": entry.video_id,
                        "relative_path": entry.relative_video_path,
                        "error": error_message,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                log(f"[{index}/{len(entries)}] FAILED video={entry.video_id} error={error_message[:200]}")
            elapsed_video = time.perf_counter() - started
            counters.elapsed_seconds = time.monotonic() - counters.started_monotonic
            progress = _progress_payload(counters, expected_chunks_total)
            progress["current_video"] = entry.video_id
            progress["last_video_status"] = status
            progress["last_video_elapsed_seconds"] = round(elapsed_video, 3)
            progress["last_video_chunks"] = chunks
            progress["last_video_retries"] = retries
            progress["last_video_error"] = error_message
            progress.update(_gpu_snapshot())
            atomic_json(progress_path, progress)
            append_jsonl(
                progress_jsonl,
                {
                    "index": index,
                    "video_id": entry.video_id,
                    "status": status,
                    "chunks": chunks,
                    "retries": retries,
                    "elapsed_video_seconds": round(elapsed_video, 3),
                    "completed": counters.completed,
                    "failed": counters.failed,
                    "total_chunks": counters.chunks,
                    "elapsed_seconds": round(counters.elapsed_seconds, 3),
                    "chunks_per_second": progress.get("chunks_per_second"),
                    "eta_hours": progress.get("eta_hours_chunk_basis"),
                    "error": error_message,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            if status == "completed":
                log(
                    f"[{index}/{len(entries)}] video={entry.video_id} chunks={chunks} "
                    f"elapsed={elapsed_video:.1f}s total_chunks={counters.chunks} "
                    f"failed={counters.failed} chunks/s={progress.get('chunks_per_second')} "
                    f"ETA={progress.get('eta_hours_chunk_basis')}h "
                    f"gpu={progress.get('gpu_util_percent')}% vram={progress.get('vram_used_mib')}MiB"
                )
            if error_message and is_systemic_error(error_message):
                stop_reason = f"systemic error: {error_message[:200]}"
                log(f"STOPPING RUN: {stop_reason}")
                break
            if index % health_every == 0 and not _service_has_model(base_url):
                log("vLLM health check failed; attempting restart")
                try:
                    if owned is not None:
                        stop_vllm(owned)
                    config = {
                        "inference": {
                            "python": os.sys.executable,
                            "qwen_snapshot": {"base": "models", "path": "Qwen3.5-4B"},
                            "video_root": {"base": "datasets", "path": str(args.dataset_root)},
                            "vllm_base_url": args.base_url,
                            **FROZEN_SERVER,
                        }
                    }
                    owned, base_url = start_vllm(config, environment, root / "logs" / "vllm.log")
                    client = QwenVLLMClient(base_url=base_url, model=QWEN_MODEL, timeout_sec=1800.0)
                    log("vLLM restarted and healthy")
                except Exception as exc:  # noqa: BLE001 - unrecoverable service failure
                    stop_reason = f"vLLM restart failed: {type(exc).__name__}: {exc}"
                    log(f"STOPPING RUN: {stop_reason}")
                    break

    counters.elapsed_seconds = time.monotonic() - counters.started_monotonic
    final_progress = _progress_payload(counters, expected_chunks_total)
    final_progress["stop_reason"] = stop_reason
    atomic_json(progress_path, final_progress)

    missing, duplicate, corrupt = audit_artifacts(root, [entry.video_id for entry in entries])
    summary = {
        "schema": RUN_SUMMARY_SCHEMA,
        "run_id": args.run_id,
        "stage": "QWEN_RETRIEVAL_ONLY",
        "status": "COMPLETE" if not missing and not corrupt else "INCOMPLETE",
        "stop_reason": stop_reason,
        "videos_total": len(entries),
        "completed": counters.completed,
        "failed": counters.failed,
        "skipped_from_cache": counters.skipped,
        "total_chunks": counters.chunks,
        "expected_chunks": expected_chunks_total,
        "retries": counters.retries,
        "wall_seconds": counters.elapsed_seconds,
        "wall_hours": counters.elapsed_seconds / 3600.0,
        "chunks_per_second": counters.chunks / counters.elapsed_seconds if counters.elapsed_seconds else None,
        "missing": len(missing),
        "duplicate": len(duplicate),
        "corrupt": len(corrupt),
        "missing_ids": missing[:20],
        "corrupt_ids": corrupt[:20],
        "execution": run_manifest["execution"],
        "code_identity": code_identity,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / "summary" / "run_summary.json", summary)
    log(
        f"=== run end status={summary['status']} completed={summary['completed']} failed={summary['failed']} "
        f"chunks={summary['total_chunks']} wall={summary['wall_hours']:.3f}h missing={summary['missing']} "
        f"corrupt={summary['corrupt']} ==="
    )
    if owned is not None:
        stop_vllm(owned)
    log_handle.close()
    _print(summary)
    return summary


def audit_artifacts(root: Path, video_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    retrieval_dir = root / "retrieval"
    present = {path.stem for path in retrieval_dir.glob("*.json")} if retrieval_dir.is_dir() else set()
    expected = set(video_ids)
    missing = sorted(expected - present)
    duplicate = sorted(present - expected)
    corrupt = []
    for video_id in sorted(expected & present):
        valid, _why = validate_artifact(retrieval_dir / f"{video_id}.json", video_id)
        if not valid:
            corrupt.append(video_id)
    return missing, duplicate, corrupt


def command_status(args) -> dict:
    root = args.run_root.resolve()
    progress_path = root / "progress.json"
    if not progress_path.is_file():
        raise Qwen1500Error(f"no progress.json at {progress_path}")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    summary_path = root / "summary" / "run_summary.json"
    payload["summary_present"] = summary_path.is_file()
    if payload["summary_present"]:
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    _print(payload)
    return payload


def command_verify(args) -> dict:
    records, _meta = load_raw1500_manifest(
        args.raw_manifest,
        expected_sha256=args.raw_manifest_sha256,
        dataset_root=args.dataset_root,
        expected_records=args.expected_records,
    )
    ids = [record.video_id for record in records]
    missing, duplicate, corrupt = audit_artifacts(args.run_root.resolve(), ids)
    result = {
        "videos": len(ids),
        "artifacts_present": len(ids) - len(missing),
        "missing": len(missing),
        "missing_ids": missing[:20],
        "duplicate": len(duplicate),
        "duplicate_ids": duplicate[:20],
        "corrupt": len(corrupt),
        "corrupt_ids": corrupt[:20],
        "status": "COMPLETE" if not missing and not duplicate and not corrupt else "INCOMPLETE",
    }
    _print(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, *, with_run_root: bool) -> None:
        sub.add_argument("--raw-manifest", required=True, type=Path)
        sub.add_argument("--raw-manifest-sha256", required=True)
        sub.add_argument("--dataset-root", required=True, type=Path)
        sub.add_argument(
            "--expected-records",
            type=int,
            default=1500,
            help="frozen record count (1500 for PHD2 Full1500; smaller only for smoke tests)",
        )
        if with_run_root:
            sub.add_argument("--run-root", required=True, type=Path)

    verify_raw = subparsers.add_parser("verify-raw")
    add_common(verify_raw, with_run_root=False)
    verify_raw.add_argument("--chunk-seconds", type=float, default=30.0)
    verify_raw.add_argument("--overlap-seconds", type=float, default=5.0)
    verify_raw.add_argument("--skip-size-check", action="store_true")

    plan = subparsers.add_parser("plan")
    add_common(plan, with_run_root=False)

    run = subparsers.add_parser("run")
    add_common(run, with_run_root=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--environment", required=True, type=Path)
    run.add_argument("--overlap-render", action="store_true")
    run.add_argument(
        "--retry-failed-rounds",
        type=int,
        default=2,
        help="failed-only retry rounds after the main pass (frozen recovery is stochastic)",
    )
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True, type=Path)

    verify = subparsers.add_parser("verify")
    add_common(verify, with_run_root=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-raw":
        command_verify_raw(args)
        return 0
    if args.command == "plan":
        command_plan(args)
        return 0
    if args.command == "run":
        command_run(args)
        return 0
    if args.command == "status":
        command_status(args)
        return 0
    result = command_verify(args)
    return 0 if result["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
