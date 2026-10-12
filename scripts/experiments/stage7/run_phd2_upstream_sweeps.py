#!/usr/bin/env python3
"""Run cache-aware Qwen concurrency and RT-DETR engineering sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

from aic_video_highlight.ftnet.phd2_upstream_benchmark import (
    PhaseState,
    ProfileEvents,
    SystemTelemetry,
    atomic_json,
    compare_detection_roots,
    compare_retrieval_roots,
    ensure_positive_omp_threads,
    load_benchmark_manifest,
    run_detection,
    run_retrieval,
)
from aic_video_highlight.ftnet.performance_decode import DecodeCounters, iter_decoded_batches
from aic_video_highlight.ftnet.real_provider import (
    RTDETR_MODEL,
    build_grid_for_entry,
    create_rtdetr_localizer,
)
from aic_video_highlight.ftnet.index import load_index, resolve_video_path
from aic_video_highlight.runtime.orchestrator import start_vllm, stop_vllm
from aic_video_highlight.runtime.paths import EnvironmentPaths


def _write_profile(root: Path, events: ProfileEvents, total_wall: float) -> None:
    events.write_csv(root / "stage_events.csv")
    rows = events.summary(total_wall)
    with (root / "stage_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ("stage",))
        writer.writeheader()
        writer.writerows(rows)


def _telemetry_summary(path: Path, phase: str | None = None) -> dict[str, float | None]:
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if phase is not None:
        rows = [row for row in rows if row.get("phase") == phase]

    def values(name: str) -> list[float]:
        result = []
        for row in rows:
            try:
                result.append(float(row[name]))
            except (KeyError, TypeError, ValueError):
                pass
        return result

    result: dict[str, float | None] = {}
    for name in ("gpu_util_percent", "vram_used_mib", "gpu_power_w", "cpu_util_percent"):
        series = values(name)
        result[f"{name}_mean"] = sum(series) / len(series) if series else None
        result[f"{name}_peak"] = max(series) if series else None
    return result


def _selected_entries(index_path: Path, ids: list[str]):
    entries, _ = load_index(index_path)
    mapping = {entry.video_id: entry for entry in entries}
    missing = [video_id for video_id in ids if video_id not in mapping]
    if missing:
        raise RuntimeError(f"sweep ids missing from index: {missing}")
    return [mapping[video_id] for video_id in ids]


def _source_duration(benchmark: dict, ids: list[str]) -> float:
    records = {str(row["youtubeId"]): row for row in benchmark["records"]}
    return sum(float(records[video_id]["duration_seconds"]) for video_id in ids)


def _baseline_qwen_arm(baseline_root: Path, ids: list[str], source_duration: float) -> dict:
    wall = 0.0
    chunks = 0
    for video_id in ids:
        payload = json.loads(
            (baseline_root / "retrieval" / f"{video_id}.json").read_text(encoding="utf-8")
        )
        wall += float(payload["timing"]["artifact_wall_sec"])
        chunks += len(payload.get("chunks", []))
    return {
        "concurrency": 1,
        "status": "PASS_BASELINE_CACHE",
        "root": str(baseline_root),
        "wall_seconds": wall,
        "chunk_count": chunks,
        "chunks_per_second": chunks / wall,
        "source_video_min_per_wall_min": source_duration / wall,
        "semantic_equivalence": True,
        "telemetry": _telemetry_summary(
            baseline_root / "system_metrics.csv", "qwen_retrieval"
        ),
    }


def qwen_sweep(args, benchmark: dict, entries) -> dict:
    ids = [entry.video_id for entry in entries]
    source_duration = _source_duration(benchmark, ids)
    output_root = args.output_root / "qwen_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    arms = [_baseline_qwen_arm(args.baseline_root, ids, source_duration)]
    environment = EnvironmentPaths.from_json(args.environment)
    base_url = "http://127.0.0.1:8000/v1"
    config = {
        "inference": {
            "python": os.sys.executable,
            "qwen_snapshot": {"base": "models", "path": "Qwen3.5-4B"},
            "video_root": {"base": "datasets", "path": str(args.dataset_root)},
            "vllm_base_url": base_url,
            "vllm_gpu_memory_utilization": 0.80,
            "vllm_max_model_len": 32768,
            "vllm_start_timeout_sec": 900,
        }
    }
    service_started = time.perf_counter()
    owned, base_url = start_vllm(config, environment, output_root / "vllm.log")
    service_load_seconds = time.perf_counter() - service_started
    service_start_count = 1 if owned is not None else 0
    try:
        requested = [2, 4, 8]
        for concurrency in requested:
            arm_root = output_root / f"c{concurrency}"
            arm_root.mkdir(parents=True, exist_ok=True)
            events = ProfileEvents()
            phase = PhaseState()
            telemetry = SystemTelemetry(arm_root / "system_metrics.csv", phase)
            telemetry.start()
            started = time.perf_counter()
            try:
                summary = run_retrieval(
                    entries,
                    dataset_root=args.dataset_root,
                    output_root=arm_root,
                    base_url=base_url,
                    concurrency=concurrency,
                    timeout_sec=1800.0,
                    events=events,
                    phase=phase,
                )
                elapsed = time.perf_counter() - started
                correctness = compare_retrieval_roots(args.baseline_root, arm_root, ids)
                row = {
                    "concurrency": concurrency,
                    "status": "PASS" if correctness["pass"] else "REJECTED_CORRECTNESS",
                    "root": str(arm_root),
                    "wall_seconds": summary["wall_seconds"],
                    "chunk_count": summary["chunk_count"],
                    "chunks_per_second": summary["chunk_count"] / summary["wall_seconds"],
                    "source_video_min_per_wall_min": source_duration / summary["wall_seconds"],
                    "semantic_equivalence": correctness["pass"],
                    "correctness": correctness,
                }
            except Exception as exc:  # noqa: BLE001 - failed arm is rejected, not fatal
                elapsed = time.perf_counter() - started
                row = {
                    "concurrency": concurrency,
                    "status": f"REJECTED_{type(exc).__name__}",
                    "error": str(exc),
                    "root": str(arm_root),
                    "wall_seconds": elapsed,
                    "semantic_equivalence": False,
                }
            finally:
                phase.set("finalize")
                telemetry.stop()
            row["telemetry"] = _telemetry_summary(arm_root / "system_metrics.csv", "qwen_retrieval")
            _write_profile(arm_root, events, elapsed)
            atomic_json(arm_root / "arm_summary.json", row)
            arms.append(row)

        arm4 = next(row for row in arms if row["concurrency"] == 4)
        arm8 = next(row for row in arms if row["concurrency"] == 8)
        growth = (
            float(arm8.get("chunks_per_second", 0.0))
            / max(float(arm4.get("chunks_per_second", 0.0)), 1e-12)
            - 1.0
        )
        peak = arm8.get("telemetry", {}).get("vram_used_mib_peak")
        test_16 = (
            arm8["status"] == "PASS"
            and growth >= 0.05
            and peak is not None
            and float(peak) < 23500.0
        )
        if test_16:
            concurrency = 16
            arm_root = output_root / "c16"
            arm_root.mkdir(parents=True, exist_ok=True)
            events = ProfileEvents()
            phase = PhaseState()
            telemetry = SystemTelemetry(arm_root / "system_metrics.csv", phase)
            telemetry.start()
            started = time.perf_counter()
            try:
                summary = run_retrieval(
                    entries,
                    dataset_root=args.dataset_root,
                    output_root=arm_root,
                    base_url=base_url,
                    concurrency=16,
                    timeout_sec=1800.0,
                    events=events,
                    phase=phase,
                )
                elapsed = time.perf_counter() - started
                correctness = compare_retrieval_roots(args.baseline_root, arm_root, ids)
                row = {
                    "concurrency": 16,
                    "status": "PASS" if correctness["pass"] else "REJECTED_CORRECTNESS",
                    "root": str(arm_root),
                    "wall_seconds": summary["wall_seconds"],
                    "chunk_count": summary["chunk_count"],
                    "chunks_per_second": summary["chunk_count"] / summary["wall_seconds"],
                    "source_video_min_per_wall_min": source_duration / summary["wall_seconds"],
                    "semantic_equivalence": correctness["pass"],
                    "correctness": correctness,
                }
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - started
                row = {
                    "concurrency": 16,
                    "status": f"REJECTED_{type(exc).__name__}",
                    "error": str(exc),
                    "root": str(arm_root),
                    "wall_seconds": elapsed,
                    "semantic_equivalence": False,
                }
            finally:
                phase.set("finalize")
                telemetry.stop()
            row["telemetry"] = _telemetry_summary(arm_root / "system_metrics.csv", "qwen_retrieval")
            _write_profile(arm_root, events, elapsed)
            atomic_json(arm_root / "arm_summary.json", row)
            arms.append(row)
    finally:
        if owned is not None:
            stop_vllm(owned)
    passing = [row for row in arms if str(row["status"]).startswith("PASS")]
    selected = max(passing, key=lambda row: float(row.get("chunks_per_second", 0.0)))
    report = {
        "subset_ids": ids,
        "source_duration_seconds": source_duration,
        "qwen_service_start_count": service_start_count,
        "qwen_service_load_seconds": service_load_seconds,
        "arms": arms,
        "selected_concurrency": selected["concurrency"],
        "selected_speedup_vs_c1": float(selected["chunks_per_second"])
        / float(arms[0]["chunks_per_second"]),
        "concurrency16_tested": any(row["concurrency"] == 16 for row in arms),
    }
    atomic_json(output_root / "sweep_summary.json", report)
    return report


def _baseline_decode_arm(baseline_root: Path, ids: list[str]) -> dict:
    rows = [
        json.loads((baseline_root / "detection_profiles" / f"{video_id}.json").read_text(encoding="utf-8"))
        for video_id in ids
    ]
    frames = sum(int(row["frames"]) for row in rows)
    wall = sum(float(row["wall_sec"]) for row in rows)
    decode = sum(float(row["profile"]["decode_wall_seconds"]) for row in rows)
    return {
        "name": "historical_batch8",
        "batch_size": 8,
        "decode_mode": "windowed_prefix",
        "status": "PASS_BASELINE_CACHE",
        "root": str(baseline_root),
        "frame_count": frames,
        "wall_seconds": wall,
        "decode_wall_seconds": decode,
        "end_to_end_fps": frames / wall,
        "video_capture_open_count": sum(int(row["profile"]["capture_open_count"]) for row in rows),
        "grabbed_frame_count": sum(int(row["profile"]["grabbed_frame_count"]) for row in rows),
        "retrieved_frame_count": sum(int(row["profile"]["retrieved_frame_count"]) for row in rows),
        "repeated_prefix_decoded_frame_estimate": sum(
            int(row["profile"]["repeated_prefix_decoded_frame_estimate"]) for row in rows
        ),
    }


def _rgb_checksums(entries, dataset_root: Path, mode: str) -> dict[str, dict]:
    result = {}
    for entry in entries:
        digest = hashlib.sha256()
        counters = DecodeCounters()
        sample = build_grid_for_entry(entry)
        for frame_ids, images in iter_decoded_batches(
            resolve_video_path(dataset_root, entry),
            sample.frames,
            batch_size=64,
            decode_window=64,
            mode=mode,
            counters=counters,
        ):
            for frame_id, image in zip(frame_ids, images):
                digest.update(int(frame_id).to_bytes(8, "little", signed=False))
                digest.update(str(tuple(image.shape)).encode("ascii"))
                digest.update(image.tobytes(order="C"))
        result[entry.video_id] = {
            "sha256": digest.hexdigest(),
            "emitted_frames": counters.emitted_frame_count,
            "capture_open_count": counters.capture_open_count,
            "grabbed_frame_count": counters.grabbed_frame_count,
        }
    return result


def rtdetr_sweep(args, entries) -> dict:
    import torch

    ids = [entry.video_id for entry in entries]
    output_root = args.output_root / "rtdetr_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    environment = EnvironmentPaths.from_json(args.environment)
    baseline = _baseline_decode_arm(args.baseline_root, ids)
    checksum_entries = [entries[0]]
    v720 = next((entry for entry in entries if entry.category == "v720"), None)
    if v720 is not None and v720.video_id != checksum_entries[0].video_id:
        checksum_entries.append(v720)
    historical_rgb = _rgb_checksums(checksum_entries, args.dataset_root, "windowed_prefix")
    onepass_rgb = _rgb_checksums(checksum_entries, args.dataset_root, "one_pass")
    rgb_checksum_pass = all(
        historical_rgb[video_id]["sha256"] == onepass_rgb[video_id]["sha256"]
        and historical_rgb[video_id]["emitted_frames"] == onepass_rgb[video_id]["emitted_frames"]
        for video_id in historical_rgb
    )
    load_started = time.perf_counter()
    localizer = create_rtdetr_localizer(
        rtdetr_snapshot=environment.models / "rtdetr_r50vd",
        rtdetr_model_id=RTDETR_MODEL,
        device="cuda",
    )
    load_seconds = time.perf_counter() - load_started
    arms = []
    try:
        pending = [8, 1, 4, 16, 32]
        position = 0
        while position < len(pending):
            batch_size = pending[position]
            position += 1
            arm_root = output_root / f"onepass_b{batch_size}"
            arm_root.mkdir(parents=True, exist_ok=True)
            events = ProfileEvents()
            phase = PhaseState()
            telemetry = SystemTelemetry(arm_root / "system_metrics.csv", phase)
            telemetry.start()
            started = time.perf_counter()
            try:
                summary = run_detection(
                    entries,
                    dataset_root=args.dataset_root,
                    output_root=arm_root,
                    rtdetr_snapshot=environment.models / "rtdetr_r50vd",
                    batch_size=batch_size,
                    decode_window=64,
                    decode_mode="one_pass",
                    events=events,
                    phase=phase,
                    localizer=localizer,
                    model_load_seconds=load_seconds,
                )
                elapsed = time.perf_counter() - started
                reference = arm_root if batch_size == 8 else output_root / "onepass_b8"
                correctness = (
                    compare_detection_roots(args.baseline_root, arm_root, ids)
                    if batch_size == 8
                    else compare_detection_roots(reference, arm_root, ids)
                )
                arm_pass = correctness["pass"] and rgb_checksum_pass
                row = {
                    **summary,
                    "status": "PASS" if arm_pass else "REJECTED_CORRECTNESS",
                    "root": str(arm_root),
                    "correctness": correctness,
                }
            except torch.cuda.OutOfMemoryError as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "batch_size": batch_size,
                    "decode_mode": "one_pass",
                    "status": "REJECTED_OOM",
                    "error": str(exc),
                    "root": str(arm_root),
                    "wall_seconds": elapsed,
                }
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - reject arm and continue
                elapsed = time.perf_counter() - started
                row = {
                    "batch_size": batch_size,
                    "decode_mode": "one_pass",
                    "status": f"REJECTED_{type(exc).__name__}",
                    "error": str(exc),
                    "root": str(arm_root),
                    "wall_seconds": elapsed,
                }
                torch.cuda.empty_cache()
            finally:
                phase.set("finalize")
                telemetry.stop()
            row["telemetry"] = _telemetry_summary(arm_root / "system_metrics.csv", "rtdetr_detection")
            _write_profile(arm_root, events, elapsed)
            atomic_json(arm_root / "arm_summary.json", row)
            arms.append(row)
            if batch_size == 32 and row.get("status") == "PASS":
                b16 = next((item for item in arms if item.get("batch_size") == 16), None)
                if b16 and b16.get("status") == "PASS" and float(row["end_to_end_fps"]) > float(b16["end_to_end_fps"]):
                    pending.extend([48, 64])
    finally:
        del localizer
        torch.cuda.empty_cache()
    onepass8 = next(row for row in arms if row.get("batch_size") == 8)
    decode = {
        "baseline": baseline,
        "one_pass": onepass8,
        "correctness_pass": onepass8.get("status") == "PASS",
        "sampled_rgb_checksum_pass": rgb_checksum_pass,
        "sampled_rgb_checksum_scope": [entry.video_id for entry in checksum_entries],
        "historical_rgb": historical_rgb,
        "onepass_rgb": onepass_rgb,
        "decode_speedup": baseline["decode_wall_seconds"]
        / float(sum(
            json.loads((Path(onepass8["root"]) / "detection_profiles" / f"{video_id}.json").read_text(encoding="utf-8"))["profile"]["decode_wall_seconds"]
            for video_id in ids
        )),
    }
    passing = [row for row in arms if row.get("status") == "PASS"]
    selected = (
        max(passing, key=lambda row: float(row["end_to_end_fps"]))
        if passing
        else baseline
    )
    report = {
        "subset_ids": ids,
        "rtdetr_model_load_count": 1,
        "rtdetr_model_load_seconds": load_seconds,
        "decode_comparison": decode,
        "arms": arms,
        "selected_batch_size": selected["batch_size"],
        "selected_decode_mode": selected["decode_mode"],
        "one_pass_promoted": bool(passing),
        "selected_speedup_vs_batch8": float(selected["end_to_end_fps"])
        / float(
            onepass8["end_to_end_fps"]
            if passing
            else baseline["end_to_end_fps"]
        ),
    }
    atomic_json(output_root / "sweep_summary.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("qwen", "rtdetr"))
    result.add_argument("--benchmark-manifest", required=True, type=Path)
    result.add_argument("--benchmark-sha256", required=True)
    result.add_argument("--dataset-root", required=True, type=Path)
    result.add_argument("--environment", required=True, type=Path)
    result.add_argument("--index", required=True, type=Path)
    result.add_argument("--baseline-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--video-id", action="append", dest="video_ids", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    ensure_positive_omp_threads()
    benchmark = load_benchmark_manifest(
        args.benchmark_manifest,
        expected_sha256=args.benchmark_sha256,
        dataset_root=args.dataset_root,
        verify_raw_sha256=False,
    )
    entries = _selected_entries(args.index, args.video_ids)
    report = qwen_sweep(args, benchmark, entries) if args.command == "qwen" else rtdetr_sweep(args, entries)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
