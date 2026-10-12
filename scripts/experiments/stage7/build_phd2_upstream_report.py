#!/usr/bin/env python3
"""Build evidence-linked tables, figures, ETA, and the PHD2 benchmark report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


COLORS = {"baseline": "#0072B2", "optimized": "#D55E00", "neutral": "#4D4D4D"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def telemetry_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "monotonic_seconds", "gpu_util_percent", "vram_used_mib", "gpu_power_w",
                "cpu_util_percent", "disk_read_bytes", "disk_write_bytes",
            ):
                try:
                    parsed[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    parsed[key] = None
            rows.append(parsed)
    return rows


def phase_stats(rows: list[dict], phase: str) -> dict:
    selected = [row for row in rows if row.get("phase") == phase]
    result = {}
    for key in ("gpu_util_percent", "vram_used_mib", "gpu_power_w", "cpu_util_percent"):
        values = [row[key] for row in selected if row.get(key) is not None]
        result[f"{key}_mean"] = sum(values) / len(values) if values else None
        result[f"{key}_peak"] = max(values) if values else None
    return result


def retrieval_stats(root: Path, ids: list[str]) -> dict:
    wall = chunks = retry = 0.0
    for video_id in ids:
        payload = load_json(root / "retrieval" / f"{video_id}.json")
        wall += float(payload["timing"]["artifact_wall_sec"])
        chunks += len(payload.get("chunks", []))
        retry += len(payload.get("provenance", {}).get("recovery_log", []))
    return {"summed_video_wall_seconds": wall, "chunk_count": int(chunks), "retry_records": int(retry)}


def detection_profiles(root: Path, ids: list[str]) -> list[dict]:
    return [load_json(root / "detection_profiles" / f"{video_id}.json") for video_id in ids]


def detection_stats(root: Path, ids: list[str], summary: dict) -> dict:
    rows = detection_profiles(root, ids)
    profiles = [row["profile"] for row in rows]
    frames = sum(int(row["frames"]) for row in rows)
    counters = summary["detection"]["counters"]
    return {
        "frames": frames,
        "wall_seconds": float(summary["detection"]["wall_seconds"]),
        "end_to_end_fps": float(summary["detection"]["end_to_end_fps"]),
        "pure_forward_fps": float(summary["detection"]["pure_forward_fps"]),
        "batch_size": int(summary["detection"]["batch_size"]),
        "peak_vram_mib": summary["detection"].get("peak_vram_mib"),
        "decode_wall_seconds": sum(float(row["decode_wall_seconds"]) for row in profiles),
        "write_wall_seconds": sum(float(row["npz_serialization_compression_write_seconds"]) for row in profiles),
        "capture_opens": int(counters["video_capture_open_count"]),
        "grabbed_frames": int(counters["grabbed_frame_count"]),
        "useful_frames": int(counters["retrieved_frame_count"]),
        "decode_amplification": float(counters["grabbed_frame_count"]) / max(float(counters["retrieved_frame_count"]), 1.0),
    }


def stage_parts(summary: dict) -> dict[str, float]:
    qwen = float((summary.get("retrieval") or {}).get("wall_seconds") or 0.0)
    qwen_load = next((float(row["wall_seconds"] or 0.0) for row in summary["profile_summary"] if row["stage"] == "qwen_service_start"), 0.0)
    detection = float((summary.get("detection") or {}).get("wall_seconds") or 0.0)
    model_load = float((summary.get("detection") or {}).get("model_load_seconds") or 0.0)
    assigned = qwen + qwen_load + detection + model_load
    return {
        "Qwen service load": qwen_load,
        "Qwen retrieval": qwen,
        "RT-DETR model load": model_load,
        "Decode + RT-DETR + write": detection,
        "Orchestration/telemetry": max(0.0, float(summary["total_wall_seconds"]) - assigned),
    }


def chunk_count(duration: float) -> int:
    return 1 if duration <= 30.0 else int(math.ceil((duration - 30.0) / 25.0)) + 1


def stage_scope(baseline: dict, optimized: dict) -> dict:
    """Separate detection-stage, Qwen, and whole-upstream measured scopes.

    Both promoted runs served Qwen artifacts from the on-disk c=1 retrieval
    cache (``retrieval.wall_seconds`` near zero), so the run-level wall time in
    each root measures the detection stage only.  The Qwen term must therefore
    be taken from the summed per-video retrieval artifact wall times, which were
    measured in a dedicated serial retrieval run.
    """

    base_qwen = float(baseline["retrieval"]["wall_seconds"])
    opt_qwen = float(optimized["retrieval"]["wall_seconds"])
    base_det = float(baseline["detection"]["wall_seconds"])
    opt_det = float(optimized["detection"]["wall_seconds"])
    return {
        "baseline_run_wall": float(baseline["total_wall_seconds"]),
        "optimized_run_wall": float(optimized["total_wall_seconds"]),
        "baseline_run_retrieval_wall": base_qwen,
        "optimized_run_retrieval_wall": opt_qwen,
        "detection_speedup": (
            float(baseline["total_wall_seconds"]) / float(optimized["total_wall_seconds"])
            if optimized["total_wall_seconds"]
            else None
        ),
        "detection_stage_baseline_seconds": base_det,
        "detection_stage_optimized_seconds": opt_det,
        "qwen_run_scope": "cached in both benchmark runs; see summed per-video retrieval wall",
    }


def eta(
    frozen: dict,
    optimized: dict,
    baseline: dict,
    optimized_root: Path,
    qwen_sweep: dict,
    *,
    baseline_root: Path | None = None,
    qwen_audit: dict | None = None,
) -> dict:
    records = frozen["records"]
    total_duration = sum(float(row["actual_duration_s"]) for row in records)
    total_chunks = sum(chunk_count(float(row["actual_duration_s"])) for row in records)
    total_frames = sum(int(math.floor(float(row["actual_duration_s"]) * 2.0 + 1e-9)) + 1 for row in records)
    # The run-level ``retrieval.wall_seconds`` is ~0 whenever the Qwen artifacts
    # are served from the on-disk cache, so the Qwen rate must come from the
    # summed per-video retrieval artifact wall instead.
    opt_qwen_stats = retrieval_stats(optimized_root, optimized["selected_video_ids"])
    base_qwen_stats = retrieval_stats(
        baseline_root if baseline_root is not None else optimized_root,
        baseline["selected_video_ids"],
    )
    qwen_fps = float(opt_qwen_stats["chunk_count"]) / float(opt_qwen_stats["summed_video_wall_seconds"])
    baseline_qwen_fps = float(base_qwen_stats["chunk_count"]) / float(base_qwen_stats["summed_video_wall_seconds"])
    qwen_rate_source = "summed per-video retrieval artifact wall (measured)"
    if qwen_audit and qwen_audit.get("promoted_qwen_chunks_per_second"):
        qwen_fps = float(qwen_audit["promoted_qwen_chunks_per_second"])
        qwen_rate_source = str(
            qwen_audit.get("promoted_qwen_rate_source", "Qwen audit promoted arm")
        )
    profiles = detection_profiles(optimized_root, optimized["selected_video_ids"])
    frame_total = sum(int(row["frames"]) for row in profiles)
    decode_keys = ("video_open_seconds", "grab_seconds", "retrieve_seconds")
    compute_keys = ("image_processor_seconds", "h2d_seconds", "rtdetr_pure_forward_cuda_seconds")
    post_keys = (
        "grid_creation_seconds", "grid_serialization_seconds", "gap_seconds",
        "postprocess_seconds", "d2h_seconds", "npz_serialization_compression_write_seconds",
    )
    decode_s = sum(sum(float(row["profile"].get(key, 0.0)) for key in decode_keys) for row in profiles)
    compute_s = sum(sum(float(row["profile"].get(key, 0.0)) for key in compute_keys) for row in profiles)
    post_s = sum(sum(float(row["profile"].get(key, 0.0)) for key in post_keys) for row in profiles)
    accounted = decode_s + compute_s + post_s
    overhead = max(0.0, float(optimized["detection"]["wall_seconds"]) - accounted)
    rtdetr_measured = (decode_s + compute_s) / frame_total * total_frames + float(optimized["detection"]["model_load_seconds"])
    post_measured = (post_s + overhead) / frame_total * total_frames
    qwen_measured = total_chunks / qwen_fps
    per_video_rt = []
    per_video_post = []
    for row in profiles:
        frames = max(int(row["frames"]), 1)
        profile = row["profile"]
        per_video_rt.append(sum(float(profile.get(key, 0.0)) for key in decode_keys + compute_keys) / frames)
        per_video_post.append(sum(float(profile.get(key, 0.0)) for key in post_keys) / frames)
    qwen_conservative = total_chunks / baseline_qwen_fps
    rtdetr_conservative = max(per_video_rt) * total_frames + float(optimized["detection"]["model_load_seconds"])
    post_conservative = max(per_video_post) * total_frames + overhead / frame_total * total_frames
    return {
        "source_duration_seconds": total_duration,
        "source_duration_minutes": total_duration / 60.0,
        "source_duration_hours": total_duration / 3600.0,
        "estimated_qwen_chunks": total_chunks,
        "estimated_2fps_frames": total_frames,
        "measured": {
            "qwen_seconds": qwen_measured,
            "rtdetr_seconds": rtdetr_measured,
            "postprocess_seconds": post_measured,
            "total_seconds": qwen_measured + rtdetr_measured + post_measured,
        },
        "conservative": {
            "qwen_seconds": qwen_conservative,
            "rtdetr_seconds": rtdetr_conservative,
            "postprocess_seconds": post_conservative,
            "total_seconds": qwen_conservative + rtdetr_conservative + post_conservative,
        },
        "method": {
            "measured": "exact frozen durations -> exact configured Qwen chunk-count formula and 2fps duration workload; scale by Optimized32 observed component throughput",
            "conservative": "Qwen uses observed serial Baseline32 chunks/s; RT-DETR/postprocess use the slowest observed Optimized32 per-video seconds/frame, with fixed model-load and measured overhead",
            "sampling_structure": "duration workload is summed over all 1500 records, avoiding equal-bin Benchmark32 count bias",
            "qwen_rate_source": qwen_rate_source,
            "qwen_rate_chunks_per_second": qwen_fps,
            "qwen_baseline_rate_chunks_per_second": baseline_qwen_fps,
            "benchmark_qwen_chunks": int(opt_qwen_stats["chunk_count"]),
            "benchmark_qwen_summed_wall_seconds": opt_qwen_stats["summed_video_wall_seconds"],
        },
    }


def whole_upstream(
    scope: dict,
    baseline_qwen_seconds: float,
    optimized_qwen_seconds: float,
    *,
    qwen_audit: dict | None = None,
) -> dict:
    """Component-wise Qwen -> RT-DETR upstream speedup, scope-explicit.

    The detection-stage speedup is the ratio of the two benchmark run walls.
    The whole-upstream ratio additionally adds the measured Qwen retrieval wall,
    which was NOT executed inside those two runs; it is therefore a sum of
    separately measured components and is labelled accordingly.
    """

    qwen_basis = "benchmark_root_retrieval_artifacts"
    audit_block = None
    if qwen_audit:
        audit_block = (qwen_audit.get("wall_time_breakdown") or {}).get(
            "benchmark32_promoted"
        )
        if audit_block and audit_block.get("summed_video_wall_seconds"):
            optimized_qwen_seconds = float(audit_block["summed_video_wall_seconds"])
            if audit_block.get("serial_baseline_summed_wall_seconds"):
                baseline_qwen_seconds = float(
                    audit_block["serial_baseline_summed_wall_seconds"]
                )
            qwen_basis = "qwen_execution_audit_promoted_arm_full_benchmark32"
    baseline_total = baseline_qwen_seconds + scope["detection_stage_baseline_seconds"]
    optimized_total = optimized_qwen_seconds + scope["detection_stage_optimized_seconds"]
    return {
        "detection_stage": {
            "baseline_seconds": scope["detection_stage_baseline_seconds"],
            "optimized_seconds": scope["detection_stage_optimized_seconds"],
            "speedup": (
                scope["detection_stage_baseline_seconds"]
                / scope["detection_stage_optimized_seconds"]
                if scope["detection_stage_optimized_seconds"]
                else None
            ),
            "measurement": "MEASURED_SINGLE_RUN_DETECTION_ONLY",
        },
        "whole_upstream": {
            "baseline_seconds": baseline_total,
            "optimized_seconds": optimized_total,
            "speedup": baseline_total / optimized_total if optimized_total else None,
            "baseline_qwen_seconds": baseline_qwen_seconds,
            "optimized_qwen_seconds": optimized_qwen_seconds,
            "qwen_basis": qwen_basis,
            "measurement": "NOT_DIRECTLY_MEASURED_COMPONENT_WISE_SUM",
            "note": (
                "Qwen retrieval and RT-DETR detection were measured in separate runs "
                "(the benchmark roots served Qwen from cache). The whole-upstream ratio "
                "sums the separately measured Qwen component with the detection run wall."
            ),
        },
        "scope": scope,
    }


def _qwen_promoted_label(audit: dict | None) -> str:
    if not audit:
        return "not measured in this report"
    return str(audit.get("promoted_qwen_execution", "not measured"))


def _qwen_gate_label(audit: dict | None) -> str:
    if not audit:
        return "NOT_MEASURED"
    gates = audit.get("correctness_gates", {})
    return (
        f"self-determinism {gates.get('self_determinism')}, "
        f"challenger equivalence {gates.get('challenger_equivalence')}, "
        f"artifact equivalence vs promoted baseline "
        f"{gates.get('artifact_semantic_equivalence_vs_promoted_baseline')}"
    )


def qwen_audit_section(audit: dict | None, eta_result: dict) -> str:
    """Render the round-2 Qwen execution audit into the report body."""

    if not audit:
        return ""
    arms = audit.get("arms", {})
    promoted = audit.get("promoted_qwen_execution")
    determinism = audit.get("self_determinism", {})
    challenger = audit.get("challenger") or {}
    breakdown = audit.get("wall_time_breakdown", {})
    serial = breakdown.get("c1_serial") or {}
    promoted_bd = breakdown.get("promoted") or {}
    lines = [
        "",
        "## Qwen execution audit (round 2)",
        "",
        f"- Determinism subset: {audit.get('subset', {}).get('count')} videos, "
        f"SHA256 `{audit.get('subset', {}).get('sha256')}`",
        f"- c=1 self-determinism A/B/C: **{determinism.get('verdict')}** "
        f"(chunk exact-match min {determinism.get('chunk_exact_match_rate_min')}, "
        f"video candidate match min {determinism.get('video_candidate_semantic_match_rate_min')})",
        f"- Promoted Qwen execution: {promoted}",
        f"- Promoted Qwen rate: {audit.get('promoted_qwen_chunks_per_second')} chunks/s "
        f"({audit.get('promoted_qwen_rate_source')})",
        "",
        "| Arm | Execution | Chunks | Summed video wall (s) | Chunk render (s) | Client request (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(arms):
        row = arms[key]
        lines.append(
            f"| {key} | {'render look-ahead' if row.get('overlap_render') else 'serial c=1'} | "
            f"{row.get('chunk_count')} | {row.get('sum_video_wall_sec'):.3f} | "
            f"{row.get('sum_render_sec'):.3f} | {row.get('sum_request_sec'):.3f} |"
        )
    for label, block in (("serial c=1", serial), ("promoted", promoted_bd)):
        per_chunk = block.get("per_chunk") or {}
        server = block.get("server") or {}
        derived = block.get("derived") or {}
        if not per_chunk:
            continue
        lines.extend(
            [
                "",
                f"### Qwen wall-time breakdown ({label})",
                "",
                "| Bucket | Per chunk (s) | Measurement |",
                "|---|---:|---|",
                f"| Chunk extraction / ffmpeg render | {per_chunk.get('chunk_render_sec'):.3f} | client timer around the frozen renderer |",
                f"| Client HTTP request (send->receive) | {per_chunk.get('client_request_sec'):.3f} | client timer around `analyze_video` |",
                f"| Server queue wait | {(server.get('queue_time_sec_mean') or 0.0):.3f} | `vllm:request_queue_time_seconds` |",
                f"| Server prefill | {(server.get('prefill_time_sec_mean') or 0.0):.3f} | `vllm:request_prefill_time_seconds` |",
                f"| Server decode / generation | {(server.get('decode_time_sec_mean') or 0.0):.3f} | `vllm:request_decode_time_seconds` |",
                f"| Server input processing (MM decode, sample, resize, tokenize) | "
                f"{((derived.get('server_input_processing_sec_total') or 0.0) / max(block.get('chunk_count', 1), 1)):.3f} | "
                "server e2e minus engine inference |",
                f"| JSON parse | {per_chunk.get('parse_sec'):.6f} | client timer |",
                f"| Candidate merge | {per_chunk.get('merge_sec'):.6f} | client timer |",
                f"| Prompt tokens / chunk | {per_chunk.get('prompt_tokens'):.1f} | server usage |",
                f"| Completion tokens / chunk | {per_chunk.get('completion_tokens'):.1f} | server usage |",
            ]
        )
    full = breakdown.get("full1500_promoted") or {}
    if full:
        lines.extend(
            [
                "",
                "### Full1500 Qwen extrapolation",
                "",
                f"- Chunks: {full.get('full1500_chunks')}",
                f"- Estimated Qwen wall: {full.get('qwen_hours'):.3f} h",
                f"- Of which chunk render: "
                f"{(full.get('chunk_render_seconds') or 0.0)/3600:.3f} h, "
                f"server prefill: {(full.get('server_prefill_seconds') or 0.0)/3600:.3f} h, "
                f"server decode: {(full.get('server_decode_seconds') or 0.0)/3600:.3f} h, "
                f"server input processing: {(full.get('server_input_processing_seconds') or 0.0)/3600:.3f} h",
                f"- Report ETA basis (measured): {eta_result['method']['qwen_rate_source']}",
            ]
        )
    if challenger:
        lines.extend(
            [
                "",
                "### Render look-ahead challenger",
                "",
                f"- Chunk raw exact-match rate vs serial c=1: {challenger.get('chunk_raw_exact_match_rate')}",
                f"- All chunks identical: {challenger.get('all_chunks_exact')}; "
                f"all videos identical: {challenger.get('all_videos_equal')}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def save_figure(fig, figures: Path, name: str, metadata: list[dict], *, alt: str, sources: list[str], transformations: list[str]) -> None:
    png = figures / f"{name}.png"
    fig.savefig(png, dpi=180, facecolor="white")
    metadata.append({"name": name, "png": str(png), "alt": alt, "sources": sources, "transformations": transformations})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--optimized-root", required=True, type=Path)
    parser.add_argument("--qwen-sweep", required=True, type=Path)
    parser.add_argument("--rtdetr-sweep", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--qwen-audit",
        type=Path,
        default=None,
        help="optional Qwen execution audit summary (round 2) to embed in the report",
    )
    args = parser.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
    root = args.output_root
    tables = root / "tables"
    figures = root / "figures"
    profiles = root / "profiles"
    correctness_dir = root / "correctness"
    machine = root / "machine_outputs"
    logs = root / "logs"
    for path in (tables, figures, profiles, correctness_dir, machine, logs, root / "benchmark_manifest"):
        path.mkdir(parents=True, exist_ok=True)

    baseline = load_json(args.baseline_root / "summary.json")
    optimized = load_json(args.optimized_root / "summary.json")
    qwen = load_json(args.qwen_sweep)
    rtdetr = load_json(args.rtdetr_sweep)
    correctness = load_json(args.correctness)
    benchmark = load_json(args.benchmark_manifest)
    frozen = load_json(args.frozen_manifest)
    qwen_audit_payload = (
        load_json(Path(args.qwen_audit))
        if args.qwen_audit is not None and Path(args.qwen_audit).is_file()
        else None
    )
    ids = baseline["selected_video_ids"]
    base_tel = telemetry_rows(args.baseline_root / "system_metrics.csv")
    opt_tel = telemetry_rows(args.optimized_root / "system_metrics.csv")
    base_det = detection_stats(args.baseline_root, ids, baseline)
    opt_det = detection_stats(args.optimized_root, ids, optimized)
    base_qwen = retrieval_stats(args.baseline_root, ids)
    opt_qwen = retrieval_stats(args.optimized_root, ids)
    base_qt = phase_stats(base_tel, "qwen_retrieval")
    opt_qt = phase_stats(opt_tel, "qwen_retrieval")
    base_dt = phase_stats(base_tel, "rtdetr_detection")
    opt_dt = phase_stats(opt_tel, "rtdetr_detection")
    eta_result = eta(
        frozen,
        optimized,
        baseline,
        args.optimized_root,
        qwen,
        baseline_root=args.baseline_root,
        qwen_audit=qwen_audit_payload,
    )
    write_json(machine / "eta.json", eta_result)
    scope = stage_scope(baseline, optimized)
    whole = whole_upstream(
        scope,
        base_qwen["summed_video_wall_seconds"],
        opt_qwen["summed_video_wall_seconds"],
        qwen_audit=qwen_audit_payload,
    )
    if args.qwen_audit is not None and Path(args.qwen_audit).is_file():
        whole["qwen_audit"] = load_json(Path(args.qwen_audit))
        whole["qwen_audit_path"] = str(args.qwen_audit)
    write_json(machine / "whole_upstream.json", whole)

    metrics = [
        ("Detection-stage wall (s) [Qwen cached]", baseline["total_wall_seconds"], optimized["total_wall_seconds"]),
        ("RTF [detection stage]", baseline["rtf"], optimized["rtf"]),
        ("Qwen run wall (s) [cached]", baseline["retrieval"]["wall_seconds"], optimized["retrieval"]["wall_seconds"]),
        ("Qwen summed video wall (s) [real c=1]", base_qwen["summed_video_wall_seconds"], opt_qwen["summed_video_wall_seconds"]),
        ("Qwen chunks/s [summed video wall]", baseline["retrieval"]["chunk_count"] / base_qwen["summed_video_wall_seconds"], optimized["retrieval"]["chunk_count"] / opt_qwen["summed_video_wall_seconds"]),
        ("Qwen source-min/wall-min [summed video wall]", baseline["source_duration_seconds"] / base_qwen["summed_video_wall_seconds"], optimized["source_duration_seconds"] / opt_qwen["summed_video_wall_seconds"]),
        ("Qwen GPU util mean (%)", base_qt.get("gpu_util_percent_mean"), opt_qt.get("gpu_util_percent_mean")),
        ("Qwen VRAM peak (MiB)", base_qt.get("vram_used_mib_peak"), opt_qt.get("vram_used_mib_peak")),
        ("Qwen power mean (W)", base_qt.get("gpu_power_w_mean"), opt_qt.get("gpu_power_w_mean")),
        ("Decode wall (s)", base_det["decode_wall_seconds"], opt_det["decode_wall_seconds"]),
        ("VideoCapture opens", base_det["capture_opens"], opt_det["capture_opens"]),
        ("Grabbed frames", base_det["grabbed_frames"], opt_det["grabbed_frames"]),
        ("Useful frames", base_det["useful_frames"], opt_det["useful_frames"]),
        ("Decode amplification", base_det["decode_amplification"], opt_det["decode_amplification"]),
        ("RT-DETR wall (s)", base_det["wall_seconds"], opt_det["wall_seconds"]),
        ("RT-DETR end-to-end FPS", base_det["end_to_end_fps"], opt_det["end_to_end_fps"]),
        ("RT-DETR pure-forward FPS", base_det["pure_forward_fps"], opt_det["pure_forward_fps"]),
        ("RT-DETR GPU util mean (%)", base_dt.get("gpu_util_percent_mean"), opt_dt.get("gpu_util_percent_mean")),
        ("RT-DETR VRAM peak telemetry (MiB)", base_dt.get("vram_used_mib_peak"), opt_dt.get("vram_used_mib_peak")),
        ("RT-DETR power mean (W)", base_dt.get("gpu_power_w_mean"), opt_dt.get("gpu_power_w_mean")),
        ("RT-DETR batch", base_det["batch_size"], opt_det["batch_size"]),
        ("Detection NPZ write (s)", base_det["write_wall_seconds"], opt_det["write_wall_seconds"]),
        ("CPU utilization mean (%)", phase_stats(base_tel, "rtdetr_detection").get("cpu_util_percent_mean"), phase_stats(opt_tel, "rtdetr_detection").get("cpu_util_percent_mean")),
    ]
    higher_better = {
        "Qwen chunks/s [summed video wall]",
        "Qwen source-min/wall-min [summed video wall]",
        "RT-DETR end-to-end FPS",
        "RT-DETR pure-forward FPS",
    }
    table_rows = []
    for name, left, right in metrics:
        if left in (None, 0) or right is None:
            speedup = None
        elif name in higher_better:
            speedup = float(right) / float(left)
        else:
            speedup = float(left) / float(right) if float(right) != 0 else None
        table_rows.append({"metric": name, "baseline": left, "optimized": right, "speedup": speedup})
    write_csv(tables / "baseline_vs_optimized.csv", table_rows)
    write_csv(tables / "qwen_concurrency_sweep.csv", [{k: v for k, v in row.items() if k not in ("correctness", "telemetry")} | {"vram_peak_mib": row.get("telemetry", {}).get("vram_used_mib_peak"), "gpu_util_mean_percent": row.get("telemetry", {}).get("gpu_util_percent_mean")} for row in qwen["arms"]])
    write_csv(tables / "rtdetr_batch_sweep.csv", [{k: v for k, v in row.items() if k not in ("correctness", "telemetry", "counters")} | {"vram_peak_mib": row.get("telemetry", {}).get("vram_used_mib_peak"), "gpu_util_mean_percent": row.get("telemetry", {}).get("gpu_util_percent_mean")} for row in rtdetr["arms"]])
    bins = ["0-1min", "1-3min", "3-10min", "10-30min"]
    composition = []
    for name in bins:
        selected = [row for row in benchmark["records"] if row["duration_bin"] == name]
        composition.append({"duration_bin": name, "video_count": len(selected), "source_seconds": sum(float(row["duration_seconds"]) for row in selected)})
    write_csv(tables / "benchmark32_duration_composition.csv", composition)
    write_csv(tables / "full1500_eta.csv", [{"estimate": key, **value} for key, value in (("MEASURED_ESTIMATE", eta_result["measured"]), ("CONSERVATIVE_ESTIMATE", eta_result["conservative"]))])

    figure_meta = []
    for label, summary, name, color in (("Baseline", baseline, "01_baseline_stage_wall", COLORS["baseline"]), ("Optimized", optimized, "02_optimized_stage_wall", COLORS["optimized"])):
        parts = stage_parts(summary)
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.bar(parts.keys(), parts.values(), color=color, hatch="//" if label == "Baseline" else "")
        ax.set(ylabel="Wall time (s)", title=f"{label} stage wall-time breakdown")
        ax.tick_params(axis="x", rotation=25)
        save_figure(fig, figures, name, figure_meta, alt=f"Bar chart of {label.lower()} wall time by measured top-level stage.", sources=[str((args.baseline_root if label == "Baseline" else args.optimized_root) / "summary.json")], transformations=["top-level non-overlapping stage aggregation", "bars start at zero"])
        plt.close(fig)

    labels = list(stage_parts(baseline))
    fig, ax = plt.subplots(figsize=(7.4, 4.3), layout="constrained")
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [stage_parts(baseline)[k] for k in labels], 0.4, label="Baseline", color=COLORS["baseline"], hatch="//")
    ax.bar([i + 0.2 for i in x], [stage_parts(optimized)[k] for k in labels], 0.4, label="Optimized", color=COLORS["optimized"])
    ax.set_xticks(list(x), labels, rotation=25)
    ax.set(ylabel="Wall time (s)", title="Before/after stage comparison")
    ax.legend()
    save_figure(fig, figures, "03_before_after_stage", figure_meta, alt="Grouped bars compare baseline and optimized wall time for each top-level stage.", sources=[str(args.baseline_root / "summary.json"), str(args.optimized_root / "summary.json")], transformations=["same categories and common zero baseline"])
    plt.close(fig)

    def timeline(rows, key, title, ylabel, name, both=None):
        fig, ax = plt.subplots(figsize=(8.0, 3.8), layout="constrained")
        ax.plot([row["monotonic_seconds"] / 60 for row in rows], [row[key] for row in rows], color=COLORS["baseline"], label="Baseline")
        if both is not None:
            ax.plot([row["monotonic_seconds"] / 60 for row in both], [row[key] for row in both], color=COLORS["optimized"], linestyle="--", label="Optimized")
            ax.legend()
        ax.set(xlabel="Run elapsed time (min)", ylabel=ylabel, title=title)
        save_figure(fig, figures, name, figure_meta, alt=title + "; raw one-second samples without smoothing.", sources=[str(args.baseline_root / "system_metrics.csv")] + ([str(args.optimized_root / "system_metrics.csv")] if both is not None else []), transformations=["raw approximately 1-second telemetry; no smoothing"])
        plt.close(fig)

    timeline(base_tel, "gpu_util_percent", "Baseline GPU utilization timeline", "GPU utilization (%)", "04_gpu_util_baseline")
    timeline(opt_tel, "gpu_util_percent", "Optimized GPU utilization timeline", "GPU utilization (%)", "05_gpu_util_optimized")
    timeline(base_tel, "vram_used_mib", "VRAM timeline", "VRAM used (MiB)", "06_vram_timeline", opt_tel)
    timeline(base_tel, "gpu_power_w", "GPU power timeline", "GPU power (W)", "07_gpu_power_timeline", opt_tel)

    qarms = [row for row in qwen["arms"] if row.get("chunks_per_second") is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    ax.plot([row["concurrency"] for row in qarms], [row["chunks_per_second"] for row in qarms], marker="o", color=COLORS["optimized"])
    ax.set(xlabel="Request concurrency", ylabel="Qwen throughput (chunks/s)", title="Qwen concurrency sweep", xticks=[row["concurrency"] for row in qarms])
    save_figure(fig, figures, "08_qwen_concurrency_throughput", figure_meta, alt="Line and point chart of measured Qwen chunks per second by request concurrency.", sources=[str(args.qwen_sweep)], transformations=["PASS and measured baseline arms shown; no smoothing"])
    plt.close(fig)

    rarms = [row for row in rtdetr["arms"] if row.get("end_to_end_fps") is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    ax.plot([row["batch_size"] for row in rarms], [row["end_to_end_fps"] for row in rarms], marker="o", color=COLORS["optimized"])
    ax.set(xlabel="RT-DETR batch size", ylabel="End-to-end throughput (frames/s)", title="RT-DETR batch sweep")
    save_figure(fig, figures, "09_rtdetr_batch_fps", figure_meta, alt="Line and point chart of RT-DETR end-to-end frames per second by batch size.", sources=[str(args.rtdetr_sweep)], transformations=["measured one-pass arms; no smoothing"])
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    vram_arms = [row for row in rarms if row.get("telemetry", {}).get("vram_used_mib_peak") is not None]
    ax.plot([row["batch_size"] for row in vram_arms], [row["telemetry"]["vram_used_mib_peak"] for row in vram_arms], marker="s", color=COLORS["baseline"])
    ax.set(xlabel="RT-DETR batch size", ylabel="Peak VRAM used (MiB)", title="RT-DETR batch vs VRAM")
    save_figure(fig, figures, "10_rtdetr_batch_vram", figure_meta, alt="Line and square chart of telemetry peak VRAM by RT-DETR batch size.", sources=[str(args.rtdetr_sweep)], transformations=["maximum raw telemetry sample per arm"])
    plt.close(fig)

    decode_comp = rtdetr["decode_comparison"]
    decode_values = [decode_comp["baseline"]["decode_wall_seconds"], sum(float(load_json(Path(decode_comp["one_pass"]["root"]) / "detection_profiles" / f"{video_id}.json")["profile"]["decode_wall_seconds"]) for video_id in rtdetr["subset_ids"])]
    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    ax.bar(["Windowed prefix", "One-pass"], decode_values, color=[COLORS["baseline"], COLORS["optimized"]], hatch=["//", ""])
    ax.set(ylabel="Decode wall time (s)", title="Decode before/after (fixed subset)")
    save_figure(fig, figures, "11_decode_before_after", figure_meta, alt="Two bars compare repeated-prefix and one-pass decode wall time on the fixed RT-DETR subset.", sources=[str(args.rtdetr_sweep)], transformations=["sum of per-video measured decode wall time; zero baseline"])
    plt.close(fig)

    overall = float(baseline["total_wall_seconds"]) / float(optimized["total_wall_seconds"])
    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    labels = ["Detection baseline", "Detection optimized", "Whole upstream baseline", "Whole upstream optimized"]
    values = [
        1.0,
        overall,
        whole["whole_upstream"]["baseline_seconds"] / whole["whole_upstream"]["baseline_seconds"],
        whole["whole_upstream"]["optimized_seconds"] / whole["whole_upstream"]["baseline_seconds"],
    ]
    ax.bar(
        labels,
        values,
        color=[COLORS["baseline"], COLORS["optimized"], COLORS["baseline"], COLORS["optimized"]],
        hatch=["//", "", "//", ""],
    )
    ax.set(
        ylabel="Wall time relative to whole-upstream baseline (-)",
        title="Speedup scope: detection stage vs whole upstream",
    )
    ax.tick_params(axis="x", rotation=18)
    save_figure(
        fig,
        figures,
        "12_speedup_scope",
        figure_meta,
        alt=(
            f"Bar chart: detection baseline 1.00, detection optimized {1/overall:.3f} of it, "
            f"whole-upstream baseline 1.00, whole-upstream optimized "
            f"{whole['whole_upstream']['optimized_seconds']/whole['whole_upstream']['baseline_seconds']:.3f} of it."
        ),
        sources=[str(args.baseline_root / "summary.json"), str(args.optimized_root / "summary.json")],
        transformations=[
            "detection bars: baseline run wall divided by optimized run wall (Qwen cached)",
            "whole-upstream bars: (summed per-video Qwen wall + detection stage wall) normalized to the baseline sum; component-wise, not one run",
            "zero baseline",
        ],
    )
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    ax.bar([row["duration_bin"] for row in composition], [row["video_count"] for row in composition], color=COLORS["baseline"], hatch="//")
    ax.set(xlabel="Duration bin", ylabel="Videos (count)", title="Benchmark32 duration composition")
    save_figure(fig, figures, "13_benchmark32_duration_composition", figure_meta, alt="Four equal-height bars show eight Benchmark32 videos in each duration bin.", sources=[str(args.benchmark_manifest)], transformations=["predeclared duration-bin counts; zero baseline"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    eta_stages = ["qwen_seconds", "rtdetr_seconds", "postprocess_seconds"]
    ax.bar(["Qwen", "RT-DETR", "Postprocess"], [eta_result["measured"][key] / 3600 for key in eta_stages], color=["#009E73", COLORS["baseline"], "#CC79A7"])
    ax.set(ylabel="Estimated wall time (hours)", title="Measured Full1500 stage-time estimate")
    save_figure(fig, figures, "14_full1500_stage_eta", figure_meta, alt="Bar chart of measured-estimate Full1500 hours for Qwen, RT-DETR, and postprocess.", sources=[str(args.frozen_manifest), str(args.optimized_root / "summary.json")], transformations=[eta_result["method"]["measured"], "bars start at zero"])
    plt.close(fig)

    timeline(base_tel, "cpu_util_percent", "CPU utilization timeline", "CPU utilization (%)", "15_cpu_util_timeline", opt_tel)
    fig, ax = plt.subplots(figsize=(8.0, 3.8), layout="constrained")
    for rows, label, color, style in ((base_tel, "Baseline read", COLORS["baseline"], "-"), (base_tel, "Baseline write", COLORS["baseline"], ":"), (opt_tel, "Optimized read", COLORS["optimized"], "--"), (opt_tel, "Optimized write", COLORS["optimized"], "-.")):
        read = "read" in label.lower()
        points_x, points_y = [], []
        for previous, current in zip(rows, rows[1:]):
            key = "disk_read_bytes" if read else "disk_write_bytes"
            if previous[key] is None or current[key] is None:
                continue
            dt = current["monotonic_seconds"] - previous["monotonic_seconds"]
            if dt > 0:
                points_x.append(current["monotonic_seconds"] / 60)
                points_y.append(max(0.0, current[key] - previous[key]) / dt / (1024 * 1024))
        ax.plot(points_x, points_y, label=label, color=color, linestyle=style)
    ax.set(xlabel="Run elapsed time (min)", ylabel="Disk throughput (MiB/s)", title="Disk I/O timeline")
    ax.legend(ncol=2)
    save_figure(fig, figures, "16_disk_io_timeline", figure_meta, alt="Four lines show baseline and optimized disk read and write throughput from raw counter differences.", sources=[str(args.baseline_root / "system_metrics.csv"), str(args.optimized_root / "system_metrics.csv")], transformations=["first differences of cumulative disk bytes divided by sample interval; negative resets clipped to zero; no smoothing"])
    plt.close(fig)
    write_json(figures / "figure_manifest.json", figure_meta)

    for src, dst in (
        (args.baseline_root / "summary.json", machine / "baseline_summary.json"),
        (args.optimized_root / "summary.json", machine / "optimized_summary.json"),
        (args.qwen_sweep, machine / "qwen_sweep_summary.json"),
        (args.rtdetr_sweep, machine / "rtdetr_sweep_summary.json"),
        (args.correctness, correctness_dir / "baseline_vs_optimized.json"),
        (args.benchmark_manifest, root / "benchmark_manifest" / args.benchmark_manifest.name),
        (args.frozen_manifest, root / "benchmark_manifest" / args.frozen_manifest.name),
    ):
        shutil.copy2(src, dst)
    for label, run_root in (("baseline", args.baseline_root), ("optimized", args.optimized_root)):
        destination = profiles / label
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("system_metrics.csv", "stage_events.csv", "stage_summary.csv", "progress.jsonl"):
            source = run_root / name
            if source.is_file():
                shutil.copy2(source, destination / name)

    report = f"""# PHD² Upstream Acceleration Pre-Experiment

Status: `PERFORMANCE_BENCHMARK_ONLY`  
Full1500 execution: **NOT STARTED**

## Data identity

- Frozen PHD² records: {len(frozen['records'])}
- Frozen PHD² SHA256: `{file_sha256(args.frozen_manifest)}`
- Exact source duration: {eta_result['source_duration_seconds']:.6f} s = {eta_result['source_duration_minutes']:.6f} min = {eta_result['source_duration_hours']:.6f} h
- Benchmark32 source duration: {baseline['source_duration_seconds']:.6f} s
- Benchmark32 SHA256: `{baseline['benchmark_sha256']}`

## Speedup scope (corrected)

The two benchmark roots below executed the **detection stage only**: both runs
served the Qwen retrieval artifacts from the on-disk `concurrency = 1` retrieval
cache (`retrieval.wall_seconds = {scope['baseline_run_retrieval_wall']:.6f} s` baseline,
`{scope['optimized_run_retrieval_wall']:.6f} s` optimized). Therefore the run-level
ratio is a **detection / RT-DETR upstream sub-chain speedup**, not a whole-upstream
end-to-end speedup.

| Scope | Baseline | Optimized | Speedup | Measurement status |
|---|---:|---:|---:|---|
| A. Detection / RT-DETR sub-chain | {scope['detection_stage_baseline_seconds']:.3f} s | {scope['detection_stage_optimized_seconds']:.3f} s | {whole['detection_stage']['speedup']:.3f}× | `MEASURED_SINGLE_RUN_DETECTION_ONLY` |
| B. Whole Qwen -> RT-DETR upstream | {whole['whole_upstream']['baseline_seconds']:.3f} s | {whole['whole_upstream']['optimized_seconds']:.3f} s | {whole['whole_upstream']['speedup']:.3f}× | `NOT_DIRECTLY_MEASURED_COMPONENT_WISE_SUM` |

Scope A is the ratio of the two run walls. Scope B adds the summed per-video
Qwen retrieval wall ({whole['whole_upstream']['baseline_qwen_seconds']:.3f} s baseline /
{whole['whole_upstream']['optimized_qwen_seconds']:.3f} s optimized, basis
`{whole['whole_upstream']['qwen_basis']}`); the Qwen and
detection walls were never executed as one continuous measurement, so scope B is
explicitly **not directly measured**.

## Promoted execution configuration

- Qwen request concurrency: **1** (semantic-equivalence gate applied)
- Qwen chunk render: **one-chunk look-ahead** ({_qwen_promoted_label(qwen_audit_payload)})
  - requests remain strictly serial (one in flight), chunk order unchanged
  - correctness: {_qwen_gate_label(qwen_audit_payload)}
- RT-DETR batch size: **{rtdetr['selected_batch_size']}** (detection/Visual256 gate applied)
- Decode: **{rtdetr.get('selected_decode_mode', 'one_pass')}**, bounded by inference batch
- Numerical acceleration: not used

## End-to-end result (detection stage)

| Metric | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Detection-stage wall (s) | {baseline['total_wall_seconds']:.3f} | {optimized['total_wall_seconds']:.3f} | {overall:.3f}× |
| RTF (detection stage) | {baseline['rtf']:.6f} | {optimized['rtf']:.6f} | {overall:.3f}× |
| Qwen chunks/s (summed video wall) | {baseline['retrieval']['chunk_count']/base_qwen['summed_video_wall_seconds']:.4f} | {optimized['retrieval']['chunk_count']/opt_qwen['summed_video_wall_seconds']:.4f} | {(baseline['retrieval']['chunk_count']/base_qwen['summed_video_wall_seconds'])/(optimized['retrieval']['chunk_count']/opt_qwen['summed_video_wall_seconds']):.3f}× |
| RT-DETR end-to-end FPS | {base_det['end_to_end_fps']:.3f} | {opt_det['end_to_end_fps']:.3f} | {opt_det['end_to_end_fps']/base_det['end_to_end_fps']:.3f}× |
| Decode amplification | {base_det['decode_amplification']:.3f}× | {opt_det['decode_amplification']:.3f}× | {base_det['decode_amplification']/opt_det['decode_amplification']:.3f}× reduction |

Full metric table: [tables/baseline_vs_optimized.csv](tables/baseline_vs_optimized.csv).

## Promoted execution configuration (RT-DETR sweep root)

- Qwen request concurrency in that sweep: **{qwen['selected_concurrency']}**
  (the sweep root predates the round-2 Qwen execution audit)
- RT-DETR batch size: **{rtdetr['selected_batch_size']}** (detection/Visual256 gate applied)
- Decode: **{rtdetr.get('selected_decode_mode', 'one_pass')}**, bounded by inference batch
- Producer/consumer overlap: RT-DETR decode one-pass; Qwen render look-ahead per the audit above
- Numerical acceleration: not used

## Correctness gate

- Overall: **{'PASS' if correctness['pass'] else 'FAIL'}**
- Qwen semantic equivalence: {correctness['qwen_semantic_equivalence']}
- Frame/grid equivalence: {correctness['frame_grid_equivalence']}
- RT-DETR equivalence: {correctness['rtdetr_equivalence']}
- Downstream: {correctness['downstream_equivalence']}

## Full1500 ETA (Full1500 was not run)

| Estimate | Qwen | RT-DETR | Postprocess | Total |
|---|---:|---:|---:|---:|
| MEASURED_ESTIMATE | {eta_result['measured']['qwen_seconds']/3600:.3f} h | {eta_result['measured']['rtdetr_seconds']/3600:.3f} h | {eta_result['measured']['postprocess_seconds']/3600:.3f} h | {eta_result['measured']['total_seconds']/3600:.3f} h |
| CONSERVATIVE_ESTIMATE | {eta_result['conservative']['qwen_seconds']/3600:.3f} h | {eta_result['conservative']['rtdetr_seconds']/3600:.3f} h | {eta_result['conservative']['postprocess_seconds']/3600:.3f} h | {eta_result['conservative']['total_seconds']/3600:.3f} h |

Measured basis: {eta_result['method']['measured']}. Conservative basis: {eta_result['method']['conservative']}. The computation sums the real 1500-video duration distribution, rather than multiplying the equal-bin Benchmark32 average.

Qwen rate basis: `{eta_result['method']['qwen_rate_source']}` at
{eta_result['method']['qwen_rate_chunks_per_second']:.6f} chunks/s
({eta_result['method']['benchmark_qwen_chunks']} chunks over
{eta_result['method']['benchmark_qwen_summed_wall_seconds']:.3f} s of summed per-video
retrieval wall). Benchmark32 Qwen chunk count ({eta_result['method']['benchmark_qwen_chunks']})
and Full1500 chunk count ({eta_result['estimated_qwen_chunks']}) use the same configured
30 s / 5 s chunk formula.
{qwen_audit_section(qwen_audit_payload, eta_result)}

## Figures and provenance

The `figures/` directory contains PNG versions only (SVG/PDF/EPS are not produced). `figures/figure_manifest.json` records source files, transformations, and alt text. Telemetry timelines use raw approximately one-second samples without smoothing. Underlying values are preserved in `tables/` and `profiles/`.

## Scope limitations

This adapter intentionally stops before PHD² Pairwise/PU supervision. It does not fabricate annotations or targets. LOC/CMP/TS/Native16 and target adaptation are not applicable to this upstream-only performance benchmark.

## Visualization methodology reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    (root / "experiment_report.md").write_text(report, encoding="utf-8", newline="\n")
    write_json(
        machine / "report_metrics.json",
        {
            "overall_speedup": overall,
            "detection_stage_speedup": whole["detection_stage"]["speedup"],
            "whole_upstream_speedup": whole["whole_upstream"]["speedup"],
            "whole_upstream_measurement": whole["whole_upstream"]["measurement"],
            "speedup_scope": scope,
            "baseline_detection": base_det,
            "optimized_detection": opt_det,
            "baseline_retrieval": base_qwen,
            "optimized_retrieval": opt_qwen,
            "eta": eta_result,
            "qwen_audit": whole.get("qwen_audit"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
