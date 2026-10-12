"""PHD2 upstream-only engineering benchmark support.

This module deliberately stops before any PHD2 supervision adapter.  It reads
raw videos and produces Qwen retrieval, uniform full-video 2 fps grid,
RT-DETR detection, and encoder GAP artifacts for performance measurement only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aic_video_highlight.runtime.paths import EnvironmentPaths

from .index import (
    DECODE_OK,
    IndexEntry,
    extract_pts_with_fallback,
    load_index,
    probe_index_entry,
    resolve_video_path,
    write_index,
)
from .performance_decode import ONE_PASS, WINDOWED_PREFIX
from .real_provider import (
    QWEN_MODEL,
    RTDETR_MODEL,
    RetrievalSettings,
    create_rtdetr_localizer,
    run_detection_for_entry,
    run_retrieval_for_entry,
)


BENCHMARK_SCHEMA = "aic.phd2.upstream-benchmark32/v1"
RUN_SCHEMA = "aic.phd2.upstream-performance-run/v1"
CORRECTNESS_SCHEMA = "aic.phd2.upstream-correctness/v1"
PERFORMANCE_ONLY = "PERFORMANCE_BENCHMARK_ONLY"


def ensure_positive_omp_threads() -> int:
    """Repair an empty/invalid inherited OpenMP setting for vLLM startup."""

    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        value = max(1, min(8, os.cpu_count() or 1))
        os.environ["OMP_NUM_THREADS"] = str(value)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    return target


def append_progress(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def load_benchmark_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    dataset_root: str | Path,
    verify_raw_sha256: bool,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Benchmark32 SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != BENCHMARK_SCHEMA:
        raise RuntimeError(f"unexpected Benchmark32 schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise RuntimeError(f"Benchmark32 must contain 32 records, found {len(records or [])}")
    ids = [str(record["youtubeId"]) for record in records]
    if len(set(ids)) != 32:
        raise RuntimeError("Benchmark32 contains duplicate youtubeId values")
    root = Path(dataset_root).expanduser().resolve()
    for record in records:
        video_path = root / str(record["relative_path"])
        if not video_path.is_file():
            raise RuntimeError(f"Benchmark32 raw video missing: {video_path}")
        expected_size = int(record["file_size_bytes"])
        if video_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Benchmark32 raw size mismatch for {record['youtubeId']}: "
                f"expected {expected_size}, got {video_path.stat().st_size}"
            )
        if verify_raw_sha256:
            expected_raw = str(record.get("metadata", {}).get("raw_sha256") or "")
            if expected_raw and sha256_file(video_path) != expected_raw:
                raise RuntimeError(f"Benchmark32 raw SHA256 mismatch: {record['youtubeId']}")
    return payload


def _base_entry(record: Mapping[str, Any]) -> IndexEntry:
    return IndexEntry(
        dataset_id=".",
        video_id=str(record["youtubeId"]),
        realized_video_id=str(record["youtubeId"]),
        category=str(record["physical_resolution_arm"]),
        split="TRAIN",
        relative_video_path=str(record["relative_path"]),
        source_sha256=str(record.get("metadata", {}).get("raw_sha256") or "0" * 64),
        annotation_identity=PERFORMANCE_ONLY,
    )


def prepare_index(
    benchmark: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    output_path: str | Path,
    ffprobe_bin: str = "ffprobe",
) -> list[IndexEntry]:
    entries = []
    for record in benchmark["records"]:
        entry = probe_index_entry(
            _base_entry(record),
            dataset_root,
            verify_sha256=False,
            extract_pts=False,
            ffprobe_bin=ffprobe_bin,
        )
        if entry.decode_status != DECODE_OK:
            raise RuntimeError(f"probe failed for {entry.video_id}: {entry.decode_error}")
        # Some PHD2 WebM streams omit container ``nb_frames``.  The generic
        # probe then estimates frame_count from duration * average fps, which
        # can be one frame larger than the authoritative per-frame PTS table.
        # For VFR/PTS_TABLE inputs, use the actual PTS row count as the physical
        # frame count; no timestamp is synthesized or interpolated.
        if entry.timestamp_mode != "CFR_FPS":
            pts_values, pts_source = extract_pts_with_fallback(
                resolve_video_path(dataset_root, entry),
                expected_frame_count=None,
                ffprobe_bin=ffprobe_bin,
            )
            entry = replace(
                entry,
                frame_count=len(pts_values),
                pts_timestamps=tuple(str(value) for value in pts_values),
                pts_source=pts_source,
                probe_source=f"{entry.probe_source}+authoritative_pts_count",
            )
        entries.append(entry)
    write_index(
        entries,
        metadata={
            "dataset_id": ".",
            "protocol_name": PERFORMANCE_ONLY,
            "protocol_version": "v1",
            "seed": benchmark.get("selection", {}).get("seed"),
            "split_manifest_sha256": benchmark.get("source_manifest_sha256"),
            "split_content_sha256": benchmark.get("source_manifest_sha256"),
        },
        output_path=output_path,
    )
    return entries


class PhaseState:
    def __init__(self) -> None:
        self._value = "startup"
        self._lock = threading.Lock()

    def set(self, value: str) -> None:
        with self._lock:
            self._value = value

    def get(self) -> str:
        with self._lock:
            return self._value


class SystemTelemetry:
    FIELDS = (
        "timestamp_utc",
        "monotonic_seconds",
        "phase",
        "gpu_util_percent",
        "vram_used_mib",
        "gpu_power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "cpu_util_percent",
        "ram_used_bytes",
        "disk_read_bytes",
        "disk_write_bytes",
    )

    def __init__(self, path: str | Path, phase: PhaseState, interval_seconds: float = 1.0):
        self.path = Path(path)
        self.phase = phase
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="system-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))

    def _gpu(self) -> list[str | None]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(command, text=True, timeout=5).splitlines()[0]
            return [item.strip() for item in output.split(",")]
        except (OSError, subprocess.SubprocessError, IndexError):
            return [None] * 5

    def _run(self) -> None:
        import datetime as dt
        import psutil

        started = time.monotonic()
        psutil.cpu_percent(interval=None)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                gpu = self._gpu()
                memory = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                writer.writerow(
                    {
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "monotonic_seconds": round(time.monotonic() - started, 6),
                        "phase": self.phase.get(),
                        "gpu_util_percent": gpu[0],
                        "vram_used_mib": gpu[1],
                        "gpu_power_w": gpu[2],
                        "sm_clock_mhz": gpu[3],
                        "memory_clock_mhz": gpu[4],
                        "cpu_util_percent": psutil.cpu_percent(interval=None),
                        "ram_used_bytes": memory.used,
                        "disk_read_bytes": getattr(disk, "read_bytes", None),
                        "disk_write_bytes": getattr(disk, "write_bytes", None),
                    }
                )
                handle.flush()
                self._stop.wait(self.interval_seconds)


class ProfileEvents:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(
        self,
        stage: str,
        wall_seconds: float,
        *,
        count: int = 0,
        unit: str = "items",
        video_id: str | None = None,
        available: bool = True,
        note: str | None = None,
    ) -> None:
        row = {
            "stage": stage,
            "wall_seconds": float(wall_seconds),
            "count": int(count),
            "unit": unit,
            "video_id": video_id,
            "available": bool(available),
            "note": note,
        }
        with self._lock:
            self.rows.append(row)

    def unavailable(self, stage: str, note: str) -> None:
        self.add(stage, 0.0, available=False, note=note)

    def summary(self, total_wall_seconds: float) -> list[dict[str, Any]]:
        names = sorted({str(row["stage"]) for row in self.rows})
        result = []
        for name in names:
            selected = [row for row in self.rows if row["stage"] == name]
            available = all(bool(row["available"]) for row in selected)
            seconds = sum(float(row["wall_seconds"]) for row in selected)
            count = sum(int(row["count"]) for row in selected)
            units = sorted({str(row["unit"]) for row in selected})
            result.append(
                {
                    "stage": name,
                    "wall_seconds": round(seconds, 6) if available else None,
                    "count": count if available else None,
                    "unit": units[0] if len(units) == 1 else "mixed",
                    "throughput": round(count / seconds, 6) if available and seconds > 0 else None,
                    "percentage_total": (
                        round(100.0 * seconds / total_wall_seconds, 6)
                        if available and total_wall_seconds > 0
                        else None
                    ),
                    "available": available,
                    "note": next((row["note"] for row in selected if row.get("note")), None),
                }
            )
        return result

    def write_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("stage", "wall_seconds", "count", "unit", "video_id", "available", "note"),
            )
            writer.writeheader()
            writer.writerows(self.rows)
        return target


def _record_map(benchmark: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(record["youtubeId"]): record for record in benchmark["records"]}


def _source_duration(benchmark: Mapping[str, Any], entries: Sequence[IndexEntry]) -> float:
    records = _record_map(benchmark)
    return sum(float(records[entry.video_id]["duration_seconds"]) for entry in entries)


def run_retrieval(
    entries: Sequence[IndexEntry],
    *,
    dataset_root: Path,
    output_root: Path,
    base_url: str,
    concurrency: int,
    timeout_sec: float,
    events: ProfileEvents,
    phase: PhaseState,
    resume: bool = False,
) -> dict[str, Any]:
    from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient

    if concurrency <= 0:
        raise ValueError("Qwen concurrency must be positive")
    cache_root = output_root / "retrieval"
    cache_root.mkdir(parents=True, exist_ok=True)
    settings = RetrievalSettings(base_url=base_url, timeout_sec=timeout_sec)

    def _one(entry: IndexEntry) -> tuple[str, dict[str, Any], float]:
        client = QwenVLLMClient(base_url=base_url, model=QWEN_MODEL, timeout_sec=timeout_sec)
        started = time.perf_counter()
        payload = run_retrieval_for_entry(
            entry,
            dataset_root=dataset_root,
            client=client,
            settings=settings,
        )
        write_started = time.perf_counter()
        atomic_json(cache_root / f"{entry.video_id}.json", payload)
        write_s = time.perf_counter() - write_started
        return entry.video_id, payload, write_s

    phase.set("qwen_retrieval")
    wall_started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    writes: dict[str, float] = {}
    cache_hits = 0
    pending = []
    for entry in entries:
        path = cache_root / f"{entry.video_id}.json"
        if resume and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("video_id") != entry.video_id:
                raise RuntimeError(f"foreign retrieval cache for {entry.video_id}: {path}")
            results[entry.video_id] = payload
            writes[entry.video_id] = 0.0
            cache_hits += 1
            events.add("retrieval_cache_hit", 0.0, count=1, unit="videos", video_id=entry.video_id)
        else:
            pending.append(entry)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="qwen-request") as executor:
        futures = {executor.submit(_one, entry): entry.video_id for entry in pending}
        try:
            for future in as_completed(futures):
                video_id, payload, write_s = future.result()
                results[video_id] = payload
                writes[video_id] = write_s
                progress_started = time.perf_counter()
                append_progress(
                    output_root / "progress.jsonl",
                    {
                        "stage": "retrieval",
                        "video_id": video_id,
                        "completed": len(results),
                        "total": len(entries),
                        "monotonic_seconds": time.monotonic(),
                    },
                )
                events.add(
                    "status_progress_write",
                    time.perf_counter() - progress_started,
                    count=1,
                    unit="writes",
                    video_id=video_id,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    wall_s = time.perf_counter() - wall_started
    chunk_count = 0
    retry_count = 0
    for entry in entries:
        payload = results[entry.video_id]
        timing = payload.get("timing", {})
        chunks = payload.get("chunks", [])
        chunk_count += len(chunks)
        recovery = payload.get("provenance", {}).get("recovery_log", [])
        retry_count += sum(1 for row in recovery if isinstance(row, dict) and row.get("attempt"))
        events.add("qwen_metadata", timing.get("probe_sec", 0.0), count=1, unit="videos", video_id=entry.video_id)
        events.add("qwen_chunk_extraction", timing.get("chunk_extraction_sec", 0.0), count=len(chunks), unit="chunks", video_id=entry.video_id)
        events.add("qwen_http_request", timing.get("model_inference_sec", 0.0), count=len(chunks), unit="chunks", video_id=entry.video_id)
        events.add("qwen_response_parse", timing.get("parsing_sec", 0.0), count=len(chunks), unit="chunks", video_id=entry.video_id)
        events.add("candidate_merge", timing.get("merging_sec", 0.0), count=len(payload.get("merged_candidates", [])), unit="candidates", video_id=entry.video_id)
        events.add("retrieval_cache_write", writes[entry.video_id], count=1, unit="files", video_id=entry.video_id)
    events.add("qwen_stage_wall", wall_s, count=chunk_count, unit="chunks")
    events.add("qwen_retry", 0.0, count=retry_count, unit="requests", note="retry latency is included in qwen_http_request")
    events.unavailable("qwen_request_queue_wait", "vLLM client boundary exposes request wall time but not per-request queue time")
    events.unavailable("qwen_model_generation", "vLLM client boundary does not separate decode/preprocess/queue/generation")
    events.unavailable("qwen_frame_sampling_preprocess", "performed inside vLLM media loader; not separately exposed")
    return {
        "wall_seconds": wall_s,
        "video_count": len(entries),
        "chunk_count": chunk_count,
        "retry_count": retry_count,
        "cache_hit_count": cache_hits,
        "executed_video_count": len(pending),
        "concurrency": concurrency,
        "cache_root": str(cache_root),
    }


def run_detection(
    entries: Sequence[IndexEntry],
    *,
    dataset_root: Path,
    output_root: Path,
    rtdetr_snapshot: Path,
    batch_size: int,
    decode_window: int,
    decode_mode: str,
    events: ProfileEvents,
    phase: PhaseState,
    localizer: Any | None = None,
    model_load_seconds: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    import torch

    cached: dict[str, dict[str, Any]] = {}
    pending = []
    for entry in entries:
        artifact = output_root / "detections" / f"{entry.video_id}.npz"
        profile_path = output_root / "detection_profiles" / f"{entry.video_id}.json"
        if resume and artifact.is_file() and profile_path.is_file():
            result = json.loads(profile_path.read_text(encoding="utf-8"))
            if result.get("video_id") != entry.video_id:
                raise RuntimeError(f"foreign detection cache for {entry.video_id}: {profile_path}")
            with np.load(artifact, allow_pickle=False) as payload:
                if int(np.asarray(payload["frames"]).shape[0]) != int(result["frames"]):
                    raise RuntimeError(f"invalid detection cache frame count: {artifact}")
            cached[entry.video_id] = result
        else:
            pending.append(entry)
    owns_localizer = localizer is None and bool(pending)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if owns_localizer:
        phase.set("rtdetr_model_load")
        load_started = time.perf_counter()
        localizer = create_rtdetr_localizer(
            rtdetr_snapshot=rtdetr_snapshot,
            rtdetr_model_id=RTDETR_MODEL,
            device="cuda",
        )
        load_s = time.perf_counter() - load_started
        events.add("rtdetr_model_load", load_s, count=1, unit="models")
    elif localizer is not None:
        load_s = float(model_load_seconds or 0.0)
        events.add("rtdetr_model_load", 0.0, count=0, unit="models", note="shared sweep model")
    else:
        load_s = 0.0
        events.add("rtdetr_model_load", 0.0, count=0, unit="models", note="all videos restored from cache")
    phase.set("rtdetr_detection")
    wall_started = time.perf_counter()
    rows = []
    executed_index = 0
    try:
        for index, entry in enumerate(entries):
            cache_hit = entry.video_id in cached
            if cache_hit:
                result = cached[entry.video_id]
                events.add("detection_cache_hit", 0.0, count=1, unit="videos", video_id=entry.video_id)
            else:
                assert localizer is not None
                result = run_detection_for_entry(
                    entry,
                    dataset_root=dataset_root,
                    work_root=output_root,
                    rtdetr_snapshot=rtdetr_snapshot,
                    rtdetr_model_id=RTDETR_MODEL,
                    device="cuda",
                    batch_size=batch_size,
                    top_k=100,
                    decode_window=decode_window,
                    decode_mode=decode_mode,
                    localizer=localizer,
                    model_load_s=load_s if executed_index == 0 else 0.0,
                    model_load_count=1 if executed_index == 0 and owns_localizer else 0,
                )
                executed_index += 1
            rows.append(result)
            if not cache_hit:
                atomic_json(output_root / "detection_profiles" / f"{entry.video_id}.json", result)
            progress_started = time.perf_counter()
            append_progress(
                output_root / "progress.jsonl",
                {
                    "stage": "detection",
                    "video_id": entry.video_id,
                    "completed": index + 1,
                    "total": len(entries),
                    "monotonic_seconds": time.monotonic(),
                },
            )
            events.add(
                "status_progress_write",
                time.perf_counter() - progress_started,
                count=1,
                unit="writes",
                video_id=entry.video_id,
            )
            profile = result["profile"]
            stage_map = {
                "grid_creation": "grid_creation_seconds",
                "grid_serialization": "grid_serialization_seconds",
                "video_open": "video_open_seconds",
                "decode_grab": "grab_seconds",
                "decode_retrieve": "retrieve_seconds",
                "image_processor": "image_processor_seconds",
                "h2d": "h2d_seconds",
                "rtdetr_pure_forward": "rtdetr_pure_forward_cuda_seconds",
                "gap": "gap_seconds",
                "rtdetr_postprocess": "postprocess_seconds",
                "d2h": "d2h_seconds",
                "detection_npz_serialization": "npz_serialization_compression_write_seconds",
            }
            frames = int(result["frames"])
            if not cache_hit:
                for stage, key in stage_map.items():
                    events.add(stage, profile.get(key, 0.0), count=frames, unit="frames", video_id=entry.video_id)
                events.add("decode_wall", profile.get("decode_wall_seconds", 0.0), count=frames, unit="frames", video_id=entry.video_id)
    finally:
        if owns_localizer:
            del localizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    wall_s = time.perf_counter() - wall_started
    frame_count = sum(int(row["frames"]) for row in rows)
    forward_cuda_s = sum(float(row["profile"]["rtdetr_pure_forward_cuda_seconds"]) for row in rows)
    totals = {
        "video_capture_open_count": sum(int(row["profile"]["capture_open_count"]) for row in rows),
        "grabbed_frame_count": sum(int(row["profile"]["grabbed_frame_count"]) for row in rows),
        "retrieved_frame_count": sum(int(row["profile"]["retrieved_frame_count"]) for row in rows),
        "emitted_frame_count": sum(int(row["profile"]["emitted_frame_count"]) for row in rows),
        "repeated_prefix_decoded_frame_estimate": sum(int(row["profile"]["repeated_prefix_decoded_frame_estimate"]) for row in rows),
        "rtdetr_forward_count": sum(int(row["profile"]["rtdetr_forward_count"]) for row in rows),
        "processed_frame_count": frame_count,
    }
    events.add("rtdetr_stage_wall", wall_s, count=frame_count, unit="frames")
    return {
        "wall_seconds": wall_s,
        "video_count": len(entries),
        "frame_count": frame_count,
        "batch_size": batch_size,
        "decode_mode": decode_mode,
        "model_load_count": 1 if owns_localizer else 0,
        "cache_hit_count": len(cached),
        "executed_video_count": len(pending),
        "model_load_seconds": load_s,
        "end_to_end_fps": frame_count / wall_s if wall_s > 0 else None,
        "pure_forward_fps": frame_count / forward_cuda_s if forward_cuda_s > 0 else None,
        "pure_forward_cuda_seconds": forward_cuda_s,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None,
        "counters": totals,
        "work_root": str(output_root),
    }


def _vllm_client_healthy(base_url: str, timeout_sec: float):
    from aic_video_highlight.retrieval.qwen_vllm_client import QwenVLLMClient

    client = QwenVLLMClient(base_url=base_url, model=QWEN_MODEL, timeout_sec=timeout_sec)
    try:
        return bool(client.health_check())
    except Exception:
        return False


def run_benchmark(
    *,
    benchmark_manifest: str | Path,
    benchmark_sha256: str,
    dataset_root: str | Path,
    environment_path: str | Path,
    run_root: str | Path,
    run_id: str,
    stages: Sequence[str],
    qwen_concurrency: int,
    rtdetr_batch_size: int,
    decode_mode: str,
    verify_raw_sha256: bool,
    entry_ids: Sequence[str] | None = None,
    index_path_override: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    from aic_video_highlight.runtime.orchestrator import start_vllm, stop_vllm

    omp_threads = ensure_positive_omp_threads()
    root = Path(run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_root).expanduser().resolve()
    environment = EnvironmentPaths.from_json(environment_path)
    benchmark = load_benchmark_manifest(
        benchmark_manifest,
        expected_sha256=benchmark_sha256,
        dataset_root=dataset,
        verify_raw_sha256=verify_raw_sha256,
    )
    index_path = (
        Path(index_path_override).expanduser().resolve()
        if index_path_override is not None
        else root / "benchmark_index.json"
    )
    if index_path.is_file():
        entries, _ = load_index(index_path)
    else:
        entries = prepare_index(benchmark, dataset_root=dataset, output_path=index_path)
    if entry_ids is not None:
        wanted = set(entry_ids)
        entries = [entry for entry in entries if entry.video_id in wanted]
        missing = wanted - {entry.video_id for entry in entries}
        if missing:
            raise RuntimeError(f"unknown benchmark video ids: {sorted(missing)}")
    records_by_id = _record_map(benchmark)
    entries.sort(key=lambda entry: list(records_by_id).index(entry.video_id))

    events = ProfileEvents()
    phase = PhaseState()
    telemetry = SystemTelemetry(root / "system_metrics.csv", phase)
    telemetry.start()
    total_started = time.perf_counter()
    qwen_summary = None
    detection_summary = None
    qwen_service_start_count = 0
    qwen_service_load_s = 0.0
    owned = None
    try:
        if "retrieval" in stages:
            phase.set("qwen_service_start")
            base_url = "http://127.0.0.1:8000/v1"
            retrieval_missing = [
                entry
                for entry in entries
                if not (root / "retrieval" / f"{entry.video_id}.json").is_file()
            ]
            needs_service = bool(retrieval_missing) or not resume
            if needs_service and not _vllm_client_healthy(base_url, 30.0):
                started = time.perf_counter()
                config = {
                    "inference": {
                        "python": os.sys.executable,
                        "qwen_snapshot": {"base": "models", "path": "Qwen3.5-4B"},
                        "video_root": {"base": "datasets", "path": str(dataset)},
                        "vllm_base_url": base_url,
                        "vllm_gpu_memory_utilization": 0.80,
                        "vllm_max_model_len": 32768,
                        "vllm_start_timeout_sec": 900,
                    }
                }
                owned, base_url = start_vllm(config, environment, root / "logs" / "vllm.log")
                qwen_service_load_s = time.perf_counter() - started
                qwen_service_start_count = 1
            events.add("qwen_service_start", qwen_service_load_s, count=qwen_service_start_count, unit="starts")
            qwen_summary = run_retrieval(
                entries,
                dataset_root=dataset,
                output_root=root,
                base_url=base_url,
                concurrency=qwen_concurrency,
                timeout_sec=1800.0,
                events=events,
                phase=phase,
                resume=resume,
            )
            if owned is not None:
                phase.set("qwen_service_stop")
                stop_vllm(owned)
                owned = None
        if "detection" in stages:
            detection_summary = run_detection(
                entries,
                dataset_root=dataset,
                output_root=root,
                rtdetr_snapshot=environment.models / "rtdetr_r50vd",
                batch_size=rtdetr_batch_size,
                decode_window=64,
                decode_mode=decode_mode,
                events=events,
                phase=phase,
                resume=resume,
            )
    finally:
        if owned is not None:
            stop_vllm(owned)
        phase.set("finalize")
        telemetry.stop()
    total_wall_s = time.perf_counter() - total_started
    source_duration_s = _source_duration(benchmark, entries)
    profile_summary = events.summary(total_wall_s)
    events.write_csv(root / "stage_events.csv")
    with (root / "stage_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "wall_seconds", "count", "unit", "throughput", "percentage_total", "available", "note"),
        )
        writer.writeheader()
        writer.writerows(profile_summary)
    summary = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "purpose": PERFORMANCE_ONLY,
        "benchmark_manifest": str(Path(benchmark_manifest).resolve()),
        "benchmark_sha256": benchmark_sha256,
        "selected_video_ids": [entry.video_id for entry in entries],
        "video_count": len(entries),
        "source_duration_seconds": source_duration_s,
        "total_wall_seconds": total_wall_s,
        "rtf": total_wall_s / source_duration_s if source_duration_s > 0 else None,
        "qwen_service_start_count": qwen_service_start_count,
        "rtdetr_model_load_count": detection_summary["model_load_count"] if detection_summary else 0,
        "retrieval": qwen_summary,
        "detection": detection_summary,
        "profile_summary": profile_summary,
        "system_metrics": str(root / "system_metrics.csv"),
        "stage_events": str(root / "stage_events.csv"),
        "stage_summary": str(root / "stage_summary.csv"),
        "omp_num_threads": omp_threads,
    }
    atomic_json(root / "summary.json", summary)
    return summary


def subset_manifest(
    benchmark: Mapping[str, Any],
    *,
    purpose: str,
    output_path: str | Path,
    benchmark_sha256: str | None = None,
) -> tuple[Path, str, list[str]]:
    records = list(benchmark["records"])
    bin_order = ("0-1min", "1-3min", "3-10min", "10-30min")
    selected: list[Mapping[str, Any]] = []
    if purpose == "qwen_sweep":
        for duration_bin in bin_order:
            pool = sorted(
                (record for record in records if record["duration_bin"] == duration_bin),
                key=lambda record: (record.get("selection_key", ""), record["youtubeId"]),
            )
            if duration_bin == "10-30min":
                selected.extend([record for record in pool if record["physical_resolution_arm"] == "v360"][:2])
            else:
                selected.append(next(record for record in pool if record["physical_resolution_arm"] == "v360"))
                selected.append(next(record for record in pool if record["physical_resolution_arm"] == "v720"))
    elif purpose == "rtdetr_sweep":
        by_duration = sorted(records, key=lambda record: (-float(record["duration_seconds"]), record["youtubeId"]))
        selected = by_duration[:3]
        selected.extend(
            record
            for record in sorted(records, key=lambda row: (float(row["duration_seconds"]), row["youtubeId"]))
            if record["physical_resolution_arm"] == "v720" and record not in selected
        )
        selected = selected[:6]
    else:
        raise ValueError(f"unknown subset purpose: {purpose}")
    payload = {
        "schema": "aic.phd2.upstream-sweep-subset/v1",
        "purpose": purpose,
        "source_benchmark_sha256": benchmark.get("source_manifest_sha256"),
        "benchmark32_sha256": benchmark_sha256,
        "count": len(selected),
        "records": selected,
    }
    path = atomic_json(output_path, payload)
    digest = sha256_file(path)
    return path, digest, [str(record["youtubeId"]) for record in selected]


def _candidate_semantics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "chunk_index": row.get("chunk_index"),
                "parse_success": row.get("parse_success"),
                "parsed_segment_count": row.get("parsed_segment_count"),
            }
            for row in payload.get("chunks", [])
        ],
        "raw_candidates": payload.get("raw_candidates", []),
        "merged_candidates": payload.get("merged_candidates", []),
    }


def compare_retrieval_roots(
    baseline_root: str | Path,
    challenger_root: str | Path,
    video_ids: Sequence[str],
) -> dict[str, Any]:
    baseline = Path(baseline_root)
    challenger = Path(challenger_root)
    rows = []
    for video_id in video_ids:
        left = json.loads((baseline / "retrieval" / f"{video_id}.json").read_text(encoding="utf-8"))
        right = json.loads((challenger / "retrieval" / f"{video_id}.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "video_id": video_id,
                "semantic_equal": _candidate_semantics(left) == _candidate_semantics(right),
            }
        )
    return {"pass": all(row["semantic_equal"] for row in rows), "rows": rows}


def compare_detection_roots(
    baseline_root: str | Path,
    challenger_root: str | Path,
    video_ids: Sequence[str],
) -> dict[str, Any]:
    baseline = Path(baseline_root)
    challenger = Path(challenger_root)
    frame_rows = []
    detection_rows = []
    for video_id in video_ids:
        with np.load(baseline / "detections" / f"{video_id}.npz", allow_pickle=False) as left, np.load(
            challenger / "detections" / f"{video_id}.npz", allow_pickle=False
        ) as right:
            grid_equal = all(
                np.array_equal(left[key], right[key])
                for key in ("frames", "timestamps", "adjacency", "target_timestamps")
            )
            frame_rows.append({"video_id": video_id, "grid_equal": grid_equal})
            metrics: dict[str, Any] = {
                "video_id": video_id,
                "labels_equal": bool(np.array_equal(left["labels"], right["labels"])),
                "offsets_equal": bool(np.array_equal(left["offsets"], right["offsets"])),
            }
            for key in ("boxes", "scores", "visual"):
                left_array = np.asarray(left[key])
                right_array = np.asarray(right[key])
                if left_array.shape != right_array.shape:
                    metrics[f"{key}_shape_equal"] = False
                    metrics[f"{key}_max_abs"] = None
                    metrics[f"{key}_mean_abs"] = None
                    metrics[f"{key}_allclose"] = False
                    metrics[f"{key}_finite"] = False
                    continue
                difference = np.abs(
                    left_array.astype(np.float64) - right_array.astype(np.float64)
                )
                atol = 1e-3 if left_array.dtype == np.float16 else 1e-6
                rtol = 1e-3 if left_array.dtype == np.float16 else 1e-5
                metrics[f"{key}_shape_equal"] = True
                metrics[f"{key}_max_abs"] = float(difference.max()) if difference.size else 0.0
                metrics[f"{key}_mean_abs"] = float(difference.mean()) if difference.size else 0.0
                metrics[f"{key}_allclose"] = bool(
                    np.allclose(left_array, right_array, rtol=rtol, atol=atol)
                )
                metrics[f"{key}_finite"] = bool(
                    np.isfinite(left_array).all() and np.isfinite(right_array).all()
                )
            metrics["pass"] = bool(
                metrics["labels_equal"]
                and metrics["offsets_equal"]
                and all(
                    metrics.get(f"{key}_shape_equal")
                    and metrics.get(f"{key}_allclose")
                    and metrics.get(f"{key}_finite")
                    for key in ("boxes", "scores", "visual")
                )
            )
            detection_rows.append(metrics)
    return {
        "pass": all(row["grid_equal"] for row in frame_rows)
        and all(row["pass"] for row in detection_rows),
        "frame_grid_equivalence": all(row["grid_equal"] for row in frame_rows),
        "rtdetr_equivalence": all(row["pass"] for row in detection_rows),
        "frame_grid": frame_rows,
        "rtdetr": detection_rows,
    }


def compare_runs(baseline_root: str | Path, optimized_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    baseline = Path(baseline_root)
    optimized = Path(optimized_root)
    baseline_summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    optimized_summary = json.loads((optimized / "summary.json").read_text(encoding="utf-8"))
    ids = baseline_summary["selected_video_ids"]
    if ids != optimized_summary["selected_video_ids"]:
        raise RuntimeError("run membership/order mismatch")
    qwen = compare_retrieval_roots(baseline, optimized, ids)
    detection = compare_detection_roots(baseline, optimized, ids)
    report = {
        "schema": CORRECTNESS_SCHEMA,
        "baseline_root": str(baseline.resolve()),
        "optimized_root": str(optimized.resolve()),
        "video_count": len(ids),
        "qwen_semantic_equivalence": qwen["pass"],
        "frame_grid_equivalence": detection["frame_grid_equivalence"],
        "rtdetr_equivalence": detection["rtdetr_equivalence"],
        "downstream_equivalence": "NOT_APPLICABLE_UPSTREAM_ONLY",
        "pass": qwen["pass"] and detection["pass"],
        "qwen": qwen["rows"],
        "frame_grid": detection["frame_grid"],
        "rtdetr": detection["rtdetr"],
    }
    atomic_json(output_path, report)
    return report


__all__ = [
    "BENCHMARK_SCHEMA",
    "CORRECTNESS_SCHEMA",
    "ONE_PASS",
    "PERFORMANCE_ONLY",
    "RUN_SCHEMA",
    "WINDOWED_PREFIX",
    "compare_runs",
    "compare_detection_roots",
    "compare_retrieval_roots",
    "load_benchmark_manifest",
    "prepare_index",
    "run_benchmark",
    "sha256_file",
    "subset_manifest",
]
