#!/usr/bin/env python3
"""VHiCraft-v1 inference orchestrator.

This is an integration layer only.  It invokes/reuses the frozen scientific
implementations for Qwen retrieval, candidate merging, frame projection,
RT-DETR/primary-subject policy, CMP-1, TS-5 Revised and FS-0.  The code in this
file is deliberately limited to artifact handoff, provenance, validation,
resume, progress and official-output assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    _REPO = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))

from aic_video_highlight.runtime.datasets import materialize_numbered_video_inputs
from aic_video_highlight.runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.runtime.paths import EnvironmentPaths
from aic_video_highlight.runtime.profiles import (
    RuntimeProfile,
    default_runtime_profile,
    load_runtime_profile,
    runtime_machine_identity,
)
from aic_video_highlight.retrieval.candidate_cache import (
    export_candidate_cache,
    replay_cache,
    validate_cache,
)
from aic_video_highlight.composition.composition_pipeline import (
    FrozenInputs,
    compose_all_frames,
    engineering_gate,
    load_policy_shard_dir,
    policy_shard_semantic_sha,
)
from aic_video_highlight.composition.fresh_pipeline import (
    FRESH_SHARD_SCHEMA_VERSION,
    FreshPipelineError,
    audit_fresh_shard,
    build_fresh_index,
    build_shard_records,
    build_ts5_by_frame,
    shared_upstream_identity,
    validate_fresh_candidate_cache_binding,
)
from aic_video_highlight.composition.vhicraft_pipeline import (
    STABILIZED,
    TARGET_RATIO,
    UNSTABILIZED_CONTROL,
    FrameCrop,
    ablation_invariants,
    assemble_prediction_lines,
    fs0_identity_holds,
    validate_prediction_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRESH_PIPELINE_SCHEMA = "aic.vhicraft.inference-pipeline/v1"
RUN_IDENTITY_SCHEMA = "aic.vhicraft.run-identity/v2"
RETRIEVAL_FINAL_ROUTE = "frozen_retrieval_v0_no_selection_no_br1_no_dtl1"
CMP1_METHOD = "primary_subject_shifted_max_ratio_crop_v1"
TS5_METHOD = "projected_state_canonical_center_ema_v1"
FS0_METHOD = "all_frames_v1"
FROZEN_CACHE_SHA256 = "4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246"
QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_PROMPT = "high_recall_retrieval_v0"
RTDETR_MODEL = "PekingU/rtdetr_r50vd"
RTDETR_REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
RTDETR_POLICY = "primary_subject_policy_v1_person_priority"


def candidate_cache_split_for_input_mode(input_mode: str) -> str:
    """Map an orchestrator input route to its explicit cache provenance split."""
    return "test" if input_mode == "numbered_videos" else "dev"


RETRIEVAL_BACKENDS = ("vllm", "transformers-bnb-nf4")


def resolve_retrieval_backend(fresh: Mapping[str, Any], environment: EnvironmentPaths) -> str:
    """Resolve the retrieval engine from the deployment environment only.

    The inference profile describes the scientific pipeline; the deployment
    environment descriptor names the execution backend.  It is the single
    authority: the ``configs/environments/*.json`` file must declare
    ``retrieval_backend`` (``vllm`` on servers, ``transformers-bnb-nf4`` for the
    local low-memory profile).  A profile's historical ``qwen_backend`` hint is
    never used as a fallback, so no host can be silently routed to the
    bitsandbytes NF4 deployment, and a host that forgets to declare its backend
    fails fast instead of guessing.
    """
    backend = environment.retrieval_backend
    if backend is None:
        raise FreshPipelineError(
            "deployment environment does not declare retrieval_backend; refusing to "
            "silently fall back to a quantized backend (set 'vllm' for server GPUs or "
            "'transformers-bnb-nf4' for the local low-memory profile)"
        )
    if backend not in RETRIEVAL_BACKENDS:
        raise FreshPipelineError(
            f"unsupported qwen_backend: {backend} "
            f"(profile hint: {fresh.get('qwen_backend')!r})"
        )
    return backend


def build_retrieval_command(
    *,
    python: str,
    manifest: Path,
    video_root: Path,
    output_dir: Path,
    dataset_name: str,
    dataset_version: str,
    runtime_profile_path: Path,
    qwen_backend: str,
    local_model_path: Path | None,
    local_compute_dtype: str,
    base_url: str | None,
    resume: bool,
) -> list[str]:
    """Assemble the frozen retrieval-runner command for the resolved backend."""
    command = [
        python, "-u", "-m", "aic_video_highlight.retrieval.runner",
        "--manifest", str(manifest), "--video-root", str(video_root),
        "--output-dir", str(output_dir), "--config", str(REPO_ROOT / "configs/highlight_retrieval.yaml"),
        "--dataset-name", dataset_name, "--dataset-version", dataset_version,
        "--model-revision", QWEN_REVISION,
        "--runtime-profile", str(runtime_profile_path),
    ]
    if qwen_backend == "transformers-bnb-nf4":
        command.extend([
            "--local-model-path", str(local_model_path),
            "--local-quantization", "bnb-nf4",
            "--local-compute-dtype", local_compute_dtype,
        ])
    elif qwen_backend == "vllm":
        command.extend(["--base-url", str(base_url)])
    else:
        raise FreshPipelineError(f"unsupported qwen_backend: {qwen_backend}")
    if resume:
        command.append("--resume")
    return command


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FreshPipelineError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _resolve(spec: Mapping[str, Any], environment: EnvironmentPaths) -> Path:
    base = str(spec["base"])
    roots = {
        "repo": environment.repo,
        "outputs": environment.outputs,
        "datasets": environment.datasets,
        "models": environment.models,
        "hf_cache": environment.hf_cache,
        "derived": environment.derived,
    }
    if base not in roots or roots[base] is None:
        raise FreshPipelineError(f"unsupported fresh input base: {base}")
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else roots[base] / path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def build_run_identity(
    *,
    config_path: Path,
    protocol_path: Path,
    selected_manifest: Path,
    profile: str,
    runtime_profile: RuntimeProfile | None = None,
    runtime_profile_path: Path | None = None,
    scope: str | None = None,
    video_ids: Sequence[str] | None = None,
    retrieval_backend: str | None = None,
) -> dict[str, Any]:
    runtime_profile = runtime_profile or default_runtime_profile()
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "git_head": _git_head(),
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(protocol_path),
        "model": {"repo_id": QWEN_MODEL, "revision": QWEN_REVISION},
        "prompt": QWEN_PROMPT,
        "input_manifest_sha256": file_sha256(selected_manifest),
        "profile": profile,
        **runtime_profile.identity_fields(),
        "runtime": runtime_machine_identity(),
    }
    if scope is not None:
        identity["scope"] = scope
    if video_ids is not None:
        identity["video_ids"] = [str(video_id) for video_id in video_ids]
    if runtime_profile_path is not None:
        identity["runtime_profile_sha256"] = file_sha256(runtime_profile_path)
    if retrieval_backend is not None:
        identity["retrieval_backend"] = retrieval_backend
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def bind_resume(root: Path, identity: Mapping[str, Any], *, resume: bool) -> int:
    """Bind every resumable artifact below *root* to one immutable run identity."""
    path = root / "run_identity.json"
    if path.exists():
        existing = _read_json(path)
        if existing != dict(identity):
            raise FreshPipelineError("resume identity mismatch; refusing foreign fresh artifacts")
        if not resume:
            raise FreshPipelineError("fresh output already exists; pass --resume to reuse it")
        return 1
    _write_json(path, identity)
    return 0


class Progress:
    def __init__(self, total_videos: int) -> None:
        self.total_videos = total_videos
        self.started = time.monotonic()

    def emit(self, component: str, done: int, *, unit: str = "videos", detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.total_videos - done) / done if done else None
        eta_text = "?" if eta is None else f"{eta:.0f}s"
        print(
            f"[vhicraft] component={component} {unit}={done}/{self.total_videos} "
            f"elapsed={elapsed:.0f}s eta={eta_text} {detail}".rstrip(),
            flush=True,
        )


def select_dev_manifest(
    source_path: Path, role_records: Sequence[Mapping[str, Any]], output_path: Path,
    *, videos: int | None, video_ids: Sequence[str] | None = None,
) -> list[str]:
    source = {str(row["video_id"]): row for row in _read_jsonl(source_path)}
    ordered = [str(row["video_id"]) for row in role_records]
    if video_ids is not None:
        requested = [str(value) for value in video_ids]
        if not requested:
            raise FreshPipelineError("--video-ids must not be empty")
        unknown = [value for value in requested if value not in set(ordered)]
        if unknown:
            raise FreshPipelineError(f"--video-ids not in Dev role: {unknown[:3]}")
        ordered = requested
    elif videos is not None:
        if videos <= 0:
            raise FreshPipelineError("--videos must be positive")
        ordered = ordered[:videos]
    missing = [video_id for video_id in ordered if video_id not in source]
    if missing:
        raise FreshPipelineError(f"Dev role is not closed over source manifest: {missing[:3]}")
    rows = [source[video_id] for video_id in ordered]
    if any(row.get("split") != "dev" for row in rows):
        raise FreshPipelineError("fresh manifest contains a non-Dev record")
    _write_jsonl(output_path, rows)
    return ordered


def _service_has_model(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as response:
            payload = json.load(response)
        return any(item.get("id") == QWEN_MODEL for item in payload.get("data", []))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def start_vllm(
    config: Mapping[str, Any], environment: EnvironmentPaths, log_path: Path
) -> tuple[subprocess.Popen[Any] | None, str]:
    fresh = config["inference"]
    base_url = str(fresh.get("vllm_base_url", "http://127.0.0.1:8000/v1"))
    if _service_has_model(base_url):
        return None, base_url
    python = str(fresh.get("python", sys.executable))
    model_path = _resolve(fresh["qwen_snapshot"], environment)
    media_path = _resolve(fresh["video_root"], environment)
    command = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", QWEN_MODEL,
        "--host", "127.0.0.1", "--port", "8000",
        "--trust-remote-code",
        "--allowed-local-media-path", str(media_path),
        "--gpu-memory-utilization", str(fresh.get("vllm_gpu_memory_utilization", 0.80)),
        "--max-model-len", str(fresh.get("vllm_max_model_len", 32768)),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    child_environment = os.environ.copy()
    # Invoking an environment's Python by absolute path does not activate that
    # environment.  vLLM/FlashInfer launches the environment-provided ``ninja``
    # executable during warm-up, so its bin directory must be on child PATH.
    child_environment["PATH"] = str(Path(python).parent) + os.pathsep + child_environment.get("PATH", "")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=child_environment,
    )
    deadline = time.monotonic() + float(fresh.get("vllm_start_timeout_sec", 900))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise FreshPipelineError(f"vLLM exited during startup ({process.returncode}); see {log_path}")
        if _service_has_model(base_url):
            log.close()
            return process, base_url
        time.sleep(2)
    stop_vllm(process)
    log.close()
    raise FreshPipelineError(f"vLLM health check timed out; see {log_path}")


def stop_vllm(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


PROCESS_RESOURCES: list[dict[str, Any]] = []


WATCHDOG_DEFAULT_SEC = 900.0


def _safe_print(text: str) -> None:
    """Best-effort console output; a console codec must never break a run."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        try:
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)
        except (UnicodeEncodeError, OSError):
            pass
    except OSError:
        pass


def _child_environment() -> dict[str, str]:
    """Numerics-neutral child environment for observable, bounded GPU runs."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    allocator = env.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if "expandable_segments" not in allocator:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{allocator},expandable_segments:True" if allocator else "expandable_segments:True"
        )
    return env


def _watchdog_expired(last_activity_sec: float, now_sec: float, limit_sec: float) -> bool:
    """True when no observable child progress happened for *limit_sec*."""
    return limit_sec > 0 and (now_sec - last_activity_sec) > limit_sec


def _terminate_child(process: subprocess.Popen[Any]) -> None:
    """Graceful first, forceful last; never raise into the orchestrator loop."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except (OSError, ValueError):
        try:
            process.terminate()
        except OSError:
            pass


def _run(
    command: Sequence[str],
    *,
    label: str,
    log_path: Path | None = None,
    watchdog_sec: float = WATCHDOG_DEFAULT_SEC,
) -> None:
    _safe_print(f"[vhicraft] {label}: {' '.join(command)}")
    try:
        import psutil
    except ImportError:
        psutil = None
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", newline="\n")
    started = time.perf_counter()
    last_activity = [started]
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_child_environment(),
        creationflags=creationflags,
    )

    def _pump() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                last_activity[0] = time.monotonic()
                record = f"[{label}] {line.rstrip()}"
                if log_handle is not None:
                    log_handle.write(record + "\n")
                    log_handle.flush()
                _safe_print(record)
        except Exception:
            pass

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    peak_rss = 0
    watchdog_triggered = False
    while process.poll() is None:
        if _watchdog_expired(last_activity[0], time.monotonic(), watchdog_sec):
            watchdog_triggered = True
            message = (
                f"[vhicraft] WATCHDOG {label}: no child progress for "
                f"{watchdog_sec:.0f}s; terminating the child process tree"
            )
            _safe_print(message)
            if log_handle is not None:
                log_handle.write(message + "\n")
                log_handle.flush()
            _terminate_child(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            break
        if psutil is not None:
            try:
                parent = psutil.Process(process.pid)
                family = [parent, *parent.children(recursive=True)]
                peak_rss = max(peak_rss, sum(item.memory_info().rss for item in family))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.25)
    process.wait()
    reader.join(timeout=10)
    if log_handle is not None:
        log_handle.flush()
        log_handle.close()
    PROCESS_RESOURCES.append(
        {
            "label": label,
            "wall_sec": time.perf_counter() - started,
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_process_tree_rss_mib": peak_rss / (1024**2),
            "watchdog_triggered": watchdog_triggered,
        }
    )
    if watchdog_triggered:
        raise FreshPipelineError(
            f"{label} watchdog: no observable progress for {watchdog_sec:.0f}s; child terminated"
        )
    if process.returncode:
        raise FreshPipelineError(f"{label} failed with exit code {process.returncode}")


def _component_marker(path: Path, *, component: str, inputs: Mapping[str, str], outputs: Mapping[str, str]) -> None:
    _write_json(path, {
        "schema_version": FRESH_PIPELINE_SCHEMA,
        "component": component,
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
    })


def _marker_matches(path: Path, *, inputs: Mapping[str, str], outputs: Sequence[Path]) -> bool:
    if not path.is_file() or not all(item.exists() for item in outputs):
        return False
    marker = _read_json(path)
    if marker.get("inputs") != dict(sorted(inputs.items())):
        raise FreshPipelineError(f"resume component input identity mismatch: {path}")
    recorded = marker.get("outputs", {})
    for output in outputs:
        if output.is_file() and recorded.get(output.name) != file_sha256(output):
            raise FreshPipelineError(f"resume component output identity mismatch: {output}")
    return True


def _count_qwen_calls(raw_dir: Path) -> int:
    return sum(len(_read_json(path).get("chunks", [])) for path in sorted(raw_dir.glob("*.json")))


def _build_fresh_inputs(
    predictions_path: Path, metadata_path: Path, index_path: Path, localization_dir: Path,
) -> tuple[FrozenInputs, dict[str, Any]]:
    predictions = {row["video_id"]: row for row in _read_jsonl(predictions_path)}
    metadata = _read_json(metadata_path)["records"]
    index = {row["video_id"]: row for row in _read_jsonl(index_path)}
    policy = tuple(load_policy_shard_dir(localization_dir / "policy_v1"))
    policy_hash = policy_shard_semantic_sha(policy)
    expected = {
        (video_id, int(item["frame"]))
        for video_id, row in predictions.items() for item in row["predictions"]
    }
    actual = {(row["video_id"], int(row["frame"])) for row in policy}
    if expected != actual or len(actual) != len(policy):
        raise FreshPipelineError(
            f"spatial composition fresh binding mismatch: expected={len(expected)} actual={len(actual)}"
        )
    inputs = FrozenInputs(
        frame_projection_predictions=predictions,
        metadata=metadata,
        index=index,
        policy_records=policy,
        raw_frames={},
        weak_reference={},
        input_hashes={
            "frame_projection_predictions": file_sha256(predictions_path),
            "video_metadata_cache": file_sha256(metadata_path),
            "dev166_index": file_sha256(index_path),
            "subject_policy_artifact": policy_hash,
        },
    )
    return inputs, {"expected_frames": len(expected), "policy_frames": len(actual), "policy_sha256": policy_hash}


def _composition_manifest(inputs: FrozenInputs) -> dict[str, Any]:
    videos = []
    for video_id in inputs.index:
        policy = sorted(
            (row for row in inputs.policy_records if row["video_id"] == video_id),
            key=lambda row: int(row["frame"]),
        )
        meta = inputs.metadata[video_id]
        videos.append({
            "video_id": video_id,
            "image_width": int(meta["width"]),
            "image_height": int(meta["height"]),
            "target_ratio_wh": [float(v) for v in inputs.index[video_id]["targetRatioWH"]],
            "frame_count": len(policy),
            "frames": [{"frame": int(row["frame"])} for row in policy],
        })
    manifest = {
        "schema_version": "aic.vhicraft.composition-manifest/v1",
        "video_count": len(videos),
        "frame_count": sum(item["frame_count"] for item in videos),
        "videos": videos,
        "input_identities": inputs.input_hashes,
        "algorithm": CMP1_METHOD,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _assemble_variant(
    *, variant: str, index_rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
    shards_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    key = "ts5" if variant == STABILIZED else "ts0"
    crops_by_video: dict[str, dict[int, FrameCrop]] = {}
    fs0 = {"new": 0, "drop": 0, "changed": 0, "frame_count_before": 0, "frame_count_after": 0}
    for entry in index_rows:
        video_id = str(entry["video_id"])
        rows = json.loads((shards_dir / f"{video_id}.json").read_text(encoding="utf-8"))
        before = [int(row["frame"]) for row in rows]
        after = list(before)  # canonical FS-0 identity
        if not fs0_identity_holds(before, after):
            raise FreshPipelineError(f"FS-0 changed the frame set: {video_id}")
        fs0["frame_count_before"] += len(before)
        fs0["frame_count_after"] += len(after)
        crops_by_video[video_id] = {
            int(row["frame"]): FrameCrop(
                int(row["frame"]), int(row[key]["x"]), int(row[key]["y"]), int(row[key]["w"])
            )
            for row in rows
        }
    lines = assemble_prediction_lines(
        crops_by_video,
        target_ratio=TARGET_RATIO,
        frame_size_by_video={v: (int(metadata[v]["width"]), int(metadata[v]["height"])) for v in crops_by_video},
        frame_count_by_video={v: int(metadata[v]["frame_count"]) for v in crops_by_video},
    )
    predictions = output_dir / "predictions.jsonl"
    _write_jsonl(predictions, lines)
    report = validate_prediction_lines(
        predictions,
        index={str(row["video_id"]): row["targetRatioWH"] for row in index_rows},
        metadata={v: metadata[v] for v in crops_by_video},
    )
    _write_json(output_dir / "fs0_identity.json", {"method": FS0_METHOD, "identity": True, **fs0})
    _write_json(output_dir / "validation.json", {
        "status": "PASS" if report.is_valid else "FAIL",
        "contract": {"is_valid": report.is_valid, "stats": dict(report.stats), "issue_count": len(report.issues)},
    })
    return {
        "variant": variant,
        "predictions_path": str(predictions),
        "video_count": len(index_rows),
        "frame_count": fs0["frame_count_after"],
        "contract": {"is_valid": report.is_valid, "stats": dict(report.stats)},
        "errors": [],
        "fs0": {"identity": True, **fs0},
    }


def run_inference(
    config: Mapping[str, Any], environment: EnvironmentPaths, *, config_path: Path,
    protocol_path: Path, resume: bool, validate_only: bool = False,
    only_video_ids: Sequence[str] | None = None, output_run_id: str | None = None,
    videos: int | None = None,
    runtime_profile: RuntimeProfile | None = None,
    runtime_profile_path: Path | None = None,
) -> dict[str, Any]:
    PROCESS_RESOURCES.clear()
    pipeline_started = time.perf_counter()
    runtime_profile = runtime_profile or default_runtime_profile()
    runtime_profile_path = runtime_profile_path or (
        REPO_ROOT / "configs" / "runtime" / f"{runtime_profile.name}.json"
    )
    fresh = config.get("inference")
    if not isinstance(fresh, Mapping):
        raise FreshPipelineError("profile config lacks inference")
    inputs_config = config.get("inputs", {})
    if not isinstance(inputs_config, Mapping):
        raise FreshPipelineError("profile config lacks inputs")
    input_mode = str(inputs_config.get("mode", "dev_role"))
    candidate_cache_split = candidate_cache_split_for_input_mode(input_mode)
    run_id = str(output_run_id or "")
    if not run_id or Path(run_id).name != run_id:
        raise FreshPipelineError("run_id is required and must be a single path segment")
    root = environment.outputs / "vhicraft" / run_id
    dataset_name = str(inputs_config.get("dataset_name", "aic_highlight_dev"))
    dataset_version = str(inputs_config.get("dataset_version", "aic_highlight_dev_v1.1"))

    def _validated(missing: list[str]) -> dict[str, Any]:
        return {
            "status": "VALIDATED" if not missing else "MISSING_ENVIRONMENT_INPUTS",
            "missing": missing,
            "scientific_identities": {
                "qwen": {"repo_id": QWEN_MODEL, "revision": QWEN_REVISION, "prompt": QWEN_PROMPT},
                "temporal": RETRIEVAL_FINAL_ROUTE,
                "rtdetr": {"model": RTDETR_MODEL, "revision": RTDETR_REVISION, "policy": RTDETR_POLICY},
                "cmp1": CMP1_METHOD,
                "ts5": TS5_METHOD,
                "fs0": FS0_METHOD,
            },
        }

    if input_mode == "numbered_videos":
        video_root = _resolve(inputs_config["video_root"], environment)
        target_ratio = inputs_config.get("target_ratio_wh", list(TARGET_RATIO))
        if validate_only:
            missing = [str(path) for path in [video_root] if not path.exists()]
            return _validated(missing)
        materialized = materialize_numbered_video_inputs(
            video_root, root / "input", target_ratio, video_ids=only_video_ids
        )
        selected_manifest = materialized["manifest_path"]
        source_index = materialized["index_path"]
        role = {"records": materialized["records"]}
        video_ids = [str(record["video_id"]) for record in materialized["records"]]
    else:
        role_path = _resolve(inputs_config["role_manifest"], environment)
        source_manifest = _resolve(fresh["dev_manifest"], environment)
        source_index = _resolve(fresh["source_index"], environment)
        video_root = _resolve(fresh["video_root"], environment)
        if validate_only:
            required = [role_path, source_manifest, source_index, video_root]
            missing = [str(path) for path in required if not path.exists()]
            return _validated(missing)
        role = _read_json(role_path)
        selected_manifest = root / "input" / "dev_selected.jsonl"
        video_ids = select_dev_manifest(
            source_manifest, role["records"], selected_manifest,
            videos=videos, video_ids=only_video_ids,
        )
    profile_name = str(config.get("profile", "dev166"))
    scope = (
        "isolated_validation"
        if input_mode == "numbered_videos" and only_video_ids is not None
        else "formal_full"
        if input_mode == "numbered_videos"
        else "development"
    )
    qwen_backend = resolve_retrieval_backend(fresh, environment)
    identity = build_run_identity(
        config_path=config_path, protocol_path=protocol_path,
        selected_manifest=selected_manifest, profile=profile_name,
        runtime_profile=runtime_profile,
        runtime_profile_path=runtime_profile_path,
        scope=scope,
        video_ids=video_ids,
        retrieval_backend=qwen_backend,
    )
    resume_count = bind_resume(root, identity, resume=resume)
    progress = Progress(len(video_ids))
    python = str(fresh.get("python", sys.executable))

    # retrieval: real Qwen calls.  The selected Dev manifest is the exact input
    # consumed by the cache component below; it cannot silently fall back to frozen.
    retrieval = root / "retrieval"
    retrieval_marker = retrieval / "fresh_identity.json"
    retrieval_inputs = {"selected_manifest": file_sha256(selected_manifest), "run_identity": identity["identity_sha256"]}
    if not _marker_matches(retrieval_marker, inputs=retrieval_inputs, outputs=[retrieval / "predictions.jsonl", retrieval / "run_config.json"]):
        if runtime_profile.uses_local_efficient_sdpa and qwen_backend != "transformers-bnb-nf4":
            raise FreshPipelineError(
                "local_efficient_sdpa requires qwen_backend=transformers-bnb-nf4"
            )
        owned_vllm: subprocess.Popen[Any] | None = None
        try:
            local_model_path = (
                _resolve(fresh["qwen_snapshot"], environment)
                if qwen_backend == "transformers-bnb-nf4"
                else None
            )
            base_url: str | None = None
            if qwen_backend == "vllm":
                owned_vllm, base_url = start_vllm(config, environment, root / "logs" / "vllm.log")
            command = build_retrieval_command(
                python=python,
                manifest=selected_manifest,
                video_root=video_root,
                output_dir=retrieval,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                runtime_profile_path=runtime_profile_path,
                qwen_backend=qwen_backend,
                local_model_path=local_model_path,
                local_compute_dtype=str(fresh.get("qwen_compute_dtype", "float16")),
                base_url=base_url,
                resume=resume,
            )
            _run(
                command,
                label="Retrieval Qwen fresh inference",
                log_path=root / "logs" / "retrieval.log",
            )
        finally:
            stop_vllm(owned_vllm)
        _component_marker(retrieval_marker, component="retrieval_qwen", inputs=retrieval_inputs, outputs={
            "predictions.jsonl": file_sha256(retrieval / "predictions.jsonl"),
            "run_config.json": file_sha256(retrieval / "run_config.json"),
        })
    else:
        resume_count += 1
    qwen_calls = _count_qwen_calls(retrieval / "raw")
    if qwen_calls <= 0:
        raise FreshPipelineError("fresh retrieval contains zero Qwen calls")
    progress.emit("retrieval", len(video_ids), detail=f"qwen_calls={qwen_calls}")

    # temporal refinement final route: canonical cache builder + frozen no-selection route.
    temporal = root / "temporal"
    cache = temporal / "candidate_cache"
    final_segments = temporal / "final_segments.jsonl"
    temporal_marker = temporal / "fresh_identity.json"
    temporal_inputs = {"retrieval_predictions": file_sha256(retrieval / "predictions.jsonl"), "retrieval_run_config": file_sha256(retrieval / "run_config.json")}
    if not _marker_matches(temporal_marker, inputs=temporal_inputs, outputs=[cache / "cache_manifest.json", final_segments]):
        if cache.exists() and any(cache.iterdir()):
            raise FreshPipelineError("unbound fresh candidate cache exists; refusing overwrite")
        manifest = export_candidate_cache({candidate_cache_split: retrieval}, output_dir=cache)
        if manifest["global_semantic_sha256"] == FROZEN_CACHE_SHA256:
            raise FreshPipelineError("fresh inference unexpectedly resolved to the historical frozen candidate cache")
        replay_cache(cache, final_segments)
        _component_marker(temporal_marker, component="temporal_final_route", inputs=temporal_inputs, outputs={
            "cache_manifest.json": file_sha256(cache / "cache_manifest.json"),
            "final_segments.jsonl": file_sha256(final_segments),
        })
    else:
        resume_count += 1
    cache_summary = validate_cache(cache)
    cache_binding = validate_fresh_candidate_cache_binding(
        _read_json(cache / "cache_manifest.json"),
        retrieval_predictions_path=retrieval / "predictions.jsonl",
        expected_video_ids=video_ids,
        expected_split=candidate_cache_split,
        forbidden_global_sha256=FROZEN_CACHE_SHA256,
    )
    if cache_summary["record_count"] != len(video_ids):
        raise FreshPipelineError("fresh candidate cache count gate failed")
    progress.emit("temporal", len(video_ids), detail=f"cache={cache_summary['global_semantic_sha256'][:12]}")

    # frame projection canonical frame projection.
    projection_dir = root / "frame_projection"
    index_path = projection_dir / "fresh_index.jsonl"
    index_rows = build_fresh_index(role["records"], _read_jsonl(source_index), video_ids=video_ids)
    _write_jsonl(index_path, index_rows)
    projection_predictions = projection_dir / "predictions.jsonl"
    metadata_cache = projection_dir / "metadata_cache.json"
    projection_marker = projection_dir / "fresh_identity.json"
    projection_inputs = {"final_segments": file_sha256(final_segments), "fresh_index": file_sha256(index_path)}
    if not _marker_matches(projection_marker, inputs=projection_inputs, outputs=[projection_predictions, metadata_cache]):
        _run([
            python, "-u", "-m", "aic_video_highlight.composition.frame_projection_runner",
            "--index", str(index_path), "--video-dir", str(video_root),
            "--segments", str(final_segments), "--out", str(projection_predictions),
            "--metadata-cache", str(metadata_cache), "--pts-extract",
        ], label="Frame projection", log_path=root / "logs" / "frame_projection.log")
        _component_marker(projection_marker, component="frame_projection", inputs=projection_inputs, outputs={
            "predictions.jsonl": file_sha256(projection_predictions), "metadata_cache.json": file_sha256(metadata_cache),
        })
    else:
        resume_count += 1
    progress.emit("frame_projection", len(video_ids))

    # subject localization canonical RT-DETR + primary_subject_policy_v1 on fresh frames.
    localization_dir = root / "localization"
    localization_marker = localization_dir / "fresh_identity.json"
    localization_inputs = {"frame_projection_predictions": file_sha256(projection_predictions), "fresh_index": file_sha256(index_path), "metadata": file_sha256(metadata_cache)}
    localization_summary_path = localization_dir / "full_dev_summary.json"
    if not _marker_matches(localization_marker, inputs=localization_inputs, outputs=[localization_summary_path]):
        rtdetr_snapshot = _resolve(fresh["rtdetr_snapshot"], environment)
        _run([
            python, "-u", "-m", "aic_video_highlight.localization.localization_runner",
            "--index", str(index_path), "--video-base", str(video_root),
            "--predictions", str(projection_predictions), "--metadata-cache", str(metadata_cache),
            "--output-dir", str(localization_dir), "--model-id", RTDETR_MODEL,
            "--local-path", str(rtdetr_snapshot), "--device", "cuda", "--dtype", str(fresh.get("rtdetr_dtype", "float32")),
            "--mode", "FRESH_PIPELINE", "--expected-videos", str(len(video_ids)),
        ], label="Subject localization RT-DETR", log_path=root / "logs" / "localization.log")
        _component_marker(localization_marker, component="localization_rtdetr", inputs=localization_inputs, outputs={
            "full_dev_summary.json": file_sha256(localization_summary_path),
        })
    else:
        resume_count += 1
    localization_summary = _read_json(localization_summary_path)
    if any(int(localization_summary.get(key, -1)) != 0 for key in ("missing", "extra", "duplicates")):
        raise FreshPipelineError("subject localization fresh frame identity gate failed")
    rtdetr_calls = int(localization_summary["frames"])
    if rtdetr_calls <= 0:
        raise FreshPipelineError("fresh subject localization contains zero RT-DETR calls")
    progress.emit("localization", len(video_ids), unit="videos", detail=f"rtdetr_frames={rtdetr_calls}")

    # spatial composition: fresh identity binding; canonical CMP-1 implementation.
    composition_dir = root / "composition"
    composition_dir.mkdir(parents=True, exist_ok=True)
    composition_marker = composition_dir / "fresh_identity.json"
    inputs, composition_binding = _build_fresh_inputs(projection_predictions, metadata_cache, index_path, localization_dir)
    composition_manifest_payload = _composition_manifest(inputs)
    composition_manifest_path = composition_dir / "manifest.json"
    _write_json(composition_manifest_path, composition_manifest_payload)
    composition_records_path = composition_dir / "cmp1_records.json"
    composition_inputs = {
        "frame_projection_predictions": file_sha256(projection_predictions),
        "localization_policy": composition_binding["policy_sha256"],
        "manifest": file_sha256(composition_manifest_path),
        "algorithm": CMP1_METHOD,
    }
    if not _marker_matches(composition_marker, inputs=composition_inputs, outputs=[composition_records_path]):
        composition_records = compose_all_frames(composition_manifest_payload, inputs, TARGET_RATIO)
        composition_gate = engineering_gate(composition_manifest_payload, composition_records)
        if any(int(composition_gate[key]) for key in ("missing", "extra", "duplicates", "invalid_crop", "cmp0_frozen_regression")):
            raise FreshPipelineError(f"spatial composition fresh CMP-1 gate failed: {composition_gate}")
        _write_json(composition_records_path, {"algorithm": CMP1_METHOD, "records": composition_records, "gate": composition_gate})
        _component_marker(composition_marker, component="composition_cmp1", inputs=composition_inputs, outputs={
            "cmp1_records.json": file_sha256(composition_records_path),
        })
    else:
        resume_count += 1
    composition_records = _read_json(composition_records_path)["records"]
    progress.emit("composition", len(video_ids), detail=f"frames={len(composition_records)}")

    # temporal stabilization: canonical TS-5 Revised.  TS-0 is the unmodified CMP-1 fork.
    stabilization_dir = root / "stabilization"
    stabilization_shards = stabilization_dir / "shards"
    stabilization_shards.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in video_ids}
    for row in composition_records:
        by_video[str(row["video_id"])].append(row)
    audits = []
    for index, video_id in enumerate(video_ids, 1):
        shard_path = stabilization_shards / f"{video_id}.json"
        marker_path = stabilization_shards / f"{video_id}.identity.json"
        shard_inputs = {"composition": file_sha256(composition_records_path), "video_id": video_id, "algorithm": TS5_METHOD}
        if not _marker_matches(marker_path, inputs=shard_inputs, outputs=[shard_path]):
            meta = inputs.metadata[video_id]
            ts5 = build_ts5_by_frame(
                by_video[video_id], width=int(meta["width"]), height=int(meta["height"]),
                target_ratio=TARGET_RATIO, alpha=0.5,
            )
            shard = build_shard_records(by_video[video_id], ts5)
            _atomic_text(shard_path, json.dumps(shard, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            _component_marker(marker_path, component="stabilization_ts5_revised", inputs=shard_inputs, outputs={shard_path.name: file_sha256(shard_path)})
        else:
            resume_count += 1
        shard_rows = json.loads(shard_path.read_text(encoding="utf-8"))
        audits.append(asdict(audit_fresh_shard(video_id, shard_rows)))
        progress.emit("stabilization", index)
    audit_path = stabilization_dir / "audit.json"
    _write_json(audit_path, {"schema": FRESH_SHARD_SCHEMA_VERSION, "method": TS5_METHOD, "videos": audits})

    upstream_paths = {
        "retrieval_predictions": retrieval / "predictions.jsonl",
        "fresh_candidate_cache_manifest": cache / "cache_manifest.json",
        "temporal_final_segments": final_segments,
        "frame_projection_predictions": projection_predictions,
        "localization_summary": localization_summary_path,
        "composition_records": composition_records_path,
    }
    shared = shared_upstream_identity(upstream_paths)
    shared_path = root / "shared_upstream_identity.json"
    _write_json(shared_path, {**shared, "variants": [STABILIZED, UNSTABILIZED_CONTROL], "fork_component": "stabilization", "only_difference": "TS-5 Revised vs TS-0"})

    metadata = _read_json(metadata_cache)["records"]
    stabilized = _assemble_variant(variant=STABILIZED, index_rows=index_rows, metadata=metadata, shards_dir=stabilization_shards, output_dir=root / "predictions")
    control = _assemble_variant(variant=UNSTABILIZED_CONTROL, index_rows=index_rows, metadata=metadata, shards_dir=stabilization_shards, output_dir=root / "control_no_stabilization")
    if not stabilized["contract"]["is_valid"] or not control["contract"]["is_valid"]:
        raise FreshPipelineError("fresh official contract validation failed")
    # The fork is checked explicitly; only x/y may differ.
    for video_id in video_ids:
        rows = json.loads((stabilization_shards / f"{video_id}.json").read_text(encoding="utf-8"))
        ts0 = {int(r["frame"]): FrameCrop(int(r["frame"]), int(r["ts0"]["x"]), int(r["ts0"]["y"]), int(r["ts0"]["w"])) for r in rows}
        ts5 = {int(r["frame"]): FrameCrop(int(r["frame"]), int(r["ts5"]["x"]), int(r["ts5"]["y"]), int(r["ts5"]["w"])) for r in rows}
        ablation_invariants(ts0, ts5)

    result = {
        "schema_version": FRESH_PIPELINE_SCHEMA,
        "status": "PASS",
        "profile": profile_name,
        "scope": scope,
        "runtime_profile": runtime_profile.name,
        "attention_backend": runtime_profile.attention_backend,
        "gqa_execution_mode": runtime_profile.gqa_execution_mode,
        "video_ids": video_ids,
        "video_count": len(video_ids),
        "qwen_calls": qwen_calls,
        "rtdetr_calls": rtdetr_calls,
        "empty_video_count": int(localization_summary.get("empty_video_count", 0)),
        "empty_video_ids": list(localization_summary.get("empty_video_ids", [])),
        "requested_frame_count": int(localization_summary.get("requested_frame_count", rtdetr_calls)),
        "localized_frame_count": int(localization_summary.get("localized_frame_count", rtdetr_calls)),
        "resume_count": resume_count,
        "identity": identity,
        "retrieval_artifact_sha256": file_sha256(retrieval / "predictions.jsonl"),
        "fresh_candidate_cache_sha256": cache_summary["global_semantic_sha256"],
        "fresh_candidate_cache_is_frozen": False,
        "fresh_candidate_cache_binding": cache_binding,
        "fresh_frame_count": rtdetr_calls,
        "localization": {"status": "PASS", **localization_summary},
        "composition": {"status": "PASS", **composition_binding, "algorithm": CMP1_METHOD},
        "stabilization": {"status": "PASS", "method": TS5_METHOD, "audit_path": str(audit_path)},
        "shared_upstream": {**shared, "proof_path": str(shared_path)},
        "variants": {STABILIZED: stabilized, UNSTABILIZED_CONTROL: control},
        "algorithmic_diff": "ZERO",
        "official_score": (
            {"status": "UNAVAILABLE", "reason": "HIDDEN_GT"}
            if input_mode == "numbered_videos"
            else {"status": "NOT_REQUESTED", "reason": "DEVELOPMENT_RUN"}
        ),
        "resources": {
            "processes": list(PROCESS_RESOURCES),
            "qwen": _read_json(retrieval / "local_backend_stats.json")
            if (retrieval / "local_backend_stats.json").is_file()
            else {},
            "rtdetr": {
                "peak_allocated_mib": localization_summary.get("peak_allocated_mib"),
                "peak_reserved_mib": localization_summary.get("peak_reserved_mib"),
                "total_wall_sec": localization_summary.get("total_wall_sec"),
            },
            "total_wall_sec": time.perf_counter() - pipeline_started,
            "cpu_offload": False,
            "serialized_qwen_rtdetr": True,
            "oom": False,
            "runtime_profile": runtime_profile.name,
            "attention_backend": runtime_profile.attention_backend,
            "gqa_execution_mode": runtime_profile.gqa_execution_mode,
            **runtime_machine_identity(),
        },
    }
    _write_json(root / "inference_result.json", result)
    try:
        from aic_video_highlight.reporting import render_run_report

        result["reporting"] = render_run_report(root)
    except Exception as exc:  # reporting must never invalidate valid predictions
        result["reporting"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        _safe_print(f"[vhicraft] reporting failed: {result['reporting']['error']}")
    _write_json(root / "inference_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="inference profile JSON")
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-ids", default=None, help="comma-separated video_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=REPO_ROOT / "configs" / "runtime" / "default.json",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _read_json(config_path)
    environment = EnvironmentPaths.from_json(args.environment)
    runtime_profile_path = args.runtime_profile.resolve()
    runtime_profile = load_runtime_profile(runtime_profile_path)
    only_video_ids = (
        [value.strip() for value in args.video_ids.split(",") if value.strip()]
        if args.video_ids
        else None
    )
    default_ids = [str(value) for value in config.get("video_ids", [])]
    selected = only_video_ids or default_ids or None
    result = run_inference(
        config, environment, config_path=config_path,
        protocol_path=REPO_ROOT / str(config["protocol"]),
        resume=args.resume, validate_only=args.validate_only,
        only_video_ids=selected, output_run_id=args.run_id,
        runtime_profile=runtime_profile,
        runtime_profile_path=runtime_profile_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in ("PASS", "VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
