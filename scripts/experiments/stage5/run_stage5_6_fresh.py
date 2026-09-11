#!/usr/bin/env python3
"""True-fresh VHiCraft-v1 orchestrator for Stage 5.6.

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

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.highlight_retrieval.candidate_cache import (
    export_candidate_cache,
    replay_cache,
    validate_cache,
)
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputs,
    compose_all_frames,
    engineering_gate,
    load_policy_shard_dir,
    policy_shard_semantic_sha,
)
from aic_video_highlight.spatial_composition.fresh_pipeline import (
    FRESH_SHARD_SCHEMA_VERSION,
    FreshPipelineError,
    audit_fresh_shard,
    build_fresh_index,
    build_shard_records,
    build_ts5_by_frame,
    shared_upstream_identity,
    validate_fresh_candidate_cache_binding,
)
from aic_video_highlight.spatial_composition.vhicraft_pipeline import (
    TARGET_RATIO,
    VC1,
    VCA0,
    FrameCrop,
    ablation_invariants,
    assemble_prediction_lines,
    fs0_identity_holds,
    validate_prediction_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRESH_PIPELINE_SCHEMA = "aic.vhicraft.true-fresh-pipeline/v1"
RUN_IDENTITY_SCHEMA = "aic.vhicraft.fresh-run-identity/v1"
STAGE4_FINAL_ROUTE = "frozen_retrieval_v0_no_selection_no_br1_no_dtl1"
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
    }
    if base not in roots:
        raise FreshPipelineError(f"unsupported fresh input base: {base}")
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else roots[base] / path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def build_run_identity(
    *, config_path: Path, protocol_path: Path, selected_manifest: Path, mode: str
) -> dict[str, Any]:
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "git_head": _git_head(),
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(protocol_path),
        "model": {"repo_id": QWEN_MODEL, "revision": QWEN_REVISION},
        "prompt": QWEN_PROMPT,
        "input_manifest_sha256": file_sha256(selected_manifest),
        "mode": mode,
    }
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

    def emit(self, stage: str, done: int, *, unit: str = "videos", detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.total_videos - done) / done if done else None
        eta_text = "?" if eta is None else f"{eta:.0f}s"
        print(
            f"[fresh] stage={stage} {unit}={done}/{self.total_videos} "
            f"elapsed={elapsed:.0f}s eta={eta_text} {detail}".rstrip(),
            flush=True,
        )


def select_dev_manifest(
    source_path: Path, role_records: Sequence[Mapping[str, Any]], output_path: Path,
    *, videos: int | None,
) -> list[str]:
    source = {str(row["video_id"]): row for row in _read_jsonl(source_path)}
    ordered = [str(row["video_id"]) for row in role_records]
    if videos is not None:
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
    fresh = config["fresh_pipeline"]
    base_url = str(fresh.get("vllm_base_url", "http://127.0.0.1:8000/v1"))
    if _service_has_model(base_url):
        return None, base_url
    python = str(fresh.get("python", sys.executable))
    model_path = _resolve(fresh["qwen_snapshot"], environment)
    command = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", QWEN_MODEL,
        "--host", "127.0.0.1", "--port", "8000",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(fresh.get("vllm_gpu_memory_utilization", 0.80)),
        "--max-model-len", str(fresh.get("vllm_max_model_len", 32768)),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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


def _run(command: Sequence[str], *, label: str) -> None:
    print(f"[fresh] {label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(list(command), cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise FreshPipelineError(f"{label} failed with exit code {completed.returncode}")


def _stage_marker(path: Path, *, stage: str, inputs: Mapping[str, str], outputs: Mapping[str, str]) -> None:
    _write_json(path, {
        "schema_version": FRESH_PIPELINE_SCHEMA,
        "stage": stage,
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
    })


def _marker_matches(path: Path, *, inputs: Mapping[str, str], outputs: Sequence[Path]) -> bool:
    if not path.is_file() or not all(item.exists() for item in outputs):
        return False
    marker = _read_json(path)
    if marker.get("inputs") != dict(sorted(inputs.items())):
        raise FreshPipelineError(f"resume stage input identity mismatch: {path}")
    recorded = marker.get("outputs", {})
    for output in outputs:
        if output.is_file() and recorded.get(output.name) != file_sha256(output):
            raise FreshPipelineError(f"resume stage output identity mismatch: {output}")
    return True


def _count_qwen_calls(raw_dir: Path) -> int:
    return sum(len(_read_json(path).get("chunks", [])) for path in sorted(raw_dir.glob("*.json")))


def _build_fresh_inputs(
    predictions_path: Path, metadata_path: Path, index_path: Path, stage5_2_dir: Path,
) -> tuple[FrozenInputs, dict[str, Any]]:
    predictions = {row["video_id"]: row for row in _read_jsonl(predictions_path)}
    metadata = _read_json(metadata_path)["records"]
    index = {row["video_id"]: row for row in _read_jsonl(index_path)}
    policy = tuple(load_policy_shard_dir(stage5_2_dir / "policy_v1"))
    policy_hash = policy_shard_semantic_sha(policy)
    expected = {
        (video_id, int(item["frame"]))
        for video_id, row in predictions.items() for item in row["predictions"]
    }
    actual = {(row["video_id"], int(row["frame"])) for row in policy}
    if expected != actual or len(actual) != len(policy):
        raise FreshPipelineError(
            f"Stage 5.3 fresh binding mismatch: expected={len(expected)} actual={len(actual)}"
        )
    inputs = FrozenInputs(
        stage5_1_predictions=predictions,
        metadata=metadata,
        index=index,
        policy_records=policy,
        raw_frames={},
        weak_reference={},
        input_hashes={
            "stage5_1_predictions": file_sha256(predictions_path),
            "video_metadata_cache": file_sha256(metadata_path),
            "dev166_index": file_sha256(index_path),
            "stage5_2_policy_artifact": policy_hash,
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
        "schema_version": "aic.vhicraft.fresh-stage5_3-manifest/v1",
        "video_count": len(videos),
        "frame_count": sum(item["frame_count"] for item in videos),
        "videos": videos,
        "input_identities": inputs.input_hashes,
        "algorithm": CMP1_METHOD,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _assemble_fresh_arm(
    *, arm: str, index_rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
    shards_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    key = "ts5" if arm == VC1 else "ts0"
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
        "arm": arm,
        "predictions_path": str(predictions),
        "video_count": len(index_rows),
        "frame_count": fs0["frame_count_after"],
        "contract": {"is_valid": report.is_valid, "stats": dict(report.stats)},
        "errors": [],
        "fs0": {"identity": True, **fs0},
    }


def run_fresh_pipeline(
    config: Mapping[str, Any], environment: EnvironmentPaths, *, config_path: Path,
    protocol_path: Path, videos: int | None, resume: bool, validate_only: bool = False,
) -> dict[str, Any]:
    fresh = config.get("fresh_pipeline")
    if not isinstance(fresh, Mapping):
        raise FreshPipelineError("execution config lacks fresh_pipeline")
    role_path = _resolve(config["inputs"]["role_manifest"], environment)
    source_manifest = _resolve(fresh["dev_manifest"], environment)
    source_index = _resolve(fresh["source_index"], environment)
    video_root = _resolve(fresh["video_root"], environment)
    if validate_only:
        required = [role_path, source_manifest, source_index, video_root]
        missing = [str(path) for path in required if not path.exists()]
        return {
            "status": "VALIDATED" if not missing else "MISSING_ENVIRONMENT_INPUTS",
            "missing": missing,
            "scientific_identities": {
                "qwen": {"repo_id": QWEN_MODEL, "revision": QWEN_REVISION, "prompt": QWEN_PROMPT},
                "stage4": STAGE4_FINAL_ROUTE,
                "rtdetr": {"model": RTDETR_MODEL, "revision": RTDETR_REVISION, "policy": RTDETR_POLICY},
                "cmp1": CMP1_METHOD,
                "ts5": TS5_METHOD,
                "fs0": FS0_METHOD,
            },
        }
    role = _read_json(role_path)
    root = environment.outputs / "stage5" / str(config["output_run_id"]) / "fresh_pipeline"
    selected_manifest = root / "input" / "dev_selected.jsonl"
    video_ids = select_dev_manifest(source_manifest, role["records"], selected_manifest, videos=videos)
    mode = "SMOKE" if videos is not None else "FORMAL"
    identity = build_run_identity(
        config_path=config_path, protocol_path=protocol_path,
        selected_manifest=selected_manifest, mode=mode,
    )
    resume_count = bind_resume(root, identity, resume=resume)
    progress = Progress(len(video_ids))
    python = str(fresh.get("python", sys.executable))

    # Stage 1: real Qwen calls.  The selected Dev manifest is the exact input
    # consumed by the cache stage below; it cannot silently fall back to frozen.
    stage1 = root / "stage1"
    stage1_marker = stage1 / "fresh_identity.json"
    stage1_inputs = {"selected_manifest": file_sha256(selected_manifest), "run_identity": identity["identity_sha256"]}
    if not _marker_matches(stage1_marker, inputs=stage1_inputs, outputs=[stage1 / "predictions.jsonl", stage1 / "run_config.json"]):
        owned_vllm: subprocess.Popen[Any] | None = None
        try:
            owned_vllm, base_url = start_vllm(config, environment, root / "logs" / "vllm.log")
            command = [
                python, "scripts/experiments/stage1/run_baseline.py",
                "--manifest", str(selected_manifest), "--video-root", str(video_root),
                "--output-dir", str(stage1), "--config", str(REPO_ROOT / "configs/highlight_retrieval.yaml"),
                "--dataset-name", "aic_highlight_dev", "--dataset-version", "aic_highlight_dev_v1.1",
                "--model-revision", QWEN_REVISION, "--base-url", base_url,
            ]
            if resume:
                command.append("--resume")
            _run(command, label="Stage1 Qwen fresh inference")
        finally:
            stop_vllm(owned_vllm)
        _stage_marker(stage1_marker, stage="stage1_qwen", inputs=stage1_inputs, outputs={
            "predictions.jsonl": file_sha256(stage1 / "predictions.jsonl"),
            "run_config.json": file_sha256(stage1 / "run_config.json"),
        })
    else:
        resume_count += 1
    qwen_calls = _count_qwen_calls(stage1 / "raw")
    if qwen_calls <= 0:
        raise FreshPipelineError("fresh Stage 1 contains zero Qwen calls")
    progress.emit("stage1", len(video_ids), detail=f"qwen_calls={qwen_calls}")

    # Stage 4 final route: canonical cache builder + frozen no-selection route.
    stage4 = root / "stage4"
    cache = stage4 / "candidate_cache"
    final_segments = stage4 / "final_segments.jsonl"
    stage4_marker = stage4 / "fresh_identity.json"
    stage4_inputs = {"stage1_predictions": file_sha256(stage1 / "predictions.jsonl"), "stage1_run_config": file_sha256(stage1 / "run_config.json")}
    if not _marker_matches(stage4_marker, inputs=stage4_inputs, outputs=[cache / "cache_manifest.json", final_segments]):
        if cache.exists() and any(cache.iterdir()):
            raise FreshPipelineError("unbound fresh candidate cache exists; refusing overwrite")
        manifest = export_candidate_cache({"dev": stage1}, output_dir=cache)
        if manifest["global_semantic_sha256"] == FROZEN_CACHE_SHA256:
            raise FreshPipelineError("VC-1 unexpectedly resolved to the historical frozen candidate cache")
        replay_cache(cache, final_segments)
        _stage_marker(stage4_marker, stage="stage4_final_route", inputs=stage4_inputs, outputs={
            "cache_manifest.json": file_sha256(cache / "cache_manifest.json"),
            "final_segments.jsonl": file_sha256(final_segments),
        })
    else:
        resume_count += 1
    cache_summary = validate_cache(cache)
    cache_binding = validate_fresh_candidate_cache_binding(
        _read_json(cache / "cache_manifest.json"),
        stage1_predictions_path=stage1 / "predictions.jsonl",
        expected_video_ids=video_ids,
        forbidden_global_sha256=FROZEN_CACHE_SHA256,
    )
    if cache_summary["record_count"] != len(video_ids):
        raise FreshPipelineError("fresh candidate cache count gate failed")
    progress.emit("stage4", len(video_ids), detail=f"cache={cache_summary['global_semantic_sha256'][:12]}")

    # Stage 5.1 canonical frame projection.
    stage51 = root / "stage5_1"
    index_path = stage51 / "fresh_index.jsonl"
    index_rows = build_fresh_index(role["records"], _read_jsonl(source_index), video_ids=video_ids)
    _write_jsonl(index_path, index_rows)
    predictions51 = stage51 / "predictions.jsonl"
    metadata51 = stage51 / "metadata_cache.json"
    stage51_marker = stage51 / "fresh_identity.json"
    stage51_inputs = {"final_segments": file_sha256(final_segments), "fresh_index": file_sha256(index_path)}
    if not _marker_matches(stage51_marker, inputs=stage51_inputs, outputs=[predictions51, metadata51]):
        _run([
            python, "scripts/experiments/stage5/run_stage5_center_crop_baseline.py",
            "--index", str(index_path), "--video-dir", str(video_root),
            "--segments", str(final_segments), "--out", str(predictions51),
            "--metadata-cache", str(metadata51), "--pts-extract",
        ], label="Stage5.1 fresh frame projection")
        _stage_marker(stage51_marker, stage="stage5_1_frame_projection", inputs=stage51_inputs, outputs={
            "predictions.jsonl": file_sha256(predictions51), "metadata_cache.json": file_sha256(metadata51),
        })
    else:
        resume_count += 1
    progress.emit("stage5.1", len(video_ids))

    # Stage 5.2 canonical RT-DETR + primary_subject_policy_v1 on fresh frames.
    stage52 = root / "stage5_2"
    stage52_marker = stage52 / "fresh_identity.json"
    stage52_inputs = {"stage5_1_predictions": file_sha256(predictions51), "fresh_index": file_sha256(index_path), "metadata": file_sha256(metadata51)}
    summary52 = stage52 / "full_dev_summary.json"
    if not _marker_matches(stage52_marker, inputs=stage52_inputs, outputs=[summary52]):
        rtdetr_snapshot = _resolve(fresh["rtdetr_snapshot"], environment)
        _run([
            python, "scripts/experiments/stage5/run_stage5_2_full_dev.py",
            "--index", str(index_path), "--video-base", str(video_root),
            "--predictions", str(predictions51), "--metadata-cache", str(metadata51),
            "--output-dir", str(stage52), "--model-id", RTDETR_MODEL,
            "--local-path", str(rtdetr_snapshot), "--device", "cuda", "--dtype", "float32",
            "--mode", "FRESH_PIPELINE", "--expected-videos", str(len(video_ids)),
        ], label="Stage5.2 fresh RT-DETR")
        _stage_marker(stage52_marker, stage="stage5_2_rtdetr", inputs=stage52_inputs, outputs={
            "full_dev_summary.json": file_sha256(summary52),
        })
    else:
        resume_count += 1
    stage52_summary = _read_json(summary52)
    if any(int(stage52_summary.get(key, -1)) != 0 for key in ("missing", "extra", "duplicates")):
        raise FreshPipelineError("Stage 5.2 fresh frame identity gate failed")
    rtdetr_calls = int(stage52_summary["frames"])
    if rtdetr_calls <= 0:
        raise FreshPipelineError("fresh Stage 5.2 contains zero RT-DETR calls")
    progress.emit("stage5.2", len(video_ids), unit="videos", detail=f"rtdetr_frames={rtdetr_calls}")

    # Stage 5.3: fresh identity binding; canonical CMP-1 implementation.
    stage53 = root / "stage5_3"
    stage53.mkdir(parents=True, exist_ok=True)
    stage53_marker = stage53 / "fresh_identity.json"
    inputs, binding53 = _build_fresh_inputs(predictions51, metadata51, index_path, stage52)
    manifest53 = _composition_manifest(inputs)
    manifest53_path = stage53 / "manifest.json"
    _write_json(manifest53_path, manifest53)
    records53_path = stage53 / "cmp1_records.json"
    stage53_inputs = {
        "stage5_1_predictions": file_sha256(predictions51),
        "stage5_2_policy": binding53["policy_sha256"],
        "manifest": file_sha256(manifest53_path),
        "algorithm": CMP1_METHOD,
    }
    if not _marker_matches(stage53_marker, inputs=stage53_inputs, outputs=[records53_path]):
        records53 = compose_all_frames(manifest53, inputs, TARGET_RATIO)
        gate53 = engineering_gate(manifest53, records53)
        if any(int(gate53[key]) for key in ("missing", "extra", "duplicates", "geometry_failures", "cmp0_frozen_regression")):
            raise FreshPipelineError(f"Stage 5.3 fresh CMP-1 gate failed: {gate53}")
        _write_json(records53_path, {"algorithm": CMP1_METHOD, "records": records53, "gate": gate53})
        _stage_marker(stage53_marker, stage="stage5_3_cmp1", inputs=stage53_inputs, outputs={
            "cmp1_records.json": file_sha256(records53_path),
        })
    else:
        resume_count += 1
    records53 = _read_json(records53_path)["records"]
    progress.emit("stage5.3", len(video_ids), detail=f"frames={len(records53)}")

    # Stage 5.4: canonical TS-5 Revised.  TS-0 is the unmodified CMP-1 fork.
    stage54 = root / "stage5_4"
    shards54 = stage54 / "shards"
    shards54.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in video_ids}
    for row in records53:
        by_video[str(row["video_id"])].append(row)
    audits = []
    for index, video_id in enumerate(video_ids, 1):
        shard_path = shards54 / f"{video_id}.json"
        marker_path = shards54 / f"{video_id}.identity.json"
        shard_inputs = {"stage5_3": file_sha256(records53_path), "video_id": video_id, "algorithm": TS5_METHOD}
        if not _marker_matches(marker_path, inputs=shard_inputs, outputs=[shard_path]):
            meta = inputs.metadata[video_id]
            ts5 = build_ts5_by_frame(
                by_video[video_id], width=int(meta["width"]), height=int(meta["height"]),
                target_ratio=TARGET_RATIO, alpha=0.5,
            )
            shard = build_shard_records(by_video[video_id], ts5)
            _atomic_text(shard_path, json.dumps(shard, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            _stage_marker(marker_path, stage="stage5_4_ts5_revised", inputs=shard_inputs, outputs={shard_path.name: file_sha256(shard_path)})
        else:
            resume_count += 1
        shard_rows = json.loads(shard_path.read_text(encoding="utf-8"))
        audits.append(asdict(audit_fresh_shard(video_id, shard_rows)))
        progress.emit("stage5.4", index)
    audit_path = stage54 / "audit.json"
    _write_json(audit_path, {"schema": FRESH_SHARD_SCHEMA_VERSION, "method": TS5_METHOD, "videos": audits})

    upstream_paths = {
        "stage1_predictions": stage1 / "predictions.jsonl",
        "fresh_candidate_cache_manifest": cache / "cache_manifest.json",
        "stage4_final_segments": final_segments,
        "stage5_1_predictions": predictions51,
        "stage5_2_summary": summary52,
        "stage5_3_records": records53_path,
    }
    shared = shared_upstream_identity(upstream_paths)
    shared_path = root / "shared_upstream_identity.json"
    _write_json(shared_path, {**shared, "arms": [VC1, VCA0], "fork_stage": "stage5_4", "only_difference": "TS-5 Revised vs TS-0"})

    metadata = _read_json(metadata51)["records"]
    vc1 = _assemble_fresh_arm(arm=VC1, index_rows=index_rows, metadata=metadata, shards_dir=shards54, output_dir=root / "vc_1")
    vca0 = _assemble_fresh_arm(arm=VCA0, index_rows=index_rows, metadata=metadata, shards_dir=shards54, output_dir=root / "vc_a0")
    if not vc1["contract"]["is_valid"] or not vca0["contract"]["is_valid"]:
        raise FreshPipelineError("fresh official contract validation failed")
    # The fork is checked explicitly; only x/y may differ.
    for video_id in video_ids:
        rows = json.loads((shards54 / f"{video_id}.json").read_text(encoding="utf-8"))
        ts0 = {int(r["frame"]): FrameCrop(int(r["frame"]), int(r["ts0"]["x"]), int(r["ts0"]["y"]), int(r["ts0"]["w"])) for r in rows}
        ts5 = {int(r["frame"]): FrameCrop(int(r["frame"]), int(r["ts5"]["x"]), int(r["ts5"]["y"]), int(r["ts5"]["w"])) for r in rows}
        ablation_invariants(ts0, ts5)

    result = {
        "schema_version": FRESH_PIPELINE_SCHEMA,
        "status": "PASS",
        "mode": mode,
        "video_ids": video_ids,
        "video_count": len(video_ids),
        "qwen_calls": qwen_calls,
        "rtdetr_calls": rtdetr_calls,
        "resume_count": resume_count,
        "identity": identity,
        "stage1_artifact_sha256": file_sha256(stage1 / "predictions.jsonl"),
        "fresh_candidate_cache_sha256": cache_summary["global_semantic_sha256"],
        "fresh_candidate_cache_is_frozen": False,
        "fresh_candidate_cache_binding": cache_binding,
        "fresh_frame_count": rtdetr_calls,
        "stage5_2": {"status": "PASS", **stage52_summary},
        "stage5_3": {"status": "PASS", **binding53, "algorithm": CMP1_METHOD},
        "stage5_4": {"status": "PASS", "method": TS5_METHOD, "audit_path": str(audit_path)},
        "shared_upstream": {**shared, "proof_path": str(shared_path)},
        "arms": {VC1: vc1, VCA0: vca0},
        "algorithmic_diff": "ZERO",
    }
    _write_json(root / "fresh_pipeline_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--videos", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _read_json(config_path)
    environment = EnvironmentPaths.from_json(args.environment)
    videos = args.videos
    if args.smoke and videos is None:
        videos = int(config.get("smoke", {}).get("max_videos", 3))
    result = run_fresh_pipeline(
        config, environment, config_path=config_path,
        protocol_path=REPO_ROOT / str(config["protocol"]), videos=videos,
        resume=args.resume, validate_only=args.validate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
