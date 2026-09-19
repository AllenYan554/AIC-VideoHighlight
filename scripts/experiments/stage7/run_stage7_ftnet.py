#!/usr/bin/env python3
"""Canonical Stage 7 FTNet runner (index probe, materialization, audits, training).

The stage launcher (``scripts/experiments/stage7/run.py``) invokes this script
with ``--config`` and ``--environment``; the config ``task`` selects the
operation:

    index_probe   build/probe the frozen index (AutoDL)
    materialize   retrieval -> detection -> assemble for a video scope
    small_gate    materialize a small TRAIN scope and freeze the idx0 gate
    retry_failed  re-run only the videos recorded in the failure journal
    normalize     TRAIN-only native normalization (111 TRAIN videos)
    integrity     full 154/154 materialized-dataset integrity gate
    audit         per-field native feature audit (TRAIN by default)
    train         FTNet training (smoke or formal) through scripts/train_ftnet.py

All paths are resolved against the environment JSON; no machine-specific
absolute path is written into data identities.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.integrity import (  # noqa: E402
    audit_native_features,
    load_idx0_decision,
    run_integrity_gate,
    run_train_normalization,
    write_idx0_decision,
)
from aic_video_highlight.ftnet.formal_reporting import generate_formal_deliverables  # noqa: E402
from aic_video_highlight.ftnet.pipeline import (  # noqa: E402
    MaterializationSettings,
    ensure_probed_index,
    load_raw_entries,
    retry_failed,
    run_materialization,
    select_and_probe,
)
from aic_video_highlight.ftnet.provider_core import IDX0_FALLBACK_NONE, IDX0_FALLBACKS  # noqa: E402
from aic_video_highlight.runtime.paths import EnvironmentPaths  # noqa: E402


def _resolve_spec(spec: Any, environment: EnvironmentPaths) -> Path:
    if isinstance(spec, (str, Path)):
        return Path(spec).expanduser().resolve()
    if not isinstance(spec, Mapping) or "path" not in spec:
        raise ValueError(f"invalid path spec: {spec!r}")
    base = str(spec.get("base", "repo"))
    roots = {
        "repo": environment.repo,
        "outputs": environment.outputs,
        "datasets": environment.datasets,
        "models": environment.models,
        "hf_cache": environment.hf_cache,
        "logs": environment.logs,
        "cache": environment.cache,
        "tmp": environment.tmp,
        "archive": environment.archive,
        "derived": environment.derived,
    }
    if base not in roots or roots[base] is None:
        raise ValueError(f"unsupported path base: {base!r}")
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else (roots[base] / path)


def _resolve_optional(spec: Any, environment: EnvironmentPaths) -> Path | None:
    if spec in (None, ""):
        return None
    return _resolve_spec(spec, environment)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    payload.setdefault("task", "materialize")
    payload["__config_path__"] = str(path)
    return payload


def _state_paths(config: Mapping[str, Any], environment: EnvironmentPaths) -> dict[str, Path]:
    state_root = _resolve_spec(
        config.get("state_root", {"base": "outputs", "path": "stage7_ftnet"}), environment
    )
    run_id = str(config.get("run_id", "default"))
    return {
        "state_root": state_root,
        "run_root": state_root / "runs" / run_id,
        "index_path": _resolve_spec(
            config.get("index_path", {"base": "outputs", "path": "stage7_ftnet/index/frozen_index.json"}),
            environment,
        ),
        "output_root": _resolve_spec(
            config.get(
                "output_root",
                {"base": "derived", "path": "VHiCraFTNet/youtube_highlights_ftnet"},
            ),
            environment,
        ),
    }


def _selection(config: Mapping[str, Any]):
    selection = config.get("selection", {})
    splits = selection.get("splits")
    video_ids = selection.get("video_ids")
    limit = selection.get("limit")
    if splits is not None and not isinstance(splits, list):
        raise ValueError("selection.splits must be a list")
    if video_ids is not None and not isinstance(video_ids, list):
        raise ValueError("selection.video_ids must be a list")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("selection.limit must be a positive integer")
    return splits, video_ids, limit


def _settings(config: Mapping[str, Any], environment: EnvironmentPaths) -> MaterializationSettings:
    model = config.get("model", {})
    paths = _state_paths(config, environment)
    idx0_fallback = str(config.get("idx0_fallback", IDX0_FALLBACK_NONE))
    if idx0_fallback not in IDX0_FALLBACKS:
        raise ValueError(f"idx0_fallback must be one of {IDX0_FALLBACKS}")
    return MaterializationSettings(
        dataset_root=environment.datasets,
        work_root=paths["run_root"],
        output_root=paths["output_root"],
        index_path=paths["index_path"],
        qwen_snapshot=_resolve_spec(model.get("qwen_snapshot", {"base": "models", "path": "Qwen3.5-4B"}), environment),
        rtdetr_snapshot=_resolve_spec(model.get("rtdetr_snapshot", {"base": "models", "path": "rtdetr_r50vd"}), environment),
        environment_path=Path(config["__environment_path__"]),
        split_manifest_path=_resolve_optional(config.get("split_manifest_path"), environment),
        rtdetr_model_id=str(model.get("rtdetr_model_id", "PekingU/rtdetr_r50vd")),
        device=str(config.get("device", "cuda")),
        detection_batch_size=int(config.get("detection_batch_size", 8)),
        detection_top_k=int(config.get("detection_top_k", 100)),
        decode_window=int(config.get("decode_window", 64)),
        retrieval_timeout_sec=float(config.get("retrieval_timeout_sec", 120.0)),
        vllm_base_url=str(config.get("vllm_base_url", "http://127.0.0.1:8000/v1")),
        vllm_gpu_memory_utilization=float(config.get("vllm_gpu_memory_utilization", 0.80)),
        vllm_max_model_len=int(config.get("vllm_max_model_len", 32768)),
        vllm_start_timeout_sec=float(config.get("vllm_start_timeout_sec", 900.0)),
        python=sys.executable,
        idx0_fallback=idx0_fallback,
        overwrite=bool(config.get("overwrite", False)),
        wave_size=int(config["wave_size"]) if config.get("wave_size") else None,
    )


def _stratified_ids(entries, splits, limit):
    """Deterministic one-per-category round robin for the small gate."""

    pool = sorted(
        (entry for entry in entries if entry.split in set(splits)),
        key=lambda entry: (entry.category, entry.video_id),
    )
    by_category: dict[str, list] = {}
    for entry in pool:
        by_category.setdefault(entry.category, []).append(entry)
    target = limit or len(pool)
    picked: list[str] = []
    index = 0
    while len(picked) < target:
        advanced = False
        for category in sorted(by_category):
            bucket = by_category[category]
            if index < len(bucket):
                picked.append(bucket[index].video_id)
                advanced = True
                if len(picked) >= target:
                    break
        if not advanced:
            break
        index += 1
    return picked


def _resolve_scope(config, entries, splits, video_ids, limit):
    if video_ids:
        return video_ids, None
    if config.get("selection", {}).get("stratified"):
        splits = splits or ["TRAIN"]
        return _stratified_ids(entries, splits, limit), None
    return None, limit


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def task_index_probe(config, environment, args) -> int:
    settings = _settings(config, environment)
    entries, metadata = ensure_probed_index(
        settings,
        rebuild=args.rebuild,
        verify_sha256=not args.no_verify_sha,
        ffprobe_bin=args.ffprobe_bin,
    )
    from aic_video_highlight.ftnet.index import audit_index, index_sha256

    audit = audit_index(entries)
    summary = {
        "task": "index_probe",
        "index_path": str(settings.index_path),
        "index_sha256": index_sha256(settings.index_path),
        "index_metadata": metadata,
        "audit": audit,
    }
    _print(summary)
    return 0 if audit["all_ok"] else 3


def task_materialize(config, environment, args) -> int:
    settings = _settings(config, environment)
    paths = _state_paths(config, environment)
    if config.get("require_idx0_decision"):
        decision = load_idx0_decision(paths["state_root"])
        if settings.idx0_fallback != IDX0_FALLBACK_NONE and decision["status"] != "REPLACE":
            raise SystemExit("config idx0_fallback contradicts the frozen idx0 decision")
        if decision["status"] == "REPLACE" and settings.idx0_fallback != decision["fallback"]:
            raise SystemExit(
                f"idx0 gate requires fallback {decision['fallback']}; set it in the config"
            )
    splits, video_ids, limit = _selection(config)
    raw_entries, _ = load_raw_entries(settings)
    video_ids, limit = _resolve_scope(config, raw_entries, splits, video_ids, limit)
    if args.validate_only or args.dry_run:
        selected, _ = select_and_probe(
            settings,
            splits=splits,
            video_ids=video_ids,
            limit=limit,
            verify_sha256=False,
        )
        _print(
            {
                "task": "materialize",
                "status": "VALIDATE_ONLY" if args.validate_only else "DRY_RUN",
                "index_path": str(settings.index_path),
                "output_root": str(settings.output_root),
                "work_root": str(settings.work_root),
                "selected": len(selected),
                "counts": {
                    split: sum(1 for entry in selected if entry.split == split)
                    for split in ("TRAIN", "VALIDATION", "CALIBRATION")
                },
                "idx0_fallback": settings.idx0_fallback,
                "qwen_snapshot": str(settings.qwen_snapshot),
                "rtdetr_snapshot": str(settings.rtdetr_snapshot),
            }
        )
        return 0
    summary = run_materialization(
        settings,
        splits=splits,
        video_ids=video_ids,
        limit=limit,
        rebuild_index=False,
        verify_sha256=not args.no_verify_sha,
    )
    _print(summary)
    return 0


def task_small_gate(config, environment, args) -> int:
    settings = _settings(config, environment)
    paths = _state_paths(config, environment)
    splits, video_ids, limit = _selection(config)
    splits = splits or ["TRAIN"]
    raw_entries, _ = load_raw_entries(settings)
    video_ids, limit = _resolve_scope(config, raw_entries, splits, video_ids, limit)
    if args.dry_run or args.validate_only:
        selected, _ = select_and_probe(
            settings, splits=splits, video_ids=video_ids, limit=limit, verify_sha256=False
        )
        _print(
            {
                "task": "small_gate",
                "status": "DRY_RUN" if args.dry_run else "VALIDATE_ONLY",
                "videos": [entry.video_id for entry in selected],
                "splits": {entry.video_id: entry.split for entry in selected},
            }
        )
        return 0
    summary = run_materialization(
        settings,
        splits=splits,
        video_ids=video_ids,
        limit=limit,
        verify_sha256=not args.no_verify_sha,
    )
    if summary["failed"]:
        _print({**summary, "gate": "BLOCKED_BY_FAILURES"})
        return 3
    selected_ids = list(summary.get("selected_video_ids", []))
    audit = audit_native_features(
        settings.output_root,
        split="TRAIN",
        video_ids=selected_ids,
    )
    decision_path = write_idx0_decision(paths["state_root"], audit, stage="small_gate")
    gate = audit["idx0_gate"]
    _print(
        {
            "task": "small_gate",
            "status": "PASS" if gate["status"] == "KEEP" else "PASS_WITH_FALLBACK",
            "videos": audit["videos"],
            "materialized": summary["assembled"],
            "failed": summary["failed"],
            "idx0_gate": gate,
            "decision_path": str(decision_path),
            "field_stats": audit["fields"],
        }
    )
    return 0


def task_retry_failed(config, environment, args) -> int:
    settings = _settings(config, environment)
    if args.dry_run or args.validate_only:
        _print({"task": "retry_failed", "status": "DRY_RUN"})
        return 0
    summary = retry_failed(settings)
    _print(summary)
    return 0 if summary.get("status") != "FAILED" else 3


def task_normalize(config, environment, args) -> int:
    paths = _state_paths(config, environment)
    if args.dry_run or args.validate_only:
        _print({"task": "normalize", "status": "DRY_RUN", "output_root": str(paths["output_root"])})
        return 0
    payload = run_train_normalization(paths["output_root"])
    _print({key: value for key, value in payload.items() if key != "train_videos"})
    return 0


def task_integrity(config, environment, args) -> int:
    settings = _settings(config, environment)
    entries, metadata = ensure_probed_index(settings, rebuild=False, verify_sha256=False)
    if args.dry_run or args.validate_only:
        _print({"task": "integrity", "status": "DRY_RUN", "entries": len(entries)})
        return 0
    report = run_integrity_gate(settings.output_root, entries=entries)
    summary = {
        key: report[key]
        for key in (
            "status",
            "total_materialized",
            "expected_total",
            "counts",
            "expected_counts",
            "missing",
            "extra",
            "duplicate",
            "corrupt",
            "nan_fields",
            "source_leakage",
        )
    }
    _print(summary)
    return 0 if report["status"] == "PASS" else 3


def task_audit(config, environment, args) -> int:
    paths = _state_paths(config, environment)
    split = str(config.get("split", "TRAIN"))
    audit = audit_native_features(paths["output_root"], split=split)
    _print(audit)
    return 0


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "aic_train_ftnet", REPO_ROOT / "scripts" / "train_ftnet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight_training(config, environment, settings, paths) -> dict[str, Any]:
    entries, _ = ensure_probed_index(settings, rebuild=False, verify_sha256=False)
    counts = {
        split: sum(1 for entry in entries if entry.split == split)
        for split in ("TRAIN", "VALIDATION", "CALIBRATION")
    }
    report: dict[str, Any] = {"counts": counts, "total": len(entries)}
    normalization = paths["output_root"] / "normalization" / "normalization_stats.json"
    sidecar = normalization.with_suffix(normalization.suffix + ".sha256")
    report["normalization_path"] = str(normalization)
    report["normalization_present"] = normalization.is_file()
    if not normalization.is_file():
        raise SystemExit("training preflight: normalization_stats.json is missing")
    digest = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else None
    report["normalization_sha256"] = digest
    integrity_path = paths["output_root"] / "audits" / "integrity_report.json"
    if integrity_path.is_file():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        report["integrity_status"] = integrity.get("status")
        report["integrity_total"] = integrity.get("total_materialized")
    else:
        report["integrity_status"] = "MISSING"
    if counts != {"TRAIN": 111, "VALIDATION": 23, "CALIBRATION": 20}:
        raise SystemExit(f"training preflight: unexpected split counts {counts}")
    if report.get("integrity_status") != "PASS":
        raise SystemExit("training preflight: integrity gate is not PASS")
    return report


def task_train(config, environment, args) -> int:
    paths = _state_paths(config, environment)
    settings = _settings(config, environment)
    preflight: dict[str, Any] = {}
    if config.get("preflight", False):
        preflight = _preflight_training(config, environment, settings, paths)
    run_id = str(config.get("run_id", ""))
    if config.get("timestamp_run_id"):
        run_id = f"{config.get('run_id_prefix', 'ftnet_youtubehl')}_{time.strftime('%Y%m%d%H%M')}"
    if not run_id:
        raise SystemExit("train config requires run_id or timestamp_run_id")
    train_config = _resolve_spec(
        config.get("train_config", {"base": "repo", "path": "configs/models/ftnet_reference.yaml"}),
        environment,
    )
    output_root = _resolve_spec(
        config.get("train_output_root", {"base": "outputs", "path": "stage7_ftnet/training"}),
        environment,
    )
    argv = [
        "--data-root",
        str(paths["output_root"]),
        "--config",
        str(train_config),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--device",
        str(config.get("device", "cuda")),
        "--num-workers",
        str(config.get("num_workers", 0)),
    ]
    max_epochs = args.max_epochs if getattr(args, "max_epochs", None) else config.get("max_epochs")
    max_steps = args.max_steps if getattr(args, "max_steps", None) else config.get("max_steps")
    if max_epochs:
        argv += ["--max-epochs", str(max_epochs)]
    if max_steps:
        argv += ["--max-steps", str(max_steps)]
    if args.resume:
        argv.append("--resume")
    if args.validate_only:
        argv.append("--validate-only")
    summary = {"task": "train", "run_id": run_id, "argv": argv, "preflight": preflight}
    if args.dry_run:
        _print({**summary, "status": "DRY_RUN"})
        return 0
    _print(summary)
    module = _load_train_module()
    exit_code = int(module.main(argv))
    if exit_code != 0:
        return exit_code
    if not bool(config.get("formal_reporting", False)):
        return 0
    formal_status = generate_formal_deliverables(
        output_root / run_id,
        data_root=paths["output_root"],
        repo_root=REPO_ROOT,
        index_path=paths["index_path"],
        training_config_path=train_config,
    )
    _print({"task": "formal_reporting", **formal_status})
    return 0 if formal_status["status"] == "COMPLETE" else 4


TASKS = {
    "index_probe": task_index_probe,
    "materialize": task_materialize,
    "small_gate": task_small_gate,
    "retry_failed": task_retry_failed,
    "normalize": task_normalize,
    "integrity": task_integrity,
    "audit": task_audit,
    "train": task_train,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-verify-sha", action="store_true")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    environment = EnvironmentPaths.from_json(args.environment)
    config["__environment_path__"] = str(args.environment)
    task = str(config["task"])
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; known: {', '.join(sorted(TASKS))}")
    print(
        f"[stage7] task={task} platform={platform.system()} python={sys.executable}",
        flush=True,
    )
    return TASKS[task](config, environment, args)


if __name__ == "__main__":
    raise SystemExit(main())
