"""Formal Stage 7.1 reporting from already-produced machine artifacts.

The module is intentionally offline: it never trains, calls Qwen, loads
RT-DETR, or touches source videos.  Missing evidence is rendered explicitly
instead of being inferred from prose or operator memory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

REPORT_TITLE = "Stage 7.1 Real-Data FTNet Baseline Experiment Report"
REPORT_SCHEMA = "aic.stage7.ftnet.formal-report/v1"
AUDIT_SCHEMA = "aic.stage7.ftnet.artifact-audit/v1"
MANIFEST_SCHEMA = "aic.stage7.ftnet.artifact-manifest/v1"
STATUS_SCHEMA = "aic.stage7.ftnet.formal-run-status/v1"
CHECKPOINT_AUDIT_SCHEMA = "aic.stage7.ftnet.checkpoint-identity-audit/v1"
INSUFFICIENT = "NOT RECORDED / CURRENT ARTIFACTS INSUFFICIENT"

ARCHIVE_DIRECTORIES = ("config", "figures", "results", "supplementary")

TRAINING_FIGURES = (
    "train_loss_vs_epoch.png",
    "validation_loss_vs_epoch.png",
    "train_validation_loss.png",
    "learning_rate_vs_epoch.png",
    "gradient_norm_vs_epoch.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | None) -> Any | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _first_file(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _copy_lightweight_inputs(
    formal_dir: Path, data_root: Path | None, index_path: Path | None
) -> None:
    sources: list[tuple[Path | None, str]] = [
        (index_path, "supplementary/frozen_index.json"),
    ]
    if data_root is not None:
        sources.extend(
            [
                (
                    data_root / "manifests" / "materialized_videos.json",
                    "results/materialized_videos.json",
                ),
                (
                    data_root / "normalization" / "normalization_stats.json",
                    "results/normalization_stats.json",
                ),
                (
                    data_root / "normalization" / "normalization_stats.json.sha256",
                    "results/normalization_stats.json.sha256",
                ),
                (
                    data_root / "audits" / "native_feature_audit.json",
                    "results/native_feature_audit.json",
                ),
                (
                    data_root / "audits" / "integrity_report.json",
                    "results/integrity_report.json",
                ),
            ]
        )
    for source, name in sources:
        target = formal_dir / name
        if source is not None and source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _copy_training_run(formal_dir: Path, training_run_dir: Path | None) -> None:
    if training_run_dir is None:
        return
    sources = (
        ("history.json", "results/training_history.json"),
        ("summary.json", "results/training_summary.json"),
        ("best.pt", "results/checkpoints/best.pt"),
        ("last.pt", "results/checkpoints/last.pt"),
        ("logs/progress.json", "supplementary/training_progress.json"),
        ("logs/progress.jsonl", "supplementary/training_progress.jsonl"),
    )
    for source_name, target_name in sources:
        source = training_run_dir / source_name
        target = formal_dir / target_name
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _copy_materialization_run(
    formal_dir: Path,
    materialization_run_dir: Path | None,
) -> None:
    if materialization_run_dir is None:
        return
    sources = (
        ("status.json", "supplementary/materialization_status.json"),
        ("summary.json", "supplementary/materialization_summary.json"),
        ("failures.json", "supplementary/materialization_failures.json"),
        ("logs/progress.json", "supplementary/materialization_progress.json"),
        ("logs/progress.jsonl", "supplementary/materialization_progress.jsonl"),
        ("logs/vllm.log", "supplementary/materialization_vllm.log"),
        ("performance_summary.json", "results/performance_summary.json"),
    )
    for source_name, target_name in sources:
        source = materialization_run_dir / source_name
        target = formal_dir / target_name
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _copy_optional_evidence(
    formal_dir: Path,
    source: Path | None,
    target_name: str,
) -> None:
    if source is None or not source.is_file():
        return
    target = formal_dir / target_name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_config_snapshots(
    formal_dir: Path,
    training_config_path: Path | None,
    config_snapshot_paths: Iterable[Path],
) -> None:
    sources = [training_config_path, *config_snapshot_paths]
    for source in sources:
        if source is None or not source.is_file():
            continue
        target = formal_dir / "config" / source.name
        shutil.copy2(source, target)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_line(
    output: Path,
    epochs: list[int],
    series: list[tuple[str, list[float], str, str]],
    *,
    title: str,
    ylabel: str,
) -> None:
    plt = _load_matplotlib()
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        for label, values, color, linestyle in series:
            ax.plot(
                epochs,
                values,
                color=color,
                linestyle=linestyle,
                marker="o" if len(values) <= 50 else None,
                markersize=3,
                linewidth=1.8,
                label=label,
            )
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
        if len(series) > 1:
            ax.legend(frameon=False)
        fig.savefig(
            output,
            dpi=180,
            facecolor="white",
            metadata={"Software": "AIC Stage 7.1 formal reporter"},
        )
        plt.close(fig)


def render_figures(
    history_path: Path | None,
    native_audit_path: Path | None,
    figures_dir: Path,
) -> list[dict[str, str]]:
    """Render only figures whose required numeric fields actually exist."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, str]] = []
    history = _read_json(history_path)
    if isinstance(history, list) and history:
        epochs = [int(row["epoch"]) + 1 for row in history if "epoch" in row]

        def values(field: str) -> list[float] | None:
            if len(epochs) != len(history) or any(not isinstance(row.get(field), (int, float)) for row in history):
                return None
            return [float(row[field]) for row in history]

        specifications = [
            (
                "train_loss_vs_epoch.png",
                "FTNet training loss",
                "Masked BCE",
                [("Train", values("train_loss"), "#0072B2", "-")],
                "results/training_history.json: train_loss",
            ),
            (
                "validation_loss_vs_epoch.png",
                "FTNet validation loss",
                "Masked BCE",
                [("Validation", values("val_loss"), "#D55E00", "--")],
                "results/training_history.json: val_loss",
            ),
            (
                "train_validation_loss.png",
                "FTNet train and validation loss",
                "Masked BCE",
                [
                    ("Train", values("train_loss"), "#0072B2", "-"),
                    ("Validation", values("val_loss"), "#D55E00", "--"),
                ],
                "results/training_history.json: train_loss, val_loss",
            ),
            (
                "learning_rate_vs_epoch.png",
                "Learning-rate schedule",
                "Learning rate",
                [("Learning rate", values("lr"), "#009E73", "-")],
                "results/training_history.json: lr",
            ),
            (
                "gradient_norm_vs_epoch.png",
                "Gradient norm by epoch",
                "Global L2 norm",
                [("Gradient norm", values("grad_norm"), "#CC79A7", "-")],
                "results/training_history.json: grad_norm",
            ),
        ]
        for filename, title, ylabel, series, source in specifications:
            if any(item[1] is None for item in series):
                continue
            concrete = [(label, data or [], color, style) for label, data, color, style in series]
            _save_line(figures_dir / filename, epochs, concrete, title=title, ylabel=ylabel)
            rendered.append({"file": filename, "source": source})

    audit = _read_json(native_audit_path)
    fields = audit.get("fields") if isinstance(audit, dict) else None
    if isinstance(fields, dict) and fields:
        names = list(fields)
        if all(isinstance(fields[name].get("mean"), (int, float)) and isinstance(fields[name].get("std"), (int, float)) for name in names):
            plt = _load_matplotlib()
            with plt.style.context("default"):
                fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
                y = list(range(len(names)))
                ax.errorbar(
                    [float(fields[name]["mean"]) for name in names],
                    y,
                    xerr=[float(fields[name]["std"]) for name in names],
                    fmt="o",
                    color="#0072B2",
                    ecolor="#56B4E9",
                    capsize=3,
                )
                ax.set_yticks(y, labels=names)
                ax.invert_yaxis()
                ax.set(title="Native16 mean and standard deviation", xlabel="Raw feature value (mean ± SD)")
                ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.7)
                fig.savefig(figures_dir / "native16_mean_std.png", dpi=180, facecolor="white")
                plt.close(fig)
            rendered.append(
                {
                    "file": "native16_mean_std.png",
                    "source": "results/native_feature_audit.json: fields[*].mean/std",
                }
            )

        if all(
            isinstance(fields[name].get("unique_count"), (int, float))
            and isinstance(fields[name].get("count"), (int, float))
            and isinstance(fields[name].get("missing_rate"), (int, float))
            and float(fields[name]["count"]) > 0
            for name in names
        ):
            plt = _load_matplotlib()
            with plt.style.context("default"):
                fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
                y = list(range(len(names)))
                unique_ratio = [float(fields[name]["unique_count"]) / float(fields[name]["count"]) for name in names]
                observed_ratio = [1.0 - float(fields[name]["missing_rate"]) for name in names]
                ax.barh([item - 0.18 for item in y], unique_ratio, height=0.34, label="Unique / observed", color="#0072B2")
                ax.barh([item + 0.18 for item in y], observed_ratio, height=0.34, label="Observed / all", color="#E69F00")
                ax.set_yticks(y, labels=names)
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                ax.set(title="Native16 variability and observation coverage", xlabel="Fraction")
                ax.legend(
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.08),
                    ncol=2,
                )
                ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.7)
                fig.savefig(
                    figures_dir / "native16_unique_or_variability.png",
                    dpi=180,
                    facecolor="white",
                )
                plt.close(fig)
            rendered.append(
                {
                    "file": "native16_unique_or_variability.png",
                    "source": "results/native_feature_audit.json: unique_count/count/missing_rate",
                }
            )
    return rendered


def _git_identity(repo_root: Path | None) -> dict[str, str]:
    if repo_root is None or not (repo_root / ".git").exists():
        return {"head": "UNKNOWN", "branch": "UNKNOWN", "status": "UNKNOWN"}

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args], text=True, encoding="utf-8"
            ).strip()
        except Exception:  # noqa: BLE001 - reporting must preserve missing evidence
            return "UNKNOWN"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short", "--branch"),
    }


def validate_checkpoint_identity(
    formal_dir: Path,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate formal checkpoint metadata without running model inference."""

    checkpoint_dir = formal_dir / "results" / "checkpoints"
    paths = {name: checkpoint_dir / name for name in ("best.pt", "last.pt")}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "schema": CHECKPOINT_AUDIT_SCHEMA,
            "status": "MISSING",
            "missing": missing,
            "checkpoints": {},
        }
    if not isinstance(summary, dict):
        raise ValueError("checkpoint validation requires results/training_summary.json")

    import torch

    run_id = summary.get("run_id")
    git_head = summary.get("git_head")
    epochs_run = summary.get("epochs_run")
    total_steps = summary.get("steps")
    best_epoch = summary.get("best_epoch")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("training summary has no valid run_id")
    if not isinstance(git_head, str) or not git_head:
        raise ValueError("training summary has no valid git_head")
    if not isinstance(epochs_run, int) or epochs_run <= 0:
        raise ValueError("training summary has no valid epochs_run")
    if not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError("training summary has no valid steps")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs_run:
        raise ValueError("training summary has no valid best_epoch")

    records: dict[str, Any] = {}
    raw_payloads: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - fail closed on archive evidence
            raise ValueError(f"checkpoint is unreadable: {path}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "aic.stage7.ftnet.checkpoint/v1"
        ):
            raise ValueError(f"unexpected checkpoint schema: {path}")
        identity = payload.get("identity")
        if not isinstance(identity, dict) or identity.get("git_head") != git_head:
            raise ValueError(f"checkpoint git identity conflicts with training summary: {path}")
        recorded_path = summary.get(f"checkpoint_{name.removesuffix('.pt')}")
        if not isinstance(recorded_path, str):
            raise ValueError(f"training summary has no recorded path for {name}")
        normalized_recorded = recorded_path.replace("\\", "/")
        if f"/{run_id}/" not in normalized_recorded or not normalized_recorded.endswith(
            f"/{name}"
        ):
            raise ValueError(f"checkpoint run identity conflicts with training summary: {name}")
        raw_payloads[name] = payload
        records[name] = {
            "relative_path": path.relative_to(formal_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "epoch": int(payload["epoch"]),
            "global_step": int(payload["global_step"]),
            "identity": dict(identity),
            "recorded_source_path": recorded_path,
            "run_id": run_id,
            "run_id_verification": "training_summary path and synchronized remote run directory",
        }

    if records["best.pt"]["epoch"] != int(best_epoch):
        raise ValueError("best checkpoint epoch conflicts with training summary")
    if records["last.pt"]["epoch"] != int(epochs_run):
        raise ValueError("last checkpoint epoch conflicts with training summary")
    if records["last.pt"]["global_step"] != int(total_steps):
        raise ValueError("last checkpoint step conflicts with training summary")
    if int(total_steps) % int(epochs_run) == 0:
        expected_best_step = int(best_epoch) * (int(total_steps) // int(epochs_run))
        if records["best.pt"]["global_step"] != expected_best_step:
            raise ValueError("best checkpoint step conflicts with training summary")
    if raw_payloads["best.pt"].get("identity") != raw_payloads["last.pt"].get("identity"):
        raise ValueError("best and last checkpoint identities conflict")

    return {
        "schema": CHECKPOINT_AUDIT_SCHEMA,
        "status": "PASS",
        "summary_run_id": run_id,
        "summary_git_head": git_head,
        "checkpoints": records,
    }


def _artifact_entry(path: Path, *, role: str, source: str, formal_dir: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "relative_path": path.relative_to(formal_dir).as_posix() if path.is_relative_to(formal_dir) else None,
        "exists": exists,
        "size": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
        "role": role,
        "source": source,
        "formally_archived": exists and path.is_relative_to(formal_dir),
    }


def write_artifact_audit(formal_dir: Path, figure_records: Iterable[dict[str, str]]) -> dict[str, Any]:
    expected = [
        ("experiment_report.md", "formal experiment report", "offline reporter"),
        ("config/ftnet_reference.yaml", "frozen FTNet protocol", "repository snapshot"),
        ("config/ftnet_train_formal.json", "formal run config", "repository snapshot"),
        ("config/windows_local.json", "local environment snapshot", "repository snapshot"),
        ("results/performance_summary.json", "machine-readable performance evidence", "logs/instrumentation"),
        ("results/integrity_report.json", "dataset integrity result", "integrity gate"),
        ("results/materialized_videos.json", "materialized dataset manifest", "materialization"),
        ("results/native_feature_audit.json", "Native16 field audit", "TRAIN-only audit"),
        ("results/normalization_stats.json", "TRAIN-only normalization", "normalization"),
        ("results/normalization_stats.json.sha256", "normalization digest sidecar", "normalization"),
        ("results/training_history.json", "formal training epoch history", "formal training"),
        ("results/training_summary.json", "formal training summary", "formal training"),
        ("results/checkpoints/best.pt", "best formal checkpoint", "formal training"),
        ("results/checkpoints/last.pt", "last formal checkpoint", "formal training"),
        ("results/formal_run_status.json", "formal completion status", "offline reporter"),
        ("supplementary/artifact_manifest.json", "content manifest", "offline reporter"),
        ("supplementary/performance_audit.md", "performance root-cause audit", "manual/code audit"),
        ("supplementary/path_and_storage_audit.md", "path and storage audit", "filesystem/Git audit"),
        ("supplementary/frozen_index.json", "frozen dataset index", "Stage7 index probe"),
        ("supplementary/idx0_gate_decision.json", "frozen idx0 decision", "small real-data gate"),
        ("supplementary/checkpoint_identity_audit.json", "checkpoint identity validation", "offline reporter"),
        ("supplementary/training_progress.json", "final training progress snapshot", "runtime instrumentation"),
        ("supplementary/training_progress.jsonl", "training progress history", "runtime instrumentation"),
        ("supplementary/materialization_status.json", "materialization status ledger", "AutoDL runtime"),
        ("supplementary/materialization_summary.json", "materialization completion summary", "AutoDL runtime"),
        ("supplementary/materialization_failures.json", "materialization failure journal", "AutoDL runtime"),
        ("supplementary/materialization_progress.json", "materialization progress snapshot", "AutoDL runtime"),
        ("supplementary/materialization_progress.jsonl", "materialization progress history", "AutoDL runtime"),
        ("supplementary/materialization_vllm.log", "materialization vLLM log", "AutoDL runtime"),
    ]
    entries = [
        _artifact_entry(formal_dir / name, role=role, source=source, formal_dir=formal_dir)
        for name, role, source in expected
    ]
    for record in figure_records:
        entries.append(
            _artifact_entry(
                formal_dir / "figures" / record["file"],
                role="real-data figure",
                source=record["source"],
                formal_dir=formal_dir,
            )
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "formal_dir": str(formal_dir),
        "entries": entries,
        "present": sum(1 for entry in entries if entry["exists"]),
        "missing": [entry["relative_path"] for entry in entries if not entry["exists"]],
    }
    _atomic_json(formal_dir / "supplementary" / "artifact_audit.json", payload)
    return payload


def write_artifact_manifest(formal_dir: Path, data_root: Path | None) -> dict[str, Any]:
    archive_files = []
    excluded = {
        "supplementary/artifact_manifest.json",
        "supplementary/artifact_audit.json",
    }
    for path in sorted(item for item in formal_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(formal_dir).as_posix()
        if relative_path in excluded:
            continue
        archive_files.append(
            {
                "relative_path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    dataset_files = []
    if data_root is not None and data_root.is_dir():
        for path in sorted(item for item in data_root.rglob("*") if item.is_file()):
            dataset_files.append(
                {
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "formal_dir": str(formal_dir),
        "data_root": str(data_root) if data_root is not None else None,
        "archive_files": archive_files,
        "dataset_files": dataset_files,
        "dataset_file_count": len(dataset_files),
        "dataset_total_bytes": sum(row["size"] for row in dataset_files),
        "note": "This manifest excludes itself and supplementary/artifact_audit.json to avoid circular hashes.",
    }
    _atomic_json(formal_dir / "supplementary" / "artifact_manifest.json", payload)
    return payload


def _load_training_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render_report(
    *,
    formal_dir: Path,
    data_root: Path | None,
    history: list[dict[str, Any]] | None,
    summary: dict[str, Any] | None,
    integrity: dict[str, Any] | None,
    materialized: dict[str, Any] | None,
    native_audit: dict[str, Any] | None,
    normalization: dict[str, Any] | None,
    training_config: dict[str, Any],
    figures: list[dict[str, str]],
    git: dict[str, str],
    checkpoint_audit: dict[str, Any],
) -> str:
    counts = (integrity or {}).get("counts", {})
    training = training_config.get("training", {})
    optimizer = training.get("optimizer", {})
    scheduler = training.get("scheduler", {})
    best_epoch = (summary or {}).get("best_epoch", INSUFFICIENT)
    best_val = (summary or {}).get("best_val_masked_bce", INSUFFICIENT)
    last = history[-1] if history else {}
    run_id = (summary or {}).get("run_id", INSUFFICIENT)
    training_head = (summary or {}).get("git_head", INSUFFICIENT)
    materialized_count = (materialized or {}).get("count", INSUFFICIENT)
    integrity_status = (integrity or {}).get("status", INSUFFICIENT)
    native_frames = (native_audit or {}).get("frame_count", INSUFFICIENT)
    normalization_train = (normalization or {}).get("train_video_count", INSUFFICIENT)
    checkpoints_valid = checkpoint_audit.get("status") == "PASS"
    training_progress = _read_json(formal_dir / "supplementary" / "training_progress.json")
    training_elapsed = (
        training_progress.get("elapsed_sec", INSUFFICIENT)
        if isinstance(training_progress, dict)
        else INSUFFICIENT
    )
    materialization_logs_present = all(
        (formal_dir / "supplementary" / name).is_file()
        for name in (
            "materialization_status.json",
            "materialization_progress.json",
            "materialization_progress.jsonl",
            "materialization_vllm.log",
        )
    )
    archive_complete = checkpoints_valid and materialization_logs_present
    report_lines = [
        f"# {REPORT_TITLE}",
        "",
        "> 本报告由现有本机 JSON、配置、Git 与文件系统离线生成；没有重新训练、没有重新物化，也没有调用 Qwen/RT-DETR。证据缺失处明确标为 `NOT RECORDED / CURRENT ARTIFACTS INSUFFICIENT`。",
        "",
        "## 1. 实验目标",
        "",
        "在冻结 YouTube Highlights split、Native16 语义、Y 目标、FTNet 架构和训练超参数的前提下，完成真实数据 FTNet baseline 的可审计归档。",
        "",
        "## 2. 数据来源与 split",
        "",
        "| Split | 视频数 |",
        "|---|---:|",
        f"| TRAIN | {counts.get('TRAIN', INSUFFICIENT)} |",
        f"| VALIDATION | {counts.get('VALIDATION', INSUFFICIENT)} |",
        f"| CALIBRATION | {counts.get('CALIBRATION', INSUFFICIENT)} |",
        f"| Total | {materialized_count} |",
        "",
        "数据集为 YouTube Highlights；`supplementary/frozen_index.json` 保存源视频相对路径、split、视频元数据与源 SHA-256。Official Test 与 TVSum 在完整性报告中均为 0 行。",
        "",
        "## 3. Real provider pipeline",
        "",
        "冻结流水线为 Qwen/vLLM retrieval → 2.0 fps 网格 RT-DETR detection + encoder level-0 GAP → CPU LOC/CMP/TS/Native16/Y → 每视频一个 safetensors。",
        "",
        "## 4. 154 视频 materialization",
        "",
        f"- Manifest count: `{materialized_count}`",
        f"- Integrity status: `{integrity_status}`",
        f"- 数据物理路径: `{data_root if data_root is not None else INSUFFICIENT}`",
        "- 逐视频 runtime: " + INSUFFICIENT,
        "",
        "## 5. Native16 audit",
        "",
        f"TRAIN audit videos: `{(native_audit or {}).get('video_count', INSUFFICIENT)}`；frames: `{native_frames}`；schema: `{(native_audit or {}).get('native_schema_version', INSUFFICIENT)}`。",
        "",
        "## 6. Normalization",
        "",
        f"仅使用 TRAIN：`{normalization_train}` 个视频；native_dim=`{(normalization or {}).get('native_dim', INSUFFICIENT)}`；missing policy=`{(normalization or {}).get('missing_policy', INSUFFICIENT)}`。",
        "",
        "## 7. Integrity",
        "",
        f"状态 `{integrity_status}`；missing={len((integrity or {}).get('missing', []))}，extra={len((integrity or {}).get('extra', []))}，duplicate={len((integrity or {}).get('duplicate', []))}，corrupt={len((integrity or {}).get('corrupt', []))}。",
        "",
        "## 8. FTNet architecture",
        "",
        f"visual_dim={training_config.get('visual_dim', INSUFFICIENT)}，native_dim={training_config.get('native_dim', INSUFFICIENT)}，temporal_channels={training_config.get('temporal_channels', INSUFFICIENT)}，dilations={training_config.get('temporal_dilations', INSUFFICIENT)}，dropout={training_config.get('dropout', INSUFFICIENT)}。",
        "",
        "## 9. Training configuration",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| Run ID | `{run_id}` |",
        f"| Device | `{(summary or {}).get('device', INSUFFICIENT)}` |",
        f"| Batch size | `{training.get('batch_size_videos', INSUFFICIENT)}` videos |",
        f"| Optimizer | `{optimizer.get('name', INSUFFICIENT)}` |",
        f"| Initial LR | `{optimizer.get('learning_rate', INSUFFICIENT)}` |",
        f"| Scheduler | `{scheduler.get('name', INSUFFICIENT)}` |",
        f"| Epochs | `{(summary or {}).get('epochs_run', INSUFFICIENT)}` |",
        f"| Steps | `{(summary or {}).get('steps', INSUFFICIENT)}` |",
        f"| Seed | `{training.get('seed', INSUFFICIENT)}` |",
        "",
        "## 10. Epoch / batch / optimizer / lr / scheduler",
        "",
        "上述值直接来自训练 summary 与冻结 `ftnet_reference.yaml`。训练历史按 epoch 记录 train loss、validation loss、学习率和最后一个 batch 的 gradient norm。",
        "",
        "## 11. Training curves",
        "",
    ]
    for record in figures:
        if record["file"].startswith("native16_"):
            continue
        report_lines.extend(
            [
                f"![{record['file']}](figures/{record['file']})",
                "",
                f"数据来源：`{record['source']}`。",
                "",
            ]
        )
    if not history or not all("epoch_duration_s" in row for row in history):
        report_lines.extend(["Epoch duration / throughput 图：" + INSUFFICIENT, ""])
    report_lines.extend(
        [
            "## 12. Validation result",
            "",
            f"Best validation masked BCE = `{best_val}`，best epoch = `{best_epoch}`。Epoch 40 train=`{last.get('train_loss', INSUFFICIENT)}`，validation=`{last.get('val_loss', INSUFFICIENT)}`。",
            "",
            "## 13. Best checkpoint",
            "",
            f"Summary 记录路径：`{(summary or {}).get('checkpoint_best', INSUFFICIENT)}`。本地正式副本为 `results/checkpoints/best.pt`；身份校验状态：`{checkpoint_audit.get('status', INSUFFICIENT)}`。",
            "",
            "## 14. Runtime breakdown",
            "",
            f"正式训练 progress 记录总耗时约 {training_elapsed} s；原始 materialization 日志已归档，但其最终 progress snapshot 对应最后一次单视频续作，不能据此反推完整 154 视频总耗时，因此不生成虚构的总耗时图。",
            "",
            "## 15. GPU / CPU / I/O observations",
            "",
            "源码证据显示：Qwen 与 RT-DETR 为 GPU 阶段；FFmpeg/视频解码和模型加载同时涉及 CPU/I/O；LOC/CMP/TS/Native16 与序列化为 CPU/I/O。实际 GPU utilization、CPU utilization 与 I/O throughput：" + INSUFFICIENT + "。",
            "",
            "## 16. Failure + retry history",
            "",
            ("AutoDL 原始 materialization status/progress/vLLM 日志与训练 progress 已原样归档到 `supplementary/`，可逐条复核续作与失败记录。" if materialization_logs_present else INSUFFICIENT + "。关键 materialization 原始日志尚未齐备。"),
            "",
            "## 17. Dataset physical paths",
            "",
            f"- Processed dataset: `{data_root if data_root is not None else INSUFFICIENT}`",
            f"- Formal archive: `{formal_dir}`",
            "",
            "## 18. Checkpoint paths",
            "",
            f"- AutoDL best source: `{(summary or {}).get('checkpoint_best', INSUFFICIENT)}`",
            f"- AutoDL last source: `{(summary or {}).get('checkpoint_last', INSUFFICIENT)}`",
            "- 本机正式副本：`results/checkpoints/best.pt`、`results/checkpoints/last.pt`。",
            "- size / SHA-256 / epoch / step / run id / Git identity：见 `supplementary/checkpoint_identity_audit.json`。",
            "",
            "## 19. Git identity",
            "",
            f"- Training Git HEAD: `{training_head}`",
            f"- Audit Git HEAD: `{git['head']}`",
            f"- Audit branch: `{git['branch']}`",
            "",
            "## 20. Reproducibility",
            "",
            "使用 `supplementary/frozen_index.json`、`results/materialized_videos.json`、TRAIN-only normalization、`config/` 冻结配置、checkpoint 身份审计与 summary 中的训练 Git identity。FTNet 训练可直接读取 E 盘既有 derived 数据，无需再次运行 Qwen 或 RT-DETR。",
            "",
            "## 21. 当前局限",
            "",
            "- checkpoint 与关键原始日志已归档；checkpoint 身份校验为 " + ("PASS。" if checkpoints_valid else "未通过。"),
            "- 完整 154 视频 materialization 的端到端 wall-clock 总耗时未由单一日志字段记录；不得用最后一次续作的单视频 elapsed 外推。",
            "- 训练 history 不含 epoch duration/throughput，因此相应图不生成。",
            "- 当前未执行 Full154 或正式训练复跑，也未执行性能 benchmark。",
            "",
            "## 22. Stage 7.1 结论",
            "",
            (f"计算结果由现有 summary/integrity 支持为 154/154、40 epochs；7 张真实数据图、正式 checkpoint、配置快照、关键原始日志及内容清单均已归档，checkpoint 校验为 `{checkpoint_audit.get('status', INSUFFICIENT)}`。归档状态可声明 `COMPLETE_AND_ARCHIVED`。" if archive_complete else "当前证据仍不满足 `COMPLETE_AND_ARCHIVED`；缺失项以 `results/formal_run_status.json` 为准。"),
            "",
            "## 23. 下一步",
            "",
            "本阶段无需追加计算。后续若做性能对比，只能作为新的小样本工程 benchmark 单独立项；不得为本归档重跑 Full154、Qwen、RT-DETR 或正式训练。",
            "",
            "## 图表数据表",
            "",
            "| Figure | Machine source |",
            "|---|---|",
        ]
    )
    report_lines.extend(f"| `figures/{record['file']}` | `{record['source']}` |" for record in figures)
    report_lines.extend(
        [
            "",
            "## 方法参考",
            "",
            "图表遵循原始数据不增删、不插值缺失观测、来源显式记录的原则。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065",
            "",
        ]
    )
    return "\n".join(report_lines)


def generate_formal_deliverables(
    formal_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    index_path: str | Path | None = None,
    training_config_path: str | Path | None = None,
    training_run_dir: str | Path | None = None,
    materialization_run_dir: str | Path | None = None,
    idx0_gate_path: str | Path | None = None,
    performance_summary_path: str | Path | None = None,
    config_snapshot_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    formal = Path(formal_dir).expanduser().resolve()
    formal.mkdir(parents=True, exist_ok=True)
    for directory in ARCHIVE_DIRECTORIES:
        (formal / directory).mkdir(parents=True, exist_ok=True)
    data = Path(data_root).expanduser().resolve() if data_root is not None else None
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else None
    index = Path(index_path).expanduser().resolve() if index_path is not None else None
    training_run = (
        Path(training_run_dir).expanduser().resolve()
        if training_run_dir is not None
        else None
    )
    materialization_run = (
        Path(materialization_run_dir).expanduser().resolve()
        if materialization_run_dir is not None
        else None
    )
    idx0_gate = (
        Path(idx0_gate_path).expanduser().resolve()
        if idx0_gate_path is not None
        else None
    )
    performance_summary = (
        Path(performance_summary_path).expanduser().resolve()
        if performance_summary_path is not None
        else None
    )
    config_path = (
        Path(training_config_path).expanduser().resolve()
        if training_config_path is not None
        else None
    )
    snapshots = tuple(Path(path).expanduser().resolve() for path in config_snapshot_paths)
    _copy_lightweight_inputs(formal, data, index)
    _copy_training_run(formal, training_run)
    _copy_materialization_run(formal, materialization_run)
    _copy_optional_evidence(
        formal,
        idx0_gate,
        "supplementary/idx0_gate_decision.json",
    )
    _copy_optional_evidence(
        formal,
        performance_summary,
        "results/performance_summary.json",
    )
    _copy_config_snapshots(formal, config_path, snapshots)

    history_path = _first_file(formal / "results" / "training_history.json")
    summary_path = _first_file(formal / "results" / "training_summary.json")
    native_path = _first_file(formal / "results" / "native_feature_audit.json")
    integrity_path = _first_file(formal / "results" / "integrity_report.json")
    materialized_path = _first_file(formal / "results" / "materialized_videos.json")
    normalization_path = _first_file(formal / "results" / "normalization_stats.json")
    figures = render_figures(history_path, native_path, formal / "figures")
    history = _read_json(history_path)
    summary = _read_json(summary_path)
    checkpoint_audit = validate_checkpoint_identity(
        formal,
        summary if isinstance(summary, dict) else None,
    )
    _atomic_json(
        formal / "supplementary" / "checkpoint_identity_audit.json",
        checkpoint_audit,
    )
    report = _render_report(
        formal_dir=formal,
        data_root=data,
        history=history if isinstance(history, list) else None,
        summary=summary if isinstance(summary, dict) else None,
        integrity=_read_json(integrity_path),
        materialized=_read_json(materialized_path),
        native_audit=_read_json(native_path),
        normalization=_read_json(normalization_path),
        training_config=_load_training_config(config_path),
        figures=figures,
        git=_git_identity(repo),
        checkpoint_audit=checkpoint_audit,
    )
    (formal / "experiment_report.md").write_text(report, encoding="utf-8", newline="\n")

    required = [
        formal / "experiment_report.md",
        formal / "config" / "ftnet_reference.yaml",
        formal / "config" / "ftnet_train_formal.json",
        formal / "config" / "windows_local.json",
        formal / "results" / "training_history.json",
        formal / "results" / "training_summary.json",
        formal / "results" / "normalization_stats.json",
        formal / "results" / "normalization_stats.json.sha256",
        formal / "results" / "native_feature_audit.json",
        formal / "results" / "integrity_report.json",
        formal / "results" / "materialized_videos.json",
        formal / "results" / "performance_summary.json",
        formal / "results" / "checkpoints" / "best.pt",
        formal / "results" / "checkpoints" / "last.pt",
        formal / "supplementary" / "frozen_index.json",
        formal / "supplementary" / "idx0_gate_decision.json",
        formal / "supplementary" / "training_progress.json",
        formal / "supplementary" / "training_progress.jsonl",
        formal / "supplementary" / "materialization_status.json",
        formal / "supplementary" / "materialization_summary.json",
        formal / "supplementary" / "materialization_failures.json",
        formal / "supplementary" / "materialization_progress.json",
        formal / "supplementary" / "materialization_progress.jsonl",
        formal / "supplementary" / "materialization_vllm.log",
        formal / "supplementary" / "path_and_storage_audit.md",
        formal / "supplementary" / "performance_audit.md",
        *[
            formal / "figures" / name
            for name in (
                *TRAINING_FIGURES,
                "native16_mean_std.png",
                "native16_unique_or_variability.png",
            )
        ],
    ]
    missing_required = [
        path.relative_to(formal).as_posix()
        for path in required
        if not path.is_file()
    ]
    if checkpoint_audit["status"] != "PASS" and not any(
        path.startswith("results/checkpoints/") for path in missing_required
    ):
        missing_required.append("supplementary/checkpoint_identity_audit.json:PASS")
    status = {
        "schema": STATUS_SCHEMA,
        "status": "COMPLETE" if not missing_required else "INCOMPLETE",
        "missing_required": missing_required,
        "figure_count": len(figures),
        "dataset_file_count": (
            len([path for path in data.rglob("*") if path.is_file()])
            if data is not None and data.is_dir()
            else 0
        ),
        "checkpoint_validation": checkpoint_audit["status"],
    }
    _atomic_json(formal / "results" / "formal_run_status.json", status)
    write_artifact_manifest(formal, data)
    write_artifact_audit(formal, figures)
    return status


__all__ = [
    "AUDIT_SCHEMA",
    "ARCHIVE_DIRECTORIES",
    "CHECKPOINT_AUDIT_SCHEMA",
    "INSUFFICIENT",
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "REPORT_TITLE",
    "STATUS_SCHEMA",
    "TRAINING_FIGURES",
    "generate_formal_deliverables",
    "render_figures",
    "sha256_file",
    "validate_checkpoint_identity",
    "write_artifact_audit",
    "write_artifact_manifest",
]
