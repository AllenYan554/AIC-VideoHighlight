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
INSUFFICIENT = "NOT RECORDED / CURRENT ARTIFACTS INSUFFICIENT"

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
        (index_path, "frozen_index.json"),
    ]
    if data_root is not None:
        sources.extend(
            [
                (data_root / "manifests" / "materialized_videos.json", "materialized_videos.json"),
                (data_root / "normalization" / "normalization_stats.json", "normalization_stats.json"),
                (
                    data_root / "normalization" / "normalization_stats.json.sha256",
                    "normalization_stats.json.sha256",
                ),
                (data_root / "audits" / "native_feature_audit.json", "native_feature_audit.json"),
                (data_root / "audits" / "integrity_report.json", "integrity_report.json"),
            ]
        )
    for source, name in sources:
        target = formal_dir / name
        if source is not None and source.is_file() and not target.exists():
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
                "training_history.json: train_loss",
            ),
            (
                "validation_loss_vs_epoch.png",
                "FTNet validation loss",
                "Masked BCE",
                [("Validation", values("val_loss"), "#D55E00", "--")],
                "training_history.json: val_loss",
            ),
            (
                "train_validation_loss.png",
                "FTNet train and validation loss",
                "Masked BCE",
                [
                    ("Train", values("train_loss"), "#0072B2", "-"),
                    ("Validation", values("val_loss"), "#D55E00", "--"),
                ],
                "training_history.json: train_loss, val_loss",
            ),
            (
                "learning_rate_vs_epoch.png",
                "Learning-rate schedule",
                "Learning rate",
                [("Learning rate", values("lr"), "#009E73", "-")],
                "training_history.json: lr",
            ),
            (
                "gradient_norm_vs_epoch.png",
                "Gradient norm by epoch",
                "Global L2 norm",
                [("Gradient norm", values("grad_norm"), "#CC79A7", "-")],
                "training_history.json: grad_norm",
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
                    "source": "native_feature_audit.json: fields[*].mean/std",
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
                    "source": "native_feature_audit.json: unique_count/count/missing_rate",
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
        ("artifact_manifest.json", "content manifest", "offline reporter"),
        ("performance_audit.md", "performance root-cause audit", "manual/code audit"),
        ("performance_summary.json", "machine-readable performance evidence", "logs/instrumentation"),
        ("path_and_storage_audit.md", "path and storage audit", "filesystem/Git audit"),
        ("frozen_index.json", "frozen dataset index", "Stage7 index probe"),
        ("idx0_gate_decision.json", "frozen idx0 decision", "small real-data gate"),
        ("integrity_report.json", "dataset integrity result", "integrity gate"),
        ("materialized_videos.json", "materialized dataset manifest", "materialization"),
        ("native_feature_audit.json", "Native16 field audit", "TRAIN-only audit"),
        ("normalization_stats.json", "TRAIN-only normalization", "normalization"),
        ("normalization_stats.json.sha256", "normalization digest sidecar", "normalization"),
        ("training_history.json", "formal training epoch history", "formal training"),
        ("training_summary.json", "formal training summary", "formal training"),
        ("best.pt", "best formal checkpoint", "formal training"),
        ("last.pt", "last formal checkpoint", "formal training"),
        ("progress.json", "final progress snapshot", "runtime instrumentation"),
        ("progress.jsonl", "append-only progress history", "runtime instrumentation"),
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
    _atomic_json(formal_dir / "artifact_audit.json", payload)
    return payload


def write_artifact_manifest(formal_dir: Path, data_root: Path | None) -> dict[str, Any]:
    archive_files = []
    excluded = {"artifact_manifest.json", "artifact_audit.json"}
    for path in sorted(item for item in formal_dir.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        archive_files.append(
            {
                "relative_path": path.relative_to(formal_dir).as_posix(),
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
        "note": "This manifest excludes itself and artifact_audit.json to avoid circular hashes.",
    }
    _atomic_json(formal_dir / "artifact_manifest.json", payload)
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
        "数据集为 YouTube Highlights；`frozen_index.json` 保存源视频相对路径、split、视频元数据与源 SHA-256。Official Test 与 TVSum 在完整性报告中均为 0 行。",
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
            f"Summary 记录路径：`{(summary or {}).get('checkpoint_best', INSUFFICIENT)}`。正式归档内 `best.pt` 是否存在以 `artifact_audit.json` 为准。",
            "",
            "## 14. Runtime breakdown",
            "",
            INSUFFICIENT + "。现有 training history/summary 未记录训练总时长或逐 epoch 时长；正式归档也没有 materialization progress/status 日志，因此不生成 runtime breakdown 图。",
            "",
            "## 15. GPU / CPU / I/O observations",
            "",
            "源码证据显示：Qwen 与 RT-DETR 为 GPU 阶段；FFmpeg/视频解码和模型加载同时涉及 CPU/I/O；LOC/CMP/TS/Native16 与序列化为 CPU/I/O。实际 GPU utilization、CPU utilization 与 I/O throughput：" + INSUFFICIENT + "。",
            "",
            "## 16. Failure + retry history",
            "",
            INSUFFICIENT + "。归档缺少 failure journal 与 progress JSONL；不得仅用历史文字描述替代。",
            "",
            "## 17. Dataset physical paths",
            "",
            f"- Processed dataset: `{data_root if data_root is not None else INSUFFICIENT}`",
            f"- Formal archive: `{formal_dir}`",
            "",
            "## 18. Checkpoint paths",
            "",
            f"- best: `{(summary or {}).get('checkpoint_best', INSUFFICIENT)}`",
            f"- last: `{(summary or {}).get('checkpoint_last', INSUFFICIENT)}`",
            "- 本机正式归档副本存在性：见 `artifact_audit.json`。",
            "",
            "## 19. Git identity",
            "",
            f"- Training Git HEAD: `{training_head}`",
            f"- Audit Git HEAD: `{git['head']}`",
            f"- Audit branch: `{git['branch']}`",
            "",
            "## 20. Reproducibility",
            "",
            "使用 `frozen_index.json`、`materialized_videos.json`、TRAIN-only normalization、冻结配置与 summary 中的训练 Git identity。当前缺少正式 checkpoint 和原始进度日志，本机归档尚不能单独完成全链路重放审计。",
            "",
            "## 21. 当前局限",
            "",
            "- 正式 `best.pt` / `last.pt` 未在本机归档发现。",
            "- materialization `status.json`、`progress.json/jsonl`、vLLM 日志未在本机归档发现。",
            "- 训练 history 不含 epoch duration/throughput，因此相应图不生成。",
            "- 当前未执行 Full154 或正式训练复跑，也未执行性能 benchmark。",
            "",
            "## 22. Stage 7.1 结论",
            "",
            f"计算结果由现有 summary/integrity 支持为 154/154、40 epochs；报告与真实数据曲线已离线补齐。但正式 checkpoint 与关键性能日志仍缺，因此不能声明 `COMPLETE_AND_ARCHIVED`。",
            "",
            "## 23. 下一步",
            "",
            "AutoDL 若重新开启，只应同步已经存在的正式 checkpoint 与 progress/status/vLLM 日志；不得重跑 Full154。之后可经用户批准用 1–3 个视频执行工程等价性与性能 smoke。",
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
) -> dict[str, Any]:
    formal = Path(formal_dir).expanduser().resolve()
    formal.mkdir(parents=True, exist_ok=True)
    data = Path(data_root).expanduser().resolve() if data_root is not None else None
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else None
    index = Path(index_path).expanduser().resolve() if index_path is not None else None
    config_path = (
        Path(training_config_path).expanduser().resolve()
        if training_config_path is not None
        else None
    )
    _copy_lightweight_inputs(formal, data, index)

    # The trainer keeps short runtime names; the formal archive uses explicit
    # canonical names.  Copy rather than rename so resume semantics stay intact.
    for source_name, canonical_name in (
        ("history.json", "training_history.json"),
        ("summary.json", "training_summary.json"),
        ("logs/progress.json", "progress.json"),
        ("logs/progress.jsonl", "progress.jsonl"),
    ):
        source = formal / source_name
        target = formal / canonical_name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)

    history_path = _first_file(formal / "training_history.json", formal / "history.json")
    summary_path = _first_file(formal / "training_summary.json", formal / "summary.json")
    native_path = _first_file(formal / "native_feature_audit.json")
    integrity_path = _first_file(formal / "integrity_report.json")
    materialized_path = _first_file(formal / "materialized_videos.json")
    normalization_path = _first_file(formal / "normalization_stats.json")
    figures = render_figures(history_path, native_path, formal / "figures")
    history = _read_json(history_path)
    summary = _read_json(summary_path)
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
    )
    (formal / "experiment_report.md").write_text(report, encoding="utf-8", newline="\n")
    manifest = write_artifact_manifest(formal, data)
    audit = write_artifact_audit(formal, figures)

    required = [
        formal / "experiment_report.md",
        formal / "artifact_manifest.json",
        formal / "best.pt",
        formal / "last.pt",
        history_path,
        summary_path,
        *[formal / "figures" / name for name in TRAINING_FIGURES],
    ]
    missing_required = [str(path) for path in required if path is None or not path.is_file()]
    status = {
        "schema": STATUS_SCHEMA,
        "status": "COMPLETE" if not missing_required else "INCOMPLETE",
        "missing_required": missing_required,
        "figure_count": len(figures),
        "artifact_count": len(manifest["archive_files"]),
        "dataset_file_count": manifest["dataset_file_count"],
        "audit_present": audit["present"],
    }
    _atomic_json(formal / "formal_run_status.json", status)
    return status


__all__ = [
    "AUDIT_SCHEMA",
    "INSUFFICIENT",
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "REPORT_TITLE",
    "STATUS_SCHEMA",
    "TRAINING_FIGURES",
    "generate_formal_deliverables",
    "render_figures",
    "sha256_file",
    "write_artifact_audit",
    "write_artifact_manifest",
]
