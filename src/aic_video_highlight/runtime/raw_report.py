"""Render factual machine evidence; never emit scientific interpretation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import atomic_write_text


def render_raw_report(machine_dir: Path, output: Path, experiment_id: str) -> None:
    evidence: dict[str, Any] = {}
    for name in ("summary", "metrics", "runtime", "validation", "artifact_manifest"):
        path = machine_dir / f"{name}.json"
        evidence[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    lines = [
        f"# Experiment Raw Report: {experiment_id}",
        "",
        "> Machine-generated factual evidence only. Scientific interpretation is intentionally excluded.",
        "",
    ]
    for name, value in evidence.items():
        lines.extend([f"## {name.replace('_', ' ').title()}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""])
    atomic_write_text(output, "\n".join(lines))


def write_ai_report_inputs(root: Path) -> None:
    atomic_write_text(
        root / "AI_REPORT_INPUTS.md",
        "# AI Final Report Inputs\n\n"
        "请将以下事实文件交给 AI：\n\n"
        "- `experiment_raw_report.md`\n"
        "- `machine/summary.json`\n"
        "- `machine/metrics.json`\n"
        "- `machine/runtime.json`\n"
        "- `machine/validation.json`\n"
        "- `machine/artifact_manifest.json`\n"
        "- `figures/`\n\n"
        "AI 根据这些证据撰写 `experiment_report.md`。实验程序不得创建或覆盖该最终科研报告。\n",
    )
