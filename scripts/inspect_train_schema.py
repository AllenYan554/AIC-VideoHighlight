#!/usr/bin/env python3
"""Read-only, streaming JSONL schema inspection with no annotation payload export."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIELDS_OF_INTEREST = (
    "timeline",
    "highlight_score",
    "confidence",
    "candidate_segments",
    "crop_keyframes",
    "trajectory",
    "summary",
    "dataset_split",
    "provenance",
    "seed_model",
    "prompt_fingerprint",
)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _walk_schema(
    value: Any,
    path: str,
    observed: dict[str, Counter[str]],
    *,
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    observed[path][_type_name(value)] += 1
    if depth >= max_depth:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            _walk_schema(child, child_path, observed, depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, list) and value:
        # One representative item per record bounds work and never copies values.
        _walk_schema(value[0], f"{path}[]", observed, depth=depth + 1, max_depth=max_depth)


def inspect_jsonl(source: Path) -> str:
    size_bytes = source.stat().st_size
    total_lines = 0
    blank_lines = 0
    valid_objects = 0
    invalid_lines: list[tuple[int, str]] = []
    top_level_sets: Counter[tuple[str, ...]] = Counter()
    field_presence: Counter[str] = Counter()
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    fixed_samples: list[tuple[int, tuple[str, ...]]] = []

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines = line_number
            if not line.strip():
                blank_lines += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if len(invalid_lines) < 20:
                    invalid_lines.append((line_number, exc.msg))
                continue
            if not isinstance(record, dict):
                if len(invalid_lines) < 20:
                    invalid_lines.append((line_number, "top-level value is not an object"))
                continue

            valid_objects += 1
            keys = tuple(sorted(record))
            top_level_sets[keys] += 1
            for field in FIELDS_OF_INTEREST:
                if field in record:
                    field_presence[field] += 1
            _walk_schema(record, "$", observed)
            if len(fixed_samples) < 5:
                fixed_samples.append((line_number, keys))

    nonblank_lines = total_lines - blank_lines
    independent_json = nonblank_lines == valid_objects and not invalid_lines
    consistent_fields = len(top_level_sets) <= 1
    lines = [
        "# train.jsonl Schema 检查记录",
        "",
        "## 确定可以看出的事实",
        "",
        f"- 只读来源：`{source}`",
        f"- 文件大小：{size_bytes} bytes",
        f"- 总行数：{total_lines}",
        f"- 空行数：{blank_lines}",
        f"- 合法顶层 JSON 对象数：{valid_objects}",
        f"- 每个非空行是否均为独立 JSON 对象：{'是' if independent_json else '否'}",
        f"- 顶层字段集合是否完全一致：{'是' if consistent_fields else '否'}",
        f"- 不同顶层字段集合数量：{len(top_level_sets)}",
        "",
        "### 重点字段在顶层的出现次数",
        "",
    ]
    lines.extend(f"- `{field}`：{field_presence[field]}" for field in FIELDS_OF_INTEREST)
    lines.extend(["", "### 重点字段的同名嵌套路径", ""])
    for field in FIELDS_OF_INTEREST:
        matching_paths = [
            path
            for path in observed
            if path.rsplit(".", 1)[-1].removesuffix("[]") == field
        ]
        if matching_paths:
            rendered = ", ".join(
                f"`{path}`（{sum(observed[path].values())} 条观测）" for path in sorted(matching_paths)
            )
            lines.append(f"- `{field}`：{rendered}")
        else:
            lines.append(f"- `{field}`：未观测到顶层或嵌套同名路径")
    lines.extend(["", "### 顶层字段集合（按出现次数，最多 20 种）", ""])
    for keys, count in top_level_sets.most_common(20):
        lines.append(f"- {count} 条：`{', '.join(keys)}`")
    if not top_level_sets:
        lines.append("- 未发现合法 JSON 对象。")

    lines.extend(["", "### 固定抽样（最多前 5 条合法记录，仅列字段名）", ""])
    for line_number, keys in fixed_samples:
        lines.append(f"- 第 {line_number} 行：`{', '.join(keys)}`")
    if not fixed_samples:
        lines.append("- 无可抽样记录。")

    lines.extend(["", "### 字段层级与观测类型（深度最多 4 层，最多 250 项）", ""])
    for path in sorted(observed)[:250]:
        type_summary = ", ".join(f"{name}×{count}" for name, count in observed[path].most_common())
        lines.append(f"- `{path}`：{type_summary}")

    lines.extend(["", "### 解析异常（最多记录 20 条，仅列行号与错误类型）", ""])
    if invalid_lines:
        lines.extend(f"- 第 {line_number} 行：{message}" for line_number, message in invalid_lines)
    else:
        lines.append("- 未发现解析异常。")

    lines.extend(
        [
            "",
            "## 暂时无法确认的标注语义",
            "",
            "- 字段名本身不能证明某字段是赛事最终人工 Ground Truth。",
            "- 即使存在 `candidate_segments`，也不能仅凭名称把它作为唯一 GT。",
            "- `seed_model`、`prompt_fingerprint`、`provenance` 等字段若存在，只能确认记录包含生成/来源元数据；不能据此确定标签由人工、模型或混合流程产生。",
            "- 在获得赛事数据说明、字段定义或可靠样例解释前，`gt_reader.py` 不会默认选择任何 GT 字段。",
            "- 本报告不包含训练标注原文、摘要内容、轨迹值或候选区间值。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inspect train.jsonl structure without modifying it")
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads" / "train.jsonl")
    parser.add_argument("--output", type=Path, default=project_root / "docs" / "train-schema-notes.md")
    args = parser.parse_args()

    if args.input.is_file():
        report = inspect_jsonl(args.input.resolve())
    else:
        report = "\n".join(
            [
                "# train.jsonl Schema 检查记录",
                "",
                "## 确定可以看出的事实",
                "",
                f"- 本轮未在只读候选路径 `{args.input}` 找到文件，因此未读取任何训练标注。",
                "",
                "## 暂时无法确认的标注语义",
                "",
                "- 当前没有本地文件可供检查，所有字段语义均待赛事说明或数据文件确认。",
                "- `gt_reader.py` 不会默认选择任何 GT 字段。",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Schema report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
