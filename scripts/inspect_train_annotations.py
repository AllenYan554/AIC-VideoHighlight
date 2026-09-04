#!/usr/bin/env python3
"""Read-only, full-file analysis for AIC-VideoHighlight JSONL annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def _is_empty(value: Any) -> bool:
    return value == "" or value == [] or value == {}


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _number(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.6g}"


def _summary(values: Iterable[float]) -> str:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return "无可用数值"
    return (
        f"n={len(data)}，min={_number(min(data))}，P25={_number(_quantile(data, 0.25))}，"
        f"median={_number(statistics.median(data))}，mean={_number(statistics.fmean(data))}，"
        f"P75={_number(_quantile(data, 0.75))}，P95={_number(_quantile(data, 0.95))}，"
        f"max={_number(max(data))}"
    )


def _percentage(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def _counter(counter: Counter[Any], *, limit: int = 12) -> str:
    if not counter:
        return "无"
    parts = [f"`{key}`×{count}" for key, count in counter.most_common(limit)]
    if len(counter) > limit:
        parts.append(f"其余 {len(counter) - limit} 种")
    return "，".join(parts)


def _field_types(types: Counter[str]) -> str:
    return ", ".join(f"{name}×{count}" for name, count in types.most_common()) or "未观测"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intervals(items: Any) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            result.append((float(start), float(end)))
    return result


def _inside(time_sec: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start - 1e-9 <= time_sec <= end + 1e-9 for start, end in intervals)


class Analysis:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.size_bytes = source.stat().st_size
        self.sha256 = _sha256(source)
        self.total_lines = 0
        self.blank_lines = 0
        self.valid_records = 0
        self.invalid_lines: list[tuple[int, str]] = []
        self.node_types: dict[str, Counter[str]] = defaultdict(Counter)
        self.node_empty: Counter[str] = Counter()
        self.node_null: Counter[str] = Counter()
        self.numeric_values: dict[str, list[float]] = defaultdict(list)
        self.list_lengths: dict[str, list[int]] = defaultdict(list)
        self.object_instances: Counter[str] = Counter()
        self.object_key_presence: dict[str, Counter[str]] = defaultdict(Counter)
        self.object_keysets: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
        self.scalar_values: dict[str, Counter[Any]] = defaultdict(Counter)
        self.segment_durations: list[float] = []
        self.segment_frame_spans: list[float] = []
        self.segment_valid_time = 0
        self.segment_valid_frame = 0
        self.segment_source_equal = 0
        self.segment_source_comparable = 0
        self.segment_within_clip = 0
        self.segment_clip_comparable = 0
        self.segment_clip_overshoots: list[float] = []
        self.candidate_count_equal_records = 0
        self.candidate_exact_records = 0
        self.candidate_both_nonempty_records = 0
        self.timeline_deltas: list[float] = []
        self.timeline_non_unit_nonfinal_deltas = 0
        self.timeline_nonmonotonic_records = 0
        self.timeline_points = 0
        self.timeline_points_in_segments = 0
        self.timeline_score_inside: list[float] = []
        self.timeline_score_outside: list[float] = []
        self.timeline_binary: list[tuple[float, bool]] = []
        self.crop_keyframe_points = 0
        self.crop_keyframes_in_segments = 0
        self.crop_roi_items = 0
        self.crop_roi_well_formed = 0
        self.crop_roi_frames_in_segments = 0
        self.crop_roi_exact_segment_frame_records = 0
        self.crop_roi_comparable_records = 0
        self.empty_crop_matches_dropped_default = 0
        self.keyframe_trajectory_length_equal_records = 0
        self.keyframe_trajectory_comparable_records = 0
        self.keyframe_trajectory_pairs = 0
        self.keyframe_trajectory_time_equal = 0
        self.keyframe_trajectory_center_equal = 0
        self.keyframe_trajectory_visible_equal = 0
        self.timeline_trajectory_length_equal_records = 0
        self.timeline_trajectory_time_equal_records = 0

    def walk(self, value: Any, path: str = "$") -> None:
        kind = _type_name(value)
        self.node_types[path][kind] += 1
        if value is None:
            self.node_null[path] += 1
        if _is_empty(value):
            self.node_empty[path] += 1
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.numeric_values[path].append(float(value))
        if isinstance(value, dict):
            self.object_instances[path] += 1
            self.object_keysets[path][tuple(sorted(value))] += 1
            for key, child in value.items():
                self.object_key_presence[path][key] += 1
                self.walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            self.list_lengths[path].append(len(value))
            for child in value:
                self.walk(child, f"{path}[]")

    def collect_scalar(self, record: dict[str, Any], path: str) -> None:
        value: Any = record
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return
            value = value[part]
        if isinstance(value, (str, int, float, bool)) or value is None:
            self.scalar_values[path][value] += 1

    def collect_record(self, record: dict[str, Any]) -> None:
        self.valid_records += 1
        self.walk(record)
        for path in (
            "annotation_mode",
            "schema_version",
            "dataset_split",
            "provenance.source",
            "provenance.seed_model",
            "provenance.prompt_version",
            "provenance.prompt_fingerprint",
            "provenance.annotator_config_sha256",
            "provenance.video_sha256",
            "provenance.source_group",
            "provenance.license",
            "provenance.temporal_fps",
            "provenance.spatial_fps",
            "provenance.n_seed_repeats",
            "quality.status",
            "quality.annotation_mode",
            "quality.spatial_source",
            "clip.free_axis",
        ):
            self.collect_scalar(record, path)
        target_ratio = record.get("targetRatioWH")
        if isinstance(target_ratio, list):
            self.scalar_values["targetRatioWH"][tuple(target_ratio)] += 1

        clip = record.get("clip", {})
        segments = record.get("segments", [])
        segment_intervals = _intervals(segments)
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                start = segment.get("start_sec")
                end = segment.get("end_sec")
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    self.segment_durations.append(float(end) - float(start))
                    self.segment_valid_time += int(float(end) >= float(start))
                    clip_start = clip.get("start_sec") if isinstance(clip, dict) else None
                    clip_end = clip.get("end_sec") if isinstance(clip, dict) else None
                    if isinstance(clip_start, (int, float)) and isinstance(clip_end, (int, float)):
                        self.segment_clip_comparable += 1
                        self.segment_within_clip += int(
                            -1e-9
                            <= float(start)
                            <= float(end)
                            <= float(clip_end) - float(clip_start) + 1e-9
                        )
                        overshoot = float(end) - (float(clip_end) - float(clip_start))
                        if overshoot > 1e-9:
                            self.segment_clip_overshoots.append(overshoot)
                start_frame = segment.get("start_frame")
                end_frame = segment.get("end_frame")
                if isinstance(start_frame, (int, float)) and isinstance(end_frame, (int, float)):
                    self.segment_frame_spans.append(float(end_frame) - float(start_frame))
                    self.segment_valid_frame += int(float(end_frame) >= float(start_frame))
                source_start = segment.get("source_start_sec")
                source_end = segment.get("source_end_sec")
                if all(isinstance(v, (int, float)) for v in (start, end, source_start, source_end)):
                    self.segment_source_comparable += 1
                    self.segment_source_equal += int(
                        math.isclose(float(clip.get("start_sec", 0)) + float(start), float(source_start), abs_tol=1e-9)
                        and math.isclose(float(clip.get("start_sec", 0)) + float(end), float(source_end), abs_tol=1e-9)
                    )

        teacher = record.get("teacher_signals", {})
        candidates = teacher.get("candidate_segments", []) if isinstance(teacher, dict) else []
        candidate_intervals = _intervals(candidates)
        if isinstance(segments, list) and isinstance(candidates, list):
            self.candidate_count_equal_records += int(len(segments) == len(candidates))
            if segments and candidates:
                self.candidate_both_nonempty_records += 1
            self.candidate_exact_records += int(segment_intervals == candidate_intervals)

        timeline = teacher.get("timeline", []) if isinstance(teacher, dict) else []
        trajectory = teacher.get("trajectory", []) if isinstance(teacher, dict) else []
        if isinstance(timeline, list):
            times = [
                float(item["time_sec"])
                for item in timeline
                if isinstance(item, dict) and isinstance(item.get("time_sec"), (int, float))
            ]
            deltas = [right - left for left, right in zip(times, times[1:])]
            self.timeline_deltas.extend(delta for delta in deltas if delta > 0)
            self.timeline_non_unit_nonfinal_deltas += sum(
                not math.isclose(delta, 1.0, abs_tol=1e-9)
                for delta in deltas[:-1]
            )
            self.timeline_nonmonotonic_records += int(any(delta <= 0 for delta in deltas))
            for item in timeline:
                if not isinstance(item, dict) or not isinstance(item.get("time_sec"), (int, float)):
                    continue
                time_sec = float(item["time_sec"])
                is_inside = _inside(time_sec, segment_intervals)
                self.timeline_points += 1
                self.timeline_points_in_segments += int(is_inside)
                score = item.get("highlight_score")
                if isinstance(score, (int, float)):
                    target = self.timeline_score_inside if is_inside else self.timeline_score_outside
                    target.append(float(score))
                    self.timeline_binary.append((float(score), is_inside))

        crop_keyframes = record.get("crop_keyframes", [])
        if isinstance(crop_keyframes, list):
            for item in crop_keyframes:
                if isinstance(item, dict) and isinstance(item.get("time_sec"), (int, float)):
                    self.crop_keyframe_points += 1
                    self.crop_keyframes_in_segments += int(
                        _inside(float(item["time_sec"]), segment_intervals)
                    )

        crop_rois = record.get("cropRois", [])
        if isinstance(crop_rois, list):
            observed_frames: list[int] = []
            for item in crop_rois:
                self.crop_roi_items += 1
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], (int, float))
                    and isinstance(item[1], list)
                    and len(item[1]) == 4
                    and all(isinstance(value, (int, float)) for value in item[1])
                ):
                    self.crop_roi_well_formed += 1
                    frame = int(item[0])
                    observed_frames.append(frame)
                    self.crop_roi_frames_in_segments += int(
                        any(
                            isinstance(segment, dict)
                            and isinstance(segment.get("start_frame"), (int, float))
                            and isinstance(segment.get("end_frame"), (int, float))
                            and int(segment["start_frame"]) <= frame <= int(segment["end_frame"])
                            for segment in segments
                        )
                    )
            expected_frames = {
                frame
                for segment in segments
                if isinstance(segment, dict)
                and isinstance(segment.get("start_frame"), (int, float))
                and isinstance(segment.get("end_frame"), (int, float))
                for frame in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1)
            }
            if expected_frames and observed_frames:
                self.crop_roi_comparable_records += 1
                self.crop_roi_exact_segment_frame_records += int(set(observed_frames) == expected_frames)
            quality = record.get("quality", {})
            if not crop_rois and isinstance(quality, dict):
                self.empty_crop_matches_dropped_default += int(
                    quality.get("spatial_source") == "dropped_center_default"
                )

        if isinstance(crop_keyframes, list) and isinstance(trajectory, list):
            self.keyframe_trajectory_comparable_records += 1
            self.keyframe_trajectory_length_equal_records += int(len(crop_keyframes) == len(trajectory))
            for keyframe, point in zip(crop_keyframes, trajectory):
                if not isinstance(keyframe, dict) or not isinstance(point, dict):
                    continue
                self.keyframe_trajectory_pairs += 1
                self.keyframe_trajectory_time_equal += int(keyframe.get("time_sec") == point.get("time_sec"))
                self.keyframe_trajectory_center_equal += int(keyframe.get("raw_center") == point.get("center"))
                self.keyframe_trajectory_visible_equal += int(keyframe.get("visible") == point.get("visible"))

        if isinstance(timeline, list) and isinstance(trajectory, list):
            self.timeline_trajectory_length_equal_records += int(len(timeline) == len(trajectory))
            timeline_times = [item.get("time_sec") for item in timeline if isinstance(item, dict)]
            trajectory_times = [item.get("time_sec") for item in trajectory if isinstance(item, dict)]
            self.timeline_trajectory_time_equal_records += int(timeline_times == trajectory_times)

    def load(self) -> None:
        with self.source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                self.total_lines = line_number
                if not line.strip():
                    self.blank_lines += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if len(self.invalid_lines) < 20:
                        self.invalid_lines.append((line_number, exc.msg))
                    continue
                if not isinstance(record, dict):
                    if len(self.invalid_lines) < 20:
                        self.invalid_lines.append((line_number, "top-level value is not an object"))
                    continue
                self.collect_record(record)

    def schema_table(self, object_path: str) -> list[str]:
        denominator = self.object_instances[object_path]
        rows = ["| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |", "|---|---|---:|---|"]
        for key in sorted(self.object_key_presence[object_path]):
            path = f"{object_path}.{key}"
            presence = self.object_key_presence[object_path][key]
            empty = self.node_empty[path] + self.node_null[path]
            rows.append(
                f"| `{key}` | {_field_types(self.node_types[path])} | "
                f"{presence}/{denominator}（{_percentage(presence, denominator)}） | "
                f"{'是，' + str(empty) + ' 次' if empty else '否'} |"
            )
        return rows

    def best_timeline_threshold(self) -> tuple[float, float, float, float, int, int]:
        best = (0.0, 0.0, 0.0, 0.0, 0, 0)
        for index in range(21):
            threshold = index / 20
            tp = sum(score >= threshold and label for score, label in self.timeline_binary)
            fp = sum(score >= threshold and not label for score, label in self.timeline_binary)
            fn = sum(score < threshold and label for score, label in self.timeline_binary)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best[3]:
                best = (threshold, precision, recall, f1, tp + fp, tp + fn)
        return best

    def _string_lengths(self, path: str) -> list[int]:
        lengths: list[int] = []
        with self.source.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value: Any = json.loads(line)
                    for part in path.removeprefix("$.").split("."):
                        value = value[part]
                    if isinstance(value, str):
                        lengths.append(len(value))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return lengths

    def render(self) -> str:
        records = self.valid_records
        nonblank = self.total_lines - self.blank_lines
        top_schema_count = len(self.object_keysets["$"])
        segment_items = self.object_instances["$.segments[]"]
        candidate_items = self.object_instances["$.teacher_signals.candidate_segments[]"]
        timeline_items = self.object_instances["$.teacher_signals.timeline[]"]
        trajectory_items = self.object_instances["$.teacher_signals.trajectory[]"]
        crop_keyframe_items = self.object_instances["$.crop_keyframes[]"]
        threshold, precision, recall, f1, predicted, positives = self.best_timeline_threshold()
        exact_one_second = sum(math.isclose(delta, 1.0, abs_tol=1e-9) for delta in self.timeline_deltas)
        lines = [
            "# AIC-VideoHighlight 训练标注结构分析",
            "",
            "## 1. 分析目的",
            "",
            "本报告用于明确赛事训练标注 `train.jsonl` 的真实 schema、字段含义及其对 Highlight Retrieval 评估的可用性。分析覆盖整个文件；原文件只读，不导出完整标注正文或可识别的视频样例。",
            "",
            "复现命令：",
            "",
            "```bash",
            "python scripts/inspect_train_annotations.py \\",
            f"  --input {self.source} \\",
            "  --output docs/train-annotation-analysis.md",
            "```",
            "",
            "## 2. 数据基本信息",
            "",
            f"- 文件：`{self.source}`",
            f"- 文件大小：{self.size_bytes:,} bytes（{self.size_bytes / 1024 / 1024:.2f} MiB）",
            f"- SHA256：`{self.sha256}`",
            f"- 总行数：{self.total_lines}",
            f"- 空行数：{self.blank_lines}",
            f"- 非空行数：{nonblank}",
            f"- 合法顶层 JSON 对象：{records}",
            f"- JSON/顶层类型异常：{len(self.invalid_lines)}",
            f"- 顶层字段集合种类：{top_schema_count}；{'完全一致' if top_schema_count == 1 else '存在差异'}",
            "- 说明：字段值为空与 schema 缺字段是两回事；下文分别统计。",
            "",
            "## 3. 顶层字段结构",
            "",
            "以下覆盖率分母为全部合法记录。",
            "",
            *self.schema_table("$"),
            "",
            "顶层字段全集：`" + "`, `".join(sorted(self.object_key_presence["$"])) + "`。",
            "",
            "顶层 schema 异常：" + ("未发现。" if top_schema_count == 1 else f"发现 {top_schema_count} 种字段集合。"),
            "",
            "## 4. 核心字段详细结构",
            "",
            f"- `schema_version`：{_counter(self.scalar_values['schema_version'])}。",
            f"- `annotation_mode`：{_counter(self.scalar_values['annotation_mode'])}。",
            f"- `dataset_split`：{_counter(self.scalar_values['dataset_split'])}；`targetRatioWH`：{_counter(self.scalar_values['targetRatioWH'])}。",
            f"- `clip` 固定字段结构见附录；`free_axis`：{_counter(self.scalar_values['clip.free_axis'])}；clip 时长分布：{_summary(end - start for start, end in zip(self.numeric_values['$.clip.start_sec'], self.numeric_values['$.clip.end_sec']))}。",
            f"- `segments`：数组；长度分布为 {_summary(self.list_lengths['$.segments'])}；元素对象共 {segment_items} 个。",
            f"- `crop_keyframes`：数组；长度分布为 {_summary(self.list_lengths['$.crop_keyframes'])}；空数组 {self.node_empty['$.crop_keyframes']} 条；元素对象共 {crop_keyframe_items} 个。",
            f"- `teacher_signals`：对象，覆盖 {self.object_key_presence['$']['teacher_signals']}/{records}；固定包含 `timeline`、`candidate_segments`、`summary`、`trajectory`。",
            f"- `provenance`：对象，覆盖 {self.object_key_presence['$']['provenance']}/{records}，记录数据来源、seed 模型、prompt/config 指纹、采样率与视频哈希。",
            f"- `quality`：对象，覆盖 {self.object_key_presence['$']['quality']}/{records}，包含接受状态、综合分数、置信度、覆盖率及重复一致性指标。",
            "",
            "## 5. teacher_signals 详细分析",
            "",
            *self.schema_table("$.teacher_signals"),
            "",
            f"- `timeline` 元素数：{timeline_items}；长度分布：{_summary(self.list_lengths['$.teacher_signals.timeline'])}。",
            f"- `candidate_segments`：{self.node_empty['$.teacher_signals.candidate_segments']}/{records} 条为空数组，元素总数 {candidate_items}。",
            f"- `summary`：非空 {records - self.node_empty['$.teacher_signals.summary']}/{records}；字符长度分布：{_summary(self._string_lengths('$.teacher_signals.summary'))}。报告不复制摘要正文。",
            f"- `trajectory` 元素数：{trajectory_items}；长度分布：{_summary(self.list_lengths['$.teacher_signals.trajectory'])}。",
            "",
            "`trajectory` 元素结构：",
            "",
            *self.schema_table("$.teacher_signals.trajectory[]"),
            "",
            "- `teacher_signals` 这一命名，加上 `provenance.seed_model` 与 prompt 指纹，直接证明文件显式保存了一组 teacher/seed 模型信号；但字段名本身不能证明是否经过人工复核。",
            "",
            "## 6. segments 详细分析",
            "",
            *self.schema_table("$.segments[]"),
            "",
            f"- 数量：共 {segment_items} 个；每条记录数组长度 {_summary(self.list_lengths['$.segments'])}；空数组 {self.node_empty['$.segments']} 条。",
            f"- `start_sec`：{_summary(self.numeric_values['$.segments[].start_sec'])}。",
            f"- `end_sec`：{_summary(self.numeric_values['$.segments[].end_sec'])}。",
            f"- 区间时长 `end_sec-start_sec`：{_summary(self.segment_durations)}。",
            f"- `start_frame`：{_summary(self.numeric_values['$.segments[].start_frame'])}；`end_frame`：{_summary(self.numeric_values['$.segments[].end_frame'])}。",
            f"- seed 分数：`min_seed_score` {_summary(self.numeric_values['$.segments[].min_seed_score'])}；`mean_seed_score` {_summary(self.numeric_values['$.segments[].mean_seed_score'])}；`max_seed_score` {_summary(self.numeric_values['$.segments[].max_seed_score'])}。",
            f"- 时间起止合法：{self.segment_valid_time}/{segment_items}；帧起止合法：{self.segment_valid_frame}/{segment_items}。",
            f"- 局部区间位于 `0..(clip.end_sec-clip.start_sec)` 内：{self.segment_within_clip}/{self.segment_clip_comparable}。",
            f"- 超出 clip 局部终点的区间：{len(self.segment_clip_overshoots)}；最大超出 {_number(max(self.segment_clip_overshoots, default=0) * 1000)} ms，属于边界舍入量级。",
            f"- `source_start/end == clip.start_sec + local start/end`：{self.segment_source_equal}/{self.segment_source_comparable}。",
            f"- 0 秒时长区间：{sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in self.segment_durations)}；0 帧跨度区间：{sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in self.segment_frame_spans)}。这是值级边界案例，不是 schema 变体。",
            "- 结构结论：它明确包含秒级起止、帧级起止、区间 ID、三种 seed 分数及 source 起止；不包含独立的 `frame`、`score` 或显式 `highlight_label` 字段。数组中的区间本身可被工程上解释为“被选中的片段”，但这不是官方 GT 身份证明。",
            "",
            "匿名化结构示例：",
            "",
            "```json",
            '{"segments":[{"segment_id":"segment_N","start_sec":0.0,"end_sec":1.0,"start_frame":0,"end_frame":30,"min_seed_score":0.0,"mean_seed_score":0.0,"max_seed_score":0.0,"source_start_sec":0.0,"source_end_sec":1.0}]}',
            "```",
            "",
            "## 7. candidate_segments 详细分析",
            "",
            f"- 字段类型：`array`×{records}；记录覆盖率 100%。",
            f"- 空数组：{self.node_empty['$.teacher_signals.candidate_segments']}/{records}（{_percentage(self.node_empty['$.teacher_signals.candidate_segments'], records)}）。",
            f"- 元素总数：{candidate_items}。由于没有任何元素，数据无法证明候选元素应有哪些键、时间单位或分数字段。",
            f"- 与 `segments` 数量完全相等的记录：{self.candidate_count_equal_records}/{records}；区间列表精确相等：{self.candidate_exact_records}/{records}；双方均非空：{self.candidate_both_nonempty_records}/{records}。",
            "- 因此，本文件中二者不是高度一致的两份区间：`segments` 非空而 `candidate_segments` 全空。不能计算有意义的时间 IoU/覆盖率，也不能把空候选字段当作 GT。",
            "",
            "## 8. timeline 详细分析",
            "",
            *self.schema_table("$.teacher_signals.timeline[]"),
            "",
            f"- 时间点总数：{timeline_items}；每条记录长度 {_summary(self.list_lengths['$.teacher_signals.timeline'])}。",
            f"- `time_sec`：{_summary(self.numeric_values['$.teacher_signals.timeline[].time_sec'])}。",
            f"- 相邻正时间差：{_summary(self.timeline_deltas)}；其中恰为 1 秒 {exact_one_second}/{len(self.timeline_deltas)}（{_percentage(exact_one_second, len(self.timeline_deltas))}）。",
            f"- 时间非严格递增记录：{self.timeline_nonmonotonic_records}/{records}；非末尾位置的非 1 秒间隔：{self.timeline_non_unit_nonfinal_deltas}。末尾可以是非整数视频终点，因此它是约 1 Hz 的秒级采样点序列，而不是逐帧标签或固定长度时间块。",
            f"- `highlight_score`：{_summary(self.numeric_values['$.teacher_signals.timeline[].highlight_score'])}。",
            f"- `confidence`：{_summary(self.numeric_values['$.teacher_signals.timeline[].confidence'])}。",
            f"- 位于 `segments` 内的 timeline 点：{self.timeline_points_in_segments}/{self.timeline_points}（{_percentage(self.timeline_points_in_segments, self.timeline_points)}）。",
            f"- 区间内 `highlight_score`：{_summary(self.timeline_score_inside)}；区间外：{_summary(self.timeline_score_outside)}。",
            f"- 在仅测试 0.05 步长阈值时，最佳点级阈值为 `score >= {_number(threshold)}`：precision={precision:.3f}、recall={recall:.3f}、F1={f1:.3f}（预测正点 {predicted}，区间内点 {positives}）。这只是对现有 `segments` 的拟合，不是官方规则。",
            f"- schema 预留 `phase` 和 `description` 语义字段，但本文件二者分别为空 {self.node_empty['$.teacher_signals.timeline[].phase']}/{timeline_items}、{self.node_empty['$.teacher_signals.timeline[].description']}/{timeline_items}，没有可用逐点文本语义。",
            "",
            "## 9. provenance 分析",
            "",
            *self.schema_table("$.provenance"),
            "",
            f"- `source`：{_counter(self.scalar_values['provenance.source'])}。",
            f"- `seed_model`：{_counter(self.scalar_values['provenance.seed_model'])}。",
            f"- `prompt_version`：{_counter(self.scalar_values['provenance.prompt_version'])}。",
            f"- `prompt_fingerprint`：{len(self.scalar_values['provenance.prompt_fingerprint'])} 个唯一值，覆盖 {sum(self.scalar_values['provenance.prompt_fingerprint'].values())}/{records}。",
            f"- `annotator_config_sha256`：{len(self.scalar_values['provenance.annotator_config_sha256'])} 个唯一值。",
            f"- `source_group`：{len(self.scalar_values['provenance.source_group'])} 个唯一值；覆盖 {sum(self.scalar_values['provenance.source_group'].values())}/{records}。不列出可识别的具体值。",
            f"- `n_seed_repeats`：{_summary(self.numeric_values['$.provenance.n_seed_repeats'])}。",
            f"- `temporal_fps`：{_summary(self.numeric_values['$.provenance.temporal_fps'])}；`spatial_fps`：{_summary(self.numeric_values['$.provenance.spatial_fps'])}。",
            f"- `video_sha256` 唯一值数：{len(self.scalar_values['provenance.video_sha256'])}；不在报告中列出具体视频哈希。",
            f"- `license`：空字符串 {self.node_empty['$.provenance.license']}/{records}；当前文件本身没有给出许可文本。",
            "- 能证明的内容：记录声明了来源数据集、seed 模型、prompt/config 版本或指纹和采样参数，且 schema 名称为 seed weak training label。",
            "- 不能证明的内容：这些字段不包含人工审核者身份、人工复核状态、官方 GT 标记或标注流程文档，因而不能单独证明人工/模型/混合生成的最终归属。",
            "- 人工标注判断：仅依据本文件，没有任何字段可被确认或高概率归类为人工标注；`clip.qvh_window` 等可能继承自源数据集的字段也缺少逐字段来源说明。",
            "",
            "## 10. crop_keyframes 分析",
            "",
            *self.schema_table("$.crop_keyframes[]"),
            "",
            f"- 非空记录：{records - self.node_empty['$.crop_keyframes']}/{records}；元素总数 {crop_keyframe_items}；长度 {_summary(self.list_lengths['$.crop_keyframes'])}。",
            f"- 时间点落入 `segments`：{self.crop_keyframes_in_segments}/{self.crop_keyframe_points}（{_percentage(self.crop_keyframes_in_segments, self.crop_keyframe_points)}）。这说明 keyframe 通常覆盖视频采样时间轴，而非仅高光区间。",
            f"- 与 `trajectory` 长度相同：{self.keyframe_trajectory_length_equal_records}/{self.keyframe_trajectory_comparable_records} 条；对齐 pair {self.keyframe_trajectory_pairs} 个。",
            f"- 对齐 pair 中：时间相同 {self.keyframe_trajectory_time_equal}/{self.keyframe_trajectory_pairs}，`raw_center == center` {self.keyframe_trajectory_center_equal}/{self.keyframe_trajectory_pairs}，`visible` 相同 {self.keyframe_trajectory_visible_equal}/{self.keyframe_trajectory_pairs}。",
            f"- `cropRois` 元素 {self.crop_roi_items} 个，满足 `[frame,[x,y,w,h]]` 结构 {self.crop_roi_well_formed}/{self.crop_roi_items}；ROI 帧位于 segment 帧范围 {self.crop_roi_frames_in_segments}/{self.crop_roi_well_formed}。",
            f"- 在可比较记录中，`cropRois` 帧集合恰好覆盖全部 segment 整数帧范围：{self.crop_roi_exact_segment_frame_records}/{self.crop_roi_comparable_records}。",
            f"- 空 `cropRois` 且 `quality.spatial_source=dropped_center_default`：{self.empty_crop_matches_dropped_default}/{self.node_empty['$.cropRois']}。",
            "- 工程含义：`crop_keyframes`/`trajectory` 描述稀疏主体中心与置信度，`cropRois` 给出选中区间内逐帧裁剪框；它们适合后续空间构图任务，但不应反向当作 Highlight Retrieval 的独立时序 GT。",
            "",
            "## 11. Highlight Retrieval 可用标签分析",
            "",
            "### 候选方案 A：使用 `segments`",
            "",
            "- 证据：全量非空；具有明确秒级和帧级边界；含 seed 分数；与逐帧 `cropRois` 高度关联。",
            "- 优点：可直接构造 temporal reference segments，无需自定义阈值。",
            "- 风险：`schema_version`、`seed_model`、`prompt_fingerprint`、`min/mean/max_seed_score` 显示其很可能是 seed/teacher 弱标签；没有官方 GT 标记或人工审核链路。",
            "- 当前可信度：三种方案中最高，适合作为“训练弱 reference / 内部开发 reference”，暂不能称赛事官方 Ground Truth。",
            "",
            "### 候选方案 B：使用 `teacher_signals.candidate_segments`",
            "",
            "- 证据：字段在所有记录中存在。",
            "- 优点：名称在未来数据版本中可能承载 teacher 原始候选。",
            "- 风险：本文件 100% 为空，元素 schema 也无法确认。",
            "- 当前可信度：不可用。",
            "",
            "### 候选方案 C：从 `timeline.highlight_score` 阈值化构造",
            "",
            f"- 证据：timeline 全量存在，约 1 Hz，含 `highlight_score` 与 `confidence`；粗网格最佳拟合阈值为 {_number(threshold)}，F1={f1:.3f}。",
            "- 优点：保留连续软分数，可调整 Recall/Precision，也可用于点级蒸馏或 ranking。",
            "- 风险：阈值、点到区间的边界扩展、短间隔合并和尾点处理均无官方定义；秒级采样也弱于已有帧级边界。",
            "- 当前可信度：适合作为辅助监督与敏感性分析，不宜替代 `segments` 作为首选 reference。",
            "",
            "结论：若下一步必须选择一个内部 Highlight Retrieval reference，`segments` 最可信；命名应明确为 seed/teacher-derived weak reference。最终 GT 身份仍需官方说明或人工抽样核验。",
            "",
            "## 12. 已确认事实",
            "",
            f"- 文件包含 {records} 条合法记录，0 空行，{len(self.invalid_lines)} 条解析/顶层类型异常，顶层 schema 完全一致。",
            "- 所有记录都包含 `segments`、`crop_keyframes`、`teacher_signals`、`provenance` 和 `quality`。",
            f"- `segments` 共 {segment_items} 个且每条记录至少一个；包含时间、帧、segment ID 和 seed 分数，不包含显式 `highlight_label`。",
            f"- `candidate_segments` 在 {records}/{records} 条中为空。",
            "- `timeline` 是按 `time_sec` 排列的约 1 Hz 采样序列，包含 `highlight_score` 和 `confidence`。",
            f"- `timeline` 与 `trajectory` 长度相同 {self.timeline_trajectory_length_equal_records}/{records} 条，时间序列完全相同 {self.timeline_trajectory_time_equal_records}/{records} 条。",
            f"- provenance 的 `source` 分布为：{_counter(self.scalar_values['provenance.source'])}。",
            f"- provenance 的 `seed_model` 分布为：{_counter(self.scalar_values['provenance.seed_model'])}。",
            "",
            "## 13. 高概率判断",
            "",
            "- `teacher_signals.timeline/summary/trajectory` 明显是 teacher/seed 模型输出或其规范化结果。依据是字段命名、seed 模型、prompt 版本/指纹和 provenance 同时存在。",
            "- `segments` 高概率由 timeline/seed 分数再经过阈值、连通区间和边界处理生成，而非独立人工 GT；其字段直接命名为 `*_seed_score`。",
            "- `quality` 高概率是自动质量门控/聚合产物，尤其是 coverage、score spread、threshold margin 和 repeat agreement。",
            "- `crop_keyframes` 高概率由 teacher trajectory 规范化而来；两者时间、中心和可见性可直接量化对应。",
            "",
            "## 14. 合理推测",
            "",
            "- `cropRois` 可能由稀疏 `crop_keyframes`/trajectory 插值后，仅在 `segments` 选中帧范围内展开，用于后续竖屏裁剪或构图。",
            "- `source=QVHighlights` 可能表示本训练文件以 QVHighlights 视频/窗口为底座，再由 seed 模型生成赛事任务所需的 generic highlight 与空间弱标签。",
            "- `clip.qvh_window`、`used_window` 与 planned/achieved ratio 可能记录从源窗口到当前训练 clip 的选段规划过程，但缺少生成器说明，不能作为事实。",
            "",
            "## 15. 当前无法确认事项",
            "",
            "- `segments` 是否被赛事官方定义为最终 reference annotation / Ground Truth。需要官方 README、字段说明或评测代码确认。",
            "- 任何字段是否经过人工审核、修订或验收。需要人工标注流程、审核日志或明确的 provenance 字段。",
            "- `clip.qvh_window`、`used_window`、比例字段及源数据元信息分别属于人工、原始数据集还是自动生成。需要字段级 lineage/生成流程文档。",
            "- `candidate_segments` 的预期元素 schema 与为何全空。需要数据生成代码或 schema 文档。",
            "- 从 timeline 到 segments 的精确阈值、插值、合并与边界规则。需要生成器实现/config；点级拟合不能证明生成规则。",
            "- `source_start_sec/source_end_sec` 的官方命名意图。数据中它们严格等于 `clip.start_sec + local start/end`，但仍需字段文档确认其坐标系定义。",
            "",
            "## 16. 推荐下一步",
            "",
            "1. 查阅赛事官方 README、训练数据说明、schema 定义与评测脚本，确认 reference 字段和时间边界约定。",
            "2. 从不同 `annotation_mode`、quality 状态和 segment 数量桶中抽样少量训练视频，人工核验 `segments` 是否确实覆盖语义高光。",
            "3. 定位生成 `seed_weak_training_label_v1` 的代码/config，用 `prompt_fingerprint` 与 `annotator_config_sha256` 对应具体版本。",
            "4. 将 `segments` 暂时命名为 `weak_reference_segments`，同时保留 timeline 软分数；在官方确认前避免在指标或文档中写作 Ground Truth。",
            "5. 对 timeline 阈值做敏感性分析时按 source/video 分组，防止同源窗口泄漏；本轮不运行正式 benchmark。",
            "",
            "## 17. 对 Highlight Retrieval Pipeline 的工程建议",
            "",
            "- Reader 层显式支持 `reference_policy=segments|timeline_threshold|none`，默认值在官方语义确认前不要暗示 official GT。",
            "- 训练可同时使用 `segments` 的区间监督与 timeline 的软分数/置信度，但在 loss 中区分 weak label 与人工 label。",
            "- 评估时记录 reference 来源、schema_version、seed_model、prompt_fingerprint 和阈值配置，保证结果可复现。",
            "- `candidate_segments` 为空时应明确报不可用，不能静默回退并声称同一指标定义。",
            "- `crop_keyframes`、trajectory 与 `cropRois` 留给空间裁剪阶段；不要让空间字段改变本轮 Highlight Retrieval 的 reference 语义。",
            "- QVHighlights 是 query-conditioned highlight detection 数据；本赛事目标更接近 generic highlight selection。可将 QVHighlights 作为后续辅助训练/泛化来源，但不能把 QVHighlights GT 直接等同赛事 GT。本轮不下载该数据集。",
            "",
            "---",
            "",
            "### 附录 A：quality 概览",
            "",
            *self.schema_table("$.quality"),
            "",
            f"- `status`：{_counter(self.scalar_values['quality.status'])}。",
            f"- `annotation_mode`：{_counter(self.scalar_values['quality.annotation_mode'])}。",
            f"- `spatial_source`：{_counter(self.scalar_values['quality.spatial_source'])}。",
            f"- `score`：{_summary(self.numeric_values['$.quality.score'])}。",
            f"- `selected_ratio`：{_summary(self.numeric_values['$.quality.selected_ratio'])}。",
            f"- `temporal_confidence`：{_summary(self.numeric_values['$.quality.temporal_confidence'])}。",
            f"- `spatial_confidence`：{_summary(self.numeric_values['$.quality.spatial_confidence'])}。",
            f"- `timeline_coverage`：{_summary(self.numeric_values['$.quality.timeline_coverage'])}。",
            f"- `trajectory_coverage`：{_summary(self.numeric_values['$.quality.trajectory_coverage'])}。",
            f"- `repeat_agreement`：{_summary(self.numeric_values['$.quality.repeat_agreement'])}。",
            f"- `score_spread`：{_summary(self.numeric_values['$.quality.score_spread'])}。",
            f"- `threshold_margin`：{_summary(self.numeric_values['$.quality.threshold_margin'])}。",
            f"- `reasons` 空数组：{self.node_empty['$.quality.reasons']}/{records}；非空元素计数：{sum(self.node_types['$.quality.reasons[]'].values())}。",
            "",
            "### 附录 B：schema 变体与异常",
            "",
        ]
        for path in (
            "$",
            "$.clip",
            "$.segments[]",
            "$.crop_keyframes[]",
            "$.teacher_signals",
            "$.teacher_signals.timeline[]",
            "$.teacher_signals.trajectory[]",
            "$.provenance",
            "$.quality",
        ):
            variants = self.object_keysets[path]
            lines.append(
                f"- `{path}`：对象 {self.object_instances[path]} 个，字段集合 {len(variants)} 种；"
                + ("一致。" if len(variants) <= 1 else "存在 schema 变体。")
            )
        if self.invalid_lines:
            lines.extend(["", "解析异常（最多 20 条，仅列行号和错误）："])
            lines.extend(f"- 第 {line} 行：{message}" for line, message in self.invalid_lines)
        else:
            lines.extend(["", "未发现 JSON 解析异常或非对象顶层记录。"])
        lines.append("")
        return "\n".join(lines)


def inspect_jsonl(source: Path) -> str:
    analysis = Analysis(source)
    analysis.load()
    return analysis.render()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fully inspect train.jsonl without modifying or copying annotation payloads"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "docs" / "train-annotation-analysis.md",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input JSONL does not exist: {args.input}")
    report = inspect_jsonl(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Analysis report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
