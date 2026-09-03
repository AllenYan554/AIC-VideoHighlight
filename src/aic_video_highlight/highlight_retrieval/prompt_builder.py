"""Versioned prompts for high-recall highlight candidate retrieval."""

from __future__ import annotations


PROMPT_VERSION = "high_recall_retrieval_v0"


def build_high_recall_prompt(chunk_duration_sec: float, max_segments: int = 5) -> str:
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be greater than zero")
    if max_segments <= 0:
        raise ValueError("max_segments must be greater than zero")
    return f"""你是通用视频高光剪辑系统中的候选高光粗召回模块。

目标不是直接决定最终成片，而是尽可能不要遗漏潜在高光事件。高召回优先，允许召回不确定候选。

综合考虑：
1. 明显动作和行为变化
2. 事件高潮、转折或结果
3. 人物明显情绪和互动变化
4. 信息量明显提升
5. 视觉表现突出
6. 镜头或场景显著变化
7. 完整且具有观看价值的小事件

不要将长时间静止、黑屏、严重模糊、无信息变化或明显重复内容无条件作为高光。候选时间段应尽量覆盖完整事件，并尽可能返回多个候选，最多 {max_segments} 个。

时间规则：视频片段时长为 {chunk_duration_sec:.3f} 秒。start_sec 和 end_sec 必须是当前片段内部的相对秒数，范围为 0 到 {chunk_duration_sec:.3f}。不要输出帧号或整段原视频的全局时间。

仅输出一个 JSON 对象，不要输出 Markdown 或解释。存在候选时格式为：
{{"has_highlight":true,"segments":[{{"start_sec":4.2,"end_sec":8.6,"score":0.86,"reason":"简短原因"}}]}}

没有候选时格式为：
{{"has_highlight":false,"segments":[]}}

Prompt 版本：{PROMPT_VERSION}
"""
