"""Tests for versioned high-recall retrieval prompts."""

import pytest

from aic_video_highlight.highlight_retrieval.pipeline import HighlightRetrievalConfig
from aic_video_highlight.highlight_retrieval.prompt_builder import (
    PROMPT_VERSION,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V3,
    PROMPT_VERSION_V4,
    SUPPORTED_PROMPT_VERSIONS,
    build_high_recall_prompt,
    build_high_recall_prompt_v1,
    build_high_recall_prompt_v2,
    build_high_recall_prompt_v3,
    build_high_recall_prompt_v4,
    build_prompt,
)


def test_v0_prompt_is_unchanged():
    prompt = build_high_recall_prompt(30.0, 5)
    assert f"Prompt 版本：{PROMPT_VERSION}" in prompt
    assert "高召回优先" in prompt
    assert "信息量明显提升" in prompt
    assert "视觉表现突出" in prompt
    assert "镜头或场景显著变化" in prompt
    assert "尽可能返回多个候选，最多 5 个" in prompt
    assert "30.000" in prompt
    assert '{"has_highlight":true,"segments":[{"start_sec":4.2,"end_sec":8.6,"score":0.86,"reason":"简短原因"}]}' in prompt
    assert '{"has_highlight":false,"segments":[]}' in prompt


def test_v1_prompt_contains_core_rules():
    prompt = build_high_recall_prompt_v1(30.0, 5)
    assert f"Prompt 版本：{PROMPT_VERSION_V1}" in prompt
    assert "有意思不等于值得成片" in prompt
    assert "动作峰值" in prompt
    assert "显著结果" in prompt
    assert "关键信息峰值" in prompt
    assert "明显审美峰值" in prompt
    assert "单纯场景切换" in prompt
    assert "铺垫过程" in prompt
    assert "重复互动" in prompt
    assert "局部显著性" in prompt
    assert "最短充分高光区间" in prompt
    assert "召回保护" in prompt
    assert "不确定就把整段视频保留" in prompt
    assert "空结果政策" in prompt
    assert "第三步的排除规则只用于在该事件内部收缩明显无效的首尾" in prompt
    assert "score 含义" in prompt
    assert "最多 5 个" in prompt
    assert "30.000" in prompt
    assert '{"has_highlight":true,"segments":[{"start_sec":4.2,"end_sec":8.6,"score":0.86,"reason":"简短原因"}]}' in prompt
    assert '{"has_highlight":false,"segments":[]}' in prompt


def test_v2_prompt_contains_core_rules():
    prompt = build_high_recall_prompt_v2(30.0, 5)
    assert f"Prompt 版本：{PROMPT_VERSION_V2}" in prompt
    assert "粗召回" in prompt
    assert "漏掉真实事件的代价高于保留少量多余候选" in prompt
    assert "不要求出现剧烈动作、高潮、冲突或巨大变化" in prompt
    assert "不能单独证明一个区间值得成为高光" in prompt
    assert "不要机械按照内容类型删除" in prompt
    assert "不要只截取一个峰值瞬间" in prompt
    assert "不要把事件本身裁掉" in prompt
    assert "分别输出多个候选" in prompt
    assert "没有明显的局部峰值不等于没有高光" in prompt
    assert "score 表示该候选最终值得进入成片的置信度" in prompt
    assert "最多 5 个" in prompt
    assert "30.000" in prompt
    assert '{"has_highlight":true,"segments":[{"start_sec":4.2,"end_sec":8.6,"score":0.86,"reason":"简短原因"}]}' in prompt
    assert '{"has_highlight":false,"segments":[]}' in prompt
    assert "不要因为\"不够强\"返回空" in prompt


def test_v3_prompt_adds_anti_truncation_rules():
    prompt = build_high_recall_prompt_v3(30.0, 5)
    assert f"Prompt 版本：{PROMPT_VERSION_V3}" in prompt
    assert "reason 简短（一两句话以内）" in prompt
    assert "确保 JSON 完整输出、不被截断" in prompt
    assert "粗召回" in prompt
    assert build_high_recall_prompt_v3(30.0, 5) != build_high_recall_prompt_v2(30.0, 5)


def test_v4_prompt_minimal_delta_from_v2():
    prompt = build_high_recall_prompt_v4(30.0, 5)
    assert f"Prompt 版本：{PROMPT_VERSION_V4}" in prompt
    assert "reason 简短，一句话以内" in prompt
    assert "粗召回" in prompt
    assert "单行紧凑" not in prompt
    assert build_high_recall_prompt_v4(30.0, 5) != build_high_recall_prompt_v2(30.0, 5)


def test_build_prompt_dispatch():
    assert build_prompt(PROMPT_VERSION, 30.0) == build_high_recall_prompt(30.0)
    assert build_prompt(PROMPT_VERSION_V1, 30.0) == build_high_recall_prompt_v1(30.0)
    assert build_prompt(PROMPT_VERSION_V2, 30.0) == build_high_recall_prompt_v2(30.0)
    assert build_prompt(PROMPT_VERSION_V3, 30.0) == build_high_recall_prompt_v3(30.0)
    assert build_prompt(PROMPT_VERSION_V4, 30.0) == build_high_recall_prompt_v4(30.0)
    with pytest.raises(ValueError):
        build_prompt("high_recall_retrieval_v99", 30.0)


def test_supported_versions():
    assert SUPPORTED_PROMPT_VERSIONS == (
        PROMPT_VERSION,
        PROMPT_VERSION_V1,
        PROMPT_VERSION_V2,
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
    )


@pytest.mark.parametrize("kwargs", [
    {"prompt_version": "high_recall_retrieval_v0"},
    {"prompt_version": "high_recall_retrieval_v1"},
    {"prompt_version": "high_recall_retrieval_v2"},
    {"prompt_version": "high_recall_retrieval_v3"},
    {"prompt_version": "high_recall_retrieval_v4"},
])
def test_config_accepts_supported_prompt_versions(kwargs):
    config = HighlightRetrievalConfig(**kwargs)
    assert config.prompt_version == kwargs["prompt_version"]


def test_config_rejects_unknown_prompt_version():
    with pytest.raises(ValueError):
        HighlightRetrievalConfig(prompt_version="high_recall_retrieval_v99")


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        build_high_recall_prompt(0, 5)
    with pytest.raises(ValueError):
        build_high_recall_prompt(30.0, 0)
    with pytest.raises(ValueError):
        build_high_recall_prompt_v1(0, 5)
    with pytest.raises(ValueError):
        build_high_recall_prompt_v1(30.0, 0)
    with pytest.raises(ValueError):
        build_high_recall_prompt_v2(0, 5)
    with pytest.raises(ValueError):
        build_high_recall_prompt_v2(30.0, 0)
