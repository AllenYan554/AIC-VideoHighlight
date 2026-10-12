"""构建 Stage 7.17 无 GT 的协议、准入工具模板与真实资源缺口报告。

只读取指定历史记录及公开数据集身份元数据，不读取任何 GT 数值或模型张量。
拒绝覆盖实验目录；运行缓存仅在 E 盘。本脚本没有下载或推理入口。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from aic_video_highlight import open_domain_highlights as H

RECORDS = Path(r"D:\CDUT\硕士\2026-09 AIC\实验记录")
BASE = RECORDS / "Stage7_VHiCraFTNet研发_2"
CANONICAL = BASE / "10_Local_Canonical_Visual_Baseline_202610021452"
TVSUM = BASE / "23_TVSum_ZeroShot_Generalization_202610081744"
REVIEW = RECORDS / "VHiCraFTNet_全实验系统性复盘_截止到10_8.md"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, encoding="utf-8", errors="replace")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def build(args) -> dict:
    record = H.contained(args.record, BASE)
    H.require(record.name.startswith("24_OpenDomain_Highlight_Evaluation_"), "实验目录命名不匹配")
    data = H.DATA_ROOT
    cache = H.contained(args.cache, Path(r"E:\ResearchData\cache\VHiCraFTNet"))
    H.require(not record.exists() and not data.exists(), "拒绝覆盖已有实验或数据方案目录")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["TMP"] = str(cache / "tmp")
    os.environ["TEMP"] = str(cache / "tmp")
    (cache / "tmp").mkdir(exist_ok=True)

    checkpoint_doc = H.read_json(TVSUM / "config/checkpoint_manifest.json")
    canonical_doc = H.read_json(CANONICAL / "results/canonical_baseline_manifest.json")
    expected = {r["seed"]: r["checkpoint_sha256"] for r in canonical_doc["per_seed"]}
    H.require({r["seed"]: r["sha256"] for r in checkpoint_doc["checkpoints"]} == expected,
              "Stage7.16 checkpoint 引用与 Stage7.10F 不一致")
    checked = H.verify_checkpoints(checkpoint_doc)
    model_code = CANONICAL / "supplementary/canonical_lib.py"
    model_sha = H.file_sha256(model_code)
    H.require(model_sha == "fc10edac86ff5f313ce76e21b5614bc39d274fa9e350f16057dde895e46eaa59",
              "冻结 Canonical 模型代码身份变化")

    yt = Path(r"E:\ResearchData\datasets\youtube_highlights\manifests")
    split = H.read_json(yt / "stage7_ftnet_split_manifest.json")
    exclusions = H.yt_exclusions(split, jsonl(yt / "candidate_manifest.jsonl"), jsonl(yt / "availability_manifest.jsonl"))
    exclusions["source_sha256"] = sorted(set(exclusions["source_sha256"]) |
        {r["sha256"] for r in jsonl(yt / "raw_manifest.jsonl") if H.HEX.fullmatch(r.get("sha256", ""))})
    exclusions["metadata_provenance"] = {name: H.file_sha256(yt / name) for name in
        ("stage7_ftnet_split_manifest.json", "candidate_manifest.jsonl", "availability_manifest.jsonl", "raw_manifest.jsonl")}

    domains = {"daily_vlog": 8, "people_events": 8, "arts_performance": 8, "making_education": 8,
               "animals_nature": 8, "travel_outdoors": 8, "news_products": 8, "animation_fiction": 8}
    protocol = {"schema": H.SCHEMA + ".protocol/v1", "stage": "7.17",
                "selected_dataset": "ODH_HUMAN_PILOT_V1", "status": "DESIGN_FROZEN_DATA_NOT_ADMITTED",
                "label_semantics": "DIRECT_HUMAN_BINARY_HIGHLIGHT", "no_query": True,
                "bin_seconds": 2.0, "tail": "保留最后不足两秒的区间，不丢弃",
                "n_raters": 3, "disagreement": "不取平均评分或多数阈值；分歧及不确定由独立第四人仲裁",
                "target_evaluation_videos": 64, "minimum_evaluation_videos": 50,
                "domain_targets": domains, "minimum_per_domain": 5,
                "annotation_practice_videos": 8, "practice_split": "ANNOTATION_PRACTICE_ONLY，独立于评价样本",
                "sampling_seed": 20261008, "source_policy": "原视频及人工标注使用权限需逐条证明；严格排除 AIC/QVHighlights 资产",
                "identity_policy": "原视频 SHA256、全部来源别名、内容家族复核；全已登记 YT 来源排除",
                "split": "EXTERNAL_TEST_ONLY，无训练/校准/模型选择",
                "universe": "完整视频全部两秒区间，不使用 Qwen candidate 或训练 loss_mask 筛选",
                "metric": "直接二元人工 GT 的每视频 AP；八 seed AP 平均后视频等权 macro AP",
                "tie_rule": "相同预测分数在同一阈值组计算，禁止排序次序获益",
                "no_positive_policy": "AP=NA，单列数量及原因，不填零、不删除视频记录",
                "all_positive_policy": "AP=1，单列数量；不以结果为由删样本",
                "bootstrap_samples": 10000, "bootstrap_seed": 20261008,
                "bootstrap_unit": "按领域分层、视频级重采样；八 seed 在每视频内固定",
                "alignment": "源 PTS，2 FPS；首尾覆盖误差≤0.25/0.75s，间隔(0,0.75]s，每 GT bin 非空；均值聚合",
                "model": {"name": "LOCAL_CANONICAL_VISUAL_BASELINE_v1", "seeds": list(H.SEEDS),
                          "model_code": str(model_code), "model_code_sha256": model_sha,
                          "audio_used": False, "normalization": "NONE", "whole_video_forward": True,
                          "checkpoint_rule": "历史 min YT VALIDATION masked BCE，保持冻结"},
                "canonical_checkpoints_sha256": H.digest_json(checkpoint_doc),
                "evaluation_code": {"path": str(REPO / "src/aic_video_highlight/open_domain_highlights.py"),
                                    "sha256": H.file_sha256(REPO / "src/aic_video_highlight/open_domain_highlights.py")},
                "cli_code": {"path": str(REPO / "scripts/experiments/stage7/open_domain_highlights.py"),
                             "sha256": H.file_sha256(REPO / "scripts/experiments/stage7/open_domain_highlights.py")},
                "visual256": {"model_id": "PekingU/rtdetr_r50vd", "revision": "df939e661d8c52e80608d1ec566561aabd25a4e7",
                              "tap": "encoder_last_hidden_state[0] GAP", "dtype": "float16", "shape": "[T,256]",
                              "sampling_fps": 2.0, "reuse": "兼容缓存优先；本轮未创建新特征"},
                "freeze_order": "合法素材→标注练习→固定子集/时轴/模型/代码→formal freeze→独立真人标注→仲裁→一次冻结推理与评价",
                "formal_evaluation": "NOT_RUN", "stop": "缺任一准入条件则阻断；不得训练、调参或凭测试结果改协议"}
    manifest = {"schema": H.SCHEMA + ".manifest/v1", "dataset": "ODH_HUMAN_PILOT_V1",
                "status": "BLOCKED_NO_ADMITTED_LEGAL_VIDEO_POOL", "domain_targets": domains,
                "exclusions_sha256": H.digest_json(exclusions), "videos": [],
                "empty_reason": "当前未能证明适合新增人工标注的本地开放领域原视频使用权；不自动下载或把不明素材填入清单"}
    decisions = [
        {"dataset": "YouTube Highlights", "size": 154, "domains": "六类；既有 TRAIN/VAL/CAL", "no_query": "YES",
         "labels": "人工 soft votes，既有 adapter 冻结", "granularity": "约两秒片段",
         "license": "原视频权利仍独立；本轮不新增用途", "media": "已有 RT-DETR Visual256",
         "split": "111/23/20 已冻结；不是独立开放领域测试", "decision": "INELIGIBLE", "reason": "本轮目标需要新的独立领域 GT",
         "source": "https://github.com/aliensunmin/DomainSpecificHighlight"},
        {"dataset": "TVSum", "size": 50, "domains": "十类摘要任务", "no_query": "标题辅助的重要性标注；非本轮无提示高光",
         "labels": "人工摘要重要性 1–5，非直接高光判断", "granularity": "两秒重要性",
         "license": "既存来源记录含 Webscope 协议要求；不自动批准重标注", "media": "已有 Stage7.16 Visual256",
         "split": "既有 EXTERNAL_TEST_ONLY；已用于 Stage7.16", "decision": "INELIGIBLE", "reason": "不能把摘要重要性二值化冒充高光 GT",
         "source": "https://openaccess.thecvf.com/content_cvpr_2015/html/Song_TVSum_Summarizing_Web_2015_CVPR_paper.html"},
        {"dataset": "PHD2", "size": 1500, "domains": "本地原视频；当前可靠内容类别不完整", "no_query": "YES，用户偏好条件",
         "labels": "个人 GIF 选择，非独立通用高光多人标注", "granularity": "秒级选择区间",
         "license": "内部研究未发现禁止但未核实", "media": "1218 条 Visual256，已有筛选偏差",
         "split": "本地 1500 TRAIN-only；不能充当独立人工高光 GT", "decision": "INELIGIBLE", "reason": "偏好语义与使用边界均不满足本轮",
         "source": "https://github.com/gifs/personalized-highlights-dataset"},
        {"dataset": "SVHighlights", "size": 320, "domains": "八类长体育直播", "no_query": "编辑集锦对齐无需自由文本；发布含固定体育 query 特征",
         "labels": "人工编辑集锦经 PSNR 自动对齐与过滤生成二元标签", "granularity": "两秒片段",
         "license": "标注/特征 CC BY-NC 4.0；原视频独立，帧需 research-only 申请", "media": "CLIP/SlowFast/PANN，非兼容 Visual256；没有本地完整原视频证明",
         "split": "官方评价来源；本轮未读取测试标签数值", "decision": "INELIGIBLE", "reason": "可用于编辑高光代理研究，但不等同独立逐区间人工 GT；兼容原视频与权限也缺失",
         "source": "https://github.com/leedongkyu2019/SVHighlights"},
        {"dataset": "Mr. HiSum", "size": 31892, "domains": "3509 多标签类别；121–300 秒", "no_query": "YES",
         "labels": "Most replayed 行为统计代理，不是直接人工区间标注", "granularity": "100 个相对时间区间；对齐到 1 FPS",
         "license": "发布资产 CC BY 4.0；另须遵守 YouTube 相关条款", "media": "YouTube-8M Inception/PCA 特征非 Visual256；本地不存在",
         "split": "官方提供划分；本轮不下载样本或测试标签", "decision": "INELIGIBLE", "reason": "仅可称重播偏好排序；不采用代理 top-percent 二值化 AP",
         "source": "https://proceedings.nips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf"},
        {"dataset": "ODH_HUMAN_PILOT_V1", "size": 0, "domains": "计划八领域×八视频，实际零视频", "no_query": "协议明确 YES；实际标注未执行",
         "labels": "未来三位真人直接二元高光判断及独立仲裁", "granularity": "完整视频两秒分箱",
         "license": "当前无准入素材，逐条原视频与人工标注授权 Gate", "media": "实际零视频、零新特征",
         "split": "计划 EXTERNAL_TEST_ONLY；与标注练习及 YT 来源隔离", "decision": "CONDITIONALLY_ELIGIBLE", "reason": "唯一选定方案，当前只达到协议/工具交付层，不满足评价准入",
         "source": "本轮 config/protocol_design.json"},
    ]
    # No existing annotation label file is opened above.
    for directory in ("config", "results", "figures", "supplementary"):
        (record / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("raw", "provenance", "manifests", "annotations", "predictions"):
        (data / directory).mkdir(parents=True, exist_ok=True)
    write_json(record / "config/checkpoint_manifest.json", checkpoint_doc)
    write_json(record / "config/protocol_design.json", protocol)
    write_text(record / "config/protocol_design.sha256", H.file_sha256(record / "config/protocol_design.json") + "  protocol_design.json\n")
    write_json(record / "config/stratified_sampling_plan.json", {"target_videos": 64, "minimum": 50, "domains": domains,
                  "practice_videos_separate": 8, "actual_admitted": 0, "sampling_seed": 20261008})
    write_json(data / "manifests/yt_exclusions.json", exclusions)
    write_json(data / "manifests/admitted_manifest.json", manifest)
    write_json(data / "manifests/candidate_manifest.json", manifest)
    write_json(record / "results/dataset_decisions.json", decisions)
    write_csv(record / "results/dataset_decisions.csv", decisions, list(decisions[0]))
    write_json(record / "results/checkpoint_verification.json", {"verified_count": 8, "checkpoints": checked,
              "model_code_sha256": model_sha, "checkpoint_tensor_load": False, "model_forward": False})
    history = {"actual_review": str(REVIEW), "requested_review_path_missing": True,
               "canonical_report": str(CANONICAL / "experiment_report.md"), "tvsum_report": str(TVSUM / "experiment_report.md"),
               "review_limitations": ["复盘里 chat-only PHD2 LOW/NOT_READY 与本会话 UNKNOWN/CONDITIONAL_READY 有差异；不修改历史记录，不据其决定新 GT",
                                      "复盘 baseline 架构/数据版本/归一化摘要与 Stage7.10F/7.16 原始冻结记录有差异；采用后者的实际身份"],
               "protected_file_sha256": {str(p): H.file_sha256(p) for p in (REVIEW, CANONICAL / "experiment_report.md",
                   CANONICAL / "results/canonical_baseline_manifest.json", TVSUM / "experiment_report.md",
                   TVSUM / "config/preregistration.json", REPO / "src/aic_video_highlight/ftnet/evaluation.py", model_code)},
               "git_head": git("rev-parse", "HEAD").strip(), "git_branch": git("branch", "--show-current").strip()}
    write_json(record / "results/history_identity.json", history)
    write_text(record / "supplementary/git_status_at_foundation.txt", git("status", "--short", "--branch"))
    write_text(record / "supplementary/git_diff_stat_at_foundation.txt", git("diff", "--stat"))
    gaps = {"planned_evaluation_videos": 64, "minimum_evaluation_videos": 50, "actual_admitted_videos": 0,
            "missing_to_target": 64, "missing_to_minimum": 50, "practice_videos_missing": 8,
            "planned_initial_human_video_assignments": 64 * 3, "completed_human_assignments": 0,
            "planned_practice_human_video_assignments": 8 * 3,
            "planned_total_initial_human_assignments_including_practice": (64 + 8) * 3,
            "human_gt_ready": False, "formal_model_evaluation": "NOT_RUN", "new_video_downloads": 0,
            "new_visual256_features": 0, "canonical_checkpoints_verified": 8,
            "person_hours": None, "person_hours_reason": "无已准入时长及标注速度实测，不能给出伪精确成本",
            "cost_formula": "初始观看下界=3×评价视频总时长；另计回看、练习、仲裁及身份/许可人工复核",
            "blockers": ["至少50条独立、每领域至少5条、原视频及人工标注使用权明确的素材",
                         "原视频 SHA256、来源别名、PTS/时长及内容家族独立性复核",
                         "独立三人真实标注与所有分歧仲裁", "正式数据子集冻结与兼容 Visual256"]}
    write_json(record / "results/resource_gaps.json", gaps)
    write_json(record / "results/admission_check.json", H.validate_manifest(manifest, exclusions))
    write_csv(record / "results/resource_counts.csv", [
        {"item": "Admitted legal videos", "observed": 0}, {"item": "Human annotated videos", "observed": 0},
        {"item": "Verified canonical checkpoints", "observed": 8}, {"item": "New videos downloaded", "observed": 0},
        {"item": "New Visual256 caches", "observed": 0}], ["item", "observed"])
    write_csv(record / "results/domain_coverage.csv", [{"domain": d, "observed": 0, "planned": n} for d, n in domains.items()],
              ["domain", "observed", "planned"])
    source_evidence = {"accessed_date": "2026-10-08", "sources": [
        {"url": decisions[3]["source"], "local_card": r"E:\ResearchData\datasets\SVHighlights\official_release\README.md",
         "claim": "编辑集锦自动对齐；发布特征/标注 CC BY-NC，原视频不随发布授权"},
        {"url": decisions[4]["source"], "additional_official_url": "https://github.com/MRHiSum/MR.HiSum",
         "claim": "31,892 条，Most replayed 代理，100 段相对重播统计；发布资产 CC BY 4.0 与平台条款分开"},
        {"url": decisions[1]["source"], "local_manifest": r"E:\ResearchData\datasets\tvsum\manifests\source_manifest.json",
         "claim": "摘要重要性；既存授权记录含 Webscope 要求，不自动接受重新标注"},
        {"url": decisions[2]["source"], "local_readme": r"E:\ResearchData\datasets\phd2\metadata\README.md",
         "claim": "个性化 GIF 选择；不冒充独立通用高光 GT"}],
        "failed_official_material_source_checks": ["studio.blender.org 返回读取失败", "peach/durian/mango.blender.org 返回402；未确认素材权利，未纳入样本池"],
        "gt_values_read": False, "notes": "只保存公开事实摘要及本地文件身份，不保存外部页面全文或下载任何视频"}
    write_json(data / "provenance/source_evidence.json", source_evidence)
    template = {"schema": H.SCHEMA + ".annotation/v1", "video_id": None, "source_sha256": None,
                "protocol_sha256": H.digest_json(protocol), "rater_id": None, "annotation_method": None,
                "tool_assistance": None, "watched_full_video": False,
                "bins": [], "template_status": "尚未绑定合法视频；不能提交为 GT；准入后由 CLI 按实际时长生成完整分箱"}
    write_json(record / "supplementary/annotation_template.json", template)
    write_json(record / "supplementary/adjudication_template.json", [{"video_id": None, "index": None,
               "source_sha256": None, "protocol_sha256": H.digest_json(protocol), "adjudicator_id": None,
               "annotation_method": None, "tool_assistance": None, "watched_full_video": False,
               "highlight": None, "reason": None}])
    write_csv(record / "supplementary/source_intake_template.csv", [], ["video_id", "domain", "source_url", "source_ids",
              "video_path", "duration_s", "source_sha256", "content_group", "license_identifier", "license_evidence_url",
              "license_evidence_path", "license_evidence_sha256", "independence_review"])
    write_json(record / "supplementary/source_record_template.json", {
               "video_id": None, "source_ids": [], "source_url": None, "source_sha256": None,
               "video_path": None, "duration_s": None, "domain": None, "content_group": None,
               "split": "EXTERNAL_TEST_ONLY", "no_query": True, "independence_review": "UNKNOWN",
               "license": {"status": "UNVERIFIED", "identifier": None,
                           "scope": "RAW_VIDEO_RESEARCH_AND_HUMAN_ANNOTATION", "evidence_url": None,
                           "evidence_path": None, "evidence_sha256": None},
               "template_status": "仅为可填写记录模板；未知项没有填写前禁止作为合法视频提交"})
    write_json(record / "config/annotation.schema.json", {"$schema": "https://json-schema.org/draft/2020-12/schema",
               "type": "object", "required": ["schema", "video_id", "source_sha256", "protocol_sha256", "rater_id",
               "annotation_method", "tool_assistance", "watched_full_video", "bins"],
               "properties": {"schema": {"const": H.SCHEMA + ".annotation/v1"}, "video_id": {"type": "string", "minLength": 1},
               "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
               "rater_id": {"type": "string", "minLength": 1}, "annotation_method": {"const": "HUMAN_MANUAL"},
               "tool_assistance": {"const": "NONE"}, "watched_full_video": {"const": True},
               "bins": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["index", "start_s", "end_s", "highlight", "uncertainty_reason"],
               "properties": {"index": {"type": "integer", "minimum": 0}, "start_s": {"type": "number", "minimum": 0},
               "end_s": {"type": "number", "exclusiveMinimum": 0}, "highlight": {"type": ["boolean", "null"]},
               "uncertainty_reason": {"type": "string"}}}}},
               "description": "静态 schema 之外，还须通过 Python 校验源身份、完整时间轴、三人独立性及先冻结要求"})
    write_text(record / "config/annotation_protocol.md", ANNOTATION_PROTOCOL)
    write_text(record / "supplementary/USAGE.md", usage(record))
    make_figures(record)
    write_text(record / "experiment_report.md", report(record, data, cache, exclusions, model_sha))
    write_json(record / "results/foundation_status.json", {"selected_dataset": "ODH_HUMAN_PILOT_V1",
               "decision": "CONDITIONALLY_ELIGIBLE", "human_gt_ready": False, "formal_model_evaluation": "NOT_RUN",
               "evaluation_protocol_frozen": "DESIGN_ONLY_NOT_EVAL_READY", "canonical_checkpoints_verified": 8,
               "final_status": "PARTIAL", "new_experiment_dir": str(record)})
    return {"record": str(record), "data": str(data), "cache": str(cache), "status": "PARTIAL", "checkpoint_verified": 8}


ANNOTATION_PROTOCOL = """# 无 Query 高光人工标注协议 v1

状态：设计冻结；未准入任何视频；没有人工 GT 或模型成绩。

## 任务语义
完整观看视频后，在没有用户查询、标题提示、类别提示或模型候选推荐的情况下，直接判断每个两秒区间是否包含值得单独保留或回看的显著精彩、情绪、趣味或事件时刻。普通背景、重复内容和单纯保持摘要完整所需的过渡，不自动成为高光。此定义仍有主观性，评价的是该规范下的多人高光判断，不代表所有人的偏好。

不要求截取固定比例，不强制存在正例，不根据播放量、重播热度、Qwen 输出、模型评分或旧数据集标签判断。不得把摘要重要性、序数评分、GIF 选择或编辑匹配标签直接映射为本协议 GT。

## 时间轴与标注过程
素材先完成原视频权限、SHA256、来源别名、实际时长/PTS、内容家族及 YT 重叠复核。标注练习使用八条独立、合法的练习视频，不进入测试子集；练习完成后才正式冻结评价子集。标注者不能看到 FTNet 输出和其他人的选择。

以源视频绝对时间建立 [2k,min(2k+2,duration)) 半开区间，完整覆盖视频，保留尾段。看完整视频再逐区间选择 true（高光）/false（非高光）/null（不确定，并写原因）。真人标注者需声明 HUMAN_MANUAL、tool_assistance=NONE、watched_full_video=true。该声明是过程记录，无法由软件代替真实的人类标注。

三个匿名但真实独立的标注者分别提交完整时间轴。三人一致且无 null 的区间直接保留；所有分歧及 null 交给独立第四位真人完整观看后给出二元仲裁和原因。保留原始三个判断，不以多数阈值或连续评分二值化制造正例。存在未解决区间时整套 GT 未就绪，不按模型结果删除困难视频。

## 一致性与质量
报告可比较的三人二元区间数、未确定比例、逐视频一致比例及 Fleiss κ；单类别退化时 κ=NA。区间不是独立研究样本，不生成区间级显著性检验或把数万个区间当视频数。记录观看确认、修订历史及匿名标注者来源；不得把 AI 填写的模板标记成人工数据。

## 冻结评价
合法素材及预定分层子集、全部源 SHA、时间轴、排除清单、8 checkpoint 与模型代码需先冻结；随后才能读取正式测试标签数值。Visual256 使用冻结 RT-DETR r50vd level-0 GAP、源 PTS 2 FPS、无新归一化；整视频输入 Canonical Visual-only，Audio 槽为零。

评价全部两秒区间，不使用 Qwen candidate/loss_mask 筛选。模型采样分数按绝对时间均值聚合，要求完整首尾及每区间覆盖。主指标为每视频八 seed AP 的平均，再对视频等权平均；同分预测按阈值组处理。没有正例的 AP=NA并登记；全正例 AP=1并登记。95% CI 用领域分层视频 bootstrap 10000次、seed=20261008，不把八个模型当八份独立测试集。报告逐视频、逐 seed、逐领域及有效 AP 数量；不混排本指标与 YT canonical pooled AP。

只评价一次，不根据新 GT 或测试结果挑 checkpoint、改网络或调参数。本轮尚未完成数据级冻结，更未进行模型评价。
"""


def usage(record: Path) -> str:
    cli = REPO / "scripts/experiments/stage7/open_domain_highlights.py"
    config = record / "config"
    common = f'--protocol "{config / "protocol_design.json"}" --checkpoints "{config / "checkpoint_manifest.json"}"'
    return f'''# 使用说明

当前样本 manifest 为空，preflight 会正确返回退出码 2。模板是尚未填写的真实人工工作表，不是 GT。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B "{cli}" preflight {common}
```

未来仅在明确获准的素材到位后，将原视频放 `{H.RAW_ROOT}`，许可证据放 `{H.DATA_ROOT / "provenance"}`。依据 source_intake_template.csv 和 source_record_template.json 填写 candidate_manifest.json 的 videos，每行必须符合模块校验；没有下载子命令，不自动获取素材。unknown/null/UNVERIFIED 必须经真实人工核实后填写；不得因需要通过工具而假填授权、独立性或时间轴。

```powershell
python -B "{cli}" select {common} --manifest "{H.DATA_ROOT / "manifests/candidate_manifest.json"}" --output "{H.DATA_ROOT / "manifests/selected_manifest.json"}"
python -B "{cli}" freeze {common} --manifest "{H.DATA_ROOT / "manifests/selected_manifest.json"}"
python -B "{cli}" template {common} --manifest "{H.DATA_ROOT / "manifests/selected_manifest.json"}" --rater-id HUMAN_01 --video-id 已准入视频ID --output "{H.DATA_ROOT / "annotations/待填写.json"}"
```

三个真人 JSON 放 annotations/；compile 提供原始三人标注以及独立仲裁列表，evaluate 同样要求原始标注，不能直接喂入伪造的汇总 GT。所有命令的输出拒绝覆盖。evaluate 有共享的一次性运行登记；失败登记不自动删除，不能换输出名字绕过重跑限制。预测生成入口本轮未运行，后续仅在数据与人工 GT 全部就绪后复用冻结 Canonical 路径。

```powershell
python -B "{cli}" compile {common} --manifest "{H.DATA_ROOT / "manifests/selected_manifest.json"}" --annotations "{H.DATA_ROOT / "annotations/三人标注列表.json"}" --adjudications "{H.DATA_ROOT / "annotations/仲裁列表.json"}" --output "{H.DATA_ROOT / "annotations/human_gt.json"}"
python -B "{cli}" evaluate {common} --manifest "{H.DATA_ROOT / "manifests/selected_manifest.json"}" --annotations "{H.DATA_ROOT / "annotations/三人标注列表.json"}" --adjudications "{H.DATA_ROOT / "annotations/仲裁列表.json"}" --predictions "{H.DATA_ROOT / "predictions/frozen_8seed.json"}" --output "{H.DATA_ROOT / "predictions/evaluation.json"}"
```

预测列表每行包含 video_id、seed、source_sha256、checkpoint_sha256、protocol_sha256、timestamps、scores；必须覆盖全部视频和八 seed。模板中的未知字段必须来自真实源文件与真实人类操作，不得照抄测试 fixtures。
'''


def make_figures(record: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = list(csv.DictReader((record / "results/resource_counts.csv").open(encoding="utf-8-sig")))
    domains = list(csv.DictReader((record / "results/domain_coverage.csv").open(encoding="utf-8-sig")))
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        values = [int(r["observed"]) for r in counts]
        axes[0].barh([r["item"] for r in counts], values, color="#0072B2")
        axes[0].set(xlim=(0, 10), xlabel="Observed count", title="Actual foundation resources (no model scores)")
        for i, value in enumerate(values): axes[0].text(value + .15, i, str(value), va="center")
        axes[1].barh([r["domain"] for r in domains], [int(r["planned"]) for r in domains],
                     facecolor="white", edgecolor="#555555", hatch="///", label="Planned only")
        axes[1].scatter([int(r["observed"]) for r in domains], range(len(domains)), color="#0072B2", marker="s", label="Admitted = 0")
        axes[1].set(xlim=(-.5, 10), xlabel="Video count", title="Coverage gap: planned ≠ observed")
        axes[1].legend(loc="lower right")
        for fmt in ("png", "svg"): fig.savefig(record / f"figures/fig01_actual_resources.{fmt}", dpi=180, facecolor="white")
        plt.close(fig)
    write_json(record / "figures/figure_sources.json", {"figure": "fig01_actual_resources",
               "sources": ["results/resource_counts.csv", "results/domain_coverage.csv"],
               "source_sha256": {n: H.file_sha256(record / n) for n in ("results/resource_counts.csv", "results/domain_coverage.csv")},
               "transformations": "原始计数直接编码；计划用斜线区分，零不等同未知成绩",
               "missing_performance": "没有 GT，不绘制 AP、置信区间或预测曲线",
               "alt_text": "八个 Canonical checkpoint 已验证；合法视频、人工标注、新视频下载和新特征均为零。八个领域各计划八视频，实际均为零。"})


def report(record: Path, data: Path, cache: Path, exclusions: dict, model_sha: str) -> str:
    return f'''# Stage 7.17 — 开放领域无 Query 高光评价基础

**结果：PARTIAL。已交付可执行人工标注与评价基础；合法样本池和独立人工 GT 缺失，正式模型评价 NOT_RUN。**

## 一、启动与历史核对
分支 {git("branch", "--show-current").strip()}，HEAD {git("rev-parse", "HEAD").strip()}。保留既有 dirty；无 commit/push。用户指定复盘旧路径不存在，实际读取 `{REVIEW}`，其内部生成时间为2026-10-08 15:43。

Stage7.10F 原始冻结记录确认 YT canonical AP=0.447946；Stage7.16 Spearman=0.0298，属于 YT→TVSum 摘要重要性零样本排序未检测到泛化，不能推广为所有领域高光失败。复盘里的 baseline 版本/归一化摘要以及 PHD2 chat-only 引用存在与原始记录/本会话差异，本轮登记在 history_identity.json，未改写任何历史结果。

## 二、数据方案决策
详见 results/dataset_decisions.csv/json。YT 是既有六类训练坐标，TVSum 是摘要重要性，PHD2 是 GIF 用户偏好，均不作为新开放领域人工高光 GT。

Mr. HiSum 原论文把 Most replayed 统计作为代理标签，提供100个相对时间区间；其 broad category coverage 并不能将行为代理变为直接人工标注，本轮不采用 top-percent 二值化 AP。[官方论文](https://proceedings.nips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf)、[官方仓库](https://github.com/MRHiSum/MR.HiSum)。

SVHighlights 是官方编辑集锦经自动对齐及过滤形成的两秒标签，适合编辑高光代理研究；发布资产许可不直接覆盖原视频，发布特征也不是冻结 RT-DETR Visual256。[官方说明](https://github.com/leedongkyu2019/SVHighlights)。本轮只读取说明与身份清单，不读取测试标签数值。

唯一选定 **ODH_HUMAN_PILOT_V1，CONDITIONALLY_ELIGIBLE（协议层）**。计划64条评价视频、八领域各八条；最低50条且每领域至少五条，另设八条独立标注练习视频。当前获准样本=0，真实标注=0，不能声称领域覆盖已经建立。没有把未确认许可的素材填进清单。

## 三、实际交付
新增独立 open_domain_highlights.py 模块、中文 CLI、定向测试和本实验生成器。支持逐条许可/来源/字节哈希与内容家族 Gate、YT 来源排除、预定分层哈希抽样、先冻结、真人模板、三人标注与独立仲裁、描述性一致性、固定时间分箱、同分处理 AP、八 seed 预测完整性及视频 bootstrap。

E 数据目录：`{data}`。candidate_manifest.json 和 admitted_manifest.json 明确为空并标记阻塞，绝非虚构素材。已登记 {len(exclusions['source_ids'])} 个 YT 来源别名、{len(exclusions['source_sha256'])} 个已有视频字节哈希用于后续排除；没有访问 AIC 资产。软件只能核验记录，不能代替人的权限与身份判断。

config/annotation_protocol.md 是可执行规范；supplementary/annotation_template.json、adjudication_template.json、source_intake_template.csv、source_record_template.json 是可填写模板；config/annotation.schema.json 和 Python 交叉校验提供实际数据契约。标注模板初始值为未知，不能作为已完成 GT。

## 四、协议冻结与模型身份
protocol_design.json 已保存字节 SHA256；这只是 **DESIGN_ONLY_NOT_EVAL_READY**，不能等同完整测试协议冻结。正式 freeze 命令要求合法原视频、独立来源、最低规模、全部领域与八 checkpoint 先闭合，且先于正式人工标签读取。当前没有创建 evaluation_freeze.json。

八个 checkpoint 已实际逐字节验证，与 Stage7.10F 的 seed/SHA 锚点一致；没有 torch.load 或模型前向。冻结模型代码 SHA256=`{model_sha}`。Stage7.16 提取/推理定义可复用，但其 TVSum GT adapter、importance 指标及读取 GT 的 preflight 不适合直接照搬；新入口将 GT 放在正式冻结之后。

现存 YT/PHD2 Visual256 不产生新的独立人工 GT；TVSum 的50条 Visual256只支持其已完成的摘要任务记录。没有兼容且获准的 ODH 缓存，本轮无新特征提取。

## 五、评价定义与验证
完整两秒 universe，源绝对时间，保留尾段；不使用 Qwen candidate 或训练 mask。直接人工 bool 高光标签，不二值化评分。每视频计算八 seed AP 再平均，视频等权 macro AP；CI 按领域分层重采样视频10000次。不把区间或seed当独立视频。无正例 AP=NA并完整登记；同分采用组阈值处理；未解决分歧阻断评价。

本轮定向测试和 CLI 实机 Gate 结果另见 results/test_results.txt、results/cli_preflight.json、results/validation_summary.json（生成后核验）。测试 fixtures 是合成契约测试输入，绝不是科研 GT 或成绩。没有真人标注，所以没有逐视频模型指标或八 seed 新成绩。

## 六、真实图表
![本轮资源计数与计划覆盖缺口](<{record / "figures/fig01_actual_resources.png"}>)

图由 resource_counts.csv/domain_coverage.csv 生成，原始表、来源 SHA、生成器与替代文字均保留。斜线代表计划，方点代表实际零样本，不绘制虚构 AP。本轮采用 scientific-visualization 本地 Skill 的计数、缺失值及图源核验原则，属于研究内部报告，不声称期刊认证。

## 七、成本、缺口与停止
实际新视频下载=0、新特征=0、模型推理=0。目标尚缺64条合法评价素材，最低门槛尚缺50条；另缺八条独立练习素材。计划评价集初始真人视频标注192份，加独立练习24份，共216份，实际完成0份；尚缺全部分歧仲裁。工时须由实际素材总时长和标注速度估计，当前 UNKNOWN；初始完整观看下界为三倍评价视频总时长。

最低下一步是提供有来源与原视频科研/标注使用依据的素材及真人标注资源，再按冻结 Gate 接入。需要付费、授权或大规模下载时停止，不擅自执行。当前符合“素材不足只交付协议、工具和精确缺口”替代分支；本轮不追加同主题调查、训练或 Fusion 开发。

```text
SELECTED_DATASET = ODH_HUMAN_PILOT_V1
DATASET_ELIGIBILITY = CONDITIONALLY_ELIGIBLE_PROTOCOL_ONLY
LABEL_SEMANTICS = DIRECT_HUMAN_BINARY_HIGHLIGHT_PLANNED
NO_QUERY_CONFIRMED = PROTOCOL_YES_COLLECTION_NOT_EXECUTED
HUMAN_GT_READY = NO
DATA_LICENSE_STATUS = NOT_ESTABLISHED_NO_ADMITTED_SAMPLES
DOMAIN_COVERAGE = PLANNED_8_DOMAINS_OBSERVED_0
SPLIT_LEAKAGE_CHECK = YT_EXCLUSION_REGISTRY_READY_NO_ADMITTED_SAMPLES
EVALUATION_PROTOCOL_FROZEN = DESIGN_ONLY_NOT_EVAL_READY
CANONICAL_CHECKPOINTS_VERIFIED = YES_8_OF_8_SHA256_ONLY
FORMAL_MODEL_EVALUATION = NOT_RUN
NEW_EXPERIMENT_DIR = {record}
TRAINING_RUN = NO
MODEL_FORWARD_RUN = NO
VIDEO_DOWNLOAD = NO
NEW_FEATURE_EXTRACTION = NO
AIC_PROTECTED_DATA_ACCESSED = NO
HISTORICAL_RESULTS_MODIFIED = NO
AUTODL = NO
COMMIT = NO
PUSH = NO
FINAL_STATUS = PARTIAL
```

运行缓存只放 `{cache}`。本轮完成替代交付后 STOP。
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args), ensure_ascii=False, indent=2))
    except (H.EligibilityError, OSError, ValueError, KeyError) as exc:
        raise SystemExit("基础产物构建阻断：" + str(exc))


if __name__ == "__main__":
    main()
