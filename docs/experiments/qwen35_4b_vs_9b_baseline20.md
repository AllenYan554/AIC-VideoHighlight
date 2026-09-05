# Qwen3.5 4B vs 9B Baseline 20 Model Scale Probe

> 本文件是实验摘要。完整方法、raw response、20 条表格与结论见 `qwen35_4b_vs_9b_model_scale_probe_report.md`。
> 2026-09-05 修订说明：保留 9B 18/20 official 历史事实，新增 non-destructive temporal salvage；salvage 不改变 official success。

## 1. Official Pipeline Result

实验只改变模型，固定 Prompt `high_recall_retrieval_v0`、FPS=2、chunk/overlap=30/5 s、merge tIoU=0.5、temperature=0、thinking=false、max tokens=512、Baseline 20 与 duration-based union-aware metrics。

| Item | 4B Official | 9B Official |
|---|---:|---:|
| Model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Processed | 20 | 20 |
| SUCCESS | 20 | **18** |
| FAILED | 0 | **2** |
| Schema success rate | 100% | **90%** |
| Formal 20/20 aggregate | 有 | **N/A** |

9B 失败样本：`qvh_000008_9x16`、`qvh_000574_9x16`。两条都不是 OOM、timeout、engine crash、视频理解失败或 truncation，而是 `ResponseParseError: segments[0].score must be a number`。无论 temporal salvage 指标如何，9B Official Pipeline Result 始终是 **18/20 SUCCESS**。

4B Official 20/20 指标：

| Metric | Value |
|---|---:|
| Precision | 0.429489 |
| Recall | 0.973001 |
| F1 | 0.550590 |
| Temporal IoU | 0.421548 |
| Mean coverage | 89.786% |
| Full-video coverage (≥99.9%) | 14/20 |
| Coverage ≥90% | 15/20 |
| Cumulative over-prediction | 238.263 s |
| Cumulative missed-reference | 3.131 s |
| Raw candidates / final segments | 78 / 74 |
| Mean inference time | 3.216 s |

## 2. Matched-18 Comparison

以下从原始结果重新计算，只比较两模型都 formal success 的相同 18 条。这是最公平的模型方向比较，但不是 9B 20/20 aggregate。

| Metric | 4B matched-18 | 9B Official matched-18 | Delta |
|---|---:|---:|---:|
| Precision | 0.403331 | 0.416344 | +0.013013 |
| Recall | 0.970809 | 0.972291 | +0.001482 |
| F1 | 0.528359 | 0.543269 | +0.014910 |
| Temporal IoU | 0.394603 | 0.407006 | +0.012403 |
| Mean coverage | 90.454% | 86.290% | -4.165 pp |
| Full-video coverage | 13/18 | 10/18 | -3 |
| Coverage ≥90% | 14/18 | 12/18 | -2 |
| Over-prediction | 228.307 s | 204.660 s | -23.647 s |
| Missed-reference | 3.058 s | 3.665 s | +0.608 s |
| Raw candidates | 68 | 52 | -16 |
| Final segments | 64 | 50 | -14 |
| Mean inference time | 3.112 s | 3.908 s | +0.796 s (+25.6%) |

## 3. Non-destructive Salvage

标记：`SALVAGED / TEMPORAL_ONLY / NON_OFFICIAL_PIPELINE_RESULT`。

只读分析脚本从已有 raw response 提取模型真实输出的 `start_sec`、`end_sec`、`reason`，复用项目正式 `duration_based_metrics`。它不重新调用模型、不补 score、不删除/截断候选、不取前 5 段、不人工改边界、不修改 `metrics.csv`、`samples/`、`raw/` 或 official report。

正式 parser 的非语义时间规则只有“`end_sec` 超过 chunk duration 时裁剪”；本次两条共 13 段全部合法且无需裁剪，因此 RAW 与 NORMALIZED 时间值一致。

### qvh_000008_9x16

- JSON syntax valid，`finish_reason=stop`。
- 6/6 有 start/end/reason，0/6 有 score。
- 6 段也违反 Prompt 的 `max_segments_per_chunk=5`。
- Prediction union：[0.000, 21.563] s，21.563000 s。
- Reference union：3 段，5.046708 s。
- Intersection / duration union：5.046708 / 21.563000 s。
- P/R/F1/tIoU：**0.234045 / 1.000000 / 0.379313 / 0.234045**。
- Coverage：99.9989%；over-prediction：16.516292 s；missed-reference：0 s。
- Raw candidates / temporal segments：6 / 6。

相对 4B 的 coverage 67.5%，9B 变宽到几乎全片。9B 用“场景切换、信息量提升、视觉突出、动作流畅”把自拍、水面、沉船鱼群、珊瑚与游泳连续切成 6 段。该视频有明显 weak-reference ambiguity，不能只凭低 Precision 判断模型能力，但可以确认 9B 没有收紧选择。

### qvh_000574_9x16

- JSON syntax valid，`finish_reason=stop`。
- segment count **7**；7/7 有 start/end/reason，0/7 有 score。
- 同时违反 missing score 与 `segment_count > 5`。
- 使用全部 7 段，Prediction union：[0.000, 30.000] s，30.000000 s。
- Reference union：[0.000, 31.031] s。
- Intersection / duration union：30.000000 / 31.031000 s。
- P/R/F1/tIoU：**1.000000 / 0.966775 / 0.983107 / 0.966775**。
- Coverage：95.5473%；over-prediction：0 s；missed-reference：1.031000 s。
- Raw candidates / temporal segments：7 / 7。

这条 positive control 的时间语义仍接近正确，但 Recall 不是 1。因为第一个 0–30 s chunk 就 parse fail，pipeline 没有生成第二个 overlap chunk 响应，salvage 不能虚构 30 s 后的输出。模型把重复的拿取/进食互动按 3–5 s 阶段拆成 7 段；语义正确不等于 schema 合规。

## 4. Salvaged Temporal Diagnostic

18 条 official success 加两条 temporal salvage 得到 20-sample research diagnostic：

| Metric | 4B Official 20 | 9B Official matched/18 | 9B Salvaged Temporal 20 | Salvaged-4B |
|---|---:|---:|---:|---:|
| Precision | 0.429489 | 0.416344 | 0.436412 | +0.006923 |
| Recall | 0.973001 | 0.972291 | 0.973401 | +0.000400 |
| F1 | 0.550590 | 0.543269 | 0.557063 | +0.006473 |
| Temporal IoU | 0.421548 | 0.407006 | 0.426347 | +0.004799 |
| Mean coverage | 89.786% | 86.290% | 87.438% | -2.348 pp |
| Full-video coverage | 14/20 | 10/18 | 11/20 | -3 |
| Coverage ≥90% | 15/20 | 12/18 | 14/20 | -1 |
| Over-prediction | 238.263 s | 204.660 s | 221.176 s | -17.087 s |
| Missed-reference | 3.131 s | 3.665 s | 4.696 s | +1.565 s |
| Raw candidates | 78 | 52 | 65 | -13 |
| Final / temporal segments | 74 | 50 | 63 | -11 |
| Schema success rate | 100% | 90% | **90%（不因 salvage 改变）** | -10 pp |
| Mean inference time | 3.216 s | 3.908 s | 3.956 s | +0.741 s |
| Peak VRAM | N/A（无可比记录） | 23781 MiB | 23781 MiB | N/A |

这套结果只能命名为 **9B Salvaged Temporal Diagnostic**，不能命名为 9B Baseline Final Metrics。

## 5. Schema Compliance Analysis

| Video | Missing score | Segment count | JSON | finish_reason | Official status |
|---|---:|---:|---:|---:|---|
| qvh_000008_9x16 | 6/6 | 6 (>5) | valid | stop | FAILED |
| qvh_000574_9x16 | 7/7 | 7 (>5) | valid | stop | FAILED |

9B 共 20 samples、25 raw chunks，finish_reason=stop 为 25/25；23/25 chunks parse success，2/25 parse failed。18 条 schema-valid 样本发出 52 个 raw candidates；两条 schema-invalid 样本另发出 13 个，所以 temporal diagnostic 共使用 65 个真实候选。failure 是字段与候选上限 contract 失守，不是内容为空。

## 6. 20 条结果索引

完整 20 行机器可读结果位于 `qwen35_4b_vs_9b_baseline20.csv`。其中 18 条 `9b_official_* = 9b_salvaged_*`；两条失败的 `9b_official_success=false`、`salvaged_temporal=true`、official metrics 为空、salvaged metrics 有真实数值。

关键变化：

- 明显改善：`000602`、`000949`、`000915`；`000089` Recall 0.735→0.997，但 coverage 同时变宽。
- 完全不变：`000023/337/531/612/067/629/134/358/363`。
- 退化或混合：`000781` 全片覆盖；`000721` Recall 1.000→0.680；`000008` salvage 变为全片；`000574` temporal 近似持平但 official schema fail。
- 五个严重过召回压力样本中：2/5 改善、2/5 不变、1/5 退化。

## 7. 推理速度、显存与部署

- GPU：RTX 4090 D，24564 MiB；BF16，无量化。
- 9B：`max-model-len=32768`、`max-num-seqs=16`、GPU memory utilization 0.95。
- 默认 256 sequences 超过可用 Mamba cache blocks（123）；降为 16 后启动成功。这是运行时并发适配，不改变串行算法实验变量。
- 9B 权重加载：17.66 GiB / 4.36 s。
- Text Smoke PASS 1.990 s；Video Smoke PASS 15.176 s。
- Baseline 峰值 23781 MiB，约余 783 MiB；OOM/timeout/engine crash 均为 0。
- matched-18 mean inference 3.112→3.908 s，9B 慢 25.6%。

## 8. Final Model Scale Conclusion

**判断：9B 相比 4B 小幅更好，但不稳定，不是质变。**

证据是 matched-18 F1/tIoU 仅 +0.0149/+0.0124；salvaged-20 仅 +0.0065/+0.0048；9/20 完全不变，11/20 仍近似全片覆盖。9B 仍用场景变化、视觉丰富、普通走动/操作、准备与完整过程作为高光理由，说明 Prompt v0 是系统性过召回的主要瓶颈。

工程侧则明显更差：schema success 100%→90%，两条都缺 score 且超候选上限。24GB 可以运行 9B，但峰值余量不足 1 GB，并付出约 25.6% matched latency 增加。

因此当前**不立即把主线从 4B 切到 9B**。先保持 4B，下一步只做 Prompt v1 单变量实验；Prompt v1 应收紧高光充分条件、排除普通场景/过程、缩短前摇与持续段、保留 Recall guardrail，并要求每段数值 score 与每 chunk ≤5 段。本轮不修改 Prompt、parser 或模型输出。

## 9. 文件与不可改写项

- 完整报告：`docs/experiments/qwen35_4b_vs_9b_model_scale_probe_report.md`
- 20 行对照：`docs/experiments/qwen35_4b_vs_9b_baseline20.csv`
- 诊断证据：`docs/experiments/data/qwen35_9b_salvaged_temporal_diagnostic.json`
- 可复现脚本：`scripts/analysis/salvage_qwen35_9b_temporal.py`

4B frozen baseline 与 9B original outputs 均不属于本报告的写入目标，必须保持原样。
