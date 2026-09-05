# Qwen3.5-4B vs Qwen3.5-9B Model Scale Probe

> 结论标签：`MODEL_SCALE_PROBE` / `OFFICIAL_AND_DIAGNOSTIC_SEPARATED`
> 诊断标签：`SALVAGED` / `TEMPORAL_ONLY` / `NON_OFFICIAL_PIPELINE_RESULT`
> 分析日期：2026-09-05
> 分析仓库：`/root/autodl-tmp/AIC-VideoHighlight-run`，`master`，起始 HEAD `f56e833ba9a4e9d1d6538f34518f9fbefa286c68`

## 1. 实验背景

Stage 2 在设计 Prompt v1 前先做模型规模探针，目的是把“模型能力不足”和“Prompt 对高光定义过宽”尽可能拆开。4B v0 已表现出很高 Recall（0.973001），但平均覆盖 89.8%，14/20 几乎覆盖全片，Precision 仅 0.429489。核心研究问题是：在 Prompt、采样和评测全部不变时，9B 能否显著减少普通内容过召回；若不能，主瓶颈更可能是 Prompt 定义，而非仅靠模型扩容可以解决。

本报告严格保存两种不同事实：

1. **9B Official Pipeline Result**：18/20 SUCCESS，2/20 FAILED，sample-level schema success rate 90%。
2. **9B Salvaged Temporal Diagnostic**：仅为研究而从两条失败 raw response 提取真实时间段；不补 score、不重推理、不修改 official output，不能称为 20/20 pipeline success。

## 2. 实验假设

- **H1（模型规模瓶颈）**：若 9B 在保持 Recall 的同时显著减少普通内容覆盖、过预测时长和全片覆盖，则模型规模是重要瓶颈。
- **H2（Prompt 瓶颈）**：若 9B 仍大量全片覆盖，并继续把场景变化、视觉丰富、普通走动/操作、准备和完整过程作为高光，则 Prompt v0 是主要瓶颈。

## 3. 单变量实验设计

算法实验只改变模型：4B → 9B。固定变量如下。

| Variable | Fixed value |
|---|---|
| Dataset | `configs/baseline_20.json`，同一 20 条 |
| Prompt | `high_recall_retrieval_v0` |
| Sampling FPS | 2.0 |
| Chunk / overlap | 30 s / 5 s |
| Merge | sorted adjacent temporal-IoU union，threshold 0.5，maximum score |
| Temperature | 0 |
| Thinking | false |
| Max new tokens | 512 |
| Timeout | 120 s |
| Reference | `weak_reference_segments` |
| Metrics | duration-based、union-aware；零时长 reference ≤0.001 s 排除出时长分母 |

9B 的 `max-num-seqs` 调整属于服务可运行性参数；Baseline 是串行请求，该调整不改变视频、Prompt、采样、生成或评测逻辑，因而不属于算法变量。

## 4. 模型与部署环境

| Item | 4B | 9B |
|---|---|---|
| Model | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` |
| Revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| vLLM | 0.28.0 | 0.28.0 |
| Python | 3.12.14 | 3.12.14 |
| PyTorch | 2.13.0 | 2.13.0 |
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB | 同左 |
| dtype | BF16 | BF16 |
| quantization | none | none |
| max-model-len | 未在 4B 留存报告中单列 | 32768 |
| max-num-seqs | 未在 4B 留存报告中单列 | 16 |
| GPU memory utilization | 未在 4B 留存报告中单列 | 0.95 |

## 5. 9B 部署过程

9B 使用统一 HF cache `/root/autodl-tmp/hf-cache`，由于 Hugging Face 直连超时，曾通过 `hf-mirror.com` 获取同一 exact revision；下载后以离线模式加载。9B cache 约 19 GB，统一 cache 约 27 GB。

初次按 vLLM 默认 `max_num_seqs=256` 启动时，权重已加载但可用 Mamba cache blocks 为 123，无法捕获 256 sequences。将并发上限降到 16 后成功启动。最终参数为 BF16、无量化、`max-model-len=32768`、`max-num-seqs=16`、`gpu-memory-utilization=0.95`。权重加载记录为 17.66 GiB / 4.36 s；服务启动后无 OOM。

这里要区分：Mamba cache 与并发上限是**运行时工程适配**；Prompt、FPS、chunk、merge、temperature、thinking、max tokens、数据集和 metrics 均未改变，仍满足模型规模探针的单变量要求。

## 6. Smoke Test

| Test | Result | finish_reason | Wall time | Evidence |
|---|---:|---:|---:|---|
| Text Smoke | PASS | stop | 1.990 s | 返回正常中文文本 |
| Video Smoke | PASS | stop | 15.176 s | 正确描述沿海道路、海面、船只、风机及车内后视镜/手机 |

Video Smoke 使用既有 `/root/autodl-tmp/datasets/sample.mp4`；未降低视觉输入、未改 FPS、未使用 Transformers 直接 generate。两项均无 OOM、timeout、HTTP error 或 engine error。

## 7. Official Baseline 结果

| Result | 4B Official | 9B Official |
|---|---:|---:|
| Processed | 20 | 20 |
| SUCCESS | 20 | 18 |
| FAILED | 0 | 2 |
| Sample schema success rate | 100% | 90% |
| Formal 20-sample aggregate | 有 | **N/A** |

4B Official 20/20：P=0.429489，R=0.973001，F1=0.550590，tIoU=0.421548。9B 不具备 20/20 正式 aggregate；失败样本为 `qvh_000008_9x16` 与 `qvh_000574_9x16`。这两个失败在本报告任何位置都不会被改写为 official success。

## 8. Schema Failure 专项分析

### 8.1 qvh_000008_9x16 raw response（完整文本）

```json
{"has_highlight":true,"segments":[{"start_sec":0.0,"end_sec":3.0,"reason":"水下自拍镜头，人物动作明显，气泡上升，视觉表现突出"},{"start_sec":3.0,"end_sec":4.0,"reason":"场景切换至水面，展示船只和人物，信息量提升"},{"start_sec":4.0,"end_sec":10.0,"reason":"水下探索场景，展示沉船结构和鱼群，视觉表现突出"},{"start_sec":10.0,"end_sec":12.0,"reason":"人物在水下游泳，动作流畅，视觉表现突出"},{"start_sec":12.0,"end_sec":17.0,"reason":"展示水下珊瑚礁和沉船细节，信息量提升，视觉表现突出"},{"start_sec":17.0,"end_sec":21.563,"reason":"人物在水下游泳，动作流畅，视觉表现突出"}]}
```

- JSON 语法有效；`finish_reason=stop`，不是 truncation。
- 6/6 段都有 `start_sec`、`end_sec`、`reason`；0/6 有 `score`。
- formal parser 首个确定性错误：`ResponseParseError: segments[0].score must be a number`。
- Prompt 上限是每 chunk 5 段；实际 6 段，因此还存在 `segment_count > 5` 的 contract violation。

### 8.2 qvh_000574_9x16 raw response（完整文本）

```json
{"has_highlight":true,"segments":[{"start_sec":0.0,"end_sec":3.0,"reason":"左侧女孩举起食物展示并做出夸张表情，吸引注意力"}, {"start_sec":3.0,"end_sec":6.0,"reason":"左侧女孩伸手去拿右侧女孩盘子里的食物，产生互动"}, {"start_sec":6.0,"end_sec":10.0,"reason":"左侧女孩成功拿到食物并吃下，右侧女孩继续专注进食"}, {"start_sec":10.0,"end_sec":15.0,"reason":"左侧女孩再次伸手去拿右侧女孩盘子里的食物，互动持续"}, {"start_sec":15.0,"end_sec":20.0,"reason":"左侧女孩再次伸手去拿右侧女孩盘子里的食物，互动持续"}, {"start_sec":20.0,"end_sec":25.0,"reason":"左侧女孩再次伸手去拿右侧女孩盘子里的食物，互动持续"}, {"start_sec":25.0,"end_sec":30.0,"reason":"左侧女孩再次伸手去拿右侧女孩盘子里的食物，互动持续"}]}
```

- JSON 语法有效；`finish_reason=stop`，不是 truncation。
- segment count 确认为 7；7/7 均有 start/end/reason，0/7 有 score。
- formal parser 首个确定性错误同样是 `segments[0].score must be a number`。
- 同时违反每 chunk 最多 5 段的 Prompt contract。

| Video | Missing score | Segment count | Other violations / exclusions |
|---|---:|---:|---|
| qvh_000008_9x16 | 6/6 | 6 (>5) | 无；JSON 合法、bounds 合法、finish=stop |
| qvh_000574_9x16 | 7/7 | 7 (>5) | 无；JSON 合法、bounds 合法、finish=stop |

两条都不是 OOM、timeout、engine crash、未理解视频或截断，而是 structured-output/schema compliance failure。`segment_count > 5` 是附加 contract violation，但当前 parser 在检查第一段 score 时已先失败。

## 9. Non-destructive Salvage 方法

独立脚本 `scripts/analysis/salvage_qwen35_9b_temporal.py` 只读 4B/9B 的 `raw/`、`predictions.jsonl`、`metrics` 与 Baseline manifest，复用项目正式 `duration_based_metrics`。规则如下：

1. 只提取模型真实输出的 `start_sec`、`end_sec`、`reason`。
2. 不补 0.5/1.0 或任何 score，不根据 reason 推断 score。
3. 不删候选、不取前 5 段、不人工重写或语义合并；`000574` 的 7 段全部进入 interval union。
4. 只镜像 formal parser 已有的时间合法化：仅当 `end_sec > chunk_duration` 时裁剪；负 start 或逆序区间仍报错。本次 13 段全部无需裁剪，故 RAW= NORMALIZED（加 chunk offset 后仍为同一局部时间）。
5. duration-based 指标对 prediction/reference 分别取时间并集，避免相邻或重叠段重复计时。
6. 只写 `docs/experiments/` 下的小型 CSV/JSON，不改 `metrics.csv`、`samples/`、`raw/` 或原 official report。

因此它只回答：“忽略 score schema 错误时，模型已经真实输出的时间选择表现如何？”所有结果必须标为 `SALVAGED / TEMPORAL_ONLY / NON_OFFICIAL_PIPELINE_RESULT`。

两条 raw 均来自 chunk 0。尤其 `000574` 的视频时长 31.398 s，模型在首个 0–30 s chunk 就触发 parse failure，pipeline 没有继续处理第二个 overlap chunk；salvage 不得虚构未发生的第二次响应。

## 10. Official Matched-18 Comparison

重新从原始 `predictions.jsonl` 与 raw 文件计算两模型共同成功的 18 条，结果如下。

| Metric | 4B matched-18 | 9B Official matched-18 | Delta (9B-4B) |
|---|---:|---:|---:|
| Mean Precision | 0.403331 | 0.416344 | +0.013013 |
| Mean Recall | 0.970809 | 0.972291 | +0.001482 |
| Mean F1 | 0.528359 | 0.543269 | +0.014910 |
| Mean Temporal IoU | 0.394603 | 0.407006 | +0.012403 |
| Mean coverage | 90.454% | 86.290% | -4.165 pp |
| Full-video coverage (≥99.9%) | 13/18 | 10/18 | -3 |
| Coverage ≥90% | 14/18 | 12/18 | -2 |
| Cumulative over-prediction | 228.307 s | 204.660 s | -23.647 s |
| Cumulative missed-reference | 3.058 s | 3.665 s | +0.608 s |
| Raw candidates | 68 | 52 | -16 |
| Final segments | 64 | 50 | -14 |
| Mean inference time | 3.112 s | 3.908 s | +0.796 s (+25.6%) |

matched-18 显示 9B 局部收紧了覆盖并略升 P/F1/tIoU，但幅度只有约 1.2–1.5 个百分点；同时 missed-reference 增加，10/18 仍是全片覆盖。这是公平的共同成功子集比较，不是 20/20 end-to-end 结论。

## 11. Salvaged Temporal 20 Comparison（DIAGNOSTIC ONLY）

| Metric | 4B Official 20/20 | 9B Official matched/18 | 9B Salvaged Temporal 20/20 | Salvaged-4B |
|---|---:|---:|---:|---:|
| Mean Precision | 0.429489 | 0.416344 | 0.436412 | +0.006923 |
| Mean Recall | 0.973001 | 0.972291 | 0.973401 | +0.000400 |
| Mean F1 | 0.550590 | 0.543269 | 0.557063 | +0.006473 |
| Mean Temporal IoU | 0.421548 | 0.407006 | 0.426347 | +0.004799 |
| Mean coverage | 89.786% | 86.290% | 87.438% | -2.348 pp |
| Full-video coverage (≥99.9%) | 14/20 | 10/18 | 11/20 | -3 |
| Coverage ≥90% | 15/20 | 12/18 | 14/20 | -1 |
| Cumulative over-prediction | 238.263 s | 204.660 s | 221.176 s | -17.087 s |
| Cumulative missed-reference | 3.131 s | 3.665 s | 4.696 s | +1.565 s |
| Raw candidates | 78 | 52 | 65 | -13 |
| Final / temporal segments | 74 | 50 | 63 | -11 |
| Schema success rate | 100% | 90% (18/20 samples) | **90%（salvage 不改变）** | -10 pp |
| Mean inference time | 3.216 s | 3.908 s | 3.956 s | +0.741 s (+23.0%) |
| Peak VRAM | 未留存可比峰值 | 23781 MiB | 23781 MiB | N/A |

该表的第三列是 18 条 formal success 加 2 条 temporal-only salvage。它是研究诊断，不是严格 end-to-end pipeline comparison，也不得命名为 “9B Baseline Final Metrics”。

## 12. 20 条逐样本结果

指标列顺序均为 P/R/F1/tIoU；9B 列对 18 条使用 official metrics，对两条失败使用明确标记的 salvage metrics。

| video_id | group | 9B source | 4B P/R/F1/tIoU | 9B temporal P/R/F1/tIoU | coverage 4B→9B | ΔP/ΔR/ΔF1/ΔtIoU |
|---|---|---|---|---|---:|---|
| qvh_000023_9x16 | single_segment | OFFICIAL | 0.783/1.000/0.879/0.783 | 0.783/1.000/0.879/0.783 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000337_9x16 | single_segment | OFFICIAL | 0.387/1.000/0.559/0.387 | 0.387/1.000/0.559/0.387 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000757_9x16 | single_segment | OFFICIAL | 0.466/1.000/0.636/0.466 | 0.499/1.000/0.665/0.499 | 100.0%→93.4% | +0.033/0.000/+0.030/+0.033 |
| qvh_000602_9x16 | single_segment | OFFICIAL | 0.107/1.000/0.193/0.107 | 0.180/1.000/0.305/0.180 | 100.0%→59.5% | +0.073/0.000/+0.111/+0.073 |
| qvh_000949_9x16 | single_segment | OFFICIAL | 0.117/1.000/0.210/0.117 | 0.170/1.000/0.291/0.170 | 100.0%→68.9% | +0.053/0.000/+0.081/+0.053 |
| qvh_000531_9x16 | double_segment | OFFICIAL | 0.505/0.824/0.626/0.456 | 0.505/0.824/0.626/0.456 | 52.1%→52.1% | 0.000/0.000/0.000/0.000 |
| qvh_000612_9x16 | double_segment | OFFICIAL | 0.546/1.000/0.707/0.546 | 0.546/1.000/0.707/0.546 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000067_9x16 | double_segment | OFFICIAL | 0.183/1.000/0.310/0.183 | 0.183/1.000/0.310/0.183 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000781_9x16 | double_segment | OFFICIAL | 0.141/1.000/0.247/0.141 | 0.133/1.000/0.235/0.133 | 94.8%→100.0% | -0.007/0.000/-0.011/-0.007 |
| qvh_000629_9x16 | double_segment | OFFICIAL | 0.086/1.000/0.158/0.086 | 0.086/1.000/0.158/0.086 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000008_9x16 | complex_multi_segment | **SALVAGED** | 0.342/0.985/0.507/0.340 | 0.234/1.000/0.379/0.234 | 67.5%→100.0% | -0.107/+0.015/-0.128/-0.106 |
| qvh_000134_9x16 | complex_multi_segment | OFFICIAL | 0.217/1.000/0.357/0.217 | 0.217/1.000/0.357/0.217 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000089_9x16 | complex_multi_segment | OFFICIAL | 0.381/0.735/0.502/0.335 | 0.406/0.997/0.577/0.406 | 66.6%→84.8% | +0.025/+0.263/+0.076/+0.071 |
| qvh_000829_9x16 | complex_multi_segment | OFFICIAL | 0.499/1.000/0.666/0.499 | 0.504/1.000/0.670/0.504 | 100.0%→99.0% | +0.005/0.000/+0.004/+0.005 |
| qvh_000358_9x16 | complex_multi_segment | OFFICIAL | 0.611/1.000/0.759/0.611 | 0.611/1.000/0.759/0.611 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000553_9x16 | boundary_or_difficult | OFFICIAL | 0.854/0.916/0.884/0.792 | 0.825/1.000/0.904/0.825 | 88.5%→100.0% | -0.029/+0.084/+0.020/+0.033 |
| qvh_000721_9x16 | boundary_or_difficult | OFFICIAL | 0.619/1.000/0.765/0.619 | 0.565/0.680/0.617/0.446 | 100.0%→74.6% | -0.055/-0.320/-0.148/-0.173 |
| qvh_000363_9x16 | boundary_or_difficult | OFFICIAL | 0.214/1.000/0.352/0.214 | 0.214/1.000/0.352/0.214 | 100.0%→100.0% | 0.000/0.000/0.000/0.000 |
| qvh_000574_9x16 | boundary_or_difficult | **SALVAGED** | 0.988/1.000/0.994/0.988 | 1.000/0.967/0.983/0.967 | 100.0%→95.5% | +0.012/-0.033/-0.011/-0.022 |
| qvh_000915_9x16 | boundary_or_difficult | OFFICIAL | 0.544/1.000/0.705/0.544 | 0.680/1.000/0.810/0.680 | 26.2%→20.9% | +0.136/0.000/+0.105/+0.136 |

9/20 指标完全相同；7/20 以 F1/tIoU 看有改善；`781`、`008`、`721` 退化，`574` 是边界更窄但漏掉尾部 1.031 s 的近似持平/轻微退化。提升不是系统性质变。

### 12.1 qvh_000008 salvage 明细

| Quantity | Value |
|---|---:|
| Prediction union | [0.000, 21.563] s |
| Prediction duration | 21.563000 s |
| Reference union | [0.959292,1.584917], [9.926583,10.260250], [16.474792,20.562208] s |
| Reference duration | 5.046708 s |
| Intersection | 5.046708 s |
| Duration union | 21.563000 s |
| Precision / Recall / F1 / tIoU | 0.234045 / 1.000000 / 0.379313 / 0.234045 |
| Coverage | 99.9989% |
| Over-prediction / missed-reference | 16.516292 s / 0.000000 s |
| Raw candidates / temporal segments | 6 / 6 |

相对 4B，9B 覆盖更多、更宽：4B coverage 67.5%，9B temporal union 几乎全片。4B 留出 3–10 s 空白；9B 用 6 个首尾相接的候选把自拍、水面切换、沉船鱼群、游泳、珊瑚细节全部解释为“视觉表现突出/信息量提升”，语义标准更宽而非收紧。不过该视频的水下沉船、鱼群、珊瑚和游泳本身有合理观看价值，weak reference 只标少量片段，故不能仅凭 Precision 判断 9B 能力；可以确定的是 9B 没有缓解这条 ambiguity，且工程 schema 失败。

### 12.2 qvh_000574 salvage 明细

| Quantity | Value |
|---|---:|
| Prediction union | [0.000, 30.000] s |
| Prediction duration | 30.000000 s |
| Reference union | [0.000, 31.031] s |
| Reference duration | 31.031000 s |
| Intersection | 30.000000 s |
| Duration union | 31.031000 s |
| Precision / Recall / F1 / tIoU | 1.000000 / 0.966775 / 0.983107 / 0.966775 |
| Coverage | 95.5473% |
| Over-prediction / missed-reference | 0.000000 s / 1.031000 s |
| Raw candidates / temporal segments | 7 / 7 |

9B 的 temporal union 仍覆盖几乎整段视频，语义上保持了这一 positive control 的高质量结果，但 Recall **不是 1**：parse failure 中止了 pipeline，未产生第二个 overlap chunk 响应，已有 raw 只到 30 s。与 4B 的全片 union 相比边界接近，F1/tIoU 仅低约 0.011/0.022。9B 把相同的“拿取—进食—再次拿取”重复互动按 3–5 s 阶段拆成 7 段，说明它识别了连续互动，却没有做事件级去重，也没有遵守最多 5 段和 score 必填约束。语义/时间表现接近正确，不等于工程输出合规。

## 13. 五个严重过召回压力样本

| Sample | 判定 | 证据 |
|---|---|---|
| qvh_000602_9x16 | 改善但不充分 | P/tIoU 0.107→0.180，coverage 100%→59.5%；仍保留“准备、动作和落水结果”的宽段及“动态丰富”的水中视角。 |
| qvh_000949_9x16 | 有限改善 | P/tIoU 0.117→0.170，coverage 100%→68.9%；仍把场景切换、走向长椅、展示食物和“完整小事件”当作高光。 |
| qvh_000629_9x16 | 不变 | P/tIoU 仍 0.086，coverage 仍 100%；房间环境、走动和手势仍被独立保留。 |
| qvh_000781_9x16 | 退化 | P/tIoU 0.141→0.133，coverage 94.8%→100%；行走、涂鸦特写和远景覆盖全片。 |
| qvh_000363_9x16 | 不变 | P/tIoU 仍 0.214，coverage 100%；清洗大米、窗外城市、烹饪过程全部保留。 |

只有 2/5 明显缩窄，2/5 完全不变，1/5 退化；H1 所需的系统性收紧没有出现。

## 14. Recall Guardrails

| Sample | 4B Recall | 9B Recall | 结论 |
|---|---:|---:|---|
| qvh_000023_9x16 | 1.000 | 1.000 official | 正对照完全保持。 |
| qvh_000574_9x16 | 1.000 | 0.967 salvaged | 时间语义几乎保持，但 official schema FAIL；不得写成 Recall=1。 |
| qvh_000089_9x16 | 0.735 | 0.997 official | 基本恢复漏掉的短动作，但 coverage 66.6%→84.8%，Precision 只小升。 |
| qvh_000721_9x16 | 1.000 | 0.680 official | 明显 Recall 回归；缩窄时漏掉 weak-reference 边界。 |

9B 没有形成稳定的 Recall guardrail：一条显著恢复，一条明显退化，一条正对照因 schema failure 无 formal metrics。

## 15. Weak-reference Ambiguity

- `qvh_000067_9x16`：reference 主要标起跳/着陆，连续蹦极过程也可能具有持续观看价值。两模型均全片覆盖、P=0.183；不能单凭 P 判错。
- `qvh_000008_9x16`：珊瑚、沉船、鱼群、水下游动都可能是合理审美型高光；9B 全片覆盖证明它未收紧，但低 P 不等价于能力差。
- `qvh_000134_9x16`：高空地貌、云层与光影缺少可复现的已标/未标语义边界；两模型结果完全一致且全片覆盖。

这些样本应作为 reference-risk 单独解释，不能替 `629/781/363` 等普通内容过召回证据免责，也不能用单点 Precision 排名模型。

## 16. Raw Reason 对比

9B 的 reason 更短、更像局部事件摘要，但选择标准未根本变化：

- **场景变化**：`949` 的“场景切换至公园”、`363` 的“场景切换至窗外城市景观”、`008` 的“场景切换至水面”。
- **视觉丰富/突出**：`363` 的城市景观、`134` 的云层与光影、`008` 的沉船鱼群/珊瑚/游泳。
- **普通走动/操作**：`629` 的房间展示、走动和手势；`781` 的行走/回头；`089` 的咖啡机操作。
- **准备和完整过程**：`602` 明确包含准备—动作—落水结果；`949` 仍用“完整小事件”；`574` 把重复互动的完整过程拆成 7 段。

`000008` 上，4B 是 4 个不连续候选，9B 是 6 个相邻候选覆盖全片；`000574` 上，4B 用 5+1 个 chunk candidates 覆盖全片，9B 在第一个 chunk 就拆成 7 段。规模增大减少了总体 raw candidates（78→65，diagnostic 20），但没有稳定改变 v0 的宽松高光语义。

## 17. 性能成本

| Metric | 4B | 9B |
|---|---:|---:|
| Mean inference，全部已请求样本 | 3.216 s/sample | 3.956 s/sample |
| Matched-18 mean inference | 3.112 s/sample | 3.908 s/sample |
| Relative matched-18 latency | baseline | +25.6% |
| Peak VRAM | 未留存可比监控值 | 23781 MiB |
| GPU capacity | 24564 MiB | 24564 MiB |
| Peak headroom | N/A | 约 783 MiB |
| OOM / timeout / engine crash during successful run | 未见 | 0 / 0 / 0 |

RTX 4090 D 24GB 足以在 BF16、无量化下部署 9B 并完成多模态请求，但峰值余量不足 1 GB，属于“可运行、余量小”。以约 23–26% 推理时延增加换来的 temporal aggregate 增益只有 F1 +0.0065、tIoU +0.0048（且是 diagnostic），当前性价比不足以支持立即切换主线。

## 18. 最终结论

### 模型能力

**9B 相比 4B：小幅更好，但不稳定。** matched-18 的 P/F1/tIoU 分别提升 0.0130/0.0149/0.0124；salvaged-20 仅提升 0.0069/0.0065/0.0048。9B 确实能在 `602/949/915` 局部缩窄区间，并在 `089` 恢复 Recall。

### 提升幅度

没有达到质变。9/20 完全相同，五个严重压力样本只有 2 个明显改善；full-video coverage 在 diagnostic 中仍有 11/20。

### Prompt 问题

H2 得到更强支持：9B 仍被 v0 的“场景变化、视觉表现、信息量、完整小事件”等宽松定义驱动。当前系统性过召回主要不能靠模型扩容解决。

### 工程稳定性

9B structured output 更差：4B 20/20 schema success，9B 18/20；两条失败均全部缺 score 且候选数超限。模型理解了视频和时间内容，但 end-to-end pipeline 仍失败。

### 部署成本

24GB 足够，但 9B 峰值 23781/24564 MiB，仅约 783 MiB 余量；无 OOM 不代表有充足安全裕度。

### 主线模型

**目前不值得立即从 4B 切到 9B。** 9B 的小幅 temporal 改善不足以抵消 schema success 从 100% 降到 90%、约 25.6% matched latency 增加和显存余量过小。主线应先保持 4B，并做 Prompt v1 单变量实验；Prompt v1 稳定后再用同一 Prompt 复测 9B。

## 19. 对 Prompt v1 的影响

本轮不修改 Prompt。下一版应按真实证据解决：

1. 明确场景切换、画面好看、信息丰富、普通走动/操作本身不是充分高光条件。
2. 以峰值动作、显著结果、强情绪/冲突、关键转折或任务成败作为正条件。
3. 强制区分准备—峰值—结果，默认输出最短可理解峰值；不可分割时才扩边界。
4. 给出负例排除：房间/城市环境展示、普通行走、静态特写、常规烹饪步骤、重复互动阶段不得独立成段。
5. 保持 Recall guardrail，但禁止“不确定就全留”；用 `023/574/089/721` 预注册护栏。
6. 加强格式自检：每段必须有数值 score、每 chunk ≤5 段、输出前验证 JSON 字段完整性。格式护栏与语义收紧应在设计文档中明确区分，便于解释 A/B 结果。

## 20. Stage 2 下一步

唯一下一步：**Prompt v1 单变量实验**。先冻结模型为当前主线 4B，只改 Prompt，执行同一 Baseline 20 A/B；第一门槛是 20/20 schema success，随后比较 Recall guardrails、严重过召回压力样本、coverage、over-prediction、P/F1/tIoU。不得在同一轮同时改变模型、FPS、chunk、merge、score threshold 或 parser。

## 可复现性与长期文件

- `docs/experiments/qwen35_4b_vs_9b_baseline20.md`
- `docs/experiments/qwen35_4b_vs_9b_baseline20.csv`
- `docs/experiments/qwen35_4b_vs_9b_model_scale_probe_report.md`
- `docs/experiments/data/qwen35_9b_salvaged_temporal_diagnostic.json`
- `scripts/analysis/salvage_qwen35_9b_temporal.py`

以上文件只保存小型报告和派生诊断；模型权重、cache、raw output 大目录、图片和 GPU 日志不进入 Git。
