# Baseline 20 Error Analysis

## 1. 分析目的

本报告对冻结的 Baseline 20 做逐视频、逐预测区间的离线证据化误差分析，目标是解释 weak-reference development metrics 中 Recall 很高（0.9730），但 Precision（0.4295）和 Temporal IoU（0.4215）明显偏低的原因。所有语义结论均联合使用 source video 代表帧、weak-reference、raw response/reason/score、chunk 与最终 prediction；仅由时间区间可确认的事实和需要画面支持的语义判断被明确区分。

## 2. Baseline 固定配置

| 项目 | 值 |
|---|---|
| Baseline 运行时 HEAD | `c324690197758081eb6a74f4d94069302ead44a9` |
| Baseline 截断修复正式 commit | `299cf8a790b365d478a45d84e797c53b9a1fb896` |
| 当前分析 HEAD | `f56e833ba9a4e9d1d6538f34518f9fbefa286c68` |
| 模型 | `Qwen/Qwen3.5-4B` |
| revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Prompt | `high_recall_retrieval_v0` |
| sampling FPS | 2.0 |
| chunk / overlap | 30 s / 5 s |
| merge threshold | temporal IoU 0.5 |
| Baseline 状态 | 20/20 SUCCESS；26/26 chunks finish_reason=stop；parse failure=0；truncation=0 |

三类 HEAD 分别代表运行基线、截断修复与当前分析状态，不能互换。本报告没有重跑推理，也没有启动 vLLM。

## 3. 方法

1. 读取 `metrics.json`、`predictions.jsonl`、20 个 `samples/*.json`、20 个 `raw/*.json`、`configs/baseline_20.json` 与 frozen index。
2. 根据 manifest 的真实 `source_video_path` 和 `clip_start_sec/clip_end_sec` 定位 20 条源视频；20/20 均存在。
3. 系统未提供 `ffmpeg`/`ffprobe`，因此未安装新依赖，改用现有 OpenCV 5.0.0 读取 duration/FPS/resolution，并在 prediction/reference 起止点、中点及 3–5 s 均匀点抽帧生成 contact sheet。
4. contact sheet 色框：绿色=prediction∩reference，红色=prediction-only，蓝色=reference-only，灰色=均不覆盖。所有临时图位于仓库外 `/root/autodl-tmp/outputs/experiments/baseline20_error_analysis_tmp/`。
5. 诊断量按区间 union 计算：coverage=prediction union/video duration；over-prediction=prediction−intersection；missed reference=reference−intersection。
6. raw candidate 与 merge 后结果逐项比对；score 诊断采用 weak-reference overlap fraction：≥0.5 为 TP-like，≤0.1 为 FP-like，其余为 Mixed。该标签仅用于诊断，不等同于人工语义真值，尤其要避开 E10 样本的误导。

## 4. 总体错误统计

总体上，14/20 prediction union 覆盖 100% 视频，15/20 覆盖至少 90%，平均 coverage 为 89.8%。累计 over-prediction 238.26 s，而 missed-reference 仅 3.13 s。这一数量级差异直接解释了 Recall 高、Precision/tIoU 低。

以下为每条视频唯一 primary category 的频次：

| Error Type | Count | Percentage | Representative Samples |
|---|---:|---:|---|
| E1 普通动作 / 普通内容过召回 | 2 | 10.0% | qvh_000629_9x16, qvh_000363_9x16 |
| E3 事件结束后拖尾 | 1 | 5.0% | qvh_000915_9x16 |
| E5 区间整体过宽 / 近全视频覆盖 | 6 | 30.0% | qvh_000602_9x16, qvh_000949_9x16, qvh_000781_9x16 |
| E7 真正漏召回 | 1 | 5.0% | qvh_000089_9x16 |
| E8 边界偏移 | 4 | 20.0% | qvh_000531_9x16, qvh_000553_9x16, qvh_000721_9x16 |
| E10 Weak-reference 可能不完整或有歧义 | 3 | 15.0% | qvh_000067_9x16, qvh_000008_9x16, qvh_000134_9x16 |
| E11 模型语义判断明显过宽 | 1 | 5.0% | qvh_000612_9x16 |
| E12 无明显错误 / 高度一致 | 2 | 10.0% | qvh_000023_9x16, qvh_000574_9x16 |

### 分组汇总

| Group | N | Mean P | Mean R | Mean tIoU | Mean Coverage | Raw candidates/video | Final segments/video | 证据化规律 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single_segment | 5 | 0.372 | 1.000 | 0.372 | 100.0% | 4.8 | 4.4 | 5/5 全片覆盖；4/5 为 E5，说明单 reference 并未阻止模型把全片切成多个候选。 |
| double_segment | 5 | 0.292 | 0.965 | 0.282 | 89.4% | 3.6 | 3.4 | 3/5 全片覆盖、1/5 覆盖 94.8%；两个短事件之间的普通片段经常未被抑制。 |
| complex_multi_segment | 5 | 0.410 | 0.944 | 0.400 | 86.8% | 3.4 | 3.2 | 既有 E10 标注歧义，也有 E7 真漏与场景级粗边界；问题不是单一来源。 |
| boundary_or_difficult | 5 | 0.644 | 0.983 | 0.631 | 82.9% | 3.8 | 3.8 | 包含最佳对照，也包含普通内容过召回；边界收缩价值最清晰。 |

## 5. 20 条逐样本分析

### qvh_000023_9x16

- Group：`single_segment`
- 指标：P=0.783, R=1.000, F1=0.879, tIoU=0.783
- Reference：[0.000, 4.104]
- Prediction：[0.000, 5.239]
- Prediction coverage：100.0%；over=1.13 s；missed=0.00 s
- Segment count：reference=1；raw=1；final=1
- Raw model behavior：模型以 0.92 分将整段寿司特写视为视觉吸引力高光。
- 视频观察：全片均为同一盘寿司的连续特写；reference 覆盖 0.00–4.10 s，prediction 仅多出结尾约 1.13 s，画面语义没有明显突变。
- 主要错误：E12 无明显错误 / 高度一致
- 次要错误：无
- Root cause：PRIMARY=R（仅解释 1.13 s 的 weak-reference 端点差异；不视为模型主错误）
- Confidence：HIGH
- 下一步启示：作为正对照保留；后续任何实验应避免使其 Recall 下降。

### qvh_000337_9x16

- Group：`single_segment`
- 指标：P=0.387, R=1.000, F1=0.559, tIoU=0.387
- Reference：[7.841, 13.013]
- Prediction：[0.000, 8.000]; [8.000, 13.347]
- Prediction coverage：100.0%；over=8.17 s；missed=0.00 s
- Segment count：reference=1；raw=2；final=2
- Raw model behavior：0–8 s 的普通户外讲话被 0.85 分判为“主要视觉内容”，8 s 后推文信息卡被 0.95 分判为核心信息点。
- 视频观察：reference 仅覆盖约 7.84–13.01 s 的推文截图；前半段为持续讲话与手势，prediction 因两个相邻候选覆盖全片。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E1; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 明确排除仅维持叙事背景的普通讲话，并要求候选边界围绕信息峰值。

### qvh_000757_9x16

- Group：`single_segment`
- 指标：P=0.466, R=1.000, F1=0.636, tIoU=0.466
- Reference：[7.520, 22.480]
- Prediction：[0.000, 4.500]; [4.500, 7.000]; [7.000, 11.000]; [11.000, 23.500]; [23.500, 31.000]; [31.000, 32.120]
- Prediction coverage：100.0%；over=17.16 s；missed=0.00 s
- Segment count：reference=1；raw=7；final=6
- Raw model behavior：模型分别以 0.82–0.92 分保留饮料、夜景、洗手间独白、机场与机舱切换，理由反复使用“视觉吸引力/场景变化/信息量”。
- 视频观察：reference 为约 7.52–22.48 s 的洗手间人物独白；前段饮料/夜景与后段机场/机舱均被纳入，prediction union 覆盖全片。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E1; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 增加“场景切换本身不是高光”及普通旅行 B-roll 负例。

### qvh_000602_9x16

- Group：`single_segment`
- 指标：P=0.107, R=1.000, F1=0.193, tIoU=0.107
- Reference：[27.661, 31.098]
- Prediction：[0.000, 4.800]; [4.800, 11.800]; [11.800, 15.500]; [15.500, 29.800]; [25.000, 29.500]; [29.500, 32.132]; [29.800, 30.000]
- Prediction coverage：100.0%；over=28.70 s；missed=0.00 s
- Segment count：reference=1；raw=7；final=7
- Raw model behavior：模型把自拍、聚餐、泳池嬉戏、跳水准备、跳水与庆祝均列为高光；15.5–29.8 s 单候选已远宽于实际跳水瞬间。
- 视频观察：reference 只标约 27.66–31.10 s 的跳水/落水；0–12 s 是自拍和聚餐，12–27 s 多为泳池活动与长准备，prediction union 覆盖全片。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E2; E3; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 收紧“高潮事件”定义，明确准备/普通社交不应自动纳入，并以 Recall 约束验证。

### qvh_000949_9x16

- Group：`single_segment`
- 指标：P=0.117, R=1.000, F1=0.210, tIoU=0.117
- Reference：[12.800, 16.567]
- Prediction：[0.000, 4.000]; [4.000, 10.000]; [10.000, 15.000]; [15.000, 20.000]; [20.000, 30.000]; [30.000, 32.133]
- Prediction coverage：100.0%；over=28.37 s；missed=0.00 s
- Segment count：reference=1；raw=7；final=6
- Raw model behavior：0.75–0.95 分候选覆盖停车场、步行、摆放食物、进食互动和自拍；理由多次直接写“铺垫/准备”。
- 视频观察：reference 约 12.80–16.57 s，集中于食物到桌与开始准备；前 10 s 是步行铺垫，20 s 后是持续进食/聊天，prediction 覆盖全片。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E2; E3; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 明令铺垫与常规持续行为不单独成候选；先不做 score threshold。

### qvh_000531_9x16

- Group：`double_segment`
- 指标：P=0.505, R=0.824, F1=0.626, tIoU=0.456
- Reference：[3.971, 5.772]; [6.406, 7.608]
- Prediction：[4.500, 9.400]
- Prediction coverage：52.1%；over=2.43 s；missed=0.53 s
- Segment count：reference=2；raw=1；final=1
- Raw model behavior：模型以 0.92 分定位 4.5–9.4 s 的喷雾整理头发连续动作。
- 视频观察：reference 为 3.97–5.77 s 与 6.41–7.61 s；prediction 找到正确动作，但错过第一段开头约 0.53 s，并延续到动作后段。
- 主要错误：E8 边界偏移
- 次要错误：E3; E7
- Root cause：PRIMARY=M
- Confidence：HIGH
- 下一步启示：Boundary Refinement：围绕动作首次出现与动作完成点做边界收缩，同时监控 Recall。

### qvh_000612_9x16

- Group：`double_segment`
- 指标：P=0.546, R=1.000, F1=0.707, tIoU=0.546
- Reference：[0.000, 2.878]; [3.045, 7.549]
- Prediction：[0.000, 10.000]; [10.000, 11.500]; [11.500, 13.514]
- Prediction coverage：100.0%；over=6.13 s；missed=0.00 s
- Segment count：reference=2；raw=3；final=3
- Raw model behavior：表演主体 0–10 s 得 0.92；10–11.5 s 仅因路人遮挡造成动态变化也得 0.85；尾段低头被解释为情感高潮。
- 视频观察：全片是同一舞台表演，reference 到约 7.55 s；后半仍是同一表演，但包含遮挡与较弱动作，weak-reference 的截止点也有一定主观性。
- 主要错误：E11 模型语义判断明显过宽
- 次要错误：E3; E10
- Root cause：PRIMARY=P；组合=P+M+R
- Confidence：MEDIUM
- 下一步启示：Prompt v1 排除遮挡/路过等无叙事价值变化；并对持续表演样本做人审标注复核。

### qvh_000067_9x16

- Group：`double_segment`
- 指标：P=0.183, R=1.000, F1=0.310, tIoU=0.183
- Reference：[0.167, 3.670]; [27.319, 29.696]
- Prediction：[0.000, 5.000]; [5.000, 13.000]; [13.000, 20.000]; [20.000, 27.000]; [25.000, 27.000]; [27.000, 30.000]; [29.000, 32.115]
- Prediction coverage：100.0%；over=26.23 s；missed=0.00 s
- Segment count：reference=2；raw=8；final=7
- Raw model behavior：模型以 0.80–0.95 分连续描述准备、下坠、反弹、空中姿态、接近平台和着陆，语义链完整。
- 视频观察：全片是一次连续蹦极/高空下坠事件；reference 只标起跳与末端着陆，遗漏中段自由落体和反弹等最具高光价值的核心过程。
- 主要错误：E10 Weak-reference 可能不完整或有歧义
- 次要错误：E5
- Root cause：PRIMARY=R
- Confidence：HIGH
- 下一步启示：优先做 weak-reference 多人复核或事件级完整标注；该样本不宜直接驱动阈值收紧。

### qvh_000781_9x16

- Group：`double_segment`
- 指标：P=0.141, R=1.000, F1=0.247, tIoU=0.141
- Reference：[0.000, 1.468]; [17.918, 19.019]
- Prediction：[0.000, 3.500]; [4.000, 17.000]; [17.500, 19.253]
- Prediction coverage：94.8%；over=15.68 s；missed=0.00 s
- Segment count：reference=2；raw=3；final=3
- Raw model behavior：中段 4–17 s 的空坡道以 0.88 分因“视觉表现突出、场景变化、信息量”入选。
- 视频观察：reference 仅覆盖开头与结尾人物奔跑；中间约 13 s 是镜头沿涂鸦坡道移动、主体人物基本不在画面，prediction coverage=94.8%。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E11; E10
- Root cause：PRIMARY=P；组合=P+M+R
- Confidence：HIGH
- 下一步启示：Prompt v1 要求候选与可说明的核心事件或叙事峰值绑定，排除单纯镜头运动。

### qvh_000629_9x16

- Group：`double_segment`
- 指标：P=0.086, R=1.000, F1=0.158, tIoU=0.086
- Reference：[0.833, 1.733]; [14.500, 15.500]
- Prediction：[0.000, 4.500]; [4.500, 13.500]; [13.500, 22.200]
- Prediction coverage：100.0%；over=20.30 s；missed=0.00 s
- Segment count：reference=2；raw=3；final=3
- Raw model behavior：4.5–13.5 s 普通房间扫镜被 0.85 分判为“丰富视觉信息”，13.5–22.2 s 普通介绍被 0.88 分保留。
- 视频观察：reference 是开头短暂人物出现与约 14.5–15.5 s 再次出现；大部分房间、天花板、床与装饰扫镜没有明显事件峰值。
- 主要错误：E1 普通动作 / 普通内容过召回
- 次要错误：E5; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 加入普通 room tour/扫镜负例，并要求输出局部峰值而非整段介绍。

### qvh_000008_9x16

- Group：`complex_multi_segment`
- 指标：P=0.342, R=0.985, F1=0.507, tIoU=0.340
- Reference：[0.959, 1.585]; [9.927, 10.260]; [16.475, 20.562]
- Prediction：[0.000, 3.000]; [10.000, 13.000]; [13.000, 17.000]; [17.000, 21.563]
- Prediction coverage：67.5%；over=9.59 s；missed=0.07 s
- Segment count：reference=3；raw=4；final=4
- Raw model behavior：模型以 0.80–0.92 分选出潜水员、水面鱼群、珊瑚/沉船与水下游动，理由具体且与画面一致。
- 视频观察：未标注的珊瑚、沉船、鱼群和水下游动同样具有明确观赏价值；仅约 0.07 s reference 因 10.0 s 候选起点造成边界漏失。
- 主要错误：E10 Weak-reference 可能不完整或有歧义
- 次要错误：E8; E11
- Root cause：PRIMARY=R
- Confidence：HIGH
- 下一步启示：先做人审 reference 复核；边界实验不应把这些合理水下片段全部当负例。

### qvh_000134_9x16

- Group：`complex_multi_segment`
- 指标：P=0.217, R=1.000, F1=0.357, tIoU=0.217
- Reference：[3.875, 5.292]; [9.875, 10.292]; [11.708, 12.125]; [13.875, 15.042]
- Prediction：[0.000, 15.750]
- Prediction coverage：100.0%；over=12.33 s；missed=0.00 s
- Segment count：reference=4；raw=1；final=1
- Raw model behavior：模型以 0.72 分将整段云层、光影和清晰度变化视为一个视觉候选。
- 视频观察：全片均为连续航拍云层/光影，weak-reference 仅间歇标出四段，未标与已标画面之间缺乏稳定可见的语义分界。
- 主要错误：E10 Weak-reference 可能不完整或有歧义
- 次要错误：E5; E11
- Root cause：PRIMARY=R
- Confidence：HIGH
- 下一步启示：对抽象风景类建立多人标注与一致性规则；在此之前不应用阈值裁掉整段。

### qvh_000089_9x16

- Group：`complex_multi_segment`
- 指标：P=0.381, R=0.735, F1=0.502, tIoU=0.335
- Reference：[3.270, 4.471]; [6.840, 9.476]; [12.646, 14.248]; [15.749, 16.016]
- Prediction：[2.000, 4.000]; [4.000, 10.000]; [10.000, 13.000]
- Prediction coverage：66.6%；over=6.81 s；missed=1.51 s
- Segment count：reference=4；raw=3；final=3
- Raw model behavior：模型识别咖啡胶囊、核心操作与合盖设置，但在 13 s 截止，没有输出后续操作。
- 视频观察：reference 的 12.65–14.25 s 后半与 15.75–16.02 s 仍有手部调整/收尾动作；prediction 未覆盖，共漏约 1.51 s。
- 主要错误：E7 真正漏召回
- 次要错误：E8
- Root cause：PRIMARY=M
- Confidence：HIGH
- 下一步启示：Boundary Refinement 加短动作结束点复核；评估采样粒度对尾部微动作的影响。

### qvh_000829_9x16

- Group：`complex_multi_segment`
- 指标：P=0.499, R=1.000, F1=0.666, tIoU=0.499
- Reference：[0.000, 5.589]; [9.426, 10.177]; [11.803, 14.389]; [15.641, 21.563]; [29.404, 30.572]
- Prediction：[0.000, 11.500]; [11.500, 15.500]; [15.500, 23.000]; [23.000, 28.500]; [28.500, 32.115]
- Prediction coverage：100.0%；over=16.10 s；missed=0.00 s
- Segment count：reference=5；raw=6；final=5
- Raw model behavior：模型把三件衣物展示、穿着效果、一般肢体动作、再次拿衣物均以 0.72–0.95 分输出，最终 union 覆盖全片。
- 视频观察：reference 选择若干展示动作，未标间隙包含持续整理、说话和姿势调整；23–28.5 s 主要是一般肢体/穿着调整，区分度较低。
- 主要错误：E5 区间整体过宽 / 近全视频覆盖
- 次要错误：E6; E11; E10
- Root cause：PRIMARY=P；组合=P+M+R
- Confidence：MEDIUM
- 下一步启示：Prompt v1 要求具体完成态/转折，不以一般肢体变化成段；另做商品展示类标注复核。

### qvh_000358_9x16

- Group：`complex_multi_segment`
- 指标：P=0.611, R=1.000, F1=0.759, tIoU=0.611
- Reference：[0.000, 2.586]; [4.504, 6.173]; [7.841, 9.593]; [12.429, 14.598]; [17.476, 24.566]
- Prediction：[0.000, 4.000]; [4.000, 17.000]; [17.000, 24.983]
- Prediction coverage：100.0%；over=9.72 s；missed=0.00 s
- Segment count：reference=5；raw=3；final=3
- Raw model behavior：模型按佛罗伦萨、埃菲尔铁塔和室内三大场景输出整段，理由依赖地标、笑容、表情与信息量。
- 视频观察：核心地标/人物段基本命中，但 prediction 将每个场景整体保留，连同过渡和普通讲话；reference 在场景内部更稀疏。
- 主要错误：E8 边界偏移
- 次要错误：E5; E11
- Root cause：PRIMARY=M；组合=M+P+R
- Confidence：MEDIUM
- 下一步启示：Boundary Refinement 或 Prompt 边界约束，使候选围绕场景内峰值而非整场景。

### qvh_000553_9x16

- Group：`boundary_or_difficult`
- 指标：P=0.854, R=0.916, F1=0.884, tIoU=0.792
- Reference：[0.000, 0.000]; [1.902, 14.014]
- Prediction：[0.000, 2.000]; [2.000, 4.000]; [4.000, 7.000]; [7.000, 9.000]; [9.000, 13.000]
- Prediction coverage：88.5%；over=1.90 s；missed=1.01 s
- Segment count：reference=2；raw=5；final=5
- Raw model behavior：模型连续输出人物开场、行车、码头、城市与海滨景观，最后停在 13 s。
- 视频观察：有效 reference 为 1.90–14.01 s，prediction union 为 0–13 s；末尾约 1.01 s 的人物自拍/收尾漏掉，开头约 1.90 s 未标人物镜头被纳入。另有一个 0–0 零时长 reference。
- 主要错误：E8 边界偏移
- 次要错误：E7; E10
- Root cause：PRIMARY=M；组合=M+R
- Confidence：HIGH
- 下一步启示：Boundary Refinement，特别检查视频末尾镜头；同时清理/审计零时长 reference。

### qvh_000721_9x16

- Group：`boundary_or_difficult`
- 指标：P=0.619, R=1.000, F1=0.765, tIoU=0.619
- Reference：[0.000, 0.120]; [1.880, 5.120]; [6.880, 10.120]; [11.880, 15.040]
- Prediction：[0.000, 2.000]; [2.000, 7.000]; [7.000, 12.000]; [12.000, 15.760]
- Prediction coverage：100.0%；over=6.00 s；missed=0.00 s
- Segment count：reference=4；raw=4；final=4
- Raw model behavior：模型按四组人物车辆卡片以 0.75/0.85 分切段，但把转场一并纳入。
- 视频观察：reference 对应四张信息卡主体；prediction 也找到四组，但按 0–2/2–7/7–12/12–15.76 s 粗切，覆盖卡片间转场。
- 主要错误：E8 边界偏移
- 次要错误：E5; E11
- Root cause：PRIMARY=M；组合=M+P
- Confidence：HIGH
- 下一步启示：Boundary Refinement 对静态卡片出现/消失帧做收缩。

### qvh_000363_9x16

- Group：`boundary_or_difficult`
- 指标：P=0.214, R=1.000, F1=0.352, tIoU=0.214
- Reference：[0.760, 4.480]; [23.280, 25.000]
- Prediction：[0.000, 9.000]; [9.000, 18.000]; [18.000, 25.440]
- Prediction coverage：100.0%；over=20.00 s；missed=0.00 s
- Segment count：reference=2；raw=3；final=3
- Raw model behavior：中间 9–18 s 城市阳台景观仅因“显著视觉变化”得 0.75；18–25.44 s 整段烹饪也被保留。
- 视频观察：reference 仅覆盖开头淘米/捞米约 0.76–4.48 s 与末尾加水约 23.28–25.00 s；中间长城市扫镜和烹饪准备未标，prediction 覆盖全片。
- 主要错误：E1 普通动作 / 普通内容过召回
- 次要错误：E5; E11
- Root cause：PRIMARY=P；组合=P+M
- Confidence：HIGH
- 下一步启示：Prompt v1 排除无事件结果的环境建立镜头与普通连续步骤。

### qvh_000574_9x16

- Group：`boundary_or_difficult`
- 指标：P=0.988, R=1.000, F1=0.994, tIoU=0.988
- Reference：[0.000, 31.031]
- Prediction：[0.000, 5.000]; [5.000, 10.000]; [10.000, 15.000]; [15.000, 20.000]; [20.000, 30.000]; [25.000, 31.398]
- Prediction coverage：100.0%；over=0.37 s；missed=0.00 s
- Segment count：reference=1；raw=6；final=6
- Raw model behavior：模型以 0.72–0.85 分连续描述两个女孩进食和制作食物，覆盖整段。
- 视频观察：reference 为 0–31.03 s，几乎覆盖全片；prediction 到 31.40 s，仅多约 0.37 s，视觉内容始终一致。
- 主要错误：E12 无明显错误 / 高度一致
- 次要错误：无
- Root cause：PRIMARY=R（仅解释约 0.37 s 端点差异；不视为模型主错误）
- Confidence：HIGH
- 下一步启示：作为正对照保留，检查 Prompt 收紧后是否仍能保住连续完整事件。

### qvh_000915_9x16

- Group：`boundary_or_difficult`
- 指标：P=0.544, R=1.000, F1=0.705, tIoU=0.544
- Reference：[0.000, 1.360]
- Prediction：[0.000, 2.500]
- Prediction coverage：26.2%；over=1.14 s；missed=0.00 s
- Segment count：reference=1；raw=1；final=1
- Raw model behavior：模型以 0.92 分定位 0–2.5 s 的金属碗展示。
- 视频观察：reference 为 0–1.36 s，展示动作命中；1.36–2.5 s 人物已转入普通说话/放松状态，prediction 明显拖尾。
- 主要错误：E3 事件结束后拖尾
- 次要错误：E8
- Root cause：PRIMARY=M
- Confidence：HIGH
- 下一步启示：Boundary Refinement 收紧动作完成点，并保留起点 Recall。

## 6. Recall < 1.0 专项分析

| Video | Recall | Missed duration | 证据化结论 |
|---|---:|---:|---|
| qvh_000531_9x16 | 0.824 | 0.53 s | 喷雾整理头发事件已找对；第一段 reference 起点比 prediction 早约 0.53 s，属于边界型漏召回。 |
| qvh_000008_9x16 | 0.985 | 0.07 s | 仅 9.93–10.00 s 的量化边缘未覆盖；主要问题反而是 weak-reference 未标多个合理水下高光。 |
| qvh_000089_9x16 | 0.735 | 1.51 s | 第三段后半和第四个短收尾动作确实未被 prediction 覆盖，是最明确的真实漏召回。 |
| qvh_000553_9x16 | 0.916 | 1.01 s | prediction 在 13 s 结束，遗漏 reference 至 14.01 s 的末尾自拍/收尾；同时 reference 含一个 0–0 零时长段。 |

四条中只有 qvh_000089_9x16 呈现相对明显的语义/短动作漏检；其余三条主要是边界量化。没有证据表明整体 Recall 问题来自 chunk crash、truncation 或 parse failure。

## 7. Double Segment 专项分析

double_segment 组均值 P=0.292、R=0.965、tIoU=0.282，为四组最弱。原因不是“模型看不到第二事件”：5 条中除 qvh_000531_9x16 的边界漏失外，reference 基本都被覆盖。更关键的是两个短 reference 之间/之后的普通内容没有被压制：qvh_000781_9x16 的涂鸦坡道、qvh_000629_9x16 的房间扫镜、qvh_000612_9x16 的遮挡与持续表演均被保留。qvh_000067_9x16 则是反例：中间自由落体本身是合理高光，低 Precision 主要暴露 weak-reference 不完整。故该组的低分由“中间负段识别失败”和“标注结构风险”共同造成，不能单归因于双事件结构或 merge。

## 8. Prompt 影响

Prompt v0 明确写有：“目标不是直接决定最终成片，而是尽可能不要遗漏潜在高光事件。高召回优先，允许召回不确定候选。”同时把“视觉表现突出”“镜头或场景显著变化”列为高光信号，并要求“尽可能返回多个候选”。这与 raw response 构成直接证据链：

- qvh_000629_9x16：普通房间扫镜因“丰富视觉信息”得 0.85。
- qvh_000781_9x16：无主体的坡道镜头因“视觉表现突出、场景变化”得 0.88。
- qvh_000363_9x16：城市阳台背景因“显著视觉变化”得 0.75。
- qvh_000949_9x16：模型理由直接把“铺垫/准备”作为保留依据，最终全片覆盖。

因此 Prompt v1 应为下一轮最高优先级，但目标不是简单“更严格”，而是明确：场景变化、普通运动、铺垫、遮挡和持续过程不自动等于高光；候选应围绕可说明的叙事/动作峰值。同时用 qvh_000023_9x16、qvh_000574_9x16 与 E10 样本做 Recall 护栏。

## 9. Score 诊断

78 个 raw candidates 按 weak-reference overlap fraction 分为：TP-like 33 个（均分 0.85，范围 0.70–0.95）、Mixed 18 个（均分 0.87，范围 0.72–0.92）、FP-like 27 个（均分 0.84，范围 0.72–0.95）。三组分数高度重叠；至少 15 个 ≥0.85 的 FP-like 候选存在，例如 qvh_000949_9x16 的 0.95 进食段、qvh_000781_9x16 的 0.88 坡道段、qvh_000629_9x16 的 0.85 房间扫镜。

结论：单一 score threshold 为 **LOW PRIORITY**。高分误召回很多，阈值会同时损失真实候选，并可能错误惩罚 qvh_000067_9x16/qvh_000134_9x16 等 E10 样本。若未来做 selection，应在 reference 复核与 Prompt 改进后重新校准，优先考虑语义类别而非仅标量阈值。

## 10. Merge 诊断

raw candidates 78 个，final segments 74 个，仅 4 次合并，发生于 qvh_000757_9x16、qvh_000949_9x16、qvh_000067_9x16、qvh_000829_9x16。四次都只把时间上已有重叠的候选合为 union；合并前后的 prediction union 完全相同，没有把两个原本分离的事件错误连接，也没有新增覆盖时长。大量全片覆盖在 raw 阶段已经形成，例如 qvh_000602_9x16、qvh_000629_9x16、qvh_000363_9x16 均无 merge 仍覆盖全片。

结论：merge threshold 为 **LOW PRIORITY**。当前 0.5 阈值只在少量 overlap 候选上起去重作用，不是 Precision/tIoU 低的主要根因。

## 11. Weak-reference 局限

明确 REFERENCE_AMBIGUITY：

- qvh_000067_9x16：reference 只标起跳与着陆，遗漏连续蹦极的自由落体/反弹核心过程。
- qvh_000008_9x16：珊瑚、沉船、鱼群和水下游动均有合理高光价值，但只部分标注。
- qvh_000134_9x16：连续云层/光影画面的已标与未标区间缺少可复现语义边界。

可能存在歧义但仍有模型过召回证据：qvh_000612_9x16、qvh_000781_9x16、qvh_000829_9x16、qvh_000358_9x16。另 qvh_000553_9x16 含 0–0 零时长 reference，应在后续评测数据审计中处理。E10 不能用来替模型普遍免责；本报告仅在视频内容与 raw reason 同时支持时标记。

## 12. 最终根因排序

按 18 条非 E12 样本的 primary root cause 统计：

1. **P — Prompt / 高光定义：9/18（50.0%）**。主导全片覆盖、普通场景变化与准备过程过召回。
2. **M — Model semantic judgment / 边界粒度：6/18（33.3%）**。主导场景级粗边界、尾部拖延及少量短动作漏召回。
3. **R — Weak-reference limitation：3/18（16.7%）**。三条有强视觉证据的 E10。
4. **S — Score / candidate selection：0/18 primary**。分数不可分离，现阶段不是首要根因。
5. **G — Merge / chunk processing：0/18 primary**。4 次 merge 不改变 union；没有证据指向阈值 0.5 是主因。

组合根因仍很常见，尤其 P+M 与 P+M+R；以上排序只统计每条的 PRIMARY ROOT CAUSE。

## 13. Stage 2 下一步实验优先级

| Priority | 实验变量 | 证据 | 预期改善指标 | Recall 风险 | 成本 |
|---:|---|---|---|---|---|
| 1 | Prompt v1：收紧高光语义，加入普通运动/场景切换/铺垫/遮挡负约束；保持单变量 | P 为 50% primary；14/20 全片覆盖；raw reason 直接复述宽松信号 | Precision、tIoU，兼顾 segment count | 中；需 E12/E10 护栏 | 低 |
| 2 | Weak-reference 人工复核/多人一致性小审计 | 3 条强 E10；1 条零时长 reference；低 Precision 部分不是真 FP | 评测可信度、实验结论稳定性 | 无模型 Recall 风险 | 中 |
| 3 | Boundary Refinement（在 Prompt v1 后） | E8 为 4/20 primary；Recall 未满多为边界型 | tIoU、Precision；少量 Recall | 中；收缩过度会漏事件 | 中 |
| 4 | 语义化 candidate filtering / reranking 诊断 | 高分 FP-like 很多，标量 score 不可分 | Precision、segment count | 中到高 | 中 |
| 5 | 单一 score threshold | TP/Mixed/FP-like 分数高度重叠 | 可能小幅 Precision | 高 | 低 |
| 6 | merge threshold | 仅 4 次合并且 union 不变 | 预计改善很小 | 低 | 低 |

下一轮应只执行 Priority 1 的受控 Prompt v1 实验，并同时建立 E12/E10 Recall 护栏；不要在同一轮混入 threshold、merge 或 Boundary Refinement，以保证因果可归因。
