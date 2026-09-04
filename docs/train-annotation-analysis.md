# AIC-VideoHighlight 训练标注结构分析

## 1. 分析目的

本报告用于明确赛事训练标注 `train.jsonl` 的真实 schema、字段含义及其对 Highlight Retrieval 评估的可用性。分析覆盖整个文件；原文件只读，不导出完整标注正文或可识别的视频样例。

复现命令：

```bash
python scripts/inspect_train_annotations.py \
  --input /root/autodl-tmp/annotations/train.jsonl \
  --output docs/train-annotation-analysis.md
```

## 2. 数据基本信息

- 文件：`/root/autodl-tmp/annotations/train.jsonl`
- 文件大小：9,277,172 bytes（8.85 MiB）
- SHA256：`7177731fb7af99e8581c0ec071d116cdb9e6652a6b2b355cd8100364a004c629`
- 总行数：987
- 空行数：0
- 非空行数：987
- 合法顶层 JSON 对象：987
- JSON/顶层类型异常：0
- 顶层字段集合种类：1；完全一致
- 说明：字段值为空与 schema 缺字段是两回事；下文分别统计。

## 3. 顶层字段结构

以下覆盖率分母为全部合法记录。

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `annotation_mode` | string×987 | 987/987（100.00%） | 否 |
| `clip` | object×987 | 987/987（100.00%） | 否 |
| `cropRois` | array×987 | 987/987（100.00%） | 是，363 次 |
| `crop_keyframes` | array×987 | 987/987（100.00%） | 是，363 次 |
| `dataset_split` | string×987 | 987/987（100.00%） | 否 |
| `provenance` | object×987 | 987/987（100.00%） | 否 |
| `quality` | object×987 | 987/987（100.00%） | 否 |
| `schema_version` | string×987 | 987/987（100.00%） | 否 |
| `segments` | array×987 | 987/987（100.00%） | 否 |
| `targetRatioWH` | array×987 | 987/987（100.00%） | 否 |
| `teacher_signals` | object×987 | 987/987（100.00%） | 否 |
| `video_id` | string×987 | 987/987（100.00%） | 否 |
| `video_path` | string×987 | 987/987（100.00%） | 是，987 次 |
| `video_url` | string×987 | 987/987（100.00%） | 是，987 次 |

顶层字段全集：`annotation_mode`, `clip`, `cropRois`, `crop_keyframes`, `dataset_split`, `provenance`, `quality`, `schema_version`, `segments`, `targetRatioWH`, `teacher_signals`, `video_id`, `video_path`, `video_url`。

顶层 schema 异常：未发现。

## 4. 核心字段详细结构

- `schema_version`：`seed_weak_training_label_v1`×987。
- `annotation_mode`：`temporal_spatial`×987。
- `dataset_split`：`train`×887，`val`×100；`targetRatioWH`：`(9.0, 16.0)`×987。
- `clip` 固定字段结构见附录；`free_axis`：`x`×987；clip 时长分布：n=987，min=5.238，P25=8.1165，median=10.444，mean=13.2383，P75=16.52，P95=32.133，max=32.134。
- `segments`：数组；长度分布为 n=987，min=1，P25=1，median=1，mean=1.43566，P75=2，P95=3，max=7；元素对象共 1417 个。
- `crop_keyframes`：数组；长度分布为 n=987，min=0，P25=0，median=9，mean=9.16008，P75=14，P95=29，max=33；空数组 363 条；元素对象共 9041 个。
- `teacher_signals`：对象，覆盖 987/987；固定包含 `timeline`、`candidate_segments`、`summary`、`trajectory`。
- `provenance`：对象，覆盖 987/987，记录数据来源、seed 模型、prompt/config 指纹、采样率与视频哈希。
- `quality`：对象，覆盖 987/987，包含接受状态、综合分数、置信度、覆盖率及重复一致性指标。

## 5. teacher_signals 详细分析

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `candidate_segments` | array×987 | 987/987（100.00%） | 是，987 次 |
| `summary` | string×987 | 987/987（100.00%） | 否 |
| `timeline` | array×987 | 987/987（100.00%） | 否 |
| `trajectory` | array×987 | 987/987（100.00%） | 否 |

- `timeline` 元素数：14228；长度分布：n=987，min=6，P25=9，median=12，mean=14.4154，P75=18，P95=33，max=33。
- `candidate_segments`：987/987 条为空数组，元素总数 0。
- `summary`：非空 987/987；字符长度分布：n=987，min=14，P25=28，median=34，mean=35.8926，P75=41，P95=56，max=78。报告不复制摘要正文。
- `trajectory` 元素数：14228；长度分布：n=987，min=6，P25=9，median=12，mean=14.4154，P75=18，P95=33，max=33。

`trajectory` 元素结构：

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `center` | array×14228 | 14228/14228（100.00%） | 否 |
| `confidence` | number×14228 | 14228/14228（100.00%） | 否 |
| `shot_id` | string×14228 | 14228/14228（100.00%） | 是，14228 次 |
| `subject` | string×14228 | 14228/14228（100.00%） | 是，14228 次 |
| `time_sec` | number×14228 | 14228/14228（100.00%） | 否 |
| `visible` | boolean×14228 | 14228/14228（100.00%） | 否 |

- `teacher_signals` 这一命名，加上 `provenance.seed_model` 与 prompt 指纹，直接证明文件显式保存了一组 teacher/seed 模型信号；但字段名本身不能证明是否经过人工复核。

## 6. segments 详细分析

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `end_frame` | number×1417 | 1417/1417（100.00%） | 否 |
| `end_sec` | number×1417 | 1417/1417（100.00%） | 否 |
| `max_seed_score` | number×1417 | 1417/1417（100.00%） | 否 |
| `mean_seed_score` | number×1417 | 1417/1417（100.00%） | 否 |
| `min_seed_score` | number×1417 | 1417/1417（100.00%） | 否 |
| `segment_id` | string×1417 | 1417/1417（100.00%） | 否 |
| `source_end_sec` | number×1417 | 1417/1417（100.00%） | 否 |
| `source_start_sec` | number×1417 | 1417/1417（100.00%） | 否 |
| `start_frame` | number×1417 | 1417/1417（100.00%） | 否 |
| `start_sec` | number×1417 | 1417/1417（100.00%） | 否 |

- 数量：共 1417 个；每条记录数组长度 n=987，min=1，P25=1，median=1，mean=1.43566，P75=2，P95=3，max=7；空数组 0 条。
- `start_sec`：n=1417，min=0，P25=0，median=3.4034，mean=5.34037，P75=7.77443，P95=18.6907，max=30.7307。
- `end_sec`：n=1417，min=0，P25=4.47113，median=7.4，mean=9.3585，P75=12.5959，P95=25.008，max=31.1。
- 区间时长 `end_sec-start_sec`：n=1417，min=0，P25=1.63333，median=2.86667，mean=4.01813，P75=5.04167，P95=11.2156，max=31.1。
- `start_frame`：n=1417，min=0，P25=0，median=96，mean=147.32，P75=216，P95=523，max=921；`end_frame`：n=1417，min=0，P25=123，median=210，mean=259.888，P75=338，P95=668.4，max=933。
- seed 分数：`min_seed_score` n=1417，min=0.0902208，P25=0.675975，median=0.72042，mean=0.694911，P75=0.740239，P95=0.767751，max=0.85235；`mean_seed_score` n=1417，min=0.0999076，P25=0.714733，median=0.74875，mean=0.736227，P75=0.789714，P95=0.834724，max=0.900239；`max_seed_score` n=1417，min=0.1，P25=0.75，median=0.75，mean=0.756942，P75=0.8，P95=0.85，max=0.93。
- 时间起止合法：1417/1417；帧起止合法：1417/1417。
- 局部区间位于 `0..(clip.end_sec-clip.start_sec)` 内：1416/1417。
- 超出 clip 局部终点的区间：1；最大超出 0.633333 ms，属于边界舍入量级。
- `source_start/end == clip.start_sec + local start/end`：1417/1417。
- 0 秒时长区间：2；0 帧跨度区间：2。这是值级边界案例，不是 schema 变体。
- 结构结论：它明确包含秒级起止、帧级起止、区间 ID、三种 seed 分数及 source 起止；不包含独立的 `frame`、`score` 或显式 `highlight_label` 字段。数组中的区间本身可被工程上解释为“被选中的片段”，但这不是官方 GT 身份证明。

匿名化结构示例：

```json
{"segments":[{"segment_id":"segment_N","start_sec":0.0,"end_sec":1.0,"start_frame":0,"end_frame":30,"min_seed_score":0.0,"mean_seed_score":0.0,"max_seed_score":0.0,"source_start_sec":0.0,"source_end_sec":1.0}]}
```

## 7. candidate_segments 详细分析

- 字段类型：`array`×987；记录覆盖率 100%。
- 空数组：987/987（100.00%）。
- 元素总数：0。由于没有任何元素，数据无法证明候选元素应有哪些键、时间单位或分数字段。
- 与 `segments` 数量完全相等的记录：0/987；区间列表精确相等：0/987；双方均非空：0/987。
- 因此，本文件中二者不是高度一致的两份区间：`segments` 非空而 `candidate_segments` 全空。不能计算有意义的时间 IoU/覆盖率，也不能把空候选字段当作 GT。

## 8. timeline 详细分析

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `confidence` | number×14228 | 14228/14228（100.00%） | 否 |
| `description` | string×14228 | 14228/14228（100.00%） | 是，14228 次 |
| `highlight_score` | number×14228 | 14228/14228（100.00%） | 否 |
| `phase` | string×14228 | 14228/14228（100.00%） | 是，14228 次 |
| `time_sec` | number×14228 | 14228/14228（100.00%） | 否 |

- 时间点总数：14228；每条记录长度 n=987，min=6，P25=9，median=12，mean=14.4154，P75=18，P95=33，max=33。
- `time_sec`：n=14228，min=0，P25=3，median=7，mean=8.64669，P75=12，P95=24，max=32.1321。
- 相邻正时间差：n=13241，min=0.252567，P25=1，median=1，mean=0.98528，P75=1，P95=1，max=1.24157；其中恰为 1 秒 12260/13241（92.59%）。
- 时间非严格递增记录：0/987；非末尾位置的非 1 秒间隔：0。末尾可以是非整数视频终点，因此它是约 1 Hz 的秒级采样点序列，而不是逐帧标签或固定长度时间块。
- `highlight_score`：n=14228，min=0，P25=0.5，median=0.65，mean=0.579606，P75=0.75，P95=0.85，max=0.93。
- `confidence`：n=14228，min=0，P25=0.8，median=0.85，mean=0.802474，P75=0.9，P95=0.95，max=0.98。
- 位于 `segments` 内的 timeline 点：6160/14228（43.29%）。
- 区间内 `highlight_score`：n=6160，min=0.1，P25=0.7，median=0.75，mean=0.723436，P75=0.8，P95=0.85，max=0.93；区间外：n=8068，min=0，P25=0.3，median=0.6，mean=0.46979，P75=0.65，P95=0.75，max=0.85。
- 在仅测试 0.05 步长阈值时，最佳点级阈值为 `score >= 0.7`：precision=0.758、recall=0.820、F1=0.787（预测正点 6664，区间内点 6160）。这只是对现有 `segments` 的拟合，不是官方规则。
- schema 预留 `phase` 和 `description` 语义字段，但本文件二者分别为空 14228/14228、14228/14228，没有可用逐点文本语义。

## 9. provenance 分析

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `annotator_config_sha256` | string×987 | 987/987（100.00%） | 否 |
| `license` | string×987 | 987/987（100.00%） | 是，987 次 |
| `n_seed_repeats` | number×987 | 987/987（100.00%） | 否 |
| `prompt_fingerprint` | string×987 | 987/987（100.00%） | 否 |
| `prompt_version` | string×987 | 987/987（100.00%） | 否 |
| `seed_model` | string×987 | 987/987（100.00%） | 否 |
| `source` | string×987 | 987/987（100.00%） | 否 |
| `source_group` | string×987 | 987/987（100.00%） | 否 |
| `spatial_fps` | number×987 | 987/987（100.00%） | 否 |
| `temporal_fps` | number×987 | 987/987（100.00%） | 否 |
| `video_sha256` | string×987 | 987/987（100.00%） | 否 |

- `source`：`QVHighlights`×987。
- `seed_model`：`api_doubao_doubao-seed-2-1-pro-260628`×987。
- `prompt_version`：`seed_temporal_compact_v2_20260812`×987。
- `prompt_fingerprint`：1 个唯一值，覆盖 987/987。
- `annotator_config_sha256`：1 个唯一值。
- `source_group`：987 个唯一值；覆盖 987/987。不列出可识别的具体值。
- `n_seed_repeats`：n=987，min=1，P25=1，median=1，mean=1，P75=1，P95=1，max=1。
- `temporal_fps`：n=987，min=1，P25=1，median=1，mean=1，P75=1，P95=1，max=1；`spatial_fps`：n=987，min=1，P25=1，median=1，mean=1，P75=1，P95=1，max=1。
- `video_sha256` 唯一值数：987；不在报告中列出具体视频哈希。
- `license`：空字符串 987/987；当前文件本身没有给出许可文本。
- 能证明的内容：记录声明了来源数据集、seed 模型、prompt/config 版本或指纹和采样参数，且 schema 名称为 seed weak training label。
- 不能证明的内容：这些字段不包含人工审核者身份、人工复核状态、官方 GT 标记或标注流程文档，因而不能单独证明人工/模型/混合生成的最终归属。
- 人工标注判断：仅依据本文件，没有任何字段可被确认或高概率归类为人工标注；`clip.qvh_window` 等可能继承自源数据集的字段也缺少逐字段来源说明。

## 10. crop_keyframes 分析

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `axis_index` | number×9041 | 9041/9041（100.00%） | 否 |
| `confidence` | number×9041 | 9041/9041（100.00%） | 否 |
| `frame` | number×9041 | 9041/9041（100.00%） | 否 |
| `free_axis` | string×9041 | 9041/9041（100.00%） | 否 |
| `free_axis_position_px` | number×9041 | 9041/9041（100.00%） | 否 |
| `raw_center` | array×9041 | 9041/9041（100.00%） | 否 |
| `subject` | string×9041 | 9041/9041（100.00%） | 是，9041 次 |
| `time_sec` | number×9041 | 9041/9041（100.00%） | 否 |
| `visible` | boolean×9041 | 9041/9041（100.00%） | 否 |

- 非空记录：624/987；元素总数 9041；长度 n=987，min=0，P25=0，median=9，mean=9.16008，P75=14，P95=29，max=33。
- 时间点落入 `segments`：3783/9041（41.84%）。这说明 keyframe 通常覆盖视频采样时间轴，而非仅高光区间。
- 与 `trajectory` 长度相同：624/987 条；对齐 pair 9041 个。
- 对齐 pair 中：时间相同 9041/9041，`raw_center == center` 9041/9041，`visible` 相同 9041/9041。
- `cropRois` 元素 98242 个，满足 `[frame,[x,y,w,h]]` 结构 98242/98242；ROI 帧位于 segment 帧范围 98242/98242。
- 在可比较记录中，`cropRois` 帧集合恰好覆盖全部 segment 整数帧范围：624/624。
- 空 `cropRois` 且 `quality.spatial_source=dropped_center_default`：363/363。
- 工程含义：`crop_keyframes`/`trajectory` 描述稀疏主体中心与置信度，`cropRois` 给出选中区间内逐帧裁剪框；它们适合后续空间构图任务，但不应反向当作 Highlight Retrieval 的独立时序 GT。

## 11. Highlight Retrieval 可用标签分析

### 候选方案 A：使用 `segments`

- 证据：全量非空；具有明确秒级和帧级边界；含 seed 分数；与逐帧 `cropRois` 高度关联。
- 优点：可直接构造 temporal reference segments，无需自定义阈值。
- 风险：`schema_version`、`seed_model`、`prompt_fingerprint`、`min/mean/max_seed_score` 显示其很可能是 seed/teacher 弱标签；没有官方 GT 标记或人工审核链路。
- 当前可信度：三种方案中最高，适合作为“训练弱 reference / 内部开发 reference”，暂不能称赛事官方 Ground Truth。

### 候选方案 B：使用 `teacher_signals.candidate_segments`

- 证据：字段在所有记录中存在。
- 优点：名称在未来数据版本中可能承载 teacher 原始候选。
- 风险：本文件 100% 为空，元素 schema 也无法确认。
- 当前可信度：不可用。

### 候选方案 C：从 `timeline.highlight_score` 阈值化构造

- 证据：timeline 全量存在，约 1 Hz，含 `highlight_score` 与 `confidence`；粗网格最佳拟合阈值为 0.7，F1=0.787。
- 优点：保留连续软分数，可调整 Recall/Precision，也可用于点级蒸馏或 ranking。
- 风险：阈值、点到区间的边界扩展、短间隔合并和尾点处理均无官方定义；秒级采样也弱于已有帧级边界。
- 当前可信度：适合作为辅助监督与敏感性分析，不宜替代 `segments` 作为首选 reference。

结论：若下一步必须选择一个内部 Highlight Retrieval reference，`segments` 最可信；命名应明确为 seed/teacher-derived weak reference。最终 GT 身份仍需官方说明或人工抽样核验。

## 12. 已确认事实

- 文件包含 987 条合法记录，0 空行，0 条解析/顶层类型异常，顶层 schema 完全一致。
- 所有记录都包含 `segments`、`crop_keyframes`、`teacher_signals`、`provenance` 和 `quality`。
- `segments` 共 1417 个且每条记录至少一个；包含时间、帧、segment ID 和 seed 分数，不包含显式 `highlight_label`。
- `candidate_segments` 在 987/987 条中为空。
- `timeline` 是按 `time_sec` 排列的约 1 Hz 采样序列，包含 `highlight_score` 和 `confidence`。
- `timeline` 与 `trajectory` 长度相同 987/987 条，时间序列完全相同 987/987 条。
- provenance 的 `source` 分布为：`QVHighlights`×987。
- provenance 的 `seed_model` 分布为：`api_doubao_doubao-seed-2-1-pro-260628`×987。

## 13. 高概率判断

- `teacher_signals.timeline/summary/trajectory` 明显是 teacher/seed 模型输出或其规范化结果。依据是字段命名、seed 模型、prompt 版本/指纹和 provenance 同时存在。
- `segments` 高概率由 timeline/seed 分数再经过阈值、连通区间和边界处理生成，而非独立人工 GT；其字段直接命名为 `*_seed_score`。
- `quality` 高概率是自动质量门控/聚合产物，尤其是 coverage、score spread、threshold margin 和 repeat agreement。
- `crop_keyframes` 高概率由 teacher trajectory 规范化而来；两者时间、中心和可见性可直接量化对应。

## 14. 合理推测

- `cropRois` 可能由稀疏 `crop_keyframes`/trajectory 插值后，仅在 `segments` 选中帧范围内展开，用于后续竖屏裁剪或构图。
- `source=QVHighlights` 可能表示本训练文件以 QVHighlights 视频/窗口为底座，再由 seed 模型生成赛事任务所需的 generic highlight 与空间弱标签。
- `clip.qvh_window`、`used_window` 与 planned/achieved ratio 可能记录从源窗口到当前训练 clip 的选段规划过程，但缺少生成器说明，不能作为事实。

## 15. 当前无法确认事项

- `segments` 是否被赛事官方定义为最终 reference annotation / Ground Truth。需要官方 README、字段说明或评测代码确认。
- 任何字段是否经过人工审核、修订或验收。需要人工标注流程、审核日志或明确的 provenance 字段。
- `clip.qvh_window`、`used_window`、比例字段及源数据元信息分别属于人工、原始数据集还是自动生成。需要字段级 lineage/生成流程文档。
- `candidate_segments` 的预期元素 schema 与为何全空。需要数据生成代码或 schema 文档。
- 从 timeline 到 segments 的精确阈值、插值、合并与边界规则。需要生成器实现/config；点级拟合不能证明生成规则。
- `source_start_sec/source_end_sec` 的官方命名意图。数据中它们严格等于 `clip.start_sec + local start/end`，但仍需字段文档确认其坐标系定义。

## 16. 推荐下一步

1. 查阅赛事官方 README、训练数据说明、schema 定义与评测脚本，确认 reference 字段和时间边界约定。
2. 从不同 `annotation_mode`、quality 状态和 segment 数量桶中抽样少量训练视频，人工核验 `segments` 是否确实覆盖语义高光。
3. 定位生成 `seed_weak_training_label_v1` 的代码/config，用 `prompt_fingerprint` 与 `annotator_config_sha256` 对应具体版本。
4. 将 `segments` 暂时命名为 `weak_reference_segments`，同时保留 timeline 软分数；在官方确认前避免在指标或文档中写作 Ground Truth。
5. 对 timeline 阈值做敏感性分析时按 source/video 分组，防止同源窗口泄漏；本轮不运行正式 benchmark。

## 17. 对 Highlight Retrieval Pipeline 的工程建议

- Reader 层显式支持 `reference_policy=segments|timeline_threshold|none`，默认值在官方语义确认前不要暗示 official GT。
- 训练可同时使用 `segments` 的区间监督与 timeline 的软分数/置信度，但在 loss 中区分 weak label 与人工 label。
- 评估时记录 reference 来源、schema_version、seed_model、prompt_fingerprint 和阈值配置，保证结果可复现。
- `candidate_segments` 为空时应明确报不可用，不能静默回退并声称同一指标定义。
- `crop_keyframes`、trajectory 与 `cropRois` 留给空间裁剪阶段；不要让空间字段改变本轮 Highlight Retrieval 的 reference 语义。
- QVHighlights 是 query-conditioned highlight detection 数据；本赛事目标更接近 generic highlight selection。可将 QVHighlights 作为后续辅助训练/泛化来源，但不能把 QVHighlights GT 直接等同赛事 GT。本轮不下载该数据集。

---

### 附录 A：quality 概览

| 字段 | 类型 | 覆盖率 | 是否观察到 null/空值 |
|---|---|---:|---|
| `annotation_mode` | string×987 | 987/987（100.00%） | 否 |
| `n_repeats` | number×987 | 987/987（100.00%） | 否 |
| `reasons` | array×987 | 987/987（100.00%） | 是，984 次 |
| `repeat_agreement` | number×987 | 987/987（100.00%） | 否 |
| `score` | number×987 | 987/987（100.00%） | 否 |
| `score_spread` | number×987 | 987/987（100.00%） | 否 |
| `selected_ratio` | number×987 | 987/987（100.00%） | 否 |
| `spatial_confidence` | number×987 | 987/987（100.00%） | 否 |
| `spatial_source` | string×987 | 987/987（100.00%） | 否 |
| `status` | string×987 | 987/987（100.00%） | 否 |
| `temporal_confidence` | number×987 | 987/987（100.00%） | 否 |
| `threshold_margin` | number×987 | 987/987（100.00%） | 否 |
| `timeline_coverage` | number×987 | 987/987（100.00%） | 否 |
| `trajectory_coverage` | number×987 | 987/987（100.00%） | 否 |

- `status`：`accepted`×987。
- `annotation_mode`：`temporal_spatial`×987。
- `spatial_source`：`seed`×624，`dropped_center_default`×363。
- `score`：n=987，min=0.800139，P25=0.841449，median=0.854163，mean=0.85872，P75=0.870062，P95=0.915533，max=0.948。
- `selected_ratio`：n=987，min=0.0201794，P25=0.250304，median=0.408163，mean=0.455264，P75=0.627903，P95=0.93589，max=0.988323。
- `temporal_confidence`：n=987，min=0.66875，P25=0.825，median=0.854348，mean=0.859515，P75=0.897788，P95=0.941667，max=0.95。
- `spatial_confidence`：n=987，min=0.294444，P25=0.843095，median=0.89，mean=0.875008，P75=0.918466，P95=0.95，max=1。
- `timeline_coverage`：n=987，min=1，P25=1，median=1，mean=1，P75=1，P95=1，max=1。
- `trajectory_coverage`：n=987，min=0.969697，P25=1，median=1，mean=0.999846，P75=1，P95=1，max=1。
- `repeat_agreement`：n=987，min=0.75，P25=0.75，median=0.75，mean=0.75，P75=0.75，P95=0.75，max=0.75。
- `score_spread`：n=987，min=0.1，P25=0.75，median=0.8，mean=0.740036，P75=0.8，P95=0.85，max=0.93。
- `threshold_margin`：n=987，min=0.00117729，P25=0.207955，median=0.362804，mean=0.409143，P75=0.534289，P95=1，max=1。
- `reasons` 空数组：984/987；非空元素计数：3。

### 附录 B：schema 变体与异常

- `$`：对象 987 个，字段集合 1 种；一致。
- `$.clip`：对象 987 个，字段集合 1 种；一致。
- `$.segments[]`：对象 1417 个，字段集合 1 种；一致。
- `$.crop_keyframes[]`：对象 9041 个，字段集合 1 种；一致。
- `$.teacher_signals`：对象 987 个，字段集合 1 种；一致。
- `$.teacher_signals.timeline[]`：对象 14228 个，字段集合 1 种；一致。
- `$.teacher_signals.trajectory[]`：对象 14228 个，字段集合 1 种；一致。
- `$.provenance`：对象 987 个，字段集合 1 种；一致。
- `$.quality`：对象 987 个，字段集合 1 种；一致。

未发现 JSON 解析异常或非对象顶层记录。
