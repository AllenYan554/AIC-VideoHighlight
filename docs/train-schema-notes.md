# train.jsonl Schema 检查记录

## 确定可以看出的事实

- 只读来源：`C:\Users\lenovo\Downloads\train.jsonl`
- 文件大小：9277172 bytes
- 总行数：987
- 空行数：0
- 合法顶层 JSON 对象数：987
- 每个非空行是否均为独立 JSON 对象：是
- 顶层字段集合是否完全一致：是
- 不同顶层字段集合数量：1

### 重点字段在顶层的出现次数

- `timeline`：0
- `highlight_score`：0
- `confidence`：0
- `candidate_segments`：0
- `crop_keyframes`：987
- `trajectory`：0
- `summary`：0
- `dataset_split`：987
- `provenance`：987
- `seed_model`：0
- `prompt_fingerprint`：0

### 重点字段的同名嵌套路径

- `timeline`：`$.teacher_signals.timeline`（987 条观测）, `$.teacher_signals.timeline[]`（987 条观测）
- `highlight_score`：`$.teacher_signals.timeline[].highlight_score`（987 条观测）
- `confidence`：`$.crop_keyframes[].confidence`（624 条观测）, `$.teacher_signals.timeline[].confidence`（987 条观测）, `$.teacher_signals.trajectory[].confidence`（987 条观测）
- `candidate_segments`：`$.teacher_signals.candidate_segments`（987 条观测）
- `crop_keyframes`：`$.crop_keyframes`（987 条观测）, `$.crop_keyframes[]`（624 条观测）
- `trajectory`：`$.teacher_signals.trajectory`（987 条观测）, `$.teacher_signals.trajectory[]`（987 条观测）
- `summary`：`$.teacher_signals.summary`（987 条观测）
- `dataset_split`：`$.dataset_split`（987 条观测）
- `provenance`：`$.provenance`（987 条观测）
- `seed_model`：`$.provenance.seed_model`（987 条观测）
- `prompt_fingerprint`：`$.provenance.prompt_fingerprint`（987 条观测）

### 顶层字段集合（按出现次数，最多 20 种）

- 987 条：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`

### 固定抽样（最多前 5 条合法记录，仅列字段名）

- 第 1 行：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`
- 第 2 行：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`
- 第 3 行：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`
- 第 4 行：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`
- 第 5 行：`annotation_mode, clip, cropRois, crop_keyframes, dataset_split, provenance, quality, schema_version, segments, targetRatioWH, teacher_signals, video_id, video_path, video_url`

### 字段层级与观测类型（深度最多 4 层，最多 250 项）

- `$`：object×987
- `$.annotation_mode`：string×987
- `$.clip`：object×987
- `$.clip.achieved_hl_ratio`：number×987
- `$.clip.end_sec`：number×987
- `$.clip.free_axis`：string×987
- `$.clip.free_axis_travel`：number×987
- `$.clip.planned_hl_ratio`：number×987
- `$.clip.qvh_window`：array×987
- `$.clip.qvh_window[]`：number×987
- `$.clip.source_vid`：string×987
- `$.clip.start_sec`：number×987
- `$.clip.used_window`：array×987
- `$.clip.used_window[]`：number×987
- `$.cropRois`：array×987
- `$.cropRois[]`：array×624
- `$.cropRois[][]`：number×624
- `$.crop_keyframes`：array×987
- `$.crop_keyframes[]`：object×624
- `$.crop_keyframes[].axis_index`：number×624
- `$.crop_keyframes[].confidence`：number×624
- `$.crop_keyframes[].frame`：number×624
- `$.crop_keyframes[].free_axis`：string×624
- `$.crop_keyframes[].free_axis_position_px`：number×624
- `$.crop_keyframes[].raw_center`：array×624
- `$.crop_keyframes[].raw_center[]`：number×624
- `$.crop_keyframes[].subject`：string×624
- `$.crop_keyframes[].time_sec`：number×624
- `$.crop_keyframes[].visible`：boolean×624
- `$.dataset_split`：string×987
- `$.provenance`：object×987
- `$.provenance.annotator_config_sha256`：string×987
- `$.provenance.license`：string×987
- `$.provenance.n_seed_repeats`：number×987
- `$.provenance.prompt_fingerprint`：string×987
- `$.provenance.prompt_version`：string×987
- `$.provenance.seed_model`：string×987
- `$.provenance.source`：string×987
- `$.provenance.source_group`：string×987
- `$.provenance.spatial_fps`：number×987
- `$.provenance.temporal_fps`：number×987
- `$.provenance.video_sha256`：string×987
- `$.quality`：object×987
- `$.quality.annotation_mode`：string×987
- `$.quality.n_repeats`：number×987
- `$.quality.reasons`：array×987
- `$.quality.reasons[]`：string×3
- `$.quality.repeat_agreement`：number×987
- `$.quality.score`：number×987
- `$.quality.score_spread`：number×987
- `$.quality.selected_ratio`：number×987
- `$.quality.spatial_confidence`：number×987
- `$.quality.spatial_source`：string×987
- `$.quality.status`：string×987
- `$.quality.temporal_confidence`：number×987
- `$.quality.threshold_margin`：number×987
- `$.quality.timeline_coverage`：number×987
- `$.quality.trajectory_coverage`：number×987
- `$.schema_version`：string×987
- `$.segments`：array×987
- `$.segments[]`：object×987
- `$.segments[].end_frame`：number×987
- `$.segments[].end_sec`：number×987
- `$.segments[].max_seed_score`：number×987
- `$.segments[].mean_seed_score`：number×987
- `$.segments[].min_seed_score`：number×987
- `$.segments[].segment_id`：string×987
- `$.segments[].source_end_sec`：number×987
- `$.segments[].source_start_sec`：number×987
- `$.segments[].start_frame`：number×987
- `$.segments[].start_sec`：number×987
- `$.targetRatioWH`：array×987
- `$.targetRatioWH[]`：number×987
- `$.teacher_signals`：object×987
- `$.teacher_signals.candidate_segments`：array×987
- `$.teacher_signals.summary`：string×987
- `$.teacher_signals.timeline`：array×987
- `$.teacher_signals.timeline[]`：object×987
- `$.teacher_signals.timeline[].confidence`：number×987
- `$.teacher_signals.timeline[].description`：string×987
- `$.teacher_signals.timeline[].highlight_score`：number×987
- `$.teacher_signals.timeline[].phase`：string×987
- `$.teacher_signals.timeline[].time_sec`：number×987
- `$.teacher_signals.trajectory`：array×987
- `$.teacher_signals.trajectory[]`：object×987
- `$.teacher_signals.trajectory[].center`：array×987
- `$.teacher_signals.trajectory[].confidence`：number×987
- `$.teacher_signals.trajectory[].shot_id`：string×987
- `$.teacher_signals.trajectory[].subject`：string×987
- `$.teacher_signals.trajectory[].time_sec`：number×987
- `$.teacher_signals.trajectory[].visible`：boolean×987
- `$.video_id`：string×987
- `$.video_path`：string×987
- `$.video_url`：string×987

### 解析异常（最多记录 20 条，仅列行号与错误类型）

- 未发现解析异常。

## 暂时无法确认的标注语义

- 字段名本身不能证明某字段是赛事最终人工 Ground Truth。
- 即使存在 `candidate_segments`，也不能仅凭名称把它作为唯一 GT。
- `seed_model`、`prompt_fingerprint`、`provenance` 等字段若存在，只能确认记录包含生成/来源元数据；不能据此确定标签由人工、模型或混合流程产生。
- 在获得赛事数据说明、字段定义或可靠样例解释前，`annotation_reader.py` 不会默认选择任何 GT 字段。
- 本报告不包含训练标注原文、摘要内容、轨迹值或候选区间值。
