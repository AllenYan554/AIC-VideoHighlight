# PHD2 Generic Pretrain Target Adapter V1

```text
PROTOCOL_ID              : PHD2_GENERIC_PRETRAIN_TARGET_V1
STATUS                   : FROZEN (user-approved)
FREEZE_TIMESTAMP         : 2026-09-24 (Asia/Shanghai)
DATASET                  : PHD2 / PHD-GIFs Personalized Highlight Detection
ROLE                     : WEAK-SUPERVISION PRETRAINING DATA ONLY
CALIBRATION_CLAIM        : NONE (ranking only; sigmoid output is NOT a calibrated probability)
RESULT_DRIVEN            : NO
SUPERSEDES               : none (first PHD2 target adapter)
APPLIES_TO               : Stage 7 VHiCraFTNet PHD2 pretraining
TRAIN_ANNOTATION_SHA256  : f89f7993b28abb29549bbdb81330e4a60a2750d65e2f3cc26d78ab4ab7be6f3d
TEST_ANNOTATION_SHA256   : 7589cc7920ae46ff569a76dfd04f8c1f510e0e6fe51e04b9ba2448f8c4cf5071 (audit only; never used)
```

本协议在任何 optimizer step 之前冻结。冻结后不得因结果表现更换标签策略，不得把未标注区域当作
hard negative。如需 amendment：新版本 + 显式用户批准 + 不覆盖本文件。

## 1. 数据来源与身份

```text
source              E:\ResearchData\datasets\phd2\metadata\training.csv   (READ-ONLY source)
                    E:\ResearchData\datasets\phd2\metadata\testing.csv    (audit only)
columns             youtubeId, start, duration, user_id, video_duration, is_last
training.csv        201,527 rows | 110,611 unique youtubeId | 12,972 users
testing.csv          20,488 rows |  10,222 unique youtubeId |    850 users
time unit           seconds
selection interval  [start, start + duration)   (half-open, seconds)
benchmark video set frozen manifest phd2_raw_1500_202609220403.json
                    SHA256 ef0a07a3e437840a5a61b4b48e4524651015f70d9f3a91a3174eac58a3aa9c63
                    manifest ids in TRAIN = 1500 ; manifest ids in TEST = 0  (TRAIN_ONLY verified)
```

官方语义（PHD² README）：每个 selection 是某用户从视频中截取高光 GIF 的 `[start, start+duration)` 片段；
用户偏好 **不** 等价于 MTurk annotator vote。

## 2. TRAIN_ONLY 与 TEST isolation

```text
TRAIN_ONLY = YES
```

- 只使用 `training.csv` 构建 target；`testing.csv` 只用于 split-integrity 审计。
- TEST 不得参与训练、阈值、统计、target construction、pair sampling。
- cloud producer 只接收 TRAIN annotation；TEST 文件不上传。
- 已审计：frozen 1500 manifest 的 id 全部在 TRAIN，出现在 TEST 的数量 = 0。

## 3. Positive / Unlabeled 定义（2 fps timeline）

对每个视频的 2 fps 采样时间线（timestep 携带真实 `timestamp` 与 `source_frame_id`）：

```text
P (Positive Evidence) : timestep.timestamp ∈ 至少一个 TRAIN selection interval [start, start+duration)
                        -> POSITIVE = TRUE

U (Unlabeled)         : 未被任何 TRAIN selection interval 覆盖的 timestep
                        -> UNLABELED = TRUE ; NOT NEGATIVE
```

- interval inclusion：half-open `[start, start + duration)`，`end` 不计入。
- boundary handling：恰好落在边界时按 half-open 归入后一区间。
- **禁止** unselected → `Y = 0`；**禁止**把 PHD² 未截取区域全部当作 hard negative。
- 若一个视频没有任何 positive timestep：该视频不产生任何 pair（skip），不得伪造 negative。

## 4. Multi-user aggregation

```text
POSITIVE_AGGREGATION = UNION of all TRAIN users' selection intervals
selection_support_count = 覆盖该 timestep 的不同 user 数（metadata / audit only）
```

- 至少一个用户选择过 → positive evidence。
- **本轮 support_count 不参与 loss**（不做 0→0 / 1/3→0.33 / 3/3→1 的 soft Y 映射）。
- support_count 只写入 materialized 数据作为审计字段。

## 5. Boundary guard

```text
BOUNDARY_GUARD = 0.5 s   ( = 2 fps sampling interval )
```

- 对每个 selection interval 的 `start` / `end` 外侧 ±0.5 s 邻域内、且属于 U 的 timestep：
  - 不是 positive；
  - **不得**用作 pairwise unlabeled comparison sample；
  - 标记 `boundary_guard = TRUE`（保留在数据中用于审计，不参与 pair sampling）。
- 目的：降低 interval 边界不确定性对 pairwise 监督的污染。

## 6. Pretraining loss: Positive-Unlabeled Pairwise Ranking

本轮不使用普通 hard-negative BCE（只有 positive evidence，unknown ≠ negative）。

```text
z_t = Y-head pre-sigmoid logit for timestep t
t_pos ∈ P ,  t_u ∈ U (legal, non-boundary-guard, same video, valid adjacency)

L_pair = softplus(-(z_pos - z_u)) = log(1 + exp(-(z_pos - z_u)))

L_total = mean over sampled pairs
```

- 目标：明确被选中的 highlight timestep 应比 unlabeled timestep 获得更高 Y-head score。
- 该 loss **不**宣称 unlabeled 是 negative。
- Q Head：本轮 OFF，不参与训练。

## 7. Deterministic pair sampling

```text
RATIO          1 : 1   (每个 positive anchor 采样一个合法 unlabeled comparison)
SEED           deterministic, fixed per run (recorded in run manifest)
ORDER          per-video deterministic ordering (sorted video_id, then ascending timestep)
```

合法 pair 约束（全部满足）：

```text
same video only           (no cross-video pair)
t_u ∉ boundary_guard
t_u has valid timestamp and source_frame_id
t_u and t_pos in a valid adjacency region (no invalid adjacency / padding)
no reuse of the same unlabeled timestep beyond the 1:1 rule within a video
```

- 若当前视频没有合法 U：跳过该 pair，不伪造 negative。
- 本轮不使用 support_count 加权：所有合法 positive anchor 等权。

## 8. Calibration 声明

PHD² pretraining 只学习**相对排序**能力。`sigmoid(z)` 不得被声明为 calibrated highlight
probability；概率校准留给后续强监督（YouTube Highlights）fine-tuning。

## 9. Failure / exclusion 规则

- 单视频异常：record → continue → 最后 failed-only retry（按现有恢复策略）。
- 无法恢复的样本：记录 `video_id / reason / raw path / stage / provenance`，从 downstream eligible
  cohort 排除。
- **严禁伪造 artifact**：不得空 candidate、不得默认 score、不得用其它视频结果替换。
- Qwen retrieval 缺失样本（`QWEN_RETRIEVAL_UNAVAILABLE`）不进入本协议。

## 10. Materialized 输出契约

每个视频保存（visual feature 为 Visual256；Delta256 训练时现算，不长期存储）：

```text
visual_features            float16 [T, 256]
timestamps                 float64 [T]
source_frame_ids           int64   [T]
adjacency_mask             bool    [T]
Native16                   float16 [T, 16]
positive_mask              bool    [T]     (P)
unlabeled_mask             bool    [T]     (U)
boundary_guard_mask        bool    [T]
selection_support_count    int32   [T]
candidate_provenance       (Qwen retrieval + RT-DETR artifact identity)
pair_sampling_provenance   (seed, order, per-video pair ledger)
protocol_version           PHD2_GENERIC_PRETRAIN_TARGET_V1
```

schema 必须 finite / no NaN，缺失值为 0 且由 `loss_mask` 语义控制。

## 11. 禁止事项

```text
FORBIDDEN: unselected -> Y=0
FORBIDDEN: hard-negative BCE on PHD2
FORBIDDEN: soft Y from support_count
FORBIDDEN: TEST usage of any kind
FORBIDDEN: cross-video / boundary-guard / invalid-adjacency pairs
FORBIDDEN: fabricating candidates or labels for failed videos
FORBIDDEN: claiming calibration or frame-exact GT
```

## 12. TCN / architecture identity for this pretrain

```text
FTNET_TCN_STATUS = REFERENCE_B5_FOR_THIS_PRETRAIN
channels = 128 ; kernel = 5 ; dilation = [1,2,4,8,16] ; blocks = 5 ; dropout = 0.1
non-causal ; depthwise Conv1D -> pointwise Conv1D -> LayerNorm -> GELU -> Dropout -> Residual
theoretical receptive field = 1 + (5-1)*(1+2+4+8+16) = 125 timesteps ~ 62.5 s at 2 fps
NOTE: reference design only; NOT claimed optimal or empirically best.
```

```text
FUTURE_REQUIRED_ABLATION (NOT RUN in this round):
  B2 [1,2] ; B3 [1,2,4] ; B4 [1,2,4,8] ; B5 [1,2,4,8,16] ; B6 [1,2,4,8,16,32]
  B5-nodilation [1,1,1,1,1] ; B5-capped [1,2,4,4,4]
  purpose: separate depth effect vs receptive-field/dilation effect
```

## 13. Version / preregistration

```text
PROTOCOL_VERSION                 = PHD2_GENERIC_PRETRAIN_TARGET_V1
PREREGISTERED                    = YES
PREREGISTRATION_TIMESTAMP        = 2026-09-24 (Asia/Shanghai)
FROZEN_BEFORE_ANY_OPTIMIZER_STEP = YES
PHD2_TARGET_PROTOCOL_FROZEN      = YES
```
