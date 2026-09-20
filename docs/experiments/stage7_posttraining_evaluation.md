# Stage 7.1 FTNet Post-Training Evaluation（best.pt @ epoch 7）

```text
STATUS              : POST_TRAINING_EVALUATION_COMPLETE
DATE                : 2026-09-20
GIT_HEAD            : 7c1b02627fb5cc60f71c37d9413ef199ccf5f5ff
PRIMARY CHECKPOINT  : best.pt  (epoch 7, step 196, sha256 0204b8c3…)
CONTROL CHECKPOINT  : last.pt  (epoch 40, step 1120, sha256 3997cb70…)
TRAINING            : NO
QWEN / RT-DETR      : NO
MATERIALIZATION     : NO (frozen dataset read-only)
OFFICIAL TEST/TVSUM : NOT ACCESSED
FORMAL REPORT       : 实验记录/Stage7_VHiCraFTNet研发/07_Stage7.1_FTNet_PostTraining_Evaluation_20260920/experiment_report.md
```

## 结论（预注册规则，development metrics）

**D — 负结果**：tau_f1（CALIBRATION F1 最大）在 VALIDATION 上 pruning=99.25%、recall=32.08%、false deletion=67.92%。

补充观察（不改变预注册判定）：

- best.pt 具有明显的阈值无关排序信号：VALIDATION AP=0.38283（正帧基准 prevalence=1.20%，FS0 F1=2.38%）。
- 存在“近安全点” tau=0.0226：VALIDATION pruning=61.01%、recall=94.34%（差 0.66pp 未达到冻结的 95% 门槛），
  且 whole positive-run deletion = 7.69%（1/13 run）。
- best vs last：VALIDATION AP 0.38283 vs 0.17967（+0.20316）；CALIBRATION 反向（0.13987 vs 0.20367，-0.06380），
  过拟合对 ranking 的影响为 split-dependent，不外推为普遍泛化下降。

## 核心总表（VALIDATION，binary reference = soft target >= 0.5）

| Method | Precision | Recall | F1 | PR-AUC | Pruning | False Deletion |
|---|---:|---:|---:|---:|---:|---:|
| FS0 (keep all) | 1.20% | 100.00% | 2.38% | N/A | 0.00% | 0.00% |
| FTNet @ tau_f1 (0.3897) | 51.52% | 32.08% | 39.53% | 0.38283 | 99.25% | 67.92% |
| FTNet @ tau_safe (0.0226) | 2.91% | 94.34% | 5.65% | — | 61.01% | 5.66% |

- Universe = loss_mask==1（有 MTurk clip 覆盖的 candidate 帧）：VALIDATION 4401 帧 / 53 正帧；CALIBRATION 2181 / 30。
- 阈值只在 CALIBRATION 选择；TRAIN 未参与任何阈值或指标。

## Positive-Run 派生代理（非人类 event GT）

- **EVENT_METRICS_AVAILABLE = NO**：原始标注是 ~2s MTurk clip 窗口 + soft vote，不存在人类 event identity。
- VALIDATION（13 runs，无 <=2s short run）：tau_f1 survival=38.46%、whole deletion=61.54%；tau_safe survival=92.31%、whole deletion=7.69%。
- CALIBRATION（8 runs，2 个 short）：tau_safe 时 short-run survival=100%、whole deletion=0%、retention=1.0。
- 短高光结论：VALIDATION 没有 short run 样本，无法给出 VALIDATION short-run 结论；CALIBRATION 证据有限且必须标注为派生代理。

## Native16 missingness-vs-Y（统计审计，universe 内）

- 候选上下文 5 维在 universe 内 100% observed（无差异）。
- 4 个主体几何/置信字段 |Δ|≈0.10–0.12 → `POTENTIAL_SHORTCUT_RISK`（subject_selection_confidence、bbox_area_ratio、
  subject_scale_dynamics、subject_frame_offset）。该标记仅表示 missingness 与 Y 相关，不证明模型使用 shortcut。

## 产物

- 正式报告与图：`实验记录/Stage7_VHiCraFTNet研发/07_Stage7.1_FTNet_PostTraining_Evaluation_20260920/`
  （config/figures/results/supplementary + experiment_report.md）
- 协议：`config/evaluation_protocol.json`（指标定义、阈值规则、tie-break、universe、verdict 规则，先于指标冻结）
- 逐帧分数：`supplementary/frame_scores_{validation,calibration}.jsonl`（两次独立运行 byte-identical）
- 代码：`src/aic_video_highlight/ftnet/evaluation.py`、`scripts/experiments/stage7/run_ftnet_evaluation.py`、
  tests `tests/test_ftnet_evaluation.py`（28 项）
