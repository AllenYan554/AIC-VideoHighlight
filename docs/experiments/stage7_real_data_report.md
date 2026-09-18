# Stage 7.1 YouTube Highlights 真实数据物化与 FTNet 40-epoch Baseline

```text
STATUS            : COMPLETE
DATE              : 2026-09-18 / 2026-09-19
GIT_HEAD          : caae18ca84b9a50b989c198756920bee4f22a08d
BRANCH            : feat/stage7-ftnet
REAL_UPSTREAM_PROVIDER = PASS
SMALL_REAL_GATE   : PASS (5 TRAIN, 0 failed)
YOUTUBE_HIGHLIGHTS_MATERIALIZED = 154/154
NORMALIZATION     : PASS (TRAIN-only, 111 videos)
INTEGRITY         : PASS (missing=0 duplicate=0 corrupt=0 NaN=0 leakage=0)
FTNET_FORMAL_TRAINING_STATUS = COMPLETE (40/40 epochs)
OFFICIAL_TEST / TVSUM = NOT ACCESSED
```

## 1. 数据身份

- split manifest: `E:\ResearchData\datasets\youtube_highlights\manifests\stage7_ftnet_split_manifest.json`
  （file sha256 611544b2… 为 Windows CRLF 文件字节；sidecar 913c0116… 为 LF 规范化哈希，
  两者差异仅为换行符，canonical `manifest_sha256 = befc0108…` 与
  `content_sha256 = 29241ff8…` 在本地 / AutoDL / 协议文档中一致。）
- split: TRAIN 111 / VALIDATION 23 / CALIBRATION 20 / TOTAL 154；官方 Test 与 TVSum 未触碰。
- 154 源视频 SHA-256 与 split manifest 全量核对通过（0 mismatch），已上传 AutoDL。

## 2. 新增代码（`feat/stage7-ftnet`）

- `src/aic_video_highlight/ftnet/index.py`：冻结索引（相对路径可移植、ffprobe 元数据、PTS 回退）。
- `sampling.py`：2.0fps 网格 + 真实 source 相邻性 `adjacency_mask`。
- `youtube_highlights.py`：MTurk soft-vote Y 目标投影（半开区间、mean-of-overlap、loss_mask）。
- `provider_core.py`：16 维 Native v1.1 + Y + mask 纯 CPU 计算（aspect-agnostic Generic CMP/TS）。
- `real_provider.py`：真实上游三阶段（Qwen vLLM 检索 / RT-DETR 检测+GAP / 物化），
  含记录式 chunk 级恢复（多余右括号 → temperature 0.3 重试；截断 → 512/1024 token 预算）。
- `pipeline.py`：分波生产、断点续跑、失败日志、进度（progress.json/jsonl、ETA、心跳）。
- `integrity.py`：TRAIN-only normalization、原生特征审计、154/154 Integrity Gate。
- `scripts/experiments/stage7/`：launcher registry、统一 runner、SHA 校验批量同步脚本。
- launcher：`launch_experiment.ps1` + `registry.py` 已注册 9 个 stage7 实验，
  Windows 解释器固定为 vhicraft env。

## 3. Small Gate（5 TRAIN，跨类别）

- 5/5 物化成功，0 失败；idx0 `fraction(value==1)=0.6891 ≤ 0.95` → **KEEP**（fallback 未启用），
  决策冻结于 `idx0_gate_decision.json`；全量 111 TRAIN 审计复核为 0.7253，与决策一致。
- 16 维全部可生成，逐维统计见 `native_feature_audit.json`（每维 unique ≈ 全量帧的 0.9×，无退化维度）。

## 4. 全量 154 生产

- 运行：AutoDL RTX 4090D，Qwen/Qwen3.5-4B (vLLM 0.28.0, revision 851bf6e8) +
  PekingU/rtdetr_r50vd (revision df939e66)，2.0fps 全视频网格，wave=20。
- 首轮 151/154，3 个 TRAIN 视频遇确定性 Qwen 输出故障；实现恢复路径后 retry 全部成功 → 154/154。
- 恢复记录写入 retrieval artifact provenance（`recovery_applied/recovery_log`）。

## 5. Normalization / Integrity

- `normalization_stats.json` sha256 = `e49abb2b…`（仅 111 TRAIN，E 盘与 AutoDL 一致）。
- Integrity Gate：154/154，missing/extra/duplicate/corrupt/NaN/Inf = 0，dim 256/16，
  official_test_rows=0，tvsum_rows=0，source_leakage=0。
- E 盘落盘：`E:\ResearchData\derived\VHiCraFTNet\youtube_highlights_ftnet\`
  （154 safetensors + manifests + normalization + audits，分批 SHA-256 校验通过）。

## 6. Formal Training

- Run ID：`ftnet_youtubehl_202609190101`，config `configs/models/ftnet_reference.yaml`
  （batch=4, AdamW 3e-4, cosine 40 epochs, grad clip 1.0, fp32, seed 20260917）。
- 40/40 epochs，1120 steps，258s；best epoch 7，`validation_masked_bce = 0.11702`。
- checkpoint：`last.pt` / `best.pt`（AutoDL `outputs/stage7_ftnet/training/<run_id>/`）。
- 本地 CUDA smoke（RTX 4060，batch=4，真实数据）与 AutoDL 2-step + resume smoke 均 PASS。

## 7. 运行证据与边界

- 训练未涉及 decoder 阈值、Q 头、SFT/LoRA/RL；VALIDATION 仅用于 checkpoint 选择。
- 报告不宣称 16 维 Native 的性能贡献；需按 FTNet.md §8.5 做预注册消融。
- AutoDL 存储：TEMPORARY_RUN_DATA_PRESENT（models 8.9G / datasets 4.1G / derived 18M / outputs 5.6G）。
