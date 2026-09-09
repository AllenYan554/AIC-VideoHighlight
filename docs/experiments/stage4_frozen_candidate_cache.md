# Stage 4.2 Frozen Candidate Cache

本模块只消费 Stage 3 已持久化的 Dev/Hard frozen v0 artifacts，不读取视频、
weak-reference、metrics 或 Audit36 标签，也不调用 Qwen/vLLM。

## Source contract

每个 `--source SPLIT=EXPERIMENT_DIR` 必须是 `dev` 或 `hard`，并采用以下任一布局：

```text
EXPERIMENT_DIR/
  run_config.json
  predictions.jsonl
  raw/<video_id>.json
```

或本地 Stage 3 归档布局：

```text
EXPERIMENT_DIR/
  config/run_config.json
  results/predictions.jsonl
  results/raw/<video_id>.json
```

只接受 `aic_highlight_dev_v1.1`。任何其他 split（包括 Heldout）都会被拒绝。

## Cache contract

- schema：`aic.frozen-candidate-cache/v1`
- ID scheme：`aic.candidate-id/v1`
- canonicalization：`aic.canonical-json/v1`
- JSON：UTF-8、Unicode 原文、key 排序、compact separators、LF 结尾、禁止 NaN/Infinity
- record：`records/<video_id>.json`
- manifest：`cache_manifest.json`

Raw candidate ID 由 video、chunk identity、chunk-local/global interval、稳定候选位置、
score、reason 与 source chunk 的 canonical SHA-256 决定。Merged candidate ID 由 video、
merged fields 与排序后的 contributor raw IDs 决定。ID 不依赖时间戳、随机数、绝对路径
或文件系统遍历顺序。

每条 video record 另保存无标签的 `source_chunks`（chunk index/bounds、finish reason、
candidate count、raw response SHA-256），以覆盖“chunk 存在但没有 candidate”的 lineage；
不把 raw response 正文或推理 latency 纳入下游 candidate cache。

Exporter 从 frozen raw response 重新调用未修改的 v0 parser，并核对持久化 parsed
candidates；随后使用与 frozen merger 相同的稳定排序与 pairwise union 步骤传播 contributor
lineage，同时调用原 merger 交叉核验。只有 parsed、merged 和 frozen prediction 三层完全一致
才会写出 cache。

Cache 只保存模型可观测候选与 frozen provenance/hash。禁止写入 weak-reference、Precision、
Recall、F1、Temporal IoU、reference coverage、Audit36、reviewer 或 adjudication 信息。

## CLI

```powershell
python scripts/experiments/stage4/run_candidate_cache.py export `
  --source dev=<DEV_EXPERIMENT_DIR> `
  --source hard=<HARD_EXPERIMENT_DIR> `
  --output-dir <CACHE_DIR>

python scripts/experiments/stage4/run_candidate_cache.py validate `
  --cache-dir <CACHE_DIR> `
  --report <VALIDATION_REPORT_JSON>

python scripts/experiments/stage4/run_candidate_cache.py replay `
  --cache-dir <CACHE_DIR> `
  --output <REPLAYED_PREDICTIONS_JSONL>

python scripts/experiments/stage4/run_candidate_cache.py compare `
  --cache-dir <CACHE_DIR> `
  --source dev=<DEV_EXPERIMENT_DIR> `
  --source hard=<HARD_EXPERIMENT_DIR> `
  --report <IDENTITY_REPORT_JSON>
```

Cache output 是 immutable：目标目录非空时 exporter 拒绝覆盖。确定性 rebuild 应使用两个
全新的目录并比较 `global_semantic_sha256` 与 canonical manifest bytes。

`--limit-per-split` 只允许用于 development smoke；Formal 431 必须省略该参数。
Manifest 的 `complete_source_export` 在且仅在每个 source 的全部 records 均已导出时为
`true`；Formal 431 Gate 必须要求该值为 `true`。
