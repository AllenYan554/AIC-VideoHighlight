# Stage 4.3 Lightweight Candidate Selection

Stage 4.3 是 Frozen Candidate Cache 之上的确定性 KEEP/DROP 层。它不读取视频、
不调用模型、不重新 merge，也不修改候选的时间、score、reason、source chunk 或 lineage。
任何 selected prediction 都由 validator 验证后，再按 candidate ID 从 Stage 4.2 cache
取回原始字段。

## Data roles

- Dev-Tune166：唯一允许选择 SEL-1/SEL-2 参数的角色。
- Hard-Stress229：参数冻结后的压力/泛化评估，不允许反馈调参。
- Audit36：所有方法和参数冻结后的 post-hoc diagnostic。
- Heldout392：保持 locked，不进入 cache、manifest、selection 或 evaluation。

Role manifests 由 Stage 4.1 私有映射中的 `video_id` 与 `split` 身份投影生成；生成器
不访问或复制 metric、stratum、rule level、人工判断或裁决字段。正式 manifests 必须
两两不交且并集恰为 Frozen Cache 的 431 条记录。

## Selector contract

正式配置冻结于 `configs/stage4_3_selection_protocol.json`。

- SEL-0：全部 KEEP；Stage 4.3 identity control。
- SEL-1：`score >= threshold`；threshold 只能来自预注册 grid。
- SEL-2：`score >= max_score(video) - delta`；ties 全保留，无 Top-1/budget。
- SEL-3：只有 context-aware 明确负向 lexical rule 命中且无正向 override 时 DROP；
  其余情况（包括 uncertain）全部 KEEP。

SEL-3 规则只来自 Stage 2 已冻结的 v0 error analysis 和 conservative-filtering 教训。
裸 `准备/铺垫/过渡/重复` 词不构成删除证据；规则禁止包含 video ID。

## Result and validation

Selection result 使用 `aic.candidate-selection-result/v1`，包含 cache/role/config hashes、
每个候选恰一条 decision、原候选 semantic hash、selected ID 列表、per-video hash 和
global hash。结果不复制候选区间，replay 只从 cache 取回候选，因此边界修改没有合法
表达方式。Validator 会重新执行 selector 并要求整个 canonical payload 精确一致，同时
拒绝 reference、metric、Audit label、Heldout、未知/重复/缺失 candidate decision。

Evaluation 是独立后置职责。只有 selection result 已验证且 replay 精确匹配后，才读取
Stage 3 frozen predictions 中的 weak reference，并复用未修改的
`baseline_experiment.evaluate_weak_references`。报告必须称这些为 weak-reference
development metrics，不能称为官方 Ground Truth 或最终质量。

## CLI

以下用占位路径说明调用关系；Formal 应使用 protocol 中锁定的真实 hashes 和 role。

```powershell
python scripts/experiments/stage4/run_candidate_selection.py validate-roles `
  --cache-dir <FORMAL_431/cache_build_a> `
  --role-dir <Stage4.3/00_PROTOCOL>

python scripts/experiments/stage4/run_candidate_selection.py select `
  --cache-dir <FORMAL_431/cache_build_a> `
  --role-manifest <Stage4.3/00_PROTOCOL/dev_tune_166.json> `
  --protocol configs/stage4_3_selection_protocol.json `
  --selector SEL-1 --parameter 0.80 `
  --output <new-selection-result.json>

python scripts/experiments/stage4/run_candidate_selection.py validate-selection `
  --cache-dir <FORMAL_431/cache_build_a> `
  --role-manifest <role-manifest.json> `
  --selection-result <selection-result.json>

python scripts/experiments/stage4/run_candidate_selection.py replay `
  --cache-dir <FORMAL_431/cache_build_a> `
  --role-manifest <role-manifest.json> `
  --selection-result <selection-result.json> `
  --output <new-selected-predictions.jsonl>

python scripts/experiments/stage4/run_candidate_selection.py evaluate `
  --cache-dir <FORMAL_431/cache_build_a> `
  --role-manifest <role-manifest.json> `
  --selection-result <selection-result.json> `
  --selected-predictions <selected-predictions.jsonl> `
  --frozen-predictions <Stage3-frozen-predictions.jsonl> `
  --output <new-evaluation.json>

python scripts/experiments/stage4/run_candidate_selection.py assess `
  --baseline-evaluation <SEL-0-evaluation.json> `
  --candidate-evaluation <candidate-evaluation.json> `
  --protocol configs/stage4_3_selection_protocol.json `
  --phase dev `
  --output <new-assessment.json>

python scripts/experiments/stage4/run_candidate_selection.py freeze-dev-parameter `
  --baseline-evaluation <SEL-0-evaluation.json> `
  --candidate-evaluation <grid-evaluation-1.json> `
  --candidate-evaluation <grid-evaluation-2.json> `
  --protocol configs/stage4_3_selection_protocol.json `
  --selector SEL-1 `
  --output <new-parameter-freeze-evidence.json>
```

所有输出均拒绝覆盖。`select` 会拒绝 protocol grid 外的参数。Formal 开始后不得修改
grid、SEL-3 rules、Recall Guardrail、Promotion Gate、Hard-Stress Gate 或 tie-break。
Audit36 跨 Dev/Hard 时重复传入两个 `--frozen-predictions`；evaluator 会拒绝跨文件
重复 video ID，不需要改写任一 frozen source。

## Formal order

1. 验证 Git、cache、roles、protocol hashes。
2. Dev166 运行 SEL-0，再运行 SEL-1 grid、SEL-2 grid 与固定 SEL-3。
3. 先过 Recall Guardrail，再按 protocol 的有序 tie-break 在 Dev166 锁定参数。
4. 写入 parameter-freeze evidence，之后原样运行 Hard229。
5. Hard 失败只报告，不回调参数。
6. 全部冻结且 Hard 完成后，才允许 Audit36 post-hoc diagnostic。

Development Smoke 只能证明 routing、determinism、validation、replay 和 evaluation
plumbing；不得用于参数选择或修改预注册协议。
