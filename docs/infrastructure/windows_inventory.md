# Windows Infrastructure Inventory

Inventory time: 2026-09-08. No file was moved during inventory.

## Repository

The only formal local repository is `D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight`, branch `master`, start HEAD `e2c791b`, initially clean and synchronized with `origin/master`.

Top-level code areas are `configs/`, `scripts/`, `src/`, `tests/`, `docs/`, and `remote/`. Empty legacy placeholders `outputs/` and `logs/` exist in the repository and contain no formal experiment evidence. There was no repo-level `tmp/`, `cache/`, or `tools/`. Existing runner scripts were flat under `scripts/`, and configs mixed scientific stages at one level. Historical paths remain untouched for compatibility.

## Experiment archive

Root: `D:\CDUT\硕士\2026-09 AIC\实验记录`

| Directory | Files | Bytes | Classification |
|---|---:|---:|---|
| `数据集快照` | 52 | 6,042,491 | KEEP |
| `Stage1_Baseline` | 37 | 3,449,399 | LEGACY_CANONICAL |
| `Stage2_误差分析与模型对比` | 95 | 4,432,318 | LEGACY_CANONICAL |
| `Stage3_数据集扩展与泛化验证` | 715 | 12,999,413 | LEGACY_CANONICAL |
| `Stage4_候选筛选与时间边界精修` | 2,356 | 76,333,595 | KEEP / active recovery |
| `Stage5_逐帧高光预测与空间构图` | 99 | 29,384,316 | KEEP / frozen through Stage 5.2 |

Stage 5.2 Full Dev local evidence is lightweight: diagnostics, frozen execution config, report, and historical working scripts. The 51,256-frame detector run is not rerun or copied into Git.

## Findings

- Historical experiment directories use mixed `formal`, `results`, `working`, `diagnostics`, and report conventions.
- Machine-generated facts and AI scientific prose are not consistently separated in older experiments.
- Existing flat runner paths are recorded in reports and must remain compatible.
- No second Windows repository was created or found in the instructed project scope.
