# Physical Cleanup — 2026-09-08

This report records the executed cleanup. Scientific JSON/JSONL content was copied byte-for-byte; no Stage 4/5 algorithm or protocol was changed. The full machine migration ledger is `/root/autodl-tmp/archive/physical_cleanup_20260908/remote_migration.tsv`.

## Before and after

| Scope | Before | After |
|---|---|---|
| `/root` AIC scattered files | 17 | 0 |
| AutoDL project checkouts | 2 (one active, one dirty old clone) | 1 active checkout |
| AutoDL top-level nonstandard AIC directories | 5 | 1 protected `_scratch` |
| Flat/scattered legacy logs | 20 across `/root`, `logs/`, and `outputs/experiments` | 20 preserved under `logs/legacy/<stage>/` |
| Windows Stage 5.2 empty directories | 6 | 0 |
| Misplaced Windows infrastructure directories under Stage 5 | 1 | 0 |
| Repo-level run artifact directories | empty tracked `outputs/`, `logs/` placeholders | removed |

## Frozen Stage 5.2 migration

| Source | Destination | Files | Bytes | Source aggregate manifest SHA-256 | Destination aggregate manifest SHA-256 | Status |
|---|---|---:|---:|---|---|---|
| `/root/autodl-tmp/AIC-VideoHighlight-run/outputs/stage5_2_full_dev` | `/root/autodl-tmp/outputs/stage5/stage5_2_full_dev` | 674 | 862,753,310 | `65e9e1cf88207bac8b6b4f5820e015bb824b992ceeddcb144b6ef82d98952087` | same | PASS; old duplicate removed |

Semantic validation after migration:

- records: 51,256; missing/extra/duplicate: 0/0/0
- raw detector: `44ae9383d20fe0afd290c10f8fb78c5bcd2ee5b0709fb7aedf61dbd83d52f399`
- policy-v1: `222258d1d7960639697bd6c3f36339443c85a27bee9dcee7742f24fda238871d`
- crop diagnostics: `f62b0160151d9b68a44d16df3626d89171e56961038bcd9c0c1f4d7203bc59d2`

## Output directory moves

For every row, source and destination manifests were byte-identical before the source was removed.

| Source under active repo `outputs/` | Destination | Files | Aggregate SHA-256 | Status |
|---|---|---:|---|---|
| `baseline_20_qwen3.5_4b_zero_shot_v0` | `outputs/stage1/...` | 87 | `c10b6e73f7f4666a354d7cc4c352df24b2a36561f7e9ff08e239ee4c0c8360c2` | PASS |
| `baseline_20_qwen3.5_4b_zero_shot_v1` | `outputs/stage1/...` | 70 | `f0a7bf42c51f574d2efe531065b99b88059d1df89600a509d809ab949786087b` | PASS |
| `baseline_20_qwen3.5_4b_zero_shot_v2` | `outputs/stage1/...` | 69 | `1f84d2ae745e526b0c5e9f01ed39f518bb0fc2efd41b970090614bd5b488af52` | PASS |
| `baseline_20_qwen3.5_9b_zero_shot_v0` | `outputs/stage1/...` | 70 | `012f76a380e5422964a71338735df25e73942589ee7802616eecd072b59c079c` | PASS |
| `smoke_prompt_v1` | `outputs/stage2/...` | 19 | `06f535567353fb7e25009863f222e2ab133e854505b023713a2e77e6e60288b0` | PASS |
| `smoke_prompt_v2_draft_a` | `outputs/stage2/...` | 37 | `502def85bd59bba708c23303191d21fe9ed3d9749bb5c549f8a5cb98799ab517` | PASS |
| `smoke_prompt_v2_draft_b` | `outputs/stage2/...` | 37 | `c22d494e5a7c455fbc8b27ed993fd0b28a8acc2faf0a38d7b9ed9e6bc1a577c8` | PASS |
| `smoke_prompt_v2_draft_c` | `outputs/stage2/...` | 37 | `9548f27c6108672039684596377e4a9f9fcf44bb7c857426217de906042c5d6d` | PASS |
| `smoke_prompt_v3` | `outputs/stage2/...` | 40 | `49c3fb328b4ac151844d8f9574cdf7a12d9681f222baeabce788c72260f311ce` | PASS |
| `smoke_prompt_v4` | `outputs/stage2/...` | 40 | `f2362928bb332e2d4b1ca3fca046d44a5f8d0a694bbf4a289775b53e9759786e` | PASS |
| `stage3_dev_full_baseline_v1` | `outputs/stage3/...` | 572 | `343a9558087ba0eea901bebe99596d3f9e846098a7a5397fc2535d3571b6bf3f` | PASS |
| `stage3_dev_pipeline_smoke_v1` | `outputs/stage3/...` | 46 | `340b5db6172c14c42add44db242220d22a4583c7296178c247f9c94819fb36a8` | PASS |
| `stage3_hard_full_baseline_v1` | `outputs/stage3/...` | 767 | `73c64377d45e33bb3e0cf6351f417711c8412d1213903a647acd1a05dbb55b2c` | PASS |
| `stage5_1_dev_baseline` | `outputs/stage5/...` | 15 | `c858948fe65ded609ddb2708850e2e465925b4a5776d7af66eb292829fb6e73c` | PASS |
| `stage5_2_formal` | `outputs/stage5/...` | 484 | `c6c6fb0214381cc2458d5583cd1eb0f75edb8e942b10fd562c19a338fc51f672` | PASS |
| `stage5_2_smoke` | `outputs/stage5/...` | 198 | `8a1f10994016a72ad0c20ee10eae5279c7eae625a1cc0fc6fed0ee5e4d6328b8` | PASS |

Additional verified output/archive moves:

- `_stage44_smoke` → `outputs/stage4/stage4_4_dev_smoke_legacy`: 10 files, aggregate `27b9df8f89b1c0ea5cfecb6a25e162d1abc0ab98b3f782fc036d53fefb706c71`.
- `outputs/frozen_baselines` → `outputs/stage1/frozen_baselines`: 1 file, aggregate `6a79927a408c6f0255967f436e6bef8eac22b2ca4f73217392e113eccf59b6e7`.
- `outputs/experiments/baseline20_error_analysis_tmp` → `archive/legacy_artifacts/stage2/...`: 21 files, aggregate `39760e2ad77575b3193ee7be2afdd43baad4eb1a6b5bedb597251bbced87eb54`.
- `logs/legacy_root_20260905` → `logs/legacy/stage1/legacy_root_20260905`: 8 files, aggregate `68b73ac37d680719c334cda2f9c466b3d2ce2c2f73d6f0146cf0a32f18444cac`.

## Root logs and drivers

Seven `/root` log files, three Stage 2 logs, and two flat Stage 4 logs were copied with matching SHA-256 and moved into `logs/legacy/stage2|stage3|stage4`. Nine historical drivers were copied with matching SHA-256 into `archive/legacy_drivers/stage3|stage4`. `stage46_formal_artifacts.tgz` moved to `archive/legacy_artifacts/stage4/` with SHA-256 `c2a8c7bdfc5f33c99feaecd59ab33945ed6475a02f2fd8cbf9bc977bfb5c397a`.

The machine ledger contains the source, destination, action, bytes, source SHA/aggregate SHA, and PASS state for every file/directory move. Destination hashes equal source hashes in every row.

## Old clone

- Removed source: `/root/autodl-tmp/AIC-VideoHighlight`
- Metadata: `/root/autodl-tmp/archive/old_clone_AIC-VideoHighlight_20260908/`
- Complete tar: `/root/autodl-tmp/archive/AIC-VideoHighlight_old_clone_20260908.tar.gz`
- Tar files: 162; bytes: 328,204
- SHA-256: `aeea264ed07a8e44ad5e25b6f6bc61c14169e750495c6097776d9a4511c013c0`
- `tar -tzf` validation and checks for `.git` and unique untracked source: PASS

## Dataset annotation cleanup

`annotations/train.jsonl` moved to `datasets/annotations/train.jsonl`; source/destination SHA-256 `7177731fb7af99e8581c0ec071d116cdb9e6652a6b2b355cd8100364a004c629`. The `.ipynb_checkpoints/train-checkpoint.jsonl` byte-identical duplicate was deleted after the same SHA was confirmed.

## Windows cleanup

`Stage5_逐帧高光预测与空间构图/04_Infrastructure_Compatibility_Import` moved to `_工程与基础设施记录/20260908_Experiment_Infrastructure_Standardization`: 7 files, 14,471 bytes, source/destination manifest equality PASS; aggregate evidence manifest SHA-256 `c80e91dfd57459a65022cf7fb8d6289c525d76996314408a8904763a23b9af3c`.

Six empty Stage 5.2 Full Dev directories and two empty repo placeholders were removed. The exact local list is stored in `empty_dirs_removed.txt`; a Stage 5.2 directory README now maps Development Smoke, Formal, and Full Dev without renaming historical artifacts.

## Intentionally retained

- `/root/autodl-tmp/_scratch/legacy_root_20260905`: contains Heldout-labelled visual files; prohibited from content access or migration in this task.
- Legacy Stage 4 top-level output directories: retained because current Stage 4 recovery depends on those paths.
- Empty Stage 4 formal snapshot/freeze/report directories and canonical root placeholders: retained for scientific structure or required canonical roots.

## Verification

- Windows repo and AutoDL active repo: clean after commit/push/pull.
- AutoDL experiment runtime import: PASS.
- AutoDL CPU tiny fake experiment and validate-only: PASS.
- Stage 5.1 predictions SHA-256: `ca44cc2831e935e82fa7b05820b1269d7f2cc0d360c030b5847a192ede11b9a5`.
- Stage 5.2 compatibility evidence: PASS, 51,256 records.
- GPU calls: 0; Qwen/vLLM calls: 0; Heldout file-content access: 0.
