# AutoDL Infrastructure Inventory

Inventory time: 2026-09-08. This was a read-only SSH inspection; no remote path was changed.

## Repositories

| Path | Size | State | Classification |
|---|---:|---|---|
| `/root/autodl-tmp/AIC-VideoHighlight-run` | 1.1 GB | `master`, clean, HEAD `e2c791b`, SSH origin | ACTIVE_REPO / KEEP |
| `/root/autodl-tmp/AIC-VideoHighlight` | 1012 KB | HEAD `962bc77`, many tracked and untracked changes | ARCHIVE_CANDIDATE; inspect unique work before any action |

## Shared roots

| Path | Size/state | Classification |
|---|---|---|
| `/root/autodl-tmp/datasets` | 11 GB | KEEP |
| `/root/autodl-tmp/models` | empty | KEEP |
| `/root/autodl-tmp/hf-cache` | 27 GB | KEEP |
| `/root/autodl-tmp/outputs` | 40 MB | LEGACY_CANONICAL / inventory before migration |
| `/root/autodl-tmp/logs` | 444 KB | KEEP; standardize future writes |
| `/root/autodl-tmp/cache` | empty | KEEP |
| `/root/autodl-tmp/_archive` | empty | LEGACY_EMPTY |
| `/root/autodl-tmp/_scratch` | 6.1 MB | REVIEW; no automatic deletion |
| `/root/autodl-tmp/tmp` | missing | CREATE in safe migration phase |
| `/root/autodl-tmp/archive` | missing | CREATE in safe migration phase |

Existing outputs include `stage4_2_formal_431`, `stage4_3_formal_431`, `stage4_4_formal_431`, `stage4_6_formal_431`, smoke retries, legacy frozen baselines, and experiment scratch. Stage 3/4 canonical recovery paths are `KEEP / LEGACY_CANONICAL`.

## Root-level scattered files

`/root` contains `dtl1_dev.log`, `formal_assessment.py`, `formal_phase_driver.py`, `phase1_dev_dtl0.sh`, Stage 3 run scripts/logs, and four `vllm_*.log` files. They are migration candidates, not deletion candidates. Provenance-bearing logs should be copied and hash-verified into stage/experiment log directories; ad-hoc drivers should be archived after their Git/uniqueness status is reviewed.

Existing `/root/autodl-tmp/logs` also has flat `stage44_*.log` files and `legacy_root_20260905/`. Preserve these until mapped to experiments.

## Risks

- The old clone is dirty and may contain unique work; never delete or overwrite it automatically.
- Stage 4 recovery is active; cache, role manifests, replay, and protocol paths must not move yet.
- `/root` log filenames lack stable experiment IDs, and outputs mix `formal`, `working`, and retry semantics.
