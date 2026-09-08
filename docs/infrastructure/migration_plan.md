# Infrastructure Migration Plan

All automated migration is copy-first and dry-run by default. `DELETE` is not part of this plan.

| Current | Target | Action | Condition |
|---|---|---|---|
| `/root/autodl-tmp/AIC-VideoHighlight-run` | same | KEEP | sole active repo |
| `/root/autodl-tmp/AIC-VideoHighlight` | `/root/autodl-tmp/archive/repos/AIC-VideoHighlight-legacy-962bc77` | ARCHIVE_CANDIDATE | only after dirty/unique diff review and copy hash verification |
| `/root/*.log` | `/root/autodl-tmp/logs/<stage>/<experiment_id>/` | COPY | map provenance first; retain source until group sign-off |
| `/root/run_stage*`, `formal_*`, `phase*`, `dtl*` | `/root/autodl-tmp/archive/legacy_drivers/<date>/` | COPY | record bytes SHA-256 and origin |
| flat `outputs/stage4_*` | current location | KEEP / LEGACY_CANONICAL | Stage 4 recovery still active |
| future formal outputs | `outputs/<stage>/<experiment_id>/` | NEW STANDARD | runner owns layout |
| future logs | `logs/<stage>/<experiment_id>/` | NEW STANDARD | no root writes |
| future scratch | `tmp/<experiment_id>/` | NEW STANDARD | disposable, never provenance source |

## Safe migration phases

1. Create missing `tmp/`, `archive/`, and stage subdirectories.
2. Generate an explicit JSON migration plan and run `scripts/tools/migrate_experiment_layout.py --dry-run`.
3. Copy only approved files with `--apply`; the tool verifies source/destination SHA-256 and writes a JSONL migration log.
4. Update launch configs/wrappers only after compatibility is proven.
5. Keep all original files until a separate human-approved cleanup task. Deletion candidates: **NONE in this task**.

No remote migration was applied during standardization. This avoids disturbing Stage 4 recovery and preserves the old dirty clone for forensic review.
