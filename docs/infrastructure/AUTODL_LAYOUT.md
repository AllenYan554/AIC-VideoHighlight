# AutoDL Layout

```text
/root/autodl-tmp/
├── AIC-VideoHighlight-run/    # only active repository
├── datasets/
├── models/
├── hf-cache/
├── outputs/<stage>/<experiment_id>/
├── logs/<stage>/<experiment_id>/
├── cache/<stage>/<experiment_id>/
├── tmp/<experiment_id>/
└── archive/
```

Do not write scripts, JSON artifacts, or logs directly under `/root` or `/root/autodl-tmp`. The active repo syncs through the SSH Git remote. Dataset, model, and cache roots are never committed.

## Actual state after 2026-09-08 cleanup

- `/root` contains zero scattered AIC Stage/DTL/vLLM files.
- `AIC-VideoHighlight-run` is the only checkout; the dirty old clone is preserved as a verified tar in `archive/`.
- Stage 1/2/3/5 ignored repo outputs were migrated into `outputs/<stage>/`; repo `outputs/` now contains no run artifacts.
- Stage 5.2 Full Dev is canonical at `outputs/stage5/stage5_2_full_dev/` with a byte manifest and unchanged semantic hashes.
- Historical logs are under `logs/legacy/<stage>/`; the logs root has no flat log files.
- Stage 4.2/4.3/4.4/4.6 formal and recovery directories retain their legacy top-level names under `outputs/` to preserve active recovery paths.
- `_scratch/legacy_root_20260905` remains in place because it contains Heldout-labelled visual material; this cleanup did not read or move those files.
