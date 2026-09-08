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

Legacy locations remain readable until separately migrated. Stage 4 cache/replay/role manifests and all Stage 5.1/5.2 frozen artifacts are protected as `LEGACY_CANONICAL`.
