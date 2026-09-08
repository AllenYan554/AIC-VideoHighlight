# Directory Standard

The Git repository contains code, tests, configs, tools, and documentation only. Formal reports and run artifacts belong in the Windows `实验记录` archive or the configured AutoDL runtime roots.

Experiment IDs use lowercase snake case and begin with a stage: `stage5_3_smoke`, `stage5_3_formal`, `stage5_3_full_dev`. One semantic experiment has one ID.

Each run output uses:

```text
<outputs>/<stage>/<experiment_id>/
├── run_manifest.json
├── status.json
├── shards/
├── artifacts/
├── metrics/
├── validation/
├── machine/
│   ├── summary.json
│   ├── metrics.json
│   ├── runtime.json
│   ├── validation.json
│   └── artifact_manifest.json
├── experiment_raw_report.md
└── AI_REPORT_INPUTS.md
```

Logs go only to `<logs>/<stage>/<experiment_id>/` as `run.log`, `stderr.log`, `progress.json`, `progress.jsonl`, and `events.jsonl`. Cache is reusable but non-authoritative. Temporary decode/merge material goes to `<tmp>/<experiment_id>/` and must never be required to interpret a completed run.

Windows and AutoDL roots live in separate environment configs. Scientific configs bind scientific inputs and protocol hashes; changing machines must not require changing the scientific protocol.

As of the 2026-09-08 physical cleanup, tracked repo-level `outputs/` and `logs/` placeholders were removed. The Windows Stage 5 tree contains only scientific records; infrastructure compatibility evidence lives under `实验记录/_工程与基础设施记录/`. Historical scientific paths are not retroactively renamed.
