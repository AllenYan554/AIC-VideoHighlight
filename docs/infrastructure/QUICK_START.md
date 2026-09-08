# Experiment Infrastructure Quick Start

1. Ask the Agent to prepare code/config/protocol/tests; do not ask it to babysit a formal run.
2. Pull the committed code through the SSH remote in the single active repo.
3. On AutoDL run `bash scripts/setup/autodl_preflight.sh`.
4. Preview paths and bindings with the experiment command plus `--dry-run`.
5. Launch the formal experiment with its canonical stage launcher and `--resume` when continuing.
6. Watch the terminal progress bar or inspect `logs/<stage>/<experiment_id>/progress.json`.
7. After Ctrl+C, use the printed resume command. Never reuse shards after a config/protocol/input/Git mismatch.
8. On completion, inspect validation and `experiment_raw_report.md`.
9. Send the raw report and files listed in `AI_REPORT_INPUTS.md` to AI.
10. AI writes the separate final `experiment_report.md`.

Infrastructure-only acceptance example on Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python scripts\experiments\stage5\run.py --experiment stage5_infra_tiny_fake --config configs\experiments\stage5\stage5_infra_tiny_fake.json --environment configs\environments\windows_local.json --dry-run
```

Future Stage 5.3 canonical shape (not yet registered or started):

```bash
cd /root/autodl-tmp/AIC-VideoHighlight-run
PYTHONPATH=src python scripts/experiments/stage5/run.py --experiment stage5_3_formal --resume
```

The launcher currently rejects `stage5_3_formal` deliberately. A later scientific task must first create and freeze its config/protocol and register its runner.

Current AutoDL evidence locations after cleanup:

- Stage 5.2 Full Dev: `/root/autodl-tmp/outputs/stage5/stage5_2_full_dev/`
- Legacy logs: `/root/autodl-tmp/logs/legacy/<stage>/`
- Migration manifests: `/root/autodl-tmp/archive/physical_cleanup_20260908/`

Do not use the removed repo-local `outputs/stage5_2_full_dev` path. Stage 4 legacy recovery paths remain unchanged until the recovery work is formally closed.
