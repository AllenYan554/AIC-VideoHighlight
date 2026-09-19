# Stage 7.1 YouTube Highlights Real-Data FTNet Baseline

```text
COMPUTE_STATUS       : COMPLETE (existing artifacts; not rerun in the audit)
LOCAL_ARCHIVE_STATUS : INCOMPLETE
AUDIT_DATE           : 2026-09-19
TRAINING_GIT_HEAD    : caae18ca84b9a50b989c198756920bee4f22a08d
AUDIT_BASE_HEAD      : 172c95176d5170fd3cbb072785c3d922cacd0cbc
BRANCH               : feat/stage7-ftnet
FULL154_RERUN        : NO
FORMAL_TRAINING_RERUN: NO
OFFICIAL_TEST / TVSUM: NOT ACCESSED
```

## Machine-verified current state

- E-drive dataset exists at `E:\ResearchData\derived\VHiCraFTNet\youtube_highlights_ftnet\`.
- 154 safetensors are present: TRAIN 111 / VALIDATION 23 / CALIBRATION 20.
- The 154-record materialization manifest resolves to 154 existing files with zero missing paths.
- Integrity artifact says PASS with missing/extra/duplicate/corrupt/NaN/Inf = 0 and Official Test / TVSum rows = 0.
- TRAIN-only normalization SHA-256 matches its sidecar and the formal-archive copy.
- Training summary records 40 epochs, 1120 steps, best epoch 7 and best validation masked BCE 0.11702357701681278.
- Training history contains 40 real epoch rows and ends at train loss 0.12357279522947372 / validation loss 0.13277547467577.

## Formal local archive

The canonical archive is outside Git:

`D:\CDUT\硕士\2026-09 AIC\实验记录\Stage7_VHiCraFTNet研发\06_Stage7.1_RealData_FTNet_Baseline_20260918`

It now contains:

- `experiment_report.md`
- `figures/` with five training curves and two Native16 audit figures
- `artifact_audit.json`
- `artifact_manifest.json` (including local SHA-256 for all 159 E-drive dataset files)
- `performance_audit.md`
- `performance_summary.json`
- `path_and_storage_audit.md`
- `formal_run_status.json`

The local formal archive is intentionally marked `INCOMPLETE` because the formal `best.pt` and `last.pt` exist only at paths recorded in `training_summary.json`; their files are not present locally. Materialization progress/status/vLLM logs are also absent from the archive.

## SHA qualification

The current machine verifies 154/154 local files, manifest consistency, tensor integrity, and lightweight artifact hashes. The historical remote-to-local SHA PASS cannot be independently replayed because no per-file remote digest manifest or retained sync output is archived and AutoDL is currently unreachable. Do not restate it as a newly verified fact.

## Performance root cause

At the training commit, RT-DETR was constructed inside the per-video detection function. Its constructor calls processor/model `from_pretrained`, so the initialization granularity was **per video**. The exported terminal history independently shows repeated `Loading weights` between consecutive detection counters.

The repair moves RT-DETR to one task-scoped shared lifecycle, makes retrieval/detection heavy stages task-scoped rather than wave-scoped, and emits per-video `performance_per_video.jsonl` plus aggregate `performance_summary.json`. Detection and encoder GAP still share the same forward; Native16, Y, split, 2.0 fps, checkpoints, prompts and training hyperparameters are unchanged.

Exact Qwen/decode/RT-DETR/LOC/CMP/TS/runtime percentages for the completed Full154 run are `NOT RECORDED / CURRENT ARTIFACTS INSUFFICIENT`. The historical 17,585 s materialization, 258 s training and 433-minute workflow values remain unverified claims until the existing remote logs are recovered.

## Storage-path conclusion

`实验运行` is the configured Windows runtime root introduced by commit `a83a646b`; `实验记录` is the permanent archive root. The runtime directory contains a local smoke run, not the formal 40-epoch run, and must not be deleted without an explicit cleanup decision.

## Remaining action

If AutoDL is reopened, only sync the already-existing formal checkpoints and raw progress/status/vLLM logs. Do not rerun Full154. Any performance validation must first be limited to 1–3 videos and explicitly approved.
