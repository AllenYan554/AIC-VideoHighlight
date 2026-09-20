# Stage 7.1 YouTube Highlights Real-Data FTNet Baseline

```text
COMPUTE_STATUS       : COMPLETE (existing artifacts; not rerun in the audit)
LOCAL_ARCHIVE_STATUS : COMPLETE_AND_ARCHIVED
AUDIT_DATE           : 2026-09-20
TRAINING_GIT_HEAD    : caae18ca84b9a50b989c198756920bee4f22a08d
BRANCH               : feat/stage7-ftnet
FULL154_RERUN        : NO
FORMAL_TRAINING_RERUN: NO
OFFICIAL_TEST / TVSUM: NOT ACCESSED
```

## Machine-verified current state

- Processed data is read directly from `E:\ResearchData\derived\VHiCraFTNet\youtube_highlights_ftnet\`.
- 154 safetensors are present: TRAIN 111 / VALIDATION 23 / CALIBRATION 20.
- The 154-record materialization manifest resolves to 154 existing files with zero missing paths.
- Integrity is PASS with missing/extra/duplicate/corrupt/NaN/Inf = 0 and Official Test / TVSum rows = 0.
- Training summary records 40 epochs, 1120 steps, best epoch 7 and best validation masked BCE 0.11702357701681278.
- `best.pt` is epoch 7 / step 196; `last.pt` is epoch 40 / step 1120. Both carry training Git identity `caae18ca84b9a50b989c198756920bee4f22a08d`.
- AutoDL and local checkpoint SHA-256 values match: best `0204b8c31b44de01ba0c54c1ead92f07549adaa324afd44d997bd57254b94899`, last `3997cb7091ae5a35dc6989ba1e03e224b93030ff871b11e6fbb452eaa941c9a1`.

## Formal local archive

The canonical archive is outside Git:

`D:\CDUT\硕士\2026-09 AIC\实验记录\Stage7_VHiCraFTNet研发\06_Stage7.1_RealData_FTNet_Baseline_20260918`

Its stable layout follows the existing Stage 4/5 convention:

```text
config/
figures/
results/
  checkpoints/best.pt
  checkpoints/last.pt
supplementary/
experiment_report.md
```

The root contains no flat JSON. Machine results, checkpoints, configuration snapshots, plots, raw runtime evidence, content manifest, audits and checkpoint identity verification live under their respective directories. See `supplementary/artifact_manifest.json` for local SHA-256 evidence.

## Performance qualification

The repair keeps one RT-DETR instance for the materialization task and makes retrieval/detection heavy stages task-scoped. Detection and encoder GAP still share one forward; Native16, Y, split, 2.0 fps, checkpoints, prompts and training hyperparameters are unchanged.

The recovered materialization progress snapshot covers the final one-video continuation, not a single uninterrupted 154-video wall-clock interval. Therefore no full-run stage percentage is inferred. No benchmark or model computation was run during archiving.

## Storage-path conclusion

`实验运行` remains the configured Windows runtime/resume root; `实验记录` is the permanent archive root. The local smoke checkpoint and logs are unique development evidence, so the runtime directory is retained. Empty, unreferenced `E:\ResearchData\archives` and `E:\ResearchData\intermediate` directories were removed; `cache` remains the configured cache root and `manifests` retains unique storage-inventory evidence.

Future FTNet training resolves `output_root` from the environment's `derived` base, so it consumes the existing E-drive materialization directly and does not need Qwen or RT-DETR.
