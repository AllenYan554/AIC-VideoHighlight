# Stage 5 Experiment Scripts — Frame Projection / Subject Localization / Spatial Composition

- `run.py`: canonical Stage 5 launcher. Single source of truth for the Stage 5 experiment
  registry (`RUNNERS` / `CONFIGS` / `LAUNCH`), auto-discovered by
  [`../registry.py`](../registry.py) and the unified PowerShell launcher.
- `run_stage5_3_composition.py`: target-ratio composition runner (CMP-0 vs CMP-1;
  `stage5_3_smoke` / `stage5_3_formal`).

Standalone frozen-stage CLIs (experiments already executed and frozen; not registry
members):

- `run_stage5_center_crop_baseline.py`: Stage 5.1 center-crop baseline runner
  (frozen temporal segments -> official JSONL).
- `run_stage5_2_loc1_smoke.py`: Stage 5.2 LOC-1 smoke runner (RT-DETR on the frozen smoke
  manifest).
- `run_stage5_2_loc1_formal.py`: Stage 5.2 formal runner (shared raw RT-DETR candidates +
  policy v0/v1 + diagnostic crops).
- `run_stage5_2_full_dev.py`: Stage 5.2 Full Dev resumable per-video RT-DETR inference over
  Dev166.
