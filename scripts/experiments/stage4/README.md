# Stage 4 Experiment Scripts — Candidate Selection / Boundary Refinement / Dense Temporal Localization

Standalone protocol CLIs (not part of the unified `scripts/experiments/registry.py`
registry):

- `run_candidate_cache.py`: build and validate the Stage 4.2 frozen candidate cache
  (`export` / `validate` / `replay` / `compare`).
- `run_candidate_selection.py`: deterministic Stage 4.3 selection without video or model
  access (`validate-roles` / `select` / `validate-selection` / `replay` / `evaluate` /
  `assess` / `freeze-dev-parameter`).
- `run_boundary_refinement.py`: Stage 4.4 temporal boundary refinement (BR-0 identity /
  BR-1 local refiner).
- `run_dense_temporal_localization.py`: Stage 4.6 DTL-0/DTL-1 development CLI.

CLI examples: `docs/experiments/stage4_frozen_candidate_cache.md`,
`docs/experiments/stage4_lightweight_candidate_selection.md`.
