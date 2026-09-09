# Stage 3 Experiment Scripts — Development Dataset Expansion & Generalization Validation

- `build_stage3_smoke_manifest.py`: deterministic builder for the fixed Stage 3 Dev
  Pipeline Smoke Set from a frozen dev split manifest (seeded, rule-based stratified
  selection; no RNG influence on the outcome).

Standalone CLI; not part of the unified `scripts/experiments/registry.py` registry.
