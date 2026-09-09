# Stage 1 Experiment Scripts — Highlight Retrieval Baseline Establishment

Canonical experiment entrypoints born in Stage 1 and reused by later retrieval stages.

- `run_highlight_retrieval.py`: single-video highlight candidate retrieval against the vLLM
  endpoint (Qwen video understanding pipeline smoke / one-off runs).
- `run_baseline.py`: reproducible weak-reference Baseline20 development experiments from a
  frozen manifest (git-HEAD-bound reports, resumable, per-sample raw journal). Stage 2
  (diagnosis & prompt/model comparison) reused this runner with prompt-version configs
  (`configs/highlight_retrieval_v*.yaml`); it has no dedicated Stage 2 script.

These runners are standalone CLIs and are not part of the unified
`scripts/experiments/registry.py` launch-spec registry (registry membership is for the
resumable experiment-runtime experiments, currently Stage 5).
