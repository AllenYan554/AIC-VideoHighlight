# Stage 2 Experiment Scripts — Highlight Retrieval Diagnosis & Optimization

Stage 2 has no dedicated experiment script: all Baseline20 error-analysis, model-scale
(Qwen3.5-4B vs 9B) and prompt-version runs reused
[`../stage1/run_baseline.py`](../stage1/run_baseline.py) with the frozen manifests and
prompt-version configs (`configs/highlight_retrieval_v1..v4.yaml`).

Analysis-only tooling from this era lives in `scripts/analysis/` (e.g. the salvaged 9B
temporal diagnostic) and `docs/experiments/` (reports).
