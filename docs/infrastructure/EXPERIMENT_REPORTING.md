# Experiment Reporting

`experiment_raw_report.md` is generated only from `machine/*.json`. It may state counts, hashes, runtime, validator output, errors, and a factual Gate state. It must not state that a method is effective, prove a scientific claim, or recommend promotion.

`experiment_report.md` is an AI-authored scientific report created after the user supplies the raw report, summary, metrics, runtime, validation, artifact manifest, and figures. It covers context, hypothesis, method, results, baseline comparison, strengths, weaknesses, failure mechanisms, interpretation, limitations, Gate conclusion, and next-stage recommendation.

Experiment programs must never create or overwrite `experiment_report.md`. `AI_REPORT_INPUTS.md` is the explicit handoff boundary.

The artifact manifest schema is `aic.artifact-manifest/v1` and records relative path, type, byte count, SHA-256, optional semantic SHA/count/schema/role, and `created_by`. The run manifest schema is `aic.experiment-run-manifest/v1` and binds run identity, Git, config/protocol hashes, input hashes, model, environment, and output/log/cache/tmp roots.

## VHiCraFTNet experiment record archive conventions (frozen 2026-09-28, Stage 7.2 follow-up)

Applies to every `D:\CDUT\硕士\2026-09 AIC\实验记录\Stage7_VHiCraFTNet研发\<NN>_<Name>_<RunID>\` record (from `23_` onward; historical records are not retroactively modified):

- `experiment_report.md` MUST embed every key result figure inline with `![说明](figures/xxx.png)` in the relevant section. A lead-in sentence before the figure and a data explanation after it are required. Listing figure filenames or writing "see figures/" is not acceptable.
- `figures/` contains PNG files only. `savefig` must write `.png` only; `.svg` / `.pdf` / `.eps` are forbidden unless the user explicitly requests them in that round.
- `results/` contains JSON / CSV / metrics / `figure_sources.json` / machine-readable evidence (figure provenance, run metrics, decisions).
- `config/` contains the frozen configuration and preregistration; `logs/` contains run logs; `checkpoints/` contains the necessary checkpoint records/manifest; `supplementary/` contains reproducible helper scripts.
- Data for figures must come from real run artifacts (history/summary/metrics JSON); fabricated or illustrative figures are forbidden.
