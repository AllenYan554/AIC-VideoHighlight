# Experiment Reporting

`experiment_raw_report.md` is generated only from `machine/*.json`. It may state counts, hashes, runtime, validator output, errors, and a factual Gate state. It must not state that a method is effective, prove a scientific claim, or recommend promotion.

`experiment_report.md` is an AI-authored scientific report created after the user supplies the raw report, summary, metrics, runtime, validation, artifact manifest, and figures. It covers context, hypothesis, method, results, baseline comparison, strengths, weaknesses, failure mechanisms, interpretation, limitations, Gate conclusion, and next-stage recommendation.

Experiment programs must never create or overwrite `experiment_report.md`. `AI_REPORT_INPUTS.md` is the explicit handoff boundary.

The artifact manifest schema is `aic.artifact-manifest/v1` and records relative path, type, byte count, SHA-256, optional semantic SHA/count/schema/role, and `created_by`. The run manifest schema is `aic.experiment-run-manifest/v1` and binds run identity, Git, config/protocol hashes, input hashes, model, environment, and output/log/cache/tmp roots.
