# Experiment Workflow

## Before a formal run

The Agent prepares code, scientific config, frozen protocol, validators, sharding/resume, progress reporting, machine evidence, smoke tests, and a Git commit. It then stops. Formal GPU work is launched by the user with one command.

Every runner checks config SHA-256, protocol SHA-256, input hashes, and Git HEAD. Resume is rejected if identity changes. Status values are `PENDING`, `RUNNING`, `INTERRUPTED`, `COMPLETED`, `FAILED`, and `VALIDATION_FAILED`.

Ctrl+C must atomically retain complete shards, flush progress, write `INTERRUPTED`, and print the resume command. GPU telemetry is optional and must never be a fatal dependency.

## During and after the run

The runner owns progress, ETA, shards, logs, validation, structured evidence, artifact hashes, and the factual raw report. On completion it points the user to `AI_REPORT_INPUTS.md`; it does not claim scientific success or promotion.

Historical Stage 1–5.2 entrypoints remain supported. New canonical stage launchers live under `scripts/experiments/<stage>/run.py`; old documented paths may remain as thin wrappers.
