"""Run reporting: tables, figures and a markdown report from real artifacts.

Every number in this package is read from the run's own artifacts
(``inference_result.json``, retrieval samples/raw calls, predictions JSONL,
localization policy shards, evaluation reports).  No value is fabricated; when a
reference evaluation is absent, no score is produced at all.
"""

from .report import render_run_report

__all__ = ["render_run_report"]
