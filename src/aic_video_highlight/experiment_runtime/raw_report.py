"""
Compatibility alias for aic_video_highlight.runtime.raw_report.

The body of this module was a byte-identical duplicate of
aic_video_highlight.runtime.raw_report. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.runtime.raw_report import (render_raw_report, write_ai_report_inputs)

__all__ = ['render_raw_report', 'write_ai_report_inputs']
