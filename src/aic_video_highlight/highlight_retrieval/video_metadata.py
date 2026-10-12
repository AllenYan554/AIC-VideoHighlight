"""
Compatibility alias for aic_video_highlight.retrieval.video_metadata.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.video_metadata. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.video_metadata import (VideoProbeError, _parse_rate, _positive_float, _probe_with_opencv, probe_video)

__all__ = ['VideoProbeError', '_parse_rate', '_positive_float', '_probe_with_opencv', 'probe_video']
