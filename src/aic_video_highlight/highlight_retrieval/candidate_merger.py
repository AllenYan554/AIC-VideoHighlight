"""
Compatibility alias for aic_video_highlight.retrieval.candidate_merger.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.candidate_merger. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.candidate_merger import (_merge_pair, merge_segments)

__all__ = ['_merge_pair', 'merge_segments']
