"""
Compatibility alias for aic_video_highlight.retrieval.video_clip.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.video_clip. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.video_clip import (COORDINATE_CONTRACT, VideoClipError, _SHOWINFO_CHECKSUM, _SHOWINFO_PTS, _extract_with_opencv, _timing_payload, extract_video_clip)

__all__ = ['COORDINATE_CONTRACT', 'VideoClipError', '_SHOWINFO_CHECKSUM', '_SHOWINFO_PTS', '_extract_with_opencv', '_timing_payload', 'extract_video_clip']
