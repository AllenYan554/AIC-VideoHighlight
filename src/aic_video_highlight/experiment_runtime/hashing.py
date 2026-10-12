"""
Compatibility alias for aic_video_highlight.runtime.hashing.

The body of this module was a byte-identical duplicate of
aic_video_highlight.runtime.hashing. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.runtime.hashing import (canonical_json_bytes, canonical_sha256, file_sha256)

__all__ = ['canonical_json_bytes', 'canonical_sha256', 'file_sha256']
