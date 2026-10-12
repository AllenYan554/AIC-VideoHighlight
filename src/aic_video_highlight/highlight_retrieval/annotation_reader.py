"""
Compatibility alias for aic_video_highlight.retrieval.annotation_reader.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.annotation_reader. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.annotation_reader import (AnnotationSchemaError, iter_jsonl_records, read_explicit_field)

__all__ = ['AnnotationSchemaError', 'iter_jsonl_records', 'read_explicit_field']
