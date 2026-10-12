"""
Compatibility alias for aic_video_highlight.retrieval.response_parser.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.response_parser. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.response_parser import (ResponseParseError, TruncatedResponseError, _JSON_FENCE, _TRUNCATION_HINTS, _finite_number, _load_json_object, parse_highlight_response)

__all__ = ['ResponseParseError', 'TruncatedResponseError', '_JSON_FENCE', '_TRUNCATION_HINTS', '_finite_number', '_load_json_object', 'parse_highlight_response']
