"""
Compatibility alias for aic_video_highlight.retrieval.temporal_metrics.

The body of this module was a byte-identical duplicate of
aic_video_highlight.retrieval.temporal_metrics. The canonical implementation now lives
there; only the import path is preserved so historical experiment
runners and their tests keep working unchanged.

Auto-generated during research-repo curation
(chore/research-repo-curation-202610121334).
"""
from __future__ import annotations

from aic_video_highlight.retrieval.temporal_metrics import (TemporalInterval, ZERO_DURATION_EPSILON_SEC, _bounds, _merge_interval_union, _set_intersection_duration, _total_duration, duration_based_metrics, temporal_intersection, temporal_iou, temporal_union)

__all__ = ['TemporalInterval', 'ZERO_DURATION_EPSILON_SEC', '_bounds', '_merge_interval_union', '_set_intersection_duration', '_total_duration', 'duration_based_metrics', 'temporal_intersection', 'temporal_iou', 'temporal_union']
