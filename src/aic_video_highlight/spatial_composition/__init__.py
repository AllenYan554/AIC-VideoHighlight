"""Stage 5 spatial composition: official output contract and crop baselines."""

from .center_crop import CenterCropBox, compute_center_crop, derived_height
from .frame_projection import (
    CFR_FPS,
    PTS_TABLE,
    FrameProjection,
    VideoTiming,
    frame_timestamp,
    frames_in_segment,
    fps_rational,
    project_segments,
)
from .metadata import (
    METADATA_SCHEMA,
    VideoMetadataProbeError,
    VideoSpatialMeta,
    extract_pts_timestamps,
    get_or_probe_metadata,
    load_metadata_cache,
    probe_spatial_meta,
    save_metadata_cache,
    timing_from_metadata,
)
from .submission import (
    SubmissionValidationError,
    build_submission_record,
    render_submission_line,
    write_predictions_jsonl,
)
from .validation import ValidationIssue, ValidationReport, validate_submission_file

__all__ = [
    "CFR_FPS",
    "PTS_TABLE",
    "CenterCropBox",
    "FrameProjection",
    "METADATA_SCHEMA",
    "SubmissionValidationError",
    "ValidationIssue",
    "ValidationReport",
    "VideoMetadataProbeError",
    "VideoSpatialMeta",
    "VideoTiming",
    "build_submission_record",
    "compute_center_crop",
    "derived_height",
    "extract_pts_timestamps",
    "frame_timestamp",
    "frames_in_segment",
    "fps_rational",
    "get_or_probe_metadata",
    "load_metadata_cache",
    "probe_spatial_meta",
    "project_segments",
    "render_submission_line",
    "save_metadata_cache",
    "timing_from_metadata",
    "validate_submission_file",
    "write_predictions_jsonl",
]
