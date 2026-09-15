"""Stage 6.1 literature frame-selection contracts (CPU-only, synthetic features)."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from aic_video_highlight.composition.frame_projection import (  # noqa: E402
    CFR_FPS,
    PTS_TABLE,
    VideoTiming,
)
from aic_video_highlight.composition.literature_frame_selection import (  # noqa: E402
    KTSConfig,
    LiteratureFrameSelectionError,
    SampledFrame,
    apply_keep_drop,
    knapSack,
    kts_change_points,
    literature_select,
    sample_frame_mapping,
    sample_frames_uniform,
    select_shots,
    shot_bounds_from_boundaries,
    summary_budget_frames,
)
from aic_video_highlight.composition.literature_features import (  # noqa: E402
    GoogleNetPool5Extractor,
)
from aic_video_highlight.composition.pgl_sum_selector import (  # noqa: E402
    PGL_SUM,
    load_pgl_sum_model,
    pgl_sum_frame_scores,
)
from aic_video_highlight.composition.vasnet_selector import (  # noqa: E402
    VASNet,
    load_vasnet_model,
    vasnet_frame_scores,
)

T = 12
D = 1024


def _features(n: int = T, dim: int = D, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=generator)


# --- model ports -----------------------------------------------------------

def test_pgl_sum_forward_shape_and_range():
    model = PGL_SUM(input_size=D, output_size=D, num_segments=4, heads=8, fusion="add", pos_enc="absolute")
    model.eval()
    scores = pgl_sum_frame_scores(model, _features())
    assert scores.shape == (T,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_pgl_sum_checkpoint_roundtrip(tmp_path):
    model = PGL_SUM(input_size=D, output_size=D, num_segments=4, heads=8, fusion="add", pos_enc="absolute")
    path = tmp_path / "pgl.pth"
    torch.save(model.state_dict(), path)
    loaded = load_pgl_sum_model(path)
    assert set(loaded.state_dict().keys()) == set(model.state_dict().keys())


def test_vasnet_forward_shape_and_checkpoint_roundtrip(tmp_path):
    model = VASNet()
    model.eval()
    scores = vasnet_frame_scores(model, _features())
    assert scores.shape == (T,)
    path = tmp_path / "vas.pth.tar"
    torch.save(model.state_dict(), path)
    loaded = load_vasnet_model(path)
    assert set(loaded.state_dict().keys()) == set(model.state_dict().keys())


def test_vasnet_xai_sum_attention_prefix_is_remapped_strictly(tmp_path):
    model = VASNet()
    xai_sum_state_dict = {
        ("attention." + key.removeprefix("att.")) if key.startswith("att.") else key: value
        for key, value in model.state_dict().items()
    }
    path = tmp_path / "xai-sum-vasnet.pth.tar"
    torch.save(xai_sum_state_dict, path)
    loaded = load_vasnet_model(path)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, expected in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_feature_extractor_honors_gpu_batch_size_without_full_stack():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch):
            self.batch_sizes.append(int(batch.shape[0]))
            return torch.zeros((batch.shape[0], D), dtype=torch.float32)

    extractor = GoogleNetPool5Extractor.__new__(GoogleNetPool5Extractor)
    extractor.torch = torch
    extractor.device = "cpu"
    extractor.batch_size = 2
    extractor.preprocess = lambda tensor: tensor
    extractor.model = FakeModel()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
    features = extractor.extract(frames)
    assert features.shape == (5, D)
    assert extractor.model.batch_sizes == [2, 2, 1]


# --- post-processing -------------------------------------------------------

def test_knapsack_selects_high_value_shots():
    selected = knapSack(5, [2, 3, 1], [1.0, 2.0, 5.0], 3)
    assert selected == [1, 2]


def test_budget_matches_upstream_definitions():
    assert summary_budget_frames("pgl_sum", 1000) == 150
    assert summary_budget_frames("vasnet", 1000) == 150


def test_kts_change_points_are_interior_and_monotonic():
    features = np.random.default_rng(0).standard_normal((24, 16))
    boundaries = kts_change_points(features, KTSConfig(ncp_max=4, vmax=1.0))
    assert list(boundaries) == sorted(set(int(b) for b in boundaries))
    assert all(0 < int(b) < 24 for b in boundaries)
    bounds = shot_bounds_from_boundaries(24, boundaries)
    assert bounds[0][0] == 0 and bounds[-1][1] == 24


def test_sample_frames_uniform_cfr():
    timing = VideoTiming("v", fps=30.0, frame_count=300, timestamp_mode=CFR_FPS)
    frames = sample_frames_uniform(timing, 2.0)
    assert frames[0] == 0
    assert frames == tuple(sorted(set(frames)))
    assert all(0 <= frame < 300 for frame in frames)


def test_sample_frames_uniform_pts_table():
    pts = tuple(Fraction(i, 4) for i in range(8))
    timing = VideoTiming("v", fps=4.0, frame_count=8, timestamp_mode=PTS_TABLE, pts_timestamps=pts)
    frames = sample_frames_uniform(timing, 2.0)
    assert frames == (0, 2, 4, 6)


def test_sample_frame_mapping_records_exact_cfr_trace():
    timing = VideoTiming("v", fps=float(Fraction(30000, 1001)), frame_count=61, timestamp_mode=CFR_FPS)
    mapping = sample_frame_mapping(timing, 2.0)
    assert all(isinstance(item, SampledFrame) for item in mapping)
    assert tuple(item.sample_index for item in mapping) == tuple(range(len(mapping)))
    assert tuple(item.source_frame for item in mapping[:5]) == (0, 15, 30, 45, 60)
    assert mapping[1].target_timestamp == Fraction(1, 2)
    assert mapping[1].source_timestamp == Fraction(15, 1) / Fraction(str(timing.fps))


def test_sample_frame_mapping_pts_uses_fixed_grid_without_drift():
    # Frame 1 is late (0.51 s). Anchoring the next target to its PTS would drift
    # to 1.01 s and choose frame 4; the fixed 2-FPS grid must choose frame 3.
    pts = tuple(Fraction(value) for value in ("0", "0.51", "0.99", "1.0", "1.02", "1.5"))
    timing = VideoTiming("v", fps=4.0, frame_count=len(pts), timestamp_mode=PTS_TABLE, pts_timestamps=pts)
    mapping = sample_frame_mapping(timing, 2.0)
    assert tuple(item.source_frame for item in mapping) == (0, 1, 3, 5)
    assert tuple(item.target_timestamp for item in mapping) == (
        Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)
    )


# --- FS-0 subset adapter ---------------------------------------------------

def _fake_scores(n: int = T) -> list[float]:
    return [0.9 if i < n // 2 else 0.1 for i in range(n)]


def test_literature_select_is_subset_of_fs0_and_drops_only():
    sample_frames = tuple(range(0, 120, 2))  # 60 samples
    scores = _fake_scores(len(sample_frames))
    # FS-0 is a strict subset of the sampled frames.
    fs0 = tuple(frame for frame in sample_frames if frame % 6 == 0)
    selection = literature_select(
        method="pgl_sum",
        sample_frames=sample_frames,
        frame_scores=scores,
        n_frames=120,
        fs0_frames=fs0,
        shot_bounds=((0, 20), (20, 40), (40, 60)),
    )
    assert selection.is_subset_of_fs0
    assert set(selection.kept_frames) <= set(fs0)
    assert set(selection.kept_frames) | set(selection.dropped_frames) == set(fs0)
    assert set(selection.kept_frames) <= set(selection.selected_source_frames)


def test_kept_frames_are_selected_intersect_fs0():
    sample_frames = tuple(range(0, 40, 2))
    scores = [1.0] * 20
    fs0 = (0, 4, 8, 12)
    selection = literature_select(
        method="vasnet",
        sample_frames=sample_frames,
        frame_scores=scores,
        n_frames=40,
        fs0_frames=fs0,
        shot_bounds=((0, 20),),
    )
    assert set(selection.kept_frames) <= set(fs0)
    assert set(selection.kept_frames) == ({0, 4, 8, 12} & set(selection.selected_source_frames))


def test_selected_keyshot_expands_to_source_interval_before_fs0_intersection():
    # Only source frames 0 and 2 are sampled, but selecting that keyshot means
    # the complete source interval [0, 4), including unsampled FS-0 frames.
    selection = literature_select(
        method="pgl_sum",
        sample_frames=(0, 2, 4, 30),
        frame_scores=(1.0, 1.0, 0.0, 0.0),
        n_frames=40,
        fs0_frames=(1, 3, 7, 31),
        shot_bounds=((0, 2), (2, 4)),
    )
    assert selection.selected_shots == (0,)
    assert selection.selected_source_frames == tuple(range(4))
    assert selection.kept_frames == (1, 3)


def test_apply_keep_drop_preserves_bbox_rows_and_drops_rejected_frames():
    rows = [{"frame": f, "bboxes": [[f, f, 10, 20]]} for f in (0, 4, 8, 12)]
    selection = literature_select(
        method="pgl_sum",
        sample_frames=tuple(range(0, 16, 4)),
        frame_scores=[1.0, 1.0, 0.0, 0.0],
        n_frames=16,
        fs0_frames=(0, 4, 8, 12),
        shot_bounds=((0, 2), (2, 4)),
    )
    # The 15% budget over a tiny synthetic video drops at least one shot; the
    # exact KEEP/DROP split is owned by the adapter, but it must be a real
    # subset of FS-0 and every surviving row must be byte-identical.
    assert set(selection.dropped_frames) <= {0, 4, 8, 12}
    assert not set(selection.kept_frames) & set(selection.dropped_frames)
    result = apply_keep_drop({"v": rows}, {"v": selection})
    kept_by_frame = {int(row["frame"]): row for row in result["v"]}
    assert set(kept_by_frame) == set(selection.kept_frames)
    for frame, row in kept_by_frame.items():
        assert row["bboxes"] == [[frame, frame, 10, 20]]


def test_select_rejects_non_monotonic_sample_frames():
    with pytest.raises(LiteratureFrameSelectionError):
        literature_select(
            method="pgl_sum",
            sample_frames=(0, 4, 2),
            frame_scores=[1.0, 1.0, 1.0],
            n_frames=16,
            fs0_frames=(0, 2, 4),
            shot_bounds=((0, 3),),
        )


def test_select_shots_returns_valid_shot_indices():
    lengths = [10, 10]
    scores = [0.5, 0.5]
    for method in ("pgl_sum", "vasnet"):
        selected = select_shots(method, scores, lengths, 10)
        assert selected == sorted(selected)
        assert all(0 <= index < 2 for index in selected)
        assert sum(lengths[index] for index in selected) <= 10
