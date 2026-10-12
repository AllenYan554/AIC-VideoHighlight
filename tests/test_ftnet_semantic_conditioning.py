from __future__ import annotations

import numpy as np
import pytest
import torch

from aic_video_highlight.ftnet.model import FTNet, FTNetConfig
from aic_video_highlight.ftnet.semantic_conditioning import (
    REASON_VECTOR_DIM,
    SemanticConditionedFTNet,
    SemanticConditioningError,
    SemanticFTNetExample,
    build_frame_semantics,
    collate_semantic_examples,
    encode_reason,
    primary_candidate_index,
    shuffle_semantics_within_split,
)

FROZEN_D0 = (
    r"E:\ResearchData\cache\VHiCraFTNet\FTNet_Delta_Representation_202609271718"
    r"\runs\D0_seed20260918\best.pt"
)


def _config() -> FTNetConfig:
    return FTNetConfig(
        visual_dim=256, branch_dim=96, native_dim=16, native_branch_dim=32,
        temporal_channels=128, temporal_kernel_size=5, temporal_dilations=(1, 2, 4, 8, 16),
        dropout=0.1, use_delta=True,
    )


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


def _masked_reference(model: FTNet, visual: torch.Tensor, sequence: torch.Tensor, adjacency: torch.Tensor):
    """Reference frozen visual-only (branch_mask=(1,0,0)) forward from a plain FTNet."""

    mask = sequence.unsqueeze(-1)
    sanitized = visual.masked_fill(~mask, 0.0)
    projected = model.visual_branch(sanitized).masked_fill(~mask, 0.0)
    zeros96 = torch.zeros(visual.shape[0], visual.shape[1], 96, device=visual.device)
    zeros32 = torch.zeros(visual.shape[0], visual.shape[1], 32, device=visual.device)
    temporal = model.fusion(torch.cat([projected, zeros96, zeros32], dim=-1)).masked_fill(~mask, 0.0)
    for block in model.temporal_blocks:
        temporal = block(temporal, sequence)
    logits = model.y_head(temporal).squeeze(-1).masked_fill(~sequence, 0.0)
    return logits


# ---------------------------------------------------------------------------
# 1. reason encoder determinism
# ---------------------------------------------------------------------------


def test_reason_encoder_is_deterministic_and_normalized() -> None:
    text = "海边冲浪者完成了一个高难度旋转动作，浪花飞溅。"
    first = encode_reason(text)
    second = encode_reason(text)
    assert first.shape == (REASON_VECTOR_DIM,)
    assert np.array_equal(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5
    assert not np.array_equal(first, encode_reason(text + " 不同。"))
    assert float(np.linalg.norm(encode_reason(""))) == 0.0


# ---------------------------------------------------------------------------
# 2. frame alignment + missing semantics
# ---------------------------------------------------------------------------


def test_frame_alignment_primary_candidate_and_missing_mask() -> None:
    timestamps = np.arange(0.0, 5.0, 0.5)  # 10 frames
    candidates = [
        {"start_sec": 0.0, "end_sec": 2.0, "score": 0.6, "reason": "a"},
        {"start_sec": 1.0, "end_sec": 4.0, "score": 0.9, "reason": "b"},
    ]
    index = primary_candidate_index(timestamps, candidates)
    assert index[3] == 1  # 1.5s covered by both -> highest score wins
    assert index[1] == 0
    assert index[9] == -1  # 4.5s outside every candidate

    semantics = build_frame_semantics(timestamps, candidates)
    assert semantics.score_valid.sum() == 8
    assert not semantics.score_valid[9]
    assert not semantics.reason_valid[9]
    assert float(semantics.score[3]) == pytest.approx(0.9)
    assert np.array_equal(semantics.reason[9], np.zeros(REASON_VECTOR_DIM, dtype=np.float32))
    assert semantics.stats["score_valid_fraction"] == pytest.approx(0.8)


def test_missing_candidate_list_yields_all_invalid() -> None:
    timestamps = np.arange(0.0, 3.0, 0.5)
    semantics = build_frame_semantics(timestamps, [])
    assert semantics.score_valid.sum() == 0
    assert semantics.reason_valid.sum() == 0
    assert float(np.abs(semantics.reason).sum()) == 0.0


def test_injected_text_encoder_is_used_and_cached() -> None:
    """Formal protocol injects a frozen sentence encoder; alignment must be encoder-agnostic."""

    calls: list[str] = []

    def fake_sentence_encoder(text: str) -> np.ndarray:
        calls.append(text)
        vector = np.zeros(8, dtype=np.float32)
        vector[0] = float(len(text)) / 100.0
        vector[1] = 1.0
        return vector

    timestamps = np.arange(0.0, 4.0, 0.5)
    candidates = [
        {"start_sec": 0.0, "end_sec": 2.0, "score": 0.6, "reason": "reason-A"},
        {"start_sec": 2.0, "end_sec": 4.0, "score": 0.9, "reason": "reason-B"},
    ]
    semantics = build_frame_semantics(
        timestamps, candidates, reason_vector_dim=8, text_encoder=fake_sentence_encoder
    )
    assert semantics.reason.shape == (8, 8)
    assert semantics.reason_valid.all()
    # one call per unique candidate (cached across frames)
    assert sorted(calls) == ["reason-A", "reason-B"]
    assert semantics.reason[0, 0] == pytest.approx(len("reason-A") / 100.0)
    assert semantics.reason[6, 0] == pytest.approx(len("reason-B") / 100.0)

    with pytest.raises(SemanticConditioningError):
        build_frame_semantics(
            timestamps, candidates, reason_vector_dim=8,
            text_encoder=lambda text: np.zeros(3, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 3. model shapes, mode gating, perturbation
# ---------------------------------------------------------------------------


def test_semantic_forward_shapes_for_every_mode() -> None:
    config = _config()
    batch, length = 2, 12
    visual = torch.randn(batch, length, 256)
    sequence, adjacency = _masks(batch, length)
    score = torch.rand(batch, length)
    score_valid = torch.ones(batch, length, dtype=torch.bool)
    reason = torch.randn(batch, length, REASON_VECTOR_DIM)
    reason_valid = torch.ones(batch, length, dtype=torch.bool)
    for mode in ("off", "score", "reason", "score_reason"):
        model = SemanticConditionedFTNet(config, semantic_mode=mode).eval()
        with torch.no_grad():
            output = model(
                visual, sequence_mask=sequence, adjacency_mask=adjacency,
                semantic_score=score, semantic_score_valid=score_valid,
                semantic_reason=reason, semantic_reason_valid=reason_valid,
            )
        assert output.logits.shape == (batch, length)
        assert bool(torch.isfinite(output.logits).all())


def test_score_and_reason_perturbation_change_logits_when_active() -> None:
    torch.manual_seed(0)
    config = _config()
    batch, length = 1, 10
    visual = torch.randn(batch, length, 256)
    sequence, adjacency = _masks(batch, length)
    score = torch.rand(batch, length)
    reason = torch.randn(batch, length, REASON_VECTOR_DIM)
    valid = torch.ones(batch, length, dtype=torch.bool)

    score_model = SemanticConditionedFTNet(config, semantic_mode="score").eval()
    with torch.no_grad():
        base = score_model(visual, sequence_mask=sequence, adjacency_mask=adjacency,
                           semantic_score=score, semantic_score_valid=valid)
        perturbed = score_model(visual, sequence_mask=sequence, adjacency_mask=adjacency,
                                semantic_score=torch.zeros_like(score), semantic_score_valid=valid)
    assert not torch.equal(base.logits, perturbed.logits)

    reason_model = SemanticConditionedFTNet(config, semantic_mode="reason").eval()
    with torch.no_grad():
        base_r = reason_model(visual, sequence_mask=sequence, adjacency_mask=adjacency,
                              semantic_reason=reason, semantic_reason_valid=valid)
        perturbed_r = reason_model(visual, sequence_mask=sequence, adjacency_mask=adjacency,
                                   semantic_reason=torch.zeros_like(reason), semantic_reason_valid=valid)
    assert not torch.equal(base_r.logits, perturbed_r.logits)


def test_invalid_semantic_frames_are_zero_slot() -> None:
    torch.manual_seed(1)
    config = _config()
    batch, length = 1, 8
    visual = torch.randn(batch, length, 256)
    sequence, adjacency = _masks(batch, length)
    score = torch.rand(batch, length)
    reason = torch.randn(batch, length, REASON_VECTOR_DIM)
    valid = torch.zeros(batch, length, dtype=torch.bool)
    model = SemanticConditionedFTNet(config, semantic_mode="score_reason").eval()
    model.capture_semantic_slots = True
    with torch.no_grad():
        model(visual, sequence_mask=sequence, adjacency_mask=adjacency,
              semantic_score=score, semantic_score_valid=valid,
              semantic_reason=reason, semantic_reason_valid=valid)
    slots = model.last_semantic_slots
    assert torch.equal(slots["score"], torch.zeros_like(slots["score"]))
    assert torch.equal(slots["reason"], torch.zeros_like(slots["reason"]))


def test_active_mode_requires_semantic_tensors() -> None:
    model = SemanticConditionedFTNet(_config(), semantic_mode="score").eval()
    sequence, adjacency = _masks(1, 6)
    with pytest.raises(SemanticConditioningError):
        model(torch.randn(1, 6, 256), sequence_mask=sequence, adjacency_mask=adjacency)


# ---------------------------------------------------------------------------
# 4. B0 invariance vs frozen baseline (real checkpoint, real validation video)
# ---------------------------------------------------------------------------


def test_semantic_off_matches_masked_baseline_bitwise() -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    import json
    from pathlib import Path

    from safetensors.numpy import load_file

    from aic_video_highlight.ftnet.native_schema import standardize_native

    payload = torch.load(FROZEN_D0, map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    values["temporal_dilations"] = tuple(values["temporal_dilations"])
    config = FTNetConfig(**values)

    baseline = FTNet(config).to("cuda")
    baseline.load_state_dict(payload["model_state_dict"])
    baseline.eval()

    candidate = SemanticConditionedFTNet(config, semantic_mode="off").to("cuda")
    shared = {
        key: value for key, value in payload["model_state_dict"].items()
        if key.startswith(("visual_branch.", "fusion.", "temporal_blocks.", "y_head."))
    }
    missing, unexpected = candidate.load_state_dict(shared, strict=False)
    assert not unexpected
    candidate.eval()

    data_root = Path(r"E:\ResearchData\derived\VHiCraFTNet\YouTubeHighlights_FTNet_v1")
    stats_payload = json.loads(
        (data_root / "normalization" / "normalization_stats.json").read_text(encoding="utf-8")
    )
    from aic_video_highlight.ftnet.native_schema import NormalizationStats

    stats = NormalizationStats(
        mean=tuple(float(v) for v in stats_payload["mean"]),
        std=tuple(float(v) for v in stats_payload["std"]),
        log1p_fields=tuple(int(v) for v in stats_payload["log1p_fields"]),
    )
    max_diff = 0.0
    for path in sorted((data_root / "validation").glob("*.safetensors"))[:4]:
        tensors = load_file(str(path))
        visual = torch.from_numpy(np.asarray(tensors["visual"], dtype=np.float32)).unsqueeze(0).to("cuda")
        raw_native = torch.from_numpy(np.asarray(tensors["native"], dtype=np.float32))
        missing_mask = torch.from_numpy(np.asarray(tensors["native_missing"], dtype=bool))
        native = standardize_native(raw_native, stats, missing_mask).unsqueeze(0).to("cuda")
        length = int(visual.shape[1])
        sequence = torch.ones(1, length, dtype=torch.bool, device="cuda")
        adjacency = torch.from_numpy(np.asarray(tensors["adjacency_mask"], dtype=bool)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            reference = _masked_reference(baseline, visual, sequence, adjacency)
            off = candidate(
                visual, native_features=native, sequence_mask=sequence, adjacency_mask=adjacency
            ).logits
        assert torch.equal(reference, off), (path.name, float((reference - off).abs().max()))
        max_diff = max(max_diff, float((reference - off).abs().max()))
    assert max_diff == 0.0


# ---------------------------------------------------------------------------
# 5. deterministic shuffle + no cross-split leakage
# ---------------------------------------------------------------------------


def _tiny_semantics(video_id: str, length: int, seed: int) -> "object":
    rng = np.random.default_rng(seed)
    candidates = [
        {"start_sec": 0.0, "end_sec": length * 0.5, "score": float(rng.random()), "reason": f"{video_id}-a"},
        {"start_sec": length * 0.25, "end_sec": length, "score": float(rng.random()), "reason": f"{video_id}-b"},
    ]
    timestamps = np.arange(length, dtype=np.float64)
    return build_frame_semantics(timestamps, candidates)


def test_shuffle_is_deterministic_and_confined_to_split() -> None:
    train = {f"tr{i}": _tiny_semantics(f"tr{i}", 8, 100 + i) for i in range(3)}
    val = {f"va{i}": _tiny_semantics(f"va{i}", 8, 200 + i) for i in range(2)}

    train_a = shuffle_semantics_within_split(train, seed=20260928, split_salt="TRAIN")
    train_b = shuffle_semantics_within_split(train, seed=20260928, split_salt="TRAIN")
    for video_id in train:
        assert np.array_equal(train_a[video_id].score, train_b[video_id].score)
        assert np.array_equal(train_a[video_id].reason, train_b[video_id].reason)

    val_shuffled = shuffle_semantics_within_split(val, seed=20260928, split_salt="VALIDATION")
    # a VALIDATION shuffle must never equal any TRAIN-shuffled value set
    train_values = sorted(float(x) for item in train_a.values() for x in item.score.tolist())
    val_values = sorted(float(x) for item in val_shuffled.values() for x in item.score.tolist())
    assert train_values != val_values
    # shuffling preserves the multiset of scores per split
    original = sorted(float(x) for item in train.values() for x in item.score.tolist())
    assert sorted(float(x) for item in train_a.values() for x in item.score.tolist()) == original


# ---------------------------------------------------------------------------
# 6. collate + tiny forward/backward + evaluator smoke
# ---------------------------------------------------------------------------


def test_collate_pads_and_defaults_missing_semantics_to_invalid() -> None:
    examples = [
        SemanticFTNetExample(
            visual_features=torch.randn(5, 256), targets=torch.tensor([1.0, 0, 1, 0, 0]),
            adjacency_mask=torch.ones(5, dtype=torch.bool), loss_mask=torch.ones(5, dtype=torch.bool),
            semantic_score=torch.rand(5), semantic_score_valid=torch.ones(5, dtype=torch.bool),
            semantic_reason=torch.randn(5, REASON_VECTOR_DIM),
            semantic_reason_valid=torch.ones(5, dtype=torch.bool),
        ),
        SemanticFTNetExample(
            visual_features=torch.randn(2, 256), targets=torch.tensor([0.0, 1.0]),
            adjacency_mask=torch.ones(2, dtype=torch.bool), loss_mask=torch.ones(2, dtype=torch.bool),
        ),
    ]
    batch = collate_semantic_examples(examples)
    assert batch.visual_features.shape == (2, 5, 256)
    assert batch.sequence_mask[1, 2:].sum() == 0
    assert batch.semantic_score_valid[1, 2:].sum() == 0
    assert float(batch.semantic_score[1, 2:].abs().sum()) == 0.0


def test_tiny_forward_backward_and_evaluator_smoke() -> None:
    from aic_video_highlight.ftnet.evaluation import average_precision
    from aic_video_highlight.ftnet.losses import masked_binary_cross_entropy
    from aic_video_highlight.ftnet.trainer import build_adamw_optimizer

    torch.manual_seed(3)
    config = _config()
    model = SemanticConditionedFTNet(config, semantic_mode="score_reason")
    optimizer = build_adamw_optimizer(model, learning_rate=3e-4, weight_decay=1e-2,
                                      betas=(0.9, 0.999), eps=1e-8)
    examples = [
        SemanticFTNetExample(
            visual_features=torch.randn(6, 256), targets=torch.randint(0, 2, (6,)).float(),
            adjacency_mask=torch.ones(6, dtype=torch.bool), loss_mask=torch.ones(6, dtype=torch.bool),
            semantic_score=torch.rand(6), semantic_score_valid=torch.ones(6, dtype=torch.bool),
            semantic_reason=torch.randn(6, REASON_VECTOR_DIM),
            semantic_reason_valid=torch.ones(6, dtype=torch.bool),
        ),
        SemanticFTNetExample(
            visual_features=torch.randn(4, 256), targets=torch.randint(0, 2, (4,)).float(),
            adjacency_mask=torch.ones(4, dtype=torch.bool), loss_mask=torch.ones(4, dtype=torch.bool),
            semantic_score=torch.rand(4), semantic_score_valid=torch.ones(4, dtype=torch.bool),
            semantic_reason=torch.randn(4, REASON_VECTOR_DIM),
            semantic_reason_valid=torch.ones(4, dtype=torch.bool),
        ),
    ]
    batch = collate_semantic_examples(examples)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch.visual_features, sequence_mask=batch.sequence_mask, adjacency_mask=batch.adjacency_mask,
        semantic_score=batch.semantic_score, semantic_score_valid=batch.semantic_score_valid,
        semantic_reason=batch.semantic_reason, semantic_reason_valid=batch.semantic_reason_valid,
    )
    loss = masked_binary_cross_entropy(output.logits, batch.targets, batch.loss_mask)
    assert bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(
        name.startswith(("score_projection", "reason_projection")) and p.grad is not None
        for name, p in model.named_parameters()
    )
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(
            batch.visual_features, sequence_mask=batch.sequence_mask, adjacency_mask=batch.adjacency_mask,
            semantic_score=batch.semantic_score, semantic_score_valid=batch.semantic_score_valid,
            semantic_reason=batch.semantic_reason, semantic_reason_valid=batch.semantic_reason_valid,
        )
    mask = batch.loss_mask.numpy()
    ap = average_precision(
        batch.targets.numpy()[mask].astype(bool), output.probabilities.numpy()[mask]
    )
    assert 0.0 <= float(ap) <= 1.0


def test_parameter_count_adds_only_semantic_projections() -> None:
    config = _config()
    off = SemanticConditionedFTNet(config, semantic_mode="off")
    active = SemanticConditionedFTNet(config, semantic_mode="score_reason")
    off_params = sum(p.numel() for p in off.parameters())
    active_params = sum(p.numel() for p in active.parameters())
    # projections exist in every mode; only their activation differs
    assert off_params == active_params == 146177
    # frozen visual-only active capacity (166945 minus unused delta 24864 / native 608)
    frozen_active = 166945 - 24864 - 608
    score_projection = 2 * 96 + 96 + 2 * 96      # Linear(2,96) + LayerNorm(96)
    reason_projection = 129 * 32 + 32 + 2 * 32   # Linear(129,32) + LayerNorm(32)
    assert off_params == frozen_active + score_projection + reason_projection
