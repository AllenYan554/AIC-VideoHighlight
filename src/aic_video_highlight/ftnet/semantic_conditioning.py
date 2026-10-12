"""Semantic-conditioned FTNet: deterministic Qwen score/reason conditioning.

This module adds an INDEPENDENTLY SWITCHABLE semantic branch to the frozen
visual-only FTNet.  The frozen baseline forward (round-09 ``DeltaRepresentationFTNet``
with ``branch_mask=(1,0,0)``) builds a 224-wide fusion input::

    [ visual_projected (96) | zero-delta (96) | zero-native (32) ]

The semantic candidate replaces the two never-used zero slots with semantic
projections and keeps everything else frozen::

    [ visual_projected (96) | score_slot (96) | reason_slot (32) ]

Because ``semantic_mode == "off"`` fills both slots with literal zeros, the
candidate forward is bitwise identical to the frozen baseline (verified by
``test_semantic_off_matches_masked_baseline``).  Only the *fusion input content*
is extended; the fusion weights (224->128), TCN and Y head are unchanged.

Semantic inputs are built deterministically from the persisted Qwen retrieval
artifact (``aic.stage7.ftnet.retrieval-artifact/v1``):
  * score  : the primary covering candidate's ``score`` in [0, 1]
  * reason : a deterministic OFFLINE hashed character-ngram vector (no model
             call, no online embedding API, no LLM retraining)
  * missing: an explicit validity channel; a frame with no covering candidate
             gets zero features and ``validity == 0`` (never silently filled)

Nothing in this module reads the Y target.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn  # noqa: F401  (kept for parity with model.py imports)

from .model import (
    FTNet,
    FTNetConfig,
    FTNetOutput,
    _projection,
    _require_finite_in_valid_region,
    _validate_bool_mask,
)

SEMANTIC_MODES = ("off", "score", "reason", "score_reason")
REASON_VECTOR_DIM = 128
REASON_NGRAM_MIN = 2
REASON_NGRAM_MAX = 3
REASON_HASH_SEED = 20260928
RETRIEVAL_REASON_KEY = "reason"


class SemanticConditioningError(ValueError):
    """Raised when semantic inputs violate the frozen contract."""


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def encode_reason(text: str, *, dim: int = REASON_VECTOR_DIM, normalize: bool = True) -> np.ndarray:
    """Deterministic offline hashed character n-gram bag-of-words vector.

    No tokenizer model, no network call, no external service.  The same text
    always maps to the same vector on any machine.
    """

    vector = np.zeros(dim, dtype=np.float64)
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return vector.astype(np.float32)
    salient = cleaned if len(cleaned) <= 400 else cleaned[:400]
    for n in range(REASON_NGRAM_MIN, REASON_NGRAM_MAX + 1):
        if len(salient) < n:
            continue
        for start in range(0, len(salient) - n + 1):
            token = f"{n}:{salient[start:start + n]}"
            index = _stable_hash(token) % dim
            vector[index] += 1.0
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
    return vector.astype(np.float32)


def primary_candidate_index(timestamps: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Deterministic single-candidate selection mirroring provider_core._context_indices.

    Order: highest score, then earliest start, then longest span, then index.
    Returns -1 for frames outside every candidate.
    """

    index = np.full(timestamps.shape[0], -1, dtype=np.int64)
    if not candidates:
        return index
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -float(candidates[i]["score"]),
            float(candidates[i]["start_sec"]),
            -(float(candidates[i]["end_sec"]) - float(candidates[i]["start_sec"])),
            i,
        ),
    )
    for candidate_index in order:
        candidate = candidates[candidate_index]
        mask = (
            (timestamps >= float(candidate["start_sec"]))
            & (timestamps < float(candidate["end_sec"]))
            & (index < 0)
        )
        index[mask] = candidate_index
    return index


@dataclass
class FrameSemantics:
    score: np.ndarray            # [T] in [0,1] (0 where missing)
    score_valid: np.ndarray      # [T] bool
    reason: np.ndarray           # [T, REASON_VECTOR_DIM] float32 (0 where missing)
    reason_valid: np.ndarray     # [T] bool
    candidate_index: np.ndarray  # [T] int64 (-1 where missing)
    stats: dict = field(default_factory=dict)


def build_frame_semantics(
    timestamps: np.ndarray,
    candidates: list[dict],
    *,
    reason_vector_dim: int = REASON_VECTOR_DIM,
    text_encoder=None,
) -> FrameSemantics:
    """Align per-candidate Qwen score/reason onto the frame timeline (deterministic).

    ``text_encoder`` is an optional injected callable ``text -> np.ndarray`` used to
    represent a reason string.  When omitted, the deterministic offline hashed
    character n-gram encoder is used (backward compatible).  A frozen sentence
    encoder can be injected for the formal protocol; the alignment rule, missing
    policy and validity channels are identical either way.
    """

    if text_encoder is None:
        def text_encoder(text: str) -> np.ndarray:
            return encode_reason(text, dim=reason_vector_dim)

    timestamps = np.asarray(timestamps, dtype=np.float64)
    frame_count = int(timestamps.shape[0])
    score = np.zeros(frame_count, dtype=np.float64)
    score_valid = np.zeros(frame_count, dtype=bool)
    reason = np.zeros((frame_count, reason_vector_dim), dtype=np.float32)
    reason_valid = np.zeros(frame_count, dtype=bool)
    candidate_index = primary_candidate_index(timestamps, candidates)

    reason_cache: dict[int, np.ndarray] = {}
    for frame_index in range(frame_count):
        candidate_id = int(candidate_index[frame_index])
        if candidate_id < 0:
            continue
        candidate = candidates[candidate_id]
        value = float(candidate.get("score", 0.0))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SemanticConditioningError(f"candidate score out of [0,1]: {value!r}")
        score[frame_index] = value
        score_valid[frame_index] = True
        text = str(candidate.get(RETRIEVAL_REASON_KEY, ""))
        if text.strip():
            if candidate_id not in reason_cache:
                vector = np.asarray(text_encoder(text), dtype=np.float32)
                if vector.shape != (reason_vector_dim,):
                    raise SemanticConditioningError(
                        f"text encoder returned {vector.shape}, expected ({reason_vector_dim},)"
                    )
                reason_cache[candidate_id] = vector
            reason[frame_index] = reason_cache[candidate_id]
            reason_valid[frame_index] = True

    stats = {
        "frames": frame_count,
        "candidates": len(candidates),
        "score_valid_frames": int(score_valid.sum()),
        "reason_valid_frames": int(reason_valid.sum()),
        "score_valid_fraction": float(score_valid.mean()) if frame_count else 0.0,
        "reason_valid_fraction": float(reason_valid.mean()) if frame_count else 0.0,
    }
    return FrameSemantics(
        score=score.astype(np.float32),
        score_valid=score_valid,
        reason=reason,
        reason_valid=reason_valid,
        candidate_index=candidate_index,
        stats=stats,
    )


def shuffle_semantics_within_split(
    semantics: dict[str, FrameSemantics],
    *,
    seed: int,
    split_salt: str,
) -> dict[str, FrameSemantics]:
    """Deterministic negative control: permute score/reason field VALUES across videos.

    The permutation is confined to one split (the caller passes one split's map),
    so no cross-split information can move.  Video ids are sorted before the
    permutation, and the RNG is derived from (seed, split_salt) only.
    """

    video_ids = sorted(semantics.keys())
    if not video_ids:
        return {}
    records: list[tuple[int, float, np.ndarray, bool, bool]] = []
    for video_id in video_ids:
        item = semantics[video_id]
        for frame_index in range(item.score.shape[0]):
            records.append((
                frame_index, float(item.score[frame_index]),
                item.reason[frame_index].copy(),
                bool(item.score_valid[frame_index]), bool(item.reason_valid[frame_index]),
            ))
    digest = hashlib.blake2b(f"{int(seed)}:{split_salt}".encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big") % (2**63 - 1))
    order = rng.permutation(len(records))

    shuffled: dict[str, FrameSemantics] = {}
    cursor = 0
    for video_id in video_ids:
        item = semantics[video_id]
        frame_count = item.score.shape[0]
        score = np.zeros(frame_count, dtype=np.float32)
        score_valid = np.zeros(frame_count, dtype=bool)
        reason = np.zeros_like(item.reason)
        reason_valid = np.zeros(frame_count, dtype=bool)
        for frame_index in range(frame_count):
            _fi, value, vector, s_ok, r_ok = records[int(order[cursor])]
            cursor += 1
            score[frame_index] = value
            score_valid[frame_index] = s_ok
            reason[frame_index] = vector
            reason_valid[frame_index] = r_ok
        shuffled[video_id] = FrameSemantics(
            score=score, score_valid=score_valid, reason=reason,
            reason_valid=reason_valid, candidate_index=item.candidate_index.copy(),
            stats={**item.stats, "shuffled": True},
        )
    return shuffled


class SemanticConditionedFTNet(FTNet):
    """Frozen visual-only FTNet with an optionally activated semantic slot pair."""

    def __init__(
        self,
        config: FTNetConfig | None = None,
        *,
        semantic_mode: str = "off",
        reason_vector_dim: int = REASON_VECTOR_DIM,
    ) -> None:
        super().__init__(config)
        if semantic_mode not in SEMANTIC_MODES:
            raise SemanticConditioningError(
                f"semantic_mode must be one of {SEMANTIC_MODES}, got {semantic_mode!r}"
            )
        self.delta_branch = None
        self.native_branch = None
        self.semantic_mode = semantic_mode
        self.reason_vector_dim = int(reason_vector_dim)
        # score slot -> 96 (frozen delta width); reason slot -> 32 (frozen native width)
        self.score_projection = _projection(2, self.config.branch_dim)
        self.reason_projection = _projection(self.reason_vector_dim + 1, self.config.native_branch_dim)
        self.capture_semantic_slots = False
        self.last_semantic_slots: dict[str, torch.Tensor] | None = None

    def _semantic_slots(
        self,
        score: torch.Tensor,
        score_valid: torch.Tensor,
        reason: torch.Tensor,
        reason_valid: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project semantic inputs; invalid frames get an EXACT zero slot.

        Missing/fabricated semantics are impossible: a frame without a valid
        candidate is multiplied by zero after projection (no bias leak).
        """

        batch, length = score.shape
        score_slot = torch.zeros(
            batch, length, self.config.branch_dim, device=score.device, dtype=score.dtype
        )
        reason_slot = torch.zeros(
            batch, length, self.config.native_branch_dim, device=score.device, dtype=score.dtype
        )
        if self.semantic_mode in ("score", "score_reason"):
            score_active = (score_valid & sequence_mask).unsqueeze(-1).to(score.dtype)
            score_input = torch.stack([score, score_valid.to(score.dtype)], dim=-1)
            score_slot = self.score_projection(score_input) * score_active
        if self.semantic_mode in ("reason", "score_reason"):
            if reason.shape[-1] != self.reason_vector_dim:
                raise SemanticConditioningError(
                    f"reason vector dim {reason.shape[-1]} != {self.reason_vector_dim}"
                )
            reason_active = (reason_valid & sequence_mask).unsqueeze(-1).to(reason.dtype)
            reason_input = torch.cat([reason, reason_valid.to(reason.dtype).unsqueeze(-1)], dim=-1)
            reason_slot = self.reason_projection(reason_input) * reason_active
        return score_slot, reason_slot

    def forward(
        self,
        visual_features: torch.Tensor,
        *,
        sequence_mask: torch.Tensor,
        adjacency_mask: torch.Tensor,
        native_features: torch.Tensor | None = None,
        semantic_score: torch.Tensor | None = None,
        semantic_score_valid: torch.Tensor | None = None,
        semantic_reason: torch.Tensor | None = None,
        semantic_reason_valid: torch.Tensor | None = None,
    ) -> FTNetOutput:
        if visual_features.ndim != 3:
            raise ValueError("visual_features must have shape [batch, time, channels]")
        batch, length, channels = visual_features.shape
        if length == 0:
            raise ValueError("FTNet requires at least one timestep")
        if channels != self.config.visual_dim:
            raise ValueError(
                f"visual_features last dimension must be {self.config.visual_dim}"
            )
        _validate_bool_mask(sequence_mask, (batch, length), "sequence_mask")
        _validate_bool_mask(adjacency_mask, (batch, length), "adjacency_mask")
        _require_finite_in_valid_region(visual_features, sequence_mask, "visual_features")

        mask = sequence_mask.unsqueeze(-1)
        sanitized_visual = visual_features.masked_fill(~mask, 0.0)
        visual_projected = self.visual_branch(sanitized_visual).masked_fill(~mask, 0.0)

        if self.semantic_mode == "off":
            score_slot = torch.zeros(
                batch, length, self.config.branch_dim, device=visual_features.device,
                dtype=visual_features.dtype,
            )
            reason_slot = torch.zeros(
                batch, length, self.config.native_branch_dim, device=visual_features.device,
                dtype=visual_features.dtype,
            )
        else:
            needs_score = self.semantic_mode in ("score", "score_reason")
            needs_reason = self.semantic_mode in ("reason", "score_reason")
            if (needs_score and semantic_score is None) or (needs_reason and semantic_reason is None):
                raise SemanticConditioningError(
                    f"semantic_mode={self.semantic_mode!r} requires the matching semantic tensors"
                )
            if needs_score and semantic_score.shape != (batch, length):
                raise SemanticConditioningError("semantic_score must align with [batch, time]")
            if needs_reason and semantic_reason.shape[:2] != (batch, length):
                raise SemanticConditioningError("semantic_reason must align with [batch, time]")
            score_valid = (
                semantic_score_valid
                if semantic_score_valid is not None
                else torch.ones(batch, length, dtype=torch.bool, device=visual_features.device)
            )
            reason_valid = (
                semantic_reason_valid
                if semantic_reason_valid is not None
                else torch.ones(batch, length, dtype=torch.bool, device=visual_features.device)
            )
            score_tensor = (
                semantic_score.to(visual_features.dtype)
                if semantic_score is not None
                else torch.zeros(batch, length, dtype=visual_features.dtype, device=visual_features.device)
            )
            reason_tensor = (
                semantic_reason.to(visual_features.dtype)
                if semantic_reason is not None
                else torch.zeros(
                    batch, length, self.reason_vector_dim,
                    dtype=visual_features.dtype, device=visual_features.device,
                )
            )
            score_slot, reason_slot = self._semantic_slots(
                score_tensor, score_valid, reason_tensor, reason_valid, sequence_mask
            )

        if self.capture_semantic_slots:
            self.last_semantic_slots = {"score": score_slot.detach(), "reason": reason_slot.detach()}

        concatenated = torch.cat([visual_projected, score_slot, reason_slot], dim=-1)
        temporal = self.fusion(concatenated).masked_fill(~mask, 0.0)
        for block in self.temporal_blocks:
            temporal = block(temporal, sequence_mask)
        logits = self.y_head(temporal).squeeze(-1).masked_fill(~sequence_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~sequence_mask, 0.0)
        return FTNetOutput(logits=logits, probabilities=probabilities, features=temporal)


@dataclass(frozen=True)
class SemanticFTNetExample:
    visual_features: torch.Tensor
    targets: torch.Tensor
    adjacency_mask: torch.Tensor
    loss_mask: torch.Tensor
    native_features: torch.Tensor | None = None
    semantic_score: torch.Tensor | None = None
    semantic_score_valid: torch.Tensor | None = None
    semantic_reason: torch.Tensor | None = None
    semantic_reason_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class SemanticFTNetBatch:
    visual_features: torch.Tensor
    targets: torch.Tensor
    sequence_mask: torch.Tensor
    adjacency_mask: torch.Tensor
    loss_mask: torch.Tensor
    native_features: torch.Tensor | None
    semantic_score: torch.Tensor | None
    semantic_score_valid: torch.Tensor | None
    semantic_reason: torch.Tensor | None
    semantic_reason_valid: torch.Tensor | None

    def to(self, device: str | torch.device) -> "SemanticFTNetBatch":
        def _move(value):
            return value.to(device) if value is not None else None

        return SemanticFTNetBatch(
            visual_features=_move(self.visual_features),
            targets=_move(self.targets),
            sequence_mask=_move(self.sequence_mask),
            adjacency_mask=_move(self.adjacency_mask),
            loss_mask=_move(self.loss_mask),
            native_features=_move(self.native_features),
            semantic_score=_move(self.semantic_score),
            semantic_score_valid=_move(self.semantic_score_valid),
            semantic_reason=_move(self.semantic_reason),
            semantic_reason_valid=_move(self.semantic_reason_valid),
        )


def collate_semantic_examples(examples: list[SemanticFTNetExample]) -> SemanticFTNetBatch:
    """Pad variable-length examples; semantic tensors default to zeros+invalid (explicit)."""

    if not examples:
        raise SemanticConditioningError("at least one example is required")
    lengths = [int(example.visual_features.shape[0]) for example in examples]
    max_length = max(lengths)
    batch_size = len(examples)
    visual_dim = int(examples[0].visual_features.shape[1])
    reason_dim = 0
    for example in examples:
        if example.semantic_reason is not None:
            reason_dim = int(example.semantic_reason.shape[1])
            break

    visual = examples[0].visual_features.new_zeros((batch_size, max_length, visual_dim))
    targets = examples[0].targets.new_zeros((batch_size, max_length))
    sequence = torch.zeros(batch_size, max_length, dtype=torch.bool)
    adjacency = torch.zeros(batch_size, max_length, dtype=torch.bool)
    loss = torch.zeros(batch_size, max_length, dtype=torch.bool)
    score = torch.zeros(batch_size, max_length)
    score_valid = torch.zeros(batch_size, max_length, dtype=torch.bool)
    reason = torch.zeros(batch_size, max_length, reason_dim)
    reason_valid = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for index, (example, length) in enumerate(zip(examples, lengths)):
        visual[index, :length] = example.visual_features
        targets[index, :length] = example.targets
        sequence[index, :length] = True
        adjacency[index, :length] = example.adjacency_mask
        adjacency[index, 0] = False
        if example.loss_mask is not None:
            loss[index, :length] = example.loss_mask
        else:
            loss[index, :length] = True
        if reason_dim and example.semantic_reason is not None:
            reason[index, :length] = example.semantic_reason
        if example.semantic_score is not None:
            score[index, :length] = example.semantic_score
        if example.semantic_score_valid is not None:
            score_valid[index, :length] = example.semantic_score_valid
        if example.semantic_reason_valid is not None:
            reason_valid[index, :length] = example.semantic_reason_valid
    return SemanticFTNetBatch(
        visual_features=visual,
        targets=targets,
        sequence_mask=sequence,
        adjacency_mask=adjacency,
        loss_mask=loss & sequence,
        native_features=None,
        semantic_score=score,
        semantic_score_valid=score_valid,
        semantic_reason=reason if reason_dim else None,
        semantic_reason_valid=reason_valid if reason_dim else None,
    )


__all__ = [
    "SEMANTIC_MODES",
    "REASON_VECTOR_DIM",
    "REASON_HASH_SEED",
    "SemanticConditioningError",
    "SemanticConditionedFTNet",
    "SemanticFTNetExample",
    "SemanticFTNetBatch",
    "FrameSemantics",
    "build_frame_semantics",
    "collate_semantic_examples",
    "encode_reason",
    "primary_candidate_index",
    "shuffle_semantics_within_split",
]
