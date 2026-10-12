"""PHD2 generic pretrain target adapter (PHD2_GENERIC_PRETRAIN_TARGET_V1).

Weak-supervision pretraining targets from PHD2 per-user highlight selections.

Frozen semantics (protocol ``PHD2_GENERIC_PRETRAIN_TARGET_V1``):

* selection intervals are half-open ``[start, start + duration)`` seconds;
* TRAIN_ONLY annotations; TEST is never read by this module;
* a timestep whose timestamp falls in at least one TRAIN selection is POSITIVE;
* a timestep covered by no selection is UNLABELED (never a negative);
* multi-user aggregation is the UNION of intervals; ``selection_support_count``
  (distinct covering users) is audit metadata only and never becomes a soft Y;
* ``BOUNDARY_GUARD = 0.5 s``: unlabeled timesteps in the +/- 0.5 s neighbourhood
  outside any interval boundary are excluded from pairwise comparison sampling;
* supervision is positive-unlabeled pairwise ranking
  ``softplus(-(z_pos - z_u))`` with a deterministic 1:1 per-video sampler.

This module never fabricates labels, candidates or artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_VERSION = "PHD2_GENERIC_PRETRAIN_TARGET_V1"
TRAIN_ANNOTATION_SHA256 = "f89f7993b28abb29549bbdb81330e4a60a2750d65e2f3cc26d78ab4ab7be6f3d"
BOUNDARY_GUARD_SEC = 0.5
PAIR_SEED_DEFAULT = 20260924


class Phd2TargetError(RuntimeError):
    """Raised when the frozen target contract cannot be honoured."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Selection:
    video_id: str
    start_sec: float
    end_sec: float
    user_id: str


@dataclass(frozen=True)
class AnnotationIndex:
    """video_id -> TRAIN selection intervals (union semantics)."""

    selections: Mapping[str, tuple[Selection, ...]]
    source_path: str
    source_sha256: str
    row_count: int
    unique_videos: int

    def intervals_for(self, video_id: str) -> tuple[Selection, ...]:
        return self.selections.get(video_id, ())


def load_train_annotations(
    path: str | Path,
    *,
    expected_sha256: str | None = TRAIN_ANNOTATION_SHA256,
    require_sha: bool = True,
) -> AnnotationIndex:
    """Read the TRAIN-only annotation CSV (never reads TEST)."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise Phd2TargetError(f"train annotation not found: {csv_path}")
    digest = sha256_file(csv_path)
    if require_sha and expected_sha256 and digest != expected_sha256:
        raise Phd2TargetError(
            f"train annotation SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    table: dict[str, list[Selection]] = {}
    rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"youtubeId", "start", "duration", "user_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise Phd2TargetError(f"annotation CSV lacks columns: {sorted(missing)}")
        for row in reader:
            video_id = str(row.get("youtubeId", "")).strip()
            if not video_id:
                continue
            try:
                start = float(row["start"])
                duration = float(row["duration"])
            except (TypeError, ValueError) as exc:
                raise Phd2TargetError(f"non-numeric start/duration for {video_id}") from exc
            if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                raise Phd2TargetError(f"invalid selection for {video_id}: {start}+{duration}")
            table.setdefault(video_id, []).append(
                Selection(
                    video_id=video_id,
                    start_sec=start,
                    end_sec=start + duration,
                    user_id=str(row.get("user_id", "")).strip(),
                )
            )
            rows += 1
    ordered = {
        video_id: tuple(sorted(items, key=lambda item: (item.start_sec, item.end_sec, item.user_id)))
        for video_id, items in table.items()
    }
    return AnnotationIndex(
        selections=ordered,
        source_path=str(csv_path.resolve()),
        source_sha256=digest,
        row_count=rows,
        unique_videos=len(ordered),
    )


def timestep_masks(
    timestamps: Sequence[float],
    selections: Sequence[Selection],
    *,
    boundary_guard_sec: float = BOUNDARY_GUARD_SEC,
) -> dict[str, list[Any]]:
    """Compute per-timestep positive / unlabeled / boundary-guard / support.

    ``legal_unlabeled`` = unlabeled AND not boundary-guard: the only timesteps
    allowed as pairwise comparison samples.
    """

    if boundary_guard_sec < 0:
        raise Phd2TargetError("boundary_guard_sec must be >= 0")
    positives: list[bool] = []
    unlabeled: list[bool] = []
    guard: list[bool] = []
    support: list[int] = []
    for value in timestamps:
        t = float(value)
        if not math.isfinite(t):
            raise Phd2TargetError("non-finite timestamp in timeline")
        covering = [item for item in selections if item.start_sec <= t < item.end_sec]
        is_positive = bool(covering)
        support_count = len({item.user_id for item in covering})
        if is_positive:
            near_boundary = False
        else:
            near_boundary = any(
                (item.start_sec - boundary_guard_sec <= t < item.start_sec)
                or (item.end_sec <= t < item.end_sec + boundary_guard_sec)
                for item in selections
            )
        positives.append(is_positive)
        unlabeled.append(not is_positive)
        guard.append(bool(near_boundary))
        support.append(int(support_count))
    legal = [u and not g for u, g in zip(unlabeled, guard)]
    return {
        "positive_mask": positives,
        "unlabeled_mask": unlabeled,
        "boundary_guard_mask": guard,
        "selection_support_count": support,
        "legal_unlabeled_mask": legal,
    }


def sample_pairs(
    timestamps: Sequence[float],
    masks: Mapping[str, Sequence[Any]],
    *,
    video_id: str,
    adjacency_mask: Sequence[bool] | None = None,
    seed: int = PAIR_SEED_DEFAULT,
    positive_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic 1:1 positive:legal-unlabeled pair ledger for one video.

    Each legal unlabeled timestep is used at most once; if a video has no legal
    unlabeled timestep, no pair is produced (never a fabricated negative).
    """

    n = len(timestamps)
    if n == 0:
        return []
    positive = list(masks["positive_mask"])
    legal = list(masks["legal_unlabeled_mask"])
    if adjacency_mask is not None and len(adjacency_mask) != n:
        raise Phd2TargetError("adjacency_mask length does not match timeline")
    positives = sorted(
        int(i)
        for i in (positive_indices if positive_indices is not None else range(n))
        if positive[i]
    )
    candidates = [
        i
        for i in range(n)
        if legal[i] and (adjacency_mask is None or bool(adjacency_mask[i]))
    ]
    if not positives or not candidates:
        return []
    rng = random.Random(f"{seed}:{video_id}")
    pool = list(candidates)
    pairs: list[dict[str, Any]] = []
    for pos_index in positives:
        if not pool:
            break
        choice = pool.pop(rng.randrange(len(pool)))
        pairs.append(
            {
                "video_id": video_id,
                "positive_index": int(pos_index),
                "positive_timestamp": float(timestamps[pos_index]),
                "unlabeled_index": int(choice),
                "unlabeled_timestamp": float(timestamps[choice]),
                "seed": int(seed),
            }
        )
    return pairs


def pairwise_ranking_loss(z_pos: Any, z_u: Any) -> Any:
    """``softplus(-(z_pos - z_u))`` = log(1 + exp(-(z_pos - z_u)))."""

    import torch

    return torch.nn.functional.softplus(-(z_pos - z_u)).mean()


def pair_audit(pairs: Sequence[Mapping[str, Any]], *, n: int) -> dict[str, Any]:
    pos = {int(p["positive_index"]) for p in pairs}
    unl = {int(p["unlabeled_index"]) for p in pairs}
    return {
        "pairs": len(pairs),
        "unique_positive": len(pos),
        "unique_unlabeled": len(unl),
        "one_to_one": len(pos) == len(unl) == len(pairs),
        "cross_video": any(str(p["video_id"]) != str(pairs[0]["video_id"]) for p in pairs),
        "index_in_range": all(0 <= int(p["positive_index"]) < n and 0 <= int(p["unlabeled_index"]) < n for p in pairs),
    }


__all__ = [
    "AnnotationIndex",
    "BOUNDARY_GUARD_SEC",
    "PAIR_SEED_DEFAULT",
    "PROTOCOL_VERSION",
    "Phd2TargetError",
    "Selection",
    "TRAIN_ANNOTATION_SHA256",
    "load_train_annotations",
    "pair_audit",
    "pairwise_ranking_loss",
    "sample_pairs",
    "sha256_file",
    "timestep_masks",
]
