"""Deterministic, group-safe split protocol for FTNet training data."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass


PROTOCOL_NAME = "VHiCraFTNet-Stage7.1"
PROTOCOL_VERSION = "1.0"
SPLIT_SEED = 20260917
SPLIT_RATIOS = (70, 15, 15)
SPLIT_ROLES = ("TRAIN", "VALIDATION", "CALIBRATION")
OFFICIAL_TRAIN = "train"
SOURCE_GROUP_NAMESPACE = "youtube_highlights|canonical_source_group"


class SplitProtocolError(RuntimeError):
    """Raised when input identities violate the frozen split protocol."""


@dataclass(frozen=True)
class SplitCandidate:
    canonical_video_id: str
    realized_video_id: str
    category: str
    upstream_official_split: str
    alignment_pass: bool
    local_video_path: str
    source_file_sha256: str
    annotation_identity: str
    original_video_id: str | None = None
    replacement_video_id: str | None = None
    upstream_official_set: str | None = None


@dataclass(frozen=True)
class AssignedSplit:
    candidate: SplitCandidate
    stage7_split: str
    source_group_key: str
    split_hash_key: str
    source_aliases: tuple[str, ...]
    replacement_original_id: str | None


@dataclass(frozen=True)
class SplitAudit:
    total: int
    counts: dict[str, int]
    category_counts: dict[str, dict[str, int]]
    source_id_leakage: int
    official_test_rows: int
    tvsum_rows: int
    non_alignment_pass_rows: int


def _split_counts(total: int) -> tuple[int, int, int]:
    counts = [total * ratio // sum(SPLIT_RATIOS) for ratio in SPLIT_RATIOS]
    for index in range(total - sum(counts)):
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _split_hash(canonical_video_id: str, category: str, seed: int) -> str:
    value = f"{PROTOCOL_NAME}|{seed}|{category}|{canonical_video_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_group_key(canonical_video_id: str) -> str:
    value = f"{SOURCE_GROUP_NAMESPACE}|{canonical_video_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _aliases(candidate: SplitCandidate) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for value in (
                    candidate.canonical_video_id,
                    candidate.realized_video_id,
                    candidate.original_video_id,
                    candidate.replacement_video_id,
                )
                if value
            }
        )
    )


def assign_stage7_splits(
    candidates: list[SplitCandidate],
    *,
    seed: int = SPLIT_SEED,
) -> tuple[AssignedSplit, ...]:
    """Assign candidates by canonical source group inside each category."""

    if not candidates:
        raise SplitProtocolError("no candidates were provided for a split")

    alias_owner: dict[str, str] = {}
    groups: dict[str, list[SplitCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.upstream_official_split.strip().lower() != OFFICIAL_TRAIN:
            raise SplitProtocolError(
                "only official train candidates are allowed: "
                f"{candidate.canonical_video_id} is "
                f"{candidate.upstream_official_split}"
            )
        if not candidate.alignment_pass:
            raise SplitProtocolError(
                f"candidate {candidate.canonical_video_id} lacks ALIGNMENT_PASS"
            )
        for alias in _aliases(candidate):
            owner = alias_owner.setdefault(alias, candidate.canonical_video_id)
            if owner != candidate.canonical_video_id:
                raise SplitProtocolError(
                    f"source alias {alias} belongs to both {owner} and "
                    f"{candidate.canonical_video_id}"
                )
        groups[candidate.canonical_video_id].append(candidate)

    groups_by_category: dict[str, list[str]] = defaultdict(list)
    for canonical_video_id in sorted(groups):
        members = groups[canonical_video_id]
        categories = {member.category for member in members}
        if len(categories) != 1:
            raise SplitProtocolError(
                f"canonical source {canonical_video_id} spans categories "
                f"{sorted(categories)}"
            )
        groups_by_category[categories.pop()].append(canonical_video_id)

    assigned: list[AssignedSplit] = []
    for category in sorted(groups_by_category):
        canonicals = sorted(
            groups_by_category[category],
            key=lambda canonical_video_id: (
                _split_hash(canonical_video_id, category, seed),
                canonical_video_id,
            ),
        )
        train_count, validation_count, calibration_count = _split_counts(
            len(canonicals)
        )
        roles = (
            ["TRAIN"] * train_count
            + ["VALIDATION"] * validation_count
            + ["CALIBRATION"] * calibration_count
        )
        for canonical_video_id, role in zip(canonicals, roles):
            members = sorted(
                groups[canonical_video_id],
                key=lambda member: (member.realized_video_id, member.category),
            )
            for member in members:
                assigned.append(
                    AssignedSplit(
                        candidate=member,
                        stage7_split=role,
                        source_group_key=_source_group_key(
                            canonical_video_id
                        ),
                        split_hash_key=_split_hash(
                            canonical_video_id, category, seed
                        ),
                        source_aliases=_aliases(member),
                        replacement_original_id=(
                            canonical_video_id
                            if member.realized_video_id != canonical_video_id
                            else None
                        ),
                    )
                )
    return tuple(assigned)


def audit_assignment(assigned: tuple[AssignedSplit, ...]) -> SplitAudit:
    """Recompute leakage and membership invariants over an assignment."""

    alias_to_splits: dict[str, set[str]] = defaultdict(set)
    for entry in assigned:
        for alias in entry.source_aliases:
            alias_to_splits[alias].add(entry.stage7_split)
    source_id_leakage = sum(
        1 for splits in alias_to_splits.values() if len(splits) > 1
    )

    counts = {role: 0 for role in SPLIT_ROLES}
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {role: 0 for role in SPLIT_ROLES}
    )
    official_test_rows = 0
    tvsum_rows = 0
    non_alignment_pass_rows = 0
    for entry in assigned:
        candidate = entry.candidate
        counts[entry.stage7_split] += 1
        category_counts[candidate.category][entry.stage7_split] += 1
        if candidate.upstream_official_split.strip().lower() != OFFICIAL_TRAIN:
            official_test_rows += 1
        if candidate.upstream_official_split.strip().lower() == "tvsum":
            tvsum_rows += 1
        if not candidate.alignment_pass:
            non_alignment_pass_rows += 1

    return SplitAudit(
        total=len(assigned),
        counts=counts,
        category_counts={
            category: category_counts[category]
            for category in sorted(category_counts)
        },
        source_id_leakage=source_id_leakage,
        official_test_rows=official_test_rows,
        tvsum_rows=tvsum_rows,
        non_alignment_pass_rows=non_alignment_pass_rows,
    )
