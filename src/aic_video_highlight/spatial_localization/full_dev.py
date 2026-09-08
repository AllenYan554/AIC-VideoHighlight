"""Resumable full-dev spatial inference: shard bookkeeping and merge gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = "raw_detector"
POLICY_DIR = "policy_v1"
CROP_DIR = "diagnostic_crop"
STATUS_DIR = "status"


class ShardStateError(RuntimeError):
    """Raised when a shard's recorded state is inconsistent with its artifacts."""


class FrameIdentityError(RuntimeError):
    """Raised when produced frame keys diverge from the Stage 5.1 frozen set."""


def canonical_sha(records) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_paths(output_dir: Path, video_id: str) -> dict[str, Path]:
    return {
        "raw": output_dir / RAW_DIR / f"{video_id}.jsonl",
        "policy": output_dir / POLICY_DIR / f"{video_id}.jsonl",
        "crop": output_dir / CROP_DIR / f"{video_id}.jsonl",
        "status": output_dir / STATUS_DIR / f"{video_id}.json",
    }


@dataclass(frozen=True, slots=True)
class ShardStatus:
    video_id: str
    frame_count: int
    raw_sha256: str
    policy_sha256: str
    crop_sha256: str


def write_shard(
    output_dir: Path,
    video_id: str,
    raw_records: list[dict],
    policy_records: list[dict],
    crop_records: list[dict],
    extra_status: dict | None = None,
) -> ShardStatus:
    paths = shard_paths(output_dir, video_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(paths["raw"], raw_records)
    _write_jsonl_atomic(paths["policy"], policy_records)
    _write_jsonl_atomic(paths["crop"], crop_records)
    status = ShardStatus(
        video_id=video_id,
        frame_count=len(policy_records),
        raw_sha256=file_sha256(paths["raw"]),
        policy_sha256=file_sha256(paths["policy"]),
        crop_sha256=file_sha256(paths["crop"]),
    )
    payload = {
        "video_id": video_id,
        "frame_count": status.frame_count,
        "raw_sha256": status.raw_sha256,
        "policy_sha256": status.policy_sha256,
        "crop_sha256": status.crop_sha256,
        "extra": dict(extra_status or {}),
    }
    _write_jsonl_atomic(paths["status"], [payload])
    return status


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def shard_is_complete(output_dir: Path, video_id: str, expected_frames: list[int]) -> bool:
    """A shard is reusable only if artifacts exist, parse, and match recorded hashes/counts."""
    paths = shard_paths(output_dir, video_id)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        status_records = [json.loads(line) for line in paths["status"].read_text(encoding="utf-8").splitlines() if line.strip()]
        status = status_records[0]
        if status.get("frame_count") != len(expected_frames) or status.get("video_id") != video_id:
            return False
        if status["raw_sha256"] != file_sha256(paths["raw"]):
            return False
        if status["policy_sha256"] != file_sha256(paths["policy"]):
            return False
        if status["crop_sha256"] != file_sha256(paths["crop"]):
            return False
        policy_keys = set()
        for line in paths["policy"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                policy_keys.add((record["video_id"], record["frame"]))
        if policy_keys != {(video_id, frame) for frame in expected_frames}:
            return False
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    return True


def load_shard_records(output_dir: Path, video_id: str) -> tuple[list[dict], list[dict], list[dict]]:
    paths = shard_paths(output_dir, video_id)

    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return read(paths["raw"]), read(paths["policy"]), read(paths["crop"])


def merge_and_verify(
    output_dir: Path,
    expected_keys: set[tuple[str, int]],
) -> dict:
    """Merge all shards and enforce the frame identity gate."""
    merged_raw: list[dict] = []
    merged_policy: list[dict] = []
    merged_crop: list[dict] = []
    produced: set[tuple[str, int]] = set()
    duplicate_keys: set[tuple[str, int]] = set()
    video_ids = sorted({path.stem for path in (output_dir / POLICY_DIR).glob("*.jsonl")})
    for video_id in video_ids:
        raw_records, policy_records, crop_records = load_shard_records(output_dir, video_id)
        merged_raw.extend(raw_records)
        merged_crop.extend(crop_records)
        for record in policy_records:
            key = (record["video_id"], record["frame"])
            if key in produced:
                duplicate_keys.add(key)
            produced.add(key)
            merged_policy.append(record)
    missing = expected_keys - produced
    extra = produced - expected_keys
    if missing or extra or duplicate_keys:
        raise FrameIdentityError(
            f"frame identity gate failed: missing={len(missing)} extra={len(extra)} duplicate={len(duplicate_keys)}"
        )
    return {
        "video_count": len(video_ids),
        "frame_count": len(produced),
        "missing_keys": len(missing),
        "extra_keys": len(extra),
        "duplicate_keys": len(duplicate_keys),
        "semantic_hashes": {
            "raw_candidates": canonical_sha(merged_raw),
            "policy_v1_decisions": canonical_sha(merged_policy),
            "crop_diagnostics": canonical_sha(merged_crop),
        },
        "merged": {
            "raw": merged_raw,
            "policy": merged_policy,
            "crop": merged_crop,
        },
    }


def pick_even(values: list, count: int) -> list:
    if len(values) <= count:
        return list(values)
    return [values[i * len(values) // count] for i in range(count)]
