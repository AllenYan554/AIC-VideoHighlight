"""Materialized FTNet dataset loader and TRAIN-only normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from .materialize import SPLIT_DIRS
from .native_schema import NormalizationStats, standardize_native


class DatasetError(RuntimeError):
    """Raised when materialized data violates the frozen contract."""


@dataclass(frozen=True)
class MaterializedVideo:
    canonical_video_id: str
    split: str
    visual: torch.Tensor
    native: torch.Tensor
    target: torch.Tensor
    adjacency_mask: torch.Tensor
    loss_mask: torch.Tensor
    timestamps: np.ndarray


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationStats(
        mean=tuple(float(value) for value in payload["mean"]),
        std=tuple(float(value) for value in payload["std"]),
        log1p_fields=tuple(int(value) for value in payload["log1p_fields"]),
    )


class MaterializedFTNetDataset:
    """Load per-video safetensors for one Stage 7 split."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        stats: NormalizationStats,
    ) -> None:
        if split not in SPLIT_DIRS:
            raise DatasetError(f"unknown split: {split}")
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.stats = stats
        self.split_dir = self.data_root / SPLIT_DIRS[split]
        if not self.split_dir.is_dir():
            raise DatasetError(f"split directory is missing: {self.split_dir}")
        self.paths = sorted(self.split_dir.glob("*.safetensors"))
        if not self.paths:
            raise DatasetError(f"no materialized videos in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def video_ids(self) -> list[str]:
        return [path.stem for path in self.paths]

    def load(self, index: int) -> MaterializedVideo:
        path = self.paths[index]
        try:
            tensors = load_file(str(path))
        except Exception as exc:
            raise DatasetError(f"unreadable safetensors: {path}") from exc
        required = {
            "visual", "native", "native_missing", "target",
            "loss_mask", "adjacency_mask", "timestamp",
        }
        missing = required - set(tensors)
        if missing:
            raise DatasetError(f"{path.name} missing keys: {sorted(missing)}")
        visual = torch.from_numpy(np.asarray(tensors["visual"], dtype=np.float32))
        raw_native = torch.from_numpy(np.asarray(tensors["native"], dtype=np.float32))
        missing_mask = torch.from_numpy(
            np.asarray(tensors["native_missing"], dtype=bool)
        )
        native = standardize_native(raw_native, self.stats, missing_mask)
        return MaterializedVideo(
            canonical_video_id=path.stem,
            split=self.split,
            visual=visual,
            native=native,
            target=torch.from_numpy(np.asarray(tensors["target"], dtype=np.float32)),
            adjacency_mask=torch.from_numpy(
                np.asarray(tensors["adjacency_mask"], dtype=bool)
            ),
            loss_mask=torch.from_numpy(np.asarray(tensors["loss_mask"], dtype=bool)),
            timestamps=np.asarray(tensors["timestamp"], dtype=np.float64),
        )


def compute_train_normalization(
    data_root: str | Path,
    *,
    output_path: str | Path,
) -> dict:
    """Compute TRAIN-only per-field mean/std on the log1p space and freeze them."""

    from .native_schema import LOG1P_Z_FIELDS, NATIVE_DIM, NATIVE_FIELDS, compute_normalization_stats

    data_root = Path(data_root).expanduser().resolve()
    train_dir = data_root / SPLIT_DIRS["TRAIN"]
    paths = sorted(train_dir.glob("*.safetensors"))
    if not paths:
        raise DatasetError("no TRAIN videos to compute normalization")
    chunks = []
    for path in paths:
        tensors = load_file(str(path))
        raw = np.asarray(tensors["native"], dtype=np.float32)
        missing = np.asarray(tensors["native_missing"], dtype=bool)
        clean = raw.copy()
        clean[missing] = np.nan
        chunks.append(clean)
    stacked = np.concatenate(chunks, axis=0)
    finite_rows = ~np.isnan(stacked).any(axis=1)
    usable = torch.from_numpy(np.nan_to_num(stacked[finite_rows], nan=0.0))
    stats = compute_normalization_stats(usable)
    payload = stats.as_dict()
    payload.update(
        {
            "train_video_count": len(paths),
            "train_videos": [path.stem for path in paths],
            "fields": list(NATIVE_FIELDS),
            "native_dim": NATIVE_DIM,
            "log1p_fields": list(LOG1P_Z_FIELDS),
            "missing_policy": "standardize_then_zero",
        }
    )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
