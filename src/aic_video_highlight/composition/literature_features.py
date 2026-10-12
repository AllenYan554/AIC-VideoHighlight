"""AIC-side adapter: 1024-D GoogleNet pool5 features for literature selectors.

PGL-SUM and VASNet do **not** ship a feature extractor.  Both were trained on
pre-computed ``eccv16_dataset_{summe,tvsum}_google_pool5.h5`` tensors: one
1024-D GoogleNet pool5 vector per sub-sampled frame, with ``picks`` recording
the original frame index of each sub-sampled frame.  To run either pretrained
model on AIC videos we must reproduce that feature space.

This module is therefore an AIC adapter, not author code.  It is deliberately
thin and lazy: torchvision is imported only when features are actually
requested, so the rest of the package stays importable in CPU-only
environments.  It is a compatible reproduction of the published 1024-D
GoogleNet pool5 feature contract; bitwise identity with the unavailable
original precomputed Caffe features is not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

FEATURE_DIM = 1024


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    frames: tuple[int, ...]
    features: np.ndarray  # [T, 1024]


class GoogleNetPool5Extractor:
    """Extract 1024-D ImageNet GoogleNet pool5 features for given frames.

    The network head (``fc``) is discarded: the returned vector is the flattened
    pre-classifier 1024-D activation, matching the Caffe ``pool5`` blob the
    SumMe/TVSum h5 files were built from.
    """

    def __init__(self, *, device: str = "cpu", batch_size: int = 64) -> None:
        try:
            import torch
            from torchvision.models import GoogLeNet_Weights, googlenet
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "GoogleNet pool5 feature extraction requires torchvision; "
                "install torchvision in the cloud environment before the Formal run"
            ) from exc

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.torch = torch
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.weights = weights
        model = googlenet(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)
        self.preprocess = weights.transforms()

    def extract(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Return a ``[T, 1024]`` array for a sequence of BGR uint8 frames."""
        if not frames_bgr:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        torch = self.torch
        chunks: list[np.ndarray] = []
        for start in range(0, len(frames_bgr), self.batch_size):
            tensors = []
            for frame in frames_bgr[start:start + self.batch_size]:
                rgb = np.asarray(frame)[:, :, ::-1].copy()
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
                tensors.append(self.preprocess(tensor))
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model(batch)
            chunks.append(
                feats.detach().cpu().numpy().astype(np.float32).reshape(len(tensors), FEATURE_DIM)
            )
        return np.concatenate(chunks, axis=0)


def read_frames_bgr(video_path: str | Path, frames: Sequence[int]) -> list[np.ndarray]:
    """Read exactly the requested frame indices from a decoded video (0-based)."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    wanted = sorted({int(frame) for frame in frames})
    collected: dict[int, np.ndarray] = {}
    try:
        index = 0
        last = wanted[-1] if wanted else -1
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                collected[index] = frame
            index += 1
    finally:
        capture.release()
    missing = [frame for frame in wanted if frame not in collected]
    if missing:
        raise RuntimeError(f"video ended before frame(s) {missing[:5]} of {len(wanted)} requested")
    return [collected[int(frame)] for frame in frames]


def extract_features(
    video_path: str | Path,
    frames: Sequence[int],
    *,
    extractor: GoogleNetPool5Extractor | None = None,
) -> FeatureBatch:
    """Convenience wrapper: decode frames, extract pool5, keep index alignment."""
    ordered = tuple(int(frame) for frame in frames)
    extractor = extractor or GoogleNetPool5Extractor()
    images = read_frames_bgr(video_path, ordered)
    return FeatureBatch(frames=ordered, features=extractor.extract(images))
