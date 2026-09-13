"""RT-DETR adapter for single-frame subject localization (offline, pretrained)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .subject_localization import SubjectCandidate, SubjectPolicyConfig, SubjectDecision, select_primary_subject

DEFAULT_MODEL_ID = "PekingU/rtdetr_r50vd"


class RTDetrLocalizer:
    """Zero-shot COCO RT-DETR localizer; deterministic single-frame inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        local_path: str | Path | None = None,
        device: str | None = None,
        torch_dtype: str = "float32",
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self.torch = torch
        self.model_id = model_id
        source = str(local_path) if local_path is not None else model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = {"float32": torch.float32, "float16": torch.float16}[torch_dtype]
        self.processor = AutoImageProcessor.from_pretrained(
            source,
            do_resize=True,
            size={"height": 640, "width": 640},
        )
        self.model = RTDetrForObjectDetection.from_pretrained(
            source,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.id2label = dict(self.model.config.id2label)

    def _decode(self, outputs, width: int, height: int, top_k: int) -> tuple[SubjectCandidate, ...]:
        logits = outputs.logits[0]
        boxes = outputs.pred_boxes[0]
        scores_all = logits.sigmoid()
        best_scores, best_labels = scores_all.max(dim=-1)
        k = min(top_k, best_scores.shape[0])
        top_scores, top_indices = best_scores.topk(k)
        candidates: list[SubjectCandidate] = []
        for rank in range(k):
            query = int(top_indices[rank].item())
            score = float(top_scores[rank].item())
            label_id = int(best_labels[query].item())
            cx, cy, bw, bh = (float(v) for v in boxes[query].tolist())
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            candidates.append(
                SubjectCandidate(
                    box=(x1, y1, x2, y2),
                    score=score,
                    label_id=label_id,
                    label=str(self.id2label.get(label_id, label_id)),
                )
            )
        return tuple(candidates)

    def localize_frames(
        self,
        video_id: str,
        frames: Iterable[tuple[int, np.ndarray]],
        config: SubjectPolicyConfig,
        provenance: dict[str, str] | None = None,
    ) -> list[SubjectDecision]:
        """frames: iterable of (frame_id, rgb_uint8_array)."""
        decisions: list[SubjectDecision] = []
        with self.torch.inference_mode():
            for frame_id, image in frames:
                height, width = image.shape[0], image.shape[1]
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self.torch_dtype)
                outputs = self.model(pixel_values=pixel_values)
                raw = self._decode(outputs, width, height, top_k=100)
                decisions.append(
                    select_primary_subject(
                        video_id=video_id,
                        frame=frame_id,
                        image_width=width,
                        image_height=height,
                        candidates=raw,
                        config=config,
                        provenance=provenance,
                    )
                )
        return decisions
