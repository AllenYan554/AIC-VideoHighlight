"""Visualization helpers for subject localization diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .subject_localization import SubjectDecision, candidate_is_valid


def _draw_box(draw: ImageDraw.ImageDraw, box, color, width: int = 3, label: str | None = None) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)


def render_decision_frame(
    image_rgb: np.ndarray,
    decision: SubjectDecision,
    reference_box: tuple[float, float, float, float] | None = None,
    center_crop_box: tuple[int, int, int] | None = None,
) -> Image.Image:
    pil = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(pil)
    if center_crop_box is not None:
        x, y, w = center_crop_box
        _draw_box(draw, (x, y, x + w, decision.image_height), (160, 160, 160), width=2, label="center-crop")
    for candidate in decision.candidates[:5]:
        color = (90, 90, 255) if candidate_is_valid(candidate.box) else (255, 140, 0)
        _draw_box(
            draw,
            candidate.box,
            color,
            width=1,
            label=f"{candidate.label} {candidate.score:.2f}",
        )
    if reference_box is not None:
        _draw_box(draw, reference_box, (60, 200, 60), width=3, label="weak-ref")
    if decision.primary is not None:
        _draw_box(
            draw,
            decision.primary.box,
            (255, 60, 60),
            width=3,
            label=f"primary {decision.primary.score:.2f}",
        )
    elif decision.fallback_box is not None:
        x, y, w = decision.fallback_box
        _draw_box(draw, (x, y, x + w, decision.image_height), (255, 60, 60), width=3, label="fallback")
    return pil


def save_decision_image(
    out_dir: Path,
    image_rgb: np.ndarray,
    decision: SubjectDecision,
    reference_box: tuple[float, float, float, float] | None = None,
    center_crop_box: tuple[int, int, int] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_decision_frame(image_rgb, decision, reference_box, center_crop_box)
    path = out_dir / f"{decision.video_id}__frame_{decision.frame}.png"
    rendered.save(path)
    return path
