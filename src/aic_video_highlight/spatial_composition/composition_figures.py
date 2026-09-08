"""Stage 5.3 smoke visualization: geometry-true figures for audit categories.

Every figure shows the frame extent with raw primary bbox, sanitized bbox,
CMP-0 crop, CMP-1 crop, and (when present) the weak reference. When the source
video is unavailable (CPU smoke without media), a schematic canvas is used and
the figure index tags it GEOMETRY_SCHEMATIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aic_video_highlight.experiment_runtime.io import atomic_write_json

COLOR_CMP0 = (120, 120, 120)
COLOR_CMP1 = (30, 90, 220)
COLOR_RAW = (220, 40, 40)
COLOR_SANITIZED = (240, 150, 30)
COLOR_WEAK = (40, 170, 70)
COLOR_FRAME = (15, 15, 15)

FIGURE_CATEGORIES = (
    "top_improvement",
    "regression",
    "strong_off_center",
    "box_too_large",
    "ambiguous",
    "non_person",
    "border_clamp",
)


@dataclass(frozen=True, slots=True)
class FigureSpec:
    category: str
    record: Mapping[str, Any]
    raw_bbox: Sequence[float] | None


def select_figures(
    records: Sequence[Mapping[str, Any]],
    box_too_large_boxes: Mapping[tuple[str, int], Sequence[float]] | None = None,
    per_category: int = 3,
) -> list[FigureSpec]:
    """Deterministic category coverage; failures are included, not only successes."""
    boxes = box_too_large_boxes or {}
    specs: list[FigureSpec] = []
    with_subject = [r for r in records if r.get("sanitized", {}).get("status") == "OK"]

    def take(category: str, rows: Sequence[Mapping[str, Any]], raw: Sequence[float] | None = None) -> None:
        for record in rows[:per_category]:
            specs.append(FigureSpec(category, record, raw))

    improvements = sorted(
        with_subject,
        key=lambda r: (r["cmp1"]["subject_visible_fraction"] - r["cmp0"]["subject_visible_fraction"], r["video_id"], r["frame"]),
        reverse=True,
    )
    take("top_improvement", improvements)
    take("regression", list(reversed(improvements)))

    off_center = sorted(
        (r for r in with_subject if r["stratum"] == "strongly_off_center"),
        key=lambda r: (r["horizontal_center_offset"], r["video_id"], r["frame"]),
        reverse=True,
    )
    take("strong_off_center", off_center)

    too_large = sorted(
        (r for r in records if "BOX_TOO_LARGE" in r["fallback_reasons"]),
        key=lambda r: (r["video_id"], r["frame"]),
    )
    for record in too_large[:per_category]:
        take("box_too_large", [record], boxes.get((record["video_id"], int(record["frame"]))))

    ambiguous = [r for r in records if r["ambiguous_candidate_count"] > 0]
    ambiguous.sort(key=lambda r: (-r["ambiguous_candidate_count"], r["video_id"], r["frame"]))
    take("ambiguous", ambiguous)

    non_person = [r for r in with_subject if r["primary_label"] not in (None, "person")]
    non_person.sort(key=lambda r: (r["video_id"], r["frame"]))
    take("non_person", non_person)

    clamped = [r for r in records if r["cmp1"].get("clamped_x") or r["cmp1"].get("clamped_y")]
    clamped.sort(key=lambda r: (r["video_id"], r["frame"]))
    take("border_clamp", clamped)
    return specs


def _load_frame_image(videos_root: Path | None, video_path: str | None, frame_index: int):
    """Decode one frame via OpenCV when media is available; otherwise None."""
    if videos_root is None or not video_path:
        return None
    try:
        import cv2
        import numpy as np

        candidate = Path(video_path)
        path = candidate if candidate.is_absolute() else videos_root / candidate
        if not path.is_file():
            return None
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
        finally:
            capture.release()
        if not ok or frame_bgr is None:
            return None
        return np.ascontiguousarray(frame_bgr[:, :, ::-1])
    except Exception:
        return None


def _scale(pil_image, max_width: int = 720):
    width, height = pil_image.size
    if width <= max_width:
        return pil_image
    ratio = max_width / width
    return pil_image.resize((max_width, max(1, int(height * ratio))))


def render_figure(
    spec: FigureSpec,
    videos_root: Path | None,
    video_path: str | None,
    weak_roi: Sequence[float] | None,
):
    from PIL import Image, ImageDraw

    record = spec.record
    width, height = int(record["image_width"]), int(record["image_height"])
    frame_array = _load_frame_image(videos_root, video_path, int(record["frame"]))
    schematic = frame_array is None
    if schematic:
        image = Image.new("RGB", (width, height), (250, 250, 250))
    else:
        image = Image.fromarray(frame_array.copy())
    draw = ImageDraw.Draw(image)

    def rect(box, color, width_px=2, label=None, dashed=False):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if dashed:
            step = 6
            for x in range(int(x1), int(x2), step * 2):
                draw.line([(x, y1), (min(x + step, x2), y1)], fill=color, width=width_px)
                draw.line([(x, y2), (min(x + step, x2), y2)], fill=color, width=width_px)
            for y in range(int(y1), int(y2), step * 2):
                draw.line([(x1, y), (x1, min(y + step, y2))], fill=color, width=width_px)
                draw.line([(x2, y), (x2, min(y + step, y2))], fill=color, width=width_px)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width_px)
        if label:
            draw.text((x1 + 2, max(0.0, y1 - 11)), label, fill=color)

    cmp0, cmp1 = record["cmp0"], record["cmp1"]
    rect((cmp0["x"], cmp0["y"], cmp0["x"] + cmp0["w"], cmp0["y"] + cmp0["h"]), COLOR_CMP0, 2, "CMP-0")
    rect((cmp1["x"], cmp1["y"], cmp1["x"] + cmp1["w"], cmp1["y"] + cmp1["h"]), COLOR_CMP1, 2, "CMP-1")
    if spec.raw_bbox is not None:
        rect(spec.raw_bbox, COLOR_RAW, 2, "raw", dashed=True)
    sanitized = record["sanitized"]
    if sanitized.get("xyxy"):
        rect(sanitized["xyxy"], COLOR_SANITIZED, 2, "sanitized")
    if weak_roi is not None:
        rect(weak_roi, COLOR_WEAK, 2, "weak-ref")
    draw.rectangle([0, 0, width - 1, height - 1], outline=COLOR_FRAME, width=1)
    return _scale(image), schematic


def render_all_figures(
    records: Sequence[Mapping[str, Any]],
    out_dir: Path,
    videos_root: Path | None = None,
    video_paths: Mapping[str, str] | None = None,
    box_too_large_boxes: Mapping[tuple[str, int], Sequence[float]] | None = None,
    weak_reference: Mapping[str, Mapping[str, Any]] | None = None,
    per_category: int = 3,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_paths = video_paths or {}
    weak_reference = weak_reference or {}
    specs = select_figures(records, box_too_large_boxes, per_category)
    index_entries: list[dict[str, Any]] = []
    counter: dict[str, int] = {}
    for spec in specs:
        counter[spec.category] = counter.get(spec.category, 0) + 1
        weak_roi = (
            weak_reference.get(spec.record["video_id"], {}).get("rois", {}).get(str(spec.record["frame"]))
        )
        image, schematic = render_figure(
            spec, videos_root, video_paths.get(spec.record["video_id"]), weak_roi
        )
        name = f"{spec.category}{counter[spec.category]}__{spec.record['video_id']}__frame_{spec.record['frame']}.png"
        image.save(out_dir / name)
        index_entries.append(
            {
                "file": name,
                "category": spec.category,
                "video_id": spec.record["video_id"],
                "frame": spec.record["frame"],
                "mode": "GEOMETRY_SCHEMATIC" if schematic else "SOURCE_FRAME",
                "cmp0_visible_fraction": spec.record["cmp0"]["subject_visible_fraction"],
                "cmp1_visible_fraction": spec.record["cmp1"]["subject_visible_fraction"],
                "stratum": spec.record["stratum"],
            }
        )
    payload = {
        "schema_version": "aic.stage5_3.figure-index/v1",
        "figure_count": len(index_entries),
        "categories": list(FIGURE_CATEGORIES),
        "figures": index_entries,
    }
    atomic_write_json(out_dir / "index.json", payload)
    return payload
