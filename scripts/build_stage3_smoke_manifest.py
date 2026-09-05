#!/usr/bin/env python3
"""Build the fixed Stage3 Dev Pipeline Smoke Set from a frozen dev split manifest.

Deterministic stratified selection:
- seed 20260905 metadata anchor; selection itself is fully rule-based (no RNG draws
  influence the outcome beyond the fixed sort order)
- strata: reference_segment_count == 1, == 2, >= 3
- within each stratum, balance annotation clip duration and reference coverage via
  fixed tercile cell traversal
- short strata borrow deterministically from the nearest adjacent stratum
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from aic_video_highlight.highlight_retrieval.baseline_experiment import load_baseline_samples

SEED = 20260905
SMOKE_SET_NAME = "Stage3 Dev Smoke Set v1"

# Fixed cell traversal order over (duration tercile, coverage tercile): diagonal
# spread first, then anti-diagonal, then the rest. Indices: 0=low, 1=mid, 2=high.
CELL_ORDER = [
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 2),
    (2, 0),
    (1, 0),
    (1, 2),
    (0, 1),
    (2, 1),
]
BORROW_ORDER = {"1": ["2", "ge3"], "2": ["1", "ge3"], "ge3": ["2", "1"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-layer", type=int, default=4)
    parser.add_argument("--dataset-name", default="aic_highlight_dev")
    parser.add_argument("--dataset-version", default="aic_highlight_dev_v1.1")
    return parser.parse_args()


def layer_of(reference_segment_count: int) -> str:
    if reference_segment_count <= 1:
        return "1"
    if reference_segment_count == 2:
        return "2"
    return "ge3"


def tercile_rank(value: float, sorted_values: list[float]) -> int:
    position = sorted_values.index(value) if value in sorted_values else 0
    # Rank by ordered position; ties resolve to the lower tercile deterministically.
    count = len(sorted_values)
    if count < 3:
        return 0
    if position < count // 3:
        return 0
    if position < (2 * count) // 3:
        return 1
    return 2


def select_within_layer(
    items: list[dict], used_video_ids: set[str], take: int
) -> list[dict]:
    available = sorted(
        (item for item in items if item["video_id"] not in used_video_ids),
        key=lambda item: item["video_id"],
    )
    if not available or take <= 0:
        return []
    duration_order = sorted(item["clip_duration_sec"] for item in available)
    coverage_order = sorted(item["reference_coverage"] for item in available)
    cells: dict[tuple[int, int], list[dict]] = {}
    for item in available:
        duration_rank = tercile_rank(item["clip_duration_sec"], duration_order)
        coverage_rank = tercile_rank(item["reference_coverage"], coverage_order)
        cells.setdefault((duration_rank, coverage_rank), []).append(item)
    chosen: list[dict] = []
    for cell in CELL_ORDER:
        candidates = sorted(cells.get(cell, []), key=lambda item: item["video_id"])
        if candidates:
            chosen.append(candidates[0])
            if len(chosen) >= take:
                return chosen
    for item in available:
        if item not in chosen:
            chosen.append(item)
            if len(chosen) >= take:
                break
    return chosen


def main() -> int:
    args = parse_args()
    samples = load_baseline_samples(
        args.manifest,
        args.video_root,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
    )
    layers: dict[str, list[dict]] = {"1": [], "2": [], "ge3": []}
    for sample in samples:
        layers[layer_of(int(sample["reference_segment_count"]))].append(sample)

    used: set[str] = set()
    smoke: list[dict] = []
    layer_counts: "OrderedDict[str, int]" = OrderedDict()
    for layer_key in ("1", "2", "ge3"):
        need = args.per_layer - len(
            [item for item in smoke if layer_of(int(item["reference_segment_count"])) == layer_key]
        )
        chosen = select_within_layer(layers[layer_key], used, need)
        layer_counts[layer_key] = len(chosen)
        for item in chosen:
            used.add(item["video_id"])
            smoke.append({**item, "smoke_layer": layer_key})
    for layer_key in ("1", "2", "ge3"):
        shortfall = args.per_layer - layer_counts[layer_key]
        for borrow_key in BORROW_ORDER[layer_key]:
            if shortfall <= 0:
                break
            chosen = select_within_layer(layers[borrow_key], used, shortfall)
            for item in chosen:
                used.add(item["video_id"])
                smoke.append({**item, "smoke_layer": layer_key, "smoke_borrowed_from": borrow_key})
            shortfall -= len(chosen)
            layer_counts[layer_key] += len(chosen)

    smoke.sort(key=lambda item: item["sample_index"])
    total = len(smoke)
    expected = args.per_layer * 3
    if total != expected:
        raise SystemExit(
            f"smoke set is incomplete: collected {total} of {expected} samples; "
            f"layer counts: {dict(layer_counts)}"
        )

    payload_lines = []
    for item in smoke:
        record = {
            "smoke_set": SMOKE_SET_NAME,
            "smoke_layer": item["smoke_layer"],
            "smoke_seed": SEED,
            "dataset_name": item.get("dataset_name"),
            "dataset_version": item.get("dataset_version"),
            "split": item["split"],
            "sample_index": item["sample_index"],
            "video_id": item["video_id"],
            "source_group": item["source_group"],
            "relative_video_path": item["relative_video_path"],
            "clip_start_sec": item["clip_start_sec"],
            "clip_end_sec": item["clip_end_sec"],
            "clip_duration_sec": item["clip_duration_sec"],
            "reference_segment_count": item["reference_segment_count"],
            "reference_coverage": item["reference_coverage"],
            "reference_path": item["reference_path"],
            "weak_reference_segments": item["weak_reference_segments"],
        }
        payload_lines.append(json.dumps(record, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
    print(f"SMOKE_SET {SMOKE_SET_NAME} total={total} layers={dict(layer_counts)}")
    print(f"OUTPUT {args.output}")
    for item in smoke:
        print(
            f"  {item['video_id']} layer={item['smoke_layer']} "
            f"duration={item['clip_duration_sec']:.2f}s coverage={item['reference_coverage']:.3f} "
            f"segments={item['reference_segment_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
