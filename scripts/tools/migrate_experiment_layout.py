#!/usr/bin/env python3
"""Apply an explicit copy-first migration plan; dry-run is the default."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aic_video_highlight.experiment_runtime.hashing import file_sha256
from aic_video_highlight.experiment_runtime.io import append_jsonl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    mode.add_argument("--apply", action="store_true", help="Copy and verify; never deletes sources")
    parser.add_argument("--log", type=Path, default=Path("migration.jsonl"))
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for item in plan.get("operations", []):
        action = item["action"]
        source_text, destination_text = item["source"], item["destination"]
        source, destination = Path(source_text), Path(destination_text)
        if action not in {"COPY", "KEEP", "ARCHIVE_CANDIDATE"}:
            raise SystemExit(f"unsupported safe action: {action}")
        result = {"action": action, "source": source_text, "destination": destination_text, "mode": "APPLY" if args.apply else "DRY_RUN"}
        if args.apply and action == "COPY":
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
            source_hash, destination_hash = file_sha256(source), file_sha256(destination)
            if source_hash != destination_hash:
                raise RuntimeError(f"copy hash mismatch: {source}")
            result.update({"source_sha256": source_hash, "destination_sha256": destination_hash, "verified": True})
        print(json.dumps(result, ensure_ascii=False))
        append_jsonl(args.log, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
