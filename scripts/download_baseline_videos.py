"""Download only the QVHighlights videos listed in a baseline manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ID = "ayushsdev/qvhighlights-videos"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a JSON array")

    seen: set[str] = set()
    for index, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        filename = item.get("filename")
        hf_path = item.get("hf_path")
        if not isinstance(filename, str) or not filename.endswith(".mp4"):
            raise ValueError(f"manifest item {index} has an invalid filename")
        if Path(filename).name != filename:
            raise ValueError(f"manifest item {index} filename must not contain a path")
        expected_hf_path = f"{filename[0]}/{filename}"
        if hf_path != expected_hf_path:
            raise ValueError(
                f"manifest item {index} hf_path must be {expected_hf_path!r}"
            )
        if filename in seen:
            raise ValueError(f"duplicate filename in manifest: {filename}")
        seen.add(filename)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download exactly the QVHighlights files named by a baseline manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip existing non-empty destination files (default: true)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".hf-download"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        print(
            "huggingface_hub is required in the active environment; no files downloaded",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    downloaded = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for item in manifest:
        filename = item["filename"]
        destination = output_dir / filename
        if args.skip_existing and destination.is_file() and destination.stat().st_size > 0:
            skipped += 1
            print(f"SKIP {filename} ({destination.stat().st_size} bytes)")
            continue

        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=item["hf_path"],
                    local_dir=staging_dir,
                )
            )
            if not cached_path.is_file() or cached_path.stat().st_size == 0:
                raise RuntimeError("download returned a missing or empty file")
            shutil.move(str(cached_path), destination)
            downloaded += 1
            print(f"OK   {filename} ({destination.stat().st_size} bytes)")
        except Exception as error:  # Report every individual file failure, then fail the run.
            failures.append((filename, f"{type(error).__name__}: {error}"))
            print(f"FAIL {filename}: {type(error).__name__}: {error}", file=sys.stderr)

    print(
        f"SUMMARY requested={len(manifest)} downloaded={downloaded} "
        f"skipped={skipped} failed={len(failures)}"
    )
    if failures:
        print("FAILED FILES", file=sys.stderr)
        for filename, message in failures:
            print(f"- {filename}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
