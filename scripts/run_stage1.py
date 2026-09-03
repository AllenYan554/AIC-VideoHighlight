#!/usr/bin/env python3
"""Run the Stage 1 coarse-recall pipeline against a vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aic_video_highlight.stage1.pipeline import Stage1Pipeline, load_stage1_config
from aic_video_highlight.stage1.qwen_client import QwenVLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 video highlight coarse recall")
    parser.add_argument("--video", required=True, type=Path, help="local video path visible to vLLM")
    parser.add_argument("--config", required=True, type=Path, help="Stage 1 YAML config")
    parser.add_argument("--output", required=True, type=Path, help="internal Stage1Result JSON path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_stage1_config(args.config)
        client = QwenVLLMClient(
            base_url=args.base_url,
            model=config.model,
            timeout_sec=config.request_timeout_sec,
        )
        if not client.health_check():
            raise RuntimeError(f"vLLM is reachable but model is not listed: {config.model}")
        result = Stage1Pipeline(client, config).run(args.video)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(args.output)
    except Exception as exc:
        print(f"Stage 1 failed: {exc}", file=sys.stderr)
        return 1
    print(f"Stage 1 result written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
