#!/usr/bin/env python3
"""Minimal video-understanding check, separate from highlight judgment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic_video_highlight.highlight_retrieval.qwen_vllm_client import QwenVLLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM video understanding")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    try:
        client = QwenVLLMClient(base_url=args.base_url, model=args.model)
        if not client.health_check():
            raise RuntimeError(f"model is not listed by vLLM: {args.model}")
        print(
            client.analyze_video(
                args.video,
                "请简要说明这段视频中发生了什么。",
                max_new_tokens=128,
                coarse_fps=2.0,
                enable_thinking=False,
            )
        )
    except Exception as exc:
        print(f"Video smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
