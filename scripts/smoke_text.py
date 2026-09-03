#!/usr/bin/env python3
"""Minimal text-only check for the OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import sys

from aic_video_highlight.stage1.qwen_client import QwenVLLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM text chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    try:
        client = QwenVLLMClient(base_url=args.base_url, model=args.model)
        if not client.health_check():
            raise RuntimeError(f"model is not listed by vLLM: {args.model}")
        print(client.chat_text("请用一句话回答：你可以正常接收文本请求吗？", max_new_tokens=64))
    except Exception as exc:
        print(f"Text smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
