"""Thin client for an externally managed OpenAI-compatible vLLM server."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class QwenVLLMClient:
    """Call vLLM only; this class never loads models or manages CUDA."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3.5-4B",
        api_key: str = "EMPTY",
        timeout_sec: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("missing dependency 'openai'; install the project environment first") from exc
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout_sec)

    def health_check(self) -> bool:
        """Return whether the configured model is visible through ``/v1/models``."""
        models = self._client.models.list()
        return any(item.id == self.model for item in models.data)

    def chat_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        extra_body: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            extra_body=extra_body or None,
        )
        return response.choices[0].message.content or ""

    def analyze_video(
        self,
        video: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        coarse_fps: float | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Analyze one local video or HTTP(S) URL using a ``video_url`` content part.

        Local paths become ``file://`` URLs and therefore require the server's
        ``--allowed-local-media-path`` flag. ``media_io_kwargs`` is supported by
        current vLLM releases; its compatibility with the selected Qwen/vLLM
        versions must be confirmed on AutoDL before pinning versions.
        """
        video_reference: str
        if isinstance(video, Path) or not str(video).startswith(("http://", "https://", "file://")):
            path = Path(video).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"video file does not exist: {path}")
            video_reference = path.as_uri()
        else:
            video_reference = str(video)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_reference}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if coarse_fps is not None:
            if coarse_fps <= 0:
                raise ValueError("coarse_fps must be greater than zero")
            request["extra_body"] = {"media_io_kwargs": {"video": {"fps": coarse_fps}}}
        if enable_thinking is not None:
            request.setdefault("extra_body", {})["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }
        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""
