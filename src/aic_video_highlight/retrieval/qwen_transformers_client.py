"""Low-VRAM local Qwen3.5 video client backed by Transformers + bitsandbytes.

This is a deployment adapter only.  It implements the same ``analyze_video``
surface as :class:`QwenVLLMClient`; prompt construction, chunking, parsing and
candidate merging remain owned by the frozen retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .qwen_vllm_client import ModelResponse
from aic_video_highlight.runtime.profiles import RuntimeProfile, default_runtime_profile


@dataclass(frozen=True, slots=True)
class LocalQwenStats:
    calls: int
    peak_vram_bytes: int
    peak_reserved_vram_bytes: int
    runtime_profile: str
    attention_backend: str
    gqa_execution_mode: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    backend_evidence: dict[str, Any]
    call_records: tuple[dict[str, Any], ...]


class QwenTransformersClient:
    """Run the exact local Qwen snapshot with deterministic 4-bit inference."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_id: str = "Qwen/Qwen3.5-4B",
        revision: str,
        device: str = "cuda:0",
        quantization: str = "bnb-nf4",
        compute_dtype: str = "float16",
        runtime_profile: RuntimeProfile | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Qwen snapshot does not exist: {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local Qwen backend")
        if quantization != "bnb-nf4":
            raise ValueError("only the reproducible bnb-nf4 deployment is supported")
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if compute_dtype not in dtypes:
            raise ValueError("compute_dtype must be float16 or bfloat16")

        self.torch = torch
        self.model_path = path
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.quantization = quantization
        self.compute_dtype_name = compute_dtype
        self.compute_dtype = dtypes[compute_dtype]
        self.runtime_profile = runtime_profile or default_runtime_profile()
        self.calls = 0
        self.call_records: list[dict[str, Any]] = []
        self._peak_vram_bytes = 0
        self._peak_reserved_vram_bytes = 0
        self._runtime_evidence = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )
        attention_implementation: str | dict[str, str] = "sdpa"
        if self.runtime_profile.uses_local_efficient_sdpa:
            from aic_video_highlight.runtime.efficient_sdpa import (
                RuntimeBackendEvidence,
                probe_efficient_backend,
                register_attention_interface,
            )

            probe_efficient_backend(device, self.compute_dtype)
            attention_name = register_attention_interface()
            attention_implementation = {
                "": "sdpa",
                "vision_config": "sdpa",
                "text_config": attention_name,
            }
            self._runtime_evidence = RuntimeBackendEvidence()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            local_files_only=True,
            quantization_config=quant_config,
            dtype=self.compute_dtype,
            device_map={"": device},
            attn_implementation=attention_implementation,
        )
        if self.runtime_profile.uses_local_efficient_sdpa:
            from aic_video_highlight.runtime.efficient_sdpa import bind_runtime_evidence

            self._bound_attention_modules = bind_runtime_evidence(
                self.model, self._runtime_evidence
            )
            text_config = self.model.config.text_config
            if int(text_config.num_key_value_heads) != 4:
                raise RuntimeError(
                    "local_efficient_sdpa expected the frozen Qwen3.5-4B 4-head KV cache"
                )
        self.model.eval()

    def health_check(self) -> bool:
        return True

    @staticmethod
    def _decode_video(video: Path, fps: float) -> tuple[np.ndarray, dict]:
        """Decode and uniformly sample frames using qwen-vl-utils' frame-count rule."""
        import cv2
        from qwen_vl_utils.vision_process import smart_nframes

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames or source_fps <= 0:
            raise RuntimeError(f"video has no decodable frames or FPS: {video}")
        total = len(frames)
        count = smart_nframes({"fps": float(fps)}, total_frames=total, video_fps=source_fps)
        indices = np.linspace(0, total - 1, count).round().astype(np.int64)
        sampled = np.stack([frames[int(index)] for index in indices], axis=0)
        return sampled, {
            "total_num_frames": total,
            "fps": source_fps,
            "width": int(sampled.shape[2]),
            "height": int(sampled.shape[1]),
            "duration": total / source_fps,
            "video_backend": "opencv",
            "frames_indices": indices.tolist(),
        }

    def analyze_video(
        self,
        video: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        coarse_fps: float | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        path = Path(video).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        fps = 2.0 if coarse_fps is None else float(coarse_fps)
        if fps <= 0:
            raise ValueError("coarse_fps must be greater than zero")
        frames, video_metadata = self._decode_video(path, fps)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            load_audio_from_video=False,
            enable_thinking=False if enable_thinking is None else bool(enable_thinking),
            processor_kwargs={
                "do_sample_frames": False,
                "video_metadata": video_metadata,
                "cap_pixels_per_frame": True,
            },
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }
        if temperature not in (0, 0.0):
            raise ValueError("frozen local deployment requires temperature=0")
        input_length = int(inputs["input_ids"].shape[1])
        sampled_frames = int(frames.shape[0])
        device_index = self.torch.device(self.device)
        self.torch.cuda.reset_peak_memory_stats(device_index)
        free_before, total_memory = self.torch.cuda.mem_get_info(device_index)
        efficient_before = (
            self._runtime_evidence.efficient_attention_calls
            if self._runtime_evidence is not None
            else 0
        )
        self.torch.cuda.synchronize(device_index)
        generate_started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation)
            self.torch.cuda.synchronize(device_index)
        except Exception as exc:
            latency = time.perf_counter() - generate_started
            peak_allocated = int(self.torch.cuda.max_memory_allocated(device_index))
            peak_reserved = int(self.torch.cuda.max_memory_reserved(device_index))
            free_after, _ = self.torch.cuda.mem_get_info(device_index)
            self._peak_vram_bytes = max(self._peak_vram_bytes, peak_allocated)
            self._peak_reserved_vram_bytes = max(
                self._peak_reserved_vram_bytes, peak_reserved
            )
            self.call_records.append(
                {
                    "sequence_length": input_length,
                    "sampled_frames": sampled_frames,
                    "generate_latency_sec": latency,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "device_total_bytes": int(total_memory),
                    "device_used_before_bytes": int(total_memory - free_before),
                    "device_used_after_bytes": int(total_memory - free_after),
                    "success": False,
                    "oom": "out of memory" in str(exc).lower(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        generate_latency = time.perf_counter() - generate_started
        peak_allocated = int(self.torch.cuda.max_memory_allocated(device_index))
        peak_reserved = int(self.torch.cuda.max_memory_reserved(device_index))
        free_after, _ = self.torch.cuda.mem_get_info(device_index)
        efficient_after = (
            self._runtime_evidence.efficient_attention_calls
            if self._runtime_evidence is not None
            else 0
        )
        if self.runtime_profile.uses_local_efficient_sdpa:
            if efficient_after <= efficient_before:
                raise RuntimeError(
                    "local_efficient_sdpa completed generate without an efficient attention call"
                )
            if (
                self._runtime_evidence.cache_kv_heads != 4
                or self._runtime_evidence.compute_kv_heads != 16
                or self._runtime_evidence.math_attention_calls != 0
            ):
                raise RuntimeError(
                    "local_efficient_sdpa runtime evidence violated 4KV->16Q/MATH=0"
                )
        self._peak_vram_bytes = max(self._peak_vram_bytes, peak_allocated)
        self._peak_reserved_vram_bytes = max(
            self._peak_reserved_vram_bytes, peak_reserved
        )
        self.call_records.append(
            {
                "sequence_length": input_length,
                "sampled_frames": sampled_frames,
                "generate_latency_sec": generate_latency,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "device_total_bytes": int(total_memory),
                "device_used_before_bytes": int(total_memory - free_before),
                "device_used_after_bytes": int(total_memory - free_after),
                "efficient_attention_calls": efficient_after - efficient_before,
                "success": True,
                "oom": False,
                "error": None,
            }
        )
        generated = output_ids[:, input_length:]
        content = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.calls += 1
        finish_reason = "length" if generated.shape[1] >= max_new_tokens else "stop"
        return ModelResponse(content=content, finish_reason=finish_reason)

    def stats(self) -> LocalQwenStats:
        return LocalQwenStats(
            calls=self.calls,
            peak_vram_bytes=self._peak_vram_bytes,
            peak_reserved_vram_bytes=self._peak_reserved_vram_bytes,
            runtime_profile=self.runtime_profile.name,
            attention_backend=self.runtime_profile.attention_backend,
            gqa_execution_mode=self.runtime_profile.gqa_execution_mode,
            torch_version=str(self.torch.__version__),
            cuda_version=str(self.torch.version.cuda or "unavailable"),
            gpu_name=str(self.torch.cuda.get_device_name(self.device)),
            backend_evidence=(
                self._runtime_evidence.as_dict()
                if self._runtime_evidence is not None
                else {
                    "attention_backend": "transformers_sdpa_auto",
                    "gqa_execution_mode": "transformers_default",
                }
            ),
            call_records=tuple(self.call_records),
        )
