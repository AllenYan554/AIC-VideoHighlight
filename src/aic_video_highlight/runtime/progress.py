"""Concise terminal progress plus machine-readable state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import append_jsonl, atomic_write_json
from .run_context import utc_now


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


@dataclass
class ProgressReporter:
    experiment_id: str
    total: int
    log_dir: Path
    started: float = field(default_factory=time.monotonic)
    last_payload: dict[str, Any] | None = None

    def update(
        self,
        processed: int,
        *,
        current_video: str | None = None,
        current_shard: str | None = None,
        errors: int = 0,
        display: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started)
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - processed)
        eta = remaining / rate if rate > 0 else None
        payload = {
            "schema_version": "aic.experiment-progress/v1",
            "experiment_id": self.experiment_id,
            "status": "RUNNING",
            "processed": processed,
            "total": self.total,
            "percent": round(100.0 * processed / self.total, 2) if self.total else 100.0,
            "current_video": current_video,
            "current_shard": current_shard,
            "elapsed_sec": round(elapsed, 3),
            "eta_sec": None if eta is None else round(eta, 3),
            "errors": errors,
            "last_update": utc_now(),
            **(display or {}),
            **extra,
        }
        self.last_payload = payload
        atomic_write_json(self.log_dir / "progress.json", payload)
        append_jsonl(self.log_dir / "progress.jsonl", payload)
        width = 24
        filled = int(width * processed / self.total) if self.total else width
        bar = "█" * filled + "-" * (width - filled)
        line = (
            f"\r{self.experiment_id} [{bar}] {processed}/{self.total} {payload['percent']:6.2f}% "
            f"elapsed={_clock(elapsed)} eta={_clock(None if eta is None else eta)} errors={errors}"
        )
        if current_video:
            line += f" current={current_video}"
        if display:
            line += " " + " ".join(f"{key}={display[key]}" for key in display)
        print(line, end="", flush=True)
        if processed >= self.total:
            print()
        return payload

    def interrupt(self) -> None:
        if self.last_payload is None:
            return
        payload = {**self.last_payload, "status": "INTERRUPTED", "last_update": utc_now()}
        atomic_write_json(self.log_dir / "progress.json", payload)
        append_jsonl(self.log_dir / "events.jsonl", {"event": "INTERRUPTED", **payload})
