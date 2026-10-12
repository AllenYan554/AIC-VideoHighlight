#!/usr/bin/env python3
"""Stage 7.1 YouTube Highlights expansion: live availability preflight recheck.

Re-checks the current YouTube availability of Stage 7.1 official-TRAIN
candidates that were previously recorded as UNAVAILABLE or ERROR and are not
present in the local raw download. The recheck is read-only (yt-dlp simulate);
it never downloads and never mutates frozen manifests.

Classification per video (same vocabulary as the frozen availability manifest):
    AVAILABLE              metadata extraction succeeded
    SIGN_IN_REQUIRED       age/login gated
    UNAVAILABLE            removed, private, or otherwise not retrievable
    ERROR                  any other extraction failure

Outputs (all write targets are explicit command-line arguments):
    <output-jsonl>  one JSON record per checked candidate
    <output-json>   aggregate summary
    <log>           progress log with ETA
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SIGN_IN_MARKERS = ("sign in", "login required", "confirm your age", "age-restricted")
UNAVAILABLE_MARKERS = (
    "this video is not available",
    "this video is unavailable",
    "video unavailable",
    "private video",
    "video has been removed",
    "no longer available",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def classify_ytdlp_failure(text: str) -> str:
    lowered = text.lower()
    for marker in SIGN_IN_MARKERS:
        if marker in lowered:
            return "SIGN_IN_REQUIRED"
    for marker in UNAVAILABLE_MARKERS:
        if marker in lowered:
            return "UNAVAILABLE"
    return "ERROR"


def yt_dlp_version(yt_dlp: str) -> str:
    try:
        completed = subprocess.run(
            [yt_dlp, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        return (completed.stdout or "").strip() or "UNKNOWN"
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def probe_video(video_id: str, *, yt_dlp: str, timeout_sec: float) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = [
        yt_dlp,
        "--simulate",
        "--skip-download",
        "--no-warnings",
        "--no-playlist",
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        "--print",
        "%(id)s\t%(duration)s\t%(title)s",
        url,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "ERROR",
            "reason": "PROBE_TIMEOUT",
            "duration_s": None,
            "title": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
    elapsed = round(time.perf_counter() - started, 3)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode == 0 and stdout:
        fields = stdout.splitlines()[-1].split("\t")
        duration = None
        title = None
        if len(fields) >= 2:
            try:
                duration = float(fields[1])
            except (TypeError, ValueError):
                duration = None
        if len(fields) >= 3:
            title = fields[2] or None
        return {
            "status": "AVAILABLE",
            "reason": None,
            "duration_s": duration,
            "title": title,
            "elapsed_s": elapsed,
        }
    combined = f"{stdout}\n{stderr}".strip()
    return {
        "status": classify_ytdlp_failure(combined),
        "reason": combined.splitlines()[-1][:240] if combined else "NO_OUTPUT",
        "duration_s": None,
        "title": None,
        "elapsed_s": elapsed,
    }


def select_recheck_scope(
    candidates: list[dict[str, Any]],
    availability_by_id: dict[str, dict[str, Any]],
    raw_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = str(candidate["canonical_video_id"])
        official_split = str(candidate.get("official_split", "")).lower()
        if official_split != "train":
            continue
        if video_id in raw_ids:
            continue
        entry = availability_by_id.get(video_id, {})
        if str(entry.get("status", "")) not in {"UNAVAILABLE", "ERROR"}:
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda row: (str(row.get("category")), str(row["canonical_video_id"])))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--availability-manifest", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--yt-dlp", default="yt-dlp")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = _read_jsonl(args.candidate_manifest)
    availability_by_id = _index_by(_read_jsonl(args.availability_manifest), "canonical_video_id")
    raw_ids = {str(row["canonical_video_id"]) for row in _read_jsonl(args.raw_manifest)}
    scope = select_recheck_scope(candidates, availability_by_id, raw_ids)
    if args.video_ids:
        wanted = set(args.video_ids)
        scope = [row for row in scope if str(row["canonical_video_id"]) in wanted]
    if args.limit:
        scope = scope[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("", encoding="utf-8")
    log_handle = args.log.open("w", encoding="utf-8")
    tool_version = yt_dlp_version(args.yt_dlp)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    started = time.perf_counter()
    total = len(scope)

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log_handle.write(f"[{stamp}] {message}\n")
        log_handle.flush()
        print(message, flush=True)

    log(f"preflight recheck start: total={total} yt_dlp={args.yt_dlp}")
    for position, candidate in enumerate(scope, start=1):
        video_id = str(candidate["canonical_video_id"])
        result = probe_video(video_id, yt_dlp=args.yt_dlp, timeout_sec=args.timeout_sec)
        record = {
            "schema": "aic.stage7.yth.expansion-preflight-recheck/v1",
            "dataset": "youtube_highlights",
            "canonical_video_id": video_id,
            "category": candidate.get("category"),
            "official_split": candidate.get("official_split"),
            "official_set": candidate.get("official_set"),
            "previous_status": availability_by_id.get(video_id, {}).get("status"),
            "previous_reason": availability_by_id.get(video_id, {}).get("reason"),
            "check_time_utc": datetime.now(timezone.utc).isoformat(),
            "yt_dlp_version": tool_version,
            **result,
        }
        records.append(record)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if position % 5 == 0 or position == total or result["status"] == "AVAILABLE":
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            eta_min = (total - position) * rate / 60.0
            log(
                f"progress {position}/{total} current={video_id} status={result['status']} "
                f"rate={rate:.1f}s/vid eta={eta_min:.1f}min counts={counts}"
            )
        with args.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema": "aic.stage7.yth.expansion-preflight-summary/v1",
        "dataset": "youtube_highlights",
        "checked_total": total,
        "status_counts": counts,
        "available_video_ids": [
            record["canonical_video_id"] for record in records if record["status"] == "AVAILABLE"
        ],
        "previous_status_counts": {},
        "yt_dlp_version": tool_version,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for record in records:
        key = str(record["previous_status"])
        summary["previous_status_counts"][key] = summary["previous_status_counts"].get(key, 0) + 1
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"preflight recheck done: {counts}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
