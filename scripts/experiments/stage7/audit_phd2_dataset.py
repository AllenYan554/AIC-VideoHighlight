#!/usr/bin/env python3
"""PHD2 dataset feasibility audit: metadata-only inventory, sampling, probe, size estimate.

Subcommands:
    inventory   parse the official CSVs and emit deterministic statistics
    sample      stratified random sample of unique videos (split x duration bin)
    probe       metadata-only YouTube probe (availability + format sizes), resumable
    estimate    availability rate with 95% CI and 360p/480p/720p storage budgets

Design constraints:
    - metadata only: videos are never downloaded; only ``extract_info`` is called
    - deterministic: sampling uses a fixed seed; probe order is the sample order
    - resumable: probe results append to a JSONL cache and completed ids are skipped
    - rate-limit safe: serial requests with sleep, exponential backoff on HTTP 429,
      and an automatic stop after repeated consecutive failures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DURATION_BINS = (
    ("0-1min", 0.0, 60.0),
    ("1-3min", 60.0, 180.0),
    ("3-10min", 180.0, 600.0),
    ("10-30min", 600.0, 1800.0),
    ("30min+", 1800.0, float("inf")),
)
RESOLUTION_CAPS = {"360p": 360, "480p": 480, "720p": 720}
SAMPLE_SEED = 20260920


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def duration_bin_label(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    for label, low, high in DURATION_BINS:
        if low <= seconds < high:
            return label
    return "unknown"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def build_unique_video_table(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    """One deterministic record per unique youtubeId (first row wins)."""

    table: dict[str, dict[str, Any]] = {}
    for split, rows in (("TRAIN", train_rows), ("TEST", test_rows)):
        for row in rows:
            video_id = row.get("youtubeId", "").strip()
            if not video_id:
                continue
            record = table.setdefault(
                video_id,
                {
                    "youtubeId": video_id,
                    "video_duration": to_float(row.get("video_duration")),
                    "splits": set(),
                    "rows": 0,
                },
            )
            record["splits"].add(split)
            record["rows"] += 1
    for record in table.values():
        record["splits"] = sorted(record["splits"])
        record["duration_bin"] = duration_bin_label(record["video_duration"])
    return table


def command_inventory(args: argparse.Namespace) -> int:
    train_rows = load_csv(args.metadata_root / "training.csv")
    test_rows = load_csv(args.metadata_root / "testing.csv")
    payload: dict[str, Any] = {
        "schema": "aic.phd2.inventory/v1",
        "generated_at_utc": utc_now(),
        "source": {
            "repository": "https://github.com/gifs/personalized-highlights-dataset",
            "files": {},
        },
        "annotation_count_note": (
            "CSV rows are highlight annotations, NOT videos; video counts below are unique youtubeId counts."
        ),
    }
    for label, name, rows in (
        ("TRAIN", "training.csv", train_rows),
        ("TEST", "testing.csv", test_rows),
    ):
        videos = {row["youtubeId"] for row in rows if row.get("youtubeId")}
        users = {row["user_id"] for row in rows if row.get("user_id")}
        highlights = [to_float(row["duration"]) for row in rows]
        unique_durations = []
        seen: set[str] = set()
        for row in rows:
            video_id = row.get("youtubeId", "")
            if video_id and video_id not in seen:
                seen.add(video_id)
                value = to_float(row.get("video_duration"))
                if value is not None and value >= 0:
                    unique_durations.append(value)
        payload["source"]["files"][name] = {
            "sha256": sha256_file(args.metadata_root / name),
            "bytes": (args.metadata_root / name).stat().st_size,
        }
        payload[label] = {
            "rows_annotations": len(rows),
            "unique_videos": len(videos),
            "unique_users": len(users),
            "highlight_duration_s": describe([v for v in highlights if v is not None]),
            "video_duration_s": describe(unique_durations),
            "total_unique_video_hours": sum(unique_durations) / 3600.0,
            "rows_is_last_true": sum(1 for row in rows if row.get("is_last") == "True"),
        }

    table = build_unique_video_table(train_rows, test_rows)
    all_durations = [
        record["video_duration"]
        for record in table.values()
        if record["video_duration"] is not None and record["video_duration"] >= 0
    ]
    bin_counts = {label: 0 for label, _low, _high in DURATION_BINS}
    bin_hours = {label: 0.0 for label, _low, _high in DURATION_BINS}
    for record in table.values():
        label = record["duration_bin"]
        if label in bin_counts:
            bin_counts[label] += 1
            if record["video_duration"] is not None and record["video_duration"] >= 0:
                bin_hours[label] += record["video_duration"] / 3600.0
    highlights_per_video: dict[str, int] = {}
    for rows in (train_rows, test_rows):
        for row in rows:
            video_id = row.get("youtubeId", "")
            if video_id:
                highlights_per_video[video_id] = highlights_per_video.get(video_id, 0) + 1
    train_ids = {row["youtubeId"] for row in train_rows}
    test_ids = {row["youtubeId"] for row in test_rows}
    payload["COMBINED"] = {
        "rows_annotations": len(train_rows) + len(test_rows),
        "unique_videos": len(table),
        "unique_users": len({row["user_id"] for row in train_rows + test_rows}),
        "train_test_video_overlap": len(train_ids & test_ids),
        "total_unique_video_hours": sum(all_durations) / 3600.0,
        "video_duration_s": describe(all_durations),
        "duration_bin_counts": bin_counts,
        "duration_bin_hours": {key: round(value, 1) for key, value in bin_hours.items()},
        "highlights_per_video": {
            "mean": statistics.mean(highlights_per_video.values()),
            "median": statistics.median(highlights_per_video.values()),
            "p90": percentile(list(highlights_per_video.values()), 90),
            "max": max(highlights_per_video.values()),
            "videos_with_single_highlight": sum(
                1 for value in highlights_per_video.values() if value == 1
            ),
            "videos_with_multiple_highlights": sum(
                1 for value in highlights_per_video.values() if value >= 2
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["COMBINED"], ensure_ascii=False, indent=2))
    return 0


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_sample(args: argparse.Namespace) -> int:
    train_rows = load_csv(args.metadata_root / "training.csv")
    test_rows = load_csv(args.metadata_root / "testing.csv")
    table = build_unique_video_table(train_rows, test_rows)
    strata: dict[tuple[str, str], list[str]] = {}
    for record in table.values():
        split = "TRAIN" if "TRAIN" in record["splits"] else "TEST"
        strata.setdefault((split, record["duration_bin"]), []).append(record["youtubeId"])
    rng = random.Random(args.seed)
    total = len(table)
    sample: list[dict[str, Any]] = []
    for (split, bin_label), ids in sorted(strata.items()):
        ids = sorted(ids)
        share = len(ids) / total
        quota = max(1, round(share * args.size))
        quota = min(quota, len(ids))
        picked = rng.sample(ids, quota)
        for video_id in picked:
            record = table[video_id]
            sample.append(
                {
                    "youtubeId": video_id,
                    "split": split,
                    "duration_bin": bin_label,
                    "video_duration": record["video_duration"],
                }
            )
    rng.shuffle(sample)
    payload = {
        "schema": "aic.phd2.sample/v1",
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "requested_size": args.size,
        "sampled": len(sample),
        "population": total,
        "strata": {
            f"{split}|{bin_label}": {
                "population": len(ids),
                "sampled": sum(
                    1
                    for row in sample
                    if row["split"] == split and row["duration_bin"] == bin_label
                ),
            }
            for (split, bin_label), ids in sorted(strata.items())
        },
        "videos": sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sampled={len(sample)} population={total} seed={args.seed}")
    for key, value in payload["strata"].items():
        print(f"  {key}: population={value['population']} sampled={value['sampled']}")
    return 0


RATE_LIMIT_MARKERS = ("http error 429", "too many requests", "rate-limited", "rate limited")
NETWORK_MARKERS = (
    "connection aborted",
    "connectionreseterror",
    "connection reset",
    "unable to download api page",
    "remote end closed connection",
)


def classify_probe_error(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in RATE_LIMIT_MARKERS):
        return "RATE_LIMITED"
    if any(marker in lowered for marker in NETWORK_MARKERS):
        return "NETWORK_ERROR"
    if "confirm your age" in lowered or "age-restricted" in lowered:
        return "AGE_RESTRICTED"
    if "sign in" in lowered or "login required" in lowered:
        return "SIGN_IN_REQUIRED"
    if "not available in your country" in lowered or "geo" in lowered and "block" in lowered:
        return "REGION_BLOCKED"
    if "private video" in lowered:
        return "PRIVATE"
    if (
        "video is unavailable" in lowered
        or "video is not available" in lowered
        or "video unavailable" in lowered
    ):
        return "UNAVAILABLE"
    if "removed" in lowered or "deleted" in lowered or "terminated" in lowered:
        return "UNAVAILABLE"
    return "UNKNOWN_ERROR"


def pick_format_size(formats: list[dict[str, Any]], height_cap: int, duration: float | None) -> dict[str, Any] | None:
    """Best video stream <= cap plus best audio-only stream; conservative size math."""

    video_options = [
        fmt
        for fmt in formats
        if fmt.get("vcodec") not in (None, "none")
        and fmt.get("height") is not None
        and int(fmt["height"]) <= height_cap
    ]
    if not video_options:
        return None

    def fmt_size(fmt: dict[str, Any]) -> float | None:
        size = fmt.get("filesize") or fmt.get("filesize_approx")
        if size:
            return float(size)
        tbr = fmt.get("tbr")
        if tbr and duration:
            return float(tbr) * 1000.0 / 8.0 * duration
        return None

    video_options.sort(
        key=lambda fmt: (
            int(fmt["height"]),
            fmt.get("tbr") or 0.0,
        ),
        reverse=True,
    )
    chosen_video = None
    for candidate in video_options:
        if fmt_size(candidate) is not None:
            chosen_video = candidate
            break
    if chosen_video is None:
        chosen_video = video_options[0]
    video_size = fmt_size(chosen_video)
    if video_size is None:
        return None
    progressive = chosen_video.get("acodec") not in (None, "none")
    audio_size = 0.0
    audio_info: dict[str, Any] | None = None
    if not progressive:
        audio_options = [
            fmt
            for fmt in formats
            if fmt.get("acodec") not in (None, "none")
            and fmt.get("vcodec") in (None, "none")
        ]
        audio_options.sort(key=lambda fmt: fmt.get("abr") or 0.0, reverse=True)
        for candidate in audio_options:
            size = fmt_size(candidate)
            if size is not None:
                audio_size = size
                audio_info = {
                    "format_id": candidate.get("format_id"),
                    "abr": candidate.get("abr"),
                    "ext": candidate.get("ext"),
                }
                break
    return {
        "height": chosen_video.get("height"),
        "format_id": chosen_video.get("format_id"),
        "ext": chosen_video.get("ext"),
        "vcodec": (chosen_video.get("vcodec") or "")[:40],
        "acodec_included": bool(progressive),
        "video_bytes": round(video_size),
        "audio_bytes": round(audio_size),
        "total_bytes": round(video_size + audio_size),
        "size_is_approx": not bool(chosen_video.get("filesize")),
        "audio": audio_info,
    }


def load_probe_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[str(record.get("youtubeId"))] = record
    return cache


def probe_one(video_id: str, duration_hint: float | None, timeout: float) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = [
        "yt-dlp",
        "--simulate",
        "--dump-single-json",
        "--no-warnings",
        "--no-playlist",
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        url,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"status": "UNKNOWN_ERROR", "error": "PROBE_TIMEOUT", "elapsed_s": round(time.perf_counter() - started, 3)}
    elapsed = round(time.perf_counter() - started, 3)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode == 0 and stdout.strip():
        info: dict[str, Any] | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    info = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
        if info is not None:
            duration = info.get("duration")
            if not isinstance(duration, (int, float)):
                duration = duration_hint
            formats = info.get("formats") or []
            record: dict[str, Any] = {
                "youtubeId": str(info.get("id") or video_id),
                "status": "AVAILABLE",
                "duration_s": duration,
                "title": str(info.get("title") or "")[:120],
                "elapsed_s": elapsed,
                "resolutions": {},
            }
            for label, cap in RESOLUTION_CAPS.items():
                record["resolutions"][label] = pick_format_size(formats, cap, duration)
            return record
        return {"status": "UNKNOWN_ERROR", "error": "PARSE_FAILURE", "elapsed_s": elapsed}
    combined = f"{stdout}\n{stderr}"
    return {
        "status": classify_probe_error(combined),
        "error": combined.strip().splitlines()[-1][:200] if combined.strip() else "NO_OUTPUT",
        "elapsed_s": elapsed,
    }


def command_probe(args: argparse.Namespace) -> int:
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    cache = load_probe_cache(args.output_jsonl)
    pending = [row for row in sample["videos"] if str(row["youtubeId"]) not in cache]
    if args.limit:
        pending = pending[: args.limit]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", encoding="utf-8")
    started = time.perf_counter()
    consecutive_rate_limits = 0
    completed_now = 0
    for position, row in enumerate(pending, start=1):
        video_id = str(row["youtubeId"])
        if consecutive_rate_limits >= args.rate_limit_abort:
            message = f"aborting after {consecutive_rate_limits} consecutive rate limits; resume later"
            print(message, flush=True)
            log.write(f"[{utc_now()}] {message}\n")
            break
        result = probe_one(video_id, row.get("video_duration"), args.timeout)
        record = {
            "youtubeId": video_id,
            "split": row.get("split"),
            "duration_bin": row.get("duration_bin"),
            "video_duration_csv": row.get("video_duration"),
            "checked_at_utc": utc_now(),
            **result,
        }
        if result["status"] == "RATE_LIMITED":
            consecutive_rate_limits += 1
            backoff = min(60.0 * (2 ** (consecutive_rate_limits - 1)), 600.0)
            log.write(
                f"[{utc_now()}] RATE_LIMITED {video_id} consecutive={consecutive_rate_limits} backoff={backoff:.0f}s\n"
            )
            time.sleep(backoff)
            continue
        consecutive_rate_limits = 0
        with args.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed_now += 1
        if position % 25 == 0 or position == len(pending):
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            eta = (len(pending) - position) * rate / 60.0
            summary = {
                "done": position,
                "total": len(pending),
                "new": completed_now,
                "rate_s": round(rate, 2),
                "eta_min": round(eta, 1),
            }
            print(f"progress {json.dumps(summary)}", flush=True)
            log.write(f"[{utc_now()}] progress {json.dumps(summary)}\n")
        time.sleep(args.sleep)
    log.close()
    print(f"probe finished: new={completed_now} cache_total={len(load_probe_cache(args.output_jsonl))}")
    return 0


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def command_estimate(args: argparse.Namespace) -> int:
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    cache = load_probe_cache(args.probe_jsonl)
    all_records = [dict(row) for row in cache.values()]
    for row in all_records:
        if row.get("status") in ("UNKNOWN_ERROR", "SIGN_IN_REQUIRED") and row.get("error"):
            row["status"] = classify_probe_error(str(row["error"]))
    probe_artifact_errors = {"PROBE_TIMEOUT", "NETWORK_ERROR"}
    timeout_records = [
        row
        for row in all_records
        if row.get("error") == "PROBE_TIMEOUT" or row.get("status") == "NETWORK_ERROR"
    ]
    records = [
        row
        for row in all_records
        if row.get("error") != "PROBE_TIMEOUT" and row.get("status") != "NETWORK_ERROR"
    ]
    available = [row for row in records if row.get("status") == "AVAILABLE"]
    status_counts: dict[str, int] = {}
    for row in records:
        status_counts[row.get("status", "UNKNOWN")] = status_counts.get(row.get("status", "UNKNOWN"), 0) + 1
    total_unique = int(inventory["COMBINED"]["unique_videos"])
    low, high = wilson_interval(len(available), len(records))
    availability_rate = len(available) / len(records) if records else 0.0
    estimated_recoverable = {
        "point": round(availability_rate * total_unique),
        "low_95": round(low * total_unique),
        "high_95": round(high * total_unique),
    }
    sizes: dict[str, dict[str, Any]] = {}
    total_stat: dict[str, dict[str, float]] = {}
    for label in RESOLUTION_CAPS:
        per_video = [
            float(row["resolutions"][label]["total_bytes"])
            for row in available
            if row.get("resolutions", {}).get(label)
        ]
        if not per_video:
            sizes[label] = {"videos_with_size": 0}
            continue
        distribution = describe(per_video)
        mean_bytes = distribution["mean"]
        stdev = statistics.pstdev(per_video)
        standard_error = stdev / math.sqrt(len(per_video))
        mean_low = max(0.0, mean_bytes - 1.96 * standard_error)
        mean_high = mean_bytes + 1.96 * standard_error
        sizes[label] = {
            "videos_with_size": len(per_video),
            "bytes_per_video": distribution,
            "mean_mib_per_video": mean_bytes / (1024 ** 2),
            "mean_bytes_ci95": {"low": mean_low, "high": mean_high},
            "estimated_total_gib": {
                "point": mean_bytes * estimated_recoverable["point"] / (1024 ** 3),
                "low_95": mean_low * estimated_recoverable["low_95"] / (1024 ** 3),
                "high_95": mean_high * estimated_recoverable["high_95"] / (1024 ** 3),
            },
        }
        total_stat[label] = {"mean": mean_bytes, "mean_low": mean_low, "mean_high": mean_high}
    budgets: dict[str, dict[str, Any]] = {}
    for label in RESOLUTION_CAPS:
        stats = total_stat.get(label)
        if not stats:
            continue
        budget_counts: dict[str, int] = {
            str(count): count for count in (1000, 5000, 10000, 20000)
        }
        budget_counts["all_recoverable"] = estimated_recoverable["point"]
        budgets[label] = {
            key: {
                "videos": count,
                "gib": stats["mean"] * count / (1024 ** 3),
                "gb": stats["mean"] * count / 1e9,
            }
            for key, count in budget_counts.items()
        }
        budgets[label]["all_recoverable_gib_low_high"] = {
            "low_95": stats["mean_low"] * estimated_recoverable["low_95"] / (1024 ** 3),
            "high_95": stats["mean_high"] * estimated_recoverable["high_95"] / (1024 ** 3),
        }
    availability_payload = {
        "schema": "aic.phd2.availability-audit/v1",
        "generated_at_utc": utc_now(),
        "method": {
            "probe": "yt-dlp --simulate --dump-single-json (metadata only, no downloads)",
            "sampling": "two stratified random samples over (split x duration_bin), seeds 20260920/20260921, deduplicated",
            "probe_started_utc": args.probe_started_utc or None,
            "probe_stopped_reason": args.probe_stopped_reason or None,
        },
        "probe_records_total": len(all_records),
        "probe_records_analyzed": len(records),
        "probe_artifacts_excluded": len(timeout_records),
        "status_counts": status_counts,
        "available": len(available),
        "availability_rate": availability_rate,
        "confidence": {
            "method": "wilson 95%",
            "low": low,
            "high": high,
        },
        "total_unique_videos": total_unique,
        "estimated_recoverable_videos": estimated_recoverable,
        "estimate_type": "ESTIMATED" if len(records) < total_unique else "EXACT",
        "notes": [
            "PROBE_TIMEOUT and NETWORK_ERROR records are excluded from the availability denominator: they reflect probe-side throttling, not video status.",
            "CSV rows with video_duration == -1 are anomalous upstream records; in this sample they were almost all unavailable.",
        ],
    }
    size_payload = {
        "schema": "aic.phd2.size-estimate/v1",
        "generated_at_utc": utc_now(),
        "recoverable_videos_point": estimated_recoverable["point"],
        "recoverable_videos_ci95": [estimated_recoverable["low_95"], estimated_recoverable["high_95"]],
        "size_per_video": sizes,
        "storage_budgets": budgets,
        "notes": [
            "Sizes are best-video-stream <= cap plus best audio-only stream; progressive formats already include audio.",
            "filesize_approx or bitrate*duration used when exact filesize is unavailable.",
            "Total-size CI combines Wilson CI on recoverable count with the 95% CI of the per-video mean size (SE = stdev/sqrt(n)).",
        ],
    }
    args.availability_output.parent.mkdir(parents=True, exist_ok=True)
    args.availability_output.write_text(
        json.dumps(availability_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.size_output.write_text(json.dumps(size_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "probe_records": len(records),
                "status_counts": status_counts,
                "availability_rate": round(availability_rate, 4),
                "wilson_95": [round(low, 4), round(high, 4)],
                "estimated_recoverable": estimated_recoverable,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inventory = sub.add_parser("inventory")
    p_inventory.add_argument("--metadata-root", required=True, type=Path)
    p_inventory.add_argument("--output", required=True, type=Path)
    p_inventory.set_defaults(func=command_inventory)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--metadata-root", required=True, type=Path)
    p_sample.add_argument("--size", type=int, default=1200)
    p_sample.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p_sample.add_argument("--output", required=True, type=Path)
    p_sample.set_defaults(func=command_sample)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--sample", required=True, type=Path)
    p_probe.add_argument("--output-jsonl", required=True, type=Path)
    p_probe.add_argument("--log", required=True, type=Path)
    p_probe.add_argument("--limit", type=int, default=None)
    p_probe.add_argument("--sleep", type=float, default=0.4)
    p_probe.add_argument("--timeout", type=float, default=60.0)
    p_probe.add_argument("--rate-limit-abort", type=int, default=5)
    p_probe.set_defaults(func=command_probe)

    p_estimate = sub.add_parser("estimate")
    p_estimate.add_argument("--inventory", required=True, type=Path)
    p_estimate.add_argument("--probe-jsonl", required=True, type=Path)
    p_estimate.add_argument("--availability-output", required=True, type=Path)
    p_estimate.add_argument("--size-output", required=True, type=Path)
    p_estimate.add_argument("--probe-started-utc", default=None)
    p_estimate.add_argument("--probe-stopped-reason", default=None)
    p_estimate.set_defaults(func=command_estimate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
