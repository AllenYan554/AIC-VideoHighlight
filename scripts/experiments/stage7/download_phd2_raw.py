#!/usr/bin/env python3
"""PHD2 99 GiB raw-video local downloader (metadata-first, resumable, rate-limit safe).

Subcommands:
    build-pool   TRAIN_ONLY candidate table + stratified user-diversified pool
    probe        metadata-only YouTube probe for pool videos (no media download)
    plan         download plan: 85/15 quality arms, duration-bin quotas, 99 GiB budget
    download     execute downloads (.part -> ffprobe validate -> sha256 -> atomic rename)
    summarize    summary JSON + failures CSV + versioned manifest

Hard rules implemented here:
    - only PHD2 TRAIN videos, and never a youtubeId present in TEST (TRAIN_ONLY)
    - metadata first, then format select, then download; never overwrite existing raw
    - per-video manifest updates immediately after every attempt (resume-safe)
    - request sleep >= 1.5s with jitter; exponential backoff on 429/ConnectionReset;
      abort after repeated IP-blocking signals; no proxies, no bypassing
    - total raw size hard cap 99 GiB with fill mode from 90 GiB
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
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
# default first-batch target mix by video count (user protocol; 30min+ excluded)
DEFAULT_BIN_RATIOS = {
    "0-1min": 0.20,
    "1-3min": 0.30,
    "3-10min": 0.40,
    "10-30min": 0.10,
    "30min+": 0.0,
}
ARM_RATIO = 0.15  # 15% of videos at <=720p, 85% at <=360p
GIB = 1024 ** 3
HARD_CAP_BYTES = 99 * GIB
FILL_MODE_BYTES = 90 * GIB
MAX_VIDEOS_PER_USER = 2
SEED = 20260920

FORMAT_SELECTORS = {
    "v360": (
        "bv*[height<=360][ext=mp4]/bv*[height<=360]/"
        "b[height<=360][ext=mp4]/b[height<=360]"
    ),
    "v720": (
        "bv*[height<=720][ext=mp4]/bv*[height<=720]/"
        "b[height<=720][ext=mp4]/b[height<=720]"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_system_proxy() -> str | None:
    """Read the Windows system proxy (same source browsers use).

    yt-dlp does not pick this up by itself, so without an explicit ``--proxy``
    it attempts a direct connection and fails behind a system-proxy setup.
    """

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = bool(winreg.QueryValueEx(key, "ProxyEnable")[0])
            server = str(winreg.QueryValueEx(key, "ProxyServer")[0] or "")
        if not enabled or not server:
            return None
        if "=" in server:  # e.g. "http=host:port;https=host:port"
            parts = {}
            for token in server.split(";"):
                if "=" in token:
                    scheme, value = token.split("=", 1)
                    parts[scheme.strip().lower()] = value.strip()
            server = parts.get("https") or parts.get("http") or next(iter(parts.values()), "")
        if not server:
            return None
        if not server.startswith(("http://", "https://", "socks4://", "socks5://")):
            server = "http://" + server
        return server
    except Exception:
        return None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def duration_bin_label(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    for label, low, high in DURATION_BINS:
        if low <= seconds < high:
            return label
    return "unknown"


def hash_key(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def build_train_only_table(metadata_root: Path) -> dict[str, dict[str, Any]]:
    """One record per TRAIN_ONLY youtubeId with users/highlights/duration."""

    train_rows = load_csv(metadata_root / "training.csv")
    test_rows = load_csv(metadata_root / "testing.csv")
    test_ids = {row["youtubeId"] for row in test_rows if row.get("youtubeId")}
    table: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        video_id = row.get("youtubeId", "").strip()
        if not video_id or video_id in test_ids:
            continue
        record = table.setdefault(
            video_id,
            {
                "youtubeId": video_id,
                "video_duration": to_float(row.get("video_duration")),
                "users": set(),
                "highlights": [],
            },
        )
        record["users"].add(str(row.get("user_id")))
        record["highlights"].append(
            {
                "start": to_float(row.get("start")),
                "duration": to_float(row.get("duration")),
                "user_id": str(row.get("user_id")),
                "is_last": row.get("is_last"),
            }
        )
    for record in table.values():
        record["users"] = sorted(record["users"])
        record["highlight_count"] = len(record["highlights"])
        record["highlight_total_s"] = sum(
            item["duration"] for item in record["highlights"] if item["duration"]
        )
        record["duration_bin"] = duration_bin_label(record["video_duration"])
    return table


def build_pool(
    table: dict[str, dict[str, Any]],
    *,
    pool_size: int,
    seed: int,
    bin_ratios: dict[str, float],
    max_per_user: int,
    reuse_available: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Stratified, user-diversified pool selection (deterministic)."""

    effective_size = max(1, int(round(pool_size)))
    needed = {
        label: int(round(effective_size * ratio))
        for label, ratio in bin_ratios.items()
        if ratio > 0
        and label in {entry[0] for entry in DURATION_BINS}
    }
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in table.values():
        label = record["duration_bin"]
        if label in needed:
            by_bin[label].append(record)
    user_counter: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def pick(record: dict[str, Any]) -> dict[str, Any]:
        primary_user = record["users"][0] if record["users"] else ""
        user_counter[primary_user] += 1
        selected_ids.add(record["youtubeId"])
        return {
            "youtubeId": record["youtubeId"],
            "user_ids": record["users"],
            "primary_user": primary_user,
            "video_duration": record["video_duration"],
            "duration_bin": record["duration_bin"],
            "highlight_count": record["highlight_count"],
            "highlight_total_s": round(record["highlight_total_s"], 3),
            "highlights": record["highlights"],
        }

    # pass 1: preferred bins with per-user cap
    for label in sorted(needed):
        candidates = sorted(
            by_bin[label], key=lambda item: hash_key(str(seed), item["youtubeId"])
        )
        taken = 0
        for record in candidates:
            if taken >= needed[label]:
                break
            if record["youtubeId"] in selected_ids:
                continue
            primary_user = record["users"][0] if record["users"] else ""
            if user_counter[primary_user] >= max_per_user:
                continue
            selected.append(pick(record))
            taken += 1
    # pass 2: progressively relax the per-user cap only when a bin is short.
    # real PHD2 bins have thousands of distinct users, so this rarely triggers.
    max_relaxed_cap = 20
    for cap in range(max_per_user + 1, max_relaxed_cap + 1):
        still_short = False
        for label in sorted(needed):
            target = needed[label]
            taken_now = sum(1 for row in selected if row["duration_bin"] == label)
            if taken_now >= target:
                continue
            still_short = True
            candidates = sorted(
                by_bin[label], key=lambda item: hash_key(str(seed), item["youtubeId"])
            )
            for record in candidates:
                if taken_now >= target:
                    break
                if record["youtubeId"] in selected_ids:
                    continue
                primary_user = record["users"][0] if record["users"] else ""
                if user_counter[primary_user] >= cap:
                    continue
                selected.append(pick(record))
                taken_now += 1
        if not still_short:
            break
    # carry size data from the feasibility probe when available (metadata cache reuse)
    if reuse_available:
        for row in selected:
            cached = reuse_available.get(row["youtubeId"])
            if cached:
                row["reused_probe"] = {
                    "duration_s": cached.get("duration_s"),
                    "resolutions": cached.get("resolutions"),
                }
    return selected


def command_build_pool(args: argparse.Namespace) -> int:
    table = build_train_only_table(args.metadata_root)
    reuse_available: dict[str, dict[str, Any]] = {}
    if args.reuse_probe and args.reuse_probe.is_file():
        for row in read_jsonl(args.reuse_probe):
            if row.get("status") == "AVAILABLE":
                reuse_available[str(row["youtubeId"])] = row
    pool = build_pool(
        table,
        pool_size=args.pool_size,
        seed=args.seed,
        bin_ratios=DEFAULT_BIN_RATIOS,
        max_per_user=MAX_VIDEOS_PER_USER,
        reuse_available=reuse_available,
    )
    user_counter = Counter(row["primary_user"] for row in pool)
    payload = {
        "schema": "aic.phd2.download-pool/v1",
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "train_only_videos": len(table),
        "pool_size": len(pool),
        "bin_targets": {
            label: int(round(args.pool_size * ratio))
            for label, ratio in DEFAULT_BIN_RATIOS.items()
            if ratio > 0
        },
        "bin_counts": dict(Counter(row["duration_bin"] for row in pool)),
        "unique_users": len(user_counter),
        "videos_per_user_max": max(user_counter.values()) if user_counter else 0,
        "reused_probe_records": sum(1 for row in pool if "reused_probe" in row),
        "videos": pool,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "train_only_videos": len(table),
                "pool_size": len(pool),
                "bin_counts": payload["bin_counts"],
                "unique_users": payload["unique_users"],
                "videos_per_user_max": payload["videos_per_user_max"],
                "reused_probe_records": payload["reused_probe_records"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_audit_module():
    import importlib.util

    module_path = Path(__file__).resolve().parent / "audit_phd2_dataset.py"
    spec = importlib.util.spec_from_file_location("aic_audit_phd2_for_download", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RATE_MARKERS = ("http error 429", "too many requests", "rate-limited", "rate limited")
NETWORK_MARKERS = (
    "connection aborted",
    "connectionreseterror",
    "connection reset",
    "unable to download api page",
    "remote end closed connection",
    "timed out",
)


def classify_download_error(text: str) -> tuple[str, bool]:
    """Return (category, is_temporary)."""

    lowered = text.lower()
    if any(marker in lowered for marker in RATE_MARKERS) or any(
        marker in lowered for marker in NETWORK_MARKERS
    ):
        return "RATE_LIMIT_OR_NETWORK", True
    if "confirm your age" in lowered or "age-restricted" in lowered:
        return "AGE_GATE", False
    if "sign in" in lowered or "login required" in lowered:
        return "SIGN_IN", False
    if "private video" in lowered:
        return "PRIVATE", False
    if "not available in your country" in lowered:
        return "REGION", False
    if "requested format is not available" in lowered or "no video formats" in lowered:
        return "FORMAT_NOT_FOUND", False
    if (
        "video is unavailable" in lowered
        or "video unavailable" in lowered
        or "video is not available" in lowered
        or "removed" in lowered
        or "terminated" in lowered
    ):
        return "UNAVAILABLE", False
    return "OTHER", False


def command_probe(args: argparse.Namespace) -> int:
    audit = load_audit_module()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    cache = audit.load_probe_cache(args.output_jsonl)
    pending = [
        row
        for row in pool["videos"]
        if str(row["youtubeId"]) not in cache and "reused_probe" not in row
    ]
    reused_appended = 0
    for row in pool["videos"]:
        if "reused_probe" in row and str(row["youtubeId"]) not in cache:
            cached = row["reused_probe"]
            record = {
                "youtubeId": row["youtubeId"],
                "split": "TRAIN",
                "duration_bin": row["duration_bin"],
                "video_duration_csv": row["video_duration"],
                "checked_at_utc": utc_now(),
                "status": "AVAILABLE" if cached.get("resolutions") else "UNKNOWN_ERROR",
                "duration_s": cached.get("duration_s"),
                "title": "",
                "elapsed_s": None,
                "resolutions": cached.get("resolutions"),
                "source": "reused_feasibility_probe",
            }
            with args.output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            reused_appended += 1
    if args.limit:
        pending = pending[: args.limit]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", encoding="utf-8")
    started = time.perf_counter()
    consecutive_blocks = 0
    completed = 0
    rng = random.Random(args.seed)
    for position, row in enumerate(pending, start=1):
        video_id = str(row["youtubeId"])
        if consecutive_blocks >= args.block_abort:
            message = f"ABORT: {consecutive_blocks} consecutive blocking signals; resume later"
            print(message, flush=True)
            log.write(f"[{utc_now()}] {message}\n")
            break
        result = audit.probe_one(video_id, row.get("video_duration"), args.timeout)
        status = str(result.get("status"))
        record = {
            "youtubeId": video_id,
            "split": "TRAIN",
            "duration_bin": row.get("duration_bin"),
            "video_duration_csv": row.get("video_duration"),
            "checked_at_utc": utc_now(),
            **result,
        }
        if status in ("NETWORK_ERROR", "RATE_LIMITED") or result.get("error") == "PROBE_TIMEOUT":
            consecutive_blocks += 1
            backoff = min(30.0 * (2 ** (consecutive_blocks - 1)), 300.0)
            log.write(
                f"[{utc_now()}] BLOCK signal {status} id={video_id} "
                f"consecutive={consecutive_blocks} backoff={backoff:.0f}s\n"
            )
            if consecutive_blocks < args.block_abort:
                time.sleep(backoff)
            continue
        consecutive_blocks = 0
        with args.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed += 1
        if position % 25 == 0 or position == len(pending):
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            eta = (len(pending) - position) * rate / 60.0
            summary = {
                "done": position,
                "total": len(pending),
                "new": completed,
                "rate_s": round(rate, 2),
                "eta_min": round(eta, 1),
            }
            print(f"progress {json.dumps(summary)}", flush=True)
            log.write(f"[{utc_now()}] progress {json.dumps(summary)}\n")
        time.sleep(args.sleep + rng.random() * args.jitter)
    log.close()
    print(
        f"probe finished: new={completed} reused_appended={reused_appended} "
        f"cache_total={len(audit.load_probe_cache(args.output_jsonl))}"
    )
    return 0


RESOLUTION_ARM_KEY = {"v360": "360p", "v720": "720p"}


def command_plan(args: argparse.Namespace) -> int:
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    probe = (
        {str(row["youtubeId"]): row for row in read_jsonl(args.probe_jsonl)}
        if args.probe_jsonl and args.probe_jsonl.is_file()
        else {}
    )
    rng = random.Random(args.seed)
    per_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool["videos"]:
        per_bin[row["duration_bin"]].append(row)
    for rows in per_bin.values():
        rows.sort(key=lambda item: hash_key(str(args.seed), item["youtubeId"]))
        rng.shuffle(rows)
    ordered: list[dict[str, Any]] = []
    ratio_order = [label for label, ratio in DEFAULT_BIN_RATIOS.items() if ratio > 0]
    pointers = {label: 0 for label in per_bin}
    made_progress = True
    while made_progress:
        made_progress = False
        for label in ratio_order:
            pointer = pointers.get(label, 0)
            rows = per_bin.get(label, [])
            if pointer < len(rows):
                ordered.append(rows[pointer])
                pointers[label] = pointer + 1
                made_progress = True
    total_planned = len(ordered)
    n720 = int(round(total_planned * ARM_RATIO))
    plan_videos: list[dict[str, Any]] = []
    assigned_720 = 0
    for index, row in enumerate(ordered):
        arm = "v360"
        if assigned_720 < n720 and ((index + 1) * n720 >= (assigned_720 + 1) * total_planned):
            arm = "v720"
            assigned_720 += 1
        record = probe.get(str(row["youtubeId"])) or {}
        sizes = record.get("resolutions") or {}
        entry = {
            "youtubeId": row["youtubeId"],
            "user_ids": row["user_ids"],
            "primary_user": row["primary_user"],
            "duration_bin": row["duration_bin"],
            "video_duration": row["video_duration"],
            "highlight_count": row["highlight_count"],
            "highlight_total_s": row["highlight_total_s"],
            "quality_arm": arm,
            "probe": {
                "duration_s": record.get("duration_s"),
                "360p": sizes.get("360p"),
                "720p": sizes.get("720p"),
            },
            "estimated_bytes": (sizes.get(RESOLUTION_ARM_KEY[arm]) or {}).get("total_bytes") or 0,
        }
        plan_videos.append(entry)
    est_total = sum(row["estimated_bytes"] for row in plan_videos)
    payload = {
        "schema": "aic.phd2.download-plan/v1",
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "budget_bytes": HARD_CAP_BYTES,
        "fill_mode_bytes": FILL_MODE_BYTES,
        "arm_ratio_target": ARM_RATIO,
        "arm_counts": dict(Counter(row["quality_arm"] for row in plan_videos)),
        "bin_counts": dict(Counter(row["duration_bin"] for row in plan_videos)),
        "planned_videos": len(plan_videos),
        "estimated_total_bytes": est_total,
        "estimated_total_gib": est_total / GIB,
        "videos": plan_videos,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "planned_videos": len(plan_videos),
                "arm_counts": payload["arm_counts"],
                "bin_counts": payload["bin_counts"],
                "estimated_total_gib": round(payload["estimated_total_gib"], 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def ffprobe_media(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or "ffprobe failed").strip()[:200])
    streams = json.loads(completed.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe: no video stream")
    return streams[0]


def duration_is_plausible(csv_seconds: float | None, actual_seconds: float | None) -> bool:
    if csv_seconds is None or actual_seconds is None or csv_seconds <= 0:
        return True
    return abs(actual_seconds - csv_seconds) <= max(5.0, 0.10 * csv_seconds)


def command_download(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest_path = args.manifest
    manifest_rows = read_jsonl(manifest_path)
    existing = {str(row["youtubeId"]): row for row in manifest_rows}
    attempt_counts: Counter[str] = Counter(str(row["youtubeId"]) for row in manifest_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root = args.raw_root
    incoming = args.incoming
    incoming.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", encoding="utf-8")
    rng = random.Random(args.seed)

    def current_raw_bytes() -> int:
        total = 0
        for row in read_jsonl(manifest_path):
            if row.get("download_status") == "SUCCESS":
                total += int(row.get("actual_bytes") or 0)
        return total

    if args.proxy is None:
        proxy = detect_system_proxy()
    elif args.proxy.lower() in ("", "none", "direct"):
        proxy = None
    else:
        proxy = args.proxy
    processed = 0
    successes = 0
    failures = 0
    successes_360 = 0
    successes_720 = 0
    consecutive_blocks = 0
    last_rest_at = 0
    started = time.perf_counter()

    def eta_hours() -> float | None:
        if successes == 0 or processed == 0:
            return None
        raw_now = current_raw_bytes()
        avg_bytes = raw_now / successes
        if avg_bytes <= 0:
            return None
        remaining_bytes = max(0, HARD_CAP_BYTES - raw_now)
        remaining_videos = remaining_bytes / avg_bytes
        avg_sec = (time.perf_counter() - started) / processed
        return remaining_videos * avg_sec / 3600.0

    def progress_line() -> str:
        raw_now = current_raw_bytes()
        ratio = f"{successes_360}:{successes_720}"
        eta = eta_hours()
        eta_text = f"{eta:.1f}h" if eta is not None else "n/a"
        return (
            f"success={successes} fail={failures} "
            f"raw={raw_now / GIB:.2f}/99.00 GiB "
            f"({raw_now / HARD_CAP_BYTES * 100:.1f}%) "
            f"360p:720p={ratio} eta={eta_text}"
        )

    success_before = sum(
        1 for row in manifest_rows if row.get("download_status") == "SUCCESS"
    )
    print(
        f"download start: plan={len(plan['videos'])} already_success={success_before} "
        f"raw_before={current_raw_bytes() / GIB:.2f} GiB proxy={proxy or 'DIRECT'}",
        flush=True,
    )
    log.write(f"[{utc_now()}] download start proxy={proxy or 'DIRECT'}\n")
    for entry in plan["videos"]:
        if args.limit and processed >= args.limit:
            break
        video_id = str(entry["youtubeId"])
        previous = existing.get(video_id)
        if previous and previous.get("download_status") == "SUCCESS":
            continue
        if previous and previous.get("download_status", "").startswith("FAILED"):
            status = str(previous.get("download_status"))
            max_attempts = 3 if status == "FAILED_TEMP" else 1
            if attempt_counts.get(video_id, 0) >= max_attempts and not args.retry_failed:
                continue
        arm = entry["quality_arm"]
        estimated = int(entry.get("estimated_bytes") or 0)
        raw_now = current_raw_bytes()
        if raw_now >= FILL_MODE_BYTES and not estimated:
            # fill mode: metadata pre-check to keep the hard cap
            audit = load_audit_module()
            pre = audit.probe_one(video_id, entry.get("video_duration"), 60.0)
            pre_status = str(pre.get("status"))
            if pre_status == "AVAILABLE":
                sizes = pre.get("resolutions") or {}
                estimated = (sizes.get(RESOLUTION_ARM_KEY[arm]) or {}).get("total_bytes") or 0
            elif pre_status in ("NETWORK_ERROR", "RATE_LIMITED") or pre.get("error") == "PROBE_TIMEOUT":
                consecutive_blocks += 1
                record = {
                    "youtubeId": video_id,
                    "download_status": "FAILED_TEMP",
                    "failure_reason": "RATE_LIMIT_OR_NETWORK",
                    "quality_arm": arm,
                    "error_excerpt": str(pre.get("error"))[:200],
                    "checked_at_utc": utc_now(),
                }
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing[video_id] = record
                log.write(
                    f"[{utc_now()}] FILL_MODE_BLOCK {pre_status} {video_id} "
                    f"consecutive={consecutive_blocks}\n"
                )
                if consecutive_blocks >= args.block_abort:
                    print("ABORT: repeated IP blocking signals during fill mode", flush=True)
                    break
                time.sleep(min(30.0 * (2 ** (consecutive_blocks - 1)), 300.0))
                continue
            else:
                category, temporary = classify_download_error(str(pre.get("error") or ""))
                if not temporary and pre_status in ("UNAVAILABLE", "PRIVATE", "AGE_RESTRICTED", "SIGN_IN_REQUIRED", "REGION_BLOCKED"):
                    category = pre_status
                record = {
                    "youtubeId": video_id,
                    "download_status": "FAILED_TEMP" if temporary else "FAILED_PERMANENT",
                    "failure_reason": category,
                    "quality_arm": arm,
                    "error_excerpt": str(pre.get("error"))[:200],
                    "checked_at_utc": utc_now(),
                }
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing[video_id] = record
                failures += 1
                processed += 1
                continue
        if estimated and raw_now + estimated > HARD_CAP_BYTES:
            record = {
                "youtubeId": video_id,
                "download_status": "SKIPPED_OVER_BUDGET",
                "quality_arm": arm,
                "estimated_bytes": estimated,
                "raw_bytes_before": raw_now,
                "checked_at_utc": utc_now(),
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing[video_id] = record
            log.write(f"[{utc_now()}] SKIP_OVER_BUDGET {video_id} est={estimated}\n")
            continue
        if consecutive_blocks >= args.block_abort:
            print("ABORT: repeated IP blocking signals; resume later", flush=True)
            break
        target_dir = raw_root / ("360p" if arm == "v360" else "720p")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{video_id}.mp4"
        if target_path.is_file():
            record = existing.get(video_id) or {}
            log.write(f"[{utc_now()}] NOTE existing file without manifest SUCCESS {video_id}\n")

        time.sleep(args.sleep + rng.random() * args.jitter)
        out_template = str(incoming / f"{video_id}.%(ext)s")
        command = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--no-mtime",
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
        ]
        if proxy:
            command += ["--proxy", proxy]
        command += [
            "-f",
            FORMAT_SELECTORS[arm],
            "-o",
            out_template,
            "--print",
            "after_move:filepath",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        t0 = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            completed = None
        elapsed = round(time.perf_counter() - t0, 2)
        record: dict[str, Any] = {
            "youtubeId": video_id,
            "split": "TRAIN",
            "user_ids": entry["user_ids"],
            "primary_user": entry["primary_user"],
            "duration_bin": entry["duration_bin"],
            "video_duration_csv": entry["video_duration"],
            "highlight_count": entry["highlight_count"],
            "highlight_total_s": entry["highlight_total_s"],
            "quality_arm": arm,
            "selected_height_cap": 360 if arm == "v360" else 720,
            "estimated_bytes": estimated,
            "proxy": proxy or "DIRECT",
            "attempted_at_utc": utc_now(),
            "elapsed_s": elapsed,
        }
        if completed is None:
            record.update(
                {
                    "download_status": "FAILED_TEMP",
                    "failure_reason": "DOWNLOAD_TIMEOUT",
                }
            )
            failures += 1
            consecutive_blocks += 1
        else:
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            paths = [
                line.strip()
                for line in stdout.splitlines()
                if line.strip().lower().endswith((".mp4", ".webm", ".mkv", ".m4a", ".mov"))
            ]
            if completed.returncode != 0 or not paths:
                category, temporary = classify_download_error(f"{stdout}\n{stderr}")
                record.update(
                    {
                        "download_status": "FAILED_TEMP" if temporary else "FAILED_PERMANENT",
                        "failure_reason": category,
                        "error_excerpt": (stderr or stdout).strip().splitlines()[-1][:200]
                        if (stderr or stdout).strip()
                        else "NO_OUTPUT",
                    }
                )
                failures += 1
                for leftover in incoming.glob(f"{video_id}.*"):
                    leftover.unlink(missing_ok=True)
                if temporary:
                    consecutive_blocks += 1
                    backoff = min(30.0 * (2 ** (consecutive_blocks - 1)), 300.0)
                    log.write(
                        f"[{utc_now()}] DOWNLOAD_BLOCK {category} {video_id} "
                        f"consecutive={consecutive_blocks} backoff={backoff:.0f}s\n"
                    )
                    time.sleep(backoff)
                else:
                    consecutive_blocks = 0
            else:
                downloaded = Path(paths[-1])
                try:
                    probe_info = ffprobe_media(downloaded)
                except RuntimeError as error:
                    record.update(
                        {
                            "download_status": "FAILED_PERMANENT",
                            "failure_reason": "VALIDATION_FAILED",
                            "error_excerpt": str(error)[:200],
                        }
                    )
                    failures += 1
                    downloaded.unlink(missing_ok=True)
                else:
                    actual_duration = to_float(probe_info.get("duration"))
                    actual_height = probe_info.get("height")
                    if not duration_is_plausible(entry["video_duration"], actual_duration):
                        record.update(
                            {
                                "download_status": "FAILED_PERMANENT",
                                "failure_reason": "DURATION_MISMATCH",
                                "ffprobe": probe_info,
                            }
                        )
                        failures += 1
                        downloaded.unlink(missing_ok=True)
                    elif actual_height is not None and int(actual_height) > record["selected_height_cap"]:
                        record.update(
                            {
                                "download_status": "FAILED_PERMANENT",
                                "failure_reason": "HEIGHT_EXCEEDS_CAP",
                                "ffprobe": probe_info,
                            }
                        )
                        failures += 1
                        downloaded.unlink(missing_ok=True)
                    else:
                        actual_bytes = downloaded.stat().st_size
                        digest = sha256_file(downloaded)
                        if target_path.exists():
                            target_path.unlink()
                        downloaded.replace(target_path)
                        record.update(
                            {
                                "download_status": "SUCCESS",
                                "path": str(target_path),
                                "relative_path": f"raw/{'360p' if arm == 'v360' else '720p'}/{video_id}.mp4",
                                "actual_bytes": actual_bytes,
                                "actual_duration_s": actual_duration,
                                "actual_height": int(actual_height) if actual_height else None,
                                "vcodec": probe_info.get("codec_name"),
                                "nb_frames": to_float(probe_info.get("nb_frames")),
                                "sha256": digest,
                                "audio_required": False,
                                "ffprobe_validation": "PASS",
                                "downloaded_at_utc": utc_now(),
                            }
                        )
                        successes += 1
                        if arm == "v360":
                            successes_360 += 1
                        else:
                            successes_720 += 1
                        consecutive_blocks = 0
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing[video_id] = record
        processed += 1
        if processed % 5 == 0:
            print(f"progress {progress_line()}", flush=True)
            log.write(f"[{utc_now()}] progress {progress_line()}\n")
        if (
            successes > 0
            and successes % args.rest_every == 0
            and successes != last_rest_at
            and not (args.limit and processed >= args.limit)
        ):
            last_rest_at = successes
            rest_seconds = rng.uniform(args.rest_min, args.rest_max)
            print(
                f"resting {rest_seconds / 60:.1f} min after {successes} successes "
                f"(anti-throttle pause) ...",
                flush=True,
            )
            log.write(f"[{utc_now()}] rest {rest_seconds:.0f}s after {successes} successes\n")
            time.sleep(rest_seconds)
    log.close()
    print(
        json.dumps(
            {
                "processed": processed,
                "successes": successes,
                "failures": failures,
                "successes_360p": successes_360,
                "successes_720p": successes_720,
                "raw_bytes": current_raw_bytes(),
                "raw_gib": round(current_raw_bytes() / GIB, 2),
                "final_line": progress_line(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = read_jsonl(args.manifest)
    latest: dict[str, dict[str, Any]] = {}
    for row in manifest:
        latest[str(row["youtubeId"])] = row
    rows = list(latest.values())
    success = [row for row in rows if row.get("download_status") == "SUCCESS"]
    failures = [row for row in rows if str(row.get("download_status", "")).startswith("FAILED")]
    skipped = [row for row in rows if row.get("download_status") == "SKIPPED_OVER_BUDGET"]
    total_bytes = sum(int(row.get("actual_bytes") or 0) for row in success)
    plan_by_id = {str(row["youtubeId"]): row for row in plan["videos"]}
    arm_counts = Counter(row.get("quality_arm") for row in success)
    bin_counts = Counter((plan_by_id.get(str(row["youtubeId"])) or {}).get("duration_bin") for row in success)
    users = Counter()
    multi = 0
    for row in success:
        plan_row = plan_by_id.get(str(row["youtubeId"])) or {}
        for user in plan_row.get("user_ids", []):
            users[user] += 1
        if (plan_row.get("highlight_count") or 0) >= 2:
            multi += 1
    failure_reasons = Counter(str(row.get("failure_reason")) for row in failures)
    summary = {
        "schema": "aic.phd2.download-summary/v1",
        "generated_at_utc": utc_now(),
        "downloads_success": len(success),
        "downloads_failed": len(failures),
        "skipped_over_budget": len(skipped),
        "raw_total_bytes": total_bytes,
        "raw_total_gib": total_bytes / GIB,
        "arm_counts": dict(arm_counts),
        "arm_fractions": {
            key: value / len(success) if success else 0.0 for key, value in arm_counts.items()
        },
        "duration_bin_counts": dict(bin_counts),
        "unique_users_selected": len(users),
        "videos_per_user_distribution": dict(Counter(users.values())),
        "single_highlight_videos": len(success) - multi,
        "multi_highlight_videos": multi,
        "failure_reasons": dict(failure_reasons),
        "failure_details": [
            {
                "youtubeId": row["youtubeId"],
                "failure_reason": row.get("failure_reason"),
                "error_excerpt": (row.get("error_excerpt") or "")[:160],
            }
            for row in failures
        ],
        "plan_planned_videos": len(plan["videos"]),
    }
    write_json(args.output_summary, summary)
    with args.output_failures_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["youtubeId", "failure_reason", "download_status", "error_excerpt"])
        for row in failures:
            writer.writerow(
                [
                    row["youtubeId"],
                    row.get("failure_reason"),
                    row.get("download_status"),
                    (row.get("error_excerpt") or "")[:200],
                ]
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2600])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pool = sub.add_parser("build-pool")
    p_pool.add_argument("--metadata-root", required=True, type=Path)
    p_pool.add_argument("--pool-size", type=int, default=5000)
    p_pool.add_argument("--seed", type=int, default=SEED)
    p_pool.add_argument("--reuse-probe", type=Path, default=None)
    p_pool.add_argument("--output", required=True, type=Path)
    p_pool.set_defaults(func=command_build_pool)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--pool", required=True, type=Path)
    p_probe.add_argument("--output-jsonl", required=True, type=Path)
    p_probe.add_argument("--log", required=True, type=Path)
    p_probe.add_argument("--limit", type=int, default=None)
    p_probe.add_argument("--sleep", type=float, default=1.5)
    p_probe.add_argument("--jitter", type=float, default=0.8)
    p_probe.add_argument("--timeout", type=float, default=60.0)
    p_probe.add_argument("--block-abort", type=int, default=6)
    p_probe.add_argument("--seed", type=int, default=SEED)
    p_probe.set_defaults(func=command_probe)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--pool", required=True, type=Path)
    p_plan.add_argument("--probe-jsonl", type=Path, default=None)
    p_plan.add_argument("--output", required=True, type=Path)
    p_plan.add_argument("--seed", type=int, default=SEED)
    p_plan.set_defaults(func=command_plan)

    p_download = sub.add_parser("download")
    p_download.add_argument("--plan", required=True, type=Path)
    p_download.add_argument("--raw-root", required=True, type=Path)
    p_download.add_argument("--incoming", required=True, type=Path)
    p_download.add_argument("--manifest", required=True, type=Path)
    p_download.add_argument("--log", required=True, type=Path)
    p_download.add_argument("--limit", type=int, default=None)
    p_download.add_argument("--sleep", type=float, default=8.0)
    p_download.add_argument("--jitter", type=float, default=4.0)
    p_download.add_argument("--timeout", type=float, default=900.0)
    p_download.add_argument("--block-abort", type=int, default=5)
    p_download.add_argument("--rest-every", type=int, default=50)
    p_download.add_argument("--rest-min", type=float, default=180.0)
    p_download.add_argument("--rest-max", type=float, default=300.0)
    p_download.add_argument(
        "--proxy",
        default=None,
        help="proxy URL; 'none' forces direct; default auto-detects the Windows system proxy",
    )
    p_download.add_argument("--retry-failed", action="store_true")
    p_download.add_argument("--seed", type=int, default=SEED)
    p_download.set_defaults(func=command_download)

    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--plan", required=True, type=Path)
    p_summary.add_argument("--manifest", required=True, type=Path)
    p_summary.add_argument("--output-summary", required=True, type=Path)
    p_summary.add_argument("--output-failures-csv", required=True, type=Path)
    p_summary.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
