#!/usr/bin/env python3
"""Batch-sync materialized FTNet safetensors from AutoDL to the Windows warehouse.

The sync is content-verified: every downloaded file's SHA-256 is compared with
the remote SHA-256 before it counts as synced.  Existing local files with the
same size are re-verified instead of re-downloaded.  Run it repeatedly during
production; it is idempotent and only transfers missing/changed files.

Usage:
  python scripts/experiments/stage7/sync_materialized.py \
      --remote-host autodl-stage1 \
      --remote-root /root/autodl-tmp/derived/VHiCraFTNet/youtube_highlights_ftnet \
      --local-root  E:/ResearchData/derived/VHiCraFTNet/youtube_highlights_ftnet \
      --batch-size 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SPLITS = ("train", "validation", "calibration")
LIGHTWEIGHT_DIRS = ("manifests", "normalization", "audits")


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_listing(host: str, root: str) -> dict[str, int]:
    command = f"find {root} -name '*.safetensors' -printf '%P %s\\n'"
    output = _run(["ssh", "-o", "BatchMode=yes", host, command]).stdout
    listing: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        relative, size = line.rsplit(" ", 1)
        listing[relative] = int(size)
    return listing


def remote_hashes(host: str, root: str, relatives: list[str]) -> dict[str, str]:
    if not relatives:
        return {}
    quoted = " ".join(f"'{item}'" for item in relatives)
    command = f"cd {root} && sha256sum {quoted}"
    output = _run(["ssh", "-o", "BatchMode=yes", host, command]).stdout
    hashes: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        hashes[name] = digest
    return hashes


def download_batch(host: str, remote_root: str, local_root: str, relatives: list[str]) -> None:
    Path(local_root).mkdir(parents=True, exist_ok=True)
    quoted = " ".join(f"'{item}'" for item in relatives)
    remote = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host, f"cd {remote_root} && tar -cf - {quoted}"],
        stdout=subprocess.PIPE,
    )
    assert remote.stdout is not None
    local = subprocess.Popen(
        ["tar", "-C", str(local_root), "-xf", "-"],
        stdin=remote.stdout,
    )
    remote.stdout.close()
    local.communicate()
    if remote.wait() != 0:
        raise RuntimeError("remote tar failed")
    if local.returncode != 0:
        raise RuntimeError("local tar failed")


def sync_metadata(host: str, remote_root: str, local_root: str) -> list[str]:
    synced = []
    for name in LIGHTWEIGHT_DIRS:
        try:
            command = f"cd {remote_root} && tar -cf - {name} 2>/dev/null"
            process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, command], stdout=subprocess.PIPE)
            assert process.stdout is not None
            extract = subprocess.Popen(["tar", "-C", str(local_root), "-xf", "-"], stdin=process.stdout)
            process.stdout.close()
            extract.communicate()
            if process.wait() == 0 and extract.returncode == 0:
                synced.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[sync] metadata {name} failed: {exc}", file=sys.stderr)
    return synced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--metadata", action="store_true", help="also sync manifests/normalization/audits")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args(argv)

    local_root = Path(args.local_root)
    listing = remote_listing(args.remote_host, args.remote_root)
    wanted = {rel: size for rel, size in listing.items() if rel.split("/", 1)[0] in set(args.splits)}
    if not wanted:
        print(json.dumps({"status": "NO_REMOTE_FILES", "remote_root": args.remote_root}))
        return 0

    missing: list[str] = []
    for relative, size in sorted(wanted.items()):
        local_path = local_root / relative
        if local_path.is_file() and local_path.stat().st_size == size:
            continue
        missing.append(relative)

    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        print(f"[sync] downloading {len(batch)} files (batch {start // args.batch_size + 1}) ...", flush=True)
        download_batch(args.remote_host, args.remote_root, str(local_root), batch)

    verify_set = sorted(wanted) if missing else sorted(wanted)
    remote_digests = remote_hashes(args.remote_host, args.remote_root, verify_set)
    mismatches = []
    verified = 0
    for relative in verify_set:
        local_path = local_root / relative
        if not local_path.is_file():
            mismatches.append({"file": relative, "error": "missing local"})
            continue
        local_digest = sha256_file(local_path)
        remote_digest = remote_digests.get(relative)
        if remote_digest is None or local_digest != remote_digest:
            mismatches.append(
                {"file": relative, "local": local_digest, "remote": remote_digest}
            )
        else:
            verified += 1

    metadata_synced = sync_metadata(args.remote_host, args.remote_root, str(local_root)) if args.metadata else []
    summary = {
        "status": "PASS" if not mismatches else "FAIL",
        "remote_files": len(wanted),
        "downloaded": len(missing),
        "verified": verified,
        "mismatches": mismatches[:10],
        "metadata_synced": metadata_synced,
        "local_root": str(local_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not mismatches else 3


if __name__ == "__main__":
    raise SystemExit(main())
