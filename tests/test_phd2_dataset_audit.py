from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "experiments" / "stage7" / "audit_phd2_dataset.py"


def _load():
    spec = importlib.util.spec_from_file_location("aic_test_phd2_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    header = "youtubeId,start,duration,user_id,video_duration,is_last"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["youtubeId"],
                    row["start"],
                    row["duration"],
                    row["user_id"],
                    row["video_duration"],
                    row.get("is_last", "False"),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_classify_probe_error_buckets() -> None:
    assert audit.classify_probe_error("ERROR: Private video") == "PRIVATE"
    assert audit.classify_probe_error("ERROR: This video is unavailable") == "UNAVAILABLE"
    assert (
        audit.classify_probe_error("ERROR: Sign in to confirm your age")
        == "AGE_RESTRICTED"
    )
    assert audit.classify_probe_error("ERROR: Please sign in") == "SIGN_IN_REQUIRED"
    assert audit.classify_probe_error("HTTP Error 429: Too Many Requests") == "RATE_LIMITED"
    assert (
        audit.classify_probe_error("ConnectionResetError: Connection aborted")
        == "NETWORK_ERROR"
    )


def test_pick_format_size_prefers_highest_under_cap_with_audio() -> None:
    formats = [
        {"format_id": "v360", "height": 360, "vcodec": "avc1", "acodec": "none",
         "tbr": 500, "filesize": 10 * 1024 * 1024},
        {"format_id": "v720", "height": 720, "vcodec": "avc1", "acodec": "none",
         "tbr": 2000, "filesize": 40 * 1024 * 1024},
        {"format_id": "a128", "height": None, "vcodec": "none", "acodec": "mp4a",
         "abr": 128, "filesize": 2 * 1024 * 1024},
    ]
    picked = audit.pick_format_size(formats, 360, 100.0)
    assert picked is not None
    assert picked["height"] == 360
    assert picked["total_bytes"] == 12 * 1024 * 1024
    picked720 = audit.pick_format_size(formats, 720, 100.0)
    assert picked720 is not None and picked720["height"] == 720


def test_wilson_interval_known_values() -> None:
    low, high = audit.wilson_interval(50, 100)
    assert low < 0.5 < high
    assert abs((low + high) / 2 - 0.5) < 0.02
    low0, high0 = audit.wilson_interval(0, 100)
    assert low0 < 1e-12 and 0.0 < high0 < 0.05


def test_inventory_counts_annotations_and_unique_videos(tmp_path: Path) -> None:
    train = [
        {"youtubeId": "v1", "start": "0", "duration": "3", "user_id": "u1", "video_duration": "100"},
        {"youtubeId": "v1", "start": "10", "duration": "2", "user_id": "u1", "video_duration": "100"},
        {"youtubeId": "v2", "start": "0", "duration": "5", "user_id": "u2", "video_duration": "50"},
    ]
    test = [
        {"youtubeId": "v3", "start": "0", "duration": "4", "user_id": "u3", "video_duration": "200"},
    ]
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_csv(metadata / "training.csv", train)
    _write_csv(metadata / "testing.csv", test)
    output = tmp_path / "inventory.json"
    code = audit.main(
        ["inventory", "--metadata-root", str(metadata), "--output", str(output)]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["TRAIN"]["rows_annotations"] == 3
    assert payload["TRAIN"]["unique_videos"] == 2
    assert payload["TRAIN"]["unique_users"] == 2
    assert payload["COMBINED"]["unique_videos"] == 3
    assert payload["COMBINED"]["rows_annotations"] == 4
    assert payload["TEST"]["unique_videos"] == 1


def test_sample_is_deterministic_and_stratified(tmp_path: Path) -> None:
    rows = []
    for index in range(40):
        duration = 30 if index < 20 else 300
        rows.append(
            {
                "youtubeId": f"vid{index:03d}",
                "start": "0",
                "duration": "3",
                "user_id": f"u{index}",
                "video_duration": str(duration),
            }
        )
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_csv(metadata / "training.csv", rows)
    _write_csv(metadata / "testing.csv", [])
    first = tmp_path / "sample1.json"
    second = tmp_path / "sample2.json"
    audit.main(
        ["sample", "--metadata-root", str(metadata), "--size", "10", "--seed", "1",
         "--output", str(first)]
    )
    audit.main(
        ["sample", "--metadata-root", str(metadata), "--size", "10", "--seed", "1",
         "--output", str(second)]
    )
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first_payload["videos"] == second_payload["videos"]
    payload = first_payload
    assert payload["sampled"] == 10
    bins = {row["duration_bin"] for row in payload["videos"]}
    assert "0-1min" in bins and "3-10min" in bins


def test_estimate_excludes_probe_artifacts_and_computes_ci(tmp_path: Path) -> None:
    inventory = {
        "COMBINED": {"unique_videos": 1000},
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    probe_path = tmp_path / "probe.jsonl"
    records = []
    for index in range(50):
        records.append(
            {
                "youtubeId": f"v{index}",
                "status": "AVAILABLE",
                "resolutions": {
                    "360p": {"total_bytes": 10 * 1024 * 1024},
                },
            }
        )
    for index in range(50):
        records.append({"youtubeId": f"u{index}", "status": "UNAVAILABLE"})
    records.append({"youtubeId": "t1", "status": "UNKNOWN_ERROR", "error": "PROBE_TIMEOUT"})
    records.append(
        {
            "youtubeId": "t2",
            "status": "UNKNOWN_ERROR",
            "error": "ERROR: Unable to download API page: ConnectionResetError",
        }
    )
    probe_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    availability_out = tmp_path / "availability.json"
    size_out = tmp_path / "size.json"
    code = audit.main(
        [
            "estimate",
            "--inventory", str(inventory_path),
            "--probe-jsonl", str(probe_path),
            "--availability-output", str(availability_out),
            "--size-output", str(size_out),
        ]
    )
    assert code == 0
    payload = json.loads(availability_out.read_text(encoding="utf-8"))
    assert payload["probe_records_analyzed"] == 100
    assert payload["probe_artifacts_excluded"] == 2
    assert payload["status_counts"]["AVAILABLE"] == 50
    assert abs(payload["availability_rate"] - 0.5) < 1e-9
    assert payload["estimated_recoverable_videos"]["point"] == 500
    size_payload = json.loads(size_out.read_text(encoding="utf-8"))
    assert abs(size_payload["size_per_video"]["360p"]["bytes_per_video"]["mean"]
               - 10 * 1024 * 1024) < 1
