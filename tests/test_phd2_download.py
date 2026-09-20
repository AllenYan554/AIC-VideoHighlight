from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "experiments" / "stage7" / "download_phd2_raw.py"


def _load():
    spec = importlib.util.spec_from_file_location("aic_test_phd2_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


downloader = _load()


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


def test_format_selectors_respect_height_caps() -> None:
    assert "height<=360" in downloader.FORMAT_SELECTORS["v360"]
    assert "height<=720" in downloader.FORMAT_SELECTORS["v720"]
    assert "1080" not in downloader.FORMAT_SELECTORS["v720"]
    assert downloader.HARD_CAP_BYTES == 99 * 1024 ** 3
    assert downloader.FILL_MODE_BYTES == 90 * 1024 ** 3


def test_classify_download_error_categories() -> None:
    assert downloader.classify_download_error("Private video") == ("PRIVATE", False)
    assert downloader.classify_download_error("This video is unavailable") == (
        "UNAVAILABLE",
        False,
    )
    assert downloader.classify_download_error("Sign in to confirm your age") == (
        "AGE_GATE",
        False,
    )
    assert downloader.classify_download_error("Please sign in") == ("SIGN_IN", False)
    assert downloader.classify_download_error(
        "Requested format is not available"
    ) == ("FORMAT_NOT_FOUND", False)
    category, temporary = downloader.classify_download_error(
        "ConnectionResetError: Connection aborted"
    )
    assert category == "RATE_LIMIT_OR_NETWORK" and temporary is True
    category, temporary = downloader.classify_download_error("HTTP Error 429")
    assert category == "RATE_LIMIT_OR_NETWORK" and temporary is True


def test_duration_plausibility() -> None:
    assert downloader.duration_is_plausible(100.0, 101.0)
    assert downloader.duration_is_plausible(30.0, 35.0)  # within 5s floor
    assert not downloader.duration_is_plausible(100.0, 130.0)
    assert downloader.duration_is_plausible(None, 100.0)


def test_train_only_table_excludes_test_overlap(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    train_rows = [
        {"youtubeId": "keep-1", "start": "0", "duration": "3", "user_id": "u1", "video_duration": "120"},
        {"youtubeId": "overlap", "start": "0", "duration": "3", "user_id": "u2", "video_duration": "90"},
        {"youtubeId": "keep-2", "start": "0", "duration": "3", "user_id": "u1", "video_duration": "4000"},
    ]
    test_rows = [
        {"youtubeId": "overlap", "start": "0", "duration": "3", "user_id": "u9", "video_duration": "90"},
    ]
    _write_csv(metadata / "training.csv", train_rows)
    _write_csv(metadata / "testing.csv", test_rows)
    table = downloader.build_train_only_table(metadata)
    assert set(table) == {"keep-1", "keep-2"}
    assert table["keep-2"]["duration_bin"] == "30min+"


def test_build_pool_user_dedup_and_bin_quotas() -> None:
    table = {}
    for index in range(200):
        vid = f"a{index:03d}"
        table[vid] = {
            "youtubeId": vid,
            "video_duration": 30.0,
            "users": [f"user-a{index}"],
            "highlights": [],
            "highlight_count": 1,
            "highlight_total_s": 3.0,
            "duration_bin": "0-1min",
        }
    for index in range(400):
        vid = f"b{index:03d}"
        table[vid] = {
            "youtubeId": vid,
            "video_duration": 120.0,
            "users": [f"user-b{index}"],
            "highlights": [],
            "highlight_count": 1,
            "highlight_total_s": 3.0,
            "duration_bin": "1-3min",
        }
    pool = downloader.build_pool(
        table,
        pool_size=300,
        seed=20260920,
        bin_ratios={"0-1min": 0.5, "1-3min": 0.5},
        max_per_user=2,
    )
    from collections import Counter

    by_user = Counter(row["primary_user"] for row in pool)
    assert max(by_user.values()) <= 2
    by_bin = Counter(row["duration_bin"] for row in pool)
    assert by_bin["0-1min"] == 150 and by_bin["1-3min"] == 150
    assert len(by_user) == 300
    assert len({row["youtubeId"] for row in pool}) == len(pool)


def test_build_pool_falls_back_when_users_are_insufficient() -> None:
    # a bin owned by very few users must still fill its quota via cap relaxation
    table = {}
    for index in range(30):
        vid = f"solo{index:03d}"
        table[vid] = {
            "youtubeId": vid,
            "video_duration": 30.0,
            "users": ["user-solo"] if index < 10 else [f"user-{index}"],
            "highlights": [],
            "highlight_count": 1,
            "highlight_total_s": 3.0,
            "duration_bin": "0-1min",
        }
    pool = downloader.build_pool(
        table,
        pool_size=20,
        seed=1,
        bin_ratios={"0-1min": 1.0},
        max_per_user=2,
    )
    assert len(pool) == 20
    from collections import Counter

    by_user = Counter(row["primary_user"] for row in pool)
    assert by_user["user-solo"] >= 2  # cap had to be relaxed to reach the quota


def test_build_pool_deterministic() -> None:
    table = {}
    for index in range(200):
        vid = f"v{index:03d}"
        table[vid] = {
            "youtubeId": vid,
            "video_duration": 100.0,
            "users": [f"u{index % 80}"],
            "highlights": [],
            "highlight_count": 1,
            "highlight_total_s": 3.0,
            "duration_bin": "1-3min",
        }
    ratios = {"1-3min": 1.0}
    first = downloader.build_pool(table, pool_size=90, seed=7, bin_ratios=ratios, max_per_user=2)
    second = downloader.build_pool(table, pool_size=90, seed=7, bin_ratios=ratios, max_per_user=2)
    assert [row["youtubeId"] for row in first] == [row["youtubeId"] for row in second]
