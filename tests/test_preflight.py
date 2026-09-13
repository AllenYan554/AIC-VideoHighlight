"""Preflight tests: readiness checks on synthetic inputs (no model, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.runtime.orchestrator import QWEN_REVISION, RTDETR_REVISION  # noqa: E402
from aic_video_highlight.runtime.paths import EnvironmentPaths  # noqa: E402
from aic_video_highlight.runtime.preflight import run_preflight  # noqa: E402


def _environment(tmp_path: Path) -> EnvironmentPaths:
    env_root = tmp_path / "env"
    outputs = env_root / "outputs"
    datasets = env_root / "datasets"
    tests = datasets / "tests"
    models = env_root / "models"
    tests.mkdir(parents=True)
    for name in ("0.mp4", "1.mp4", "2.mp4"):
        (tests / name).write_bytes(b"x")
    for repo_id, revision in (
        ("Qwen3.5-4B", QWEN_REVISION),
        ("PekingU-rtdetr_r50vd", RTDETR_REVISION),
    ):
        trees = models / repo_id / ".cache" / "huggingface" / "trees"
        trees.mkdir(parents=True)
        (trees / f"{revision}.json").write_text("{}", encoding="utf-8")
    return EnvironmentPaths(
        name="preflight_test",
        repo=REPO,
        datasets=datasets,
        models=models,
        hf_cache=models,
        outputs=outputs,
        logs=env_root / "logs",
        cache=env_root / "cache",
        tmp=env_root / "tmp",
        archive=env_root / "archive",
    )


def _profile(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "aic.vhicraft.inference-profile/v1",
        "profile": "official_test",
        "run_type": "LOCAL_QUANTIZED_DEPLOYMENT",
        "protocol": "configs/vhicraft_v1_protocol.json",
        "inputs": {
            "mode": "numbered_videos",
            "video_root": {"base": "datasets", "path": "tests"},
            "target_ratio_wh": [9, 16],
        },
        "inference": {
            "qwen_snapshot": {"base": "models", "path": "Qwen3.5-4B"},
            "rtdetr_snapshot": {"base": "models", "path": "PekingU-rtdetr_r50vd"},
            "qwen_backend": "transformers-bnb-nf4",
            "qwen_compute_dtype": "float16",
            "rtdetr_dtype": "float32",
        },
    }
    path = tmp_path / "official_test.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_preflight_numbered_videos_pass(tmp_path):
    environment = _environment(tmp_path)
    profile_path = _profile(tmp_path)
    result = run_preflight(
        json.loads(profile_path.read_text(encoding="utf-8")),
        environment,
        config_path=profile_path,
        protocol_path=REPO / "configs" / "vhicraft_v1_protocol.json",
        run_id="vhicraft_official_test_202601010000",
    )
    assert result["status"] == "PASS", result["failed_checks"]
    assert result["video_count"] == 3
    assert result["official_score_available"] is False
    assert "LOCAL_OFFICIAL_SCORE_UNAVAILABLE" in result["official_score_note"]
    names = {check["name"] for check in result["checks"]}
    for expected in (
        "video_root",
        "video_id_unique",
        "video_files_nonempty",
        "target_ratio_matches_release",
        "output_root_writable",
        "run_id_free",
        "qwen_snapshot_revision",
        "rtdetr_snapshot_revision",
        "contract_pipeline",
        "visualization_pipeline",
        "git_head",
    ):
        assert expected in names


def test_preflight_fails_when_run_id_exists(tmp_path):
    environment = _environment(tmp_path)
    (environment.outputs / "vhicraft" / "vhicraft_official_test_202601010000").mkdir(parents=True)
    profile_path = _profile(tmp_path)
    result = run_preflight(
        json.loads(profile_path.read_text(encoding="utf-8")),
        environment,
        config_path=profile_path,
        protocol_path=REPO / "configs" / "vhicraft_v1_protocol.json",
        run_id="vhicraft_official_test_202601010000",
    )
    assert result["status"] == "FAIL"
    assert "run_id_free" in result["failed_checks"]


def test_preflight_marks_gt_like_file(tmp_path):
    environment = _environment(tmp_path)
    (environment.datasets / "tests" / "ground_truth.json").write_text("{}", encoding="utf-8")
    profile_path = _profile(tmp_path)
    result = run_preflight(
        json.loads(profile_path.read_text(encoding="utf-8")),
        environment,
        config_path=profile_path,
        protocol_path=REPO / "configs" / "vhicraft_v1_protocol.json",
    )
    assert result["official_score_available"] is True
    assert "ground_truth.json" in result["official_ground_truth_files"]
