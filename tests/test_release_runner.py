"""VHiCraft-v1 release tests (synthetic fixtures only; no dataset, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from scripts.run_vhicraft import main  # noqa: E402
from aic_video_highlight.evaluation.contract import validate_contract  # noqa: E402


def _fixtures(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    frames = []
    for frame in range(4):
        frames.append({
            "frame": frame,
            "ts0": {"x": 180, "y": 0, "w": 168, "crop_w": 168, "crop_h": 300},
            "ts5": {"x": 100 + frame, "y": 0, "w": 168, "crop_w": 168, "crop_h": 300},
        })
    (shards / "v1.json").write_text(json.dumps(frames), encoding="utf-8")
    role = {"records": [{"video_id": "v1", "split": "dev"}], "record_count": 1}
    (tmp_path / "role.json").write_text(json.dumps(role), encoding="utf-8")
    metadata = {"records": {"v1": {"width": 534, "height": 300, "frame_count": 1000}}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return shards, tmp_path / "role.json", tmp_path / "metadata.json"


def test_assemble_produces_valid_official_jsonl(tmp_path):
    shards, role, metadata = _fixtures(tmp_path)
    output = tmp_path / "predictions.jsonl"
    code = main([
        "assemble", "--shards-dir", str(shards), "--role-manifest", str(role),
        "--metadata-cache", str(metadata), "--output", str(output),
        "--stabilization", "stabilized", "--validate",
    ])
    assert code == 0
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["video_id"] == "v1"
    assert lines[0]["targetRatioWH"] == [9, 16]
    assert lines[0]["predictions"][1]["bboxes"] == [101, 0, 168]
    report = validate_contract(output)
    assert report["is_valid"] is True


def test_raw_stabilization_uses_ts0_geometry(tmp_path):
    shards, role, metadata = _fixtures(tmp_path)
    output = tmp_path / "predictions_raw.jsonl"
    assert main([
        "assemble", "--shards-dir", str(shards), "--role-manifest", str(role),
        "--metadata-cache", str(metadata), "--output", str(output), "--stabilization", "raw",
    ]) == 0
    line = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert line["predictions"][0]["bboxes"] == [180, 0, 168]


def test_limit_and_release_vs_pipeline_identity(tmp_path):
    from aic_video_highlight.composition.vhicraft_pipeline import (
        FrameCrop,
        assemble_prediction_lines,
    )

    shards, role, metadata = _fixtures(tmp_path)
    output = tmp_path / "predictions_limit.jsonl"
    assert main([
        "assemble", "--shards-dir", str(shards), "--role-manifest", str(role),
        "--metadata-cache", str(metadata), "--output", str(output), "--limit", "1",
    ]) == 0
    direct = assemble_prediction_lines(
        {"v1": {f: FrameCrop(f, 100 + f, 0, 168) for f in range(4)}},
        target_ratio=(9, 16),
        frame_size_by_video={"v1": (534, 300)},
        frame_count_by_video={"v1": 1000},
    )
    expected = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in direct)
    assert output.read_text(encoding="utf-8") == expected


def test_validate_mode(tmp_path):
    shards, role, metadata = _fixtures(tmp_path)
    output = tmp_path / "p.jsonl"
    main(["assemble", "--shards-dir", str(shards), "--role-manifest", str(role),
          "--metadata-cache", str(metadata), "--output", str(output)])
    report = tmp_path / "report.json"
    assert main(["validate", "--predictions", str(output), "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["is_valid"] is True and payload["line_count"] == 1


def test_manifest_loads():
    manifest = json.loads((REPO / "configs" / "vhicraft_v1_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((REPO / "configs" / "vhicraft_v1_protocol.json").read_text(encoding="utf-8"))
    assert manifest["framework_name"] == "VHiCraft"
    assert manifest["version"] == "v1"
    assert manifest["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert protocol["component_identities"]["stabilization_method"] == "projected_state_canonical_center_ema_v1"
    assert protocol["component_identities"]["prediction_selection_policy"] == "FS-0 (all_frames_v1)"
    assert manifest["heldout_lock"]["allowed_access"] == 0


def test_profiles_load():
    for name in ("smoke", "dev166"):
        payload = json.loads((REPO / "configs" / "profiles" / f"{name}.json").read_text(encoding="utf-8"))
        assert payload["profile"] == name
        assert payload["inference"]["qwen_backend"] == "transformers-bnb-nf4"
        assert payload["protocol"] == "configs/vhicraft_v1_protocol.json"


def test_cli_help_exits_zero(capsys):
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
