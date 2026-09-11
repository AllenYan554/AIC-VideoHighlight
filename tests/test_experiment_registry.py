"""Tests for the experiment launch-spec registry interface (single source of truth)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "scripts" / "experiments" / "registry.py"
STAGE5_LAUNCHER_PATH = REPO_ROOT / "scripts" / "experiments" / "stage5" / "run.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    return _load("aic_test_registry", REGISTRY_PATH)


@pytest.fixture(scope="module")
def stage5():
    return _load("aic_test_stage5_launcher", STAGE5_LAUNCHER_PATH)


def test_launch_metadata_covers_registered_experiments(registry, stage5):
    assert set(stage5.LAUNCH) == set(stage5.RUNNERS) == set(stage5.CONFIGS)
    for name, meta in stage5.LAUNCH.items():
        assert meta["target"] in registry.VALID_TARGETS
        assert meta["gpu"] in registry.VALID_GPU


def test_list_matches_stage_registry(registry, stage5):
    entries = registry.list_experiments()
    names = [entry["experiment"] for entry in entries]
    assert names == sorted(stage5.RUNNERS)
    by_name = {entry["experiment"]: entry for entry in entries}
    for name, meta in stage5.LAUNCH.items():
        assert by_name[name]["target"] == meta["target"]
        assert by_name[name]["gpu"] == meta["gpu"]
        assert by_name[name]["stage"] == "stage5"


def test_describe_autodl_spec(registry):
    spec = registry.describe("stage5_6_vhicraft_formal")
    assert spec["schema_version"] == "aic.experiment-launch-spec/v1"
    assert spec["experiment"] == "stage5_6_vhicraft_formal"
    assert spec["target"] == "AUTODL"
    assert spec["gpu"] == "REQUIRED"
    assert spec["stage_launcher"] == "scripts/experiments/stage5/run.py"
    assert spec["canonical_args"] == ["--experiment", "stage5_6_vhicraft_formal"]
    assert spec["config"] == "configs/experiments/stage5/stage5_6_vhicraft_formal.json"
    assert spec["environment_config"] == "configs/environments/autodl.json"
    assert spec["remote_repo"] == "/root/autodl-tmp/AIC-VideoHighlight-run"
    assert spec["supports"] == {"resume": True, "dry_run": True, "validate_only": True}


def test_describe_vhicraft_smoke_and_ablation(registry):
    for name in ("stage5_6_vhicraft_smoke", "stage5_6_vhicraft_ablation"):
        spec = registry.describe(name)
        assert spec["experiment"] == name
        assert spec["target"] == "AUTODL"
        assert spec["gpu"] == "REQUIRED"
        assert spec["canonical_args"] == ["--experiment", name]
        assert spec["config"] == f"configs/experiments/stage5/{name}.json"


def test_describe_final_v1_stages(registry):
    for name in ("stage5_4_amendment4_revised_formal", "stage5_5_dev_formal"):
        spec = registry.describe(name)
        assert spec["target"] == "AUTODL"
        assert spec["gpu"] == "NONE"


def test_describe_windows_spec(registry):
    for name in ("stage5_infra_tiny_fake", "stage5_3_smoke"):
        spec = registry.describe(name)
        assert spec["target"] == "WINDOWS"
        assert spec["gpu"] == "NONE"
        assert spec["environment_config"] == "configs/environments/windows_local.json"
        assert spec["config"]


def test_describe_unknown_experiment_is_friendly(registry):
    with pytest.raises(KeyError) as excinfo:
        registry.describe("stage5_9_missing")
    message = str(excinfo.value)
    assert "unknown experiment: stage5_9_missing" in message
    for known in registry.list_experiments():
        assert known["experiment"] in message


def test_cli_list_and_describe(registry, capsys):
    assert registry.main(["list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "aic.experiment-launch-spec/v1"
    assert payload["experiments"]

    assert registry.main(["describe", "--experiment", "stage5_3_smoke"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["experiment"] == "stage5_3_smoke"


def test_cli_unknown_experiment_exit_code(registry, capsys):
    assert registry.main(["describe", "--experiment", "nope"]) == 2
    captured = capsys.readouterr()
    assert "unknown experiment: nope" in captured.err
    assert "stage5_3_formal" in captured.err


def _write_stage_launcher(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.py"
    path.write_text(body, encoding="utf-8")
    return path


def _redirect_stage_launchers(registry, monkeypatch, path: Path) -> None:
    monkeypatch.setattr(registry, "STAGE_LAUNCHERS", {"stage5": path})


def test_registry_rejects_inconsistent_metadata(registry, tmp_path, monkeypatch):
    body = (
        "RUNNERS = {'a': 'runner.py'}\n"
        "CONFIGS = {'a': 'config.json'}\n"
        "LAUNCH = {'a': {'target': 'LINUX', 'gpu': 'NONE'}}\n"
    )
    _redirect_stage_launchers(registry, monkeypatch, _write_stage_launcher(tmp_path, body))
    with pytest.raises(RuntimeError, match="invalid launch target"):
        registry.list_experiments()


def test_registry_rejects_missing_launch_entry(registry, tmp_path, monkeypatch):
    body = (
        "RUNNERS = {'a': 'runner.py'}\n"
        "CONFIGS = {'a': 'config.json'}\n"
        "LAUNCH = {}\n"
    )
    _redirect_stage_launchers(registry, monkeypatch, _write_stage_launcher(tmp_path, body))
    with pytest.raises(RuntimeError, match="missing LAUNCH metadata"):
        registry.list_experiments()


def test_registry_rejects_orphan_launch_entry(registry, tmp_path, monkeypatch):
    body = (
        "RUNNERS = {'a': 'runner.py'}\n"
        "CONFIGS = {'a': 'config.json'}\n"
        "LAUNCH = {'a': {'target': 'WINDOWS', 'gpu': 'NONE'}, 'ghost': {'target': 'WINDOWS', 'gpu': 'NONE'}}\n"
    )
    _redirect_stage_launchers(registry, monkeypatch, _write_stage_launcher(tmp_path, body))
    with pytest.raises(RuntimeError, match="without registered runner"):
        registry.list_experiments()
