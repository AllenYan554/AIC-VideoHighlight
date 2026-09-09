from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_video_highlight.experiment_runtime.artifacts import build_artifact_manifest
from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths, path_flavour, validate_experiment_id
from aic_video_highlight.experiment_runtime.progress import ProgressReporter
from aic_video_highlight.experiment_runtime.raw_report import render_raw_report, write_ai_report_inputs
from aic_video_highlight.experiment_runtime.run_context import RunContext, RunIdentityMismatch
from aic_video_highlight.experiment_runtime.shards import ShardStore


def environment(tmp_path: Path) -> EnvironmentPaths:
    return EnvironmentPaths.from_dict({
        "name": "test", "repo": str(tmp_path), "datasets": str(tmp_path / "data"),
        "models": str(tmp_path / "models"), "hf_cache": str(tmp_path / "hf"),
        "outputs": str(tmp_path / "outputs"), "logs": str(tmp_path / "logs"),
        "cache": str(tmp_path / "cache"), "tmp": str(tmp_path / "tmp"),
        "archive": str(tmp_path / "archive"),
    })


def context(tmp_path: Path, config_text: str = "{}") -> RunContext:
    config = tmp_path / "config.json"
    protocol = tmp_path / "protocol.json"
    config.write_text(config_text, encoding="utf-8")
    protocol.write_text("{}", encoding="utf-8")
    env = environment(tmp_path)
    return RunContext("stage5_infra_test", "stage5", "TEST", tmp_path, env.for_experiment("stage5", "stage5_infra_test"), config, protocol)


def test_experiment_id_and_portable_path_handling():
    assert validate_experiment_id("stage5_3_formal") == "stage5_3_formal"
    with pytest.raises(ValueError):
        validate_experiment_id("Stage 5.3 Formal")
    assert path_flavour(r"D:\experiments\logs") == "windows"
    assert path_flavour("/root/autodl-tmp/logs") == "posix"


def test_environment_path_resolution(tmp_path):
    paths = environment(tmp_path).for_experiment("stage5", "stage5_3_formal")
    assert paths.logs == tmp_path / "logs" / "stage5" / "stage5_3_formal"
    assert paths.output == tmp_path / "outputs" / "stage5" / "stage5_3_formal"


def test_manifest_resume_and_hash_mismatch(tmp_path):
    first = context(tmp_path)
    manifest = first.start()
    assert manifest["schema_version"] == "aic.experiment-run-manifest/v1"
    assert manifest["launch_provenance"] == {
        "launch_source": "NON_INTERACTIVE_DIRECT_RUN",
        "launch_mode": "direct",
        "interactive_child": False,
        "target": "UNKNOWN",
    }
    assert "launch_provenance" not in first.identity()
    assert json.loads(first.status_path.read_text())["status"] == "RUNNING"
    assert context(tmp_path).start(resume=True)["resume_mode"] is True
    with pytest.raises(RunIdentityMismatch, match="config_sha256"):
        context(tmp_path, '{"changed": true}').start(resume=True)


def test_powershell_launch_provenance_is_operational_not_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_EXPERIMENT_LAUNCHER", "powershell_v1")
    monkeypatch.setenv("AIC_EXPERIMENT_LAUNCH_MODE", "interactive_child")
    monkeypatch.setenv("AIC_EXPERIMENT_INTERACTIVE_CHILD", "true")
    monkeypatch.setenv("AIC_EXPERIMENT_LAUNCH_TARGET", "AUTODL")
    run = context(tmp_path)
    identity = run.identity()
    manifest = run.start()
    assert "launch_provenance" not in identity
    assert manifest["launch_provenance"] == {
        "launch_source": "powershell_v1",
        "launch_mode": "interactive_child",
        "interactive_child": True,
        "target": "AUTODL",
    }


def test_protocol_hash_mismatch(tmp_path):
    initial = context(tmp_path)
    initial.start()
    initial.protocol_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RunIdentityMismatch, match="protocol_sha256"):
        initial.start(resume=True)


def test_shard_resume_and_artifact_hashing(tmp_path):
    store = ShardStore(tmp_path / "shards", "a" * 64)
    data = store.write("part_001", [{"value": 1}])
    assert store.is_complete("part_001")
    manifest = build_artifact_manifest(tmp_path, [data])
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["sha256"] == file_sha256(data)
    data.write_text("tampered", encoding="utf-8")
    assert not store.is_complete("part_001")


def test_progress_and_safe_interrupt(tmp_path):
    reporter = ProgressReporter("stage5_infra_test", 2, tmp_path)
    reporter.update(1, current_shard="one")
    reporter.interrupt()
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "INTERRUPTED"
    assert "INTERRUPTED" in (tmp_path / "events.jsonl").read_text()


def test_raw_report_is_factual_and_ai_report_is_separate(tmp_path):
    machine = tmp_path / "machine"
    machine.mkdir()
    payloads = {
        "summary": {"records": 3}, "metrics": {"score": 1.0}, "runtime": {"seconds": 2},
        "validation": {"status": "PASS"}, "artifact_manifest": {"artifacts": []},
    }
    for name, payload in payloads.items():
        (machine / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    render_raw_report(machine, tmp_path / "experiment_raw_report.md", "stage5_infra_test")
    write_ai_report_inputs(tmp_path)
    raw = (tmp_path / "experiment_raw_report.md").read_text(encoding="utf-8")
    assert "Scientific interpretation is intentionally excluded" in raw
    assert not (tmp_path / "experiment_report.md").exists()
    assert "experiment_report.md" in (tmp_path / "AI_REPORT_INPUTS.md").read_text(encoding="utf-8")


def test_status_completion_and_validation_failure(tmp_path):
    run = context(tmp_path)
    run.start()
    run.set_status("VALIDATION_FAILED", reason="synthetic")
    assert json.loads(run.status_path.read_text())["status"] == "VALIDATION_FAILED"
    run.set_status("COMPLETED", validation="PASS")
    assert json.loads(run.manifest_path.read_text())["finished_at"] is not None


def test_canonical_hash_stable_key_order():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
