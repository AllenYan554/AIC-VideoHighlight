"""Read-only preflight for a release inference profile.

Preflight never loads models and never runs inference.  It validates that a
profile's inputs, models, output destination, contract pipeline and reporting
pipeline are ready.  For official-test inputs only file names/sizes and index
fields are inspected; video content is never decoded or analyzed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from aic_video_highlight.runtime.datasets import scan_numbered_videos, validate_target_ratio
from aic_video_highlight.runtime.hashing import file_sha256
from aic_video_highlight.runtime.paths import EnvironmentPaths

_GT_HINTS = ("gt", "label", "annotation", "evaluator", "groundtruth", "ground_truth", "score")


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def _resolve(spec: Mapping[str, Any], environment: EnvironmentPaths) -> Path:
    roots = {
        "repo": environment.repo,
        "outputs": environment.outputs,
        "datasets": environment.datasets,
        "models": environment.models,
        "hf_cache": environment.hf_cache,
    }
    base = str(spec["base"])
    if base not in roots:
        raise ValueError(f"unsupported preflight input base: {base}")
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else roots[base] / path


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _minimal_result(video_id: str, ratio: list[float]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "profile": "preflight",
        "video_ids": [video_id],
        "video_count": 1,
        "qwen_calls": 0,
        "rtdetr_calls": 0,
        "empty_video_count": 0,
        "empty_video_ids": [],
        "identity": {"git_head": "preflight"},
        "resources": {
            "processes": [],
            "qwen": {},
            "rtdetr": {},
            "total_wall_sec": 0.0,
            "oom": False,
        },
        "variants": {"stabilized": {"contract": {"is_valid": True}}},
    }


def run_preflight(
    config: Mapping[str, Any],
    environment: EnvironmentPaths,
    *,
    config_path: Path,
    protocol_path: Path,
    run_id: str | None = None,
    min_free_mib: int = 2048,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    checks: list[dict[str, Any]] = []
    profile = str(config.get("profile", ""))
    checks.append(
        _check(
            "profile_schema",
            str(config.get("schema_version", "")) == "aic.vhicraft.inference-profile/v1",
            str(config.get("schema_version", "")),
        )
    )
    checks.append(_check("protocol_exists", protocol_path.is_file(), str(protocol_path)))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}

    inference = config.get("inference", {})
    inputs = config.get("inputs", {})
    mode = str(inputs.get("mode", "dev_role"))
    official_gt_files: list[str] = []
    duration_coverage: str | None = None

    if mode == "numbered_videos":
        try:
            video_root = _resolve(inputs["video_root"], environment)
            files = scan_numbered_videos(video_root)
            ids = [path.stem for path in files]
            checks.append(_check("video_root", True, f"{len(files)} videos in {video_root}"))
            checks.append(_check("video_id_unique", len(ids) == len(set(ids)), f"{len(set(ids))} unique ids"))
            empty_files = [path.name for path in files if path.stat().st_size <= 0]
            checks.append(_check("video_files_nonempty", not empty_files, f"empty: {empty_files[:3]}"))
            ratio = validate_target_ratio(inputs.get("target_ratio_wh"))
            checks.append(_check("target_ratio_valid", True, str(ratio)))
            from aic_video_highlight.composition.vhicraft_pipeline import TARGET_RATIO

            checks.append(
                _check(
                    "target_ratio_matches_release",
                    tuple(ratio) == tuple(float(value) for value in TARGET_RATIO),
                    f"profile={ratio} release={tuple(TARGET_RATIO)}",
                )
            )
            others = [
                path.name
                for path in video_root.iterdir()
                if path.is_file() and path.suffix.lower() != ".mp4"
            ]
            official_gt_files = [
                name for name in others if any(hint in name.lower() for hint in _GT_HINTS)
            ]
            checks.append(
                _check(
                    "official_ground_truth_presence",
                    True,
                    f"non-video files={len(others)}; gt-like={official_gt_files}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("numbered_videos_input", False, f"{type(exc).__name__}: {exc}"))
    else:
        try:
            role_path = _resolve(inputs["role_manifest"], environment)
            dev_manifest = _resolve(inference["dev_manifest"], environment)
            source_index = _resolve(inference["source_index"], environment)
            video_root = _resolve(inference["video_root"], environment)
            missing = [str(path) for path in (role_path, dev_manifest, source_index, video_root) if not path.exists()]
            checks.append(_check("dev_role_inputs", not missing, f"missing={missing}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("dev_role_inputs", False, f"{type(exc).__name__}: {exc}"))

    output_root = environment.outputs / "vhicraft"
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        writable = os.access(output_root, os.W_OK)
        checks.append(_check("output_root_writable", writable, str(output_root)))
    except OSError as exc:
        checks.append(_check("output_root_writable", False, str(exc)))
    if run_id:
        target = output_root / run_id
        checks.append(_check("run_id_free", not target.exists(), str(target)))
    free_bytes = shutil.disk_usage(environment.outputs).free
    checks.append(
        _check(
            "disk_free",
            free_bytes >= min_free_mib * 2**20,
            f"{free_bytes / 2**30:.1f} GiB free (need {min_free_mib} MiB)",
        )
    )

    from aic_video_highlight.runtime.orchestrator import QWEN_REVISION, RTDETR_REVISION

    try:
        qwen_snapshot = _resolve(inference["qwen_snapshot"], environment)
        rtdetr_snapshot = _resolve(inference["rtdetr_snapshot"], environment)
        qwen_trees = qwen_snapshot / ".cache" / "huggingface" / "trees" / f"{QWEN_REVISION}.json"
        rtdetr_trees = rtdetr_snapshot / ".cache" / "huggingface" / "trees" / f"{RTDETR_REVISION}.json"
        checks.append(_check("qwen_snapshot_revision", qwen_snapshot.is_dir() and qwen_trees.is_file(), f"{qwen_snapshot} @ {QWEN_REVISION}"))
        checks.append(_check("rtdetr_snapshot_revision", rtdetr_snapshot.is_dir() and rtdetr_trees.is_file(), f"{rtdetr_snapshot} @ {RTDETR_REVISION}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("model_snapshots", False, f"{type(exc).__name__}: {exc}"))

    workspace = Path(tempfile.mkdtemp(prefix="vhicraft_preflight_"))
    try:
        from aic_video_highlight.evaluation.contract import validate_contract

        ratio = [9.0, 16.0]
        synthetic = {
            "video_id": "_preflight_",
            "targetRatioWH": ratio,
            "predictions": [{"frame": 0, "bboxes": [0, 0, 144]}],
        }
        predictions = workspace / "predictions.jsonl"
        predictions.write_text(json.dumps(synthetic, ensure_ascii=False) + "\n", encoding="utf-8")
        contract_report = validate_contract(predictions)
        checks.append(
            _check(
                "contract_pipeline",
                bool(contract_report.get("is_valid")),
                f"issue_count={contract_report.get('issue_count')}",
            )
        )

        from aic_video_highlight.reporting import render_run_report

        run_dir = workspace / "run"
        (run_dir / "predictions").mkdir(parents=True)
        (run_dir / "inference_result.json").write_text(
            json.dumps(_minimal_result("_preflight_", ratio), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "predictions" / "predictions.jsonl").write_text(
            json.dumps(synthetic, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rendered = render_run_report(run_dir)
        ready = (
            (run_dir / "tables" / "summary.csv").is_file()
            and (run_dir / "report" / "run_report.md").is_file()
            and any((run_dir / "figures").glob("*.png"))
        )
        checks.append(_check("visualization_pipeline", ready, ",".join(rendered["figures"])))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("contract_visualization_pipeline", False, f"{type(exc).__name__}: {exc}"))
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    head = _git_head(environment.repo)
    checks.append(_check("git_head", bool(head), str(head)))
    expected_model = protocol.get("component_identities", {}).get("model", {})
    if expected_model:
        checks.append(
            _check(
                "protocol_model_identity",
                expected_model.get("revision") == QWEN_REVISION,
                f"protocol={expected_model.get('revision')} release={QWEN_REVISION}",
            )
        )

    failed = [check["name"] for check in checks if check["status"] == "FAIL"]
    gt_present = bool(official_gt_files)
    return {
        "schema_version": "aic.vhicraft.preflight/v1",
        "status": "PASS" if not failed else "FAIL",
        "profile": profile,
        "input_mode": mode,
        "timestamp": timestamp,
        "run_id": run_id,
        "suggested_run_id": run_id or f"vhicraft_{profile}_{timestamp}",
        "git_head": head,
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(protocol_path),
        "model_revisions": {"qwen": QWEN_REVISION, "rtdetr": RTDETR_REVISION},
        "video_count": len(ids) if mode == "numbered_videos" and "ids" in locals() else None,
        "duration_coverage": duration_coverage,
        "official_ground_truth_files": official_gt_files,
        "official_score_available": gt_present,
        "official_score_note": (
            "LOCAL_OFFICIAL_SCORE_UNAVAILABLE: official test ground truth is hidden; "
            "the real official score is obtained only after submitting predictions."
            if not gt_present
            else "A local evaluator-like file exists, but running it still requires explicit user authorization."
        ),
        "checks": checks,
        "failed_checks": failed,
    }
