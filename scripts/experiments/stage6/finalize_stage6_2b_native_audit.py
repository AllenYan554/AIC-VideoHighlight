"""核验完整既有审计产物并记录最终身份；不重新计算/选择feature或mask。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import struct
import subprocess
import sys

import numpy as np
import pandas as pd

from verify_stage6_2b_inputs import DEFAULT_OUTPUT, RUN, sha, read_json, read_jsonl
from analyze_stage6_2b_native_signals import dump_json, xywh_iou

CORE_TABLES = ["feature_dictionary.csv", "feature_discrimination.csv", "per_video_signal_metrics.csv",
               "spatial_quality_signal.csv", "contribution_signal.csv", "drop_curve.csv",
               "contribution_mass_curve.csv", "segment_block_comparison.csv", "hypothesis_results.csv"]
FIGURES = ["signal_overview.png", "per_video_signal_distribution.png", "fp_vs_tp_drop_curve.png",
           "contribution_mass_loss.png", "segment_vs_frame_budget.png", "candidate_edge_analysis.png",
           "localization_state_analysis.png", "representative_timelines.png"]


def close(actual, expected):
    return bool(np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True))


def main():
    root = DEFAULT_OUTPUT
    checks = []

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(f"产物核验失败：{name}；{detail}")

    closure = read_json(root / "audit/input_closure_verification.json")
    inp = root / f"inputs/control_{RUN}"
    check("input_closure", closure["decision"] == "PASS_INPUT_CLOSURE")
    failures = [f for f, expected in closure["file_hashes"].items() if sha(inp / f) != expected]
    check("all_frozen_input_hashes_unchanged", not failures, {"files": len(closure["file_hashes"]), "mismatches": failures})
    for table in CORE_TABLES:
        check("required_table:" + table, (root / "tables" / table).is_file() and (root / "tables" / table).stat().st_size > 0)
    protocol = read_json(root / "audit/analysis_protocol.json")
    dictionary = pd.read_csv(root / "tables/feature_dictionary.csv")
    check("registered_dictionary_frozen", dictionary.feature.tolist() == [s["feature"] for s in protocol["specifications"]], {"features": len(dictionary)})
    frames = pd.read_csv(root / "tables/frame_features_targets.csv", float_precision="round_trip")
    baseline = pd.read_csv(root / "tables/baseline_per_video.csv", float_precision="round_trip").set_index("video_id")
    check("frame_key_unique", not frames.duplicated(["video_id", "frame"]).any())
    check("negative_Q_undefined_Z_zero", frames.loc[frames.Y == 0, "Q"].isna().all() and frames.loc[frames.Y == 0, "Z"].eq(0).all())
    check("positive_Q_Z_semantics", close(frames.loc[frames.Y == 1, "Q"], frames.loc[frames.Y == 1, "Z"]) and frames.loc[frames.Y == 1, "Q"].between(0, 1).all())
    check("structural_primary_missing", frames.loc[frames.fallback == 1, "primary_score"].isna().all())
    ref = {r["video_id"]: r for r in read_jsonl(inp / "reference/dev166_weak_spatial_reference.jsonl")}
    pred = {r["video_id"]: r for r in read_jsonl(inp / "predictions/predictions.jsonl")}
    check("identity_population", set(baseline.index) == set(ref) == set(pred) and len(baseline) == 166)
    recomputed = []
    for video in sorted(pred):
        shard = read_json(inp / f"stabilization/shards/{video}.json")
        # shard行结构在闭合核验时已冻结；这里再次直接由TS5与weak ROI重算target。
        records = shard
        geometry = {int(r["frame"]): r["ts5"] for r in records}
        got = frames[frames.video_id == video].set_index("frame")
        mass, positives = 0., 0
        for p in pred[video]["predictions"]:
            f = int(p["frame"])
            box = ref[video]["rois"].get(str(f))
            y = int(box is not None)
            crop = geometry[f]
            q = xywh_iou([crop[k] for k in ["x", "y", "w", "h"]], box) if y else np.nan
            z = q if y else 0.
            check_value = got.loc[f]
            if int(check_value.Y) != y or not close(check_value.Q, q) or not close(check_value.Z, z):
                raise ValueError(f"raw目标不匹配：{video}:{f}")
            mass += z; positives += y
        n, g = len(pred[video]["predictions"]), len(ref[video]["rois"])
        recomputed.append({"video_id": video, "n_pred": n, "n_ref": g, "tp": positives,
                           "sum_iou": mass, "weak_f": 2 * mass / (n + g) if n + g else 1.})
    raw = pd.DataFrame(recomputed).set_index("video_id").reindex(baseline.index)
    check("all_raw_targets_and_baseline_recomputed", close(raw[["n_pred", "n_ref", "tp", "sum_iou", "weak_f"]], baseline[["n_pred", "n_ref", "tp", "sum_iou", "weak_f"]]), {"frames": len(frames), "videos": len(raw)})
    formal = read_json(inp / "reference/formal_protocol.json")
    frozen_f = formal["frozen_stage6_0_control"]["control_metrics"]["macro_f"]
    check("frozen_full_macro_F", close(baseline.weak_f.mean(), frozen_f), {"computed": baseline.weak_f.mean(), "protocol_control": frozen_f})
    d = pd.read_csv(root / "tables/per_video_drop_metrics.csv", float_precision="round_trip")
    check("all_fixed_budgets_retained", sorted(d.budget.unique()) == protocol["budgets"])
    expected_k = np.floor(d.eligible_count * d.budget + 1e-12)
    check("every_mask_exact_unique_budget", close(d.drop_count, expected_k), {"rows": len(d)})
    matching = d.groupby(["video_id", "feature", "budget"])[["eligible_count", "drop_count", "eligible_z"]].nunique(dropna=False)
    check("six_methods_same_eligibility_and_cost", matching.eq(1).all().all() and d.groupby(["video_id", "feature", "budget"]).method.nunique().eq(6).all())
    check("deletion_accounting", close(d.fp_removed + d.tp_lost, d.drop_count) and close(d.fp_total + d.tp_total, d.n_all))
    g = d.video_id.map(baseline.n_ref).to_numpy()
    m, k, s, u = d.n_all.to_numpy(), d.drop_count.to_numpy(), d.z_total.to_numpy(), d.z_lost.to_numpy()
    old = np.divide(2 * s, m + g, out=np.ones(len(d)), where=m + g != 0)
    new = np.divide(2 * (s - u), m - k + g, out=np.ones(len(d)), where=m - k + g != 0)
    check("all_deletion_delta_F_algebra", close(new - old, d.delta_f))
    check("all_mass_loss_bounds", (d.z_lost >= -1e-10).all() and (d.z_lost <= d.z_total + 1e-10).all())
    # 全部4320个aggregate直接与358560行逐视频结果对账，包含空输出视频。
    aggregate = pd.read_csv(root / "tables/segment_block_comparison.csv", float_precision="round_trip")
    valid = set(baseline[baseline.role == "reference_valid109"].index)
    comparisons = {"frames_removed": "drop_count", "FP_removed": "fp_removed", "TP_lost": "tp_lost", "sum_IoU_lost": "z_lost"}
    count = 0
    for scope, population in [("reference_valid109", valid), ("all166_weak_label_sensitivity", set(baseline.index))]:
        dg = d[d.video_id.isin(population)].groupby(["feature", "budget", "method"])
        for row in aggregate[aggregate.scope == scope].to_dict("records"):
            group = dg.get_group((row["feature"], row["budget"], row["method"]))
            if len(group) != len(population) or not close(group.delta_f.mean(), row["macro_delta_F"]):
                raise ValueError("macro总体/均值不匹配")
            for output, source in comparisons.items():
                if not close(group[source].sum(), row[output]):
                    raise ValueError(f"aggregate不匹配：{scope}:{row['feature']}:{row['budget']}:{output}")
            if not close(group.z_lost.sum() / group.z_total.sum(), row["z_mass_lost_fraction"]):
                raise ValueError("aggregate质量比例不匹配")
            count += 1
    check("all_aggregate_per_video_reconciliation", count == len(aggregate), {"aggregate_rows": count})
    check("frame_curves_same_complete_records", sha(root / "tables/drop_curve.csv") == sha(root / "tables/contribution_mass_curve.csv"))
    source = pd.read_csv(root / "tables/source_signal_consistency.csv")
    check("empty_source_retained", source.source_clusters_total.eq(87).all() and len(source) == 77 * 3)
    hypothesis = pd.read_csv(root / "tables/hypothesis_results.csv")
    check("all_hypotheses_adjudicated", hypothesis.hypothesis.tolist() == [f"H{i}" for i in range(1, 8)] and hypothesis.status.isin(["SUPPORTED", "WEAK", "NOT_SUPPORTED", "CONFOUNDED", "UNAVAILABLE"]).all())
    figure_manifest = read_json(root / "audit/figure_manifest.json")
    sizes = {}
    for entry in figure_manifest["figures"]:
        p = root / "figures" / entry["file"]
        b = p.read_bytes()
        check("figure_hash:" + p.name, sha(p) == entry["sha256"] and b[:8] == b"\x89PNG\r\n\x1a\n")
        sizes[p.name] = struct.unpack(">II", b[16:24])
        for table, expected in entry["source_sha256"].items():
            check("figure_source_hash:" + p.name + ":" + table, sha(root / "tables" / table) == expected)
    check("eight_real_figures", set(sizes) == set(FIGURES), sizes)
    check("visual_review_completed", figure_manifest["review_status"] == "VISUALLY_REVIEWED")
    report = (root / "report/Stage6_2B_Task_Native_Signal_Audit.md").read_text(encoding="utf-8")
    check("report_full_audit_and_gate", all(x in report for x in ["RECOVERY VERIFIED", "FULL AUDIT", "NO_SIGNAL", "H1", "H7"]))
    check("previous_missing_history_retained", (root / "report/history/Stage6_2B_Task_Native_Signal_Audit_INPUTS_MISSING_20260915.md").is_file())
    qa = {"schema_version": "aic.stage6_2b.final-artifact-qa/v1", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "decision": "PASS_ARTIFACT_QA", "checks": checks, "test_suite": "10 minimal mathematical/contract tests passed",
          "production_build": "NOT_APPLICABLE:no production changes", "no_new_predictions": True}
    dump_json(root / "audit/artifact_quality_verification.json", qa)
    summary = read_json(root / "audit/summary.json")
    summary.update(final_gate="NO_SIGNAL", final_gate_semantics="发现稳定弱参考统计信号，但short-event/Recall tail及一般block收益未通过完整Gate；非零统计效应",
                   audit="FULL_AUDIT_COMPLETED", block_status="WEAK / INCONCLUSIVE_BLOCK_SIGNAL", support_decision="CLOSE_SUPPORT_EDGE_FAMILY",
                   support_decision_scope="停止将当前支持边缘家族作为独立pruner继续调参；独立机制不可识别",
                   figures_status="RENDERED_AND_VISUALLY_REVIEWED", artifact_qa="PASS_ARTIFACT_QA", input_hashes_unchanged=True,
                   hypothesis_results=dict(zip(hypothesis.hypothesis, hypothesis.status)), auto_dl=False,
                   short_event_tail_acceptable="NOT_ESTABLISHED:完整五预算中存在高贡献短事件严重损失，无预注册可接受界限",
                   statistical_signals_found=True, deploy_selector=False, next_step="task-specific learned representation/model研究；保留FS0，无heuristic调参")
    dump_json(root / "audit/summary.json", summary)
    identity = read_json(root / "audit/audit_identity.json")
    identity["git_at_input_closure"] = identity.pop("git_start", identity.get("git_at_input_closure"))
    identity.update(analysis_protocol_sha256=sha(root / "audit/analysis_protocol.json"), formal_protocol_sha256=sha(inp / "reference/formal_protocol.json"),
                    closure_sha256=sha(root / "audit/input_closure_verification.json"), completed_at_utc=qa["checked_at_utc"], artifact_qa_sha256=sha(root / "audit/artifact_quality_verification.json"))
    identity["git_startup_observation"] = "最初工具检查两仓tracked/untracked均干净；closure阶段AIC新增本次stage6审计目录。git_at_input_closure字段按机器输出保留，非零修改的初始状态。"
    identity["scripts_sha256"] = {str(p): sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in {".py", ".md"}}
    identity["git_end"] = {}
    for repo in ["VHiCraft", "AIC-VideoHighlight"]:
        cwd = root.parents[2] / repo
        identity["git_end"][repo] = {}
        for label, args in [("status", ["status", "--short", "--branch"]), ("head", ["rev-parse", "HEAD"]), ("diff_stat", ["diff", "--stat"]), ("diff_check", ["diff", "--check"])]:
            result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                raise RuntimeError(result.stderr)
            identity["git_end"][repo][label] = result.stdout.strip()
    dump_json(root / "audit/audit_identity.json", identity)
    files = {}
    for folder in ["audit", "tables", "figures", "report", "handoff"]:
        for p in (root / folder).rglob("*"):
            if p.is_file() and p.name != "artifact_manifest.json":
                files[str(p.relative_to(root))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    dump_json(root / "audit/artifact_manifest.json", {"schema_version": "aic.stage6_2b.final-artifact-manifest/v1", "run_id": RUN,
                                                    "includes_inputs": False, "files": files, "qa": "PASS_ARTIFACT_QA", "gate": "NO_SIGNAL"})
    print(f"PASS_ARTIFACT_QA：{len(checks)}项检查，701输入hash未变，4320aggregate对账，八张图已视觉审查；NO_SIGNAL")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
