"""补齐完整逐视频分位enrichment和source-equal统计；不增加feature/方向/预算。"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from analyze_stage6_2b_native_signals import DEFAULT_OUTPUT, SEED, BOOTSTRAPS, numeric_summary, spearman, dump_json


def main():
    r = DEFAULT_OUTPUT
    protocol = json.loads((r / "audit/analysis_protocol.json").read_text(encoding="utf-8"))
    df = pd.read_csv(r / "tables/frame_features_targets.csv", float_precision="round_trip")
    main_df = df[df.role == "reference_valid109"]
    byvideo = {v: g for v, g in main_df.groupby("video_id", sort=True)}
    bysource = {v: g for v, g in main_df.groupby("source_id", sort=True)}
    rows, enrichment = [], []
    for spec in protocol["specifications"]:
        f, direction = spec["feature"], spec["direction"]
        if not spec["available"] or not direction:
            continue
        for kind, groups in [("video", byvideo), ("source", bysource)]:
            for key, gg in groups.items():
                g = gg[gg[f].notna()]
                if not len(g):
                    continue
                x = g[f].to_numpy() * direction
                order = np.lexsort((g.tie_hash.to_numpy(dtype=np.uint64), x))
                constant = g[f].nunique() < 2
                quantile_means = []
                for q, indices in enumerate(np.array_split(order, 5), 1):
                    z = g.iloc[indices]
                    qp = z.loc[z.Y == 1, "Q"]
                    mean_z = float(z.Z.mean()) if len(z) else np.nan
                    quantile_means.append(mean_z)
                    rows.append({"cluster_kind": kind, "cluster_id": key, "feature": f, "rank_quantile": q, "n": len(z),
                                 "constant_feature_in_cluster": constant, "Y_mean": z.Y.mean(), "Q_n": int(z.Y.sum()),
                                 "Q_mean": qp.mean(), "Q_median": qp.median(), "Z_mean": mean_z, "Z_mass": z.Z.sum(),
                                 "cluster_Z_mean": g.Z.mean(), "Z_density_enrichment": mean_z / g.Z.mean() if g.Z.mean() > 0 else np.nan})
                enrichment.append({"cluster_kind": kind, "cluster_id": key, "feature": f, "constant": constant,
                                   "low_quintile_Z_enrichment": quantile_means[0] / g.Z.mean() if g.Z.mean() > 0 and not constant else np.nan,
                                   "high_minus_low_Z_mean": quantile_means[-1] - quantile_means[0] if not constant else np.nan,
                                   "overall_Z_mean": g.Z.mean(), "n": len(g)})
    pd.DataFrame(rows).to_csv(r / "tables/per_cluster_signal_quantiles.csv", index=False, encoding="utf-8")
    pd.DataFrame(enrichment).to_csv(r / "tables/per_cluster_contribution_enrichment.csv", index=False, encoding="utf-8")
    source_summary(r, protocol)
    print(f"完成完整逐video/source分位与enrichment：{len(rows)}行")


def source_summary(r, protocol):
    source = pd.read_csv(r / "tables/per_source_signal_metrics.csv")
    baseline = pd.read_csv(r / "tables/baseline_per_video.csv")
    source_ids = sorted(set(baseline.loc[baseline.role == "reference_valid109", "source_id"]))
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)), size=BOOTSTRAPS)
    results = []
    for spec in protocol["specifications"]:
        f = spec["feature"]
        values = source[source.feature == f].set_index("source_id").reindex(source_ids)
        for target, metric, baseline in [("Y", "auroc", .5), ("Q", "q_spearman", 0), ("Z", "z_spearman", 0)]:
            v = values[metric].to_numpy(dtype=float)
            good = np.isfinite(v)
            if good.any():
                denominator = W[:, good].sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sample = W[:, good] @ v[good] / denominator
                sample = sample[np.isfinite(sample)]
                ci = np.quantile(sample, [.025, .975])
            else:
                ci = [np.nan, np.nan]
            results.append({"scope": "reference_valid109", "feature": f, "target": target,
                            "source_clusters_total": len(source_ids), "computable_sources": int(good.sum()),
                            "source_equal_mean": numeric_summary(v)["mean"], "source_equal_median": numeric_summary(v)["median"],
                            "source_direction_positive_count": int((v[good] > baseline).sum()),
                            "source_sign_consistency": float((v[good] > baseline).mean()) if good.any() else np.nan,
                            "source_equal_ci_low": ci[0], "source_equal_ci_high": ci[1]})
    pd.DataFrame(results).to_csv(r / "tables/source_signal_consistency.csv", index=False, encoding="utf-8")
    print(f"source一致性{len(results)}行；bootstrap总体保留{len(source_ids)}个source，含空输出source")


def event_risk(r, protocol):
    df = pd.read_csv(r / "tables/frame_features_targets.csv", float_precision="round_trip")
    events = pd.read_csv(r / "tables/reference_events.csv")
    from analyze_stage6_2b_native_signals import exact_budget
    rows = []
    byvideo = {v: g for v, g in df[df.role == "reference_valid109"].groupby("video_id")}
    for s in protocol["specifications"]:
        if not s["available"] or not s["direction"]:
            continue
        f = s["feature"]
        for v, g in byvideo.items():
            x = g[f].to_numpy() * s["direction"]
            valid = np.flatnonzero(np.isfinite(x))
            order = valid[np.lexsort((g.tie_hash.to_numpy(dtype=np.uint64)[valid], x[valid]))]
            frames, y, z = g.frame.to_numpy(), g.Y.to_numpy(), g.Z.to_numpy()
            ev = events[events.video_id == v]
            for b in protocol["budgets"]:
                k = exact_budget(len(valid), b)
                drop = np.zeros(len(g), dtype=bool);drop[order[:k]] = True
                for e in ev.to_dict("records"):
                    mask = (frames >= e["start_frame"]) & (frames <= e["end_frame"]) & (y == 1)
                    lost = mask & drop
                    tp, mass = int(mask.sum()), float(z[mask].sum())
                    lt, lm = int(lost.sum()), float(z[lost].sum())
                    rows.append({"feature": f, "budget": b, "method": "native_frame_ranking", "video_id": v, "source_id": g.source_id.iloc[0],
                                 "event_id": e["event_id"], "duration_sec": e["duration_sec"], "reference_frames": e["reference_frames"],
                                 "short_within_video_quartile": e["short_within_video_quartile"], "baseline_matched": tp, "TP_lost": lt,
                                 "event_erased": bool(tp and lt == tp), "baseline_IoU_mass": mass, "IoU_mass_lost": lm,
                                 "IoU_mass_retained": mass - lm, "event_frame_recall_delta": -lt / e["reference_frames"],
                                 "event_IoU_recall_delta": -lm / e["reference_frames"],
                                 "constant_eligible_feature": bool(len(valid) and len(np.unique(x[valid])) < 2),
                                 "identity": "POSTHOC_DIAGNOSTIC_MASK_EVENT_RISK"})
    pd.DataFrame(rows).to_csv(r / "tables/per_event_native_drop_risk.csv", index=False, encoding="utf-8")
    print(f"原生逐帧mask完整event风险：{len(rows)}行；所有已注册方向/预算，不挑案例")


def block_variable_strata(r):
    d = pd.read_csv(r / "tables/per_video_drop_metrics.csv", float_precision="round_trip")
    base = pd.read_csv(r / "tables/baseline_per_video.csv")
    valid = set(base.loc[base.role == "reference_valid109", "video_id"])
    d = d[d.video_id.isin(valid)]
    from analyze_stage6_2b_native_signals import ClusterBootstrap
    vs = sorted(valid)
    source = dict(zip(base.video_id, base.source_id))
    boot = ClusterBootstrap(vs, source)
    output = []
    for (f, b), g in d.groupby(["feature", "budget"]):
        frame = g[g.method == "native_frame_ranking"].set_index("video_id").reindex(vs)
        for method in ["native_atomic_block", "fixed_temporal_grid", "candidate_parent_region"]:
            region = g[g.method == method].set_index("video_id").reindex(vs)
            delta = (region.delta_f - frame.delta_f).to_numpy()
            for kind in ["variable", "constant"]:
                keep = (frame.constant_eligible_feature == (kind == "constant")).to_numpy() & (frame.eligible_count.to_numpy() > 0)
                values = np.where(keep, delta, np.nan)
                ci = boot.mean_ci(values, True)
                output.append({"scope": "reference_valid109", "feature": f, "budget": b, "method": method,
                               "feature_variation_stratum": kind, "videos": int(keep.sum()), "sources": len({source[v] for v, k in zip(vs, keep) if k}),
                               "macro_delta_F_region_minus_frame": numeric_summary(values)["mean"],
                               "source_ci_low": ci[0], "source_ci_high": ci[1],
                               "videos_region_better": int(((delta > 1e-12) & keep).sum()), "videos_region_worse": int(((delta < -1e-12) & keep).sum())})
    pd.DataFrame(output).to_csv(r / "tables/block_frame_variable_strata.csv", index=False, encoding="utf-8")
    tails = []
    for (f, b, method), g in d.groupby(["feature", "budget", "method"]):
        row = {"feature": f, "budget": b, "method": method, "scope": "reference_valid109"}
        for metric in ["delta_frame_recall", "delta_iou_recall", "short_delta_recall"]:
            vals = g[metric].dropna()
            for q in [0., .01, .1, .25, .5]:
                row[metric + "_q" + str(q)] = vals.quantile(q) if len(vals) else np.nan
        tails.append(row)
    pd.DataFrame(tails).to_csv(r / "tables/recall_tail_by_method.csv", index=False, encoding="utf-8")
    print(f"block分层paired对照：{len(output)}行；Recall tail：{len(tails)}行")


def reference_geometry_audit(r):
    """复现已披露的标签几何混杂诊断；不新增部署feature或删除规则。"""
    reference = r / "inputs/control_vhicraft_dev166_202609141233/reference/dev166_weak_spatial_reference.jsonl"
    rows = []
    for line in reference.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        w, h = record["video_width"], record["video_height"]
        for frame, box in record["rois"].items():
            x, y, bw, bh = box
            cx, cy = x + bw / 2, y + bh / 2
            rows.append({"video_id": record["video_id"], "frame": int(frame),
                         "reference_center_offset_x": abs(cx / w - .5), "reference_center_offset_y": abs(cy / h - .5),
                         "horizontal_center_pixel_rounding": abs(cx - w / 2) <= .5, "reference_area_ratio": bw * bh / (w * h)})
    geometry = pd.DataFrame(rows)
    geometry.to_csv(r / "tables/weak_reference_geometry.csv", index=False, encoding="utf-8")
    dump_json(r / "audit/weak_reference_geometry_audit.json", {
        "identity": "WEAK_REFERENCE_GEOMETRY_DIAGNOSTIC_ONLY", "n_reference_frames": len(geometry),
        "video_count": geometry.video_id.nunique(),
        "center_definition": "ROI center within 0.5 pixel of image horizontal center;pixel rounding semantics only,no selector",
        "horizontal_center_pixel_rounding_fraction": geometry.horizontal_center_pixel_rounding.mean(),
        "center_offset_x_quantiles": geometry.reference_center_offset_x.quantile([0, .25, .5, .75, 1]).to_dict(),
        "center_offset_y_quantiles": geometry.reference_center_offset_y.quantile([0, .25, .5, .75, 1]).to_dict(),
        "video_equal_center_fraction": geometry.groupby("video_id").horizontal_center_pixel_rounding.mean().mean(),
        "new_inference_features": False})
    frames = pd.read_csv(r / "tables/frame_features_targets.csv", float_precision="round_trip")
    positive = frames[frames.Y == 1].merge(geometry, on=["video_id", "frame"], validate="one_to_one")
    quality = []
    for centered, group in positive.groupby("horizontal_center_pixel_rounding", sort=True):
        for feature in ["horizontal_center_offset_abs", "shift_x_abs", "bbox_center_extremeness"]:
            observed = group[group[feature].notna()]
            values = [spearman(-g[feature].to_numpy(), g.Q.to_numpy()) for _, g in observed.groupby("video_id")]
            quality.append({"reference_centered_pixel_rounding": bool(centered), "feature": feature,
                            "positive_n": len(observed), "videos": observed.video_id.nunique(),
                            "pooled_Q_spearman": spearman(-observed[feature].to_numpy(), observed.Q.to_numpy()),
                            "video_mean_Q_rho": numeric_summary(values)["mean"],
                            "computable_videos": int(np.isfinite(values).sum())})
    pd.DataFrame(quality).to_csv(r / "tables/reference_center_strata_quality.csv", index=False, encoding="utf-8")
    print(f"标签几何诊断：{len(geometry)}个reference帧；不新增feature/方向/预算")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--source-only" in sys.argv:
        source_summary(DEFAULT_OUTPUT, json.loads((DEFAULT_OUTPUT / "audit/analysis_protocol.json").read_text(encoding="utf-8")))
    elif "--event-risk-only" in sys.argv:
        event_risk(DEFAULT_OUTPUT, json.loads((DEFAULT_OUTPUT / "audit/analysis_protocol.json").read_text(encoding="utf-8")))
    elif "--block-strata-only" in sys.argv:
        block_variable_strata(DEFAULT_OUTPUT)
    elif "--reference-geometry-only" in sys.argv:
        reference_geometry_audit(DEFAULT_OUTPUT)
    else:
        main()
