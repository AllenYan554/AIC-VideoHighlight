"""本地真实CSV→八张审计图；只使用已安装Matplotlib，不读取视频。"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(r"D:\CDUT\硕士\2026-09 AIC\实验记录\Stage6_对VHiCraft的优化\03_Stage6_2B_Task_Native_Signal_Audit")
SEED = 20260915
SCOPE = "reference_valid109"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#8C5AA6", "#4D4D4D", "#B58800"]
REPRESENTATIVES = {
    "temporal": ["interior_position", "support_ratio", "edge_distance_min"],
    "localization": ["primary_score", "fallback", "ambiguous"],
    "composition": ["ts5_subject_visible_fraction", "crop_constraint_pressure", "shift_x_abs"],
    "consistency": ["confidence_mean", "bbox_iou_persistence", "crop_velocity"],
}
METHODS = ["random_frames", "random_contiguous_interval", "native_frame_ranking", "native_atomic_block", "fixed_temporal_grid", "candidate_parent_region"]
METHOD_NAMES = ["随机帧（200次均值）", "随机连续区间", "原生逐帧排序", "原生边界atomic block", "固定1s grid", "全部candidate parents"]


def load(name, fields=None):
    with (ROOT / "tables" / name).open(encoding="utf-8", newline="") as f:
        return [{k: v for k, v in row.items() if fields is None or k in fields} for row in csv.DictReader(f)]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def a(rows, key):
    return np.array([num(x.get(key)) for x in rows])


def subset(rows, **conditions):
    return [x for x in rows if all(x.get(k) == str(v) for k, v in conditions.items())]


def save(fig, filename, caption, tables):
    destination = ROOT / "figures" / filename
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    manifests.append({"file": filename, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                      "dpi": 180, "caption_zh": caption, "source_tables": tables,
                      "source_sha256": {p: hashlib.sha256((ROOT / "tables" / p).read_bytes()).hexdigest() for p in tables},
                      "medium": "本地研究审计报告；非期刊投稿", "publisher_requirements": "NOT_APPLICABLE",
                      "uncertainty": "图中CI为2000次source-cluster percentile 95% CI；随机mask为200次Monte Carlo均值",
                      "missing": "NA不连线；不存在的guard数据不画；连续序列在source-frame间断处断开"})


def ordered_curve(rows, feature, method=None):
    g = subset(rows, scope=SCOPE, feature=feature)
    if method:
        g = subset(g, method=method)
    return sorted(g, key=lambda x: num(x["budget"]))


def main():
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                         "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7})
    dictionary = load("feature_dictionary.csv")
    discrimination = subset(load("feature_discrimination.csv"), scope=SCOPE)
    quality = subset(load("spatial_quality_signal.csv"), scope=SCOPE)
    contribution = subset(load("contribution_signal.csv"), scope=SCOPE)
    drops = load("drop_curve.csv")
    blocks = load("segment_block_comparison.csv")
    metrics = subset(load("per_video_signal_metrics.csv"), role=SCOPE)
    features = [r["feature"] for r in dictionary]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(13, len(features) * .23)), layout="constrained", sharey=True)
    for ax, rows, value, low, high, title, baseline in [
        (axes[0], discrimination, "per_video_auc_mean", "source_auc_mean_ci_low", "source_auc_mean_ci_high", "Y：video-equal AUROC", .5),
        (axes[1], quality, "per_video_rho_mean", "source_mean_ci_low", "source_mean_ci_high", "Q：video-equal Spearman（仅正例）", 0),
        (axes[2], contribution, "per_video_rho_mean", "source_mean_ci_low", "source_mean_ci_high", "Z：video-equal Spearman", 0),
    ]:
        lookup = {x["feature"]: x for x in rows}
        y = np.arange(len(features))
        vals = np.array([num(lookup[f].get(value)) for f in features])
        lo = np.array([num(lookup[f].get(low)) for f in features]);hi = np.array([num(lookup[f].get(high)) for f in features])
        ax.hlines(y, lo, hi, color=COLORS[0], alpha=.8)
        ax.scatter(vals, y, s=16, marker="o", color=COLORS[0])
        for j, v in enumerate(vals):
            if not math.isfinite(v):
                ax.text(baseline, j, "NA", ha="center", va="center", fontsize=7, color="#777777")
        ax.axvline(baseline, color="#555555", linestyle="--", linewidth=.8)
        ax.set(title=title, xlabel="机制预定义方向；source-cluster 95% CI")
        ax.set_yticks(y, features, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("完整原生信号清单：保留常数、缺失与负结果｜弱参考代理，非官方GT", fontsize=13)
    save(fig, "signal_overview.png", "全部注册feature的Y/Q/Z跨视频均值与source-cluster CI。NA表示不可计算，不表示零效应。完整负结果同时保留在CSV。", ["feature_dictionary.csv", "feature_discrimination.csv", "spatial_quality_signal.csv", "contribution_signal.csv"])

    reps = [f for fs in REPRESENTATIVES.values() for f in fs]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
    rng = np.random.default_rng(SEED)
    for ax, feature in zip(axes.flat, reps):
        rows = subset(metrics, feature=feature)
        vals = a(rows, "auroc");vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.scatter(rng.uniform(-.22, .22, len(vals)), vals, s=18, alpha=.6, color=COLORS[0])
            ax.boxplot(vals, positions=[0], widths=.55, showfliers=False, patch_artist=False)
        else:
            ax.text(.5, .5, "不可计算 / 常数", ha="center", transform=ax.transAxes)
        ax.axhline(.5, linestyle="--", color="#555555")
        ax.set(title=f"{feature}\n可计算视频 n={len(vals)}", ylim=(0, 1), xticks=[], ylabel="固定方向AUROC")
    fig.suptitle("预先固定的四family代表：逐视频分布，不按视频翻转方向", fontsize=13)
    save(fig, "per_video_signal_distribution.png", "12个事先固定的family代表，展示每个可计算视频的AUROC。抖动仅分开重叠点，固定seed。全清单见signal_overview及CSV。", ["per_video_signal_metrics.csv"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for ax, (family, fs) in zip(axes.flat, REPRESENTATIVES.items()):
        for j, f in enumerate(fs):
            rows = ordered_curve(drops, f)
            ax.plot(a(rows, "budget") * 100, a(rows, "FP_removed_fraction"), "o-", color=COLORS[j], label=f + "：FP")
            ax.plot(a(rows, "budget") * 100, a(rows, "TP_lost_fraction"), "x--", color=COLORS[j], label=f + "：TP")
        ax.set(title=family, xlabel="删除预算：OBSERVED eligibility内 %", ylabel="各自基线FP/TP的损失比例", ylim=(0, 1))
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle("固定5/10/20/30/50%预算：FP与TP分别用自身总量归一化", fontsize=13)
    save(fig, "fp_vs_tp_drop_curve.png", "同预算原生逐帧排序：实线FP removed fraction、虚线TP lost fraction；两条线分母不同，不能直接比较绝对数量。score结构缺失帧不impute，预算只在observed层。", ["drop_curve.csv"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for ax, (family, fs) in zip(axes.flat, REPRESENTATIVES.items()):
        ax.plot([0, .55], [0, .55], "--", color="#555555", label="贡献按unique成本同比损失")
        for j, f in enumerate(fs):
            rows = ordered_curve(drops, f)
            x, y = a(rows, "fraction_of_all_control_frames"), a(rows, "z_mass_lost_fraction")
            ax.plot(x, y, "o-", color=COLORS[j], label=f)
            ax.vlines(x, a(rows, "source_z_loss_ci_low"), a(rows, "source_z_loss_ci_high"), color=COLORS[j], alpha=.7)
        ax.set(title=family, xlabel="实际删除帧 / 该scope全部control帧", ylabel="ΣZ丢失 / 基线ΣZ", xlim=(0, .55), ylim=(0, 1))
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle("实际成本与贡献质量损失：点估计及source-cluster 95% CI", fontsize=13)
    save(fig, "contribution_mass_loss.png", "贡献损失比例与实际全部control帧成本比较，避免把结构缺失score的条件预算冒充全体预算。误差线是相同pooled ΣZ比值的source-cluster CI；所有五预算保留。", ["drop_curve.csv", "contribution_mass_curve.csv"])

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), layout="constrained")
    for ax, f in zip(axes.flat, reps):
        for j, method in enumerate(METHODS):
            rows = ordered_curve(blocks, f, method)
            x = a(rows, "budget") * 100;y = a(rows, "macro_delta_F")
            ax.plot(x, y, marker=["o", "x", "s", "^", "D", "+"][j], color=COLORS[j], linewidth=1., label=METHOD_NAMES[j])
        ax.axhline(0, color="#555555", linestyle="--", linewidth=.7)
        ax.set(title=f, xlabel="相同eligible unique-frame预算 %", ylabel="弱参考macro ΔF")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("结构额外价值：同视频、同eligibility、同精确unique成本，六种对照", fontsize=13)
    save(fig, "segment_vs_frame_budget.png", "六种对照在12个预先固定代表上的全部预算。每次mask严格unique预算匹配；随机200次均值。单component不足时随机连续区间使用已登记的component内多区间方案；完整CI与paired contrasts见CSV。", ["segment_block_comparison.csv", "block_frame_paired_contrasts.csv"])

    quant = subset(load("signal_quantiles.csv"), scope=SCOPE, feature="interior_position")
    state = load("target_state_strata.csv")
    hist = load("historical_support_edge_reaudit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for key, label, color in [("y_mean", "Y membership", COLORS[0]), ("q_mean", "Q（仅正例）", COLORS[1]), ("z_mean", "Z contribution", COLORS[2])]:
        axes[0, 0].plot(a(quant, "rank_quantile"), a(quant, key), "o-", color=color, label=label)
    axes[0, 0].set(title="candidate位置：从边缘→内部的五个rank层", xlabel="interior_position rank quintile", ylabel="弱参考均值", ylim=(0, 1))
    axes[0, 0].legend()
    supp = subset(state, stratifier="support_stratum")
    axes[0, 1].bar(np.arange(len(supp)), a(supp, "Z_mean"), color=COLORS[0])
    axes[0, 1].set(xticks=np.arange(len(supp)), xticklabels=[r["stratum"] + "\nn=" + r["n"] for r in supp], title="cross-chunk曝光与support：完整层", ylabel="Z均值", ylim=(0, 1))
    for j, arm in enumerate(["historical_fs1_edge", "historical_fs2_edge"]):
        g = subset(hist, method=arm)
        axes[1, 0].bar(j - .17, np.nansum(a(g, "fp_removed")), width=.34, color=COLORS[0], label="FP" if j == 0 else None)
        axes[1, 0].bar(j + .17, np.nansum(a(g, "tp_lost")), width=.34, color=COLORS[1], hatch="//", label="弱参考positive" if j == 0 else None)
    axes[1, 0].set(xticks=[0, 1], xticklabels=["历史FS1", "历史FS2"], title="当前frozen control上重新应用历史定义", ylabel="删除帧绝对数量")
    axes[1, 0].legend()
    control = load("support_position_control.csv")
    values = a(control, "delta_z_unanimous_minus_non");values = values[np.isfinite(values)]
    if len(values):
        axes[1, 1].scatter(np.arange(len(values)), values, marker="o", color=COLORS[0])
    axes[1, 1].axhline(0, color="#555555", linestyle="--")
    axes[1, 1].set(title=f"E≥2且控制video/位置层：可比较cell n={len(values)}", xlabel="可比较video×position cells", ylabel="unanimous−nonunanimous的Z均值差")
    fig.suptitle("候选边缘与Stage5.5 support：低触发与位置混杂均需披露", fontsize=13)
    save(fig, "candidate_edge_analysis.png", "位置五层来自标签无关feature rank，非部署trim比例。支持层报告全部曝光与样本量；历史FS1/FS2在当前control重算。控制video及位置后若无可比较cell，显示n=0，不能声称独立support作用。", ["signal_quantiles.csv", "target_state_strata.csv", "historical_support_edge_reaudit.csv", "support_position_control.csv"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    for ax, stratifier in zip(axes, ["fallback", "ambiguous", "primary_label"]):
        rows = subset(state, stratifier=stratifier)
        if stratifier == "primary_label":
            grouped = []
            for label, selected in [("person", [r for r in rows if r["stratum"] == "person"]),
                                    ("non-person", [r for r in rows if r["stratum"] not in {"person", "STRUCTURAL_MISSING"}]),
                                    ("primary absent", [r for r in rows if r["stratum"] == "STRUCTURAL_MISSING"])]:
                n, qn = np.nansum(a(selected, "n")), np.nansum(a(selected, "Q_n"))
                grouped.append({"stratum": label, "n": str(int(n)), "Q_n": str(int(qn)),
                                "Y_mean": str(np.nansum(a(selected, "Y_mean") * a(selected, "n")) / n) if n else "",
                                "Q_mean": str(np.nansum(a(selected, "Q_mean") * a(selected, "Q_n")) / qn) if qn else "",
                                "Z_mean": str(np.nansum(a(selected, "Z_mean") * a(selected, "n")) / n) if n else ""})
            rows = grouped
        xs = np.arange(len(rows))
        for j, (key, label) in enumerate([("Y_mean", "Y"), ("Q_mean", "Q：仅正例"), ("Z_mean", "Z")]):
            ax.bar(xs + (j - 1) * .24, a(rows, key), width=.24, color=COLORS[j], hatch=["", "//", ".."][j], label=label)
        ax.set(xticks=xs, xticklabels=[r["stratum"] + "\nn=" + r["n"] + ", Qn=" + r["Q_n"] for r in rows], ylim=(0, 1), title=stratifier, ylabel="弱参考代理均值")
        ax.legend()
    fig.suptitle("定位状态：membership、conditional IoU和零扩展贡献分别展示", fontsize=13)
    save(fig, "localization_state_analysis.png", "PRIMARY/fallback、ambiguous/clean及person/non-person/primary absent的Y/Q/Z分布。全部真实类别合并成预定义person/non-person，完整类别仍保留CSV。Q只包含positive，Qn单独报告；fallback的primary score不存在，不填0。", ["target_state_strata.csv", "missing_target_strata.csv"])

    timeline_fields = {"video_id", "source_id", "frame", "timestamp", "role", "Y", "Q", "Z", "E_t", "S_t", "primary_score", "fallback", "ambiguous", "interior_position", "ts5_subject_visible_fraction", "tie_hash", "crop_constraint_pressure"}
    frames_data = [x for x in load("frame_features_targets.csv", timeline_fields) if x["role"] == SCOPE]
    byvideo = {}
    for row in frames_data:
        byvideo.setdefault(row["video_id"], []).append(row)
    membership = load("candidate_multi_membership.csv", {"video_id", "frame", "parent_id"})
    flookup = {(r["video_id"], int(r["frame"])): r for r in frames_data}
    parentrows = {}
    for row in membership:
        key = row["video_id"], row["parent_id"]
        fkey = row["video_id"], int(row["frame"])
        if fkey in flookup:
            parentrows.setdefault(key, []).append(flookup[fkey])
    high = max(parentrows.items(), key=lambda kv: (np.mean(a(kv[1], "Z")), len(kv[1]), kv[0]))[0][0]
    low = min(parentrows.items(), key=lambda kv: (np.mean(a(kv[1], "Z")), -len(kv[1]), kv[0]))[0][0]
    detail = load("per_video_drop_metrics.csv", {"video_id", "feature", "budget", "method", "delta_f"})
    misleading_rows = [r for r in detail if r["video_id"] in byvideo and r["feature"] == "primary_score" and r["method"] == "native_frame_ranking" and math.isclose(num(r["budget"]), .1)]
    misleading = min(misleading_rows, key=lambda x: (num(x["delta_f"]), x["video_id"]))["video_id"]
    def longest_fp(rows):
        longest = current = 0;previous = None
        for r in rows:
            frame = int(r["frame"])
            if num(r["Y"]) == 0:
                current = current + 1 if previous is not None and frame == previous + 1 else 1
                longest = max(longest, current)
            else:
                current = 0
            previous = frame
        return longest
    severe = max(byvideo, key=lambda v: (longest_fp(byvideo[v]), v))
    ev = [r for r in load("reference_events.csv") if r["video_id"] in byvideo and num(r["baseline_matched"]) > 0]
    shortcase = min(ev, key=lambda r: (num(r["duration_sec"]), r["video_id"], num(r["event_id"])))["video_id"]
    cases = [("高贡献parent（最大mean Z）", high), ("低贡献parent（最小mean Z，长区间优先）", low),
             ("误导signal：固定primary_score/10%最差ΔF", misleading), ("严重weak FP：最长连续Y=0区域", severe),
             ("短事件风险：最短有预测覆盖reference run", shortcase)]
    fig, axes = plt.subplots(5, 3, figsize=(20, 15), layout="constrained")
    case_manifest = []
    for i, (case, video) in enumerate(cases):
        rows = sorted(byvideo[video], key=lambda r: int(r["frame"]))
        t, fs = a(rows, "timestamp"), a(rows, "frame")
        gaps = np.r_[False, np.diff(fs) != 1]
        def broken(key):
            vals = a(rows, key).copy();vals[gaps] = np.nan;return vals
        for key, color, label in [("E_t", COLORS[0], "E_t eligible chunks"), ("S_t", COLORS[1], "S_t supported chunks")]:
            axes[i, 0].plot(t, broken(key), marker=".", markersize=2, color=color, label=label)
        axes[i, 0].set(title=case + "\n" + video, xlabel="canonical source timestamp / s", ylabel="chunk数量")
        axes[i, 0].legend(loc="upper right")
        for key, color, label, style in [("primary_score", COLORS[0], "primary score", "-"), ("ts5_subject_visible_fraction", COLORS[2], "TS5主体可见率", "-"),
                                         ("fallback", COLORS[1], "fallback", "--"), ("ambiguous", COLORS[3], "ambiguous", ":"), ("interior_position", COLORS[4], "parent interior位置", "-")]:
            axes[i, 1].plot(t, broken(key), linestyle=style, linewidth=.8, color=color, label=label)
        axes[i, 1].set(ylim=(-.05, 1.05), xlabel="时间 / s", ylabel="真实native features（NA断线）")
        axes[i, 1].legend(loc="upper right", fontsize=6)
        for key, color, label, style in [("Y", COLORS[0], "Y weak membership", "--"), ("Q", COLORS[1], "Q conditional IoU", "-"), ("Z", COLORS[2], "Z contribution", "-")]:
            axes[i, 2].plot(t, broken(key), linestyle=style, linewidth=.9, color=color, label=label)
        # 每个case固定展示同一个interior_position/10%诊断mask，不根据case挑预算。
        x = a(rows, "interior_position")
        h = np.array([int(r["tie_hash"]) for r in rows], dtype=np.uint64)
        k = int(math.floor(len(rows) * .1))
        ix = np.lexsort((h, x))[:k]
        axes[i, 2].scatter(t[ix], np.full(len(ix), -.035), s=7, marker="|", color="#333333", label="固定interior/10%删除mask")
        axes[i, 2].set(ylim=(-.07, 1.05), xlabel="时间 / s", ylabel="弱参考targets；非官方GT")
        axes[i, 2].legend(loc="upper right", fontsize=6)
        case_manifest.append({"case": case, "video_id": video, "selection_definition": case, "displayed_mask": "interior_position/native_frame/10%"})
    fig.suptitle("五类真实timeline：包含失败案例和最短事件，不重解码；所有曲线在frame断点断开", fontsize=14)
    save(fig, "representative_timelines.png", "案例按已披露的极值规则抽取：高/低贡献parent、固定score/10%的最差ΔF、最长Y=0区间、最短有覆盖reference run。全视频范围显示，Q负例NA。每例统一画interior_position/10%mask，仅作诊断。", ["frame_features_targets.csv", "candidate_multi_membership.csv", "per_video_drop_metrics.csv", "reference_events.csv"])
    (ROOT / "audit/timeline_case_manifest.json").write_text(json.dumps(case_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": "aic.stage6_2b.figure-provenance/v1", "seed": SEED, "python": sys.version,
                "matplotlib": matplotlib.__version__, "numpy": np.__version__, "figures": manifests,
                "review_status": "PENDING_VISUAL_REVIEW", "raw_inputs_modified": False}
    (ROOT / "audit/figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    captions = ["# Stage6.2B 真实图表说明", "", "全部图来自恢复control的派生CSV；参考为teacher/seed weak reference。配套CSV提供完整数值与缺失状态。"]
    for m in manifests:
        captions += ["", "## " + m["file"], "", m["caption_zh"]]
    (ROOT / "figures/captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    print("八张真实数据图已生成", flush=True)


manifests = []
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
