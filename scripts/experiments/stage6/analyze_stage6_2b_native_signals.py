"""Stage6.2B 完整只读原生信号审计；所有mask仅为POSTHOC_DIAGNOSTIC_MASK。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from verify_stage6_2b_inputs import DEFAULT_OUTPUT, RUN, PRED_SHA, REF_SHA, verify, sha, project

SEED = 20260915
BOOTSTRAPS = 2000
RANDOM_REPLICATES = 200
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
RADIUS_SEC = 0.5
GRID_SEC = 1.0
IDENTITY = "POSTHOC_DIAGNOSTIC_ONLY"

# 定义在任何Y/Q/Z关系计算前冻结。direction=+1表示原值较大预期贡献较高，-1相反。
SPECS = []


def spec(name, family, definition, unit="1", direction=1, missing="OBSERVED", available=True):
    SPECS.append(dict(feature=name, family=family, definition=definition, unit=unit, direction=direction,
                      missing_semantics=missing, derived=True, safe_for_inference=available, available=available))


for name, definition, unit, direction in [
    ("raw_score_mean", "覆盖该帧raw spans的score均值；无raw覆盖为结构缺失", "1", 1),
    ("raw_score_max", "覆盖该帧raw spans的最大score", "1", 1),
    ("merged_score_mean", "全部包含该帧merged parents的score均值", "1", 1),
    ("merged_score_max", "全部包含该帧merged parents的最大score", "1", 1),
    ("candidate_duration_mean", "全部所属merged parents的duration均值", "s", 1),
    ("distance_start_min", "到全部所属parent开始时间距离的最小值", "s", 1),
    ("distance_end_min", "到全部所属parent结束时间距离的最小值", "s", 1),
    ("edge_distance_min", "到任一所属parent最近边缘的秒数", "s", 1),
    ("normalized_position_mean", "所属parents内(start距离/duration)均值；有符号位置仅描述", "1", 1),
    ("interior_position", "所属parents的2*min(position,1-position)均值", "1", 1),
    ("parent_count", "merged candidate多归属数量，不选择单个父candidate", "count", 1),
    ("raw_overlap_count", "投影覆盖该帧的raw candidate数量", "count", 1),
    ("contributor_count_mean", "所属parents的contributor raw id数量均值", "count", 1),
    ("raw_span_disagreement", "所属parents内contributor两两1-temporalIoU的均值；单contributor为0", "1", -1),
    ("score_dispersion", "所属parents内contributor score标准差的均值", "1", -1),
    ("merge_only_expansion", "属于merged union但不在任何raw投影内为1", "bool", -1),
    ("E_t", "原始chunk窗口覆盖该帧的不同chunk数；曝光量非动作证据", "count", 1),
    ("S_t", "eligible chunks中至少一个raw span覆盖该帧的不同chunk数", "count", 1),
    ("support_ratio", "S_t/E_t；E=0为结构缺失", "1", 1),
    ("chunk_overlap_consistency", "S_t/E_t的语义重复列，保留但不算独立证据", "1", 1),
    ("support_deficit", "E_t-S_t", "count", -1),
    ("historical_fs1_edge", "冻结FS1定义：有S>=2 core，E>=2且S=1的边缘连续run；多parent KEEP并集", "bool", -1),
    ("historical_fs2_edge", "冻结FS2定义：有unanimous core，E>=2且S<E的边缘run；多parent KEEP并集", "bool", -1),
]:
    spec(name, "temporal", definition, unit, direction,
         "STRUCTURAL_MISSING:no_raw_coverage" if name.startswith("raw_score") else
         "STRUCTURAL_MISSING:E_t=0" if name in {"support_ratio", "chunk_overlap_consistency"} else "OBSERVED")

for name, definition, unit, direction, missing in [
    ("is_primary", "LOC status=PRIMARY", "bool", 1, "OBSERVED"),
    ("fallback", "LOC status!=PRIMARY", "bool", -1, "OBSERVED"),
    ("primary_score", "真实LOC primary.score；fallback不填0", "1", 1, "STRUCTURAL_MISSING:primary_absent"),
    ("is_person", "在有primary时label=person", "bool", 1, "STRUCTURAL_MISSING:primary_absent"),
    ("candidate_count", "LOC保存的全部detector候选数；不叫可靠候选数", "count", 1, "OBSERVED"),
    ("invalid_candidate_count", "LOC保存的无效候选数", "count", -1, "OBSERVED"),
    ("ambiguous", "LOC保存的ambiguous bool", "bool", -1, "OBSERVED"),
    ("ambiguous_candidate_count", "LOC保存的ambiguous_candidate_count", "count", -1, "OBSERVED"),
    ("bbox_area_ratio", "有效sanitized primary xyxy面积/(W*H)", "1", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("bbox_center_x", "sanitized primary中心x/W；有符号位置仅描述", "1", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("bbox_center_y", "sanitized primary中心y/H；有符号位置仅描述", "1", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("bbox_center_extremeness", "abs(cx/W-0.5)+abs(cy/H-0.5)", "1", -1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("bbox_edge_distance", "primary到image四边最小距离/min(W,H)", "1", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("sanitization_amount", "四个sanitized clamp量之和/min(W,H)", "1", -1, "STRUCTURAL_MISSING:primary_absent"),
    ("neighbor_bbox_iou", "仅连续source frame且两帧有效primary的sanitized bbox IoU", "1", 1, "STRUCTURAL_MISSING:neighbor_or_primary_absent"),
    ("center_velocity", "连续primary中心的归一化位移/真实时间差", "s^-1", -1, "STRUCTURAL_MISSING:neighbor_or_primary_absent"),
    ("center_acceleration", "连续三个有效primary的归一化速度向量差/时间差范数", "s^-2", -1, "STRUCTURAL_MISSING:neighbor_or_primary_absent"),
    ("scale_velocity", "连续primary的abs(log面积比)/dt", "s^-1", -1, "STRUCTURAL_MISSING:neighbor_or_primary_absent"),
    ("fallback_run", "连续source-frame fallback run全长秒数；正常帧0", "s", -1, "OBSERVED"),
    ("ambiguous_run", "连续source-frame ambiguous run全长秒数；clean帧0", "s", -1, "OBSERVED"),
]:
    spec(name, "localization", definition, unit, direction, missing)

for name, definition, unit, direction, missing in [
    ("subject_visible_fraction", "保存的CMP1主体可见率；不是reference IoU", "1", 1, "STRUCTURAL_MISSING:primary_absent"),
    ("subject_center_inside", "primary存在时保存的CMP1主体中心在crop内", "bool", 1, "STRUCTURAL_MISSING:primary_absent"),
    ("horizontal_center_offset_abs", "abs(CMP horizontal_center_offset)", "1", -1, "STRUCTURAL_MISSING:primary_absent"),
    ("shift_x_abs", "abs(CMP1 shift_x_norm)", "1", -1, "OBSERVED"),
    ("shift_y_abs", "abs(CMP1 shift_y_norm)", "1", -1, "OBSERVED"),
    ("clamped_x", "CMP1 clamped_x", "bool", -1, "OBSERVED"),
    ("clamped_y", "CMP1 clamped_y", "bool", -1, "OBSERVED"),
    ("final_crop_edge_margin", "frozen TS5 crop到image四边最小margin/min(W,H)", "1", 1, "OBSERVED"),
    ("ts5_subject_visible_fraction", "sanitized主体面积中位于frozen TS5 crop的比例；不是reference IoU", "1", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("ts5_subject_center_inside", "有效primary中心是否在frozen TS5 crop内", "bool", 1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
    ("crop_constraint_pressure", "(1-TS5主体可见率)+abs(CMP shifts)+(ideal到clamped位置距离归一化)", "1", -1, "STRUCTURAL_MISSING:primary_absent_or_invalid"),
]:
    spec(name, "composition", definition, unit, direction, missing)

for name, definition, unit, direction, missing in [
    ("confidence_mean", "同一候选连续component内±0.5s有效primary score均值", "1", 1, "STRUCTURAL_MISSING:no_local_primary"),
    ("confidence_median", "同一component内±0.5s有效primary score中位数", "1", 1, "STRUCTURAL_MISSING:no_local_primary"),
    ("confidence_std", "同一component内±0.5s至少2个有效primary score的总体std", "1", -1, "STRUCTURAL_MISSING:insufficient_local_primary"),
    ("confidence_persistence", "同一component内±0.5s primary帧占比", "1", 1, "OBSERVED"),
    ("bbox_iou_persistence", "同一component内±0.5s有效neighbor bbox IoU均值", "1", 1, "STRUCTURAL_MISSING:no_local_bbox_pairs"),
    ("bbox_iou_std", "同一component内±0.5s至少2个有效neighbor bbox IoU的std", "1", -1, "STRUCTURAL_MISSING:insufficient_local_bbox_pairs"),
    ("center_velocity_mean", "同一component内±0.5s有效center velocity均值", "s^-1", -1, "STRUCTURAL_MISSING:no_local_bbox_pairs"),
    ("center_velocity_std", "同一component内±0.5s至少2个有效center velocity的std", "s^-1", -1, "STRUCTURAL_MISSING:insufficient_local_bbox_pairs"),
    ("crop_velocity", "连续TS5归一化crop中心向量的速度范数", "s^-1", -1, "STRUCTURAL_MISSING:component_start"),
    ("crop_acceleration", "连续三个TS5 crop的速度向量差/dt范数", "s^-2", -1, "STRUCTURAL_MISSING:component_first_two"),
    ("crop_scale_velocity", "连续TS5 crop的abs(log宽度比)/dt", "s^-1", -1, "STRUCTURAL_MISSING:component_start"),
    ("isolated_anomaly", "fallback或ambiguous的严格单source-frame run", "bool", -1, "OBSERVED"),
    ("geometry_change_proxy", "center velocity与同一component内±0.5s velocity中位数之差绝对值；非shot detector", "s^-1", -1, "STRUCTURAL_MISSING:no_local_bbox_pairs"),
]:
    spec(name, "consistency", definition, unit, direction, missing)

for name, family, definition in [
    ("reliable_candidate_count", "localization", "完整detector候选/严格reliability规则未保存，无法从candidate_count推断"),
    ("primary_secondary_coexistence", "composition", "policy表没有完整secondary bbox集合，不能严格派生"),
    ("exact_ts5_guard_correction", "consistency", "pipeline未序列化精确逐帧guard correction，禁止TS5-TS0替代"),
    ("subject_identity_continuity", "consistency", "没有track ID，geometry consistency不能叫subject identity continuity"),
    ("shot_boundary", "consistency", "没有真实shot detector，geometry change proxy不叫shot boundary"),
]:
    spec(name, family, definition, direction=0, missing="UNAVAILABLE", available=False)


def dump_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def log(text):
    print(text, flush=True)


def write_table(output, name, rows):
    path = output / "tables" / name
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def xywh_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0., min(ax + aw, bx + bw) - max(ax, bx)) * max(0., min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.


def component_ids(frames):
    return np.cumsum(np.r_[True, np.diff(frames) != 1]) - 1


def run_lengths(flags, frames, fps):
    result = np.zeros(len(flags))
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < len(flags) and flags[j] and frames[j] == frames[j - 1] + 1:
            j += 1
        result[i:j] = (j - i) / fps
        i = j
    return result


def tied_hash(video, frame):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{video}:{int(frame)}".encode()).digest()[:8], "big")


def freeze_protocol(output, data):
    reasons = sorted({r for rows in data["loc"].values() for row in rows.values() for r in row["fallback_reasons"]})
    for reason in reasons:
        spec("reason_" + reason.lower(), "localization", f"LOC fallback_reasons包含{reason}；PRIMARY为0", "bool", -1)
    protocol = {"schema_version": "aic.stage6_2b.native-audit-protocol/v1", "identity": IDENTITY,
                "run_id": RUN, "predictions_sha256": PRED_SHA, "reference_sha256": REF_SHA,
                "seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "random_mask_replicates": RANDOM_REPLICATES,
                "budgets": BUDGETS, "neighborhood_radius_sec": RADIUS_SEC, "grid_sec": GRID_SEC,
                "primary_scope": "reference_valid109", "sensitivity_scope": "all166_weak_label_sensitivity",
                "source_cluster": "source_ytid (比source_group更保守，所有同YouTube源一起重采样)",
                "directions": "机制预定义，全视频同方向；有符号position/x/y只描述，不参与Gate证据",
                "tie_break": "label-blind SHA256(seed:video_id:frame)，禁止按frame先后处理常数feature",
                "missing": "不impute；连续score预算只在OBSERVED eligibility内，所有对照同eligibility+unique budget",
                "quantiles": "5个rank-equal组，ties用同label-blind hash；constant不解释quantile为信息",
                "regions": "D: candidate/raw/chunk边界atomic regions；E:源时间0锚定1s grid；F:所有candidate parents按novel unique cost贪心",
                "partial_region": "最后region不足完整预算时固定取其中心连续piece；不根据Y/Q/Z选边界",
                "random_intervals": "若一个component可容纳K，均匀抽所有合法长度K起点；否则按component长度分配精确K，逐component随机interval",
                "uncertainty": "2000次video/source cluster重采样；pooled AUROC用精确cluster pair kernel；AP/Q/Z pooled为描述点估计，CI用于对应video-equal均值",
                "multiple_comparisons": "探索性完整family census；无p值、无确认性显著性结论；不选feature/window/budget进入production",
                "short_event": "reference-only连续run按within-video长度排序，最短四分位为风险层；不是部署minimum duration",
                "specifications": SPECS}
    p = output / "audit/analysis_protocol.json"
    if p.exists():
        previous = json.loads(p.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(protocol)):
            raise ValueError("已冻结protocol不一致，不能看结果后更改定义")
    else:
        dump_json(p, protocol)
    return protocol


def edge_diagnostic_flags(E, S, parents, method):
    kept = np.zeros(len(E), dtype=bool)
    for indices in parents:
        a = np.array(sorted(indices), dtype=int)
        if not len(a):
            continue
        core = np.any(S[a] >= 2) if method == 1 else np.any((E[a] >= 2) & (S[a] == E[a]))
        drop = (E[a] >= 2) & ((S[a] == 1) if method == 1 else (S[a] < E[a]))
        local_keep = np.ones(len(a), dtype=bool)
        if core:
            left = 0
            while left < len(a) and drop[left]:
                local_keep[left] = False
                left += 1
            right = len(a) - 1
            while right >= 0 and drop[right]:
                local_keep[right] = False
                right -= 1
        kept[a[local_keep]] = True
    return (~kept).astype(float)


def extract(data):
    records, memberships, blocks = [], [], {}
    reason_names = [x["feature"] for x in SPECS if x["feature"].startswith("reason_")]
    reason_lookup = {r.lower(): r for rows in data["loc"].values() for row in rows.values() for r in row["fallback_reasons"]}
    for v in sorted(data["pred"]):
        pred = data["pred"][v]
        meta, cache = data["meta"][v], data["candidates"][v]
        frames = np.array([x["frame"] for x in pred["predictions"]], dtype=int)
        if not len(frames):
            blocks[v] = {"parents": [], "atomic": [], "grid": []}
            continue
        n, fps = len(frames), meta["fps"]
        times = frames / fps
        comp = component_ids(frames)
        lookup = {int(f): i for i, f in enumerate(frames)}
        raw = cache["raw_candidates"]
        raw_by_id = {x["raw_candidate_id"]: x for x in raw}
        rawsets = [{lookup[f] for f in project(x["start_sec"], x["end_sec"], meta) if f in lookup} for x in raw]
        parents = cache["merged_candidates"]
        parentsets = [{lookup[f] for f in project(x["start_sec"], x["end_sec"], meta) if f in lookup} for x in parents]
        chunks = cache["source_chunks"]
        chunksets = {c["chunk_index"]: {lookup[f] for f in project(c["chunk_start_sec"], c["chunk_end_sec"], meta) if f in lookup} for c in chunks}
        supported = {k: set() for k in chunksets}
        for r, indices in zip(raw, rawsets):
            supported[r["chunk_index"]].update(indices)
        E, S = np.zeros(n), np.zeros(n)
        for c, indices in chunksets.items():
            for i in indices:
                E[i] += 1
                S[i] += i in supported[c]
        fs1, fs2 = edge_diagnostic_flags(E, S, parentsets, 1), edge_diagnostic_flags(E, S, parentsets, 2)
        rows, boxes, cropcenters = [], [], []
        for i, p in enumerate(pred["predictions"]):
            f = p["frame"]
            row = {x["feature"]: np.nan for x in SPECS}
            belong = [j for j, indices in enumerate(parentsets) if i in indices]
            raw_belong = [j for j, indices in enumerate(rawsets) if i in indices]
            ps = [parents[j] for j in belong]
            rs = [raw[j] for j in raw_belong]
            positions = [(times[i] - a["start_sec"]) / (a["end_sec"] - a["start_sec"]) for a in ps]
            disp, disagree = [], []
            for a in ps:
                contributors = [raw_by_id[x] for x in a["contributor_raw_candidate_ids"]]
                disp.append(float(np.std([x["score"] for x in contributors])))
                pairs = []
                for j, x in enumerate(contributors):
                    for y in contributors[j + 1:]:
                        intersection = max(0., min(x["end_sec"], y["end_sec"]) - max(x["start_sec"], y["start_sec"]))
                        union = max(x["end_sec"], y["end_sec"]) - min(x["start_sec"], y["start_sec"])
                        pairs.append(1 - intersection / union)
                disagree.append(float(np.mean(pairs)) if pairs else 0.)
            row.update(raw_score_mean=float(np.mean([x["score"] for x in rs])) if rs else np.nan,
                       raw_score_max=max((x["score"] for x in rs), default=np.nan),
                       merged_score_mean=float(np.mean([x["score"] for x in ps])), merged_score_max=max(x["score"] for x in ps),
                       candidate_duration_mean=float(np.mean([x["end_sec"] - x["start_sec"] for x in ps])),
                       distance_start_min=min(times[i] - x["start_sec"] for x in ps),
                       distance_end_min=min(x["end_sec"] - times[i] for x in ps),
                       edge_distance_min=min(min(times[i] - x["start_sec"], x["end_sec"] - times[i]) for x in ps),
                       normalized_position_mean=float(np.mean(positions)), interior_position=float(np.mean([2 * min(x, 1 - x) for x in positions])),
                       parent_count=len(ps), raw_overlap_count=len(rs), contributor_count_mean=float(np.mean([len(x["contributor_raw_candidate_ids"]) for x in ps])),
                       raw_span_disagreement=float(np.mean(disagree)), score_dispersion=float(np.mean(disp)), merge_only_expansion=float(not rs),
                       E_t=E[i], S_t=S[i], support_ratio=S[i] / E[i] if E[i] else np.nan,
                       chunk_overlap_consistency=S[i] / E[i] if E[i] else np.nan, support_deficit=E[i] - S[i],
                       historical_fs1_edge=fs1[i], historical_fs2_edge=fs2[i])
            for j in belong:
                memberships.append({"video_id": v, "frame": f, "parent_id": parents[j]["merged_candidate_id"],
                                    "parent_start_sec": parents[j]["start_sec"], "parent_end_sec": parents[j]["end_sec"], "parent_score": parents[j]["score"]})
            l, c, t = data["loc"][v][f], data["cmp"][v, f], data["ts"][v][f]["ts5"]
            primary, W, H = l.get("primary"), meta["width"], meta["height"]
            g = c["cmp1"]
            row.update(is_primary=float(primary is not None), fallback=float(primary is None),
                       primary_score=primary["score"] if primary else np.nan, is_person=float(primary["label"] == "person") if primary else np.nan,
                       candidate_count=l["candidate_count"], invalid_candidate_count=l["invalid_candidate_count"],
                       ambiguous=float(l["ambiguous"]), ambiguous_candidate_count=l["ambiguous_candidate_count"],
                       shift_x_abs=abs(g["shift_x_norm"]), shift_y_abs=abs(g["shift_y_norm"]), clamped_x=float(g["clamped_x"]), clamped_y=float(g["clamped_y"]),
                       final_crop_edge_margin=min(t["x"], t["y"], W - t["x"] - t["w"], H - t["y"] - t["h"]) / min(W, H))
            for name in reason_names:
                row[name] = float(reason_lookup[name[len("reason_"):]] in l["fallback_reasons"])
            box = None
            if primary:
                x0, y0, x1, y1 = c["sanitized"]["xyxy"]
                if x1 > x0 and y1 > y0:
                    box = [x0, y0, x1 - x0, y1 - y0]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    area = box[2] * box[3]
                    crop = [t["x"], t["y"], t["w"], t["h"]]
                    intersection = max(0., min(x1, crop[0] + crop[2]) - max(x0, crop[0])) * max(0., min(y1, crop[1] + crop[3]) - max(y0, crop[1]))
                    visible = intersection / area
                    row.update(bbox_area_ratio=area / (W * H), bbox_center_x=cx / W, bbox_center_y=cy / H,
                               bbox_center_extremeness=abs(cx / W - .5) + abs(cy / H - .5), bbox_edge_distance=min(x0, y0, W - x1, H - y1) / min(W, H),
                               sanitization_amount=sum(c["sanitized"].get("clamp_" + k, 0.) for k in ["left", "right", "top", "bottom"]) / min(W, H),
                               subject_visible_fraction=g["subject_visible_fraction"], subject_center_inside=float(g["subject_center_inside"]),
                               horizontal_center_offset_abs=abs(c["horizontal_center_offset"]) if c["horizontal_center_offset"] is not None else np.nan,
                               ts5_subject_visible_fraction=visible,
                               ts5_subject_center_inside=float(crop[0] <= cx <= crop[0] + crop[2] and crop[1] <= cy <= crop[1] + crop[3]),
                               crop_constraint_pressure=1 - visible + abs(g["shift_x_norm"]) + abs(g["shift_y_norm"]) + abs(g["ideal_x"] - g["x"]) / W + abs(g["ideal_y"] - g["y"]) / H)
            boxes.append(box)
            cropcenters.append([(t["x"] + t["w"] / 2) / W, (t["y"] + t["h"] / 2) / H])
            row.update(video_id=v, source_id=data["selected"][v]["source_ytid"], frame=f, timestamp=times[i], component=int(comp[i]),
                       role="reference_valid109" if v in data["valid_ids"] else "empty_reference57", tie_hash=tied_hash(v, f),
                       primary_label=primary["label"] if primary else "STRUCTURAL_MISSING")
            rows.append(row)
        fallback = np.array([x["fallback"] for x in rows], dtype=bool)
        ambiguous = np.array([x["ambiguous"] for x in rows], dtype=bool)
        fr, ar = run_lengths(fallback, frames, fps), run_lengths(ambiguous, frames, fps)
        velocities, cropvelocities = {}, {}
        for i, row in enumerate(rows):
            row["fallback_run"], row["ambiguous_run"] = fr[i], ar[i]
            row["isolated_anomaly"] = float((fallback[i] and math.isclose(fr[i] * fps, 1.)) or (ambiguous[i] and math.isclose(ar[i] * fps, 1.)))
            if i and frames[i] == frames[i - 1] + 1:
                dt = times[i] - times[i - 1]
                cropvelocities[i] = (np.array(cropcenters[i]) - cropcenters[i - 1]) / dt
                row["crop_velocity"] = float(np.linalg.norm(cropvelocities[i]))
                row["crop_scale_velocity"] = abs(math.log(data["ts"][v][int(frames[i])]["ts5"]["w"] / data["ts"][v][int(frames[i - 1])]["ts5"]["w"])) / dt
                if i - 1 in cropvelocities:
                    row["crop_acceleration"] = float(np.linalg.norm(cropvelocities[i] - cropvelocities[i - 1]) / dt)
                if boxes[i] is not None and boxes[i - 1] is not None:
                    row["neighbor_bbox_iou"] = xywh_iou(boxes[i], boxes[i - 1])
                    velocities[i] = (np.array([row["bbox_center_x"], row["bbox_center_y"]]) - [rows[i - 1]["bbox_center_x"], rows[i - 1]["bbox_center_y"]]) / dt
                    row["center_velocity"] = float(np.linalg.norm(velocities[i]))
                    row["scale_velocity"] = abs(math.log(boxes[i][2] * boxes[i][3] / (boxes[i - 1][2] * boxes[i - 1][3]))) / dt
                    if i - 1 in velocities:
                        row["center_acceleration"] = float(np.linalg.norm(velocities[i] - velocities[i - 1]) / dt)
        for i, row in enumerate(rows):
            ix = np.flatnonzero((comp == comp[i]) & (np.abs(times - times[i]) <= RADIUS_SEC + 1e-12))
            row["confidence_persistence"] = float(np.mean([rows[j]["is_primary"] for j in ix]))
            for source, targets in [("primary_score", ["confidence_mean", "confidence_median", "confidence_std"]),
                                    ("neighbor_bbox_iou", ["bbox_iou_persistence", "bbox_iou_std"]),
                                    ("center_velocity", ["center_velocity_mean", "center_velocity_std"])]:
                values = np.array([rows[j][source] for j in ix], dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    row[targets[0]] = float(np.mean(values))
                    if len(targets) == 3:
                        row[targets[1]] = float(np.median(values))
                    if len(values) >= 2:
                        row[targets[-1]] = float(np.std(values))
                    if source == "center_velocity" and math.isfinite(row["center_velocity"]):
                        row["geometry_change_proxy"] = abs(row["center_velocity"] - float(np.median(values)))
            # 三个target独立于全部feature；Q负例保持NaN。
            roi = data["ref"][v]["rois"].get(str(int(frames[i])))
            t = data["ts"][v][int(frames[i])]["ts5"]
            q = xywh_iou([t["x"], t["y"], t["w"], t["h"]], roi) if roi is not None else np.nan
            row.update(Y=int(roi is not None), Q=q, Z=q if roi is not None else 0.)
        # atomic partition保留完整multi-membership signature及chunk边界，不选parent。
        signatures = [tuple(j for j, a in enumerate(parentsets + rawsets + list(chunksets.values())) if i in a) for i in range(n)]
        atomic = []
        start = 0
        for i in range(1, n + 1):
            if i == n or comp[i] != comp[i - 1] or signatures[i] != signatures[i - 1]:
                atomic.append(np.arange(start, i));start = i
        grid = []
        for key in sorted({(int(comp[i]), int(math.floor(times[i] / GRID_SEC))) for i in range(n)}):
            grid.append(np.array([i for i in range(n) if (int(comp[i]), int(math.floor(times[i] / GRID_SEC))) == key]))
        blocks[v] = {"parents": [np.array(sorted(a)) for a in parentsets], "atomic": atomic, "grid": grid}
        records.extend(rows)
    return pd.DataFrame(records), memberships, blocks


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    values = x[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    r = np.empty(len(x))
    r[order] = np.repeat((starts + ends + 1) / 2., ends - starts)
    return r


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    n1, n0 = int(labels.sum()), int((labels == 0).sum())
    if not n1 or not n0 or len(np.unique(scores)) < 2:
        return np.nan
    return float((ranks(scores)[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels):
    if not np.sum(labels) or len(np.unique(scores)) < 2:
        return np.nan
    order = np.argsort(-np.asarray(scores), kind="stable")
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)]
    tp = np.cumsum(y)[ends - 1]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / ends) / y.sum())


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    a, b = ranks(x), ranks(y)
    a -= a.mean();b -= b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


class ClusterBootstrap:
    def __init__(self, videos, source_by_video):
        self.videos = list(videos)
        self.sources = sorted(set(source_by_video[v] for v in videos))
        source_ix = {s: i for i, s in enumerate(self.sources)}
        rng = np.random.default_rng(SEED)
        n, s = len(videos), len(self.sources)
        self.Wv = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS).astype(float)
        Ws = rng.multinomial(s, np.full(s, 1 / s), size=BOOTSTRAPS).astype(float)
        self.source_indices = np.array([source_ix[source_by_video[v]] for v in videos])
        self.Ws = Ws[:, self.source_indices]

    @staticmethod
    def ci(samples):
        samples = np.asarray(samples)
        samples = samples[np.isfinite(samples)]
        return tuple(np.quantile(samples, [.025, .975])) if len(samples) else (np.nan, np.nan)

    def mean_ci(self, values, source=False):
        values = np.asarray(values, dtype=float)
        good = np.isfinite(values)
        if not np.any(good):
            return (np.nan, np.nan)
        W = self.Ws if source else self.Wv
        denominator = W[:, good].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = W[:, good] @ values[good] / denominator
        return self.ci(samples)

    def ratio_ci(self, numerator, denominator, source=False):
        W = self.Ws if source else self.Wv
        num, den = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = (W @ num) / (W @ den)
        return self.ci(samples)


def pooled_auc_kernel(xs, ys):
    """A[i,j]是positive_i胜过negative_j的Mann–Whitney pair计数，ties计0.5。"""
    n = len(xs)
    A = np.zeros((n, n))
    positives = [x[y == 1] for x, y in zip(xs, ys)]
    negatives = [np.sort(x[y == 0]) for x, y in zip(xs, ys)]
    for i, p in enumerate(positives):
        for j, neg in enumerate(negatives):
            if len(p) and len(neg):
                left, right = np.searchsorted(neg, p, side="left"), np.searchsorted(neg, p, side="right")
                A[i, j] = np.sum((left + right) / 2.)
    return A, np.array([len(x) for x in positives]), np.array([len(x) for x in negatives])


def pooled_auc_ci(xs, ys, boot):
    if not sum(map(len, xs)):
        return (np.nan,) * 4
    combined = np.concatenate(xs)
    if len(np.unique(combined)) < 2:
        return (np.nan,) * 4
    A, n1, n0 = pooled_auc_kernel(xs, ys)
    cis = []
    for W in [boot.Wv, boot.Ws]:
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = np.einsum("bi,bi->b", W @ A, W) / ((W @ n1) * (W @ n0))
        cis.extend(boot.ci(samples))
    return tuple(cis)


def numeric_summary(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"mean": float(a.mean()) if len(a) else np.nan, "median": float(np.median(a)) if len(a) else np.nan,
            "iqr": float(np.quantile(a, .75) - np.quantile(a, .25)) if len(a) else np.nan, "n": len(a)}


def signal_audit(df, output, videos, source_by_video):
    per_video, per_source, discrim, spatial, contribution, quantiles, dictionary = [], [], [], [], [], [], []
    scopes = {"reference_valid109": [v for v in videos if df.loc[df.video_id == v, "role"].eq("reference_valid109").any()],
              "all166_weak_label_sensitivity": list(videos)}
    # 空输出视频仍进入boot/macro scope，auc不可计算，不能从实验身份丢掉。
    valid_list = (output / f"inputs/control_{RUN}/reference/reference_valid_109.txt").read_text().split()
    scopes["reference_valid109"] = sorted(valid_list)
    indexed = {v: df[df.video_id == v] for v in videos}
    boots = {k: ClusterBootstrap(vs, source_by_video) for k, vs in scopes.items()}
    for number, s in enumerate(SPECS, 1):
        feature, direction = s["feature"], s["direction"]
        values = df[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        constant_videos = sum(indexed[v][feature].nunique(dropna=True) <= 1 for v in videos if indexed[v][feature].notna().any())
        observed_videos = sum(indexed[v][feature].notna().any() for v in videos)
        sources = {"temporal": "temporal/candidate_cache/records/{video_id}.json + metadata_cache.json",
                   "localization": "localization/policy_v1/{video_id}.jsonl + composition/cmp1_records.json",
                   "composition": "composition/cmp1_records.json + stabilization/shards/{video_id}.json",
                   "consistency": "localization/policy_v1/{video_id}.jsonl + frozen stabilization/shards/{video_id}.json"}
        dictionary.append(s | {"source_file": str(output / f"inputs/control_{RUN}") + "/" + sources[s["family"]],
                                "granularity": "unique control source frame", "available_rate": float(finite.mean()),
                                "constant_rate": constant_videos / observed_videos if observed_videos else np.nan,
                                "constant_video_count": constant_videos, "observed_video_count": observed_videos,
                                "dominant_value_rate": float(df[feature].value_counts().iloc[0] / finite.sum()) if finite.sum() else np.nan,
                                "missing_state": "UNAVAILABLE" if not s["available"] else "STRUCTURAL_MISSING" if (~finite).any() else "OBSERVED",
                                "saved_or_derived": "NOT_SAVED_BY_PIPELINE" if feature == "exact_ts5_guard_correction" else "DERIVED_FROM_SAVED_TABLES"})
        video_metrics = {}
        for v in videos:
            g = indexed[v]
            observed = g[g[feature].notna()]
            x = observed[feature].to_numpy(dtype=float) * direction
            y = observed.Y.to_numpy(dtype=int)
            qmask = y == 1
            ym = auc(x, y) if direction else np.nan
            qp = spearman(x[qmask], observed.Q.to_numpy()[qmask]) if direction else np.nan
            zp = spearman(x, observed.Z.to_numpy()) if direction else np.nan
            vm = {"video_id": v, "source_id": source_by_video[v], "role": "reference_valid109" if v in valid_list else "empty_reference57",
                  "feature": feature, "direction": direction, "n_all": len(g), "n_observed": len(observed),
                  "missing_count": len(g) - len(observed), "unique_values": len(np.unique(x)), "constant": bool(len(x) and len(np.unique(x)) < 2),
                  "all_positive": bool(len(y) and y.sum() == len(y)), "all_negative": bool(len(y) and not y.sum()),
                  "prevalence": y.mean() if len(y) else np.nan, "auroc": ym, "ap": average_precision(x, y) if direction else np.nan,
                  "q_spearman": qp, "z_spearman": zp}
            per_video.append(vm);video_metrics[v] = vm
        for source in sorted(set(source_by_video.values())):
            g = df[(df.source_id == source) & (df.role == "reference_valid109") & df[feature].notna()]
            x, y = g[feature].to_numpy() * direction, g.Y.to_numpy()
            per_source.append({"source_id": source, "feature": feature, "n": len(g), "auroc": auc(x, y) if direction else np.nan,
                               "q_spearman": spearman(x[y == 1], g.Q.to_numpy()[y == 1]) if direction else np.nan,
                               "z_spearman": spearman(x, g.Z.to_numpy()) if direction else np.nan})
        for scope, vs in scopes.items():
            g = df[df.video_id.isin(vs) & df[feature].notna()]
            x, y = g[feature].to_numpy(dtype=float) * direction, g.Y.to_numpy(dtype=int)
            xs = [indexed[v].loc[indexed[v][feature].notna(), feature].to_numpy() * direction for v in vs]
            ys = [indexed[v].loc[indexed[v][feature].notna(), "Y"].to_numpy() for v in vs]
            vm = [video_metrics[v] for v in vs]
            aucs = [r["auroc"] for r in vm]
            aps = [r["ap"] for r in vm]
            qrs, zrs = [r["q_spearman"] for r in vm], [r["z_spearman"] for r in vm]
            bp = boots[scope]
            av = np.array(aucs, dtype=float);av = av[np.isfinite(av)]
            ci = pooled_auc_ci(xs, ys, bp) if direction and s["available"] else (np.nan,) * 4
            ysumm = numeric_summary(aucs)
            discrim.append({"scope": scope, "feature": feature, "family": s["family"], "direction": direction,
                            "n_observed": len(g), "prevalence": y.mean() if len(y) else np.nan,
                            "pooled_auroc": auc(x, y) if direction else np.nan,
                            "pooled_auroc_raw": auc(g[feature].to_numpy(), y) if direction else np.nan,
                            "pooled_ap": average_precision(x, y) if direction else np.nan,
                            "pooled_auc_video_ci_low": ci[0], "pooled_auc_video_ci_high": ci[1],
                            "pooled_auc_source_ci_low": ci[2], "pooled_auc_source_ci_high": ci[3],
                            "per_video_auc_mean": ysumm["mean"], "per_video_auc_median": ysumm["median"], "per_video_auc_iqr": ysumm["iqr"],
                            "computable_videos": ysumm["n"], "count_gt_05": int((av > .5).sum()), "count_gt_055": int((av > .55).sum()),
                            "count_gt_060": int((av > .6).sum()), "sign_consistency": float((av > .5).mean()) if len(av) else np.nan,
                            "constant_videos": sum(r["constant"] for r in vm), "all_positive": sum(r["all_positive"] for r in vm),
                            "all_negative": sum(r["all_negative"] for r in vm), "missing_videos": sum(not r["n_observed"] for r in vm),
                            "video_auc_mean_ci_low": bp.mean_ci(aucs)[0], "video_auc_mean_ci_high": bp.mean_ci(aucs)[1],
                            "source_auc_mean_ci_low": bp.mean_ci(aucs, True)[0], "source_auc_mean_ci_high": bp.mean_ci(aucs, True)[1],
                            "video_ap_mean": numeric_summary(aps)["mean"], "video_ap_mean_ci_low": bp.mean_ci(aps)[0],
                            "video_ap_mean_ci_high": bp.mean_ci(aps)[1], "source_ap_mean_ci_low": bp.mean_ci(aps, True)[0], "source_ap_mean_ci_high": bp.mean_ci(aps, True)[1]})
            for target, target_values, correlations, destination in [("Q", g.Q.to_numpy(), qrs, spatial), ("Z", g.Z.to_numpy(), zrs, contribution)]:
                mask = y == 1 if target == "Q" else np.ones(len(y), dtype=bool)
                summ = numeric_summary(correlations)
                cr = np.array(correlations);cr = cr[np.isfinite(cr)]
                destination.append({"scope": scope, "feature": feature, "family": s["family"], "target": target, "n": int(mask.sum()),
                                    "pooled_spearman": spearman(x[mask], target_values[mask]) if direction else np.nan,
                                    "per_video_rho_mean": summ["mean"], "per_video_rho_median": summ["median"], "per_video_rho_iqr": summ["iqr"],
                                    "computable_videos": summ["n"], "sign_consistency": float((cr > 0).mean()) if len(cr) else np.nan,
                                    "video_mean_ci_low": bp.mean_ci(correlations)[0], "video_mean_ci_high": bp.mean_ci(correlations)[1],
                                    "source_mean_ci_low": bp.mean_ci(correlations, True)[0], "source_mean_ci_high": bp.mean_ci(correlations, True)[1]})
            if len(g) and direction:
                order = np.lexsort((g.tie_hash.to_numpy(dtype=np.uint64), x))
                for qi, ix in enumerate(np.array_split(order, 5), 1):
                    gg = g.iloc[ix]
                    qq = gg.loc[gg.Y == 1, "Q"]
                    quantiles.append({"scope": scope, "feature": feature, "rank_quantile": qi, "n": len(gg), "constant_feature": g[feature].nunique() < 2,
                                      "y_mean": gg.Y.mean(), "q_mean": qq.mean(), "q_median": qq.median(), "z_mean": gg.Z.mean(), "z_median": gg.Z.median(),
                                      "positive_count": int(gg.Y.sum())})
        if number % 5 == 0:
            log(f"信号统计 {number}/{len(SPECS)}: {feature}")
    for name, rows in [("feature_dictionary.csv", dictionary), ("feature_discrimination.csv", discrim),
                       ("per_video_signal_metrics.csv", per_video), ("per_source_signal_metrics.csv", per_source),
                       ("spatial_quality_signal.csv", spatial), ("contribution_signal.csv", contribution), ("signal_quantiles.csv", quantiles)]:
        write_table(output, name, rows)
    return boots


def reference_events(data, df):
    events, shortest = [], {}
    for v in sorted(data["ref"]):
        fs = sorted(int(x) for x in data["ref"][v]["rois"])
        groups = []
        if fs:
            starts = [0] + [i for i in range(1, len(fs)) if fs[i] != fs[i - 1] + 1]
            for a, b in zip(starts, starts[1:] + [len(fs)]):
                groups.append(fs[a:b])
        short_ids = set(np.argsort([len(x) for x in groups], kind="stable")[:max(1, math.ceil(len(groups) / 4))]) if groups else set()
        shortest[v] = set()
        for j, g in enumerate(groups):
            short = j in short_ids
            if short:
                shortest[v].update(g)
            events.append({"video_id": v, "event_id": j, "start_frame": g[0], "end_frame": g[-1], "reference_frames": len(g),
                           "duration_sec": len(g) / data["meta"][v]["fps"], "short_within_video_quartile": short,
                           "baseline_matched": int(df[(df.video_id == v) & df.frame.isin(g)].Y.sum())})
    return events, shortest


def exact_budget(n, fraction):
    return int(math.floor(n * fraction))


def contiguous_groups(indices, frames):
    a = np.array(sorted(indices), dtype=int)
    if not len(a):
        return []
    splits = np.flatnonzero(np.diff(frames[a]) != 1) + 1
    return list(np.split(a, splits))


def random_interval_mask(eligible, frames, k, rng):
    if not k:
        return np.array([], dtype=int), True
    groups = contiguous_groups(eligible, frames)
    options = [max(0, len(g) - k + 1) for g in groups]
    total = sum(options)
    if total:
        draw = int(rng.integers(total))
        for g, count in zip(groups, options):
            if draw < count:
                return g[draw:draw + k], True
            draw -= count
    sizes = np.array([len(g) for g in groups])
    quotas = np.floor(k * sizes / sizes.sum()).astype(int)
    remainder_order = np.lexsort((np.arange(len(groups)), -(k * sizes / sizes.sum() - quotas)))
    for j in remainder_order[:k - quotas.sum()]:
        quotas[j] += 1
    chosen = []
    for g, q in zip(groups, quotas):
        if q:
            a = int(rng.integers(len(g) - q + 1))
            chosen.extend(g[a:a + q])
    return np.array(sorted(chosen), dtype=int), False


def centered_region_piece(indices, frames, count):
    groups = contiguous_groups(indices, frames)
    chosen = []
    # 对重叠parent的novel部分按component长→短确定性分配；不读target。
    for g in sorted(groups, key=lambda a: (-len(a), int(frames[a[0]]))):
        q = min(count - len(chosen), len(g))
        if q:
            start = (len(g) - q) // 2
            chosen.extend(g[start:start + q])
        if len(chosen) == count:
            break
    return np.array(sorted(chosen), dtype=int)


def native_region_mask(eligible, frames, score, hashes, regions, k):
    eligible_set = set(map(int, eligible))
    units = []
    for r in regions:
        a = np.array(sorted(eligible_set.intersection(map(int, r))), dtype=int)
        if len(a):
            units.append((float(np.mean(score[a])), int(np.min(hashes[a])), a))
    units.sort(key=lambda x: (x[0], x[1]))
    chosen = set()
    for _, _, a in units:
        remaining = k - len(chosen)
        if remaining <= 0:
            break
        novel = set(map(int, a)) - chosen
        if len(novel) <= remaining:
            chosen.update(novel)
        else:
            chosen.update(map(int, centered_region_piece(novel, frames, remaining)))
    if len(chosen) != k:
        raise ValueError("region coverage不足，不能偷偷按frame填预算")
    return np.array(sorted(chosen), dtype=int)


def measure_mask(g, dropped, reference_count, short_frames, events):
    dropped = np.array(sorted(set(map(int, dropped))), dtype=int)
    n, tp, S = len(g), float(g.Y.sum()), float(g.Z.sum())
    k = len(dropped)
    lost_tp = float(g.Y.to_numpy()[dropped].sum()) if k else 0.
    U = float(g.Z.to_numpy()[dropped].sum()) if k else 0.
    old_den, new_den = n + reference_count, n - k + reference_count
    old_f = 2 * S / old_den if old_den else 1.
    new_f = 2 * (S - U) / new_den if new_den else 1.
    frames = g.frame.to_numpy()
    dframes = set(map(int, frames[dropped]))
    predset = set(map(int, frames))
    short_lost = sum(f in short_frames for f in dframes)
    erased = 0
    for e in events:
        baseline = {f for f in predset if e["start_frame"] <= f <= e["end_frame"] and f in short_frames}
        if e["short_within_video_quartile"] and baseline and baseline <= dframes:
            erased += 1
    return {"drop_count": k, "fp_removed": k - lost_tp, "tp_lost": lost_tp, "z_lost": U,
            "fp_total": n - tp, "tp_total": tp, "z_total": S,
            "fp_fraction": (k - lost_tp) / (n - tp) if n > tp else np.nan,
            "tp_fraction": lost_tp / tp if tp else np.nan,
            "z_fraction": U / S if S else np.nan, "delta_f": new_f - old_f,
            "delta_iou_recall": -U / reference_count if reference_count else np.nan,
            "delta_frame_recall": -lost_tp / reference_count if reference_count else np.nan,
            "short_ref_total": len(short_frames), "short_tp_lost": short_lost,
            "short_delta_recall": -short_lost / len(short_frames) if short_frames else np.nan,
            "short_events_erased": erased,
            "mask_run_count": len(contiguous_groups(dropped, frames))}


def aggregate_masks(rows, boot, feature, budget, method, scope):
    def a(key):
        return np.array([r[key] for r in rows], dtype=float)
    fp, tp, z = a("fp_removed"), a("tp_lost"), a("z_lost")
    fp0, tp0, z0 = a("fp_total"), a("tp_total"), a("z_total")
    ef = np.divide(z, a("eligible_z"), out=np.full(len(rows), np.nan), where=a("eligible_z") > 0)
    result = {"scope": scope, "feature": feature, "method": method, "budget": budget, "identity": "POSTHOC_DIAGNOSTIC_MASK",
              "videos": len(rows), "eligible_frames": a("eligible_count").sum(), "frames_removed": a("drop_count").sum(),
              "fraction_of_eligible": a("drop_count").sum() / a("eligible_count").sum() if a("eligible_count").sum() else np.nan,
              "fraction_of_all_control_frames": a("drop_count").sum() / a("n_all").sum() if a("n_all").sum() else np.nan,
              "FP_removed": fp.sum(), "TP_lost": tp.sum(), "FP_removed_fraction": fp.sum() / fp0.sum() if fp0.sum() else np.nan,
              "TP_lost_fraction": tp.sum() / tp0.sum() if tp0.sum() else np.nan,
              "fp_minus_tp": fp.sum() / fp0.sum() - tp.sum() / tp0.sum() if fp0.sum() and tp0.sum() else np.nan,
              "sum_IoU_lost": z.sum(), "z_mass_lost": z.sum(), "z_mass_lost_fraction": z.sum() / z0.sum() if z0.sum() else np.nan,
              "z_mass_retained_fraction": 1 - z.sum() / z0.sum() if z0.sum() else np.nan,
              "z_loss_fraction_of_eligible_mass": z.sum() / a("eligible_z").sum() if a("eligible_z").sum() else np.nan,
              "video_equal_z_loss_eligible": numeric_summary(ef)["mean"],
              "macro_delta_F": numeric_summary(a("delta_f"))["mean"],
              "IoU_weighted_Recall_delta": numeric_summary(a("delta_iou_recall"))["mean"],
              "frame_Recall_delta": numeric_summary(a("delta_frame_recall"))["mean"],
              "recall_tail_p10": float(np.nanquantile(a("delta_frame_recall"), .1)) if np.isfinite(a("delta_frame_recall")).any() else np.nan,
              "recall_tail_min": float(np.nanmin(a("delta_frame_recall"))) if np.isfinite(a("delta_frame_recall")).any() else np.nan,
              "videos_delta_F_positive": int((a("delta_f") > 1e-12).sum()), "videos_delta_F_negative": int((a("delta_f") < -1e-12).sum()),
              "short_frame_recall_delta": -a("short_tp_lost").sum() / a("short_ref_total").sum() if a("short_ref_total").sum() else np.nan,
              "short_events_erased": a("short_events_erased").sum(), "mean_mask_runs": a("mask_run_count").mean(),
              "single_interval_feasible_videos": sum(r.get("single_interval_feasible", False) for r in rows),
              "all_unique_budgets_matched": True, "random_replicates": RANDOM_REPLICATES if method.startswith("random") else 0}
    for label, source in [("video", False), ("source", True)]:
        fci = boot.mean_ci(a("delta_f"), source)
        zci = boot.ratio_ci(z, z0, source)
        reci = boot.mean_ci(a("delta_frame_recall"), source)
        efci = boot.mean_ci(ef, source)
        W = boot.Ws if source else boot.Wv
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = W @ fp / (W @ fp0) - W @ tp / (W @ tp0)
        diffci = boot.ci(samples)
        for metric, interval in [("delta_F", fci), ("z_loss", zci), ("fp_minus_tp", diffci), ("frame_recall", reci), ("eligible_z_loss_video_mean", efci)]:
            result[label + "_" + metric + "_ci_low"], result[label + "_" + metric + "_ci_high"] = interval
    return result


def masks_audit(df, data, blocks, output, boots, events, short):
    videos = sorted(data["pred"])
    groups = {v: df[df.video_id == v].reset_index(drop=True) for v in videos}
    eventgroups = {v: [e for e in events if e["video_id"] == v] for v in videos}
    scopes = {"reference_valid109": sorted(data["valid_ids"]), "all166_weak_label_sensitivity": videos}
    aggregate, detail, contrasts = [], [], []
    random_cache = {}
    descriptive_only = {"normalized_position_mean", "bbox_center_x", "bbox_center_y"}
    for ni, s in enumerate(SPECS, 1):
        if not s["available"] or not s["direction"]:
            continue
        feature = s["feature"]
        for budget in BUDGETS:
            outcomes = {m: {} for m in ["random_frames", "random_contiguous_interval", "native_frame_ranking", "native_atomic_block", "fixed_temporal_grid", "candidate_parent_region"]}
            for v in videos:
                g = groups[v]
                score = g[feature].to_numpy(dtype=float) * s["direction"]
                eligible = np.flatnonzero(np.isfinite(score))
                hashes = g.tie_hash.to_numpy(dtype=np.uint64)
                frames = g.frame.to_numpy(dtype=int)
                k = exact_budget(len(eligible), budget)
                common = {"video_id": v, "source_id": data["selected"][v]["source_ytid"], "feature": feature, "budget": budget,
                          "eligible_count": len(eligible), "eligible_z": float(g.Z.to_numpy()[eligible].sum()), "n_all": len(g),
                          "constant_eligible_feature": bool(len(eligible) and len(np.unique(score[eligible])) < 2),
                          "descriptive_direction_only": feature in descriptive_only}
                cache_key = (v, budget, tuple(map(int, eligible)))
                if cache_key not in random_cache:
                    random_results = {}
                    seed = tied_hash(v, int(budget * 1000))
                    rng = np.random.default_rng(seed)
                    for method in ["random_frames", "random_contiguous_interval"]:
                        replicates = []
                        feasible = True
                        for repeat in range(RANDOM_REPLICATES):
                            if method == "random_frames":
                                d = rng.choice(eligible, size=k, replace=False) if k else np.array([], dtype=int)
                            else:
                                d, feasible = random_interval_mask(eligible, frames, k, rng)
                            assert len(set(map(int, d))) == k and set(map(int, d)) <= set(map(int, eligible))
                            replicates.append(measure_mask(g, d, len(data["ref"][v]["rois"]), short[v], eventgroups[v]))
                        random_results[method] = {key: numeric_summary([r[key] for r in replicates])["mean"] for key in replicates[0]}
                        random_results[method]["single_interval_feasible"] = feasible if method == "random_contiguous_interval" else False
                    random_cache[cache_key] = random_results
                for method, r in random_cache[cache_key].items():
                    outcomes[method][v] = common | r
                ranked = eligible[np.lexsort((hashes[eligible], score[eligible]))][:k]
                deterministic = {"native_frame_ranking": ranked}
                for method, region_key in [("native_atomic_block", "atomic"), ("fixed_temporal_grid", "grid"), ("candidate_parent_region", "parents")]:
                    deterministic[method] = native_region_mask(eligible, frames, score, hashes, blocks[v][region_key], k)
                for method, d in deterministic.items():
                    assert len(set(map(int, d))) == k and set(map(int, d)) <= set(map(int, eligible))
                    r = common | measure_mask(g, d, len(data["ref"][v]["rois"]), short[v], eventgroups[v])
                    outcomes[method][v] = r
                    detail.append(r | {"method": method, "identity": "POSTHOC_DIAGNOSTIC_MASK"})
                # 随机结果是200次合法mask的均值，不写任意一次的新prediction。
                for method in ["random_frames", "random_contiguous_interval"]:
                    detail.append(outcomes[method][v] | {"method": method, "identity": "POSTHOC_DIAGNOSTIC_MASK_MC_MEAN"})
            for scope, vs in scopes.items():
                for method, mapping in outcomes.items():
                    aggregate.append(aggregate_masks([mapping[v] for v in vs], boots[scope], feature, budget, method, scope) |
                                     {"descriptive_direction_only": feature in descriptive_only})
                native = outcomes["native_frame_ranking"]
                for method in ["native_atomic_block", "fixed_temporal_grid", "candidate_parent_region"]:
                    delta = [outcomes[method][v]["delta_f"] - native[v]["delta_f"] for v in vs]
                    mass = [native[v]["z_lost"] - outcomes[method][v]["z_lost"] for v in vs]
                    for clustering in ["video", "source"]:
                        b = boots[scope]
                        ci = b.mean_ci(delta, clustering == "source")
                        contrasts.append({"scope": scope, "feature": feature, "budget": budget, "method": method,
                                          "contrast": "region_minus_frame", "cluster": clustering, "macro_delta_F_advantage": numeric_summary(delta)["mean"],
                                          "delta_F_ci_low": ci[0], "delta_F_ci_high": ci[1], "IoU_mass_saved": sum(mass),
                                          "videos_better": sum(x > 1e-12 for x in delta), "videos_worse": sum(x < -1e-12 for x in delta)})
        if ni % 5 == 0:
            log(f"预算/结构诊断 {ni}/{len(SPECS)}: {feature}")
    write_table(output, "drop_curve.csv", [r for r in aggregate if r["method"] == "native_frame_ranking"])
    write_table(output, "contribution_mass_curve.csv", [r for r in aggregate if r["method"] == "native_frame_ranking"])
    write_table(output, "segment_block_comparison.csv", aggregate)
    write_table(output, "per_video_drop_metrics.csv", detail)
    write_table(output, "block_frame_paired_contrasts.csv", contrasts)


def strata_audit(df, data, output, boots, events, short):
    strata, support_control, history, base = [], [], [], []
    for v in sorted(data["pred"]):
        g = df[df.video_id == v].reset_index(drop=True)
        S, M, G = float(g.Z.sum()), len(g), len(data["ref"][v]["rois"])
        base.append({"video_id": v, "source_id": data["selected"][v]["source_ytid"], "role": "reference_valid109" if v in data["valid_ids"] else "empty_reference57",
                     "n_pred": M, "n_ref": G, "tp": int(g.Y.sum()), "fp": M - int(g.Y.sum()), "sum_iou": S,
                     "weak_precision": S / M if M else 0., "weak_recall": S / G if G else 0., "weak_f": 2 * S / (M + G) if M + G else 1.})
        if len(g):
            edge_order = np.lexsort((g.tie_hash.to_numpy(dtype=np.uint64), g.interior_position.to_numpy()))
            labels = np.empty(len(g), dtype=int)
            for q, ix in enumerate(np.array_split(edge_order, 5), 1):
                labels[ix] = q
            for q in range(1, 6):
                gg = g[(labels == q) & (g.E_t >= 2)]
                lo, hi = gg[gg.support_ratio < 1], gg[gg.support_ratio == 1]
                support_control.append({"video_id": v, "source_id": data["selected"][v]["source_ytid"], "position_rank_quintile": q,
                                        "n_nonunanimous": len(lo), "n_unanimous": len(hi), "E_ge2_frames": len(gg),
                                        "delta_y_unanimous_minus_non": hi.Y.mean() - lo.Y.mean() if len(hi) and len(lo) else np.nan,
                                        "delta_q_unanimous_minus_non": hi.loc[hi.Y == 1, "Q"].mean() - lo.loc[lo.Y == 1, "Q"].mean(),
                                        "delta_z_unanimous_minus_non": hi.Z.mean() - lo.Z.mean() if len(hi) and len(lo) else np.nan})
        eventv = [e for e in events if e["video_id"] == v]
        for arm in ["historical_fs1_edge", "historical_fs2_edge"]:
            ix = np.flatnonzero(g[arm].to_numpy() == 1)
            history.append({"video_id": v, "source_id": data["selected"][v]["source_ytid"], "method": arm} |
                           measure_mask(g, ix, G, short[v], eventv))
    main = df[df.role == "reference_valid109"].copy()
    main["support_stratum"] = np.select([main.E_t < 2, main.support_ratio == 1], ["E_lt2", "E_ge2_unanimous"], default="E_ge2_nonunanimous")
    main["geometry_stratum"] = np.where(main.center_velocity.notna(), "geometry_pair_observed", "geometry_pair_missing")
    for column in ["fallback", "ambiguous", "support_stratum", "geometry_stratum", "primary_label", "historical_fs1_edge", "historical_fs2_edge"]:
        for value, g in main.groupby(column, dropna=False):
            q = g.loc[g.Y == 1, "Q"]
            strata.append({"stratifier": column, "stratum": str(value), "n": len(g), "Y_mean": g.Y.mean(), "Q_n": len(q),
                           "Q_mean": q.mean(), "Q_median": q.median(), "Z_mean": g.Z.mean(), "Z_sum": g.Z.sum(),
                           "videos": g.video_id.nunique(), "sources": g.source_id.nunique()})
    missing_rows = []
    for s in SPECS:
        for state, mask in [("OBSERVED", main[s["feature"]].notna()),
                            ("UNAVAILABLE" if not s["available"] else "STRUCTURAL_MISSING", main[s["feature"]].isna())]:
            g = main[mask]
            if len(g):
                missing_rows.append({"feature": s["feature"], "state": state, "n": len(g), "Y_mean": g.Y.mean(),
                                     "Q_n": int(g.Y.sum()), "Q_mean": g.loc[g.Y == 1, "Q"].mean(), "Z_mean": g.Z.mean()})
    write_table(output, "target_state_strata.csv", strata)
    write_table(output, "missing_target_strata.csv", missing_rows)
    write_table(output, "support_position_control.csv", support_control)
    write_table(output, "historical_support_edge_reaudit.csv", history)
    write_table(output, "reference_events.csv", events)
    write_table(output, "baseline_per_video.csv", base)
    return base


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    output = DEFAULT_OUTPUT
    started = time.time()
    closure, data = verify(output)
    if closure["decision"] != "PASS_INPUT_CLOSURE":
        log(closure["decision"])
        return 2
    for sub in ["audit", "tables", "figures", "report"]:
        (output / sub).mkdir(exist_ok=True)
    protocol = freeze_protocol(output, data)
    identity = {"schema_version": "aic.stage6_2b.audit-identity/v1", "identity": IDENTITY, "launch_mode": "LOCAL_OFFLINE_AUDIT",
                "run_id": RUN, "predictions_sha256": PRED_SHA, "reference_sha256": REF_SHA,
                "protocol_sha256": sha(output / "audit/analysis_protocol.json"), "closure_sha256": sha(output / "audit/input_closure_verification.json"),
                "git_start": closure["git"], "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
                "scope": "frozen recovered local Dev166 only", "seed": SEED, "network": False, "new_predictions": False, "training": False}
    dump_json(output / "audit/audit_identity.json", identity)
    log("PASS_INPUT_CLOSURE → protocol已冻结 → feature/Y/Q/Z派生")
    df, memberships, blocks = extract(data)
    assert len(df) == 52019 and df.loc[df.Y == 0, "Q"].isna().all() and df.loc[df.Y == 0, "Z"].eq(0).all()
    assert np.allclose(df.loc[df.Y == 1, "Q"], df.loc[df.Y == 1, "Z"])
    write_table(output, "frame_features_targets.csv", df.to_dict("records"))
    write_table(output, "candidate_multi_membership.csv", memberships)
    log(f"真实派生完成: {len(df)}帧，{len(SPECS)}项feature（含不可用项），Y正例={int(df.Y.sum())}")
    videos = sorted(data["pred"])
    source_by_video = {v: data["selected"][v]["source_ytid"] for v in videos}
    boots = signal_audit(df, output, videos, source_by_video)
    events, short = reference_events(data, df)
    base = strata_audit(df, data, output, boots, events, short)
    masks_audit(df, data, blocks, output, boots, events, short)
    # 计算结束再次验证输入字节，任何变化停止而不是给信号Gate。
    after, _ = verify(output)
    if after["decision"] != "PASS_INPUT_CLOSURE":
        log(after["decision"])
        return 2
    for scope, vs in [("reference_valid109", data["valid_ids"]), ("all166_weak_label_sensitivity", set(videos))]:
        bb = [b for b in base if b["video_id"] in vs]
        log(f"BASELINE {scope}: videos={len(bb)}, frames={sum(x['n_pred'] for x in bb)}, macroF={np.mean([x['weak_f'] for x in bb]):.9f}")
    summary = {"schema_version": "aic.stage6_2b.native-computation-summary/v1", "identity": IDENTITY,
               "input_closure": "PASS_INPUT_CLOSURE", "recovery": "RECOVERY_VERIFIED", "audit": "FULL_AUDIT_STATISTICS_COMPLETED",
               "final_gate": "PENDING_EVIDENCE_REVIEW", "features": len(SPECS), "videos": len(videos), "frames": len(df),
               "valid_videos": len(data["valid_ids"]), "valid_frames": int((df.role == "reference_valid109").sum()),
               "Y_positive": int(df.Y.sum()), "source_clusters_all": len(set(source_by_video.values())),
               "source_clusters_valid": len({source_by_video[v] for v in data["valid_ids"]}),
               "budgets": BUDGETS, "bootstrap_replicates": BOOTSTRAPS, "random_replicates": RANDOM_REPLICATES,
               "seconds": time.time() - started, "figures_status": "PENDING_RENDER", "new_predictions": False,
               "input_hashes_unchanged": closure["file_hashes"] == after["file_hashes"]}
    dump_json(output / "audit/summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
