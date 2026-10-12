"""Independent, fail-closed human highlight annotation and evaluation tools.

No model imports, network, video decoding, training, or automatic data discovery.
The caller supplies an explicitly admitted manifest and frozen protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "vhicraft.open-domain-highlight"
SEEDS = tuple(range(20260917, 20260925))
RAW_ROOT = Path(r"E:\ResearchData\datasets\open_domain_highlight_pilot_v1\raw")
DATA_ROOT = RAW_ROOT.parent
HEX = re.compile(r"^[0-9a-f]{64}$")


class EligibilityError(ValueError):
    """An admission, annotation, freeze, or evaluation gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EligibilityError(message)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def contained(path: str | Path, root: str | Path) -> Path:
    result = Path(path).resolve()
    require(result.is_relative_to(Path(root).resolve()), "路径超出获准目录")
    return result


def bins_for_duration(duration: float, width: float = 2.0) -> list[dict]:
    require(finite(duration) and duration > 0, "视频时长必须为有限正数")
    require(finite(width) and width > 0, "时间分箱宽度无效")
    return [{"index": i, "start_s": i * width, "end_s": min((i + 1) * width, duration)}
            for i in range(math.ceil(duration / width))]


def yt_exclusions(split: dict, candidates: list[dict], availability: list[dict]) -> dict:
    """Use identity metadata only; never read YT/AIC videos or GT values."""
    fields = ("canonical_video_id", "realized_video_id", "original_video_id",
              "replacement_video_id", "replacement_original_id", "replacement_id",
              "actual_id", "actual_source_id", "canonical_source_group")
    rows = split["entries"] + candidates + availability
    return {"schema": SCHEMA + ".exclusions/v1", "dataset": "youtube_highlights",
            "source_ids": sorted({r[k] for r in rows for k in fields
                                  if isinstance(r.get(k), str) and r[k]}),
            "source_sha256": sorted({r["source_sha256"] for r in split["entries"]
                                     if HEX.fullmatch(r.get("source_sha256", ""))}),
            "scope": "全部已登记 YT 候选、冻结 split 与替代来源；不包含 AIC"}


def validate_manifest(manifest: dict, exclusions: dict, *, raw_root: Path = RAW_ROOT,
                      verify_files: bool = True) -> dict:
    require(manifest.get("schema") == SCHEMA + ".manifest/v1", "manifest schema 不匹配")
    require(manifest.get("dataset") == "ODH_HUMAN_PILOT_V1", "数据方案不匹配")
    require(exclusions.get("schema") == SCHEMA + ".exclusions/v1", "来源排除清单缺失")
    rows = manifest.get("videos")
    require(isinstance(rows, list), "videos 必须为列表")
    require(manifest.get("exclusions_sha256") == digest_json(exclusions), "来源排除清单身份变化")
    ids, hashes, groups, aliases = set(), set(), set(), set()
    domains = Counter()
    held_ids, held_hashes = set(exclusions["source_ids"]), set(exclusions["source_sha256"])
    for row in rows:
        vid = row.get("video_id")
        require(isinstance(vid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", vid) is not None,
                "video_id 格式无效")
        require(vid not in ids, "重复 video_id")
        ids.add(vid)
        require(row.get("split") == "EXTERNAL_TEST_ONLY", "不得用于训练或调参")
        require(row.get("no_query") is True, "无 Query 条件未确认")
        domain = row.get("domain")
        require(domain in manifest.get("domain_targets", {}), "领域未在冻结分层计划中登记")
        domains[domain] += 1
        require(domains[domain] <= manifest["domain_targets"][domain], "超出预定领域配额")
        require(finite(row.get("duration_s")) and row["duration_s"] > 0, "视频时间轴未知")
        sha = row.get("source_sha256", "")
        require(isinstance(sha, str) and HEX.fullmatch(sha) is not None, "原视频 SHA256 缺失")
        require(sha not in hashes and sha not in held_hashes, "视频字节重复或来源泄漏")
        hashes.add(sha)
        source_ids = row.get("source_ids")
        require(isinstance(source_ids, list) and source_ids and
                all(isinstance(s, str) and s for s in source_ids), "来源 ID 缺失")
        require(len(source_ids) == len(set(source_ids)), "单视频存在重复来源 ID")
        require(not set(source_ids) & (aliases | held_ids), "来源 ID 重复或与 YT 重叠")
        aliases.update(source_ids)
        group = row.get("content_group")
        require(isinstance(group, str) and group and group not in groups, "内容家族重复或未知")
        groups.add(group)
        require(row.get("independence_review") == "HUMAN_CONFIRMED_NO_KNOWN_REUPLOAD_OR_EXCERPT",
                "缺少内容独立性人工复核")
        license_ = row.get("license", {})
        require(license_.get("status") == "VERIFIED" and
                license_.get("scope") == "RAW_VIDEO_RESEARCH_AND_HUMAN_ANNOTATION",
                "未证明原视频科研及人工标注使用权限")
        require(isinstance(license_.get("identifier"), str) and license_["identifier"].strip(),
                "许可证或授权标识缺失")
        for key in ("source_url",):
            require(isinstance(row.get(key), str) and row[key].startswith("https://"), "官方来源 URL 缺失")
        require(isinstance(license_.get("evidence_url"), str) and
                license_["evidence_url"].startswith("https://"), "许可来源 URL 缺失")
        require(HEX.fullmatch(license_.get("evidence_sha256", "")) is not None, "许可证据 SHA 缺失")
        video = contained(row.get("video_path", ""), raw_root)
        proof = contained(license_.get("evidence_path", ""), raw_root.parent / "provenance")
        if verify_files:
            require(video.is_file() and file_sha256(video) == sha, "原视频身份不匹配")
            require(proof.is_file() and file_sha256(proof) == license_["evidence_sha256"], "许可证据身份不匹配")
    return {"status": "PASS" if rows else "BLOCKED_EMPTY_POOL", "admitted_videos": len(rows),
            "domain_counts": dict(domains), "source_id_overlap": 0,
            "content_duplication_exhaustive": False,
            "human_gt_ready": False, "evaluation_ready": False}


def stratified_selection(manifest: dict, exclusions: dict, *, seed: int = 20261008,
                         raw_root: Path = RAW_ROOT) -> dict:
    """Select by predeclared source identity; no labels or model scores are inputs."""
    # A candidate pool can exceed quotas; validate each source independently first.
    expanded = dict(manifest)
    expanded["domain_targets"] = {d: len(manifest["videos"]) for d in manifest["domain_targets"]}
    validate_manifest(expanded, exclusions, raw_root=raw_root)
    grouped = defaultdict(list)
    for row in manifest["videos"]:
        grouped[row["domain"]].append(row)
    selected = []
    for domain, n in manifest["domain_targets"].items():
        ordered = sorted(grouped[domain], key=lambda r: hashlib.sha256(
            f"ODH_PILOT_V1|{seed}|{domain}|{r['source_sha256']}".encode()).hexdigest())
        selected.extend(ordered[:n])
    return {**manifest, "videos": selected, "selection_seed": seed,
            "selection_method": "按领域与原视频 SHA256 的预定哈希顺序；不读取 GT 或模型成绩"}


def verify_checkpoints(document: dict) -> list[dict]:
    rows = document.get("checkpoints", [])
    require(len(rows) == 8 and {r.get("seed") for r in rows} == set(SEEDS), "必须包含八个固定 seed")
    result = []
    for row in rows:
        require(HEX.fullmatch(row.get("sha256", "")) is not None, "checkpoint SHA 无效")
        path = Path(row["path"])
        require(path.is_file(), "冻结 checkpoint 缺失")
        actual = file_sha256(path)
        require(actual == row["sha256"], "冻结 checkpoint 身份不一致")
        result.append({"seed": row["seed"], "path": str(path), "sha256": actual,
                       "bytes": path.stat().st_size, "verified": True})
    return result


def make_freeze(protocol: dict, manifest: dict, exclusions: dict, checkpoints: dict,
                *, raw_root: Path = RAW_ROOT) -> dict:
    require(protocol.get("schema") == SCHEMA + ".protocol/v1", "协议 schema 不匹配")
    require(protocol.get("label_semantics") == "DIRECT_HUMAN_BINARY_HIGHLIGHT", "GT 语义不匹配")
    require(protocol.get("bin_seconds") == 2.0 and protocol.get("n_raters") == 3,
            "协议分箱或标注者数量变化")
    audit = validate_manifest(manifest, exclusions, raw_root=raw_root)
    require(audit["admitted_videos"] >= 50, "正式试点至少需要 50 个独立合法视频")
    require(protocol.get("no_query") is True and
            protocol.get("domain_targets") == manifest.get("domain_targets") and
            len(manifest["domain_targets"]) == 8 and
            set(manifest["domain_targets"].values()) == {8}, "样本分层与八领域冻结设计不一致")
    require(all(audit["domain_counts"].get(d, 0) >= 5 for d in manifest["domain_targets"]),
            "每个预定领域至少需要 5 个视频")
    require(protocol.get("canonical_checkpoints_sha256") == digest_json(checkpoints),
            "checkpoint 清单与 Canonical 协议锚点不一致")
    checked = verify_checkpoints(checkpoints)
    code_identities = {}
    for key, path_key, sha_key in (("model", "model_code", "model_code_sha256"),
                                  ("evaluation_code", "path", "sha256"),
                                  ("cli_code", "path", "sha256")):
        code = protocol.get(key, {})
        expected = code.get(sha_key, "")
        require(isinstance(expected, str) and HEX.fullmatch(expected) is not None,
                "冻结代码身份缺失：" + key)
        path = Path(code.get(path_key, ""))
        require(path.is_file() and file_sha256(path) == expected, "冻结代码身份变化：" + key)
        code_identities[key] = expected
    return {"schema": SCHEMA + ".freeze/v1", "protocol_sha256": digest_json(protocol),
            "manifest_sha256": digest_json(manifest), "exclusions_sha256": digest_json(exclusions),
            "checkpoints_sha256": digest_json(checkpoints), "checkpoints": checked,
            "code_sha256": code_identities,
            "labels_read": False, "status": "FROZEN_BEFORE_ANNOTATION"}


def verify_freeze(freeze: dict, protocol: dict, manifest: dict,
                  exclusions: dict, checkpoints: dict, *, raw_root: Path = RAW_ROOT) -> None:
    require(freeze.get("schema") == SCHEMA + ".freeze/v1" and
            freeze.get("status") == "FROZEN_BEFORE_ANNOTATION" and freeze.get("labels_read") is False,
            "未完成读取标签前的正式冻结")
    rebuilt = make_freeze(protocol, manifest, exclusions, checkpoints, raw_root=raw_root)
    for key in ("protocol_sha256", "manifest_sha256", "exclusions_sha256", "checkpoints_sha256", "code_sha256"):
        require(freeze.get(key) == rebuilt[key], "冻结身份变化：" + key)


def annotation_template(video: dict, protocol_sha: str, rater_id: str) -> dict:
    require(isinstance(rater_id, str) and rater_id.strip(), "标注者 ID 缺失")
    return {"schema": SCHEMA + ".annotation/v1", "video_id": video["video_id"],
            "source_sha256": video["source_sha256"], "protocol_sha256": protocol_sha,
            "rater_id": rater_id, "annotation_method": None, "tool_assistance": None,
            "watched_full_video": False,
            "bins": [{**b, "highlight": None, "uncertainty_reason": "尚未人工标注"}
                     for b in bins_for_duration(video["duration_s"])]}


def validate_annotation(annotation: dict, video: dict, protocol_sha: str) -> None:
    require(annotation.get("schema") == SCHEMA + ".annotation/v1", "标注 schema 不匹配")
    require(annotation.get("video_id") == video["video_id"] and
            annotation.get("source_sha256") == video["source_sha256"], "标注视频身份不一致")
    require(annotation.get("protocol_sha256") == protocol_sha, "标注协议身份不一致")
    require(annotation.get("annotation_method") == "HUMAN_MANUAL" and
            annotation.get("tool_assistance") == "NONE" and
            annotation.get("watched_full_video") is True, "未确认真人独立完整观看与标注")
    require(isinstance(annotation.get("rater_id"), str) and annotation["rater_id"].strip(), "标注者 ID 缺失")
    expected = bins_for_duration(video["duration_s"])
    actual = annotation.get("bins", [])
    require(len(actual) == len(expected), "必须标注完整视频 universe")
    for ref, row in zip(expected, actual):
        require(type(row.get("index")) is int and finite(row.get("start_s")) and
                finite(row.get("end_s")) and all(row.get(k) == ref[k] for k in ref),
                "标注区间未对齐固定时间轴")
        require(type(row.get("highlight")) is bool or row.get("highlight") is None,
                "只接受直接二元判断或不确定；不接受评分二值化")
        if row.get("highlight") is None:
            require(isinstance(row.get("uncertainty_reason"), str) and row["uncertainty_reason"].strip(),
                    "不确定判断须说明原因")


def compile_human_gt(manifest: dict, protocol: dict, annotations: list[dict],
                     adjudications: list[dict] | None = None) -> dict:
    """Preserve all raters; disagreements require a fourth blinded human judgment."""
    require(protocol.get("label_semantics") == "DIRECT_HUMAN_BINARY_HIGHLIGHT" and
            protocol.get("n_raters") == 3 and protocol.get("bin_seconds") == 2.0,
            "人工高光标注协议不匹配")
    sha = digest_json(protocol)
    videos = {v["video_id"]: v for v in manifest["videos"]}
    require(bool(videos), "没有合法样本，不能生成 GT")
    grouped = defaultdict(list)
    for annotation in annotations:
        vid = annotation.get("video_id")
        require(vid in videos, "标注包含未准入视频")
        validate_annotation(annotation, videos[vid], sha)
        grouped[vid].append(annotation)
    resolved = {}
    for row in adjudications or []:
        key = (row.get("video_id"), row.get("index"))
        require(key not in resolved and key[0] in videos and type(key[1]) is int, "重复或未知仲裁区间")
        require(row.get("annotation_method") == "HUMAN_MANUAL" and row.get("tool_assistance") == "NONE"
                and row.get("watched_full_video") is True, "仲裁须由真人完整观看")
        require(row.get("source_sha256") == videos[key[0]]["source_sha256"] and
                row.get("protocol_sha256") == sha, "仲裁身份不一致")
        require(type(row.get("highlight")) is bool and isinstance(row.get("reason"), str)
                and row["reason"].strip(), "仲裁必须给出二元判断及原因")
        resolved[key] = row
    output, uncertain, agreement_rows, used, quality_rows = [], [], [], set(), []
    for vid, video in videos.items():
        raters = grouped[vid]
        require(len(raters) == 3 and len({a["rater_id"] for a in raters}) == 3,
                "每条视频必须有三位不同标注者")
        labels = []
        complete_rows, uncertain_votes, needs_adjudication, n_adjudicated = [], 0, 0, 0
        for b in bins_for_duration(video["duration_s"]):
            votes = [a["bins"][b["index"]]["highlight"] for a in raters]
            uncertain_votes += votes.count(None)
            if all(type(v) is bool for v in votes):
                agreement_rows.append(votes)
                complete_rows.append(votes)
            key = (vid, b["index"])
            if all(type(v) is bool for v in votes) and len(set(votes)) == 1:
                require(key not in resolved, "一致区间不得用仲裁覆盖")
                label = votes[0]
            elif key in resolved:
                needs_adjudication += 1
                require(isinstance(resolved[key].get("adjudicator_id"), str) and
                        resolved[key]["adjudicator_id"].strip() and
                        resolved[key]["adjudicator_id"] not in {a["rater_id"] for a in raters},
                        "仲裁者必须独立于三位初始标注者")
                label = resolved[key]["highlight"]
                used.add(key)
                n_adjudicated += 1
            else:
                needs_adjudication += 1
                label = None
                uncertain.append({"video_id": vid, **b, "votes": votes})
            labels.append({**b, "highlight": label, "votes": votes})
        output.append({"video_id": vid, "source_sha256": video["source_sha256"],
                       "domain": video["domain"], "rater_ids": [a["rater_id"] for a in raters], "bins": labels})
        quality_rows.append({"video_id": vid, "n_bins": len(labels), "uncertain_votes": uncertain_votes,
                             "uncertain_vote_fraction": uncertain_votes / (3 * len(labels)),
                             "bins_with_uncertain_vote": sum(None in b["votes"] for b in labels),
                             "bins_requiring_adjudication": needs_adjudication,
                             "adjudicated_bins": n_adjudicated,
                             "unresolved_bins": needs_adjudication - n_adjudicated,
                             "agreement": annotation_agreement(complete_rows)})
    require(set(resolved) == used, "仲裁包含越界或未使用区间")
    total_bins = sum(r["n_bins"] for r in quality_rows)
    total_uncertain_votes = sum(r["uncertain_votes"] for r in quality_rows)
    return {"schema": SCHEMA + ".human-gt/v1", "label_semantics": "DIRECT_HUMAN_BINARY_HIGHLIGHT",
            "manifest_sha256": digest_json(manifest), "protocol_sha256": sha,
            "annotations_sha256": digest_json(annotations), "adjudications_sha256": digest_json(adjudications or []),
            "human_gt_ready": not uncertain, "unresolved_bins": uncertain,
            "annotation_quality": {"n_bins": total_bins, "uncertain_votes": total_uncertain_votes,
                                   "uncertain_vote_fraction": total_uncertain_votes / (3 * total_bins),
                                   "per_video": quality_rows},
            "agreement": annotation_agreement(agreement_rows), "videos": output}


def annotation_agreement(rows: list[list[bool]]) -> dict:
    if not rows:
        return {"complete_binary_bins": 0, "unanimous_fraction": None,
                "pairwise_agreement": None, "fleiss_kappa": None}
    require(all(len(r) == 3 and all(type(v) is bool for v in r) for r in rows), "一致性输入无效")
    positive = [sum(r) for r in rows]
    observed = sum((p * (p - 1) + (3 - p) * (2 - p)) / 6 for p in positive) / len(rows)
    prevalence = sum(positive) / (3 * len(rows))
    chance = prevalence ** 2 + (1 - prevalence) ** 2
    return {"complete_binary_bins": len(rows),
            "unanimous_fraction": sum(p in (0, 3) for p in positive) / len(rows),
            "pairwise_agreement": observed,
            "fleiss_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
            "scope": "完整三人二元区间的描述性统计；区间不是独立样本，不生成区间级 CI"}


def binary_average_precision(labels: list[bool], scores: list[float]) -> float | None:
    require(bool(labels) and len(labels) == len(scores), "AP 时间轴或长度不匹配")
    require(all(type(v) is bool for v in labels), "AP 仅接受直接人工二元标签")
    require(all(finite(s) for s in scores), "模型分数含非有限值")
    positives = sum(labels)
    if not positives:
        return None
    grouped = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        grouped[score][0] += int(label)
        grouped[score][1] += 1
    tp = n = 0
    ap = 0.0
    for score in sorted(grouped, reverse=True):
        pos, count = grouped[score]
        tp += pos
        n += count
        ap += pos / positives * tp / n
    return ap


def pool_to_bins(timestamps: list[float], scores: list[float], duration: float) -> list[float]:
    require(bool(timestamps) and len(timestamps) == len(scores), "预测时间轴缺失或长度不匹配")
    require(all(finite(t) and 0 <= t < duration for t in timestamps) and
            all(finite(s) for s in scores), "预测包含无效时间或分数")
    require(timestamps[0] <= .25 and timestamps[-1] >= duration - .75,
            "预测未覆盖完整视频首尾")
    require(all(0 < b - a <= .75 for a, b in zip(timestamps, timestamps[1:])),
            "预测有重复时间戳、乱序或采样缺口")
    grouped = [[] for _ in bins_for_duration(duration)]
    for t, score in zip(timestamps, scores):
        grouped[int(t // 2)].append(score)
    require(all(grouped), "存在未覆盖的 GT 时间区间")
    return [sum(g) / len(g) for g in grouped]


def evaluate_predictions(gt: dict, predictions: list[dict], manifest: dict,
                         checkpoints: dict, *, bootstrap_samples: int = 10000,
                         bootstrap_seed: int = 20261008) -> dict:
    require(gt.get("schema") == SCHEMA + ".human-gt/v1" and
            gt.get("label_semantics") == "DIRECT_HUMAN_BINARY_HIGHLIGHT" and
            gt.get("human_gt_ready") is True and not gt.get("unresolved_bins"), "人工 GT 尚未就绪")
    require(gt.get("manifest_sha256") == digest_json(manifest), "GT manifest 身份变化")
    videos = {v["video_id"]: v for v in manifest["videos"]}
    truths = {v["video_id"]: v for v in gt["videos"]}
    require(bool(videos) and len(truths) == len(gt["videos"]) and set(truths) == set(videos), "GT 视频集合不一致")
    ck = {r["seed"]: r["sha256"] for r in checkpoints["checkpoints"]}
    require(len(checkpoints["checkpoints"]) == 8 and set(ck) == set(SEEDS), "八 seed 身份缺失")
    require(type(bootstrap_samples) is int and bootstrap_samples >= 1, "bootstrap 次数无效")
    actual = {}
    for row in predictions:
        key = (row.get("video_id"), row.get("seed"))
        require(key not in actual and key[0] in videos and key[1] in ck, "预测重复或视频/seed 未登记")
        require(row.get("checkpoint_sha256") == ck[key[1]] and
                row.get("source_sha256") == videos[key[0]]["source_sha256"] and
                row.get("protocol_sha256") == gt.get("protocol_sha256"), "预测身份不匹配")
        actual[key] = row
    require(set(actual) == {(v, s) for v in videos for s in SEEDS}, "预测必须完整覆盖所有视频和八 seed")
    per_video, by_domain = [], defaultdict(list)
    for vid, video in videos.items():
        truth = truths[vid]
        require(truth.get("source_sha256") == video["source_sha256"] and
                truth.get("domain") == video["domain"], "GT 来源或领域不匹配")
        expected = bins_for_duration(video["duration_s"])
        require(len(truth["bins"]) == len(expected) and all(
            all(b.get(k) == e[k] for k in e) for b, e in zip(truth["bins"], expected)), "GT 时间轴不匹配")
        labels = [b["highlight"] for b in truth["bins"]]
        per_seed = {}
        for seed in SEEDS:
            row = actual[(vid, seed)]
            scores = pool_to_bins(row["timestamps"], row["scores"], video["duration_s"])
            per_seed[str(seed)] = binary_average_precision(labels, scores)
        values = [v for v in per_seed.values() if v is not None]
        mean = sum(values) / len(values) if values else None
        per_video.append({"video_id": vid, "domain": video["domain"], "n_bins": len(labels),
                          "positive_bins": sum(labels), "ap_by_seed": per_seed, "mean_ap": mean,
                          "na_reason": "NO_POSITIVE_GT" if mean is None else None})
        if mean is not None:
            by_domain[video["domain"]].append(mean)
    eligible = [r["mean_ap"] for r in per_video if r["mean_ap"] is not None]
    primary = sum(eligible) / len(eligible) if eligible else None
    interval = None
    if eligible:
        rng = random.Random(bootstrap_seed)
        samples = []
        for _ in range(bootstrap_samples):
            picked = [rng.choice(values) for values in by_domain.values() for _ in values]
            samples.append(sum(picked) / len(picked))
        samples.sort()
        def quantile(q):
            x = (len(samples) - 1) * q
            lo = int(x)
            return samples[lo] + (samples[min(lo + 1, len(samples) - 1)] - samples[lo]) * (x - lo)
        interval = [quantile(.025), quantile(.975)]
    return {"schema": SCHEMA + ".evaluation/v1", "label_semantics": gt["label_semantics"],
            "metric": "每视频八 seed AP 平均后，视频等权 macro AP；非 ensemble AP",
            "n_videos": len(videos), "n_ap_defined": len(eligible), "n_no_positive": len(videos) - len(eligible),
            "n_all_positive": sum(r["positive_bins"] == r["n_bins"] for r in per_video),
            "primary_macro_ap": primary, "ci95_stratified_video_bootstrap": interval,
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
            "per_video": per_video, "domain_means": {d: sum(v) / len(v) for d, v in by_domain.items()},
            "macro_ap_by_seed": {str(s): sum(r["ap_by_seed"][str(s)] for r in per_video
                                              if r["mean_ap"] is not None) / len(eligible)
                                 if eligible else None for s in SEEDS},
            "domain_summary": {d: {"n_videos": sum(r["domain"] == d for r in per_video),
                                   "n_ap_defined": len(by_domain[d]),
                                   "mean_ap": sum(by_domain[d]) / len(by_domain[d]) if by_domain[d] else None}
                               for d in manifest["domain_targets"]},
            "limitations": "CI 只覆盖抽样视频差异；不把八个模型或同视频区间当独立视频"}
