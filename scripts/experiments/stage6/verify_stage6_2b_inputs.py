"""Stage6.2B 独立本地输入闭合；只读恢复输入，无网络或模型调用。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

RUN = "vhicraft_dev166_202609141233"
HEAD = "4725a278937e8042efe051a9ae4f6584bfa796da"
PRED_SHA = "3cf4566b9cda13ca6dd34f78b3a05558d4dfe3b6b57910499982e47d6f9a31c1"
REF_SHA = "702f0bf1c74c33b3240a1d24409640d7b4bab8d62a4b84a1857838c6dcb5d5e2"
DEFAULT_OUTPUT = Path(r"D:\CDUT\硕士\2026-09 AIC\实验记录\Stage6_对VHiCraft的优化\03_Stage6_2B_Task_Native_Signal_Audit")
FORMAL_REPO = Path(r"D:\CDUT\硕士\2026-09 AIC\VHiCraft")


def duplicate_safe(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"重复JSON key: {k}")
        result[k] = v
    return result


def strict_json(text):
    return json.loads(text, object_pairs_hook=duplicate_safe,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(x, *, newline=False, spaced=False):
    kwargs = {} if spaced else {"separators": (",", ":")}
    text = json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False, **kwargs)
    return hashlib.sha256((text + ("\n" if newline else "")).encode("utf-8")).hexdigest()


def project(start, end, meta):
    """与冻结frame_projection契约一致：[start,end)，Fraction(str(float))。"""
    if end <= start:
        return []
    if meta["timestamp_mode"] == "CFR_FPS":
        rate = Fraction(str(float(meta["fps"])))
        first = math.ceil(Fraction(str(float(start))) * rate)
        stop = math.ceil(Fraction(str(float(end))) * rate)
    else:
        from bisect import bisect_left
        pts = [Fraction(str(float(t))) for t in meta["pts_timestamps"]]
        first, stop = bisect_left(pts, Fraction(str(float(start)))), bisect_left(pts, Fraction(str(float(end))))
    return list(range(max(0, first), min(int(meta["frame_count"]), stop)))


def index_unique(rows, key):
    out = {}
    for row in rows:
        k = key(row)
        if k in out:
            raise ValueError(f"重复连接键: {k}")
        out[k] = row
    return out


def git_snapshot(repo):
    return {name: subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()
            for name, args in [("status", ["status", "--short", "--branch"]),
                               ("branch", ["branch", "--show-current"]), ("head", ["rev-parse", "HEAD"]),
                               ("log", ["log", "--oneline", "--decorate", "-10"]),
                               ("diff_stat", ["diff", "--stat"]), ("diff_check", ["diff", "--check"])]}


def verify(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError("输出限定03实验记录目录")
    root = output / "inputs" / f"control_{RUN}"
    checks, failures = {}, []

    def check(name, ok, detail=None, kind="IDENTITY_MISMATCH"):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append({"check": name, "kind": kind, "detail": detail})

    rm = read_json(output / "audit/recovery_manifest.json")
    check("recovery_run_and_root", rm["run_id"] == RUN and Path(rm["local_control_root"]).resolve() == root.resolve())
    hashes, hash_errors = {}, []
    for x in rm["files"]:
        p = root / x["relative_path"]
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
            hash_errors.append({"path": x["relative_path"], "error": "UNSAFE_PATH"})
        elif not p.is_file():
            hash_errors.append({"path": x["relative_path"], "error": "MISSING"})
        else:
            h = sha(p)
            hashes[x["relative_path"]] = h
            if h != x["sha256_source"] or h != x["sha256_destination"] or p.stat().st_size != x["size_bytes"]:
                hash_errors.append({"path": x["relative_path"], "error": "HASH_OR_SIZE_MISMATCH"})
            if x["source_environment"] == "AUTODL" and (x["source_run_id"] != RUN or
                    not x["source_location"].startswith(f"/root/autodl-tmp/outputs/vhicraft/{RUN}/")):
                hash_errors.append({"path": x["relative_path"], "error": "MIXED_RUN_PROVENANCE"})
    check("all_recovery_file_hashes", not hash_errors, {"checked": len(rm["files"]), "errors": hash_errors},
          "INPUTS_MISSING" if any(x["error"] == "MISSING" for x in hash_errors) else "IDENTITY_MISMATCH")
    check("predictions_sha256", hashes.get("predictions/predictions.jsonl") == PRED_SHA, hashes.get("predictions/predictions.jsonl"))
    check("weak_reference_sha256", hashes.get("reference/dev166_weak_spatial_reference.jsonl") == REF_SHA,
          hashes.get("reference/dev166_weak_spatial_reference.jsonl"))
    if failures:
        data = {}
    else:
        protocol = read_json(root / "reference/formal_protocol.json")["frozen_stage6_0_control"]
        # 原位协议与恢复便捷副本必须字节相同。
        original_protocol = output.parent / "01_Stage6_1_Frame_Selection/00_Preparation/manifests/formal_protocol.json"
        check("protocol_copy_and_identity", sha(original_protocol) == hashes["reference/formal_protocol.json"] and
              protocol["run_id"] == RUN and protocol["run_git_head"] == HEAD and
              protocol["predictions_sha256"] == PRED_SHA and protocol["weak_spatial_reference_sha256"] == REF_SHA)
        pred = index_unique(read_jsonl(root / "predictions/predictions.jsonl"), lambda x: x["video_id"])
        selected = index_unique(read_jsonl(root / "input/dev_selected.jsonl"), lambda x: x["video_id"])
        ref = index_unique(read_jsonl(root / "reference/dev166_weak_spatial_reference.jsonl"), lambda x: x["video_id"])
        source = index_unique(read_jsonl(root / "reference/dev.jsonl"), lambda x: x["video_id"])
        final = index_unique(read_jsonl(root / "temporal/final_segments.jsonl"), lambda x: x["video_id"])
        meta = read_json(root / "frame_projection/metadata_cache.json")["records"]
        fresh_index = index_unique(read_jsonl(root / "frame_projection/fresh_index.jsonl"), lambda x: x["video_id"])
        run = read_json(root / "run_identity.json")
        shared = read_json(root / "shared_upstream_identity.json")
        vids = set(pred)
        valid_ids = set((root / "reference/reference_valid_109.txt").read_text().split())
        empty_ids = set((root / "reference/empty_reference_57.txt").read_text().split())
        check("selected_video_set", len(vids) == 166 and vids == set(selected) == set(ref) == set(run["video_ids"]) == valid_ids | empty_ids,
              {"count": len(vids), "video_ids": sorted(vids), "valid_roles": len(valid_ids), "empty_roles": len(empty_ids)})
        check("selected_source_mapping", all(selected[v]["source_ytid"] == source[v]["source_ytid"] and
              selected[v]["source_group"] == source[v]["source_group"] and selected[v]["sha256"] == source[v]["sha256"] and
              Path(meta[v]["video_path"]).name == Path(selected[v]["relative_video_path"]).name and
              fresh_index[v]["source_group"] == selected[v]["source_group"] for v in vids),
              {"source_group_count": len({selected[v]["source_group"] for v in vids}),
               "source_ytid_count": len({selected[v]["source_ytid"] for v in vids})})
        check("run_identity_chain", run["git_head"] == HEAD and run["profile"] == "dev166" and run["scope"] == "development" and
              run["input_manifest_sha256"] == hashes["input/dev_selected.jsonl"] == protocol["input_selected_manifest_sha256"] and
              run["identity_sha256"] == canonical_sha({k: v for k, v in run.items() if k != "identity_sha256"}),
              {"head": run["git_head"], "run_id_field_saved": "run_id" in run,
               "run_binding": "protocol + exact predictions hash + selected hash + upstream chain; run_identity本身无run_id字段"})
        shared_paths = {"composition_records": "composition/cmp1_records.json", "frame_projection_predictions": "frame_projection/predictions.jsonl",
                        "fresh_candidate_cache_manifest": "temporal/candidate_cache/cache_manifest.json", "localization_summary": "localization/full_dev_summary.json",
                        "retrieval_predictions": "retrieval/predictions.jsonl", "temporal_final_segments": "temporal/final_segments.jsonl"}
        check("shared_upstream_hashes", all(shared["artifacts"][k] == hashes[p] for k, p in shared_paths.items()), shared["artifacts"])
        # 身份文件无run_id时，验证其已记录inputs/outputs与本run实际字节hash。
        bindings = {
            "frame_projection": ({"final_segments": "temporal/final_segments.jsonl", "fresh_index": "frame_projection/fresh_index.jsonl"},
                                 {"metadata_cache.json": "frame_projection/metadata_cache.json", "predictions.jsonl": "frame_projection/predictions.jsonl"}),
            "temporal": ({"retrieval_predictions": "retrieval/predictions.jsonl", "retrieval_run_config": "retrieval/run_config.json"},
                         {"cache_manifest.json": "temporal/candidate_cache/cache_manifest.json", "final_segments.jsonl": "temporal/final_segments.jsonl"}),
            "localization": ({"frame_projection_predictions": "frame_projection/predictions.jsonl", "fresh_index": "frame_projection/fresh_index.jsonl", "metadata": "frame_projection/metadata_cache.json"},
                             {"full_dev_summary.json": "localization/full_dev_summary.json"}),
            "composition": ({"frame_projection_predictions": "frame_projection/predictions.jsonl", "manifest": "composition/manifest.json"},
                            {"cmp1_records.json": "composition/cmp1_records.json"}),
        }
        for component, (ins, outs) in bindings.items():
            ident = read_json(root / component / "fresh_identity.json")
            check(component + "_hash_binding", all(ident["inputs"][k] == hashes[p] for k, p in ins.items()) and
                  all(ident["outputs"][k] == hashes[p] for k, p in outs.items()))
        cmp_rows = read_json(root / "composition/cmp1_records.json")["records"]
        cmp = index_unique(cmp_rows, lambda x: (x["video_id"], x["frame"]))
        loc, ts, candidates = {}, {}, {}
        cache_manifest = read_json(root / "temporal/candidate_cache/cache_manifest.json")
        check("cache_manifest_semantic_hash", canonical_sha({k: v for k, v in cache_manifest.items() if k != "global_semantic_sha256"}, newline=True) == cache_manifest["global_semantic_sha256"])
        check("cache_dev_only", {x["split"] for x in cache_manifest["records"]} == {"dev"} and {x["video_id"] for x in cache_manifest["records"]} == vids)
        line_errors, geometry_errors, time_errors, loc_cmp_errors = [], [], [], []
        for v in sorted(vids):
            m = meta[v]
            if not (m["video_id"] == v and m["width"] > 0 and m["height"] > 0 and m["frame_count"] > 0 and math.isfinite(m["fps"]) and m["fps"] > 0 and
                    m["timestamp_mode"] in {"CFR_FPS", "PTS_TABLE"}):
                time_errors.append(v)
            if m["timestamp_mode"] == "PTS_TABLE" and (m["pts_timestamps"] is None or len(m["pts_timestamps"]) != m["frame_count"]):
                time_errors.append(v)
            fs = [x["frame"] for x in pred[v]["predictions"]]
            if fs != sorted(set(fs)) or any(isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < m["frame_count"] for f in fs):
                time_errors.append(v)
            l = index_unique(read_jsonl(root / f"localization/policy_v1/{v}.jsonl"), lambda x: x["frame"])
            t = index_unique(read_json(root / f"stabilization/shards/{v}.json"), lambda x: x["frame"])
            loc[v], ts[v] = l, t
            if set(l) != set(fs) or set(t) != set(fs):
                line_errors.append({"video": v, "error": "LOC_TS_FRAME_SET"})
            side = read_json(root / f"stabilization/shards/{v}.identity.json")
            if side["inputs"]["composition"] != hashes["composition/cmp1_records.json"] or side["inputs"]["video_id"] != v or side["outputs"][v + ".json"] != hashes[f"stabilization/shards/{v}.json"]:
                line_errors.append({"video": v, "error": "TS_SIDECAR_IDENTITY"})
            c = read_json(root / f"temporal/candidate_cache/records/{v}.json")
            candidates[v] = c
            entry = next(x for x in cache_manifest["records"] if x["video_id"] == v)
            raw_ids = {x["raw_candidate_id"] for x in c["raw_candidates"]}
            chunk_ids = {x["chunk_index"] for x in c["source_chunks"]}
            if c["video_id"] != v or c["split"] != "dev" or c["semantic_sha256"] != entry["semantic_sha256"] or canonical_sha({k: val for k, val in c.items() if k != "semantic_sha256"}, newline=True) != c["semantic_sha256"] or c["source_provenance"]["git_commit_head"] != HEAD or any(not set(x["contributor_raw_candidate_ids"]) <= raw_ids for x in c["merged_candidates"]) or any(x["chunk_index"] not in chunk_ids for x in c["raw_candidates"]):
                line_errors.append({"video": v, "error": "CANDIDATE_LINEAGE_HASH"})
            union = set()
            for x in c["merged_candidates"]:
                union.update(project(x["start_sec"], x["end_sec"], m))
            final_union = set()
            for x in final[v]["merged_prediction_segments"]:
                final_union.update(project(x["start_sec"], x["end_sec"], m))
            if union != set(fs) or final_union != set(fs):
                line_errors.append({"video": v, "error": "CANDIDATE_PROJECTED_FRAME_SET"})
            for p in pred[v]["predictions"]:
                f = p["frame"]
                if f not in t or (v, f) not in cmp or f not in l:
                    continue
                g = t[f]["ts5"]
                if p["bboxes"] != [g["x"], g["y"], g["w"]] or not math.isclose(g["h"], g["w"] * pred[v]["targetRatioWH"][1] / pred[v]["targetRatioWH"][0], abs_tol=1e-9):
                    geometry_errors.append([v, f])
                a, b = l[f], cmp[v, f]
                primary = a.get("primary")
                if a["video_id"] != v or a["image_width"] != m["width"] or a["image_height"] != m["height"] or a["status"] != b["localization_status"] or a["ambiguous"] != b["ambiguous"] or (None if primary is None else primary["score"]) != b["primary_score"]:
                    loc_cmp_errors.append([v, f])
                expected_time = f / m["fps"] if m["timestamp_mode"] == "CFR_FPS" else m["pts_timestamps"][f]
                if not math.isclose(a["source_timestamp"], expected_time, abs_tol=1e-9):
                    time_errors.append([v, f])
        expected = {(v, p["frame"]) for v in vids for p in pred[v]["predictions"]}
        policy_rows = [row for v in sorted(loc) for _, row in sorted(loc[v].items())]
        check("policy_semantic_hash", canonical_sha(policy_rows, spaced=True) == read_json(root / "composition/fresh_identity.json")["inputs"]["localization_policy"])
        check("prediction_LOC_CMP_TS_join", len(expected) == 52019 and set(cmp) == expected and not line_errors and not loc_cmp_errors,
              {"frames": len(expected), "lineage_errors": line_errors, "loc_cmp_errors": loc_cmp_errors[:20]})
        check("frozen_TS5_bbox_exact", not geometry_errors, {"compared": len(expected), "mismatches": geometry_errors[:20]})
        check("timing_coverage_and_frame_mapping", not time_errors and vids <= set(meta),
              {"modes": dict(Counter(meta[v]["timestamp_mode"] for v in vids)), "errors": time_errors[:20]})
        ref_errors = []
        for v in vids:
            if ref[v]["video_width"] != meta[v]["width"] or ref[v]["video_height"] != meta[v]["height"]:
                ref_errors.append({"video": v, "error": "REFERENCE_DIMENSIONS"})
            for f, box in ref[v]["rois"].items():
                if not 0 <= int(f) < meta[v]["frame_count"] or len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= 0 or box[3] <= 0:
                    ref_errors.append({"video": v, "frame": f, "error": "REFERENCE_FRAME_OR_BOX"})
        check("reference_join_and_coordinate_contract", not ref_errors, {"errors": ref_errors[:20],
              "video_sha256_match_manifest_false": sum(not ref[v].get("video_sha256_match_manifest", False) for v in vids),
              "provenance_limit": "weak teacher/seed reference;身份hash闭合不证明teacher源视频一致或官方GT有效性"})
        data = {"root": root, "pred": pred, "selected": selected, "ref": ref, "meta": meta, "candidates": candidates,
                "loc": loc, "cmp": cmp, "ts": ts, "valid_ids": valid_ids, "empty_ids": empty_ids}
    result = {"schema_version": "aic.stage6_2b.codex-independent-input-closure/v1", "verified_by": "Codex",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": "POSTHOC_DIAGNOSTIC_ONLY_INPUT_CLOSURE",
              "run_id": RUN, "recovery_manifest_sha256": sha(output / "audit/recovery_manifest.json"),
              "file_hashes": hashes, "checks": checks, "failures": failures,
              "auxiliary_optional": {"exact_TS5_guard": "NOT_SAVED_BY_PIPELINE"},
              "decision": ("INPUTS_MISSING" if any(x["kind"] == "INPUTS_MISSING" for x in failures) else "IDENTITY_MISMATCH") if failures else "PASS_INPUT_CLOSURE",
              "network_used": False, "statistics_executed": False,
              "git": {"VHiCraft": git_snapshot(FORMAL_REPO), "AIC-VideoHighlight": git_snapshot(Path(__file__).resolve().parents[3])}}
    destination = output / "audit/input_closure_verification.json"
    if destination.exists():
        history = output / "audit/history"
        history.mkdir(exist_ok=True)
        previous = history / ("input_closure_verification_" + sha(destination)[:12] + ".json")
        if not previous.exists():
            shutil.copyfile(destination, previous)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, data


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result, _ = verify(args.output)
    print(json.dumps({"decision": result["decision"], "checks": result["checks"], "failures": result["failures"]}, ensure_ascii=False))
    raise SystemExit(0 if result["decision"] == "PASS_INPUT_CLOSURE" else 2)
