"""中文 CLI：样本准入、分层抽样、先冻结、人工模板、GT 合并与评价。

不下载素材，不调用模型；只评价外部生成且身份核验通过的八 seed 预测。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from aic_video_highlight import open_domain_highlights as H


def write_new(path: str | Path, value: dict) -> None:
    target = H.contained(path, H.DATA_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def flat_annotations(paths: list[str]) -> list[dict]:
    result = []
    for path in paths:
        doc = H.read_json(H.contained(path, H.DATA_ROOT / "annotations"))
        result.extend(doc if isinstance(doc, list) else [doc])
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "select", "preflight", "freeze", "template", "compile", "evaluate"))
    p.add_argument("--manifest", default=str(H.DATA_ROOT / "manifests" / "admitted_manifest.json"))
    p.add_argument("--exclusions", default=str(H.DATA_ROOT / "manifests" / "yt_exclusions.json"))
    p.add_argument("--protocol", required=True, help="新实验 config 中的 protocol_design.json")
    p.add_argument("--checkpoints", required=True, help="新实验 config 中的 checkpoint_manifest.json")
    p.add_argument("--freeze", default=str(H.DATA_ROOT / "manifests" / "evaluation_freeze.json"))
    p.add_argument("--output", help="输出必须在本轮 E 盘数据目录中；拒绝覆盖")
    p.add_argument("--rater-id", help="真实独立标注者匿名 ID")
    p.add_argument("--video-id", help="生成该已准入视频的标注模板")
    p.add_argument("--annotations", nargs="+", help="E 盘 annotations 下的真人标注 JSON")
    p.add_argument("--adjudications", help="E 盘 annotations 下的独立仲裁 JSON 列表")
    p.add_argument("--predictions", help="E 盘 predictions 下的八 seed 预测 JSON 列表")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = H.read_json(H.contained(args.manifest, H.DATA_ROOT / "manifests"))
        exclusions = H.read_json(H.contained(args.exclusions, H.DATA_ROOT / "manifests"))
        protocol, checkpoints = H.read_json(args.protocol), H.read_json(args.checkpoints)
        H.require(H.digest_json(checkpoints) == protocol["canonical_checkpoints_sha256"],
                  "checkpoint 清单与 Canonical 协议锚点不一致")
        if args.command == "select":
            H.require(bool(args.output), "必须指定新输出文件")
            selected = H.stratified_selection(manifest, exclusions)
            write_new(args.output, selected)
            result = {"selected_videos": len(selected["videos"])}
        else:
            audit = H.validate_manifest(manifest, exclusions)
            if args.command in ("validate", "preflight"):
                checked = H.verify_checkpoints(checkpoints) if args.command == "preflight" else []
                result = {**audit, "canonical_checkpoints_verified": len(checked),
                          "formal_model_evaluation": "NOT_RUN",
                          "blockers": ["缺少至少 50 条合法独立视频及真人高光标注"] if not manifest["videos"] else []}
            elif args.command == "freeze":
                result = H.make_freeze(protocol, manifest, exclusions, checkpoints)
                write_new(args.freeze, result)
            else:
                freeze = H.read_json(H.contained(args.freeze, H.DATA_ROOT / "manifests"))
                # This gate precedes reading annotations or predictions.
                H.verify_freeze(freeze, protocol, manifest, exclusions, checkpoints)
                H.require(bool(args.output), "必须指定新输出文件")
                if args.command == "template":
                    videos = {v["video_id"]: v for v in manifest["videos"]}
                    H.require(args.video_id in videos and bool(args.rater_id), "视频或标注者 ID 未指定")
                    result = H.annotation_template(videos[args.video_id], H.digest_json(protocol), args.rater_id)
                    H.contained(args.output, H.DATA_ROOT / "annotations")
                    write_new(args.output, result)
                else:
                    H.require(bool(args.annotations), "必须提供真人原始标注；不能只提交合成 GT")
                    annotations = flat_annotations(args.annotations)
                    adjudications = H.read_json(H.contained(args.adjudications, H.DATA_ROOT / "annotations")) if args.adjudications else []
                    gt = H.compile_human_gt(manifest, protocol, annotations, adjudications)
                    if args.command == "compile":
                        result = gt
                        write_new(args.output, result)
                    else:
                        H.require(gt["human_gt_ready"], "仍有分歧或不确定区间，正式评价阻断")
                        H.require(bool(args.predictions), "未提供冻结模型预测")
                        predicted = H.read_json(H.contained(args.predictions, H.DATA_ROOT / "predictions"))
                        # A shared once-only ledger prevents changing output names to repeat evaluation.
                        write_new(H.DATA_ROOT / "manifests" / "formal_evaluation_started.json",
                                  {"freeze_sha256": H.digest_json(freeze), "annotations_sha256": H.digest_json(annotations),
                                   "predictions_sha256": H.digest_json(predicted), "training": False})
                        result = H.evaluate_predictions(gt, predicted, manifest, checkpoints,
                                                        bootstrap_samples=protocol["bootstrap_samples"],
                                                        bootstrap_seed=protocol["bootstrap_seed"])
                        result.update({"freeze_sha256": H.digest_json(freeze), "gt_sha256": H.digest_json(gt),
                                       "predictions_sha256": H.digest_json(predicted), "formal_model_evaluation": "COMPLETED"})
                        write_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("status", "").startswith("BLOCKED") or result.get("human_gt_ready") is False and args.command == "compile" else 0
    except (H.EligibilityError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "formal_model_evaluation": "NOT_RUN"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
