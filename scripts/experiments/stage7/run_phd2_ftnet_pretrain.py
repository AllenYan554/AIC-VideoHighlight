#!/usr/bin/env python3
"""PHD2 5-block FTNet pretraining (PHD2_GENERIC_PRETRAIN_TARGET_V1).

Trains the frozen reference FTNet (REFERENCE_B5_FOR_THIS_PRETRAIN) with the
positive-unlabeled pairwise ranking objective on the materialized PHD2 dataset.

Frozen training configuration (no tuning in this round):
    AdamW lr=3e-4 wd=1e-2 betas=(0.9,0.999) eps=1e-8
    CosineAnnealingLR T_max=40 eta_min=1e-6
    batch=4 videos, epochs=40, grad clip=1.0, AMP off

Delta256 is computed inside the model from adjacent Visual256 features; it is
never stored.  The Q head is off.  No hard-negative BCE is used anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.model import FTNet, FTNetConfig  # noqa: E402
from aic_video_highlight.ftnet.native_schema import (  # noqa: E402
    compute_normalization_stats,
    standardize_native,
)
from aic_video_highlight.ftnet.phd2_targets import pairwise_ranking_loss  # noqa: E402

PROTOCOL_VERSION = "PHD2_GENERIC_PRETRAIN_TARGET_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_reference_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_config_from_yaml(payload: dict) -> FTNetConfig:
    return FTNetConfig(
        visual_dim=int(payload["visual_dim"]),
        branch_dim=int(payload["branch_dim"]),
        use_delta=bool(payload.get("use_delta", True)),
        native_dim=int(payload["native_dim"]),
        native_branch_dim=int(payload["native_branch_dim"]),
        temporal_channels=int(payload["temporal_channels"]),
        temporal_kernel_size=int(payload["temporal_kernel_size"]),
        temporal_dilations=tuple(int(v) for v in payload["temporal_dilations"]),
        dropout=float(payload["dropout"]),
    )


def git_identity(repo: Path) -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short"])}


class MaterializedDataset:
    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def load(self, index: int) -> dict:
        return load_file(str(self.files[index]))


def collate(samples: list[dict], stats):
    max_len = max(int(s["timestamps"].shape[0]) for s in samples)
    batch = len(samples)
    visual = torch.zeros((batch, max_len, 256), dtype=torch.float32)
    native = torch.zeros((batch, max_len, 16), dtype=torch.float32)
    sequence_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    adjacency = torch.zeros((batch, max_len), dtype=torch.bool)
    pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for i, sample in enumerate(samples):
        length = int(sample["timestamps"].shape[0])
        visual[i, :length] = sample["visual"].to(torch.float32)
        raw_native = sample["native"].to(torch.float32)
        missing = sample["native_missing"].to(torch.bool)
        native[i, :length] = standardize_native(raw_native, stats, missing)
        sequence_mask[i, :length] = True
        adjacency[i, :length] = sample["adjacency_mask"].to(torch.bool)
        pi = sample["pair_positive_index"].to(torch.int64)
        ui = sample["pair_unlabeled_index"].to(torch.int64)
        if pi.numel():
            pairs.append((i, pi, ui))
    return visual, native, sequence_mask, adjacency, pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="materialized root (contains materialized/)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None, help="debug only; formal run leaves unset")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    reference = load_reference_config(args.config)
    training = reference["training"]
    optimizer_cfg = training["optimizer"]
    scheduler_cfg = training["scheduler"]
    seed = int(training.get("seed", 20260917))
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    torch.manual_seed(seed)
    random.seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(training.get("deterministic", {}).get("use_deterministic_algorithms", False)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(training.get("deterministic", {}).get("cudnn_benchmark", False))
        torch.backends.cudnn.deterministic = bool(training.get("deterministic", {}).get("cudnn_deterministic", True))

    run_dir = args.output_root / args.run_id
    for name in ("checkpoints", "metrics", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "train.log"
    log_handle = log_path.open("a", encoding="utf-8", newline="\n")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    files = sorted((args.data_root / "materialized").glob("*.safetensors"))
    if not files:
        raise SystemExit("no materialized safetensors found")
    dataset = MaterializedDataset(files)
    log(f"=== PHD2 FTNet pretraining run_id={args.run_id} videos={len(files)} protocol={PROTOCOL_VERSION} ===")

    # TRAIN-only normalization over the materialized set (all PHD2 cohort videos are TRAIN).
    raw_parts = []
    frames_total = 0
    for path in files:
        sample = load_file(str(path))
        raw_parts.append(sample["native"].to(torch.float32))
        frames_total += int(sample["timestamps"].shape[0])
    stats = compute_normalization_stats(torch.cat(raw_parts, dim=0))
    atomic_json(run_dir / "metrics" / "normalization_stats.json", stats.as_dict())
    log(f"normalization stats computed over {frames_total} frames")

    model = FTNet(model_config_from_yaml(reference)).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(v) for v in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_cfg["t_max_epochs"]), eta_min=float(scheduler_cfg["eta_min"])
    )

    identity = {
        "schema": "aic.phd2.ftnet-pretrain-manifest/v1",
        "run_id": args.run_id,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": reference.get("architecture"),
        "tcn_reference": "REFERENCE_B5_FOR_THIS_PRETRAIN",
        "tcn_blocks": len(reference["temporal_dilations"]),
        "tcn_dilations": list(reference["temporal_dilations"]),
        "visual_dim": reference["visual_dim"],
        "native_dim": reference["native_dim"],
        "training": {k: training[k] for k in ("optimizer", "scheduler", "batch_size_videos", "max_epochs", "gradient_clipping", "precision", "seed")},
        "batch_size": batch_size,
        "epochs": epochs,
        "videos": len(files),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": args.device,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "materialized_root": str(args.data_root),
        "materialized_file_count": len(files),
        "training_config_sha256": sha256_file(args.config),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **git_identity(REPO_ROOT),
    }
    atomic_json(run_dir / "run_manifest.json", identity)
    log(f"model params={total_params} device={args.device} pytorch={torch.__version__} cuda={torch.version.cuda}")

    start_epoch = 0
    last_path = run_dir / "checkpoints" / "last.pt"
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        log(f"resumed from {last_path} at epoch {start_epoch}")

    metrics_path = run_dir / "metrics" / "history.jsonl"
    best_loss = math.inf
    global_step = 0
    rng = random.Random(seed)
    training_started = time.monotonic()

    for epoch in range(start_epoch, epochs):
        model.train()
        order = list(range(len(dataset)))
        rng.shuffle(order)
        epoch_losses: list[float] = []
        epoch_pos = epoch_neg = epoch_correct = 0
        epoch_started = time.monotonic()
        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start:batch_start + batch_size]
            samples = [dataset.load(i) for i in batch_indices]
            visual, native, sequence_mask, adjacency, pairs = collate(samples, stats)
            if not pairs:
                continue
            visual = visual.to(args.device)
            native = native.to(args.device)
            sequence_mask = sequence_mask.to(args.device)
            adjacency = adjacency.to(args.device)
            output = model(visual, sequence_mask=sequence_mask, adjacency_mask=adjacency, native_features=native)
            logits = output.logits
            z_pos_parts = []
            z_u_parts = []
            for b, pi, ui in pairs:
                z_pos_parts.append(logits[b, pi.to(args.device)])
                z_u_parts.append(logits[b, ui.to(args.device)])
            z_pos = torch.cat(z_pos_parts)
            z_u = torch.cat(z_u_parts)
            loss = pairwise_ranking_loss(z_pos, z_u)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} step {global_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(training["gradient_clipping"]["max_norm"]),
                norm_type=float(training["gradient_clipping"]["norm_type"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.item()))
            with torch.no_grad():
                epoch_pos += int(z_pos.numel())
                epoch_correct += int((z_pos > z_u).sum().item())
            if args.max_steps and global_step >= args.max_steps:
                break
        scheduler.step()
        epoch_seconds = time.monotonic() - epoch_started
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        elapsed = time.monotonic() - training_started
        remaining = (epochs - epoch - 1) * (elapsed / (epoch - start_epoch + 1))
        row = {
            "epoch": epoch + 1,
            "steps": len(epoch_losses),
            "loss": mean_loss,
            "ranking_accuracy": (epoch_correct / epoch_pos) if epoch_pos else None,
            "pairs": epoch_pos,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 3),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(remaining, 3),
        }
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
        gpu_util = None
        try:
            gpu_util = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        log(f"epoch={epoch+1}/{epochs} steps={row['steps']} loss={mean_loss} rank_acc={row['ranking_accuracy']} "
            f"lr={row['lr']:.3e} time={epoch_seconds:.1f}s elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m "
            f"vram={vram}MiB gpu={gpu_util}%")
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": model_config_from_yaml(reference).__dict__,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": args.run_id,
            "loss": mean_loss,
        }
        torch.save(checkpoint, last_path)
        if mean_loss is not None and mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(checkpoint, run_dir / "checkpoints" / "best.pt")
            atomic_json(run_dir / "checkpoints" / "best_identity.json",
                        {"epoch": epoch + 1, "loss": mean_loss, "monitor": "train_pairwise_loss",
                         "tie_break": "earliest_epoch"})
        if args.max_steps and global_step >= args.max_steps:
            log("max-steps reached; stopping (debug mode)")
            break

    total_elapsed = time.monotonic() - training_started
    last_epoch_done = epoch + 1 if "epoch" in locals() else 0
    summary = {
        "schema": "aic.phd2.ftnet-pretrain-summary/v1",
        "run_id": args.run_id,
        "status": "COMPLETE" if global_step > 0 else "FAILED",
        "protocol_version": PROTOCOL_VERSION,
        "epochs_completed": last_epoch_done,
        "epochs_target": epochs,
        "global_steps": global_step,
        "best_train_loss": None if best_loss is math.inf else best_loss,
        "wall_seconds": round(total_elapsed, 3),
        "wall_hours": round(total_elapsed / 3600.0, 4),
        "checkpoints": {"last": str(last_path), "best": str(run_dir / "checkpoints" / "best.pt")},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(run_dir / "summary" / "training_summary.json", summary)
    log(f"=== training end status={summary['status']} epochs={summary['epochs_completed']}/{epochs} "
        f"best_loss={summary['best_train_loss']} wall={summary['wall_hours']}h ===")
    log_handle.close()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
