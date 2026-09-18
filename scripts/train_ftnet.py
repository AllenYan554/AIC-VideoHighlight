#!/usr/bin/env python3
"""Formal FTNet training entrypoint (single implementation for local and AutoDL).

The frozen optimization protocol lives in the referenced config and is never
modified here.  Only device, data root, output root and worker count vary
between environments.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.experiment_runtime.progress import ProgressReporter  # noqa: E402
from aic_video_highlight.ftnet.checkpoint import (  # noqa: E402
    TrainingIdentity,
    load_checkpoint,
    save_checkpoint,
)
from aic_video_highlight.ftnet.dataset import (  # noqa: E402
    MaterializedFTNetDataset,
    load_normalization_stats,
)
from aic_video_highlight.ftnet.losses import masked_binary_cross_entropy  # noqa: E402
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig  # noqa: E402
from aic_video_highlight.ftnet.trainer import (  # noqa: E402
    FTNetExample,
    build_adamw_optimizer,
    build_cosine_scheduler,
    collate_ftnet_examples,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "models" / "ftnet_reference.yaml"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config_from_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _examples_from_video(video) -> FTNetExample:
    return FTNetExample(
        visual_features=video.visual,
        native_features=video.native,
        targets=video.target,
        adjacency_mask=video.adjacency_mask,
        loss_mask=video.loss_mask,
    )


def _loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    generator = torch.Generator().manual_seed(seed)

    class _Map(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, index: int) -> FTNetExample:
            return _examples_from_video(dataset.load(index))

    def _collate(examples):
        return collate_ftnet_examples(examples)

    return torch.utils.data.DataLoader(
        _Map(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=generator if shuffle else None,
    )


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


@torch.no_grad()
def evaluate(model, dataset, *, batch_size, num_workers, device) -> dict:
    model.eval()
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=num_workers)
    total_loss = 0.0
    total_frames = 0
    videos = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(
            batch.visual_features,
            native_features=batch.native_features,
            sequence_mask=batch.sequence_mask,
            adjacency_mask=batch.adjacency_mask,
        )
        loss = masked_binary_cross_entropy(output.logits, batch.targets, batch.loss_mask)
        frames = int(batch.loss_mask.sum().item())
        total_loss += float(loss.item()) * frames
        total_frames += frames
        videos += batch.visual_features.shape[0]
    return {
        "validation_masked_bce": total_loss / max(total_frames, 1),
        "videos": videos,
        "frames": total_frames,
    }


def run(args) -> int:
    cfg = _config_from_yaml(Path(args.config))
    training = cfg["training"]
    seed = int(training["seed"])
    _seed_everything(seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("PRECHECK FAIL: cuda requested but torch.cuda.is_available() is false", file=sys.stderr)
        return 2
    if device == "cuda":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id
    run_dir = output_root / run_id
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stats_path = data_root / "normalization" / "normalization_stats.json"
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATE_ONLY",
            "data_root": str(data_root),
            "stats_exists": stats_path.is_file(),
            "git_head": _git_head(),
        }, indent=2))
        return 0 if stats_path.is_file() else 3

    stats = load_normalization_stats(stats_path)
    train_dataset = MaterializedFTNetDataset(data_root, "TRAIN", stats)
    val_dataset = MaterializedFTNetDataset(data_root, "VALIDATION", stats)

    model = FTNet(FTNetConfig(visual_dim=cfg["visual_dim"], branch_dim=cfg["branch_dim"],
                              native_dim=cfg["native_dim"], native_branch_dim=cfg["native_branch_dim"],
                              temporal_channels=cfg["temporal_channels"],
                              temporal_kernel_size=cfg["temporal_kernel_size"],
                              temporal_dilations=tuple(cfg["temporal_dilations"]),
                              dropout=cfg["dropout"], use_delta=cfg["use_delta"])).to(device)
    opt_cfg = training["optimizer"]
    optimizer = build_adamw_optimizer(model, learning_rate=opt_cfg["learning_rate"],
                                      weight_decay=opt_cfg["weight_decay"],
                                      betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"])
    sched_cfg = training["scheduler"]
    scheduler = build_cosine_scheduler(optimizer, max_epochs=sched_cfg["t_max_epochs"],
                                       eta_min=sched_cfg["eta_min"])
    batch_size = int(training["batch_size_videos"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    clip = training["gradient_clipping"]["max_norm"]
    identity = TrainingIdentity(
        git_head=_git_head(),
        split_manifest_sha256="UNSET",
        feature_manifest_sha256="UNSET",
        label_adapter="soft_vote_target",
        seed=seed,
    )
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    start_epoch = 0
    if args.resume and last_path.is_file():
        loaded = load_checkpoint(last_path, model=model, optimizer=optimizer,
                                 scheduler=scheduler, expected_config=model.config)
        start_epoch = loaded["epoch"]
        print(f"RESUME from epoch {start_epoch}")

    print(f"FTNet Formal Training  Run: {run_id}")
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"TRAIN={len(train_dataset)} VALIDATION={len(val_dataset)} native_dim={cfg['native_dim']}")
    print(f"batch={batch_size} epochs={max_epochs} amp={training['precision']['amp']}")

    history = []
    best_metric = float("inf")
    best_epoch = -1
    total_steps = max_epochs * max(1, len(train_dataset) // batch_size)
    reporter = ProgressReporter(experiment_id="ftnet_training", total=total_steps, log_dir=log_dir)
    step = 0
    last_heartbeat = time.monotonic()
    for epoch in range(start_epoch, max_epochs):
        model.train()
        loader = _loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed + epoch,
                         num_workers=args.num_workers)
        running = 0.0
        running_frames = 0
        grad_norm_value = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.visual_features, native_features=batch.native_features,
                           sequence_mask=batch.sequence_mask, adjacency_mask=batch.adjacency_mask)
            loss = masked_binary_cross_entropy(output.logits, batch.targets, batch.loss_mask)
            loss.backward()
            grad_norm_value = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip, norm_type=2.0, error_if_nonfinite=True).detach().cpu())
            optimizer.step()
            frames = int(batch.loss_mask.sum().item())
            running += float(loss.item()) * frames
            running_frames += frames
            step += 1
            gpu_mem = (torch.cuda.memory_allocated() / (1024 ** 3)) if device == "cuda" else 0.0
            reporter.update(step, display={
                "epoch": f"{epoch + 1}/{max_epochs}",
                "loss": f"{running / max(running_frames, 1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "gnorm": f"{grad_norm_value:.3f}",
                "gpu_gb": f"{gpu_mem:.1f}",
            })
            now = time.monotonic()
            if now - last_heartbeat >= 45.0:
                print(f"\n[HEARTBEAT] epoch={epoch + 1} step={step} elapsed={now - reporter.started:.0f}s "
                      f"loss={running / max(running_frames, 1):.4f} gpu_gb={gpu_mem:.1f}")
                last_heartbeat = now
        scheduler.step()
        train_loss = running / max(running_frames, 1)
        val = evaluate(model, val_dataset, batch_size=batch_size, num_workers=args.num_workers, device=device)
        metric = val["validation_masked_bce"]
        improved = metric < best_metric
        if improved:
            best_metric, best_epoch = metric, epoch
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": metric,
                        "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm_value})
        save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch + 1, global_step=step, identity=identity, config=model.config)
        if improved:
            save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch + 1, global_step=step, identity=identity, config=model.config)
        print(f"\nEpoch {epoch + 1}/{max_epochs} train={train_loss:.4f} val={metric:.4f} "
              f"best_epoch={best_epoch + 1} lr={optimizer.param_groups[0]['lr']:.2e}")
        if args.max_steps and step >= args.max_steps:
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id, "git_head": _git_head(), "device": device,
        "best_epoch": best_epoch + 1, "best_val_masked_bce": best_metric,
        "epochs_run": len(history), "steps": step,
        "checkpoint_last": str(last_path), "checkpoint_best": str(best_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
