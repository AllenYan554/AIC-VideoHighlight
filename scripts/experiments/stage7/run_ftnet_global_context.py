#!/usr/bin/env python3
"""FTNet Global-Context training entrypoint (LOCAL ONLY).

Canonical FTNet training loop (identical to the frozen visual-only baseline
contract: seeding, loader, AdamW, cosine, grad clip, min-VAL-BCE checkpoint
rule, per-epoch validation AP) with exactly one architectural change:

    G1 candidate: Local + Global(masked mean) + Relative(context) slots,
    fusion input stays 224, TCN / Y head unchanged, loss = masked BCE only.

Paired initialization: the candidate is built from the same RNG stream as the
baseline (shared modules bitwise identical for the same seed; Relative
Projection from an independent derived sub-seed, see global_context.py).

This script is research tooling, not production.  It never touches the
production VHiCraFTNet repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(r"D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight")
sys.path.insert(0, str(REPO / "src"))

from aic_video_highlight.ftnet.checkpoint import TrainingIdentity, save_checkpoint  # noqa: E402
from aic_video_highlight.ftnet.dataset import MaterializedFTNetDataset, load_normalization_stats  # noqa: E402
from aic_video_highlight.ftnet.evaluation import (  # noqa: E402
    average_precision,
    binary_reference,
    load_video_frames,
    universe_mask,
)
from aic_video_highlight.ftnet.global_context import GlobalContextFTNet, build_initialized_pair  # noqa: E402
from aic_video_highlight.ftnet.losses import masked_binary_cross_entropy  # noqa: E402
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig  # noqa: E402
from aic_video_highlight.ftnet.native_schema import standardize_native  # noqa: E402
from aic_video_highlight.ftnet.trainer import (  # noqa: E402
    FTNetExample,
    build_adamw_optimizer,
    build_cosine_scheduler,
    collate_ftnet_examples,
)

SPLIT_DIR_NAMES = {"TRAIN": "train", "VALIDATION": "validation", "CALIBRATION": "calibration"}
EXPECTED_CANDIDATE_PARAMS = 144641
EXPECTED_BASELINE_PARAMS = 166945


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    generator = torch.Generator().manual_seed(seed)

    class _Map(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, index: int) -> FTNetExample:
            video = dataset.load(index)
            return FTNetExample(
                visual_features=video.visual,
                native_features=None,
                targets=video.target,
                adjacency_mask=video.adjacency_mask,
                loss_mask=video.loss_mask,
            )

    return torch.utils.data.DataLoader(
        _Map(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_ftnet_examples, generator=generator if shuffle else None,
    )


@torch.no_grad()
def evaluate_bce(model, dataset, *, batch_size, device) -> float:
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=0)
    total_loss = 0.0
    total_frames = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(
            batch.visual_features,
            sequence_mask=batch.sequence_mask,
            adjacency_mask=batch.adjacency_mask,
        )
        loss = masked_binary_cross_entropy(output.logits, batch.targets, batch.loss_mask)
        frames = int(batch.loss_mask.sum().item())
        total_loss += float(loss.item()) * frames
        total_frames += frames
    return total_loss / max(total_frames, 1)


@torch.no_grad()
def validation_ap(model, data_root: Path, stats, device: str):
    """Per-video forward (frozen score_video semantics) -> pooled validation AP."""

    from safetensors.numpy import load_file

    model.eval()
    root = data_root / SPLIT_DIR_NAMES["VALIDATION"]
    paths = sorted(root.glob("*.safetensors"))
    ys, ps = [], []
    for path in paths:
        video_id = path.stem
        frames = load_video_frames(data_root, "VALIDATION", video_id, category="")
        tensors = load_file(str(path))
        visual = torch.from_numpy(np.asarray(tensors["visual"], dtype=np.float32)).unsqueeze(0)
        length = int(visual.shape[1])
        sequence_mask = torch.ones(1, length, dtype=torch.bool)
        adjacency = torch.from_numpy(np.asarray(tensors["adjacency_mask"], dtype=bool)).unsqueeze(0)
        output = model(
            visual.to(device),
            sequence_mask=sequence_mask.to(device),
            adjacency_mask=adjacency.to(device),
        )
        probabilities = output.probabilities[0].detach().cpu().numpy().astype(np.float64)
        u = universe_mask(frames.loss_mask)
        ys.append(binary_reference(frames.target, frames.loss_mask, 0.5)[u])
        ps.append(probabilities[u])
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=bool)
    p = np.concatenate(ps) if ps else np.zeros(0, dtype=float)
    return float(average_precision(y, p))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = cfg["training"]
    experiment = cfg.get("experiment_round23", {})
    context_mode = str(experiment.get("context_mode", "global_relative"))
    relative_dim = int(experiment.get("relative_dim", 32))

    seed = int(training["seed"])
    seed_everything(seed)
    device = args.device
    if device == "cuda":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    data_root = args.data_root.expanduser().resolve()
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "manifests" / "materialized_videos.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = load_normalization_stats(data_root / "normalization" / "normalization_stats.json")
    train_dataset = MaterializedFTNetDataset(data_root, "TRAIN", stats)
    val_dataset = MaterializedFTNetDataset(data_root, "VALIDATION", stats)

    model_config = FTNetConfig(
        visual_dim=cfg["visual_dim"], branch_dim=cfg["branch_dim"],
        native_dim=cfg["native_dim"], native_branch_dim=cfg["native_branch_dim"],
        temporal_channels=cfg["temporal_channels"],
        temporal_kernel_size=cfg["temporal_kernel_size"],
        temporal_dilations=tuple(cfg["temporal_dilations"]),
        dropout=cfg["dropout"], use_delta=cfg["use_delta"],
    )
    baseline, model, init_evidence = build_initialized_pair(
        model_config, seed=seed, context_mode=context_mode, relative_dim=relative_dim
    )
    del baseline
    model = model.to(device)
    rng_before_count = torch.get_rng_state()
    n_params_baseline = int(sum(p.numel() for p in FTNet(model_config).parameters()))
    torch.set_rng_state(rng_before_count)
    n_params = int(sum(p.numel() for p in model.parameters()))
    assert n_params == EXPECTED_CANDIDATE_PARAMS, f"candidate params changed: {n_params}"
    assert n_params_baseline == EXPECTED_BASELINE_PARAMS, n_params_baseline
    assert init_evidence["shared_all_zero_diff"], "paired initialization failed"

    init_payload = {
        "schema": "aic.stage7_3.global_context.initialization-equivalence/v1",
        "seed": seed,
        "context_mode": context_mode,
        "relative_dim": relative_dim,
        "shared_max_abs_diff": init_evidence["shared_max_abs_diff"],
        "shared_all_zero_diff": init_evidence["shared_all_zero_diff"],
        "relative_projection_seed": init_evidence["relative_projection_seed"],
        "rng_state_restored": init_evidence["rng_state_restored"],
        "baseline_params_total": n_params_baseline,
        "candidate_params_total": n_params,
    }
    (run_dir / "init_equivalence.json").write_text(json.dumps(init_payload, indent=2), encoding="utf-8")
    if args.evidence_dir is not None:
        evidence_dir = args.evidence_dir.expanduser().resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / f"initialization_equivalence_seed{seed}.json").write_text(
            json.dumps(init_payload, indent=2), encoding="utf-8"
        )

    opt_cfg = training["optimizer"]
    optimizer = build_adamw_optimizer(model, learning_rate=opt_cfg["learning_rate"],
                                      weight_decay=opt_cfg["weight_decay"],
                                      betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"])
    sched_cfg = training["scheduler"]
    scheduler = build_cosine_scheduler(optimizer, max_epochs=sched_cfg["t_max_epochs"], eta_min=sched_cfg["eta_min"])
    batch_size = int(training["batch_size_videos"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    clip = training["gradient_clipping"]["max_norm"]

    code_fingerprint = {
        "repo_model_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/model.py"),
        "repo_global_context_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/global_context.py"),
        "repo_losses_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/losses.py"),
        "repo_trainer_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/trainer.py"),
        "repo_dataset_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/dataset.py"),
        "repo_evaluation_py": sha256_file(REPO / "src/aic_video_highlight/ftnet/evaluation.py"),
        "runner_py": sha256_file(Path(__file__).resolve()),
        "run_config_sha256": sha256_file(args.config),
    }
    identity = TrainingIdentity(
        git_head="de7c78449cad282c6c264fb7536d9d7997620c18+dirty-stage7.3",
        split_manifest_sha256=sha256_file(manifest_path),
        feature_manifest_sha256=sha256_file(data_root / "normalization" / "normalization_stats.json"),
        label_adapter="soft_vote_target",
        seed=seed,
    )
    extra = {
        "experiment_id": "23_FTNet_v21_GlobalContext_202609282054",
        "run_id": args.run_id,
        "arm": "G1_GLOBAL_CONTEXT",
        "context_mode": context_mode,
        "relative_dim": relative_dim,
        "relative_projection_seed": init_evidence["relative_projection_seed"],
        "fusion_input_dim": cfg["branch_dim"] * 2 + relative_dim,
        "dataset_root": str(data_root),
        "train_videos": len(train_dataset),
        "validation_videos": len(val_dataset),
        "baseline_params_total": n_params_baseline,
        "candidate_params_total": n_params,
        "shared_init_max_abs_diff": init_evidence["shared_max_abs_diff"],
        "loss": "masked_bce_with_logits",
        "amp": False,
        "precision": "float32",
        "checkpoint_selection": "min_validation_masked_bce_tie_earliest",
        "code_fingerprint": code_fingerprint,
    }
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"

    history = []
    best_metric, best_epoch, best_ap, best_bce = float("inf"), -1, None, None
    step = 0
    total_steps = max_epochs * math.ceil(len(train_dataset) / batch_size)
    print(f"run={args.run_id} seed={seed} device={device} arm=G1_GLOBAL_CONTEXT "
          f"context={context_mode} rel_dim={relative_dim} fusion_in={extra['fusion_input_dim']} "
          f"TRAIN={len(train_dataset)} VALIDATION={len(val_dataset)} batch={batch_size} "
          f"epochs={max_epochs} steps={total_steps} init=PAIRED params={n_params} "
          f"(baseline {n_params_baseline}, delta {n_params - n_params_baseline:+d}) AMP=OFF "
          f"shared_max_abs_diff={init_evidence['shared_max_abs_diff']}", flush=True)
    t_start = time.perf_counter()
    throughput_window: list[float] = []

    for epoch in range(0, max_epochs):
        model.train()
        loader = make_loader(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed + epoch,
            num_workers=args.num_workers,
        )
        running_bce, running_frames = 0.0, 0
        grad_norm_value = 0.0
        epoch_t0 = time.perf_counter()
        steps_in_epoch = 0
        for batch in loader:
            timing = len(throughput_window) < 20
            if timing and device == "cuda":
                torch.cuda.synchronize()
            step_t0 = time.perf_counter()
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch.visual_features,
                sequence_mask=batch.sequence_mask,
                adjacency_mask=batch.adjacency_mask,
            )
            loss = masked_binary_cross_entropy(output.logits, batch.targets, batch.loss_mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} step {step + 1}")
            loss.backward()
            grad_norm_value = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip, norm_type=2.0, error_if_nonfinite=True).detach().cpu())
            optimizer.step()
            frames = int(batch.loss_mask.sum().item())
            bce_value = float(loss.detach().item())
            running_bce += bce_value * frames
            running_frames += frames
            step += 1
            steps_in_epoch += 1
            if timing:
                if device == "cuda":
                    torch.cuda.synchronize()
                throughput_window.append(time.perf_counter() - step_t0)
            if steps_in_epoch % args.progress_every == 0 or steps_in_epoch == len(loader) or step == 20:
                elapsed = time.perf_counter() - t_start
                per_step = elapsed / max(step, 1)
                eta = per_step * (total_steps - step)
                print(
                    f"[seed {seed}] [epoch {epoch + 1}/{max_epochs}] step {steps_in_epoch}/{len(loader)} "
                    f"({step}/{total_steps}) bce={bce_value:.4f} | elapsed {format_eta(elapsed)} "
                    f"ETA {format_eta(eta)} | VRAM {gpu_memory_mb():.0f}MB", flush=True)
            if step == 20 and throughput_window:
                mean_step = sum(throughput_window) / len(throughput_window)
                print(f"[seed {seed}] throughput: {1.0 / mean_step:.2f} steps/s (first 20) -> "
                      f"projected {format_eta(mean_step * total_steps)} + validation", flush=True)
        scheduler.step()
        train_bce = running_bce / max(running_frames, 1)
        val_bce = evaluate_bce(model, val_dataset, batch_size=batch_size, device=device)
        ap = validation_ap(model, data_root, stats, device)
        assert math.isfinite(train_bce) and math.isfinite(val_bce) and math.isfinite(ap), f"NaN/Inf at epoch {epoch+1}"
        model.train()
        improved = val_bce < best_metric
        if improved:
            best_metric, best_epoch = val_bce, epoch
            best_ap, best_bce = ap, val_bce
        epoch_seconds = round(time.perf_counter() - epoch_t0, 3)
        history.append({
            "epoch": epoch + 1, "train_bce": train_bce, "val_bce": val_bce, "val_ap": ap,
            "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm_value,
            "epoch_seconds": epoch_seconds, "gpu_memory_mb": gpu_memory_mb(),
        })
        save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch + 1, global_step=step, identity=identity, config=model.config, extra=extra)
        if improved:
            save_checkpoint(best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch + 1, global_step=step, identity=identity, config=model.config, extra=extra)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (epoch + 1) * (max_epochs - epoch - 1)
        print(f"[seed {seed}] epoch {epoch + 1}/{max_epochs} train_bce={train_bce:.4f} "
              f"val_bce={val_bce:.4f} val_ap={ap:.4f} best_epoch={best_epoch + 1} best_ap={best_ap:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} epoch_s={epoch_seconds:.1f} "
              f"elapsed={format_eta(elapsed)} ETA={format_eta(eta)}", flush=True)

    wall = time.perf_counter() - t_start
    summary = {
        "run_id": args.run_id, "seed": seed, "device": device, "arm": "G1_GLOBAL_CONTEXT",
        "context_mode": context_mode, "relative_dim": relative_dim,
        "init": "PAIRED_SHARED_INIT", "shared_init_max_abs_diff": init_evidence["shared_max_abs_diff"],
        "epochs_run": len(history), "steps": step,
        "best_epoch": best_epoch + 1, "best_val_masked_bce": best_bce, "ap_at_best": best_ap,
        "max_in_training_ap": max(h["val_ap"] for h in history),
        "final_epoch_ap": history[-1]["val_ap"] if history else None,
        "final_epoch_val_bce": history[-1]["val_bce"] if history else None,
        "train_wall_sec": round(wall, 3),
        "checkpoint_best": str(best_path),
        "checkpoint_best_sha256": sha256_file(best_path),
        "checkpoint_last": str(last_path),
        "checkpoint_last_sha256": sha256_file(last_path),
        "baseline_params_total": n_params_baseline,
        "candidate_params_total": n_params,
        "architecture": {"blocks": len(cfg["temporal_dilations"]), "dilations": cfg["temporal_dilations"],
                         "fusion_input_dim": extra["fusion_input_dim"], "relative_dim": relative_dim},
        "peak_vram_mb": round(gpu_memory_mb(), 1),
        "code_fingerprint": code_fingerprint,
        "nan_inf_detected": False,
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "seed", "best_epoch", "best_val_masked_bce",
                                              "ap_at_best", "max_in_training_ap", "train_wall_sec",
                                              "peak_vram_mb", "checkpoint_best_sha256")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
