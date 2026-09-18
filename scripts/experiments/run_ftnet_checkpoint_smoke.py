"""Verify FTNet save -> terminate -> reload -> resume with real subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.checkpoint import (  # noqa: E402
    TrainingIdentity,
    checkpoint_sha256,
    load_checkpoint,
    save_checkpoint,
)
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig  # noqa: E402
from aic_video_highlight.ftnet.trainer import (  # noqa: E402
    FTNetExample,
    build_adamw_optimizer,
    build_cosine_scheduler,
    collate_ftnet_examples,
    train_step,
)

IDENTITY = TrainingIdentity(
    git_head="smoke",
    split_manifest_sha256="0" * 64,
    feature_manifest_sha256="0" * 64,
    label_adapter="smoke_target",
    seed=20260917,
)


def _batch(device: str, generator: torch.Generator):
    examples = [
        FTNetExample(
            visual_features=torch.randn(6 + index, 256, generator=generator),
            targets=torch.randint(0, 2, (6 + index,), generator=generator).float(),
            adjacency_mask=torch.tensor([False] + [True] * (5 + index)),
        )
        for index in range(2)
    ]
    return collate_ftnet_examples(examples).to(device)


def _model_bundle(device: str):
    torch.manual_seed(20260917)
    model = FTNet(FTNetConfig(native_dim=0)).to(device)
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = build_cosine_scheduler(optimizer, max_epochs=40, eta_min=1e-6)
    return model, optimizer, scheduler


def _optimizer_state_restored(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and bool(value.abs().sum() > 0):
                return True
    return False


def run_phase_save(checkpoint: Path, device: str) -> dict:
    generator = torch.Generator().manual_seed(20260917)
    model, optimizer, scheduler = _model_bundle(device)
    batch = _batch(device, generator)
    result = train_step(model, batch, optimizer, max_grad_norm=1.0)
    scheduler.step()
    digest = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        identity=IDENTITY,
        config=model.config,
    )
    return {
        "phase": "save",
        "loss": result.loss,
        "gradient_norm": result.gradient_norm,
        "epoch": 1,
        "global_step": 1,
        "scheduler_last_epoch": scheduler.last_epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "checkpoint_sha256": digest,
        "rng_probe": torch.randn(3).tolist(),
    }


def run_phase_resume(checkpoint: Path, device: str) -> dict:
    model, optimizer, scheduler = _model_bundle(device)
    loaded = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=IDENTITY,
        expected_config=model.config,
    )
    probe = torch.randn(3).tolist()
    optimizer_state_restored = _optimizer_state_restored(optimizer)
    learning_rate_at_load = optimizer.param_groups[0]["lr"]
    generator = torch.Generator().manual_seed(99)
    batch = _batch(device, generator)
    result = train_step(model, batch, optimizer, max_grad_norm=1.0)
    scheduler.step()
    return {
        "phase": "resume",
        "resumed_epoch": loaded["epoch"],
        "resumed_global_step": loaded["global_step"],
        "checkpoint_sha256": loaded["sha256"],
        "optimizer_state_restored": optimizer_state_restored,
        "scheduler_last_epoch_after_resume": scheduler.last_epoch,
        "learning_rate_at_load": learning_rate_at_load,
        "learning_rate_after_resume": optimizer.param_groups[0]["lr"],
        "loss": result.loss,
        "gradients_finite": result.gradients_finite,
        "parameters_updated": result.parameters_updated,
        "rng_probe": probe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("save", "resume", "all"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or Path(tempfile.gettempdir()) / "ftnet_checkpoint_smoke.pt"

    if args.phase == "save":
        print(json.dumps(run_phase_save(checkpoint, args.device), sort_keys=True))
        return 0
    if args.phase == "resume":
        print(json.dumps(run_phase_resume(checkpoint, args.device), sort_keys=True))
        return 0

    save_output = subprocess.run(
        [sys.executable, __file__, "--phase", "save", "--checkpoint", str(checkpoint), "--device", args.device],
        capture_output=True, text=True, check=True,
    )
    save_report = json.loads(save_output.stdout)
    resume_output = subprocess.run(
        [sys.executable, __file__, "--phase", "resume", "--checkpoint", str(checkpoint), "--device", args.device],
        capture_output=True, text=True, check=True,
    )
    resume_report = json.loads(resume_output.stdout)

    checks = {
        "resumed_epoch": resume_report["resumed_epoch"] == 1,
        "resumed_global_step": resume_report["resumed_global_step"] == 1,
        "optimizer_state_restored": resume_report["optimizer_state_restored"] is True,
        "scheduler_restored": resume_report["scheduler_last_epoch_after_resume"] == 2,
        "learning_rate_restored": abs(
            resume_report["learning_rate_at_load"]
            - save_report["learning_rate"]
        ) < 1e-12,
        "rng_restored": resume_report["rng_probe"] == save_report["rng_probe"],
        "checkpoint_sha_matches": (
            resume_report["checkpoint_sha256"] == save_report["checkpoint_sha256"]
            == checkpoint_sha256(checkpoint)
        ),
        "gradients_finite": resume_report["gradients_finite"] is True,
        "parameters_updated": resume_report["parameters_updated"] is True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(
        json.dumps(
            {
                "status": status,
                "formal_training_started": False,
                "device": args.device,
                "checkpoint": str(checkpoint),
                "save": save_report,
                "resume": resume_report,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
