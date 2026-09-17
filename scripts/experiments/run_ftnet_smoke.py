"""Run one or two synthetic FTNet optimizer steps; never a formal training job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aic_video_highlight.ftnet.model import FTNet, FTNetConfig  # noqa: E402
from aic_video_highlight.ftnet.trainer import (  # noqa: E402
    FTNetExample,
    collate_ftnet_examples,
    train_step,
)


def _example(length: int, native_dim: int, generator: torch.Generator) -> FTNetExample:
    native = None
    if native_dim > 0:
        native = torch.randn(length, native_dim, generator=generator)
    return FTNetExample(
        visual_features=torch.randn(length, 256, generator=generator),
        native_features=native,
        targets=torch.randint(0, 2, (length,), generator=generator).float(),
        adjacency_mask=torch.tensor([False] + [True] * (length - 1)),
    )


def run_smoke(*, device: str, steps: int, native_dim: int) -> dict[str, object]:
    if steps not in {1, 2}:
        raise ValueError("smoke steps must be 1 or 2")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")

    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(17)
    model = FTNet(FTNetConfig(native_dim=native_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = collate_ftnet_examples(
        [_example(5, native_dim, generator), _example(9, native_dim, generator)]
    ).to(device)
    results = [train_step(model, batch, optimizer) for _ in range(steps)]
    return {
        "status": "PASS",
        "formal_training_started": False,
        "device": device,
        "steps": steps,
        "losses": [result.loss for result in results],
        "gradients_finite": all(result.gradients_finite for result in results),
        "parameters_updated": all(result.parameters_updated for result in results),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_dim": native_dim,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--native-dim", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_smoke(
        device=args.device,
        steps=args.steps,
        native_dim=args.native_dim,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
