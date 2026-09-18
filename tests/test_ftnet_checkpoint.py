from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from aic_video_highlight.ftnet.checkpoint import (
    CheckpointError,
    TrainingIdentity,
    capture_rng_state,
    checkpoint_sha256,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig
from aic_video_highlight.ftnet.trainer import (
    FTNetExample,
    build_adamw_optimizer,
    build_cosine_scheduler,
    collate_ftnet_examples,
    train_step,
)


def _identity(seed: int = 20260917) -> TrainingIdentity:
    return TrainingIdentity(
        git_head="a" * 40,
        split_manifest_sha256="b" * 64,
        feature_manifest_sha256="c" * 64,
        label_adapter="soft_vote_target",
        seed=seed,
    )


def _build(batch_size: int = 2, length: int = 7):
    torch.manual_seed(20260917)
    model = FTNet(FTNetConfig(native_dim=0))
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = build_cosine_scheduler(optimizer, max_epochs=40, eta_min=1e-6)
    examples = [
        FTNetExample(
            visual_features=torch.randn(length + index, 256),
            targets=torch.randint(0, 2, (length + index,)).float(),
            adjacency_mask=torch.tensor([False] + [True] * (length + index - 1)),
        )
        for index in range(batch_size)
    ]
    batch = collate_ftnet_examples(examples)
    return model, optimizer, scheduler, batch


def _weights(model: FTNet) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_checkpoint_roundtrip_restores_model_optimizer_scheduler(tmp_path: Path) -> None:
    model, optimizer, scheduler, batch = _build()
    train_step(model, batch, optimizer, max_grad_norm=1.0)
    scheduler.step()
    before = _weights(model)

    path = tmp_path / "best.pt"
    digest = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=3,
        identity=_identity(),
        config=model.config,
    )
    assert digest == checkpoint_sha256(path)

    fresh, fresh_optimizer, fresh_scheduler, _ = _build()
    loaded = load_checkpoint(
        path,
        model=fresh,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        expected_identity=_identity(),
        expected_config=model.config,
    )

    assert loaded["epoch"] == 1
    assert loaded["global_step"] == 3
    assert loaded["identity"]["seed"] == 20260917
    for old, new in zip(before, _weights(fresh), strict=True):
        assert torch.equal(old, new)
    assert fresh_scheduler.last_epoch == scheduler.last_epoch


def test_checkpoint_resume_continues_deterministically(tmp_path: Path) -> None:
    reference_model, reference_optimizer, reference_scheduler, batch = _build()
    train_step(reference_model, batch, reference_optimizer, max_grad_norm=1.0)
    reference_scheduler.step()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=reference_model,
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        epoch=1,
        global_step=1,
        identity=_identity(),
        config=reference_model.config,
    )
    train_step(reference_model, batch, reference_optimizer, max_grad_norm=1.0)
    reference_scheduler.step()

    model, optimizer, scheduler, batch = _build()
    train_step(model, batch, optimizer, max_grad_norm=1.0)
    load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=_identity(),
        expected_config=model.config,
    )
    train_step(model, batch, optimizer, max_grad_norm=1.0)
    scheduler.step()

    for expected, actual in zip(
        _weights(reference_model), _weights(model), strict=True
    ):
        assert torch.equal(expected, actual)


def test_checkpoint_restores_rng_state(tmp_path: Path) -> None:
    torch.manual_seed(7)
    random.seed(7)
    model, optimizer, scheduler, _ = _build()
    path = tmp_path / "rng.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        identity=_identity(),
        config=model.config,
    )
    expected = torch.randn(4)

    torch.manual_seed(999)
    random.seed(999)
    load_checkpoint(path, model=model)
    assert torch.equal(torch.randn(4), expected)


def test_checkpoint_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _build()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        identity=_identity(),
        config=model.config,
    )
    other = TrainingIdentity(
        git_head="d" * 40,
        split_manifest_sha256="b" * 64,
        feature_manifest_sha256="c" * 64,
        label_adapter="soft_vote_target",
        seed=20260917,
    )
    with pytest.raises(CheckpointError, match="identity mismatch"):
        load_checkpoint(path, model=model, expected_identity=other)


def test_checkpoint_config_mismatch_fails_closed(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _build()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        identity=_identity(),
        config=model.config,
    )
    with pytest.raises(CheckpointError, match="config mismatch"):
        load_checkpoint(path, model=model, expected_config=FTNetConfig(native_dim=16))


def test_missing_or_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="does not exist"):
        load_checkpoint(tmp_path / "missing.pt")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not-a-torch-checkpoint")
    with pytest.raises(CheckpointError, match="unreadable"):
        load_checkpoint(corrupt)


def test_atomic_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _build()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        identity=_identity(),
        config=model.config,
    )
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_capture_and_restore_rng_state_is_reversible() -> None:
    torch.manual_seed(11)
    state = capture_rng_state()
    first = torch.randn(3)
    restore_rng_state(state)
    assert torch.equal(torch.randn(3), first)
