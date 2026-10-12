from pathlib import Path

import pytest

from aic_video_highlight.ftnet.path_resolver import (
    DatasetRootResolutionError,
    resolve_dataset_root,
)


def test_cli_data_root_has_highest_priority(tmp_path: Path) -> None:
    cli_root = tmp_path / "cli"
    env_root = tmp_path / "env"
    cli_root.mkdir()
    env_root.mkdir()

    resolved = resolve_dataset_root(
        cli_data_root=cli_root,
        environ={"VHICRAFT_DATA_ROOT": str(env_root)},
    )

    assert resolved == cli_root.resolve()


def test_environment_data_root_is_used_when_cli_is_absent(tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    env_root.mkdir()

    resolved = resolve_dataset_root(
        environ={"VHICRAFT_DATA_ROOT": str(env_root)},
    )

    assert resolved == env_root.resolve()


def test_config_data_root_is_resolved_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    data_root = config_dir / "portable-data"
    data_root.mkdir(parents=True)
    config_path = config_dir / "datasets.yaml"
    config_path.write_text("dataset_root: portable-data\n", encoding="utf-8")

    resolved = resolve_dataset_root(environ={}, config_path=config_path)

    assert resolved == data_root.resolve()


def test_nonexistent_selected_root_fails_without_silent_fallback(tmp_path: Path) -> None:
    fallback = tmp_path / "fixture"
    fallback.mkdir()

    with pytest.raises(DatasetRootResolutionError, match="does not exist"):
        resolve_dataset_root(
            cli_data_root=tmp_path / "missing",
            environ={},
            repo_root=tmp_path,
            test_fallback="fixture",
        )


def test_external_drive_move_only_changes_root_value(tmp_path: Path) -> None:
    simulated_d_drive = tmp_path / "D-drive" / "datasets"
    simulated_e_drive = tmp_path / "E-drive" / "datasets"
    simulated_d_drive.mkdir(parents=True)
    simulated_e_drive.mkdir(parents=True)

    before = resolve_dataset_root(cli_data_root=simulated_d_drive, environ={})
    after = resolve_dataset_root(cli_data_root=simulated_e_drive, environ={})

    assert before.name == after.name == "datasets"
    assert before != after


def test_repo_relative_fallback_is_explicit_and_test_only(tmp_path: Path) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "data"
    fixture.mkdir(parents=True)

    resolved = resolve_dataset_root(
        environ={},
        repo_root=tmp_path,
        test_fallback=Path("tests") / "fixtures" / "data",
    )

    assert resolved == fixture.resolve()
