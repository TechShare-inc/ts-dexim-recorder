"""Unit tests for dexim.recorder.cli.config_utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from dexim.recorder.cli.config_utils import (
    DEXIM_CONFIG_DIR_ENV,
    get_config_dir,
    get_recorder_config_dir,
    list_named_configs,
    resolve_config_path,
)

# ---------------------------------------------------------------------------
# get_config_dir
# ---------------------------------------------------------------------------


class TestGetConfigDir:
    def test_explicit_override_wins(self, tmp_path: Path) -> None:
        result = get_config_dir(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_env_var_fallback(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(DEXIM_CONFIG_DIR_ENV, str(tmp_path))
        result = get_config_dir()
        assert result == tmp_path.resolve()

    def test_explicit_overrides_env_var(self, tmp_path: Path, monkeypatch) -> None:
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv(DEXIM_CONFIG_DIR_ENV, str(tmp_path))
        result = get_config_dir(str(other))
        assert result == other.resolve()

    def test_default_is_config(self, monkeypatch) -> None:
        monkeypatch.delenv(DEXIM_CONFIG_DIR_ENV, raising=False)
        result = get_config_dir()
        assert result.name == "config"

    def test_tilde_expanded(self, monkeypatch) -> None:
        monkeypatch.delenv(DEXIM_CONFIG_DIR_ENV, raising=False)
        result = get_config_dir("~/some_config")
        assert "~" not in str(result)


# ---------------------------------------------------------------------------
# get_recorder_config_dir
# ---------------------------------------------------------------------------


class TestGetRecorderConfigDir:
    def test_path_is_under_dexim_recorder(self, tmp_path: Path) -> None:
        result = get_recorder_config_dir(str(tmp_path))
        assert result == tmp_path / "dexim" / "recorder"


# ---------------------------------------------------------------------------
# resolve_config_path
# ---------------------------------------------------------------------------


class TestResolveConfigPath:
    def test_explicit_yaml_path_resolved(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "my.yaml"
        cfg_file.touch()
        result = resolve_config_path(str(cfg_file))
        assert result == cfg_file.resolve()

    def test_explicit_yml_extension_resolved(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "my.yml"
        cfg_file.touch()
        result = resolve_config_path(str(cfg_file))
        assert result == cfg_file.resolve()

    def test_explicit_path_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_config_path(str(tmp_path / "missing.yaml"))

    def test_named_config_in_recorder_dir(self, tmp_path: Path) -> None:
        recorder_dir = tmp_path / "dexim" / "recorder"
        recorder_dir.mkdir(parents=True)
        cfg_file = recorder_dir / "my_config.yaml"
        cfg_file.touch()

        result = resolve_config_path("my_config", config_dir=str(tmp_path))
        assert result == cfg_file.resolve()

    def test_named_config_fallback_to_dexim_dir(self, tmp_path: Path) -> None:
        dexim_dir = tmp_path / "dexim"
        dexim_dir.mkdir(parents=True)
        cfg_file = dexim_dir / "fallback.yaml"
        cfg_file.touch()

        result = resolve_config_path("fallback", config_dir=str(tmp_path))
        assert result == cfg_file.resolve()

    def test_named_config_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_config_path("no_such_config", config_dir=str(tmp_path))

    def test_recorder_dir_takes_priority_over_dexim_dir(self, tmp_path: Path) -> None:
        # Both dirs have the name; recorder/ should win.
        recorder_dir = tmp_path / "dexim" / "recorder"
        recorder_dir.mkdir(parents=True)
        dexim_dir = tmp_path / "dexim"
        (recorder_dir / "shared.yaml").touch()
        (dexim_dir / "shared.yaml").touch()

        result = resolve_config_path("shared", config_dir=str(tmp_path))
        assert result == (recorder_dir / "shared.yaml").resolve()


# ---------------------------------------------------------------------------
# list_named_configs
# ---------------------------------------------------------------------------


class TestListNamedConfigs:
    def test_returns_sorted_names(self, tmp_path: Path) -> None:
        recorder_dir = tmp_path / "dexim" / "recorder"
        recorder_dir.mkdir(parents=True)
        for name in ["zebra", "alpha", "mango"]:
            (recorder_dir / f"{name}.yaml").touch()

        result = list_named_configs(config_dir=str(tmp_path))
        assert result == ["alpha", "mango", "zebra"]

    def test_empty_when_dir_absent(self, tmp_path: Path) -> None:
        result = list_named_configs(config_dir=str(tmp_path))
        assert result == []

    def test_excludes_non_yaml_files(self, tmp_path: Path) -> None:
        recorder_dir = tmp_path / "dexim" / "recorder"
        recorder_dir.mkdir(parents=True)
        (recorder_dir / "valid.yaml").touch()
        (recorder_dir / "ignored.txt").touch()
        (recorder_dir / "ignored.json").touch()

        result = list_named_configs(config_dir=str(tmp_path))
        assert result == ["valid"]
