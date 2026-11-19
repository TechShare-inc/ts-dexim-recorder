"""Tests for the dexim-recorder CLI module."""

from __future__ import annotations

import click
from click.testing import CliRunner

from dexim.recorder.cli import recorder_group, standalone_app


class TestRecorderGroup:
    """Verify the recorder CLI group structure."""

    def test_group_is_click_group(self) -> None:
        assert isinstance(recorder_group, click.Group)

    def test_group_has_run_command(self) -> None:
        assert "run" in recorder_group.commands

    def test_group_has_status_command(self) -> None:
        assert "status" in recorder_group.commands

    def test_group_has_config_subgroup(self) -> None:
        assert "config" in recorder_group.commands
        config_cmd = recorder_group.commands["config"]
        assert isinstance(config_cmd, click.Group)

    def test_config_subgroup_has_show(self) -> None:
        config_cmd = recorder_group.commands["config"]
        assert isinstance(config_cmd, click.Group)
        assert "show" in config_cmd.commands

    def test_config_subgroup_has_validate(self) -> None:
        config_cmd = recorder_group.commands["config"]
        assert isinstance(config_cmd, click.Group)
        assert "validate" in config_cmd.commands

    def test_config_subgroup_has_list(self) -> None:
        config_cmd = recorder_group.commands["config"]
        assert isinstance(config_cmd, click.Group)
        assert "list" in config_cmd.commands


class TestHelpOutput:
    """Verify that --help exits cleanly for all commands."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_group_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "status" in result.output
        assert "config" in result.output

    def test_run_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--endpoint" in result.output
        assert "--format" in result.output
        assert "--lerobot-vcodec" in result.output
        run_command = recorder_group.commands["run"]
        parameter_names = {param.name for param in run_command.params}
        assert "lerobot_parallel_encoding" in parameter_names
        assert "lerobot_encoder_threads" in parameter_names

    def test_status_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["status", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--format" in result.output

    def test_config_show_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["config", "show", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_NAME" in result.output

    def test_config_validate_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["config", "validate", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_NAME" in result.output

    def test_config_list_help(self) -> None:
        result = self.runner.invoke(recorder_group, ["config", "list", "--help"])
        assert result.exit_code == 0
        assert "--config-dir" in result.output


class TestStandaloneApp:
    """Verify standalone_app entry point."""

    def test_standalone_app_is_callable(self) -> None:
        assert callable(standalone_app)
