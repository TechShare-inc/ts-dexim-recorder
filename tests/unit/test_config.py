"""Unit tests for dexim.recorder.config."""

from __future__ import annotations

import pytest
import yaml

from dexim.recorder.config import RecorderNodeConfig, load_config

# ---------------------------------------------------------------------------
# RecorderNodeConfig validation
# ---------------------------------------------------------------------------


class TestRecorderNodeConfigValidation:
    def test_valid_config_constructs(self) -> None:
        cfg = RecorderNodeConfig(
            data_endpoints=["tcp://localhost:5600"],
            storage_format="hdf5",
            output_dir="out/episodes",
            fps=30,
        )
        assert cfg.node_id == "recorder"
        assert cfg.storage_format == "hdf5"

    def test_empty_endpoints_raises(self) -> None:
        with pytest.raises(ValueError, match="data_endpoints"):
            RecorderNodeConfig(data_endpoints=[])

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="storage_format"):
            RecorderNodeConfig(
                data_endpoints=["tcp://localhost:5600"],
                storage_format="parquet",
            )

    def test_zero_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            RecorderNodeConfig(
                data_endpoints=["tcp://localhost:5600"],
                fps=0,
            )

    def test_negative_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            RecorderNodeConfig(
                data_endpoints=["tcp://localhost:5600"],
                fps=-10,
            )

    def test_lerobot_format_valid(self) -> None:
        cfg = RecorderNodeConfig(
            data_endpoints=["tcp://localhost:5600"],
            storage_format="lerobot",
        )
        assert cfg.storage_format == "lerobot"

    def test_multiple_endpoints_accepted(self) -> None:
        cfg = RecorderNodeConfig(
            data_endpoints=[
                "tcp://localhost:5600",
                "tcp://localhost:5601",
                "tcp://localhost:5602",
            ],
        )
        assert len(cfg.data_endpoints) == 3

    def test_defaults(self) -> None:
        cfg = RecorderNodeConfig(data_endpoints=["tcp://localhost:5600"])
        assert cfg.node_id == "recorder"
        assert cfg.storage_format == "lerobot"
        assert cfg.fps == 30
        assert cfg.default_task == ""
        assert cfg.master_clock_topic is None
        assert cfg.continuous_topics == []
        assert cfg.topic_dtypes == {}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path) -> None:
        cfg_data = {
            "data_endpoints": ["tcp://localhost:5600"],
            "storage_format": "hdf5",
            "output_dir": "out/episodes",
            "fps": 15,
        }
        cfg_file = tmp_path / "recorder.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))

        assert cfg.storage_format == "hdf5"
        assert cfg.fps == 15
        assert cfg.data_endpoints == ["tcp://localhost:5600"]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_format_in_yaml_raises(self, tmp_path) -> None:
        cfg_data = {
            "data_endpoints": ["tcp://localhost:5600"],
            "storage_format": "invalid",
        }
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        with pytest.raises(ValueError, match="storage_format"):
            load_config(str(cfg_file))

    def test_empty_yaml_file_raises(self, tmp_path) -> None:
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")

        # Empty file → empty dict → missing data_endpoints → empty list error
        with pytest.raises((ValueError, TypeError)):
            load_config(str(cfg_file))

    def test_optional_fields_have_defaults(self, tmp_path) -> None:
        cfg_data = {"data_endpoints": ["tcp://localhost:5600"]}
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(str(cfg_file))
        assert cfg.node_id == "recorder"
        assert cfg.fps == 30
