"""Unit tests for offline raw HDF5 to LeRobot conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dexim.recorder.backends.raw_hdf5_writer import RawHDF5Writer
from dexim.recorder.metadata import EpisodeMetadata
from dexim.recorder.raw_conversion import convert_raw_dataset


def _metadata(index: int) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        task_id="test_task",
        task_description="Test task",
        start_time=float(index),
        end_time=float(index + 1),
        num_frames=0,
        num_topics=1,
        topics=["action/arm/joint_cmd"],
    )


def test_converts_all_raw_episodes_and_finalizes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    writer = RawHDF5Writer(
        str(raw_dir),
        topic_to_feature={"action/arm/joint_cmd": "action"},
        features={"action": {"dtype": "float32", "shape": [2]}},
        master_clock_topic="action/arm/joint_cmd",
    )
    for index in range(2):
        writer.write_episode(
            {"action/arm/joint_cmd": [(index + 0.1, {"q": [0.1, 0.2]})]},
            _metadata(index),
        )

    lerobot_writer = MagicMock()
    with patch(
        "dexim.recorder.raw_conversion.LeRobotWriter",
        return_value=lerobot_writer,
    ):
        converted = convert_raw_dataset(
            input_dir=str(raw_dir),
            dataset_path=str(tmp_path / "lerobot"),
            repo_id="local/test",
        )

    assert converted == 2
    assert lerobot_writer.write_episode.call_count == 2
    lerobot_writer.close.assert_called_once_with()


def test_refuses_existing_output_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    writer = RawHDF5Writer(str(raw_dir))
    writer.write_episode(
        {"action/arm/joint_cmd": [(0.1, {"q": [0.1, 0.2]})]},
        _metadata(0),
    )
    output_dir = tmp_path / "lerobot"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        convert_raw_dataset(
            input_dir=str(raw_dir),
            dataset_path=str(output_dir),
            repo_id="local/test",
        )
