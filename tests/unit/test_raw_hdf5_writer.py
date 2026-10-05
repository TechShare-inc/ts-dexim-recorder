"""Unit tests for lossless raw HDF5 episode storage."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dexim.recorder.backends.raw_hdf5_writer import (
    RawHDF5Writer,
    read_raw_episode,
)
from dexim.recorder.metadata import EpisodeMetadata


def _metadata(index: int = 0) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        task_id="pick_cup",
        task_description="Pick up the cup",
        start_time=1.0,
        end_time=2.0,
        num_frames=0,
        num_topics=2,
        topics=["action/arm/joint_cmd", "observation/cam/video_frame"],
    )


def _buffers():
    return {
        "action/arm/joint_cmd": [
            (1.01, {"q": [0.1, 0.2]}),
            (1.03, {"q": np.asarray([0.3, 0.4], dtype=np.float32)}),
        ],
        "observation/cam/video_frame": [
            (1.02, {"color": b"jpeg-bytes", "width": 640, "height": 480}),
        ],
    }


def test_round_trip_preserves_raw_timestamps_and_payloads(tmp_path: Path) -> None:
    writer = RawHDF5Writer(
        str(tmp_path),
        topic_to_feature={"action/arm/joint_cmd": "action"},
        features={"action": {"dtype": "float32", "shape": [2]}},
        fps=30,
        master_clock_topic="observation/cam/video_frame",
    )
    writer.write_episode(_buffers(), _metadata())

    path = tmp_path / "episode_000000.raw.h5"
    buffers, metadata, schema = read_raw_episode(path)

    assert metadata.episode_index == 0
    assert metadata.task_id == "pick_cup"
    assert buffers["action/arm/joint_cmd"][0] == (1.01, {"q": [0.1, 0.2]})
    np.testing.assert_array_equal(
        buffers["action/arm/joint_cmd"][1][1]["q"],
        np.asarray([0.3, 0.4], dtype=np.float32),
    )
    assert buffers["observation/cam/video_frame"][0][1]["color"] == b"jpeg-bytes"
    assert schema["topic_to_feature"] == {"action/arm/joint_cmd": "action"}
    assert schema["master_clock_topic"] == "observation/cam/video_frame"


def test_file_contains_lerobot_schema_metadata(tmp_path: Path) -> None:
    writer = RawHDF5Writer(
        str(tmp_path),
        topic_to_feature={"action/arm/joint_cmd": "action"},
        features={"action": {"dtype": "float32", "shape": [2]}},
    )
    writer.write_episode(_buffers(), _metadata())

    with h5py.File(tmp_path / "episode_000000.raw.h5", "r") as hf:
        assert hf.attrs["storage_format"] == "dexim_raw_hdf5"
        assert hf.attrs["schema_version"] == 1
        assert hf.attrs["num_topics"] == 2
        assert hf.attrs["num_samples"] == 3


def test_existing_episode_is_not_overwritten(tmp_path: Path) -> None:
    writer = RawHDF5Writer(str(tmp_path))
    writer.write_episode(_buffers(), _metadata())
    with pytest.raises(FileExistsError):
        writer.write_episode(_buffers(), _metadata())


def test_next_episode_index_continues_existing_session(tmp_path: Path) -> None:
    writer = RawHDF5Writer(str(tmp_path))
    writer.write_episode(_buffers(), _metadata(index=3))
    assert writer.next_episode_index == 4
