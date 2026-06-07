"""Unit tests for dexim.recorder.backends.hdf5_writer.HDF5Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexim.recorder.backends.hdf5_writer import HDF5Writer
from dexim.recorder.metadata import EpisodeMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(index: int = 0) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        task_id="pick_cup",
        task_description="Pick up the red cup",
        start_time=0.0,
        end_time=2.0,
        num_frames=3,
        num_topics=2,
        topics=["obs/arm/joint_state", "obs/cam/video_frame"],
    )


def _make_frames(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": float(i),
            "obs/arm/joint_state": [float(i)] * 6,
            "obs/cam/video_frame": np.ones((4, 4, 3), dtype=np.float32) * (i / n),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _resolve_dtype: pattern matching
# ---------------------------------------------------------------------------


class TestResolveDtype:
    """Test _resolve_dtype via the public write_episode output dtypes."""

    def _write_and_open(
        self,
        tmp_path: Path,
        topic: str,
        column: list[Any],
        topic_dtypes: dict[str, str] | None = None,
    ) -> h5py.File:
        frames = [{"timestamp": float(i), topic: v} for i, v in enumerate(column)]
        writer = HDF5Writer(output_dir=str(tmp_path), topic_dtypes=topic_dtypes)
        writer.write_episode(frames, _make_metadata())
        return h5py.File(tmp_path / "episode_0000.h5", "r")

    def test_joint_state_is_float32(self, tmp_path: Path) -> None:
        with self._write_and_open(
            tmp_path, "obs/arm/joint_state", [[0.1] * 6] * 2
        ) as f:
            assert f["obs/arm/joint_state"].dtype == np.float32

    def test_video_frame_is_uint8(self, tmp_path: Path) -> None:
        frames_data = [np.ones((4, 4, 3), dtype=np.float32) * 0.5] * 2
        with self._write_and_open(
            tmp_path, "observation/cam/video_frame", frames_data
        ) as f:
            assert f["observation/cam/video_frame"].dtype == np.uint8

    def test_depth_is_uint16(self, tmp_path: Path) -> None:
        frames_data = [np.ones((4, 4), dtype=np.float32) * 0.5] * 2
        with self._write_and_open(tmp_path, "observation/cam1/depth", frames_data) as f:
            assert f["observation/cam1/depth"].dtype == np.uint16

    def test_unknown_topic_defaults_to_float32(self, tmp_path: Path) -> None:
        with self._write_and_open(tmp_path, "custom/sensor", [[1.0, 2.0]] * 2) as f:
            assert f["custom/sensor"].dtype == np.float32

    def test_explicit_override_wins(self, tmp_path: Path) -> None:
        overrides = {"obs/arm/joint_state": "float64"}
        with self._write_and_open(
            tmp_path, "obs/arm/joint_state", [[0.1] * 6] * 2, topic_dtypes=overrides
        ) as f:
            assert f["obs/arm/joint_state"].dtype == np.float64

    def test_action_joint_cmd_is_float32(self, tmp_path: Path) -> None:
        with self._write_and_open(
            tmp_path, "action/arm/joint_cmd", [[0.1] * 6] * 2
        ) as f:
            assert f["action/arm/joint_cmd"].dtype == np.float32


# ---------------------------------------------------------------------------
# write_episode: file content
# ---------------------------------------------------------------------------


class TestWriteEpisode:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(_make_frames(), _make_metadata())
        assert (tmp_path / "episode_0000.h5").exists()

    def test_episode_index_in_filename(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(_make_frames(), _make_metadata(index=7))
        assert (tmp_path / "episode_0007.h5").exists()

    def test_creates_output_dir_if_absent(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        writer = HDF5Writer(output_dir=str(nested))
        writer.write_episode(_make_frames(), _make_metadata())
        assert (nested / "episode_0000.h5").exists()

    def test_timestamp_dataset_present(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        frames = _make_frames(3)
        writer.write_episode(frames, _make_metadata())
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            assert "timestamp" in f
            assert len(f["timestamp"]) == 3

    def test_task_metadata_in_attrs(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(_make_frames(), _make_metadata())
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            assert f.attrs["task"] == "pick_cup"
            assert f.attrs["task_description"] == "Pick up the red cup"

    def test_all_topics_written(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        frames = _make_frames(3)
        writer.write_episode(frames, _make_metadata())
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            assert "obs/arm/joint_state" in f
            assert "obs/cam/video_frame" in f

    def test_joint_state_shape(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(_make_frames(3), _make_metadata())
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            # 3 frames × 6 joints
            assert f["obs/arm/joint_state"].shape == (3, 6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_frames_skipped(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode([], _make_metadata())
        assert not (tmp_path / "episode_0000.h5").exists()

    def test_none_values_for_topic_skipped(self, tmp_path: Path) -> None:
        frames = [
            {"timestamp": 0.0, "obs/arm/joint_state": None},
            {"timestamp": 1.0, "obs/arm/joint_state": None},
        ]
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(frames, _make_metadata())
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            assert "obs/arm/joint_state" not in f

    def test_close_is_noop(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.close()  # should not raise

    def test_multiple_episodes_independent_files(self, tmp_path: Path) -> None:
        writer = HDF5Writer(output_dir=str(tmp_path))
        writer.write_episode(_make_frames(2), _make_metadata(index=0))
        writer.write_episode(_make_frames(4), _make_metadata(index=1))
        with h5py.File(tmp_path / "episode_0000.h5", "r") as f:
            assert len(f["timestamp"]) == 2
        with h5py.File(tmp_path / "episode_0001.h5", "r") as f:
            assert len(f["timestamp"]) == 4
