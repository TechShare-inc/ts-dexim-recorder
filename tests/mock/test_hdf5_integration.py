"""End-to-end integration test: DataRecorderNode -> EpisodeWriterQueue -> HDF5Writer.

Uses real EpisodeWriterQueue and HDF5Writer (no mock for I/O), but patches
ManagedNode.__init__ to skip ZMQ setup so no ports are bound.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from dexim.recorder.config import RecorderNodeConfig
from dexim.recorder.node import DataRecorderNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hdf5_config(tmp_path: Path) -> RecorderNodeConfig:
    return RecorderNodeConfig(
        node_id="test-recorder",
        data_endpoints=["tcp://localhost:5600"],
        storage_format="hdf5",
        output_dir=str(tmp_path / "episodes"),
        default_task="integration_test",
        fps=10,
    )


@pytest.fixture()
def hdf5_node(hdf5_config, patch_managed_node):
    """DataRecorderNode with real HDF5Writer and EpisodeWriterQueue, ZMQ patched.

    ``patch_managed_node`` replaces ManagedNode.__init__ with a fake that installs
    MagicMock _ctx and _poller, so _initialize_data_sockets() runs safely without
    binding real ZMQ ports.
    """
    node = DataRecorderNode(config=hdf5_config)
    yield node
    # Drain the writer queue gracefully
    node.on_shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_buffers(node: DataRecorderNode, n_frames: int = 5) -> None:
    """Directly populate node buffers without going through ZMQ."""
    with node._buffer_lock:
        for i in range(n_frames):
            t = float(i)
            node._buffers["obs/arm/joint_state"].append((t, [float(i)] * 6))
            node._buffers["obs/cam/video_frame"].append(
                (t, [[[i % 256] * 3 for _ in range(4)] for _ in range(4)])
            )


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestHdf5EndToEnd:
    def test_episode_file_created_after_stop_recording(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=5)
        hdf5_node.on_stop_recording()

        # Wait for writer thread to flush
        hdf5_node._writer_queue.shutdown(timeout=10.0)

        episode_file = tmp_path / "episodes" / "episode_0000.h5"
        assert episode_file.exists(), f"Expected {episode_file} to exist"

    def test_episode_has_correct_dataset_structure(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=5)
        hdf5_node.on_stop_recording()
        hdf5_node._writer_queue.shutdown(timeout=10.0)

        with h5py.File(tmp_path / "episodes" / "episode_0000.h5", "r") as f:
            assert "timestamp" in f
            assert "obs/arm/joint_state" in f
            assert f["obs/arm/joint_state"].shape[0] == 5  # 5 frames

    def test_task_metadata_written_to_attrs(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=3)
        hdf5_node.on_stop_recording()
        hdf5_node._writer_queue.shutdown(timeout=10.0)

        with h5py.File(tmp_path / "episodes" / "episode_0000.h5", "r") as f:
            assert f.attrs["task"] == "integration_test"

    def test_episode_counter_increments_across_episodes(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=3)
        hdf5_node.on_stop_recording()

        # Re-enable recording for second episode
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=3)
        hdf5_node.on_stop_recording()

        hdf5_node._writer_queue.shutdown(timeout=10.0)

        assert (tmp_path / "episodes" / "episode_0000.h5").exists()
        assert (tmp_path / "episodes" / "episode_0001.h5").exists()
        assert hdf5_node._episode_counter == 2

    def test_discard_does_not_write_file(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        _inject_buffers(hdf5_node, n_frames=3)
        hdf5_node.on_discard_recording()
        hdf5_node._writer_queue.shutdown(timeout=5.0)

        assert not (tmp_path / "episodes" / "episode_0000.h5").exists()
        assert hdf5_node._episode_counter == 0

    def test_empty_buffers_does_not_write_file(
        self, hdf5_node, hdf5_config, tmp_path: Path
    ) -> None:
        hdf5_node.on_start_recording()
        # No data injected
        hdf5_node.on_stop_recording()
        hdf5_node._writer_queue.shutdown(timeout=5.0)

        assert not (tmp_path / "episodes" / "episode_0000.h5").exists()
