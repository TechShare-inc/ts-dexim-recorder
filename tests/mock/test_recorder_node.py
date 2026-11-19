"""Mock tests for DataRecorderNode.

Tests validate the recorder's buffer management, episode submission, and
lifecycle hooks without any ZMQ connections, filesystem writes, or background
threads.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dexim.recorder.config import RecorderNodeConfig
from dexim.recorder.node import DataRecorderNode, _build_backend

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------


@pytest.fixture()
def recorder_config() -> RecorderNodeConfig:
    """Minimal valid RecorderNodeConfig for test construction."""
    return RecorderNodeConfig(
        node_id="test-recorder",
        data_endpoints=["tcp://localhost:5600"],
        storage_format="hdf5",
        output_dir="/tmp/dexim_recorder_test",
        default_task="pick_and_place",
    )


# ---------------------------------------------------------------------------
# Node construction helper
# ---------------------------------------------------------------------------


@pytest.fixture()
def recorder_node(
    recorder_config,
    patch_managed_node,
    patch_writer_queue,
    patch_build_backend,
) -> DataRecorderNode:
    """Construct a DataRecorderNode with all external I/O patched out."""
    return DataRecorderNode(config=recorder_config)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_node_id_set(self, recorder_node, recorder_config):
        assert recorder_node.node_id == recorder_config.node_id

    def test_data_sockets_created(self, recorder_node, recorder_config):
        # One data socket per endpoint
        assert len(recorder_node._sub_data_sockets) == len(
            recorder_config.data_endpoints
        )

    def test_data_sockets_registered_with_poller(self, recorder_node):
        poller = recorder_node._poller
        for sock in recorder_node._sub_data_sockets:
            poller.register.assert_any_call(sock, 1)  # zmq.POLLIN == 1

    def test_episode_counter_starts_at_zero(self, recorder_node):
        assert recorder_node._episode_counter == 0

    def test_lerobot_encoding_config_reaches_backend(self) -> None:
        config = RecorderNodeConfig(
            data_endpoints=["tcp://localhost:5600"],
            storage_format="lerobot",
            lerobot_vcodec="h264",
            lerobot_parallel_encoding=True,
            lerobot_encoder_threads=2,
        )

        writer_cls = MagicMock()
        fake_module = MagicMock(LeRobotWriter=writer_cls)
        with patch.dict(
            sys.modules,
            {"dexim.recorder.backends.lerobot_writer": fake_module},
        ):
            _build_backend(config)

        call_kwargs = writer_cls.call_args.kwargs
        assert call_kwargs["vcodec"] == "h264"
        assert call_kwargs["parallel_encoding"] is True
        assert call_kwargs["encoder_threads"] == 2


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class TestOnStart:
    def test_on_start_clears_buffers(self, recorder_node):
        recorder_node._buffers["some/topic"].append((1.0, [1, 2, 3]))
        recorder_node.on_start()
        assert len(recorder_node._buffers) == 0


class TestOnStop:
    def test_waits_for_active_episode_write(
        self, recorder_node, patch_writer_queue
    ) -> None:
        patch_writer_queue.pending_count.return_value = 1

        recorder_node.on_stop()

        patch_writer_queue.wait_until_idle.assert_called_once_with(
            timeout=recorder_node.heartbeat_interval
        )

    def test_reports_heartbeat_while_writer_is_busy(
        self, recorder_node, patch_writer_queue
    ) -> None:
        patch_writer_queue.pending_count.return_value = 1
        patch_writer_queue.wait_until_idle.side_effect = [False, True]
        recorder_node.report_status = MagicMock()

        recorder_node.on_stop()

        recorder_node.report_status.assert_called_once()


class TestOnStartRecording:
    def test_clears_buffers_on_start_recording(self, recorder_node):
        recorder_node._buffers["obs/cam/frame"].append((0.5, b"img"))
        recorder_node.on_start_recording()
        assert len(recorder_node._buffers) == 0


class TestOnStopRecording:
    def test_empty_buffers_does_not_submit(self, recorder_node, patch_writer_queue):
        recorder_node.on_stop_recording()
        patch_writer_queue.submit.assert_not_called()

    def test_submits_episode_when_buffers_not_empty(
        self, recorder_node, patch_writer_queue, recorder_config
    ):
        recorder_node.on_start_recording()
        recorder_node._buffers["obs/arm/joint_state"].append((1.0, [0.1, 0.2]))
        recorder_node._buffers["obs/cam/video_frame"].append((1.0, b"frame"))

        recorder_node.on_stop_recording()

        patch_writer_queue.submit.assert_called_once()
        call_kwargs = patch_writer_queue.submit.call_args.kwargs
        assert call_kwargs["metadata"].episode_index == 0
        assert call_kwargs["metadata"].task_id == recorder_config.default_task
        assert "obs/arm/joint_state" in call_kwargs["buffers"]
        assert "obs/cam/video_frame" in call_kwargs["buffers"]

    def test_episode_counter_increments(self, recorder_node, patch_writer_queue):
        recorder_node._buffers["topicA"].append((1.0, 42))
        recorder_node.on_stop_recording()
        assert recorder_node._episode_counter == 1
        recorder_node._buffers["topicA"].append((2.0, 99))
        recorder_node.on_stop_recording()
        assert recorder_node._episode_counter == 2

    def test_buffers_cleared_after_stop_recording(
        self, recorder_node, patch_writer_queue
    ):
        recorder_node._buffers["topicA"].append((1.0, 42))
        recorder_node.on_stop_recording()
        assert len(recorder_node._buffers) == 0

    def test_submitted_buffers_are_deep_copy(self, recorder_node, patch_writer_queue):
        original = [0.1, 0.2, 0.3]
        recorder_node._buffers["obs/arm/joint_state"].append((1.0, original))
        recorder_node.on_stop_recording()

        submitted = patch_writer_queue.submit.call_args.kwargs["buffers"]
        submitted_data = submitted["obs/arm/joint_state"][0][1]
        # Mutating original should not affect the submitted copy
        original.append(999)
        assert submitted_data == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Message buffering
# ---------------------------------------------------------------------------


class TestHandleDataMessage:
    def _make_mock_sock(self, frames):
        """Return a mock ZMQ socket that yields `frames` on recv_multipart."""

        sock = MagicMock()
        sock.recv_multipart.return_value = frames
        return sock

    def test_discards_when_not_recording(self, recorder_node):
        recorder_node.is_recording = False
        sock = self._make_mock_sock([b"topic", b"payload"])
        recorder_node._handle_data_message(sock)
        assert len(recorder_node._buffers) == 0

    def test_buffers_when_recording(self, recorder_node):
        import time

        import msgpack

        recorder_node.is_recording = True
        ts = time.time()
        payload = msgpack.packb({"timestamp": ts, "data": [1.0, 2.0]})
        topic_bytes = b"obs/arm/joint_state"

        sock = self._make_mock_sock([topic_bytes, payload])

        with patch("dexim.recorder.node.unpack_data_message") as mock_unpack:
            mock_unpack.return_value = {
                "topic": "obs/arm/joint_state",
                "timestamp": ts,
                "data": [1.0, 2.0],
            }
            recorder_node._handle_data_message(sock)

        assert "obs/arm/joint_state" in recorder_node._buffers
        assert len(recorder_node._buffers["obs/arm/joint_state"]) == 1
        ts_actual, data = recorder_node._buffers["obs/arm/joint_state"][0]
        assert ts_actual == pytest.approx(ts)
        assert data == [1.0, 2.0]

    def test_ignores_malformed_frames(self, recorder_node):
        recorder_node.is_recording = True
        # 3-frame message is not the expected 2-frame format
        sock = self._make_mock_sock([b"a", b"b", b"c"])
        recorder_node._handle_data_message(sock)
        assert len(recorder_node._buffers) == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestOnShutdown:
    def test_shutdown_calls_writer_queue_shutdown(
        self, recorder_node, patch_writer_queue
    ):
        recorder_node.on_shutdown()
        patch_writer_queue.shutdown.assert_called_once()

    def test_shutdown_calls_backend_close(self, recorder_node, patch_build_backend):
        recorder_node.on_shutdown()
        patch_build_backend.close.assert_called_once()


# ---------------------------------------------------------------------------
# Buffer stats
# ---------------------------------------------------------------------------


class TestGetBufferStats:
    def test_returns_correct_counts(self, recorder_node):
        recorder_node._buffers["topicA"].extend([(1.0, 1), (2.0, 2)])
        recorder_node._buffers["topicB"].append((1.0, 42))
        stats = recorder_node.get_buffer_stats()
        assert stats == {"topicA": 2, "topicB": 1}

    def test_empty_buffers_returns_empty_dict(self, recorder_node):
        assert recorder_node.get_buffer_stats() == {}

    def test_status_counts_active_writer_job(
        self, recorder_node, patch_writer_queue
    ) -> None:
        patch_writer_queue.pending_count.return_value = 1

        info = recorder_node.get_status_info()

        assert info.writer_queue_depth == 1
