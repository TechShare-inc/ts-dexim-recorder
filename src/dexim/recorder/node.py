"""DataRecorderNode — multi-stream episode recorder.

Extends ``ManagedNode`` directly (no RecorderNode intermediary) and owns
all ZMQ subscription, buffering, and episode lifecycle responsibilities.

Data-flow pipeline::

    ZMQ PUB endpoints
        │  (one SUB socket per endpoint, all on shared Poller)
        ▼
    _handle_data_message()
        │  is_recording → buffer (topic → [(timestamp, data)])
        │  not recording → drain and discard
        ▼
    on_stop_recording()
        │  deep-copy buffers → submit to EpisodeWriterQueue
        ▼
    EpisodeWriterQueue (single daemon thread)
        │  align_episode_data() → backend.write_episode()
        ▼
    HDF5Writer / LeRobotWriter
"""

from __future__ import annotations

import copy
import threading
from collections import defaultdict
from typing import Any

import zmq
from dexim.core.messages import unpack_data_message
from dexim.core.nodes.managed import ManagedNode
from loguru import logger

from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.backends.hdf5_writer import HDF5Writer
from dexim.recorder.config import RecorderNodeConfig
from dexim.recorder.writer_thread import EpisodeWriterQueue

__all__ = ["DataRecorderNode"]


def _build_backend(config: RecorderNodeConfig) -> StorageBackend:
    """Construct the storage backend from config.

    Args:
        config: Recorder node configuration.

    Returns:
        Configured ``StorageBackend`` instance.

    Raises:
        ValueError: For an unknown ``storage_format``.
    """
    if config.storage_format == "hdf5":
        return HDF5Writer(
            output_dir=config.output_dir,
            topic_dtypes=config.topic_dtypes or None,
        )
    if config.storage_format == "lerobot":
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        return LeRobotWriter(
            dataset_path=config.lerobot_dataset_path,
            repo_id=config.lerobot_repo_id,
            features=config.features,
            topic_to_feature=config.topic_to_feature,
            fps=config.fps,
            push_to_hub=config.push_to_hub,
        )
    raise ValueError(
        f"Unknown storage_format {config.storage_format!r}. "
        "Supported: 'hdf5', 'lerobot'."
    )


class DataRecorderNode(ManagedNode):
    """Records multi-stream ZMQ data as timestamped episodes.

    Subscribes to a list of ZMQ PUB endpoints and buffers all data messages
    while ``is_recording`` is ``True``.  When a STOP_REC command is received,
    the buffers are deep-copied and submitted asynchronously to an
    ``EpisodeWriterQueue`` for alignment and storage.

    Args:
        config: Fully validated ``RecorderNodeConfig`` instance.
    """

    def __init__(self, config: RecorderNodeConfig) -> None:
        """Initialise ZMQ sockets, storage backend, and writer queue.

        Args:
            config: ``RecorderNodeConfig`` instance (validated in __post_init__).
        """
        super().__init__(
            node_id=config.node_id,
            control_endpoint=config.control_endpoint,
            status_endpoint=config.status_endpoint,
        )
        self._config = config

        # Storage backend — validated before sockets are opened.
        self._backend: StorageBackend = _build_backend(config)

        # Single-thread writer queue (fixes unbounded thread spawning bug).
        self._writer_queue = EpisodeWriterQueue(
            backend=self._backend,
            master_clock_topic=config.master_clock_topic,
            continuous_topics=config.continuous_topics,
        )

        # Episode state.
        self._episode_counter: int = 0
        self._buffers: dict[str, list[tuple[float, Any]]] = defaultdict(list)
        self._buffer_lock = threading.Lock()

        # Data-plane SUB sockets (one per endpoint, all on shared _poller).
        self._sub_data_sockets: list[zmq.Socket] = []
        self._initialize_data_sockets()

        logger.success(
            f"DataRecorderNode ready — "
            f"format={config.storage_format!r}, "
            f"endpoints={config.data_endpoints}"
        )

    # ------------------------------------------------------------------
    # ZMQ setup / teardown
    # ------------------------------------------------------------------

    def _initialize_data_sockets(self) -> None:
        """Create one SUB socket per data endpoint and register with poller."""
        if self._ctx is None:
            self._ctx = zmq.Context.instance()
        for endpoint in self._config.data_endpoints:
            sub: zmq.Socket = self._ctx.socket(zmq.SUB)
            sub.connect(endpoint)
            sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
            self._sub_data_sockets.append(sub)
            if self._poller is not None:
                self._poller.register(sub, zmq.POLLIN)
            logger.debug(f"Subscribed to data endpoint: {endpoint}")

    def _cleanup_zmq(self) -> None:
        """Close data SUB sockets, then delegate to ``ManagedNode``."""
        for sock in self._sub_data_sockets:
            try:
                sock.close(linger=0)
            except Exception:
                pass
        self._sub_data_sockets.clear()
        super()._cleanup_zmq()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop_iteration(self) -> None:
        """Poll data sockets and buffer messages when recording."""
        if self._poller is None:
            return
        events = dict(self._poller.poll(timeout=10))  # 10 ms
        for sock in self._sub_data_sockets:
            if sock in events:
                self._handle_data_message(sock)

    def _handle_data_message(self, sock: zmq.Socket) -> None:
        """Receive one message and buffer it if recording is active.

        When not recording, messages are received and discarded to prevent
        socket buffer back-pressure.

        Args:
            sock: ZMQ SUB socket with data waiting.
        """
        if not self.is_recording:
            try:
                sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            return

        try:
            frames = sock.recv_multipart(flags=zmq.NOBLOCK)
            if len(frames) != 2:
                return

            message = unpack_data_message(frames)
            topic: str = message.get("topic", "")
            timestamp = message.get("timestamp")
            data = message.get("data")

            if not topic or timestamp is None or data is None:
                return

            with self._buffer_lock:
                self._buffers[topic].append((float(timestamp), data))

        except zmq.Again:
            pass
        except Exception as exc:
            logger.debug(f"Error handling data message: {exc}")

    # ------------------------------------------------------------------
    # ManagedNode lifecycle hooks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Clear buffers when the node enters active operation."""
        with self._buffer_lock:
            self._buffers.clear()
        logger.info(f"{self.node_id}: started — buffers cleared")

    def on_pause(self) -> None:
        """Hold current state on pause."""
        logger.info(f"{self.node_id}: paused")

    def on_stop(self) -> None:
        """Safe-stop (no additional action needed for a recorder)."""
        logger.info(f"{self.node_id}: stopped")

    def on_start_recording(self) -> None:
        """Clear buffers at the start of each recording window."""
        with self._buffer_lock:
            self._buffers.clear()
        logger.info(f"{self.node_id}: recording started — buffers cleared")

    def on_stop_recording(self) -> None:
        """Deep-copy buffers and submit the episode to the writer queue."""
        with self._buffer_lock:
            if not self._buffers:
                logger.warning(
                    f"{self.node_id}: STOP_REC received but buffers are empty"
                )
                return
            episode_buffers = copy.deepcopy(dict(self._buffers))
            self._buffers.clear()

        self._episode_counter += 1
        episode_number = self._episode_counter
        logger.info(
            f"{self.node_id}: queuing episode {episode_number} "
            f"({len(episode_buffers)} topics)"
        )
        self._writer_queue.submit(
            buffers=episode_buffers,
            episode_number=episode_number,
            task=self._config.task,
        )

    def on_shutdown(self) -> None:
        """Drain the writer queue and finalize the storage backend."""
        logger.info(f"{self.node_id}: shutting down — draining writer queue…")
        self._writer_queue.shutdown(timeout=60.0)
        try:
            self._backend.close()
        except Exception as exc:
            logger.error(f"{self.node_id}: backend.close() failed — {exc}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_buffer_stats(self) -> dict[str, int]:
        """Return per-topic sample counts in the current buffer.

        Returns:
            Dict mapping topic string to the number of buffered samples.
        """
        with self._buffer_lock:
            return {topic: len(msgs) for topic, msgs in self._buffers.items()}
