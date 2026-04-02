"""Serialized episode writer queue.

``EpisodeWriterQueue`` owns a single daemon thread that serializes calls to
``align_episode_data`` and ``backend.write_episode``, replacing the unbounded
per-episode thread spawning pattern in the legacy implementation.  This
eliminates the risk of unbounded thread accumulation when episodes are
recorded in rapid succession.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from loguru import logger

from dexim.recorder.alignment import align_episode_data
from dexim.recorder.backends.base import StorageBackend

__all__ = ["EpisodeWriterQueue"]


class _ShutdownSentinel:
    """Typed sentinel to signal the writer thread to stop."""


_SHUTDOWN = _ShutdownSentinel()


class EpisodeWriterQueue:
    """Serialized writer queue for episode data.

    Episodes are submitted via :meth:`submit` from the recorder's main thread
    and processed one-by-one by a single background daemon thread.

    Args:
        backend: Storage backend (``HDF5Writer`` or ``LeRobotWriter``).
        master_clock_topic: Explicit master clock topic, or ``None`` for
            auto-detection.
        continuous_topics: Topics to align with linear interpolation.
    """

    def __init__(
        self,
        backend: StorageBackend,
        master_clock_topic: str | None = None,
        continuous_topics: list[str] | None = None,
    ) -> None:
        self._backend = backend
        self._master_clock_topic = master_clock_topic
        self._continuous_topics: list[str] = list(continuous_topics or [])
        self._queue: queue.Queue[
            tuple[dict[str, list[tuple[float, Any]]], int, str] | _ShutdownSentinel
        ] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="episode-writer",
            daemon=True,
        )
        self._thread.start()
        logger.debug("EpisodeWriterQueue started")

    def submit(
        self,
        buffers: dict[str, list[tuple[float, Any]]],
        episode_number: int,
        task: str = "",
    ) -> None:
        """Queue an episode for writing.

        Non-blocking: returns immediately; the write happens in the
        background thread.

        Args:
            buffers: Deep copy of the episode data buffers (topic → samples).
            episode_number: Sequential episode index (1-based).
            task: Task name to embed in episode metadata.
        """
        self._queue.put((buffers, episode_number, task))
        logger.debug(f"Episode {episode_number} queued for writing")

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain the queue, process remaining episodes, then stop the thread.

        Args:
            timeout: Maximum seconds to wait for the thread to finish after
                the sentinel is enqueued.
        """
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "EpisodeWriterQueue.shutdown: timed out — worker thread still alive"
            )
        else:
            logger.debug("EpisodeWriterQueue shut down cleanly")

    # ------------------------------------------------------------------
    # Worker (background daemon thread)
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Dequeue and write episodes until the shutdown sentinel is received."""
        while True:
            item = self._queue.get()
            if isinstance(item, _ShutdownSentinel):
                logger.debug("EpisodeWriterQueue worker: shutdown sentinel received")
                break
            buffers, episode_number, task = item
            self._write_one(buffers, episode_number, task)

    def _write_one(
        self,
        buffers: dict[str, list[tuple[float, Any]]],
        episode_number: int,
        task: str,
    ) -> None:
        """Align buffers and write a single episode to the backend.

        Args:
            buffers: Episode data buffers.
            episode_number: Episode index.
            task: Task label.
        """
        try:
            logger.info(f"Writing episode {episode_number} ({len(buffers)} topics)…")
            frames = align_episode_data(
                buffers,
                master_clock_topic=self._master_clock_topic,
                continuous_topics=self._continuous_topics,
            )
            self._backend.write_episode(frames, episode_number, task)
            logger.success(f"Episode {episode_number} written ({len(frames)} frames)")
        except Exception as exc:
            logger.error(
                f"Episode {episode_number} write failed — {exc}",
                exc_info=True,
            )
