"""Serialized episode writer queue.

``EpisodeWriterQueue`` owns a single background thread that serializes calls
to ``align_episode_data`` and ``backend.write_episode``, replacing the
unbounded per-episode thread spawning pattern in the legacy implementation.
This eliminates the risk of unbounded thread accumulation when episodes are
recorded in rapid succession.

The thread is intentionally **non-daemon** so that Python's interpreter does
not kill it mid-encode when the main thread exits.  Call :meth:`shutdown`
before the process exits to drain the queue cleanly.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from loguru import logger

from dexim.recorder.alignment import align_episode_data
from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.metadata import EpisodeMetadata

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
            tuple[dict[str, list[tuple[float, Any]]], EpisodeMetadata]
            | _ShutdownSentinel
        ] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="episode-writer",
            daemon=False,  # must NOT be daemon: Python must not kill it mid-encode
        )
        self._thread.start()
        logger.debug("EpisodeWriterQueue started")

    def submit(
        self,
        buffers: dict[str, list[tuple[float, Any]]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Queue an episode for writing.

        Non-blocking: returns immediately; the write happens in the
        background thread.

        Args:
            buffers: Deep copy of the episode data buffers (topic → samples).
            metadata: Episode metadata snapshot.
        """
        self._queue.put((buffers, metadata))
        logger.debug(f"Episode {metadata.episode_index} queued for writing")

    def qsize(self) -> int:
        """Return the number of episodes currently waiting to be written."""
        return self._queue.qsize()

    def shutdown(self, timeout: float = 120.0) -> None:
        """Drain the queue, process remaining episodes, then stop the thread.

        The join loop absorbs ``KeyboardInterrupt`` so that a second Ctrl+C
        cannot abort the drain and leave the dataset in an inconsistent state.

        Args:
            timeout: Maximum seconds to wait for the thread to finish after
                the sentinel is enqueued.  Defaults to 120 s to accommodate
                slow video encoding.
        """
        self._queue.put(_SHUTDOWN)
        deadline = time.monotonic() + timeout
        while self._thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "EpisodeWriterQueue.shutdown: timed out — worker thread still alive"
                )
                break
            try:
                self._thread.join(timeout=min(remaining, 2.0))
            except KeyboardInterrupt:
                logger.warning(
                    "EpisodeWriterQueue.shutdown: Ctrl+C ignored — encoding still running, please wait"
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
            buffers, metadata = item
            self._write_one(buffers, metadata)

    def _write_one(
        self,
        buffers: dict[str, list[tuple[float, Any]]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Align buffers and write a single episode to the backend.

        Args:
            buffers: Episode data buffers.
            metadata: Episode metadata snapshot.
        """
        try:
            logger.info(
                f"Writing episode {metadata.episode_index} ({len(buffers)} topics)…"
            )
            frames = align_episode_data(
                buffers,
                master_clock_topic=self._master_clock_topic,
                continuous_topics=self._continuous_topics,
            )
            self._backend.write_episode(frames, metadata)
            logger.success(
                f"Episode {metadata.episode_index} written ({len(frames)} frames)"
            )
        except Exception as exc:
            logger.error(
                f"Episode {metadata.episode_index} write failed — {exc}",
                exc_info=True,
            )
