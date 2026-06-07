"""Abstract base class for episode storage backends."""

from __future__ import annotations

import abc
from typing import Any

from dexim.recorder.metadata import EpisodeMetadata

__all__ = ["StorageBackend"]


class StorageBackend(abc.ABC):
    """Contract for episode storage backends.

    All methods are called from a single serialized ``EpisodeWriterQueue``
    daemon thread, so implementations do not need to be re-entrant.
    The exception is ``LeRobotWriter``, which adds an internal lock as a
    safety net in case the caller pattern ever changes.
    """

    @abc.abstractmethod
    def write_episode(
        self,
        frames: list[dict[str, Any]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Persist one episode to the storage medium.

        Args:
            frames: List of aligned frame dicts.  Each dict maps topic string
                -> data value, plus a ``"timestamp"`` key (float).
            metadata: Episode metadata snapshot (index, task, timing, topics).
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Flush and finalize the storage medium."""
