"""HDF5 storage backend — one file per episode.

File naming: ``episode_NNNN.h5`` (zero-padded to 4 digits).

Topic → dtype mapping uses pattern matching on the topic string so the
recorder stays format-agnostic.  Override individual topics via
``topic_dtypes`` in the config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from loguru import logger

from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.metadata import EpisodeMetadata

__all__ = ["HDF5Writer"]

# Pattern → (numpy dtype string, optional float→int scale factor).
# Applied in order; first match wins.
_DEFAULT_DTYPE_PATTERNS: list[tuple[re.Pattern[str], str, float | None]] = [
    # Video frames: observation/<id>/video_frame  OR  legacy obs_image_*
    (re.compile(r"observation/.+/video_frame|obs_image_"), "uint8", 255.0),
    # Depth frames: observation/<id>/depth  OR  legacy obs_depth_*
    (re.compile(r"observation/.+/depth|obs_depth_"), "uint16", None),
    # Joint states: observation/<id>/joint_state  OR  legacy obs_joint_state_*
    (re.compile(r"observation/.+/joint_state|obs_joint_state_"), "float32", None),
    # Joint commands: action/<id>/joint_cmd  OR  legacy action_joint_cmd_*
    (re.compile(r"action/.+/joint_cmd|action_joint_cmd_"), "float32", None),
]

_COMPRESSION = "gzip"
_COMPRESSION_OPTS = 4


class HDF5Writer(StorageBackend):
    """Writes each episode as a standalone HDF5 file.

    Args:
        output_dir: Directory to write episode files into.  Created if absent.
        topic_dtypes: Optional per-topic dtype overrides keyed by exact topic
            string (e.g. ``{"observation/cam1/video_frame": "uint8"}``).
    """

    def __init__(
        self,
        output_dir: str,
        topic_dtypes: dict[str, str] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._topic_dtypes: dict[str, str] = topic_dtypes or {}

    def write_episode(
        self,
        frames: list[dict[str, Any]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Write aligned frames to ``episode_NNNN.h5``.

        Args:
            frames: Aligned frame list from ``align_episode_data``.
            metadata: Episode metadata snapshot (index, task, timing, topics).
        """
        if not frames:
            logger.warning(f"Episode {metadata.episode_index}: no frames to write — skipped")
            return

        filepath = self._output_dir / f"episode_{metadata.episode_index:04d}.h5"

        # Collect all topic keys present across any frame (excluding "timestamp")
        all_topics: set[str] = set()
        for frame in frames:
            all_topics.update(k for k in frame if k != "timestamp")

        timestamps = np.array([f["timestamp"] for f in frames], dtype=np.float64)

        with h5py.File(filepath, "w") as hf:
            hf.attrs["task"] = metadata.task_id
            hf.attrs["task_description"] = metadata.task_description
            hf.attrs["start_time"] = metadata.start_time
            hf.attrs["end_time"] = metadata.end_time
            hf.create_dataset(
                "timestamp",
                data=timestamps,
                compression=_COMPRESSION,
                compression_opts=_COMPRESSION_OPTS,
            )
            for topic in sorted(all_topics):
                column = [f.get(topic) for f in frames]
                self._write_topic_dataset(hf, topic, column)

        logger.info(f"Episode {metadata.episode_index} → {filepath} ({len(frames)} frames)")

    def close(self) -> None:
        """No-op: each episode is an independent file."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_topic_dataset(
        self,
        hf: h5py.File,
        topic: str,
        column: list[Any],
    ) -> None:
        """Stack column data and write as a compressed HDF5 dataset.

        Args:
            hf: Open HDF5 file handle.
            topic: Topic string used as the dataset name.
            column: Per-frame data values (may contain None for missing frames).
        """
        valid = [v for v in column if v is not None]
        if not valid:
            logger.debug(f"  topic={topic!r}: all frames None — skipped")
            return

        try:
            arr = np.stack([np.asarray(v) for v in valid])
        except (ValueError, TypeError) as exc:
            logger.warning(f"  topic={topic!r}: cannot stack values — {exc}")
            return

        dtype_str, scale = self._resolve_dtype(topic)
        if scale is not None and arr.dtype.kind == "f":
            arr = (arr * scale).clip(0, np.iinfo(dtype_str).max)
        arr = arr.astype(dtype_str)

        hf.create_dataset(
            topic,
            data=arr,
            compression=_COMPRESSION,
            compression_opts=_COMPRESSION_OPTS,
        )

    def _resolve_dtype(self, topic: str) -> tuple[str, float | None]:
        """Return the numpy dtype and optional float→int scale for a topic.

        Checks ``topic_dtypes`` overrides first, then pattern-matches.

        Args:
            topic: Topic string.

        Returns:
            Tuple of ``(dtype_str, scale_factor)``.  ``scale_factor`` is
            ``None`` when no floating-point scaling is required.
        """
        if topic in self._topic_dtypes:
            return self._topic_dtypes[topic], None
        for pattern, dtype_str, scale in _DEFAULT_DTYPE_PATTERNS:
            if pattern.search(topic):
                return dtype_str, scale
        return "float32", None
