"""dexim-recorder — multi-stream episode data collection for imitation learning.

Supports HDF5 and LeRobot v3 storage backends with temporal alignment across
N ZMQ data streams.
"""

from __future__ import annotations

from dexim.recorder.alignment import align_episode_data
from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.backends.hdf5_writer import HDF5Writer
from dexim.recorder.config import RecorderNodeConfig, load_config
from dexim.recorder.metadata import EpisodeMetadata, TaskInfo
from dexim.recorder.node import DataRecorderNode
from dexim.recorder.writer_thread import EpisodeWriterQueue

__version__ = "0.1.0"

__all__ = [
    "DataRecorderNode",
    "EpisodeMetadata",
    "EpisodeWriterQueue",
    "HDF5Writer",
    "RecorderNodeConfig",
    "StorageBackend",
    "TaskInfo",
    "align_episode_data",
    "load_config",
]
