"""Configuration dataclasses for DataRecorderNode.

Provides a config hierarchy for the recorder:
- storage format selection (hdf5 / lerobot)
- data endpoint list
- temporal alignment settings
- backend-specific settings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dexim.core.messages import CTRL_PUB_ENDPOINT, STATUS_PULL_ENDPOINT

__all__ = [
    "RecorderNodeConfig",
    "load_config",
]


@dataclass
class RecorderNodeConfig:
    """Complete configuration for DataRecorderNode.

    Attributes:
        node_id: Unique node identifier.
        data_endpoints: ZMQ endpoints to subscribe to (non-empty list).
        storage_format: Storage backend -- ``"hdf5"`` or ``"lerobot"``.
        output_dir: Directory for HDF5 episode files or base path for the
            LeRobot dataset.
        default_task: Fallback task name used when no SET_TASK command has been
            received. Runtime task metadata supplied via SET_TASK takes
            precedence over this value.
        master_clock_topic: Topic used as the timing reference for alignment.
            Auto-selected when ``None`` (video_frame topic > most-samples).
        continuous_topics: Topics aligned with linear interpolation.
            All other topics use nearest-neighbour alignment.
            Topics containing ``/joint_state`` are always treated as continuous
            regardless of this list.
        topic_dtypes: HDF5 dtype overrides keyed by exact topic string.
            Pattern-based defaults apply when the topic is not listed here.
        topic_to_feature: Mapping from topic string to LeRobot feature name.
            Required for the LeRobot backend.
        features: LeRobot feature schema dict required when creating a new
            dataset.
        lerobot_dataset_path: Root path for the LeRobot dataset directory.
        lerobot_repo_id: HuggingFace repo ID (e.g. ``"org/dataset-name"``).
        push_to_hub: Push dataset to HuggingFace Hub on close.
        fps: Frame rate used in LeRobot dataset creation.
        lerobot_vcodec: Video codec passed to LeRobot.
        lerobot_parallel_encoding: Encode camera streams concurrently when
            saving an episode. Disabled by default to avoid CPU oversubscription.
        lerobot_encoder_threads: Maximum encoder threads per camera stream.
        control_endpoint: ZMQ control plane endpoint.
        status_endpoint: ZMQ status plane endpoint.
    """

    node_id: str = "recorder"
    data_endpoints: list[str] = field(default_factory=list)
    storage_format: str = "lerobot"
    output_dir: str = "output/episodes"
    default_task: str = ""
    master_clock_topic: str | None = None
    continuous_topics: list[str] = field(default_factory=list)
    topic_dtypes: dict[str, str] = field(default_factory=dict)
    topic_to_feature: dict[str, str] = field(default_factory=dict)
    features: dict[str, Any] | None = None
    lerobot_dataset_path: str = "output/lerobot"
    lerobot_repo_id: str = ""
    push_to_hub: bool = False
    fps: int = 30
    lerobot_vcodec: str = "libsvtav1"
    lerobot_parallel_encoding: bool = False
    lerobot_encoder_threads: int = 4
    control_endpoint: str = CTRL_PUB_ENDPOINT
    status_endpoint: str = STATUS_PULL_ENDPOINT

    def __post_init__(self) -> None:
        """Validate configuration on construction.

        Raises:
            ValueError: When ``data_endpoints`` is empty, ``storage_format``
                is unknown, or ``fps`` is non-positive.
        """
        if not self.data_endpoints:
            raise ValueError("data_endpoints must be a non-empty list")
        allowed_formats = {"hdf5", "lerobot"}
        if self.storage_format not in allowed_formats:
            raise ValueError(
                f"storage_format must be one of {allowed_formats}, "
                f"got {self.storage_format!r}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.lerobot_vcodec.strip():
            raise ValueError("lerobot_vcodec must be non-empty")
        if self.lerobot_encoder_threads <= 0:
            raise ValueError(
                "lerobot_encoder_threads must be positive, "
                f"got {self.lerobot_encoder_threads}"
            )


def load_config(config_path: str) -> RecorderNodeConfig:
    """Load RecorderNodeConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Populated RecorderNodeConfig instance.

    Raises:
        FileNotFoundError: When the config file does not exist.
        ValueError: When the loaded config fails validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return RecorderNodeConfig(**data)
