"""Episode and task metadata dataclasses for dexim-recorder."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TaskInfo", "EpisodeMetadata"]


@dataclass
class TaskInfo:
    """Active task metadata, updated at runtime via the SET_TASK control command.

    Attributes:
        task_id: Short machine-readable identifier (e.g. ``"pick_red_cup"``).
        task_description: Human-readable natural language instruction
            (e.g. ``"Pick up the red cup and place it on the tray"``).
    """

    task_id: str = ""
    task_description: str = ""


@dataclass
class EpisodeMetadata:
    """Metadata snapshot for a single recorded episode.

    Created at STOP_REC time and passed to the storage backend alongside the
    aligned frame data.

    Attributes:
        episode_index: 0-based index within the dataset.
        task_id: Task active at START_REC time.
        task_description: Natural language instruction active at START_REC time.
        start_time: Wall-clock epoch (seconds) at START_REC.
        end_time: Wall-clock epoch (seconds) at STOP_REC.
        num_frames: Estimated frame count (0 when unknown before alignment).
        num_topics: Number of subscribed streams present in this episode.
        topics: Topic names present in the episode buffers.
    """

    episode_index: int
    task_id: str
    task_description: str
    start_time: float
    end_time: float
    num_frames: int
    num_topics: int
    topics: list[str] = field(default_factory=list)
