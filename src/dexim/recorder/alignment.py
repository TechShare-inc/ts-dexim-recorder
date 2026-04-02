"""Temporal alignment for multi-stream episode buffers.

Aligns N buffered data streams to a single master clock, producing a
list of aligned frame dicts suitable for storage backends.

Master clock priority:
    1. Explicit ``master_clock_topic`` from config.
    2. First topic matching ``*/video_frame`` (camera streams have stable rates).
    3. Topic with the most data points.

Stream classification:
    Continuous (linear interpolation): topics containing ``/joint_state``
        OR listed in the caller-supplied ``continuous_topics``.
    Discrete (nearest-neighbour): all other topics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

__all__ = ["align_episode_data"]

# Sub-string patterns that qualify a topic as a continuous signal.
_CONTINUOUS_PATTERNS = ("/joint_state",)

# Sub-string that marks a topic as a candidate master clock (camera frame rate
# is typically the most stable in a multi-sensor setup).
_VIDEO_PATTERN = "/video_frame"


def align_episode_data(
    buffers: dict[str, list[tuple[float, Any]]],
    master_clock_topic: str | None = None,
    continuous_topics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Align multi-stream buffers to a single master clock.

    Args:
        buffers: Map from topic string to a sorted list of
            ``(timestamp, data)`` tuples.
        master_clock_topic: Explicit master clock topic name.
            Auto-detected when ``None``.
        continuous_topics: Topics that should use linear interpolation.
            Topics not listed here (and not matching ``/joint_state``) use
            nearest-neighbour alignment.

    Returns:
        List of dicts, one per master-clock frame.  Each dict maps topic
        string → aligned data value, plus a ``"timestamp"`` key (float).

    Raises:
        ValueError: When ``buffers`` is empty or the master clock stream
            has no samples.
    """
    if not buffers:
        raise ValueError("buffers is empty — no data to align")

    _continuous: set[str] = set(continuous_topics or [])

    master_topic = _select_master_clock(buffers, master_clock_topic)
    master_samples = buffers[master_topic]
    if not master_samples:
        raise ValueError(f"Master clock topic {master_topic!r} has no samples")

    master_times = np.array([ts for ts, _ in master_samples], dtype=np.float64)

    logger.debug(
        f"Alignment: master={master_topic!r}, "
        f"{len(master_times)} frames, {len(buffers)} topics"
    )

    aligned_frames: list[dict[str, Any]] = []
    for i, t in enumerate(master_times):
        frame: dict[str, Any] = {"timestamp": float(t)}

        for topic, samples in buffers.items():
            if topic == master_topic:
                frame[topic] = master_samples[i][1]
                continue

            if not samples:
                frame[topic] = None
                continue

            is_continuous = topic in _continuous or any(
                p in topic for p in _CONTINUOUS_PATTERNS
            )
            frame[topic] = (
                _interpolate(t, samples)
                if is_continuous
                else _nearest_neighbour(t, samples)
            )

        aligned_frames.append(frame)

    logger.debug(f"Alignment complete: {len(aligned_frames)} frames")
    return aligned_frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_master_clock(
    buffers: dict[str, list[tuple[float, Any]]],
    explicit: str | None,
) -> str:
    """Select the topic to use as the timing master.

    Priority: explicit config > ``/video_frame`` topic > most-samples topic.

    Args:
        buffers: Episode buffers.
        explicit: Caller-supplied topic name, or ``None`` for auto-detection.

    Returns:
        Selected master clock topic string.

    Raises:
        ValueError: When ``explicit`` is given but not present in ``buffers``.
    """
    if explicit is not None:
        if explicit not in buffers:
            raise ValueError(
                f"master_clock_topic {explicit!r} not found in buffers. "
                f"Available: {list(buffers)}"
            )
        return explicit

    # Prefer video/image topics as master clock (stable frame rate source)
    for topic in buffers:
        if _VIDEO_PATTERN in topic and buffers[topic]:
            logger.debug(f"Auto-selected video master clock: {topic!r}")
            return topic

    # Fall back to the stream with the most samples
    master = max(buffers, key=lambda t: len(buffers[t]))
    logger.debug(f"Auto-selected most-samples master clock: {master!r}")
    return master


def _nearest_neighbour(
    target: float,
    samples: list[tuple[float, Any]],
) -> Any:
    """Return the datum from the sample closest in time to ``target``.

    Args:
        target: Target timestamp.
        samples: List of ``(timestamp, data)`` pairs.

    Returns:
        Data value from the temporally closest sample.
    """
    times = np.array([ts for ts, _ in samples], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - target)))
    return samples[idx][1]


def _interpolate(
    target: float,
    samples: list[tuple[float, Any]],
) -> Any:
    """Linearly interpolate between the two samples bracketing ``target``.

    Boundary cases clamp to the first or last sample when ``target`` is
    outside the range of available data.  Non-numeric types fall back to
    nearest-neighbour.

    Args:
        target: Target timestamp.
        samples: List of ``(timestamp, data)`` pairs sorted by time.

    Returns:
        Interpolated (or boundary-clamped / nearest) data value.
    """
    times = np.array([ts for ts, _ in samples], dtype=np.float64)

    # Boundary clamping
    if target <= times[0]:
        return samples[0][1]
    if target >= times[-1]:
        return samples[-1][1]

    idx_after = int(np.searchsorted(times, target, side="right"))
    idx_before = idx_after - 1

    t_before, d_before = samples[idx_before]
    t_after, d_after = samples[idx_after]

    dt = t_after - t_before
    if dt == 0.0:
        return d_before

    alpha = (target - t_before) / dt

    try:
        d_b = np.asarray(d_before, dtype=np.float64)
        d_a = np.asarray(d_after, dtype=np.float64)
        result = (1.0 - alpha) * d_b + alpha * d_a
        # Return same container type as the input data
        if isinstance(d_before, list):
            return result.tolist()
        return result
    except (TypeError, ValueError):
        # Non-numeric: fall back to nearest-neighbour
        return d_before if alpha < 0.5 else d_after
