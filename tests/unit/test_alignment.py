"""Unit tests for dexim.recorder.alignment.align_episode_data."""

from __future__ import annotations

import pytest

from dexim.recorder.alignment import align_episode_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _buf(times, values=None):
    """Build a ``[(timestamp, data)]`` list for a single topic."""
    if values is None:
        values = list(range(len(times)))
    return list(zip(times, values))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_buffers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            align_episode_data({})

    def test_explicit_master_not_in_buffers_raises(self):
        buffers = {"obs/cam/video_frame": _buf([1.0, 2.0])}
        with pytest.raises(ValueError, match="master_clock_topic"):
            align_episode_data(buffers, master_clock_topic="missing/topic")

    def test_master_clock_with_no_samples_raises(self):
        buffers = {"topicA": []}
        with pytest.raises(ValueError, match="no samples"):
            align_episode_data(buffers, master_clock_topic="topicA")


# ---------------------------------------------------------------------------
# Master clock auto-selection
# ---------------------------------------------------------------------------


class TestMasterClockSelection:
    def test_explicit_master_clock_used(self):
        buffers = {
            "topicA": _buf([1.0, 2.0]),
            "topicB": _buf([1.0, 2.0, 3.0]),
        }
        frames = align_episode_data(buffers, master_clock_topic="topicA")
        assert len(frames) == 2
        assert frames[0]["timestamp"] == pytest.approx(1.0)
        assert frames[1]["timestamp"] == pytest.approx(2.0)

    def test_video_frame_preferred_over_most_samples(self):
        buffers = {
            "obs/cam/video_frame": _buf([1.0, 2.0, 3.0]),
            "obs/arm/joint_state": _buf([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
        }
        # video_frame has fewer samples but should still be master
        frames = align_episode_data(buffers)
        # Master is video_frame -> 3 frames
        assert len(frames) == 3
        assert frames[0]["timestamp"] == pytest.approx(1.0)

    def test_most_samples_fallback_when_no_video(self):
        buffers = {
            "topicA": _buf([1.0, 2.0]),
            "topicB": _buf([1.0, 1.5, 2.0, 2.5]),
        }
        # topicB has more samples -> selected as master
        frames = align_episode_data(buffers)
        assert len(frames) == 4


# ---------------------------------------------------------------------------
# Topic alignment: nearest-neighbour (discrete)
# ---------------------------------------------------------------------------


class TestNearestNeighbour:
    def test_nearest_before(self):
        master = _buf([1.0, 2.0, 3.0])
        discrete = _buf([0.9, 1.6, 2.7])  # offsets: 0.1, 0.4, 0.3
        frames = align_episode_data(
            {"master": master, "discrete": discrete},
            master_clock_topic="master",
        )
        # At t=1.0: closest is 0.9 -> index 0
        assert frames[0]["discrete"] == 0
        # At t=2.0: closest is 1.6 (dist 0.4) vs 2.7 (dist 0.7) -> index 1
        assert frames[1]["discrete"] == 1
        # At t=3.0: closest is 2.7 (dist 0.3) vs 1.6 (dist 1.4) -> index 2
        assert frames[2]["discrete"] == 2

    def test_empty_discrete_topic_yields_none(self):
        buffers = {
            "master": _buf([1.0, 2.0]),
            "empty": [],
        }
        frames = align_episode_data(buffers, master_clock_topic="master")
        assert frames[0]["empty"] is None
        assert frames[1]["empty"] is None


# ---------------------------------------------------------------------------
# Topic alignment: linear interpolation (continuous)
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
    def test_joint_state_interpolated_by_pattern(self):
        master = _buf([0.0, 1.0, 2.0])
        joint = _buf([0.0, 2.0], values=[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        buffers = {"master": master, "obs/arm/joint_state": joint}
        frames = align_episode_data(buffers, master_clock_topic="master")
        # t=0 -> exactly first sample
        assert frames[0]["obs/arm/joint_state"] == pytest.approx([0.0, 0.0, 0.0])
        # t=1 -> midpoint between [0,0,0] and [2,2,2]
        assert frames[1]["obs/arm/joint_state"] == pytest.approx([1.0, 1.0, 1.0])
        # t=2 -> exactly last sample
        assert frames[2]["obs/arm/joint_state"] == pytest.approx([2.0, 2.0, 2.0])

    def test_explicit_continuous_topic_interpolated(self):
        master = _buf([0.0, 0.5, 1.0])
        pressure = _buf([0.0, 1.0], values=[0.0, 10.0])
        buffers = {"master": master, "sensor/pressure": pressure}
        frames = align_episode_data(
            buffers,
            master_clock_topic="master",
            continuous_topics=["sensor/pressure"],
        )
        assert frames[0]["sensor/pressure"] == pytest.approx(0.0)
        assert frames[1]["sensor/pressure"] == pytest.approx(5.0)
        assert frames[2]["sensor/pressure"] == pytest.approx(10.0)

    def test_interpolation_clamps_below_range(self):
        master = _buf([0.0])
        joint = _buf([1.0, 2.0], values=[100.0, 200.0])
        buffers = {"master": master, "obs/arm/joint_state": joint}
        frames = align_episode_data(buffers, master_clock_topic="master")
        # t=0 is before joint timestamps [1.0, 2.0] -> clamped to first
        assert frames[0]["obs/arm/joint_state"] == pytest.approx(100.0)

    def test_interpolation_clamps_above_range(self):
        master = _buf([5.0])
        joint = _buf([1.0, 2.0], values=[100.0, 200.0])
        buffers = {"master": master, "obs/arm/joint_state": joint}
        frames = align_episode_data(buffers, master_clock_topic="master")
        # t=5 is after joint timestamps -> clamped to last
        assert frames[0]["obs/arm/joint_state"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Frame dict structure
# ---------------------------------------------------------------------------


class TestFrameStructure:
    def test_timestamp_in_every_frame(self):
        buffers = {
            "obs/cam/video_frame": _buf([1.0, 2.0, 3.0]),
            "obs/arm/joint_state": _buf([1.0, 2.0, 3.0]),
        }
        frames = align_episode_data(buffers)
        for frame in frames:
            assert "timestamp" in frame

    def test_all_topics_present_in_every_frame(self):
        buffers = {
            "topicA": _buf([1.0, 2.0]),
            "topicB": _buf([0.9, 2.1]),
        }
        frames = align_episode_data(buffers, master_clock_topic="topicA")
        for frame in frames:
            assert "topicA" in frame
            assert "topicB" in frame

    def test_returns_list_of_dicts(self):
        buffers = {"topicA": _buf([1.0])}
        result = align_episode_data(buffers)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
