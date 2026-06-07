"""Unit tests for dexim.recorder.writer_thread.EpisodeWriterQueue."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

from dexim.recorder.metadata import EpisodeMetadata
from dexim.recorder.writer_thread import EpisodeWriterQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(index: int = 0, task_id: str = "test") -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        task_id=task_id,
        task_description="",
        start_time=0.0,
        end_time=1.0,
        num_frames=0,
        num_topics=1,
        topics=["topicA"],
    )


def _make_buffers() -> dict[str, list[tuple[float, Any]]]:
    return {"topicA": [(1.0, [1.0, 2.0]), (2.0, [3.0, 4.0])]}


# ---------------------------------------------------------------------------
# Submit and process
# ---------------------------------------------------------------------------


class TestSubmitAndProcess:
    def test_backend_write_episode_called(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        done = threading.Event()
        backend.write_episode.side_effect = lambda *a, **kw: done.set()

        with patch(
            "dexim.recorder.writer_thread.align_episode_data",
            return_value=[{"timestamp": 1.0}],
        ):
            q = EpisodeWriterQueue(backend=backend)
            q.submit(buffers=_make_buffers(), metadata=_make_metadata())
            done.wait(timeout=5.0)
            q.shutdown(timeout=5.0)

        backend.write_episode.assert_called_once()
        args = backend.write_episode.call_args
        assert args[0][1].episode_index == 0

    def test_alignment_called_with_buffers(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        done = threading.Event()
        backend.write_episode.side_effect = lambda *a, **kw: done.set()

        buffers = _make_buffers()
        with patch(
            "dexim.recorder.writer_thread.align_episode_data", return_value=[]
        ) as mock_align:
            q = EpisodeWriterQueue(
                backend=backend,
                master_clock_topic="topicA",
                continuous_topics=["topicA"],
            )
            q.submit(buffers=buffers, metadata=_make_metadata())
            done.wait(timeout=5.0)
            q.shutdown(timeout=5.0)

        mock_align.assert_called_once_with(
            buffers,
            master_clock_topic="topicA",
            continuous_topics=["topicA"],
        )


# ---------------------------------------------------------------------------
# FIFO ordering
# ---------------------------------------------------------------------------


class TestFifoOrdering:
    def test_three_episodes_processed_in_order(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        written_indices: list[int] = []
        finished = threading.Event()

        def _write(frames, metadata):
            written_indices.append(metadata.episode_index)
            if len(written_indices) == 3:
                finished.set()

        backend.write_episode.side_effect = _write

        with patch("dexim.recorder.writer_thread.align_episode_data", return_value=[]):
            q = EpisodeWriterQueue(backend=backend)
            for i in range(3):
                q.submit(buffers=_make_buffers(), metadata=_make_metadata(index=i))
            finished.wait(timeout=5.0)
            q.shutdown(timeout=5.0)

        assert written_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Shutdown drains queue
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_joins_thread(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        with patch("dexim.recorder.writer_thread.align_episode_data", return_value=[]):
            q = EpisodeWriterQueue(backend=backend)
            q.submit(buffers=_make_buffers(), metadata=_make_metadata())
            q.shutdown(timeout=5.0)

        assert not q._thread.is_alive()

    def test_shutdown_processes_pending_items(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        with patch("dexim.recorder.writer_thread.align_episode_data", return_value=[]):
            q = EpisodeWriterQueue(backend=backend)
            for i in range(5):
                q.submit(buffers=_make_buffers(), metadata=_make_metadata(index=i))
            q.shutdown(timeout=5.0)

        assert backend.write_episode.call_count == 5


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_failing_episode_does_not_block_next(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        call_count = 0
        finished = threading.Event()

        def _write(frames, metadata):
            nonlocal call_count
            call_count += 1
            if metadata.episode_index == 0:
                raise RuntimeError("disk full")
            if call_count == 2:
                finished.set()

        backend.write_episode.side_effect = _write

        with patch("dexim.recorder.writer_thread.align_episode_data", return_value=[]):
            q = EpisodeWriterQueue(backend=backend)
            q.submit(buffers=_make_buffers(), metadata=_make_metadata(index=0))
            q.submit(buffers=_make_buffers(), metadata=_make_metadata(index=1))
            finished.wait(timeout=5.0)
            q.shutdown(timeout=5.0)

        assert backend.write_episode.call_count == 2


# ---------------------------------------------------------------------------
# Qsize
# ---------------------------------------------------------------------------


class TestQsize:
    def test_qsize_reflects_pending_items(self) -> None:
        backend = MagicMock(spec=["write_episode", "close"])
        processing = threading.Event()
        unblock = threading.Event()

        def _slow_write(frames, metadata):
            processing.set()
            unblock.wait(timeout=5.0)

        backend.write_episode.side_effect = _slow_write

        with patch(
            "dexim.recorder.writer_thread.align_episode_data",
            return_value=[{"timestamp": 1.0}],
        ):
            q = EpisodeWriterQueue(backend=backend)
            # Submit several items; the first will block the worker
            for i in range(4):
                q.submit(buffers=_make_buffers(), metadata=_make_metadata(index=i))

            processing.wait(timeout=5.0)
            # 4 submitted, 1 being processed -> at most 3 remaining in queue
            assert q.qsize() <= 3
            unblock.set()
            q.shutdown(timeout=5.0)
