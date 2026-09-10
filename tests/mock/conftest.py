"""Shared fixtures for mock (no-hardware) tests of dexim-recorder.

Strategy
--------
- ``ManagedNode.__init__`` ZMQ setup is patched so no ZMQ ports are bound.
- ``EpisodeWriterQueue`` is patched to a no-op spy to avoid spawning
  background threads during tests.
- ``_build_backend`` is patched to return a ``MagicMock``-based storage
  backend so no filesystem writes occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from dexim.core.nodes import ManagedNode

# ---------------------------------------------------------------------------
# ManagedNode I/O patch
# ---------------------------------------------------------------------------


@pytest.fixture()
def patch_managed_node():
    """Patch only ``ManagedNode`` ZMQ setup, preserving its runtime state.

    Keeping the real base initializer prevents this fixture from drifting when
    ManagedNode adds lifecycle state.
    """

    def _fake_initialize_zmq(self: ManagedNode) -> None:
        self._ctx = MagicMock()
        self._sub_control = MagicMock()
        self._push_status = MagicMock()
        self._poller = MagicMock()

    with patch.object(ManagedNode, "_initialize_zmq", _fake_initialize_zmq):
        yield


# ---------------------------------------------------------------------------
# EpisodeWriterQueue spy
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_writer_queue():
    """Return a MagicMock that records calls to submit() and shutdown()."""
    writer_queue = MagicMock(
        spec=["submit", "shutdown", "pending_count", "wait_until_idle"]
    )
    writer_queue.pending_count.return_value = 0
    writer_queue.wait_until_idle.return_value = True
    return writer_queue


@pytest.fixture()
def patch_writer_queue(mock_writer_queue):
    """Patch ``EpisodeWriterQueue`` in node.py to return ``mock_writer_queue``."""
    with patch(
        "dexim.recorder.node.EpisodeWriterQueue",
        return_value=mock_writer_queue,
    ):
        yield mock_writer_queue


# ---------------------------------------------------------------------------
# Storage backend mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_backend():
    """Return a MagicMock standing in for a StorageBackend."""
    return MagicMock(spec=["write_episode", "close"])


@pytest.fixture()
def patch_build_backend(mock_backend):
    """Patch ``_build_backend`` in node.py to return ``mock_backend``."""
    with patch(
        "dexim.recorder.node._build_backend",
        return_value=mock_backend,
    ):
        yield mock_backend
