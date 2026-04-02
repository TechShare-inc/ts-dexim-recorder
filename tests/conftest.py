"""Shared test fixtures for dexim-recorder test suites.

Provides:
- ``propagate_loguru``: autouse fixture that routes loguru messages into
  pytest's ``caplog`` so ``caplog.messages`` assertions work with loguru.
"""

from __future__ import annotations

import logging

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def propagate_loguru(caplog):
    """Route loguru messages into pytest's caplog for assertion in tests."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
    )
    caplog.set_level(logging.DEBUG)
    yield
    logger.remove(handler_id)
