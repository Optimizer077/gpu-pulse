"""Shared pytest setup."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _event_loop_for_bare_widgets() -> Iterator[None]:
    """Give tests that build Textual widgets outside an app a current event loop on Python 3.9.

    On 3.9, creating a widget calls ``asyncio.get_event_loop()``, which raises once an earlier
    ``app.run_test()`` has closed the loop and cleared it. The dashboard itself always creates
    its widgets inside its own running loop, so this only affects unit tests. Newer Pythons
    don't need it, so nothing is done there.
    """
    if sys.version_info >= (3, 10):
        yield
        return
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        return
    yield
