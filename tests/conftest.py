"""Shared test fixtures.

The engine-backed tests are skipped automatically when no Stockfish binary can be
found, so the suite still passes on a machine without one - which is also the
degradation path the library itself has to support.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn_generator.book import OpeningBook, get_book          # noqa: E402
from pgn_generator.config import EngineConfig, build_config   # noqa: E402
from pgn_generator.engine import EngineManager                # noqa: E402

#: Fast, deterministic engine settings. Deep enough to be meaningful, shallow
#: enough that the whole suite runs on a slow CPU.
TEST_ENGINE = {
    "depth": 8,
    "critical_depth": 10,
    "multipv": 3,
    "critical_multipv": 4,
    "hash_mb": 32,
}


def find_engine() -> Optional[str]:
    return EngineConfig().resolved_path()


ENGINE_PATH = find_engine()
requires_engine = pytest.mark.skipif(
    ENGINE_PATH is None,
    reason="no Stockfish binary found (set PGNGEN_ENGINE_PATH to enable engine tests)",
)


@pytest.fixture(scope="session")
def book() -> OpeningBook:
    return get_book()


@pytest.fixture(scope="session")
def engine_config() -> dict:
    config = dict(TEST_ENGINE)
    if ENGINE_PATH:
        config["path"] = ENGINE_PATH
    return config


@pytest.fixture
def config(engine_config):
    return build_config({"engine": engine_config, "main_line_moves": 6, "variations": 1})


@pytest.fixture
def engine(config) -> Iterator[EngineManager]:
    manager = EngineManager(config.engine, deterministic=config.deterministic)
    with manager:
        yield manager


@pytest.fixture
def overrides(engine_config) -> dict:
    """Config overrides for fast end-to-end generation."""
    return {"engine": dict(engine_config), "main_line_moves": 6, "variations": 1}
