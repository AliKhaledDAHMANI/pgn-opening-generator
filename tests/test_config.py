"""Tests for configuration handling."""

from __future__ import annotations

import os

import pytest

from pgn_generator.config import Config, EngineConfig, build_config
from pgn_generator.errors import ConfigError


def test_defaults_are_sane() -> None:
    config = build_config({})
    assert config.mode == "gm"
    assert config.deterministic is True
    assert config.engine.depth >= 12
    assert config.output.comments is True
    assert config.target_plies == config.main_line_moves * 2


def test_mode_presets_apply_and_are_overridable() -> None:
    trap = build_config({}, mode="trap")
    assert trap.mode == "trap"
    assert trap.style == "sharp_tactical"
    assert trap.output.annotations == "rich"

    shortened = build_config({"main_line_moves": 5}, mode="trap")
    assert shortened.main_line_moves == 5
    assert shortened.output.annotations == "rich"   # preset survives


def test_engine_mode_disables_theory_preference() -> None:
    config = build_config({}, mode="engine")
    assert config.prefer_theory is False
    assert config.style == "engine_best"


def test_engine_best_style_forces_objective_selection() -> None:
    config = build_config({"style": "engine_best", "prefer_theory": True})
    assert config.prefer_theory is False


def test_deterministic_strips_time_limits_and_threads() -> None:
    config = build_config({"engine": {"time_ms": 500, "threads": 8}})
    assert config.deterministic is True
    assert config.engine.time_ms is None
    assert config.engine.threads == 1


def test_non_deterministic_keeps_time_limits() -> None:
    config = build_config({"deterministic": False, "engine": {"time_ms": 500, "threads": 4}})
    assert config.engine.time_ms == 500
    assert config.engine.threads == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "nonsense"},
        {"style": "nonsense"},
        {"aggressiveness": 2.0},
        {"main_line_moves": 0},
        {"variations": -1},
        {"engine": {"depth": 0}},
        {"engine": {"multipv": 0}},
        {"output": {"annotations": "loud"}},
        {"output": {"evals": "sometimes"}},
        {"unknown_key": 1},
        {"engine": {"unknown": 1}},
        {"output": {"unknown": 1}},
    ],
)
def test_invalid_config_is_rejected(overrides) -> None:
    with pytest.raises(ConfigError):
        build_config(overrides)


def test_required_engine_conflicts_with_disabled_engine() -> None:
    with pytest.raises(ConfigError):
        build_config({"engine": {"enabled": False, "required": True}})


def test_round_header_alias() -> None:
    config = build_config({"output": {"round": "3"}})
    assert config.output.round_ == "3"
    assert config.to_dict()["output"]["round"] == "3"


def test_config_round_trips_through_dict() -> None:
    config = build_config({"mode": "training", "engine": {"depth": 14}})
    clone = Config.from_dict(_strip_round(config.to_dict()))
    assert clone.to_dict() == config.to_dict()


def _strip_round(data: dict) -> dict:
    output = dict(data["output"])
    output["round_"] = output.pop("round")
    return {**data, "output": output}


def test_environment_overrides_are_applied(monkeypatch) -> None:
    monkeypatch.setenv("PGNGEN_ENGINE_DEPTH", "22")
    monkeypatch.setenv("PGNGEN_ENGINE_HASH", "64")
    config = build_config({})
    assert config.engine.depth == 22
    assert config.engine.hash_mb == 64
    assert config.engine.critical_depth >= 22


def test_disable_engine_environment_flag(monkeypatch) -> None:
    monkeypatch.setenv("PGNGEN_DISABLE_ENGINE", "1")
    config = build_config({})
    assert config.engine.enabled is False


def test_explicit_engine_path_beats_environment(monkeypatch) -> None:
    monkeypatch.setenv("PGNGEN_ENGINE_PATH", "/nonexistent/from-env")
    config = build_config({"engine": {"path": "/nonexistent/explicit"}})
    assert config.engine.path == "/nonexistent/explicit"


def test_resolved_path_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("PGNGEN_ENGINE_PATH", raising=False)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent-directory")
    engine = EngineConfig(path="/nonexistent/stockfish")
    assert engine.resolved_path() is None


def test_max_plies_caps_target_length() -> None:
    config = build_config({"main_line_moves": 12, "max_plies": 9})
    assert config.target_plies == 9
