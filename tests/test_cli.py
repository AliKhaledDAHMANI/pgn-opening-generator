"""Tests for the command-line interface."""

from __future__ import annotations

import json
import os

import chess
import pytest

from pgn_generator.cli import build_parser, main
from pgn_generator.pgn import parse_pgn

from .conftest import ENGINE_PATH, requires_engine


def _engine_args() -> list:
    return ["--engine", ENGINE_PATH] if ENGINE_PATH else ["--no-engine"]


FAST = ["--depth", "8", "--critical-depth", "10", "--multipv", "3", "--moves-count", "5"]


def test_parser_accepts_the_documented_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "sharp Italian",
            "--mode", "trap",
            "--style", "gambit",
            "--side", "white",
            "--moves-count", "9",
            "--variations", "2",
            "--depth", "14",
            "--format", "json",
            "--no-arrows",
        ]
    )
    assert args.request == "sharp Italian"
    assert args.mode == "trap"
    assert args.style == "gambit"
    assert args.main_line_moves == 9
    assert args.depth == 14
    assert args.format == "json"
    assert args.arrows is False


def test_no_request_and_no_position_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_pgn_output_without_engine(capsys) -> None:
    code = main(["Italian Game", "--no-engine", "--moves-count", "4", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("[Event ")
    game, errors = parse_pgn(captured.out)
    assert game is not None and errors == []


def test_json_envelope_without_engine(capsys) -> None:
    code = main(["Italian Game", "--no-engine", "--moves-count", "4", "--format", "json", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["validation"]["engine_validated"] is False
    assert any("ENGINE VALIDATION UNAVAILABLE" in w for w in payload["warnings"])
    game, errors = parse_pgn(payload["pgn"])
    assert game is not None and errors == []


def test_illegal_move_reports_json_error(capsys) -> None:
    code = main(
        ["continue", "--moves", "1.e4 e5 2.Nf3 Nf6 3.Bxf7", "--no-engine", "--format", "json"]
    )
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "illegal_move"
    assert payload["error"]["details"]["legal_sample"]


def test_illegal_move_reports_plain_text_error(capsys) -> None:
    code = main(["continue", "--moves", "1.e4 e5 2.Nf3 Nf6 3.Bxf7", "--no-engine"])
    captured = capsys.readouterr()
    assert code == 1
    assert "illegal_move" in captured.err


def test_lenient_moves_recovers(capsys) -> None:
    code = main(
        [
            "continue",
            "--moves", "1.e4 e5 2.Nf3 Nf6 3.Bxf7",
            "--lenient-moves",
            "--no-engine",
            "--moves-count", "5",
            "--format", "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["main_line"].startswith("1. e4 e5 2. Nf3 Nf6")


def test_invalid_fen_reports_json_error(capsys) -> None:
    code = main(["analyse", "--fen", "garbage", "--no-engine", "--format", "json"])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "invalid_fen"


def test_require_engine_fails_loudly(capsys) -> None:
    code = main(
        [
            "Italian Game",
            "--engine", "/nonexistent/stockfish",
            "--require-engine",
            "--format", "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "engine_unavailable"


def test_bad_config_file_reports_an_error(tmp_path, capsys) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"mode": "nonsense"}', encoding="utf-8")
    code = main(["Italian Game", "--config", str(path), "--no-engine", "--format", "json"])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "invalid_config"


def test_config_file_is_applied(tmp_path, capsys) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"main_line_moves": 3, "output": {"event": "From Config"}}), encoding="utf-8"
    )
    code = main(["Italian Game", "--config", str(path), "--no-engine", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    assert '[Event "From Config"]' in captured.out


def test_flags_override_the_config_file(tmp_path, capsys) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output": {"event": "From Config"}}), encoding="utf-8")
    code = main(
        ["Italian Game", "--config", str(path), "--event", "From Flag", "--no-engine", "--quiet"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert '[Event "From Flag"]' in captured.out


def test_out_file_is_written(tmp_path, capsys) -> None:
    target = tmp_path / "line.pgn"
    code = main(
        ["Italian Game", "--no-engine", "--moves-count", "4", "--out", str(target), "--quiet"]
    )
    capsys.readouterr()
    assert code == 0
    text = target.read_text(encoding="utf-8")
    game, errors = parse_pgn(text)
    assert game is not None and errors == []


def test_headers_can_be_set(capsys) -> None:
    code = main(
        [
            "Italian Game",
            "--no-engine",
            "--moves-count", "4",
            "--event", "Test Event",
            "--white", "Alice",
            "--black", "Bob",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert '[Event "Test Event"]' in captured.out
    assert '[White "Alice"]' in captured.out
    assert '[Black "Bob"]' in captured.out


def test_fen_flag_sets_the_start_position(capsys) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    code = main(["best line", "--fen", fen, "--no-engine", "--moves-count", "6", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    game, errors = parse_pgn(captured.out)
    assert game is not None and errors == []
    assert game.board().fen() == chess.Board(fen).fen()


def test_threads_flag_disables_determinism(capsys) -> None:
    code = main(
        ["Italian Game", "--no-engine", "--moves-count", "3", "--threads", "2", "--format", "json"]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["engine"]["deterministic"] is False


@requires_engine
def test_engine_run_produces_validated_json(capsys) -> None:
    code = main(
        ["main line of the Italian Game", *_engine_args(), *FAST, "--format", "json", "--trace"]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["validation"]["ok"] is True
    assert payload["validation"]["engine_validated"] is True
    assert payload["opening"]["family"] == "Italian Game"
    assert payload["trace"]
    game, errors = parse_pgn(payload["pgn"])
    assert game is not None and errors == []


@requires_engine
def test_trap_mode_via_cli(capsys) -> None:
    code = main(
        [
            "trap in the Two Knights Defense",
            *_engine_args(),
            "--mode", "trap",
            "--depth", "8",
            "--critical-depth", "10",
            "--moves-count", "6",
            "--format", "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["mode"] == "trap"


@requires_engine
def test_module_entry_point_runs(capsys) -> None:
    """``python -m pgn_generator`` must reach the same code path."""
    import runpy
    import sys

    argv = sys.argv
    sys.argv = ["pgn-generator", "Italian Game", "--no-engine", "--moves-count", "3", "--quiet"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("pgn_generator", run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv = argv
    captured = capsys.readouterr()
    assert captured.out.startswith("[Event ")
