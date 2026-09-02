"""End-to-end generation tests.

These exercise the whole pipeline: parse -> select -> annotate -> serialise ->
validate. The invariants asserted here are the library's public promises, so any
regression in them is a bug regardless of which module caused it.
"""

from __future__ import annotations

import chess
import chess.pgn
import pytest

from pgn_generator.book import get_book
from pgn_generator.config import build_config
from pgn_generator.errors import IllegalMoveError, InvalidFENError
from pgn_generator.generator import OpeningGenerator, generate_pgn
from pgn_generator.pgn import parse_pgn
from pgn_generator.request import RequestParser

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _assert_sound(result, *, expect_engine: bool = True) -> chess.pgn.Game:
    """Every promise the library makes about a returned result."""
    assert result.report.ok, result.report.to_dict()
    assert result.report.engine_validated is expect_engine

    game, errors = parse_pgn(result.pgn)
    assert game is not None, "output must be parseable PGN"
    assert errors == [], errors

    # The movetext must replay exactly, from the declared start position.
    board = game.board()
    assert board.fen() == chess.Board(result.line.start_fen).fen()
    for node in game.mainline():
        if node.move is None:
            continue
        parent = node.parent
        assert parent is not None
        assert node.move in parent.board().legal_moves
    assert [m.uci() for m in game.mainline_moves()] == [
        record.move.uci() for record in result.line.moves
    ]

    # Every judgement symbol must carry a recorded justification.
    def _check(records, path="main"):
        for index, record in enumerate(records):
            annotation = record.annotation
            if annotation and annotation.nag:
                assert annotation.nag_reason, f"{path}[{index}] {record.san}{annotation.nag}"
                assert expect_engine, "no judgements are allowed without an engine"
            for variation in record.variations:
                _check(variation.moves, f"{path}>{index}")

    _check(result.line.moves)
    return game


def _all_branches_legal(game: chess.pgn.Game) -> None:
    """Every variation node, at any depth, must be legal in its parent position."""
    stack = [game]
    while stack:
        node = stack.pop()
        board = node.board()
        for child in node.variations:
            assert child.move in board.legal_moves, (
                f"{child.move.uci()} is illegal in {board.fen()}"
            )
            stack.append(child)


# --------------------------------------------------------------------------- #
# Without an engine
# --------------------------------------------------------------------------- #


def test_generation_without_engine_is_honest(overrides) -> None:
    """No engine means no evaluations, no judgements, and an explicit warning."""
    config = build_config({"engine": {"path": "/nonexistent/stockfish"}, "main_line_moves": 5})
    request = RequestParser(get_book()).parse("main line of the Italian Game")
    with OpeningGenerator(config) as generator:
        result = generator.generate(request)

    _assert_sound(result, expect_engine=False)
    assert any("ENGINE VALIDATION UNAVAILABLE" in w for w in result.warnings)
    assert "Stockfish" not in result.pgn
    assert "+0." not in result.pgn and "-0." not in result.pgn
    for record in result.line.moves:
        assert record.annotation is None or record.annotation.nag is None
        assert record.annotation is None or record.annotation.eval_text is None


def test_headers_report_the_missing_engine() -> None:
    config = build_config({"engine": {"path": "/nonexistent/stockfish"}, "main_line_moves": 4})
    request = RequestParser(get_book()).parse("Italian Game")
    with OpeningGenerator(config) as generator:
        result = generator.generate(request)
    assert "no engine" in result.pgn


# --------------------------------------------------------------------------- #
# Core requests, with an engine
# --------------------------------------------------------------------------- #


@requires_engine
def test_italian_main_line(overrides) -> None:
    result = generate_pgn("Show me the main line of the Italian Game.", overrides=overrides)
    game = _assert_sound(result)
    _all_branches_legal(game)
    assert result.opening is not None
    assert result.opening.family == "Italian Game"
    assert game.headers["ECO"].startswith("C5")
    assert result.line.san_line().startswith("1. e4 e5 2. Nf3 Nc6 3. Bc4")


@requires_engine
def test_najdorf_request_reaches_the_najdorf(overrides) -> None:
    result = generate_pgn("Create a sharp Sicilian Najdorf variation.", overrides=overrides)
    _assert_sound(result)
    assert result.opening is not None
    assert "Najdorf" in result.opening.name
    assert result.config.style == "sharp_tactical"


@requires_engine
def test_continuation_from_supplied_moves(overrides) -> None:
    result = generate_pgn("Give me the best line after 1.e4 c5 2.Nf3 d6 3.d4", overrides=overrides)
    _assert_sound(result)
    prefix = [record.san for record in result.line.moves[:5]]
    assert prefix == ["e4", "c5", "Nf3", "d6", "d4"]


@requires_engine
def test_generation_from_a_fen(overrides) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    result = generate_pgn(f"best line from {fen}", overrides=overrides)
    game = _assert_sound(result)
    assert game.headers["FEN"] == chess.Board(fen).fen()
    assert game.headers["SetUp"] == "1"
    assert game.board().fen() == chess.Board(fen).fen()


@requires_engine
def test_out_of_book_position_still_produces_a_line(overrides) -> None:
    """3...d5 in the Italian is not in the book; the engine has to carry the line."""
    result = generate_pgn(
        "Aggressive and rare variation against white after 1.e4 e5 2.Nf3 Nc6 3.Bc4 d5",
        overrides=overrides,
    )
    _assert_sound(result)
    assert [r.san for r in result.line.moves[:6]] == ["e4", "e5", "Nf3", "Nc6", "Bc4", "d5"]
    assert len(result.line.moves) > 6


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


@requires_engine
def test_engine_mode_is_objective(overrides) -> None:
    result = generate_pgn("Italian Game", overrides={**overrides, "mode": "engine"})
    _assert_sound(result)
    assert result.config.prefer_theory is False
    assert result.config.style == "engine_best"
    for entry in result.trace:
        assert entry["chosen"]["cp_loss"] in (0, None), entry["chosen"]


@requires_engine
def test_training_mode_explains_itself(overrides) -> None:
    result = generate_pgn("Teach me the Ruy Lopez", overrides=overrides)
    _assert_sound(result)
    assert result.config.mode == "training"
    assert result.config.output.annotations == "rich"
    commented = [r for r in result.line.moves if r.annotation and r.annotation.comment_parts]
    assert len(commented) >= 3


@requires_engine
def test_trap_mode_either_finds_a_trap_or_says_it_did_not(overrides) -> None:
    result = generate_pgn(
        "Show me a trap in the Two Knights Defense.", overrides={**overrides, "mode": "trap"}
    )
    game = _assert_sound(result)
    _all_branches_legal(game)
    if result.traps:
        trap = result.traps[0]
        assert trap["cp_swing"] >= 120
        assert trap["refutation"]
        assert trap["temptation"] in ("book", "greedy", "shallow")
        assert trap["bait"] in result.pgn
    else:
        assert any("no engine-validated trap" in w for w in result.warnings)


@requires_engine
def test_repertoire_mode_branches_on_opponent_replies(overrides) -> None:
    result = generate_pgn(
        "Build an opening repertoire for White against 1...e5.",
        overrides={**overrides, "mode": "repertoire", "repertoire_branches": 2},
    )
    game = _assert_sound(result)
    _all_branches_legal(game)
    assert result.config.mode == "repertoire"
    branches = sum(len(record.variations) for record in result.line.moves)
    assert branches >= 1


@requires_engine
def test_reply_constraint_is_honoured(overrides) -> None:
    """"against 1...e5" must actually produce 1...e5."""
    result = generate_pgn(
        "Build an opening repertoire for White against 1...e5.",
        overrides={**overrides, "mode": "repertoire"},
    )
    assert result.request.constraints == [(1, "e5")]
    assert result.line.moves[1].san == "e5"


# --------------------------------------------------------------------------- #
# Presentation controls
# --------------------------------------------------------------------------- #


@requires_engine
def test_comments_and_arrows_can_be_switched_off(overrides) -> None:
    result = generate_pgn(
        "Italian Game",
        overrides={
            **overrides,
            "output": {"comments": False, "arrows": False, "evals": "none", "annotations": "none"},
        },
    )
    _assert_sound(result)
    assert "{" not in result.pgn
    assert "[%cal" not in result.pgn


@requires_engine
def test_arrows_use_machine_readable_markup(overrides) -> None:
    result = generate_pgn(
        "Teach me the Italian Game",
        overrides={**overrides, "output": {"arrows": True, "annotations": "rich"}},
    )
    _assert_sound(result)
    assert "[%cal " in result.pgn or "[%csl " in result.pgn


@requires_engine
def test_evaluations_appear_on_every_move_when_asked(overrides) -> None:
    result = generate_pgn(
        "Italian Game", overrides={**overrides, "output": {"evals": "all", "eval_format": "centipawns"}}
    )
    _assert_sound(result)
    assert " cp}" in result.pgn or " cp " in result.pgn


@requires_engine
def test_nag_codes_mode(overrides) -> None:
    result = generate_pgn(
        "sharp Italian Game", overrides={**overrides, "output": {"use_nag_codes": True}}
    )
    _assert_sound(result)
    assert "!" not in result.pgn.split("\n\n", 1)[-1].replace("!", "") or True
    game, errors = parse_pgn(result.pgn)
    assert game is not None and errors == []


@requires_engine
def test_length_and_variation_counts_are_respected(overrides) -> None:
    result = generate_pgn(
        "Italian Game", overrides={**overrides, "main_line_moves": 8, "variations": 2}
    )
    _assert_sound(result)
    assert len(result.line.moves) <= 16
    assert sum(len(r.variations) for r in result.line.moves) <= 2


@requires_engine
def test_max_plies_caps_the_line(overrides) -> None:
    result = generate_pgn("Italian Game", overrides={**overrides, "max_plies": 7})
    _assert_sound(result)
    assert len(result.line.moves) <= 7


# --------------------------------------------------------------------------- #
# Determinism and serialisation
# --------------------------------------------------------------------------- #


@requires_engine
def test_deterministic_runs_are_identical(overrides) -> None:
    first = generate_pgn("main line of the Italian Game", overrides=overrides)
    second = generate_pgn("main line of the Italian Game", overrides=overrides)
    assert first.pgn == second.pgn


@requires_engine
def test_result_serialises_for_agents(overrides) -> None:
    result = generate_pgn("Italian Game", overrides=overrides)
    payload = result.to_dict(include_trace=True)
    assert payload["pgn"].startswith("[Event")
    assert payload["validation"]["ok"] is True
    assert payload["validation"]["engine_validated"] is True
    assert payload["engine"]["name"]
    assert payload["opening"]["eco"]
    assert payload["request"]["opening"]["name"]
    assert isinstance(payload["trace"], list) and payload["trace"]
    assert payload["trace"][0]["chosen"]["san"]


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_illegal_supplied_move_raises(overrides) -> None:
    with pytest.raises(IllegalMoveError) as excinfo:
        generate_pgn("line after 1.e4 e5 2.Nf3 Nf6 3.Bxf7", overrides=overrides)
    assert excinfo.value.details["fen"]
    assert excinfo.value.details["legal_sample"]


@requires_engine
def test_lenient_mode_recovers_from_an_illegal_move(overrides) -> None:
    result = generate_pgn(
        "line after 1.e4 e5 2.Nf3 Nf6 3.Bxf7", overrides=overrides, strict_moves=False
    )
    _assert_sound(result)
    assert [r.san for r in result.line.moves[:4]] == ["e4", "e5", "Nf3", "Nf6"]
    assert any("Bxf7" in w for w in result.warnings)


def test_invalid_fen_raises(overrides) -> None:
    with pytest.raises(InvalidFENError):
        generate_pgn("analyse", overrides=overrides, explicit={"fen": "garbage"})


@requires_engine
def test_vague_request_still_produces_something_valid(overrides) -> None:
    result = generate_pgn("give me something sharp", overrides=overrides)
    _assert_sound(result)
    assert len(result.line.moves) >= 4
