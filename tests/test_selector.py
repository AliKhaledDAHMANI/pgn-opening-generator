"""Tests for move selection and trap finding."""

from __future__ import annotations

import chess
import pytest

from pgn_generator.config import build_config
from pgn_generator.selector import (
    MoveSelector,
    TrapFinder,
    position_is_critical,
)
from pgn_generator.features import analyse_move

from .conftest import requires_engine


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


# --------------------------------------------------------------------------- #
# Criticality heuristic (no engine needed)
# --------------------------------------------------------------------------- #


def test_check_makes_a_position_critical() -> None:
    board = _board("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7+")
    assert board.is_check()
    assert position_is_critical(board)


def test_quiet_opening_position_is_not_critical() -> None:
    board = _board("e4", "e5")
    assert position_is_critical(board) is False


def test_a_sacrifice_makes_the_next_position_critical() -> None:
    board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Nxd5")
    features = analyse_move(board, board.parse_san("Nxf7"))
    after = board.copy(stack=False)
    after.push(board.parse_san("Nxf7"))
    assert position_is_critical(after, previous=features)


# --------------------------------------------------------------------------- #
# Selection without an engine
# --------------------------------------------------------------------------- #


def test_selection_without_engine_uses_theory_only(book) -> None:
    from pgn_generator.engine import EngineManager

    config = build_config({"engine": {"path": "/nonexistent/stockfish"}})
    with EngineManager(config.engine) as engine:
        selector = MoveSelector(config, book, engine)
        board = chess.Board()
        selection = selector.select(board)
        assert selection is not None
        assert selection.engine_available is False
        assert selection.chosen.theory is not None
        assert selection.chosen.move in board.legal_moves
        assert selection.chosen.cp_loss is None      # no invented numbers


def test_selection_returns_none_outside_book_without_engine(book) -> None:
    from pgn_generator.engine import EngineManager

    config = build_config({"engine": {"path": "/nonexistent/stockfish"}})
    with EngineManager(config.engine) as engine:
        selector = MoveSelector(config, book, engine)
        board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "d5")   # out of book
        assert selector.select(board) is None


# --------------------------------------------------------------------------- #
# Selection with an engine
# --------------------------------------------------------------------------- #


@requires_engine
def test_every_selected_move_is_legal(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    board = chess.Board()
    for _ in range(10):
        selection = selector.select(board)
        if selection is None:
            break
        assert selection.chosen.move in board.legal_moves
        board.push(selection.chosen.move)


@requires_engine
def test_engine_mode_takes_the_top_move(book, engine_config) -> None:
    config = build_config({"engine": dict(engine_config)}, mode="engine")
    with_engine = MoveSelector(config, book, engine := _manager(config))
    try:
        board = _board("e4", "e5", "Nf3")
        selection = with_engine.select(board, force_engine_best=True)
        assert selection is not None
        assert selection.chosen.engine_rank == 1
        assert selection.chosen.cp_loss == 0
    finally:
        engine.close()


def _manager(config):
    from pgn_generator.engine import EngineManager

    manager = EngineManager(config.engine, deterministic=config.deterministic)
    manager.start()
    return manager


@requires_engine
def test_theory_is_preferred_when_it_is_sound(config, book, engine) -> None:
    """In the Italian, the selector should stay in the book."""
    selector = MoveSelector(config, book, engine)
    board = _board("e4", "e5", "Nf3", "Nc6")
    selection = selector.select(board)
    assert selection is not None
    assert selection.chosen.theory is not None
    assert selection.chosen.san in {"Bc4", "Bb5", "d4", "Nc3", "Bc5", "Nf6"}


@requires_engine
def test_bad_moves_are_rejected_by_the_cp_filter(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")
    selection = selector.select(board)
    assert selection is not None
    # Nxe5 loses a piece; it must not be chosen, and if scored it must be rejected.
    blunder = board.parse_san("Nxe5")
    assert selection.chosen.move != blunder
    scored = selector.score_specific(board, blunder, critical=True)
    assert scored.cp_loss is not None and scored.cp_loss > 100


@requires_engine
def test_styles_produce_different_lines(book, engine_config) -> None:
    lines = {}
    for style in ("solid", "gambit"):
        config = build_config({"style": style, "engine": dict(engine_config)})
        manager = _manager(config)
        try:
            selector = MoveSelector(config, book, manager)
            board = chess.Board()
            sans = []
            for _ in range(8):
                selection = selector.select(board)
                if selection is None:
                    break
                sans.append(selection.chosen.san)
                board.push(selection.chosen.move)
            lines[style] = sans
        finally:
            manager.close()
    assert lines["solid"] != lines["gambit"], f"styles collapsed to the same line: {lines}"


@requires_engine
def test_prefer_moves_steers_the_choice(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    board = chess.Board()
    target = board.parse_san("d4")
    selection = selector.select(board, prefer_moves=[target])
    assert selection is not None
    assert selection.chosen.move == target
    assert any("requested" in reason for reason in selection.chosen.reasons)


@requires_engine
def test_excluded_moves_are_not_chosen(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    board = chess.Board()
    first = selector.select(board)
    assert first is not None
    second = selector.select(board, exclude=[first.chosen.move])
    assert second is not None
    assert second.chosen.move != first.chosen.move


@requires_engine
def test_alternatives_are_reported_with_reasons(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    board = _board("e4", "e5", "Nf3", "Nc6")
    selection = selector.select(board)
    assert selection is not None
    assert selection.alternatives
    assert selection.chosen.reasons
    payload = selection.chosen.to_dict()
    assert payload["san"] == selection.chosen.san
    assert "components" in payload


@requires_engine
def test_terminal_position_yields_no_selection(config, book, engine) -> None:
    selector = MoveSelector(config, book, engine)
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert selector.select(mate) is None


# --------------------------------------------------------------------------- #
# Traps
# --------------------------------------------------------------------------- #


@requires_engine
def test_trap_finder_returns_a_verified_trap(book, engine_config) -> None:
    """The Two Knights (4.d4 Nxe4?) is a real, engine-verifiable trap."""
    config = build_config({"engine": dict(engine_config)}, mode="trap")
    manager = _manager(config)
    try:
        finder = TrapFinder(config, book, manager)
        board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")
        trap = finder.find(board, chess.WHITE)
        assert trap is not None

        # Every component must be legal in its own position.
        assert trap.bait in board.legal_moves
        after_bait = board.copy(stack=False)
        after_bait.push(trap.bait)
        assert trap.victim in after_bait.legal_moves
        after_victim = after_bait.copy(stack=False)
        after_victim.push(trap.victim)
        assert trap.refutation in after_victim.legal_moves

        assert trap.cp_swing >= 120
        assert trap.temptation in ("book", "greedy", "shallow")
        assert board.fen() == trap.origin_fen
    finally:
        manager.close()


@requires_engine
def test_trap_finder_reports_nothing_in_a_dry_position(book, engine_config) -> None:
    """A quiet, well-known position should not produce a fabricated trap."""
    config = build_config({"engine": dict(engine_config)}, mode="trap")
    manager = _manager(config)
    try:
        finder = TrapFinder(config, book, manager, min_cp_swing=250)
        board = _board("d4", "d5", "c4", "e6", "Nc3", "Nf6")
        trap = finder.find(board, chess.WHITE)
        assert trap is None or trap.cp_swing >= 250
    finally:
        manager.close()


def test_trap_finder_needs_an_engine(book) -> None:
    from pgn_generator.engine import EngineManager

    config = build_config({"engine": {"path": "/nonexistent/stockfish"}}, mode="trap")
    with EngineManager(config.engine) as manager:
        finder = TrapFinder(config, book, manager)
        assert finder.find(chess.Board(), chess.WHITE) is None
        assert finder.find_all(chess.Board(), chess.WHITE) == []
