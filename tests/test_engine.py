"""Tests for the engine wrapper, including its degradation behaviour."""

from __future__ import annotations

import chess
import pytest

from pgn_generator.config import EngineConfig, build_config
from pgn_generator.engine import MATE_SCORE_CP, EngineManager, Score
from pgn_generator.errors import EngineUnavailableError

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Score arithmetic (no engine needed)
# --------------------------------------------------------------------------- #


def test_score_perspective_flips_for_black() -> None:
    score = Score(cp=120)
    assert score.cp_for(chess.WHITE) == 120
    assert score.cp_for(chess.BLACK) == -120
    assert score.for_side(chess.BLACK).cp == -120


def test_mate_scores_map_to_large_centipawns() -> None:
    white_mates = Score(cp=MATE_SCORE_CP - 3, mate=3)
    assert white_mates.is_mate
    assert white_mates.cp > 9000
    black_mates = white_mates.for_side(chess.BLACK)
    assert black_mates.cp < -9000


# --------------------------------------------------------------------------- #
# Degradation without an engine
# --------------------------------------------------------------------------- #


def test_missing_engine_degrades_with_a_warning() -> None:
    config = build_config({"engine": {"path": "/nonexistent/stockfish"}})
    with EngineManager(config.engine) as engine:
        assert engine.available is False
        assert engine.analyse(chess.Board()) is None
        assert engine.evaluate(chess.Board()) is None
        assert any("ENGINE VALIDATION UNAVAILABLE" in w for w in engine.warnings)
        assert engine.info()["available"] is False


def test_missing_engine_raises_when_required() -> None:
    config = build_config({"engine": {"path": "/nonexistent/stockfish", "required": True}})
    with pytest.raises(EngineUnavailableError):
        with EngineManager(config.engine):
            pass


def test_disabled_engine_reports_itself() -> None:
    config = build_config({"engine": {"enabled": False}})
    with EngineManager(config.engine) as engine:
        assert engine.available is False
        assert any("disabled" in w.lower() for w in engine.warnings)


def test_terminal_positions_need_no_engine() -> None:
    config = build_config({"engine": {"path": "/nonexistent/stockfish"}})
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    with EngineManager(config.engine) as engine:
        analysis = engine.analyse(mate)
        assert analysis is not None
        assert analysis.candidates == []
        score = engine.evaluate(mate)
        assert score is not None and score.cp < -9000   # White is mated


# --------------------------------------------------------------------------- #
# Real engine
# --------------------------------------------------------------------------- #


@requires_engine
def test_engine_starts_and_reports_identity(engine: EngineManager) -> None:
    assert engine.available
    assert engine.name and "Stockfish" in engine.name
    info = engine.info()
    assert info["available"] is True
    assert info["threads"] == 1          # deterministic mode
    assert info["deterministic"] is True


@requires_engine
def test_multipv_returns_distinct_ranked_moves(engine: EngineManager) -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    analysis = engine.analyse(board)
    assert analysis is not None
    assert len(analysis.candidates) >= 2
    assert [c.rank for c in analysis.candidates] == sorted(c.rank for c in analysis.candidates)
    moves = [c.move for c in analysis.candidates]
    assert len(moves) == len(set(moves))
    for candidate in analysis.candidates:
        assert candidate.move in board.legal_moves


@requires_engine
def test_cp_loss_is_zero_for_the_best_move(engine: EngineManager) -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    analysis = engine.analyse(board)
    assert analysis is not None
    best = analysis.best
    assert best is not None
    assert analysis.cp_loss(best.move) == 0


@requires_engine
def test_cp_loss_is_positive_for_a_bad_move(engine: EngineManager) -> None:
    """Losing a piece must score clearly worse than the best move."""
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"):
        board.push_san(san)
    analysis = engine.analyse(board, multipv=1)
    assert analysis is not None
    best_score = analysis.best_score
    blunder = engine.score_move(board, board.parse_san("Nxe5"), critical=False)
    assert best_score is not None and blunder is not None
    assert best_score.cp_for(chess.WHITE) - blunder.cp_for(chess.WHITE) > 100


@requires_engine
def test_analysis_is_cached(engine: EngineManager) -> None:
    board = chess.Board()
    board.push_san("e4")
    first = engine.analyse(board)
    hits_before = engine.cache_hits
    second = engine.analyse(board)
    assert first is second
    assert engine.cache_hits == hits_before + 1


@requires_engine
def test_root_moves_restrict_the_search(engine: EngineManager) -> None:
    board = chess.Board()
    move = board.parse_san("a3")
    analysis = engine.analyse(board, multipv=1, root_moves=[move])
    assert analysis is not None
    assert analysis.candidates[0].move == move


@requires_engine
def test_depth_override_is_honoured(engine: EngineManager) -> None:
    board = chess.Board()
    analysis = engine.analyse(board, depth=4, multipv=1)
    assert analysis is not None
    assert analysis.candidates
    assert analysis.candidates[0].depth is not None
    assert analysis.candidates[0].depth <= 6


@requires_engine
def test_mate_is_reported_as_a_mate_score(engine: EngineManager) -> None:
    """Back-rank mate in one: Ra8#. The engine must report mate, not centipawns."""
    board = chess.Board("7k/5ppp/8/8/8/8/5PPP/R6K w - - 0 1")
    assert board.is_checkmate() is False
    analysis = engine.analyse(board, critical=True, multipv=1)
    assert analysis is not None
    best = analysis.best
    assert best is not None
    assert board.san(best.move) == "Ra8#"
    assert best.score.is_mate
    assert best.score.mate == 1
    assert best.score.cp > 9000


@requires_engine
def test_score_move_returns_none_for_illegal_moves(engine: EngineManager) -> None:
    board = chess.Board()
    illegal = chess.Move.from_uci("e2e5")
    assert engine.score_move(board, illegal) is None


@requires_engine
def test_deterministic_runs_agree(engine_config) -> None:
    """Same settings, same machine, same answer - twice."""
    board = chess.Board()
    for san in ("d4", "d5", "c4"):
        board.push_san(san)
    results = []
    for _ in range(2):
        config = build_config({"engine": dict(engine_config)})
        with EngineManager(config.engine, deterministic=True) as engine:
            analysis = engine.analyse(board)
            assert analysis is not None
            results.append([(c.move.uci(), c.score.cp) for c in analysis.candidates])
    assert results[0] == results[1]
