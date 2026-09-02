"""Tests for chess feature extraction.

These are the facts the annotator builds its claims on, so they are pinned
tightly: a wrong "sacrifice" or "fork" here becomes a wrong comment in the PGN.
"""

from __future__ import annotations

import chess
import pytest

from pgn_generator.features import (
    analyse_move,
    analyse_position,
    best_capture_see,
    material_balance,
    material_risk_of_quiet_move,
    static_exchange_eval,
)


def _line(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _features(board: chess.Board, san: str):
    return analyse_move(board, board.parse_san(san))


# --------------------------------------------------------------------------- #
# Static exchange evaluation
# --------------------------------------------------------------------------- #


def test_see_even_trade_is_zero() -> None:
    board = _line("e4", "e5", "Nf3", "d5")
    assert static_exchange_eval(board, board.parse_san("exd5")) == 0


def test_see_free_pawn_is_positive() -> None:
    board = _line("d4", "e5")   # e5 is undefended
    assert static_exchange_eval(board, board.parse_san("dxe5")) == 100


def test_see_losing_capture_is_negative() -> None:
    # Nxf7 in the Fried Liver gives up a knight for a pawn.
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Nxd5")
    assert static_exchange_eval(board, board.parse_san("Nxf7")) < -150


def test_best_capture_see_reports_the_profitable_capture() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5")
    # Black to move; taking on g5 with the queen loses material to Nxg5.
    assert best_capture_see(board, chess.G5) == 0
    board_free = _line("d4", "e5")
    assert best_capture_see(board_free, chess.E5) == 100


def test_quiet_move_risk_detects_a_dropped_piece() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")
    after = board.copy(stack=False)
    after.push_san("Nxe5")   # loses a knight to Nxe5
    assert material_risk_of_quiet_move(after, chess.E5) >= 200


def test_material_balance_counts_from_whites_view() -> None:
    assert material_balance(chess.Board()) == 0
    board = _line("d4", "e5", "dxe5")
    assert material_balance(board) == 100


# --------------------------------------------------------------------------- #
# Sacrifice detection
# --------------------------------------------------------------------------- #


def test_normal_development_is_not_a_sacrifice() -> None:
    board = _line("e4", "e5")
    features = _features(board, "Nf3")
    assert not features.is_sacrifice
    assert features.develops_piece
    assert features.risk_cp == 0


def test_even_recapture_is_not_a_sacrifice() -> None:
    board = _line("e4", "e5", "Nf3", "d5")
    features = _features(board, "exd5")
    assert not features.is_sacrifice


def test_dropping_a_knight_is_flagged() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6")
    features = _features(board, "Nxe5")   # Nxe5 loses a knight for a pawn
    assert features.is_sacrifice
    assert features.invested_cp >= 200


def test_fried_liver_knight_sacrifice_is_flagged() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Nxd5")
    features = _features(board, "Nxf7")
    assert features.is_sacrifice
    assert 150 < features.invested_cp < 350
    assert any(m.kind == "material_investment" for m in features.motifs)


def test_gambit_pawn_is_flagged_as_an_investment() -> None:
    board = _line("e4", "e5")
    features = _features(board, "f4")   # King's Gambit
    assert features.is_sacrifice
    assert features.invested_cp == 100


# --------------------------------------------------------------------------- #
# Tactical motifs
# --------------------------------------------------------------------------- #


def test_knight_fork_on_king_and_rook() -> None:
    board = chess.Board("r3k2r/8/8/3N4/8/8/8/4K3 w - - 0 1")
    features = _features(board, "Nc7+")
    assert features.gives_check
    assert set(features.forks) == {chess.A8, chess.E8}
    assert any(m.kind == "fork" for m in features.motifs)


def test_a_capturable_forking_piece_is_not_a_fork() -> None:
    """A "fork" that can simply be taken is a sacrifice, not a fork."""
    board = chess.Board("5r1k/6pp/8/6N1/2Q5/8/6PP/7K w - - 0 1")
    features = _features(board, "Qg8+")
    assert features.forks == []
    assert features.is_sacrifice


def test_checkmate_is_detected() -> None:
    """Fool's mate: 1.f3 e5 2.g4 Qh4#."""
    board = _line("f3", "e5", "g4")
    features = _features(board, "Qh4#")
    assert features.is_checkmate
    assert features.gives_check
    assert any(m.kind == "mate" for m in features.motifs)


def test_quiet_move_is_not_checkmate() -> None:
    board = _line("e4", "e5")
    features = _features(board, "Nf3")
    assert not features.is_checkmate
    assert not features.gives_check


def test_pin_against_the_queen_is_reported() -> None:
    """Bg5 pins the f6 knight to the d8 queen, through the empty e7 square."""
    board = _line("d4", "Nf6", "Nf3", "e6")
    features = _features(board, "Bg5")
    assert (chess.F6, chess.D8) in features.creates_pin
    assert any(m.kind == "pin" for m in features.motifs)


def test_recapture_does_not_claim_a_pawn_pin() -> None:
    """Bxc6 pins the b7 pawn to the a8 rook, which is not worth reporting."""
    board = _line("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    features = _features(board, "Bxc6")
    assert features.creates_pin == []


def test_castling_and_development_flags() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
    features = _features(board, "O-O")
    assert features.is_castling and features.castles_short
    assert any(m.kind == "castling" for m in features.motifs)


def test_forced_move_is_marked() -> None:
    board = chess.Board("k7/7R/1K6/8/8/8/8/8 b - - 0 1")
    assert board.legal_moves.count() == 1
    features = analyse_move(board, next(iter(board.legal_moves)))
    assert features.forced


def test_opening_a_file_for_the_rooks() -> None:
    board = _line("d4", "e5")
    features = _features(board, "dxe5")
    assert features.opens_file_for_rook


# --------------------------------------------------------------------------- #
# Position features
# --------------------------------------------------------------------------- #


def test_position_features_on_the_exchange_spanish() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6")
    position = analyse_position(board)
    # White traded bishop for knight: 330 vs 320 by the classical values used here.
    assert position.material == -10
    assert position.development["white"] == 1
    assert position.development["black"] == 0
    assert [chess.square_name(s) for s in position.structure["black"].doubled] == ["c6", "c7"]
    assert position.structure["black"].islands >= 2


def test_isolated_queens_pawn_is_detected() -> None:
    """White has a d-pawn with no c- or e-pawn: the classic IQP."""
    board = chess.Board("rnbqkb1r/pp3ppp/4pn2/8/3P4/2N5/PP3PPP/R1BQKBNR w KQkq - 0 6")
    position = analyse_position(board)
    assert position.iqp["white"] is True
    assert position.iqp["black"] is False


def test_castled_detection() -> None:
    board = _line("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O")
    position = analyse_position(board)
    assert position.castled["white"] is True
    assert position.castled["black"] is False
