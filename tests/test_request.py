"""Tests for natural-language request parsing."""

from __future__ import annotations

import chess
import pytest

from pgn_generator.errors import IllegalMoveError, InvalidFENError, RequestError
from pgn_generator.request import RequestParser, extract_fen, parse_move_tokens


@pytest.fixture
def parser(book) -> RequestParser:
    return RequestParser(book)


# --------------------------------------------------------------------------- #
# Move-text parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.e4 c5 2.Nf3 d6 3.d4", ["e4", "c5", "Nf3", "d6", "d4"]),
        ("1. e4 e5 2. Nf3", ["e4", "e5", "Nf3"]),
        ("e4 c5 Nf3", ["e4", "c5", "Nf3"]),
        ("1.e4 e5 2.Nf3 Nc6 3.Bb5 a6!?", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6!?"]),
        ("1.e4 e5 2.Bc4 Nf6 3.Nf3 Nxe4 *", ["e4", "e5", "Bc4", "Nf6", "Nf3", "Nxe4"]),
        ("1.e4 e5 2.Nf3 Nc6 3.Bb5 O-O", ["e4", "e5", "Nf3", "Nc6", "Bb5", "O-O"]),
        ("1.e4 e5 2.Nf3 Nc6 3.Bb5 0-0", ["e4", "e5", "Nf3", "Nc6", "Bb5", "O-O"]),
        ("e2e4 c7c5", ["e2e4", "c7c5"]),
        ("1.e4 e5 then something else entirely", ["e4", "e5"]),
        # A FEN's trailing fields must not be read as moves.
        ("w KQkq c6 0 2", []),
        ("theory please", []),
    ],
)
def test_parse_move_tokens(text, expected) -> None:
    assert parse_move_tokens(text) == expected


def test_extract_fen_finds_and_completes_a_fen() -> None:
    text = "look at r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4 please"
    fen = extract_fen(text)
    assert fen is not None
    assert chess.Board(fen).turn == chess.BLACK


def test_extract_fen_completes_a_four_field_fen() -> None:
    fen = extract_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3")
    assert fen is not None
    board = chess.Board(fen)
    assert board.turn == chess.BLACK


def test_prose_is_not_mistaken_for_moves(parser: RequestParser) -> None:
    request = parser.parse("give me something sharp and instructive from theory")
    assert request.start_moves == []


# --------------------------------------------------------------------------- #
# Whole-request interpretation
# --------------------------------------------------------------------------- #


def test_main_line_request(parser: RequestParser) -> None:
    request = parser.parse("Show me the main line of the Italian Game.")
    assert request.opening_entry is not None
    assert request.opening_entry.eco == "C50"
    assert request.style == "theoretical"


def test_sharp_najdorf_request(parser: RequestParser) -> None:
    request = parser.parse("Create a sharp Sicilian Najdorf variation.")
    assert request.opening_entry is not None
    assert "Najdorf" in request.opening_entry.name
    assert request.style == "sharp_tactical"


def test_trap_request_sets_trap_mode(parser: RequestParser) -> None:
    request = parser.parse("Give me a GM-style trap against the King's Indian.")
    assert request.mode == "trap"
    assert request.opening_entry is not None
    assert "King's Indian Defense" in request.opening_entry.name


def test_repertoire_request(parser: RequestParser) -> None:
    request = parser.parse("Build an opening repertoire for White against 1...e5.")
    assert request.mode == "repertoire"
    assert request.side == "white"
    assert request.constraints == [(1, "e5")]


def test_explicit_move_count(parser: RequestParser) -> None:
    request = parser.parse("Create a 15-move opening variation with tactical annotations.")
    assert request.overrides["main_line_moves"] == 15
    assert request.style == "sharp_tactical"


def test_continues_from_supplied_moves(parser: RequestParser) -> None:
    request = parser.parse("Give me the best line after 1.e4 c5 2.Nf3 d6 3.d4")
    assert request.start_moves == ["e4", "c5", "Nf3", "d6", "d4"]
    assert request.opening_entry is not None
    assert request.opening_entry.family == "Sicilian Defense"


def test_rare_request_relaxes_theory(parser: RequestParser) -> None:
    request = parser.parse(
        "Aggressive and rare variation against white after 1.e4 e5 2.Nf3 Nc6 3.Bc4 d5"
    )
    assert request.style == "aggressive"
    assert request.overrides["prefer_theory"] is False
    assert request.overrides["engine"]["multipv"] >= 6
    assert request.start_moves == ["e4", "e5", "Nf3", "Nc6", "Bc4", "d5"]


def test_kingside_focus_is_recorded(parser: RequestParser) -> None:
    request = parser.parse("Create an aggressive Sicilian line where Black attacks the kingside.")
    assert "kingside" in request.focus
    assert request.opening_entry is not None
    assert request.opening_entry.family == "Sicilian Defense"


def test_fen_request_classifies_the_position(parser: RequestParser) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    request = parser.parse(f"Show the best response after this position: {fen}")
    assert request.start_fen is not None
    assert chess.Board(request.start_fen).fen() == chess.Board(fen).fen()
    assert request.opening_entry is not None
    assert request.opening_entry.family == "Ruy Lopez"


def test_fen_fields_are_not_read_as_moves(parser: RequestParser) -> None:
    """An en-passant field ("c6") must not become a move."""
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    request = parser.parse(f"best line from {fen}")
    assert request.start_fen == chess.Board(fen).fen()
    assert request.start_moves == []


def test_presentation_flags_from_prose(parser: RequestParser) -> None:
    request = parser.parse("Ruy Lopez main line, no comments, no arrows")
    output = request.overrides["output"]
    assert output["comments"] is False
    assert output["arrows"] is False


def test_explicit_values_beat_inference(parser: RequestParser) -> None:
    request = parser.parse(
        "sharp trap in the Italian",
        explicit={"mode": "training", "style": "solid", "side": "black"},
    )
    assert request.mode == "training"
    assert request.style == "solid"
    assert request.side == "black"


def test_unresolvable_opening_is_a_warning_not_an_error(parser: RequestParser) -> None:
    request = parser.parse("Generate a line where Black sacrifices a pawn for initiative.")
    assert request.opening_entry is None
    assert request.style == "gambit"


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_illegal_move_raises_with_context(parser: RequestParser) -> None:
    with pytest.raises(IllegalMoveError) as excinfo:
        parser.parse("line after 1.e4 e5 2.Nf3 Nf6 3.Bxf7")
    details = excinfo.value.details
    assert details["move"].startswith("Bxf7")
    assert details["ply"] == 4
    assert details["legal_sample"]
    assert details["accepted_prefix"] == ["e2e4", "e7e5", "g1f3", "g8f6"]


def test_lenient_mode_keeps_the_legal_prefix(parser: RequestParser) -> None:
    request = parser.parse("line after 1.e4 e5 2.Nf3 Nf6 3.Bxf7", strict_moves=False)
    assert request.start_moves == ["e4", "e5", "Nf3", "Nf6"]
    assert any("Bxf7" in warning for warning in request.warnings)


def test_invalid_fen_raises(parser: RequestParser) -> None:
    with pytest.raises(InvalidFENError):
        parser.parse("analyse this", explicit={"fen": "not-a-fen at all"})


def test_fen_without_kings_is_rejected(parser: RequestParser) -> None:
    with pytest.raises(InvalidFENError):
        parser.parse("analyse", explicit={"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})


def test_finished_position_is_rejected(parser: RequestParser) -> None:
    # Fool's mate: no continuation exists.
    with pytest.raises(InvalidFENError):
        parser.parse(
            "continue",
            explicit={"fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"},
        )


def test_unreadable_explicit_moves_raise(parser: RequestParser) -> None:
    with pytest.raises(RequestError):
        parser.parse("go", explicit={"moves": "not moves at all"})


def test_request_serialises(parser: RequestParser) -> None:
    request = parser.parse("Show me the main line of the Italian Game.")
    payload = request.to_dict()
    assert payload["opening"]["eco"] == "C50"
    assert isinstance(payload["inferences"], list)
