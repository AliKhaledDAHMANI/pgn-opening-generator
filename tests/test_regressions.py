"""Regression tests for issues found during development.

Each test here corresponds to a specific bug that reached working code once. They
are kept separate from the unit tests because they document *why* a rule exists,
not just that it holds.
"""

from __future__ import annotations

import chess
import pytest

from pgn_generator.book import expand_aliases, get_book
from pgn_generator.config import build_config
from pgn_generator.errors import OpeningNotFoundError
from pgn_generator.features import analyse_move
from pgn_generator.pgn import parse_pgn
from pgn_generator.request import RequestParser, parse_move_tokens

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Request parsing
# --------------------------------------------------------------------------- #


def test_fen_en_passant_field_is_not_a_move(book) -> None:
    """"... w KQkq c6 0 2" used to be parsed as the move c6, which is illegal."""
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    request = RequestParser(book).parse(f"best line from {fen}")
    assert request.start_moves == []
    assert request.start_fen == chess.Board(fen).fen()


def test_prose_after_a_move_list_stops_the_scan() -> None:
    assert parse_move_tokens("1.e4 e5 then something else") == ["e4", "e5"]


def test_bare_prose_yields_no_moves() -> None:
    """"be4" and friends must not be mistaken for moves."""
    assert parse_move_tokens("theory please") == []
    assert parse_move_tokens("something sharp") == []


def test_alias_expansion_is_not_recursive() -> None:
    """"Sicilian Najdorf" once expanded to "sicilian defense sicilian defense ..."."""
    expanded = expand_aliases("Sicilian Najdorf")
    assert expanded.count("sicilian defense") == 1


def test_generic_words_alone_do_not_resolve_an_opening(book) -> None:
    """"Frobnicator Attack" used to resolve to "Bongcloud Attack" on one word."""
    for query in ("Frobnicator Attack", "Blah Blah Defense", "gambit game", "the attack"):
        with pytest.raises(OpeningNotFoundError):
            book.resolve(query)


def test_kings_indian_without_a_qualifier_is_the_defense(book) -> None:
    """A trap "against the King's Indian" is not about the King's Indian Attack."""
    entry, _ = book.resolve("King's Indian")
    assert entry.name == "King's Indian Defense"


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def test_quiet_development_is_not_a_sacrifice() -> None:
    """SEE of a non-capture used to flag every developing move as a sacrifice."""
    board = chess.Board()
    for san in ("e4", "e5"):
        board.push_san(san)
    features = analyse_move(board, board.parse_san("Nf3"))
    assert not features.is_sacrifice


def test_even_recapture_is_not_a_sacrifice() -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "d5"):
        board.push_san(san)
    features = analyse_move(board, board.parse_san("exd5"))
    assert not features.is_sacrifice


def test_recapture_does_not_claim_a_pawn_pin() -> None:
    """Bxc6 was reported as pinning the b7 pawn to the a8 rook, which is noise."""
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        board.push_san(san)
    features = analyse_move(board, board.parse_san("Bxc6"))
    assert features.creates_pin == []


def test_pin_through_an_empty_square_is_found() -> None:
    """Only the square directly behind the target was checked, missing Bg5xNf6-Qd8."""
    board = chess.Board()
    for san in ("d4", "Nf6", "Nf3", "e6"):
        board.push_san(san)
    features = analyse_move(board, board.parse_san("Bg5"))
    assert (chess.F6, chess.D8) in features.creates_pin


def test_a_capturable_forker_is_not_a_fork() -> None:
    """Qg8+ "forks" f8 and h8 but is simply taken; it is a sacrifice."""
    board = chess.Board("5r1k/6pp/8/6N1/2Q5/8/6PP/7K w - - 0 1")
    features = analyse_move(board, board.parse_san("Qg8+"))
    assert features.forks == []
    assert features.is_sacrifice


# --------------------------------------------------------------------------- #
# PGN writing
# --------------------------------------------------------------------------- #


def test_suffix_does_not_swallow_the_following_token() -> None:
    """Appending "!" without restoring the separator glued tokens together."""
    from pgn_generator.annotate import MoveAnnotation
    from pgn_generator.pgn import LineData, MoveRecord, line_to_pgn

    board = chess.Board()
    records = []
    for san in ("e4", "e5", "Nf3", "Nc6"):
        move = board.parse_san(san)
        annotation = MoveAnnotation(san=san, nag="!", nag_reason="test") if san == "Nf3" else None
        records.append(MoveRecord(move=move, san=board.san(move), annotation=annotation))
        board.push(move)
    line = LineData(start_fen=chess.Board().fen(), from_initial=True, moves=records)
    pgn, _game = line_to_pgn(line, build_config({}))
    assert "Nf3! Nc6" in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []


def test_custom_position_keeps_its_fen_header_even_when_disabled() -> None:
    """``include_fen_headers: never`` must not make the movetext unreplayable."""
    from pgn_generator.pgn import LineData, MoveRecord, line_to_pgn

    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    move = board.parse_san("a6")
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=[MoveRecord(move=move, san="a6")],
    )
    config = build_config({"output": {"include_fen_headers": "never"}})
    pgn, _game = line_to_pgn(line, config)
    assert '[SetUp "1"]' in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    assert parsed.board().fen() == chess.Board(fen).fen()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@requires_engine
def test_repertoire_does_not_branch_on_a_pinned_reply(overrides) -> None:
    """"against 1...e5" asks about e5; branching to 1...c5 answers the wrong question."""
    from pgn_generator.generator import generate_pgn

    result = generate_pgn(
        "Build an opening repertoire for White against 1...e5.",
        overrides={**overrides, "mode": "repertoire", "repertoire_branches": 3},
    )
    assert result.line.moves[1].san == "e5"
    assert result.line.moves[1].variations == []


@requires_engine
def test_trap_bait_is_matched_by_position_not_by_move(overrides) -> None:
    """A repeated move elsewhere in the line must not attach the trap branch."""
    from pgn_generator.generator import generate_pgn

    result = generate_pgn(
        "Show me a trap in the Two Knights Defense.",
        overrides={**overrides, "mode": "trap", "main_line_moves": 8},
    )
    if not result.traps:
        pytest.skip("no trap found at this depth")
    trap = result.traps[0]
    board = chess.Board(result.line.start_fen)
    branch_ply = None
    for ply, record in enumerate(result.line.moves):
        if record.variations and any(v.kind == "trap" for v in record.variations):
            branch_ply = ply
            break
        board.push(record.move)
    if branch_ply is None:
        pytest.skip("the trap was played in the main line")
    # The position before the branch must be the one the trap was found in.
    parent = chess.Board(result.line.start_fen)
    for record in result.line.moves[: branch_ply - 1]:
        parent.push(record.move)
    assert parent.fen() == trap["origin_fen"]


@requires_engine
def test_book_labels_are_not_repeated_in_one_line(overrides) -> None:
    """Successive book entries share a family name; only the new part is printed."""
    from pgn_generator.generator import generate_pgn

    result = generate_pgn("Teach me the Italian Game", overrides={**overrides, "main_line_moves": 9})
    labels = []
    for record in result.line.moves:
        if not record.annotation:
            continue
        for part in record.annotation.comment_parts:
            if part.endswith(")") and "(" in part:
                labels.append(part)
    assert len(labels) == len(set(labels)), f"repeated labels: {labels}"


@requires_engine
def test_plans_are_stated_once_per_line(overrides) -> None:
    """"Black fights for the centre" appeared on every second move."""
    from pgn_generator.generator import generate_pgn

    result = generate_pgn(
        "Teach me the Ruy Lopez", overrides={**overrides, "main_line_moves": 10}
    )
    sentences = []
    for record in result.line.moves:
        if record.annotation:
            sentences.extend(record.annotation.comment_parts)
    plans = [s for s in sentences if "fights for the centre" in s or "tucks the king" in s]
    assert len(plans) == len(set(plans)), f"repeated plan sentences: {plans}"
