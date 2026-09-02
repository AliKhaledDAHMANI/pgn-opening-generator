"""Tests for the validation pipeline.

Each check is exercised against a deliberately broken line, because a validator
that cannot fail is not a validator.
"""

from __future__ import annotations

import chess
import pytest

from pgn_generator.annotate import MoveAnnotation
from pgn_generator.config import build_config
from pgn_generator.pgn import LineData, MoveRecord, VariationRecord, line_to_pgn
from pgn_generator.validate import validate


def _line(sans=("e4", "e5", "Nf3", "Nc6", "Bc4"), **annotations) -> LineData:
    board = chess.Board()
    records = []
    for san in sans:
        move = board.parse_san(san)
        records.append(
            MoveRecord(move=move, san=board.san(move), annotation=annotations.get(san))
        )
        board.push(move)
    return LineData(
        start_fen=chess.Board().fen(),
        from_initial=True,
        moves=records,
        headers={"ECO": "C50", "Opening": "Italian Game"},
    )


def _validate(line: LineData, *, book, config=None, **kwargs):
    """Serialise ``line`` and run the pipeline over it."""
    config = config or build_config({})
    pgn, _game = line_to_pgn(line, config)
    arguments = dict(
        line=line,
        pgn_text=pgn,
        config=config,
        book=book,
        engine_available=True,
        analysed_positions=len(line.moves),
    )
    arguments.update(kwargs)
    return validate(**arguments)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_clean_line_passes_every_check(book) -> None:
    report = _validate(_line(), book=book)
    assert report.ok
    assert report.engine_validated is True
    for check in ("legality", "san", "pgn", "engine", "annotations", "fen", "opening"):
        assert check in report.checks_run


def test_report_serialises(book) -> None:
    payload = _validate(_line(), book=book).to_dict()
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert isinstance(payload["checks_run"], list)


# --------------------------------------------------------------------------- #
# Legality and SAN
# --------------------------------------------------------------------------- #


def test_illegal_move_is_caught(book) -> None:
    line = _line()
    # Replace 3.Bc4 with a move that is illegal in that position.
    line.moves[4] = MoveRecord(move=chess.Move.from_uci("a1a5"), san="Ra5")
    report = _validate(line, book=book)
    assert not report.ok
    assert any(f.check == "legality" for f in report.errors)


def test_wrong_san_is_caught(book) -> None:
    line = _line()
    line.moves[0] = MoveRecord(move=line.moves[0].move, san="e5")   # e2e4 is not "e5"
    report = _validate(line, book=book)
    assert not report.ok
    assert any(f.check == "san" for f in report.errors)


# --------------------------------------------------------------------------- #
# Variations
# --------------------------------------------------------------------------- #


def test_variation_with_a_wrong_start_fen_is_caught(book) -> None:
    line = _line()
    branch_board = chess.Board()
    for record in line.moves[:4]:
        branch_board.push(record.move)
    move = branch_board.parse_san("Bb5")
    line.moves[4].variations.append(
        VariationRecord(
            start_ply=4,
            start_fen=chess.Board().fen(),          # wrong: claims the initial position
            moves=[MoveRecord(move=move, san="Bb5")],
            purpose="Ruy Lopez",
        )
    )
    report = _validate(line, book=book)
    assert not report.ok
    assert any(f.check == "variations" for f in report.errors)


def test_variation_with_an_illegal_move_is_caught(book) -> None:
    line = _line()
    branch_board = chess.Board()
    for record in line.moves[:4]:
        branch_board.push(record.move)
    line.moves[4].variations.append(
        VariationRecord(
            start_ply=4,
            start_fen=branch_board.fen(),
            moves=[MoveRecord(move=chess.Move.from_uci("h8h1"), san="Rh1")],
            purpose="nonsense",
        )
    )
    report = _validate(line, book=book)
    assert not report.ok
    assert any(f.check in ("legality", "variations") for f in report.errors)


def test_legal_variation_passes(book) -> None:
    line = _line()
    branch_board = chess.Board()
    for record in line.moves[:4]:
        branch_board.push(record.move)
    move = branch_board.parse_san("Bb5")
    line.moves[4].variations.append(
        VariationRecord(
            start_ply=4,
            start_fen=branch_board.fen(),
            moves=[MoveRecord(move=move, san="Bb5")],
            purpose="Ruy Lopez",
        )
    )
    report = _validate(line, book=book)
    assert report.ok


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


def test_unjustified_nag_is_rejected(book) -> None:
    line = _line(Bc4=MoveAnnotation(san="Bc4", nag="!!"))   # no nag_reason
    report = _validate(line, book=book)
    assert not report.ok
    assert any(f.check == "annotations" for f in report.errors)


def test_justified_nag_is_accepted(book) -> None:
    annotation = MoveAnnotation(san="Bc4", nag="!", nag_reason="the only move that holds (180 cp)")
    report = _validate(_line(Bc4=annotation), book=book)
    assert report.ok


def test_nag_without_an_engine_is_rejected(book) -> None:
    annotation = MoveAnnotation(san="Bc4", nag="!", nag_reason="looks nice")
    report = _validate(_line(Bc4=annotation), book=book, engine_available=False)
    assert not report.ok
    assert any(f.check == "annotations" for f in report.errors)


# --------------------------------------------------------------------------- #
# Engine coverage
# --------------------------------------------------------------------------- #


def test_missing_engine_is_a_warning_and_flips_the_flag(book) -> None:
    report = _validate(_line(), book=book, engine_available=False)
    assert report.ok                       # still a valid PGN
    assert report.engine_validated is False
    assert any("ENGINE VALIDATION UNAVAILABLE" in f.message for f in report.warnings)


def test_unanalysed_critical_position_warns(book) -> None:
    line = _line()
    report = _validate(
        line,
        book=book,
        critical_positions=["some-fen-that-was-never-analysed"],
        analysed_fens=[],
    )
    assert report.ok
    assert any(f.check == "engine" for f in report.warnings)


# --------------------------------------------------------------------------- #
# Start position
# --------------------------------------------------------------------------- #


def test_requested_fen_must_match(book) -> None:
    line = _line()
    other = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    report = _validate(line, book=book, requested_fen=other)
    assert not report.ok
    assert any(f.check == "fen" for f in report.errors)


def test_requested_moves_must_be_a_prefix(book) -> None:
    line = _line()
    report = _validate(line, book=book, requested_moves=["d4", "d5"])
    assert not report.ok
    assert any(f.check == "fen" for f in report.errors)


def test_matching_prefix_passes(book) -> None:
    line = _line()
    report = _validate(line, book=book, requested_moves=["e4", "e5"])
    assert report.ok


def test_custom_start_position_round_trips(book) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    records = []
    for san in ("a6", "Ba4", "Nf6"):
        move = board.parse_san(san)
        records.append(MoveRecord(move=move, san=board.san(move)))
        board.push(move)
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=records,
        headers={"FEN": chess.Board(fen).fen(), "SetUp": "1"},
    )
    report = _validate(line, book=book, requested_fen=fen)
    assert report.ok


def test_custom_start_position_always_gets_a_fen_header(book) -> None:
    """A line from a custom position is meaningless without its FEN.

    ``include_fen_headers: "never"`` cannot suppress it: correctness wins over
    presentation, and the writer emits the header regardless.
    """
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    move = board.parse_san("a6")
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=[MoveRecord(move=move, san="a6")],
        headers={},                     # deliberately no FEN/SetUp
    )
    config = build_config({"output": {"include_fen_headers": "never"}})
    pgn, _game = line_to_pgn(line, config)
    assert '[SetUp "1"]' in pgn
    assert f'[FEN "{chess.Board(fen).fen()}"]' in pgn
    report = _validate(line, config=config, book=book)
    assert report.ok


def test_stripped_fen_header_is_caught(book) -> None:
    """If the FEN header is lost in transit, the round-trip check must notice."""
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    move = board.parse_san("a6")
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=[MoveRecord(move=move, san="a6")],
        headers={},
    )
    config = build_config({})
    pgn, _game = line_to_pgn(line, config)
    damaged = "\n".join(
        row for row in pgn.splitlines() if not row.startswith(("[FEN ", "[SetUp "))
    )
    report = validate(
        line=line,
        pgn_text=damaged,
        config=config,
        book=book,
        engine_available=True,
    )
    assert not report.ok
    assert any(f.check == "pgn" for f in report.errors)


def test_wrong_fen_header_is_caught(book) -> None:
    """A FEN header describing a different position must fail the round-trip."""
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    move = board.parse_san("a6")
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=[MoveRecord(move=move, san="a6")],
        headers={},
    )
    config = build_config({})
    pgn, _game = line_to_pgn(line, config)
    other = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    damaged = pgn.replace(chess.Board(fen).fen(), other)
    report = validate(
        line=line,
        pgn_text=damaged,
        config=config,
        book=book,
        engine_available=True,
    )
    assert not report.ok
    assert any(f.check == "pgn" for f in report.errors)


# --------------------------------------------------------------------------- #
# Opening
# --------------------------------------------------------------------------- #


def test_matching_opening_passes(book) -> None:
    entry, _ = book.resolve("Italian Game")
    report = _validate(
        _line(),
        book=book,
        requested_eco=entry.eco,
        requested_name=entry.name,
        requested_uci=list(entry.uci),
    )
    assert report.ok
    assert not any(f.check == "opening" for f in report.warnings)


def test_wrong_opening_warns(book) -> None:
    entry, _ = book.resolve("Sicilian Najdorf")
    report = _validate(
        _line(),
        book=book,
        requested_eco=entry.eco,
        requested_name=entry.name,
        requested_uci=list(entry.uci),
    )
    assert report.ok                     # legal, just not what was asked for
    assert any(f.check == "opening" for f in report.warnings)


def test_line_shorter_than_the_book_line_warns(book) -> None:
    entry, _ = book.resolve("Sicilian Najdorf")
    board = chess.Board()
    records = []
    for uci in entry.uci[:4]:
        move = chess.Move.from_uci(uci)
        records.append(MoveRecord(move=move, san=board.san(move)))
        board.push(move)
    line = LineData(
        start_fen=chess.Board().fen(), from_initial=True, moves=records, headers={}
    )
    report = _validate(
        line,
        book=book,
        requested_eco=entry.eco,
        requested_name=entry.name,
        requested_uci=list(entry.uci),
    )
    assert report.ok
    assert any("shorter" in f.message for f in report.warnings)
