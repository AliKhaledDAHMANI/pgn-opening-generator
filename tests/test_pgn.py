"""Tests for PGN assembly, suffix rendering and round-tripping."""

from __future__ import annotations

import chess
import chess.pgn
import pytest

from pgn_generator.annotate import MoveAnnotation
from pgn_generator.config import build_config
from pgn_generator.pgn import (
    LineData,
    MoveRecord,
    VariationRecord,
    line_to_pgn,
    parse_pgn,
)


def _records(board: chess.Board, sans, **annotations) -> list:
    """Build move records for ``sans``, attaching any annotations given by SAN."""
    records = []
    for san in sans:
        move = board.parse_san(san)
        records.append(
            MoveRecord(move=move, san=board.san(move), annotation=annotations.get(san))
        )
        board.push(move)
    return records


def _italian_line(**annotations) -> LineData:
    board = chess.Board()
    records = _records(board, ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"], **annotations)
    return LineData(
        start_fen=chess.Board().fen(),
        from_initial=True,
        moves=records,
        headers={"ECO": "C50", "Opening": "Italian Game"},
    )


def test_headers_and_movetext() -> None:
    config = build_config({})
    pgn, _game = line_to_pgn(_italian_line(), config)
    assert '[Event "Opening Analysis"]' in pgn
    assert '[ECO "C50"]' in pgn
    assert '[Opening "Italian Game"]' in pgn
    assert '[Result "*"]' in pgn
    assert "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *" in pgn


def test_round_trip_preserves_moves_and_nags() -> None:
    config = build_config({})
    annotation = MoveAnnotation(san="Bc4", nag="!", comment_parts=["Italian Game (C50)"])
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    assert "Bc4!" in pgn

    parsed, errors = parse_pgn(pgn)
    assert parsed is not None
    assert errors == []
    assert [m.uci() for m in parsed.mainline_moves()] == [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"
    ]
    nags = {node.san(): node.nags for node in parsed.mainline() if node.move is not None}
    assert nags["Bc4"] == {1}


@pytest.mark.parametrize("suffix,code", [("!", 1), ("?", 2), ("!!", 3), ("??", 4), ("!?", 5), ("?!", 6)])
def test_every_suffix_round_trips(suffix, code) -> None:
    config = build_config({})
    annotation = MoveAnnotation(san="Bc4", nag=suffix)
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    assert f"Bc4{suffix}" in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    nags = {node.san(): node.nags for node in parsed.mainline() if node.move is not None}
    assert nags["Bc4"] == {code}


def test_nag_code_mode_writes_dollar_tokens() -> None:
    config = build_config({"output": {"use_nag_codes": True}})
    annotation = MoveAnnotation(san="Bc4", nag="!!")
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    assert "Bc4 $3" in pgn
    assert "Bc4!!" not in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []


def test_suffix_and_comment_together() -> None:
    config = build_config({})
    annotation = MoveAnnotation(
        san="Bc4",
        nag="!",
        comment_parts=["White targets f7."],
        eval_text="= +0.20",
    )
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    assert "Bc4! {= +0.20 White targets f7.}" in pgn
    parsed, _errors = parse_pgn(pgn)
    assert parsed is not None
    comments = {node.san(): node.comment for node in parsed.mainline() if node.move is not None}
    assert "White targets f7." in comments["Bc4"]


def test_arrow_markup_is_machine_readable() -> None:
    config = build_config({})
    annotation = MoveAnnotation(san="Bc4")
    annotation.arrows = [("G", chess.C4, chess.F7), ("R", chess.D1, chess.H5)]
    annotation.highlights = [("R", chess.F7)]
    annotation.comment_parts = ["The f7 square is the target."]
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    assert "[%cal Gc4f7,Rd1h5]" in pgn
    assert "[%csl Rf7]" in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []


def test_nested_variation_branches_from_the_right_position() -> None:
    config = build_config({})
    line = _italian_line()

    # Alternative to 3...Bc5 (index 5): 3...Nf6 4.O-O.
    branch_board = chess.Board()
    for record in line.moves[:5]:
        branch_board.push(record.move)
    start_fen = branch_board.fen()
    variation_records = _records(branch_board.copy(stack=True), ["Nf6", "O-O"])
    line.moves[5].variations.append(
        VariationRecord(
            start_ply=5,
            start_fen=start_fen,
            moves=variation_records,
            purpose="Two Knights Defense",
            kind="theory",
        )
    )

    pgn, _game = line_to_pgn(line, config)
    assert "(" in pgn and ")" in pgn
    assert "3... Nf6 4. O-O" in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []

    # The sideline hangs off the node *before* 3...Bc5, i.e. the parent of ply 5.
    parent = parsed
    for _ in range(5):
        parent = parent.variations[0]
    assert len(parent.variations) == 2, "the alternative must be a sibling of 3...Bc5"
    parent_board = parent.board()
    main, sideline = parent.variations
    assert parent_board.san(main.move) == "Bc5"
    assert parent_board.san(sideline.move) == "Nf6"
    assert sideline.move in parent_board.legal_moves
    assert "Two Knights" in (sideline.starting_comment or "")

    # And the sideline continues legally from there.
    sideline_board = parent_board.copy(stack=False)
    sideline_board.push(sideline.move)
    follow = sideline.variations[0]
    assert sideline_board.san(follow.move) == "O-O"


def test_custom_start_position_emits_fen_headers() -> None:
    config = build_config({})
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    board = chess.Board(fen)
    records = _records(board, ["a6", "Ba4", "Nf6"])
    line = LineData(
        start_fen=chess.Board(fen).fen(),
        from_initial=False,
        moves=records,
        headers={"FEN": chess.Board(fen).fen(), "SetUp": "1"},
    )
    pgn, _game = line_to_pgn(line, config)
    assert '[SetUp "1"]' in pgn
    assert f'[FEN "{chess.Board(fen).fen()}"]' in pgn

    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    assert parsed.board().fen() == chess.Board(fen).fen()
    assert [m.uci() for m in parsed.mainline_moves()] == ["a7a6", "b5a4", "g8f6"]


def test_intro_and_closing_comments_are_placed() -> None:
    config = build_config({})
    line = _italian_line()
    line.intro_comment = "Main theoretical line."
    line.closing_comment = "Engine assessment = +0.15."
    pgn, _game = line_to_pgn(line, config)
    assert pgn.index("{Main theoretical line.}") < pgn.index("1. e4")
    assert "Engine assessment = +0.15." in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    assert "Main theoretical" in parsed.comment


def test_comment_free_output_still_carries_the_moves() -> None:
    """``comments: False`` is enforced by the annotator; the writer stays faithful."""
    config = build_config({"output": {"comments": False}})
    line = _italian_line()
    pgn, _game = line_to_pgn(line, config)
    assert "{" not in pgn
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    assert [m.uci() for m in parsed.mainline_moves()] == [r.move.uci() for r in line.moves]


def test_column_wrapping_stays_parseable() -> None:
    config = build_config({"output": {"columns": 40}})
    annotation = MoveAnnotation(san="Bc4", nag="!", comment_parts=["A long comment " * 6])
    pgn, _game = line_to_pgn(line_data := _italian_line(Bc4=annotation), config)
    assert max(len(line) for line in pgn.splitlines() if not line.startswith("[")) <= 200
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
    assert [m.uci() for m in parsed.mainline_moves()] == [
        r.move.uci() for r in line_data.moves
    ]


def test_parse_pgn_reports_garbage() -> None:
    parsed, errors = parse_pgn("this is not a pgn at all")
    assert parsed is None or errors or not list(parsed.mainline_moves())


def test_curly_braces_in_comments_do_not_break_the_pgn() -> None:
    config = build_config({})
    annotation = MoveAnnotation(san="Bc4", comment_parts=["a }brace{ inside"])
    pgn, _game = line_to_pgn(_italian_line(Bc4=annotation), config)
    parsed, errors = parse_pgn(pgn)
    assert parsed is not None and errors == []
