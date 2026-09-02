"""Tests for annotation discipline.

The contract these tests defend: a judgement symbol is never decoration. It is
attached only when Stockfish's numbers justify it, and ``!!`` needs more than
"best move".
"""

from __future__ import annotations

import chess
import pytest

from pgn_generator.annotate import (
    Annotator,
    MoveAnnotation,
    eval_symbol,
    format_score,
)
from pgn_generator.config import build_config
from pgn_generator.engine import Score
from pgn_generator.features import analyse_move
from pgn_generator.selector import MoveSelector, ScoredMove

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Score formatting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cp,expected",
    [(0, "="), (80, "+/="), (200, "+/-"), (400, "+-"), (-80, "=/+"), (-200, "-/+"), (-400, "-+")],
)
def test_eval_symbols(cp, expected) -> None:
    assert eval_symbol(Score(cp=cp)) == expected


def test_pawn_format() -> None:
    config = build_config({"output": {"eval_format": "pawns"}})
    assert format_score(Score(cp=35), config, prefix=False) == "+0.35"
    assert format_score(Score(cp=-140), config, prefix=False) == "-1.40"


def test_centipawn_format() -> None:
    config = build_config({"output": {"eval_format": "centipawns"}})
    assert format_score(Score(cp=35), config, prefix=False) == "+35 cp"


def test_verbose_format_names_the_engine() -> None:
    config = build_config({"output": {"eval_format": "verbose"}})
    assert format_score(Score(cp=35), config) == "Stockfish: +0.35"


def test_mate_scores_are_rendered_as_mate() -> None:
    config = build_config({"output": {"eval_format": "pawns"}})
    assert format_score(Score(cp=9997, mate=3), config, prefix=False) == "#3"
    assert format_score(Score(cp=-9998, mate=-2), config, prefix=False) == "#-2"


# --------------------------------------------------------------------------- #
# Comment rendering
# --------------------------------------------------------------------------- #


def test_comment_puts_markup_first_when_asked() -> None:
    annotation = MoveAnnotation(san="Bc4")
    annotation.arrows = [("G", chess.C4, chess.F7)]
    annotation.highlights = [("R", chess.F7)]
    annotation.comment_parts = ["Targets f7."]
    annotation.eval_text = "= +0.20"
    body = annotation.comment(arrow_prefix=True)
    assert body.startswith("[%cal Gc4f7][%csl Rf7]")
    assert "= +0.20 Targets f7." in body


def test_duplicate_arrows_are_collapsed() -> None:
    annotation = MoveAnnotation(san="Bc4")
    annotation.arrows = [("G", chess.C4, chess.F7), ("G", chess.C4, chess.F7)]
    assert annotation.visual_markup() == "[%cal Gc4f7]"


def test_empty_annotation_has_no_content() -> None:
    assert MoveAnnotation(san="e4").has_content is False


# --------------------------------------------------------------------------- #
# Judgement discipline
# --------------------------------------------------------------------------- #


def _candidate(board: chess.Board, san: str, *, cp_loss: int, score_cp: int = 0) -> ScoredMove:
    move = board.parse_san(san)
    features = analyse_move(board, move)
    candidate = ScoredMove(move=move, san=features.san, features=features, source="engine")
    candidate.cp_loss = cp_loss
    candidate.score_after = Score(cp=score_cp)
    return candidate


def test_no_engine_means_no_judgement(book) -> None:
    """Without engine numbers there is no basis for a ! or ?, so none is emitted."""
    config = build_config({})
    annotator = Annotator(config, book)
    board = chess.Board()
    board.push_san("e4")
    candidate = _candidate(board, "e5", cp_loss=0)
    nag, reason = annotator.judge(
        candidate, selection=None, board_before=board, engine_available=False
    )
    assert nag is None and reason is None


@pytest.mark.parametrize(
    "cp_loss,expected",
    [(0, None), (30, None), (60, "?!"), (150, "?"), (400, "??")],
)
def test_loss_thresholds(book, cp_loss, expected) -> None:
    config = build_config({})
    annotator = Annotator(config, book)
    board = chess.Board()
    board.push_san("e4")
    candidate = _candidate(board, "e5", cp_loss=cp_loss)
    nag, reason = annotator.judge(
        candidate, selection=None, board_before=board, engine_available=True
    )
    assert nag == expected
    if expected is not None:
        assert reason and str(cp_loss) in reason


def test_annotations_off_suppresses_everything(book) -> None:
    config = build_config({"output": {"annotations": "none"}})
    annotator = Annotator(config, book)
    board = chess.Board()
    board.push_san("e4")
    candidate = _candidate(board, "e5", cp_loss=400)
    nag, _reason = annotator.judge(
        candidate, selection=None, board_before=board, engine_available=True
    )
    assert nag is None


def test_forced_move_gets_no_praise(book) -> None:
    config = build_config({})
    annotator = Annotator(config, book)
    board = chess.Board("k7/7R/1K6/8/8/8/8/8 b - - 0 1")
    move = next(iter(board.legal_moves))
    features = analyse_move(board, move)
    assert features.forced
    candidate = ScoredMove(move=move, san=features.san, features=features, source="engine")
    candidate.cp_loss = 0
    nag, _reason = annotator.judge(
        candidate, selection=None, board_before=board, engine_available=True
    )
    assert nag is None


def test_checkmate_always_earns_a_mark(book) -> None:
    config = build_config({})
    annotator = Annotator(config, book)
    board = chess.Board()
    for san in ("f3", "e5", "g4"):
        board.push_san(san)
    candidate = _candidate(board, "Qh4#", cp_loss=0)
    nag, reason = annotator.judge(
        candidate, selection=None, board_before=board, engine_available=True
    )
    assert nag == "!"
    assert "checkmate" in (reason or "")


@requires_engine
def test_sacrifice_confirmation_drops_false_positives(book, config, engine) -> None:
    """...e5 in the King's Indian is not a sacrifice, whatever SEE says."""
    annotator = Annotator(config, book, engine=engine)
    board = chess.Board()
    for san in ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2"):
        board.push_san(san)
    move = board.parse_san("e5")
    features = analyse_move(board, move)
    after = board.copy(stack=False)
    after.push(move)
    annotator.confirm_investment(board, after, features)
    assert features.is_sacrifice is False
    assert features.invested_cp == 0
    assert not any(m.kind == "material_investment" for m in features.motifs)


@requires_engine
def test_sacrifice_confirmation_keeps_real_investments(book, config, engine) -> None:
    """A knight thrown at f7 for a pawn stays a sacrifice."""
    annotator = Annotator(config, book, engine=engine)
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Nxd5"):
        board.push_san(san)
    move = board.parse_san("Nxf7")
    features = analyse_move(board, move)
    after = board.copy(stack=False)
    after.push(move)
    annotator.confirm_investment(board, after, features)
    assert features.is_sacrifice is True
    assert features.invested_cp >= 90


@requires_engine
def test_evaluations_are_only_emitted_when_real(book, config, engine) -> None:
    annotator = Annotator(config, book, engine=engine)
    board = chess.Board()
    board.push_san("e4")
    selector = MoveSelector(config, book, engine)
    selection = selector.select(board)
    assert selection is not None
    after = board.copy(stack=False)
    after.push(selection.move)

    annotation = annotator.annotate_move(
        board_before=board,
        board_after=after,
        candidate=selection.chosen,
        selection=selection,
        ply=1,
        engine_available=True,
        is_critical=True,
    )
    assert annotation.eval_text is not None

    # Same move, but the caller says there is no engine: no numbers may appear.
    annotator_no_engine = Annotator(config, book)
    silent = annotator_no_engine.annotate_move(
        board_before=board,
        board_after=after,
        candidate=selection.chosen,
        selection=selection,
        ply=1,
        engine_available=False,
        is_critical=True,
    )
    assert silent.eval_text is None
    assert silent.nag is None


@requires_engine
def test_book_names_are_not_repeated(book, config, engine) -> None:
    annotator = Annotator(config, book, engine=engine)
    board = chess.Board()
    selector = MoveSelector(config, book, engine)
    labels = []
    for _ in range(6):
        selection = selector.select(board)
        if selection is None:
            break
        after = board.copy(stack=False)
        after.push(selection.move)
        annotation = annotator.annotate_move(
            board_before=board,
            board_after=after,
            candidate=selection.chosen,
            selection=selection,
            ply=len(board.move_stack),
            engine_available=True,
        )
        labels.extend(part for part in annotation.comment_parts if "(" in part and ")" in part)
        board.push(selection.move)
    assert len(labels) == len(set(labels)), f"repeated opening labels: {labels}"
