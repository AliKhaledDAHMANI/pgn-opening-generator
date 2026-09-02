"""PGN assembly.

Two responsibilities:

* build a ``chess.pgn.Game`` from a generated line plus its annotations, including
  nested variations that always start from the right position;
* serialise it with move suffixes (``!``, ``?!``) rendered the way human readers
  expect, while remaining parseable by standard PGN tools.

On suffix rendering: the PGN standard stores judgements as NAGs (``$1``, ``$5``).
Viewers such as Lichess, ChessBase and SCID also accept the inline suffix form
(``Nf3!``), and it is what people actually read, so it is the default.
``output.use_nag_codes`` switches to strict ``$n`` output. Both round-trip through
python-chess.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import chess.pgn

from .annotate import NAG_CODES, MoveAnnotation
from .config import Config

#: Reverse mapping for reading NAGs back out of a parsed game.
NAG_SUFFIXES: Dict[int, str] = {code: suffix for suffix, code in NAG_CODES.items()}


@dataclass
class MoveRecord:
    """One move of a generated line, with everything needed to render it."""

    move: chess.Move
    san: str
    annotation: Optional[MoveAnnotation] = None
    #: Sidelines attached *at this move* (alternatives to it).
    variations: List["VariationRecord"] = field(default_factory=list)

    @property
    def nag(self) -> Optional[str]:
        return self.annotation.nag if self.annotation else None


@dataclass
class VariationRecord:
    """A sideline: the position it starts from, its moves, and why it exists."""

    #: Ply index in the parent line at which this variation branches (0-based:
    #: the variation replaces the parent's move at this index).
    start_ply: int
    #: FEN of the position the variation starts from, used for validation.
    start_fen: str
    moves: List[MoveRecord]
    purpose: str = ""
    #: Nested sidelines inside this variation.
    kind: str = "alternative"   # alternative | refutation | engine | theory | trap

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_ply": self.start_ply,
            "start_fen": self.start_fen,
            "kind": self.kind,
            "purpose": self.purpose,
            "moves": [record.san for record in self.moves],
        }


@dataclass
class LineData:
    """A complete generated analysis ready for PGN serialisation."""

    start_fen: str
    #: ``True`` when the line starts from the standard initial position.
    from_initial: bool
    moves: List[MoveRecord]
    headers: Dict[str, str] = field(default_factory=dict)
    intro_comment: Optional[str] = None
    #: Comment appended after the final move.
    closing_comment: Optional[str] = None

    def san_line(self) -> str:
        board = chess.Board(self.start_fen) if not self.from_initial else chess.Board()
        return board.variation_san([record.move for record in self.moves]) if self.moves else ""

    def final_board(self) -> chess.Board:
        board = chess.Board(self.start_fen) if not self.from_initial else chess.Board()
        for record in self.moves:
            board.push(record.move)
        return board


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _apply_annotation(
    node: chess.pgn.ChildNode,
    annotation: Optional[MoveAnnotation],
    config: Config,
) -> None:
    if annotation is None:
        return
    if annotation.nag and config.output.annotations != "none":
        code = NAG_CODES.get(annotation.nag)
        if code is not None:
            node.nags.add(code)
    comment = annotation.comment(arrow_prefix=True)
    if comment:
        node.comment = comment


def build_game(line: LineData, config: Config) -> chess.pgn.Game:
    """Assemble a :class:`chess.pgn.Game` from ``line``.

    Variations are attached to the *parent* of the move they replace, which is what
    makes ``(3...Nf6 4.O-O)`` branch from the position before ``3...a6`` rather than
    after it.
    """
    game = chess.pgn.Game()
    output = config.output

    game.headers["Event"] = output.event
    game.headers["Site"] = output.site
    game.headers["Date"] = output.date
    game.headers["Round"] = output.round_
    game.headers["White"] = output.white
    game.headers["Black"] = output.black
    game.headers["Result"] = output.result
    for key, value in line.headers.items():
        if value:
            game.headers[key] = value
    for key, value in output.extra_headers.items():
        if value:
            game.headers[key] = value

    if not line.from_initial:
        # ``setup`` writes the FEN/SetUp headers, which a custom start position
        # cannot be replayed without.
        game.setup(chess.Board(line.start_fen))
    if line.intro_comment:
        game.comment = line.intro_comment

    _build_nodes(game, line.moves, config, board=game.board())

    if line.closing_comment:
        node: chess.pgn.GameNode = game
        while node.variations:
            node = node.variations[0]
        if isinstance(node, chess.pgn.ChildNode):
            node.comment = (node.comment + " " + line.closing_comment).strip() if node.comment else line.closing_comment
        else:  # no moves at all
            game.comment = (game.comment + " " + line.closing_comment).strip()
    return game


def _build_nodes(
    parent: chess.pgn.GameNode,
    records: Sequence[MoveRecord],
    config: Config,
    *,
    board: chess.Board,
) -> None:
    """Add ``records`` as the main continuation of ``parent``."""
    node = parent
    for record in records:
        # Variations branch from the position *before* this move.
        for variation in record.variations:
            _build_variation(node, variation, config, board=board)
        child = node.add_main_variation(record.move)
        _apply_annotation(child, record.annotation, config)
        board.push(record.move)
        node = child


def _build_variation(
    parent: chess.pgn.GameNode,
    variation: VariationRecord,
    config: Config,
    *,
    board: chess.Board,
) -> None:
    if not variation.moves:
        return
    branch_board = board.copy(stack=False)
    first = variation.moves[0]
    node = parent.add_variation(first.move)
    if variation.purpose and config.output.comments:
        node.starting_comment = variation.purpose
    _apply_annotation(node, first.annotation, config)
    branch_board.push(first.move)
    for record in variation.moves[1:]:
        node = node.add_main_variation(record.move)
        _apply_annotation(node, record.annotation, config)
        branch_board.push(record.move)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


class _SuffixExporter(chess.pgn.StringExporter):
    """String exporter that renders judgement NAGs as inline suffixes.

    python-chess writes NAGs as ``$n`` tokens after the move. Here the suffix is
    folded into the SAN token itself (``Nf3!``), which is how chess literature and
    every mainstream viewer displays it. Non-judgement NAGs (positional
    assessments, novelty markers) keep the ``$n`` form.
    """

    def __init__(self, *, columns: Optional[int], suffix_nags: bool = True) -> None:
        super().__init__(columns=columns, headers=True, comments=True, variations=True)
        self.suffix_nags = suffix_nags
        self._pending_suffix: Optional[str] = None

    def visit_nag(self, nag: int) -> None:
        if self.suffix_nags and nag in NAG_SUFFIXES:
            # Only one judgement suffix per move; the first one wins.
            if self._pending_suffix is None:
                self._pending_suffix = NAG_SUFFIXES[nag]
            return
        self._flush_suffix()
        super().visit_nag(nag)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self._flush_suffix()
        super().visit_move(board, move)

    def visit_comment(self, comment: str) -> None:
        self._flush_suffix()
        super().visit_comment(comment)

    def begin_variation(self):  # type: ignore[override]
        self._flush_suffix()
        return super().begin_variation()

    def end_variation(self) -> None:
        self._flush_suffix()
        super().end_variation()

    def visit_result(self, result: str) -> None:
        self._flush_suffix()
        super().visit_result(result)

    def result(self) -> str:
        self._flush_suffix()
        return super().result()

    def _flush_suffix(self) -> None:
        """Attach the pending suffix to the token that was just written.

        The exporter emits tokens with a trailing space, so the space is removed,
        the suffix appended, and the space restored - otherwise the next token
        would be glued onto the suffix.
        """
        if self._pending_suffix is None:
            return
        suffix = self._pending_suffix
        self._pending_suffix = None
        if self.current_line.rstrip():
            self.current_line = self.current_line.rstrip() + suffix + " "
        elif self.lines:
            self.lines[-1] = self.lines[-1].rstrip() + suffix


def game_to_pgn(game: chess.pgn.Game, config: Config) -> str:
    """Serialise a game to PGN text."""
    exporter = _SuffixExporter(
        columns=config.output.columns,
        suffix_nags=not config.output.use_nag_codes,
    )
    text = game.accept(exporter)
    return _tidy(text)


_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_BRACE_SPACE = re.compile(r"\{\s+")
_BRACE_SPACE_END = re.compile(r"\s+\}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([)\]])")
_PAREN_OPEN = re.compile(r"\(\s+")


def _tidy(text: str) -> str:
    """Normalise whitespace produced by the exporter without breaking PGN."""
    lines: List[str] = []
    for line in text.split("\n"):
        if line.startswith("["):
            lines.append(line)
            continue
        cleaned = _BRACE_SPACE.sub("{", line)
        cleaned = _BRACE_SPACE_END.sub("}", cleaned)
        cleaned = _PAREN_OPEN.sub("(", cleaned)
        cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
        cleaned = _MULTI_SPACE.sub(" ", cleaned)
        lines.append(cleaned.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def line_to_pgn(line: LineData, config: Config) -> Tuple[str, chess.pgn.Game]:
    """Build and serialise in one step."""
    game = build_game(line, config)
    return game_to_pgn(game, config), game


def parse_pgn(text: str) -> Tuple[Optional[chess.pgn.Game], List[str]]:
    """Parse PGN text, returning the game and any parser errors."""
    handle = io.StringIO(text)
    try:
        game = chess.pgn.read_game(handle)
    except Exception as exc:  # noqa: BLE001 - parser robustness
        return None, [f"parser raised {type(exc).__name__}: {exc}"]
    if game is None:
        return None, ["no game found in PGN text"]
    errors = [str(error) for error in game.errors]
    return game, errors
