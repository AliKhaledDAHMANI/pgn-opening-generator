"""Annotations, comments, evaluations and board arrows.

Rules this module enforces, so that output is never decorative noise:

* A NAG (``!``, ``!?``, ``?!``, ``?``, ``??``, ``!!``) is only attached when the
  engine's centipawn numbers support it. Without an engine, no judgement NAGs are
  emitted at all - only the objective ``+``/``#`` that SAN already carries.
* ``!!`` additionally requires a concrete, verifiable reason (a material
  investment or a forced mate) - "best move" alone never earns it.
* Comments are built from facts: book names come from the ECO data set, tactical
  claims from :mod:`~pgn_generator.features`, numbers from Stockfish.
* Evaluations are printed only when the engine actually produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chess

from .book import BookEntry, OpeningBook
from .config import Config
from .engine import Score
from .features import (
    Motif,
    MoveFeatures,
    analyse_position,
    material_balance,
    square_name,
)
from .selector import ScoredMove, SelectionResult

# --------------------------------------------------------------------------- #
# NAG mapping
# --------------------------------------------------------------------------- #

#: Suffix -> numeric NAG, per the PGN standard (``$1``..``$6``).
NAG_CODES: Dict[str, int] = {"!": 1, "?": 2, "!!": 3, "??": 4, "!?": 5, "?!": 6}

#: Thresholds in centipawns, applied to the loss vs. the engine's best move.
BLUNDER_CP = 250
MISTAKE_CP = 120
INACCURACY_CP = 55

#: Depth used for the "would anyone spot this?" check behind ``!`` annotations.
_OBVIOUS_DEPTH = 6

#: How far along the engine's principal variation a claimed sacrifice is followed
#: before deciding whether the material is really gone.
_INVESTMENT_PV_PLIES = 8

#: Minimum material still missing at the end of the PV for a move to be called a
#: sacrifice (just under a pawn, to tolerate the odd tempo-for-pawn line).
_CONFIRMED_INVESTMENT_CP = 90

#: Judgement suffixes; anything here must carry a recorded justification.
JUDGEMENT_NAGS = ("!", "!!", "?", "??", "!?", "?!")

#: Motifs that describe a concrete tactical point and always deserve a sentence.
_TACTICAL_MOTIFS = (
    "mate",
    "mate_threat",
    "fork",
    "pin",
    "discovered_attack",
    "material_investment",
    "hanging",
    "king_attack",
)

#: Structural observations: informative once, tedious when repeated. Only used at
#: the ``rich`` annotation level, and only the first time each is true.
_STRUCTURAL_MOTIFS = ("open_file",)

#: Motif kinds whose highlighted squares are targets/weaknesses (red rather than
#: informational yellow).
_RED_SQUARE_MOTIFS = ("mate", "fork", "threat", "hanging", "pin", "mate_threat", "king_attack")


@dataclass
class MoveAnnotation:
    """Everything attached to one move in the PGN."""

    san: str
    nag: Optional[str] = None
    nag_reason: Optional[str] = None
    comment_parts: List[str] = field(default_factory=list)
    eval_text: Optional[str] = None
    arrows: List[Tuple[str, int, int]] = field(default_factory=list)   # (colour, from, to)
    highlights: List[Tuple[str, int]] = field(default_factory=list)   # (colour, square)

    def comment(self, *, arrow_prefix: bool = False) -> str:
        """Render the PGN comment body (without the surrounding braces)."""
        pieces: List[str] = []
        visual = self.visual_markup()
        if visual and arrow_prefix:
            pieces.append(visual)
        text = " ".join(part.strip() for part in self.comment_parts if part and part.strip())
        if self.eval_text:
            text = f"{self.eval_text} {text}".strip() if text else self.eval_text
        if text:
            pieces.append(text)
        if visual and not arrow_prefix:
            pieces.append(visual)
        return " ".join(pieces).strip()

    def visual_markup(self) -> str:
        """Lichess/ChessBase-compatible ``[%cal ...][%csl ...]`` markup."""
        out = ""
        if self.arrows:
            seen: List[str] = []
            for colour, src, dst in self.arrows:
                token = f"{colour}{square_name(src)}{square_name(dst)}"
                if token not in seen:
                    seen.append(token)
            out += "[%cal " + ",".join(seen) + "]"
        if self.highlights:
            seen_sq: List[str] = []
            for colour, square in self.highlights:
                token = f"{colour}{square_name(square)}"
                if token not in seen_sq:
                    seen_sq.append(token)
            out += "[%csl " + ",".join(seen_sq) + "]"
        return out

    @property
    def has_content(self) -> bool:
        return bool(self.nag or self.comment_parts or self.eval_text or self.arrows or self.highlights)

    def to_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "nag": self.nag,
            "nag_reason": self.nag_reason,
            "comment": self.comment(),
            "eval": self.eval_text,
        }


# --------------------------------------------------------------------------- #
# Evaluation formatting
# --------------------------------------------------------------------------- #


def format_score(score: Score, config: Config, *, prefix: bool = True) -> str:
    """Render an engine score in the configured format.

    Always from White's point of view unless ``output.eval_white_pov`` is off, in
    which case the sign follows the side to move (handled by the caller passing a
    side-relative score).
    """
    fmt = config.output.eval_format
    if score.mate is not None:
        mate = score.mate
        if mate == 0:
            body = "mate"
        else:
            body = f"#{abs(mate)}" if mate > 0 else f"#-{abs(mate)}"
        if fmt == "verbose":
            side = "White" if mate > 0 else "Black"
            return f"Stockfish: mate in {abs(mate)} for {side}" if mate else "Stockfish: mate"
        return f"Stockfish: {body}" if prefix and fmt == "centipawns" else body

    if fmt == "centipawns":
        value = f"{score.cp:+d} cp"
        return f"Stockfish: {value}" if prefix else value
    pawns = score.cp / 100.0
    value = f"{pawns:+.2f}"
    if fmt == "verbose":
        return f"Stockfish: {value}"
    return value


def eval_symbol(score: Score) -> str:
    """Positional assessment symbol for a score (=, +/-, etc.)."""
    cp = score.cp
    if score.mate is not None:
        return "+-" if cp > 0 else "-+"
    if cp >= 300:
        return "+-"
    if cp >= 150:
        return "+/-"
    if cp >= 70:
        return "+/="
    if cp > -70:
        return "="
    if cp > -150:
        return "=/+"
    if cp > -300:
        return "-/+"
    return "-+"


# --------------------------------------------------------------------------- #
# The annotator
# --------------------------------------------------------------------------- #


class Annotator:
    """Turns selection results into justified NAGs, comments and arrows.

    The engine is optional but strongly recommended: without it no judgement NAG
    is emitted at all, since there would be no basis for one.
    """

    #: Minimum plies between ordinary comments, per annotation level. Important
    #: moves (judgements, tactics, new opening names) bypass this.
    _COMMENT_SPACING = {"none": 999, "minimal": 6, "standard": 3, "rich": 2}

    def __init__(self, config: Config, book: OpeningBook, engine: Optional[Any] = None) -> None:
        self.config = config
        self.book = book
        self.engine = engine
        self._last_comment_ply: int = -99
        self._last_book_label: Optional[str] = None
        self._obvious_cache: Dict[Tuple[str, str], bool] = {}
        #: Strategic observations already made, so each is stated once per line.
        self._stated_plans: set = set()
        #: Structural motifs already reported (e.g. "the e-file opens").
        self._stated_observations: set = set()

    # -- NAGs -------------------------------------------------------------- #

    def judge(
        self,
        candidate: ScoredMove,
        *,
        selection: Optional[SelectionResult],
        board_before: chess.Board,
        engine_available: bool,
        intentional: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Decide the NAG for a move, with the reason that justifies it.

        ``intentional`` is ``True`` for moves the generator chose and ``False`` for
        moves inserted in order to be refuted, where a ``?`` is the point rather
        than a defect of the analysis.
        """
        if self.config.output.annotations == "none":
            return None, None
        if not engine_available:
            # Without engine numbers there is no defensible basis for a judgement.
            return None, None

        features = candidate.features
        loss = candidate.cp_loss

        if features.is_checkmate:
            return "!", "delivers checkmate"
        if loss is None:
            return None, None

        # -- bad moves ------------------------------------------------------ #
        if loss >= BLUNDER_CP:
            return "??", f"loses {loss} cp compared with the best move"
        if loss >= MISTAKE_CP:
            return "?", f"loses {loss} cp compared with the best move"
        if loss >= INACCURACY_CP:
            if features.is_sacrifice or features.is_forcing:
                return "?!", f"sharp, but objectively {loss} cp worse than the best move"
            return "?!", f"objectively {loss} cp worse than the best move"

        # -- good moves ----------------------------------------------------- #
        # Praise requires two things: the move must be (near-)best, and it must be
        # hard to find. "Best" alone would decorate every recapture.
        if loss > 10:
            return None, None
        if features.forced:
            return None, None   # the only legal move is not praiseworthy

        gap = self._second_best_gap(selection)
        only_move = gap is not None and gap >= 150
        clearly_best = gap is not None and gap >= 60

        if features.is_sacrifice and features.invested_cp >= 250 and only_move:
            detail = (
                f"gives up {features.invested_cp} cp of material and is the only move that keeps "
                f"the advantage ({gap} cp better than the second choice)"
            )
            return "!!", detail
        if features.is_sacrifice and features.invested_cp >= 90:
            return "!", (
                f"invests {features.invested_cp} cp of material and is still the engine's top "
                "choice"
            )
        if features.forks and clearly_best and not self._is_obvious(board_before, candidate):
            return "!", f"a double attack worth {gap} cp more than the alternative"
        if features.is_promotion and features.promotion_type not in (chess.QUEEN, None):
            return "!", "an underpromotion that is objectively best"
        if only_move and not self._is_obvious(board_before, candidate):
            return "!", (
                f"the only move that holds the evaluation ({gap} cp better than the alternative)"
            )
        return None, None

    def _is_obvious(self, board: chess.Board, candidate: ScoredMove) -> bool:
        """True when a quick glance already finds this move.

        Implemented as a deliberately shallow engine search: if the move heads a
        depth-limited search, any player would spot it, so a ``!`` would be
        decoration rather than information. Without an engine, captures and checks
        are treated as obvious and everything else as not, which is conservative in
        the direction of *fewer* annotations.
        """
        features = candidate.features
        if self.engine is None or not getattr(self.engine, "available", False):
            return features.is_capture or features.gives_check

        key = (board.epd(en_passant="legal"), candidate.move.uci())
        cached = self._obvious_cache.get(key)
        if cached is not None:
            return cached

        shallow = self.engine.analyse(board, depth=_OBVIOUS_DEPTH, multipv=1)
        verdict = bool(
            shallow is not None
            and shallow.candidates
            and shallow.candidates[0].move == candidate.move
        )
        self._obvious_cache[key] = verdict
        return verdict

    def confirm_investment(
        self, board_before: chess.Board, board_after: chess.Board, features: MoveFeatures
    ) -> None:
        """Check a candidate sacrifice against the engine, and drop it if unreal.

        :func:`~pgn_generator.features.static_exchange_eval` only looks at capture
        sequences on the arrival square, so it flags plenty of ordinary moves whose
        material comes straight back a move later (``...e5`` in the King's Indian
        being the classic case). This walks the engine's principal variation from
        the resulting position and keeps the sacrifice label only when the material
        is *still* missing at the end of it.

        Without an engine the label is dropped entirely: an unverified sacrifice
        claim is worse than no claim.
        """
        if not features.is_sacrifice:
            return
        confirmed = self._investment_deficit(board_before, board_after, features)
        if confirmed is not None and confirmed >= _CONFIRMED_INVESTMENT_CP:
            features.invested_cp = confirmed
            return
        features.is_sacrifice = False
        features.is_exchange_sacrifice = False
        features.invested_cp = 0
        features.motifs = [m for m in features.motifs if m.kind != "material_investment"]

    def _investment_deficit(
        self, board_before: chess.Board, board_after: chess.Board, features: MoveFeatures
    ) -> Optional[int]:
        """Material the mover is still down at the end of the engine's PV."""
        if self.engine is None or not getattr(self.engine, "available", False):
            return None
        analysis = self.engine.analyse(board_after, critical=True, multipv=1)
        if analysis is None or not analysis.candidates:
            return None
        pv = analysis.candidates[0].pv
        if not pv:
            return None
        probe = board_after.copy(stack=False)
        for move in pv[:_INVESTMENT_PV_PLIES]:
            if move not in probe.legal_moves:  # pragma: no cover - stale PV
                break
            probe.push(move)
        before = material_balance(board_before)
        after = material_balance(probe)
        # Positive means the mover has less material than before the move.
        deficit = (before - after) if features.color == chess.WHITE else (after - before)
        return max(0, deficit)

    def _second_best_gap(self, selection: Optional[SelectionResult]) -> Optional[int]:
        """Centipawns between the engine's best and second-best move."""
        if selection is None or selection.analysis is None:
            return None
        candidates = selection.analysis.candidates
        if len(candidates) < 2:
            return None
        turn = selection.analysis.turn
        return int(candidates[0].score.cp_for(turn) - candidates[1].score.cp_for(turn))

    # -- comments ---------------------------------------------------------- #

    def annotate_move(
        self,
        *,
        board_before: chess.Board,
        board_after: chess.Board,
        candidate: ScoredMove,
        selection: Optional[SelectionResult],
        ply: int,
        engine_available: bool,
        force_comment: bool = False,
        extra_comment: Optional[str] = None,
        intentional: bool = True,
        is_critical: bool = False,
    ) -> MoveAnnotation:
        """Build the full annotation for one move.

        Three independent switches decide what ends up in the PGN comment braces:
        ``output.comments`` (prose), ``output.evals`` (engine numbers) and
        ``output.arrows`` (``[%cal]``/``[%csl]`` markup). Turning off prose leaves
        the numbers and arrows in place, which is what a caller who wants a compact
        but still machine-readable line asks for.
        """
        output = self.config.output
        annotation = MoveAnnotation(san=candidate.san)

        # SEE-based sacrifice detection is only a candidate signal; confirm it
        # against the engine before any comment or NAG can rely on it.
        self.confirm_investment(board_before, board_after, candidate.features)

        nag, reason = self.judge(
            candidate,
            selection=selection,
            board_before=board_before,
            engine_available=engine_available,
            intentional=intentional,
        )
        annotation.nag = nag
        annotation.nag_reason = reason
        judged = nag in JUDGEMENT_NAGS

        if not output.comments and not force_comment:
            self._maybe_add_eval(annotation, candidate, engine_available, is_critical=is_critical)
            if output.arrows:
                self._add_visuals(annotation, candidate, important=judged or is_critical)
            return annotation

        level = output.annotations
        book_note = self._book_note(board_after, candidate)
        tactical = self._tactical_note(candidate, board_after, nag, level)

        # Importance decides whether this move may spend comment budget at all.
        important = bool(
            force_comment
            or extra_comment
            or judged
            or book_note
            or (tactical and (candidate.features.is_sacrifice or candidate.features.forks or is_critical))
        )
        spacing = self._COMMENT_SPACING.get(level, 3)
        recently_commented = (ply - self._last_comment_ply) < spacing
        if not important and (recently_commented or level in ("none", "minimal")):
            self._maybe_add_eval(annotation, candidate, engine_available, is_critical=is_critical)
            if output.arrows and judged:
                self._add_visuals(annotation, candidate, important=False)
            return annotation

        parts: List[str] = []
        if extra_comment:
            parts.append(extra_comment.strip())
        if book_note:
            parts.append(book_note)
        if tactical:
            parts.append(tactical)
        if level == "rich" and len(parts) < 3:
            plan = self._plan_note(board_after, candidate, level)
            if plan:
                parts.append(plan)
        if judged and reason and level in ("standard", "rich"):
            parts.append(f"({reason})")

        annotation.comment_parts = [p for p in parts if p][:4]
        if annotation.comment_parts:
            self._last_comment_ply = ply
        self._maybe_add_eval(annotation, candidate, engine_available, is_critical=is_critical)
        if output.arrows:
            self._add_visuals(annotation, candidate, important=important)
        return annotation

    def _book_note(self, board_after: chess.Board, candidate: ScoredMove) -> Optional[str]:
        """Name the resulting position, but only when the name is genuinely new.

        Successive book entries often repeat the family ("Italian Game: Two Knights
        Defense" -> "... Knight Attack"), so only the part that changed is printed.
        """
        theory = candidate.theory
        entry: Optional[BookEntry] = theory.entry if theory else None
        if entry is None:
            match = self.book.classify(board_after)
            entry = match.entry if match and match.exact else None
        if entry is None:
            return None
        label = f"{entry.name} ({entry.eco})"
        if label == self._last_book_label:
            return None
        previous = self._last_book_label
        self._last_book_label = label
        if previous:
            previous_name = previous.rsplit(" (", 1)[0]
            if entry.name.startswith(previous_name + ","):
                tail = entry.name[len(previous_name) + 1:].strip()
                if tail:
                    return f"{tail} ({entry.eco})"
        return label

    def _tactical_note(
        self,
        candidate: ScoredMove,
        board_after: chess.Board,
        nag: Optional[str],
        level: str,
    ) -> Optional[str]:
        """One sentence describing the concrete point of the move.

        Only motifs that carry real information are reported; "developing the
        knight" and "with check" are visible in the move itself. Structural
        observations (an opening file) are reserved for ``rich`` output and are
        stated once per file, since a recapture opens the same file every time.
        """
        features = candidate.features
        motifs = [m for m in features.motifs if m.kind in _TACTICAL_MOTIFS]
        if not motifs and level == "rich":
            motifs = [
                m
                for m in features.motifs
                if m.kind in _STRUCTURAL_MOTIFS and self._is_new_observation(m)
            ]
        if not motifs:
            return None
        lead = motifs[0]
        colour = "White" if features.color == chess.WHITE else "Black"

        if lead.kind == "mate":
            return "Checkmate."
        if lead.kind == "material_investment":
            if nag in ("?", "??"):
                return f"{colour} {lead.text}, but the engine finds no compensation."
            if nag in ("!", "!!"):
                return f"{colour} {lead.text} for the initiative."
            return f"{colour} {lead.text}."
        if lead.kind == "hanging":
            return lead.text[0].upper() + lead.text[1:] + "."
        sentence = lead.text
        if len(motifs) > 1 and motifs[1].kind != lead.kind:
            sentence = f"{sentence}, {motifs[1].text}"
        return sentence[0].upper() + sentence[1:] + "."

    def _is_new_observation(self, motif: Motif) -> bool:
        """True the first time a given structural observation is made."""
        key = (motif.kind, motif.text)
        if key in self._stated_observations:
            return False
        self._stated_observations.add(key)
        return True

    def _plan_note(self, board_after: chess.Board, candidate: ScoredMove, level: str) -> Optional[str]:
        """Strategic context: structure, king safety, plans.

        Each observation is made once per line. Repeating "Black fights for the
        centre" every second move is noise, not instruction.
        """
        features = candidate.features
        position = analyse_position(board_after)
        colour = "White" if features.color == chess.WHITE else "Black"
        side_key = "white" if features.color == chess.WHITE else "black"
        structure = position.structure[side_key]

        candidates: List[Tuple[str, str]] = []
        if features.is_castling:
            candidates.append((f"castled:{side_key}", f"{colour} tucks the king away and connects the rooks."))
        if features.central_pawn_move:
            candidates.append(
                (
                    f"centre:{side_key}",
                    f"{colour} fights for the centre; the resulting structure defines both sides' plans.",
                )
            )
        if position.iqp[side_key]:
            candidates.append(
                (
                    f"iqp:{side_key}",
                    f"{colour} accepts an isolated d-pawn: piece activity and the d5 square in "
                    "exchange for a long-term structural weakness.",
                )
            )
        if structure.isolated:
            squares = [square_name(s) for s in structure.isolated[:2]]
            if len(squares) == 1:
                text = f"{colour}'s pawn on {squares[0]} is isolated and will need piece support."
            else:
                text = (
                    f"{colour}'s pawns on {' and '.join(squares)} are isolated and will need "
                    "piece support."
                )
            candidates.append((f"isolated:{side_key}:{','.join(squares)}", text))
        if level == "rich" and features.develops_piece:
            candidates.append(
                (f"develops:{side_key}", f"{colour} develops with tempo and prepares to castle.")
            )
        if level == "rich" and position.open_files:
            files = ", ".join(chess.FILE_NAMES[f] for f in position.open_files[:2])
            candidates.append(
                (
                    f"open:{files}",
                    f"The {files}-file is open, so rook play becomes the main strategic resource.",
                )
            )
        if (
            level == "rich"
            and not position.castled[side_key]
            and len(board_after.move_stack) >= 10
        ):
            candidates.append(
                (
                    f"king-centre:{side_key}",
                    f"{colour}'s king is still in the centre, which makes the coming exchanges sharper.",
                )
            )
        if level == "rich":
            development = position.development
            if abs(development["white"] - development["black"]) >= 2:
                leader = "White" if development["white"] > development["black"] else "Black"
                candidates.append(
                    (
                        f"development-lead:{leader}",
                        f"{leader} is ahead in development and should look to open the position.",
                    )
                )

        for key, text in candidates:
            if key in self._stated_plans:
                continue
            self._stated_plans.add(key)
            return text
        return None

    def _maybe_add_eval(
        self,
        annotation: MoveAnnotation,
        candidate: ScoredMove,
        engine_available: bool,
        *,
        is_critical: bool,
    ) -> None:
        """Attach an engine evaluation, but only a real one."""
        mode = self.config.output.evals
        if mode == "none" or not engine_available:
            return
        score = candidate.score_after
        if score is None:
            return
        if mode == "critical" and not (
            is_critical
            or annotation.nag in ("!", "!!", "?", "??", "?!", "!?")
            or candidate.features.is_sacrifice
            or candidate.features.is_checkmate
        ):
            return
        text = format_score(score, self.config, prefix=self.config.output.eval_format == "verbose")
        if self.config.output.eval_format == "pawns":
            annotation.eval_text = f"{eval_symbol(score)} {text}"
        else:
            annotation.eval_text = text

    def _add_visuals(
        self,
        annotation: MoveAnnotation,
        candidate: ScoredMove,
        *,
        important: bool,
    ) -> None:
        """Add arrows/highlights for the leading motif only.

        Green arrows show the idea being executed, red squares mark targets and
        weaknesses, yellow marks squares of structural interest. Capped at three of
        each so viewers stay readable.
        """
        features = candidate.features
        threshold = 1.2 if important else 2.0
        motifs = [m for m in features.motifs if m.weight >= threshold]
        if not motifs:
            return
        arrows: List[Tuple[str, int, int]] = []
        squares: List[Tuple[str, int]] = []
        for motif in motifs[:2]:
            arrow_colour = "R" if motif.kind == "hanging" else "G"
            square_colour = "R" if motif.kind in _RED_SQUARE_MOTIFS else "Y"
            for src, dst in motif.arrows:
                if src is not None and dst is not None:
                    arrows.append((arrow_colour, src, dst))
            for square in motif.squares:
                if square is not None:
                    squares.append((square_colour, square))
        annotation.arrows = arrows[:3]
        annotation.highlights = squares[:3]

    # -- opening / summary comments ---------------------------------------- #

    def opening_intro(
        self,
        board: chess.Board,
        entry: Optional[BookEntry],
        request_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Comment placed before the first move."""
        pieces: List[str] = []
        if request_summary:
            pieces.append(request_summary.strip().rstrip("."))
        if entry is not None:
            pieces.append(f"{entry.name} ({entry.eco})")
        return ". ".join(p for p in pieces if p) + "." if pieces else None

    def closing_note(
        self,
        board: chess.Board,
        *,
        final_score: Optional[Score],
        entry: Optional[BookEntry],
        engine_available: bool,
        reason: str,
    ) -> str:
        """Comment placed on the last move, summarising the position reached."""
        parts: List[str] = []
        if entry is not None:
            parts.append(f"Position reached: {entry.name} ({entry.eco})")
        if final_score is not None and engine_available:
            symbol = eval_symbol(final_score)
            parts.append(f"engine assessment {symbol} {format_score(final_score, self.config, prefix=False)}")
        if reason:
            parts.append(reason)
        text = "; ".join(parts)
        return text[0].upper() + text[1:] + "." if text else ""

    def variation_intro(self, purpose: str) -> str:
        """Starting comment that states why a sideline exists."""
        return purpose.strip().rstrip(".") + "."
