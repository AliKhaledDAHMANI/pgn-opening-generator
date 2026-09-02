"""Move selection: theory + engine + style, blended into a GM-like choice.

The selector never takes the engine's top move blindly, and never plays a theory
move the engine rejects. For every position it:

1. builds a candidate set from the ECO book and the engine's MultiPV list;
2. rejects anything that loses more than the configured centipawn tolerance;
3. scores survivors on engine quality, theory breadth, style fit and
   practicality, weighted per style/mode;
4. returns the winner together with the reasons, so the annotator can explain it.

Without an engine, selection falls back to theory-only (and the caller must
report that no engine validation happened).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess

from .book import BookEntry, OpeningBook, TheoryMove
from .config import Config
from .engine import Analysis, Candidate, EngineManager, Score
from .features import MoveFeatures, analyse_move, static_exchange_eval

#: Style weights over the scoring components. Every component is normalised to
#: roughly 0..1 before weighting.
STYLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    #                     engine  theory  attack  solid  tactics  practical
    "classical_gm":      {"engine": 1.00, "theory": 0.85, "attack": 0.15, "solid": 0.35, "tactics": 0.15, "practical": 0.45},
    "sharp_tactical":    {"engine": 0.80, "theory": 0.45, "attack": 0.75, "solid": 0.00, "tactics": 0.95, "practical": 0.20},
    "aggressive":        {"engine": 0.75, "theory": 0.40, "attack": 1.00, "solid": 0.00, "tactics": 0.60, "practical": 0.20},
    "solid":             {"engine": 0.95, "theory": 0.80, "attack": 0.00, "solid": 0.95, "tactics": 0.00, "practical": 0.55},
    "positional":        {"engine": 0.95, "theory": 0.70, "attack": 0.10, "solid": 0.70, "tactics": 0.05, "practical": 0.45},
    "gambit":            {"engine": 0.55, "theory": 0.55, "attack": 0.80, "solid": 0.00, "tactics": 0.65, "practical": 0.15},
    "practical":         {"engine": 0.85, "theory": 0.75, "attack": 0.20, "solid": 0.40, "tactics": 0.15, "practical": 1.00},
    "engine_best":       {"engine": 1.00, "theory": 0.00, "attack": 0.00, "solid": 0.00, "tactics": 0.00, "practical": 0.00},
    "theoretical":       {"engine": 0.80, "theory": 1.00, "attack": 0.10, "solid": 0.25, "tactics": 0.10, "practical": 0.40},
}

#: Extra centipawn tolerance granted to gambit-flavoured styles, since an
#: objectively "worse" move is the whole point of a gambit.
STYLE_TOLERANCE_BONUS: Dict[str, int] = {
    "gambit": 45,
    "sharp_tactical": 25,
    "aggressive": 25,
    "classical_gm": 0,
    "solid": -10,
    "positional": 0,
    "practical": 0,
    "engine_best": -1000,   # forces engine top move
    "theoretical": 10,
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class ScoredMove:
    """A candidate move with its scoring breakdown."""

    move: chess.Move
    san: str
    features: MoveFeatures
    #: ``theory``, ``engine`` or ``both``.
    source: str
    theory: Optional[TheoryMove] = None
    engine: Optional[Candidate] = None
    #: Score after the move, from White's point of view (engine, may be ``None``).
    score_after: Optional[Score] = None
    cp_loss: Optional[int] = None
    components: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    reasons: List[str] = field(default_factory=list)
    rejected: Optional[str] = None

    @property
    def in_book(self) -> bool:
        return self.theory is not None

    @property
    def engine_rank(self) -> Optional[int]:
        return self.engine.rank if self.engine else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.move.uci(),
            "source": self.source,
            "cp_loss": self.cp_loss,
            "engine_rank": self.engine_rank,
            "eval_cp": self.score_after.cp if self.score_after else None,
            "book_name": self.theory.entry.name if self.theory and self.theory.entry else None,
            "score": round(self.total, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "reasons": list(self.reasons),
            "rejected": self.rejected,
        }


@dataclass
class SelectionResult:
    """Everything the annotator needs about one selected move."""

    chosen: ScoredMove
    alternatives: List[ScoredMove]
    analysis: Optional[Analysis]
    critical: bool
    engine_available: bool
    #: Book classification of the position *before* the move, if any.
    book_entry: Optional[BookEntry] = None

    @property
    def move(self) -> chess.Move:
        return self.chosen.move

    def top_alternatives(self, count: int) -> List[ScoredMove]:
        return [alt for alt in self.alternatives if alt.rejected is None][:count]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def position_is_critical(board: chess.Board, *, previous: Optional[MoveFeatures] = None) -> bool:
    """Heuristic: does this position deserve the deeper engine settings?

    True when the position is forcing or unbalanced enough that a shallow search
    could mislead: checks, heavy capture tension, a piece just invested, or a
    mating threat in the air.
    """
    if board.is_check():
        return True
    captures = sum(1 for _ in board.generate_legal_captures())
    if captures >= 3:
        return True
    if previous is not None:
        if previous.is_sacrifice or previous.is_promotion or previous.forks:
            return True
        if previous.gives_check or previous.creates_threat_of_mate:
            return True
        if previous.king_zone_pressure >= 2:
            return True
    return False


def _attack_component(features: MoveFeatures, board: chess.Board) -> float:
    """How much the move increases pressure on the enemy king / initiative."""
    score = 0.0
    if features.gives_check:
        score += 0.45
    if features.is_checkmate:
        score += 1.0
    score += min(0.35, 0.12 * features.king_zone_pressure)
    if features.attacks_king_zone:
        score += 0.12
    if features.is_capture and features.see_cp >= 0:
        score += 0.10
    if features.attacks_higher_value:
        score += 0.12
    if features.creates_threat_of_mate:
        score += 0.35
    # Pawn storms towards the enemy king.
    if features.is_pawn_move:
        rank = chess.square_rank(features.to_square)
        advanced = rank >= 4 if features.color == chess.WHITE else rank <= 3
        enemy_king = board.king(not features.color)
        if advanced and enemy_king is not None:
            if abs(chess.square_file(features.to_square) - chess.square_file(enemy_king)) <= 2:
                score += 0.22
    return _clamp(score)


def _tactics_component(features: MoveFeatures) -> float:
    score = 0.0
    if features.forks:
        score += 0.45
    if features.creates_pin:
        score += 0.25
    if features.discovered_attack:
        score += 0.25
    if features.is_sacrifice:
        score += 0.40
    if features.is_promotion:
        score += 0.20
    if features.is_capture and features.captures_defended_piece:
        score += 0.10
    return _clamp(score)


def _solid_component(features: MoveFeatures, board: chess.Board) -> float:
    score = 0.0
    if features.is_castling:
        score += 0.55
    if features.develops_piece:
        score += 0.30
    if features.is_quiet:
        score += 0.15
    if features.central_pawn_move:
        score += 0.15
    if features.is_sacrifice:
        score -= 0.60
    if features.leaves_hanging:
        score -= 0.35
    # Structural care: avoid loosening the pawns in front of one's own king.
    if features.is_pawn_move:
        king = board.king(features.color)
        if king is not None and abs(chess.square_file(features.to_square) - chess.square_file(king)) <= 1:
            home_rank = 1 if features.color == chess.WHITE else 6
            if chess.square_rank(features.from_square) == home_rank:
                score -= 0.20
    return _clamp(score, -1.0, 1.0)


def _practical_component(candidate: ScoredMove, board: chess.Board) -> float:
    """Ease of play: known theory, no immediate need to walk a tightrope."""
    score = 0.0
    if candidate.theory is not None:
        score += 0.35 + _clamp(candidate.theory.breadth / 120.0) * 0.35
    features = candidate.features
    if features.is_sacrifice:
        score -= 0.25
    if features.forced:
        score += 0.10
    if features.is_castling or features.develops_piece:
        score += 0.15
    if candidate.cp_loss is not None:
        score += _clamp(1.0 - candidate.cp_loss / 60.0) * 0.20
    return _clamp(score, -1.0, 1.0)


def _gambit_bonus(features: MoveFeatures, style: str) -> float:
    """Reward offering a pawn (not a piece) when the style asks for gambits."""
    if style not in ("gambit", "aggressive", "sharp_tactical"):
        return 0.0
    if not features.is_sacrifice:
        return 0.0
    if 90 <= features.invested_cp <= 200:
        return 0.35 if style == "gambit" else 0.20
    return 0.10 if style == "gambit" else 0.0


def _focus_bonus(features: MoveFeatures, board: chess.Board, focus_side: Optional[str]) -> float:
    """Nudge play towards a requested wing ("attacks the kingside")."""
    if not focus_side:
        return 0.0
    file_index = chess.square_file(features.to_square)
    if focus_side == "kingside":
        on_wing = file_index >= 4
    elif focus_side == "queenside":
        on_wing = file_index <= 3
    else:
        return 0.0
    if not on_wing:
        return 0.0
    bonus = 0.10
    enemy_king = board.king(not features.color)
    if enemy_king is not None:
        enemy_wing = chess.square_file(enemy_king) >= 4 if focus_side == "kingside" else chess.square_file(enemy_king) <= 3
        if enemy_wing and (features.attacks_king_zone or features.is_pawn_move):
            bonus += 0.20
    return bonus


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #


class MoveSelector:
    """Blends book theory, Stockfish output and style into one move choice."""

    def __init__(
        self,
        config: Config,
        book: OpeningBook,
        engine: EngineManager,
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config
        self.book = book
        self.engine = engine
        self.rng = rng or random.Random(config.seed)

    # -- public API -------------------------------------------------------- #

    def select(
        self,
        board: chess.Board,
        *,
        previous: Optional[MoveFeatures] = None,
        critical: Optional[bool] = None,
        style: Optional[str] = None,
        force_engine_best: bool = False,
        exclude: Sequence[chess.Move] = (),
        must_be_theory: bool = False,
        prefer_moves: Sequence[chess.Move] = (),
        focus_side: Optional[str] = None,
        max_probes: int = 3,
    ) -> Optional[SelectionResult]:
        """Pick the next move for ``board``.

        ``style`` overrides the configured style (used for the opponent's side in
        repertoire mode). ``prefer_moves`` gives a strong bonus to specific moves,
        which is how requested openings are steered without breaking legality.
        ``focus_side`` ("kingside"/"queenside") nudges play towards one wing.
        ``max_probes`` caps how many extra single-move searches are spent on
        candidates that fell outside the MultiPV window.
        Returns ``None`` for terminal positions or when nothing survives filtering.
        """
        if board.is_game_over(claim_draw=False):
            return None

        effective_style = style or self.config.style
        is_critical = position_is_critical(board, previous=previous) if critical is None else critical
        analysis = self.engine.analyse(board, critical=is_critical)
        theory = self.book.theory(board) if self.config.prefer_theory or must_be_theory else []
        theory_by_move = {t.move: t for t in theory}

        candidates = self._build_candidates(
            board,
            analysis=analysis,
            theory_by_move=theory_by_move,
            exclude=set(exclude),
            must_be_theory=must_be_theory,
        )
        if not candidates:
            return None

        if analysis is not None:
            self._probe_missing(board, candidates, analysis, max_probes=max_probes)

        max_breadth = max((c.theory.breadth for c in candidates if c.theory), default=1)
        tolerance = self._tolerance(effective_style)

        for candidate in candidates:
            self._score_candidate(
                candidate,
                board=board,
                style=effective_style,
                analysis=analysis,
                max_breadth=max_breadth,
                tolerance=tolerance,
                force_engine_best=force_engine_best,
                prefer_moves=set(prefer_moves),
                focus_side=focus_side,
            )

        survivors = [c for c in candidates if c.rejected is None]
        if not survivors:
            # Everything failed the quality filter: fall back to the objectively
            # best move so output stays sound rather than stylish.
            survivors = sorted(candidates, key=lambda c: (c.cp_loss if c.cp_loss is not None else 10**6))[:1]
            if survivors:
                survivors[0].rejected = None
                survivors[0].reasons.append("selected as the safest available move (style filter empty)")

        survivors.sort(key=lambda c: (-c.total, c.cp_loss if c.cp_loss is not None else 10**6, c.san))
        chosen = survivors[0]
        others = [c for c in candidates if c is not chosen]
        others.sort(key=lambda c: (-c.total, c.san))

        book_match = self.book.classify(board)
        return SelectionResult(
            chosen=chosen,
            alternatives=others,
            analysis=analysis,
            critical=is_critical,
            engine_available=analysis is not None,
            book_entry=book_match.entry if book_match else None,
        )

    def _probe_missing(
        self,
        board: chess.Board,
        candidates: Sequence[ScoredMove],
        analysis: Analysis,
        *,
        max_probes: int,
    ) -> None:
        """Search individual candidates that MultiPV did not cover.

        Bounded by ``max_probes`` because each probe is a full search. Book moves
        with the widest theory are probed first, since those are the ones most
        likely to be chosen.
        """
        missing = [c for c in candidates if c.score_after is None]
        missing.sort(key=lambda c: -(c.theory.breadth if c.theory else 0))
        for candidate in missing[:max_probes]:
            self._attach_engine_numbers(board, candidate, analysis, critical=analysis.critical)

    def score_specific(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        critical: bool = False,
    ) -> ScoredMove:
        """Build a :class:`ScoredMove` for one given move (no style filtering)."""
        analysis = self.engine.analyse(board, critical=critical)
        theory = {t.move: t for t in self.book.theory(board)}
        candidate = self._make_candidate(board, move, analysis, theory.get(move))
        if analysis is not None:
            self._attach_engine_numbers(board, candidate, analysis, critical=critical)
        return candidate

    # -- candidate construction -------------------------------------------- #

    def _build_candidates(
        self,
        board: chess.Board,
        *,
        analysis: Optional[Analysis],
        theory_by_move: Dict[chess.Move, TheoryMove],
        exclude: set,
        must_be_theory: bool,
    ) -> List[ScoredMove]:
        moves: List[chess.Move] = []
        if must_be_theory:
            moves = [m for m in theory_by_move if m not in exclude]
        else:
            for move in theory_by_move:
                if move not in exclude:
                    moves.append(move)
            if analysis is not None:
                for cand in analysis.candidates:
                    if cand.move not in moves and cand.move not in exclude:
                        moves.append(cand.move)
            elif not moves:
                # No engine and no theory: nothing defensible to choose from.
                return []

        candidates: List[ScoredMove] = []
        for move in moves:
            if move not in board.legal_moves:  # pragma: no cover - book artefact
                continue
            candidate = self._make_candidate(board, move, analysis, theory_by_move.get(move))
            candidates.append(candidate)
        return candidates

    def _make_candidate(
        self,
        board: chess.Board,
        move: chess.Move,
        analysis: Optional[Analysis],
        theory: Optional[TheoryMove],
    ) -> ScoredMove:
        features = analyse_move(board, move)
        engine_candidate = analysis.by_move(move) if analysis else None
        source = "both" if theory is not None and engine_candidate is not None else (
            "theory" if theory is not None else "engine"
        )
        candidate = ScoredMove(
            move=move,
            san=features.san,
            features=features,
            source=source,
            theory=theory,
            engine=engine_candidate,
        )
        if engine_candidate is not None:
            candidate.score_after = engine_candidate.score
            candidate.cp_loss = analysis.cp_loss(move) if analysis else None
        return candidate

    def _attach_engine_numbers(
        self, board: chess.Board, candidate: ScoredMove, analysis: Analysis, *, critical: bool
    ) -> None:
        """Fill in eval/cp-loss for a move outside the MultiPV window."""
        if candidate.score_after is not None:
            return
        score = self.engine.score_move(board, candidate.move, critical=critical)
        if score is None:
            return
        candidate.score_after = score
        best = analysis.best_score
        if best is not None:
            candidate.cp_loss = max(0, best.cp_for(analysis.turn) - score.cp_for(analysis.turn))

    # -- scoring ------------------------------------------------------------ #

    def _tolerance(self, style: str) -> int:
        base = self.config.style_cp_tolerance
        bonus = STYLE_TOLERANCE_BONUS.get(style, 0)
        aggression_bonus = int(round(30 * (self.config.aggressiveness - 0.5) * 2)) if style != "engine_best" else 0
        return max(0, base + bonus + max(0, aggression_bonus))

    def _score_candidate(
        self,
        candidate: ScoredMove,
        *,
        board: chess.Board,
        style: str,
        analysis: Optional[Analysis],
        max_breadth: int,
        tolerance: int,
        force_engine_best: bool,
        prefer_moves: set,
        focus_side: Optional[str] = None,
    ) -> None:
        weights = STYLE_WEIGHTS.get(style, STYLE_WEIGHTS["classical_gm"])
        features = candidate.features

        allowance = tolerance
        if candidate.in_book:
            allowance = max(allowance, self.config.theory_cp_tolerance)
        if candidate.features.is_checkmate:
            allowance = 10**6

        if candidate.cp_loss is not None and candidate.cp_loss > allowance:
            candidate.rejected = f"loses {candidate.cp_loss} cp (limit {allowance})"
        if force_engine_best and candidate.engine_rank not in (None, 1):
            candidate.rejected = "not the engine's first choice"

        engine_component = 0.0
        if candidate.cp_loss is not None:
            engine_component = _clamp(1.0 - candidate.cp_loss / 100.0)
        elif candidate.engine_rank is not None:
            engine_component = _clamp(1.0 - 0.15 * (candidate.engine_rank - 1))

        theory_component = 0.0
        if candidate.theory is not None:
            breadth = candidate.theory.breadth
            theory_component = 0.45 + 0.55 * _clamp(breadth / max(1, max_breadth))
            if candidate.theory.entry is not None:
                theory_component = min(1.0, theory_component + 0.08)

        attack_component = _attack_component(features, board)
        tactics_component = _tactics_component(features)
        solid_component = _solid_component(features, board)
        practical_component = _practical_component(candidate, board)

        aggression = self.config.aggressiveness
        attack_component *= 0.6 + 0.8 * aggression
        solid_component *= 1.4 - 0.8 * aggression

        components = {
            "engine": engine_component,
            "theory": theory_component,
            "attack": attack_component,
            "solid": solid_component,
            "tactics": tactics_component,
            "practical": practical_component,
        }
        total = sum(weights.get(key, 0.0) * value for key, value in components.items())
        total += _gambit_bonus(features, style)
        total += _focus_bonus(features, board, focus_side)
        if candidate.move in prefer_moves:
            total += 1.5
            candidate.reasons.append("matches the requested line")
        if candidate.engine_rank == 1:
            total += 0.10
        if features.forced:
            total += 0.50

        candidate.components = components
        candidate.total = total
        candidate.reasons.extend(self._reasons(candidate, style))

    def _reasons(self, candidate: ScoredMove, style: str) -> List[str]:
        out: List[str] = []
        theory = candidate.theory
        if theory is not None:
            if theory.entry is not None:
                out.append(f"book line: {theory.entry.name} ({theory.entry.eco})")
            else:
                out.append(f"stays in the ECO book ({theory.breadth} catalogued continuations)")
        if candidate.engine_rank == 1:
            out.append("engine's first choice")
        elif candidate.engine_rank is not None:
            out.append(f"engine's #{candidate.engine_rank} choice")
        if candidate.cp_loss is not None and candidate.cp_loss > 0:
            out.append(f"{candidate.cp_loss} cp behind the top move")
        features = candidate.features
        if features.is_sacrifice:
            out.append(f"invests {features.invested_cp} cp of material")
        if features.gives_check:
            out.append("forcing (check)")
        if features.forks:
            out.append("creates a double attack")
        if style in ("aggressive", "sharp_tactical", "gambit") and features.attacks_king_zone:
            out.append("increases pressure near the enemy king")
        return out


# --------------------------------------------------------------------------- #
# Trap search
# --------------------------------------------------------------------------- #


@dataclass
class TrapCandidate:
    """A tempting-but-losing reply plus its engine-verified refutation."""

    #: FEN of the position the bait is played from, so the generator can tell when
    #: the line has actually arrived at the trap.
    origin_fen: str
    bait: chess.Move                 # hero move that sets the trap
    bait_san: str
    victim: chess.Move               # tempting mistake
    victim_san: str
    refutation: chess.Move           # punishing move
    refutation_san: str
    cp_swing: int                    # centipawns the victim loses
    eval_after_refutation: Score
    victim_in_book: bool
    #: Why the mistake is tempting: ``book``, ``greedy`` (wins material by SEE) or
    #: ``shallow`` (near-best in a shallow search).
    temptation: str
    refutation_features: MoveFeatures
    followup: List[chess.Move] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "origin_fen": self.origin_fen,
            "bait": self.bait_san,
            "mistake": self.victim_san,
            "refutation": self.refutation_san,
            "cp_swing": self.cp_swing,
            "eval_after_refutation_cp": self.eval_after_refutation.cp,
            "mistake_in_book": self.victim_in_book,
            "temptation": self.temptation,
        }


class TrapFinder:
    """Finds engine-validated opening traps.

    A trap is a concrete triple: the hero plays ``bait``, the opponent answers with
    a move that *looks natural* but is bad, and a refutation punishes it. All three
    conditions are checked mechanically, so nothing is reported on vibes:

    * **Sound bait** - the bait itself may lose no more than ``bait_tolerance``
      centipawns. A losing bait would make the "trap" a gift.
    * **Natural mistake** - the reply must be one a human would actually play: a
      book move (``book``), a capture that wins material by static exchange
      (``greedy``), or a near-best move in a deliberately shallow search
      (``shallow``, i.e. what the position looks like at a glance).
    * **Verified punishment** - the refutation must swing the deep evaluation by at
      least ``min_cp_swing`` centipawns. Forcing punishments (check, capture,
      sacrifice, fork) qualify at that threshold; a quiet punishment must clear the
      higher ``min_cp_swing_quiet`` bar, because "you are simply worse now" is only
      a trap when the resulting advantage is decisive.
    """

    def __init__(
        self,
        config: Config,
        book: OpeningBook,
        engine: EngineManager,
        *,
        min_cp_swing: int = 120,
        min_cp_swing_quiet: int = 220,
        max_baits: int = 4,
        max_replies: int = 4,
        shallow_depth: int = 5,
        shallow_tolerance: int = 40,
        bait_tolerance: int = 70,
    ) -> None:
        self.config = config
        self.book = book
        self.engine = engine
        self.min_cp_swing = min_cp_swing
        self.min_cp_swing_quiet = max(min_cp_swing, min_cp_swing_quiet)
        self.max_baits = max_baits
        self.max_replies = max_replies
        self.shallow_depth = shallow_depth
        self.shallow_tolerance = shallow_tolerance
        self.bait_tolerance = bait_tolerance

    def find(
        self,
        board: chess.Board,
        hero: chess.Color,
        *,
        bait_moves: Optional[Sequence[chess.Move]] = None,
    ) -> Optional[TrapCandidate]:
        """Search for the best trap with ``hero`` to move in ``board``."""
        traps = self.find_all(board, hero, bait_moves=bait_moves)
        return traps[0] if traps else None

    def find_all(
        self,
        board: chess.Board,
        hero: chess.Color,
        *,
        bait_moves: Optional[Sequence[chess.Move]] = None,
        limit: int = 3,
    ) -> List[TrapCandidate]:
        """All traps found, best first."""
        if board.turn != hero or board.is_game_over(claim_draw=False):
            return []
        if not self.engine.available:
            return []

        baits: List[Tuple[chess.Move, Optional[int]]]
        baits = [(m, None) for m in bait_moves] if bait_moves else self._bait_candidates(board)

        found: List[TrapCandidate] = []
        for bait, bait_loss in baits[: self.max_baits]:
            if bait not in board.legal_moves:
                continue
            if bait_loss is not None and bait_loss > self.bait_tolerance:
                continue
            probe = board.copy(stack=False)
            probe.push(bait)
            if probe.is_game_over(claim_draw=False):
                continue
            baseline = self.engine.analyse(probe, critical=True)
            if baseline is None or not baseline.candidates:
                continue
            best_reply_cp = baseline.candidates[0].score.cp_for(probe.turn)
            book_replies = [t.move for t in self.book.theory(probe)]

            for reply, temptation in self._tempting_replies(probe, book_replies):
                after = probe.copy(stack=False)
                after.push(reply)
                punished = self.engine.analyse(after, critical=True)
                if punished is None or not punished.candidates:
                    continue
                reply_cp = -punished.candidates[0].score.cp_for(after.turn)
                swing = best_reply_cp - reply_cp
                if swing < self.min_cp_swing:
                    continue
                refutation = punished.candidates[0].move
                features = analyse_move(after, refutation)
                forcing = self._is_forcing(features)
                if not forcing and swing < self.min_cp_swing_quiet:
                    continue
                found.append(
                    TrapCandidate(
                        origin_fen=board.fen(),
                        bait=bait,
                        bait_san=board.san(bait),
                        victim=reply,
                        victim_san=probe.san(reply),
                        refutation=refutation,
                        refutation_san=features.san,
                        cp_swing=int(swing),
                        eval_after_refutation=punished.candidates[0].score,
                        victim_in_book=reply in book_replies,
                        temptation=temptation,
                        refutation_features=features,
                        followup=list(punished.candidates[0].pv[:6]),
                    )
                )
        found.sort(key=self._trap_key, reverse=True)
        return found[:limit]

    @staticmethod
    def _is_forcing(features: MoveFeatures) -> bool:
        """A punishment that visibly wins something, rather than a quiet edge."""
        return bool(
            features.gives_check
            or features.is_checkmate
            or features.is_sacrifice
            or features.forks
            or (features.is_capture and features.see_cp > 0)
        )

    def _trap_key(self, trap: TrapCandidate) -> Tuple[int, int, int]:
        """Prefer the most natural mistake, then the most spectacular punishment."""
        temptation_rank = {"book": 3, "greedy": 2, "shallow": 1}.get(trap.temptation, 0)
        spectacle = 0
        if trap.refutation_features.is_sacrifice:
            spectacle += 2
        if trap.refutation_features.gives_check:
            spectacle += 1
        if trap.refutation_features.forks:
            spectacle += 1
        return (temptation_rank, spectacle, trap.cp_swing)

    def _bait_candidates(self, board: chess.Board) -> List[Tuple[chess.Move, Optional[int]]]:
        """Sound moves worth trying as bait, book theory first."""
        analysis = self.engine.analyse(board, critical=False)
        out: List[Tuple[chess.Move, Optional[int]]] = []
        seen: set = set()
        for theory in self.book.theory(board, limit=6):
            loss = analysis.cp_loss(theory.move) if analysis else None
            if loss is None or loss <= self.bait_tolerance:
                out.append((theory.move, loss))
                seen.add(theory.move)
        if analysis is not None:
            for cand in analysis.candidates[:4]:
                if cand.move in seen:
                    continue
                loss = analysis.cp_loss(cand.move)
                if loss is None or loss <= self.bait_tolerance:
                    out.append((cand.move, loss))
                    seen.add(cand.move)
        return out

    def _tempting_replies(
        self, board: chess.Board, book_replies: Sequence[chess.Move]
    ) -> List[Tuple[chess.Move, str]]:
        """Replies a human would plausibly play, tagged with why."""
        tagged: Dict[chess.Move, str] = {}

        for move in book_replies:
            if move in board.legal_moves:
                tagged[move] = "book"

        # Greedy captures: static exchange says they win material, which is exactly
        # the bait most opening traps rely on.
        for move in board.generate_legal_captures():
            if move in tagged:
                continue
            if static_exchange_eval(board, move) > 0:
                tagged[move] = "greedy"

        shallow = self.engine.analyse(
            board, depth=self.shallow_depth, multipv=max(3, self.max_replies)
        )
        if shallow is not None and shallow.candidates:
            shallow_best = shallow.candidates[0].score.cp_for(shallow.turn)
            for cand in shallow.candidates:
                if cand.move in tagged:
                    continue
                if shallow_best - cand.score.cp_for(shallow.turn) <= self.shallow_tolerance:
                    tagged[cand.move] = "shallow"

        order = {"book": 0, "greedy": 1, "shallow": 2}
        items = sorted(tagged.items(), key=lambda kv: order.get(kv[1], 9))
        return items[: self.max_replies + 2]
