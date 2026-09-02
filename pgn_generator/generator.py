"""Line generation: the orchestration layer.

Flow for a single request:

1. Resolve the starting position (FEN / supplied moves / requested opening).
2. Walk forward, asking :class:`~pgn_generator.selector.MoveSelector` for each
   move, until the opening has been demonstrated or the move budget runs out.
3. Attach annotations, comments, evaluations and arrows via
   :class:`~pgn_generator.annotate.Annotator`.
4. Add sidelines (engine alternatives, book theory, refutations, traps).
5. Serialise to PGN and run the full validation pipeline.

Nothing is returned unless validation passes. The mode-specific behaviour
(``gm``/``engine``/``training``/``trap``/``repertoire``) lives in
:class:`OpeningGenerator`, but the chess work is always delegated to the selector
and the engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

from .annotate import Annotator, MoveAnnotation
from .book import BookEntry, OpeningBook, get_book
from .config import Config, build_config
from .engine import EngineManager, Score
from .errors import GenerationError, ValidationFailedError
from .features import MoveFeatures
from .pgn import LineData, MoveRecord, VariationRecord, line_to_pgn
from .request import Request, RequestParser
from .selector import MoveSelector, ScoredMove, SelectionResult, TrapCandidate, TrapFinder
from .validate import ValidationReport, validate

#: How many positions along the line the trap search may examine before giving up.
_TRAP_ATTEMPTS = 6

#: Minimum plies before ``stop_when_out_of_theory`` is allowed to end the line.
_MIN_PLIES_BEFORE_EARLY_STOP = 8


@dataclass
class GenerationResult:
    """Everything the caller gets back from a successful generation."""

    pgn: str
    line: LineData
    config: Config
    request: Request
    report: ValidationReport
    engine_info: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    #: Per-move trace: chosen move, engine numbers, reasons, alternatives.
    trace: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    opening: Optional[BookEntry] = None
    #: Traps found in ``trap`` mode.
    traps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_trace: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pgn": self.pgn,
            "opening": self.opening.to_dict() if self.opening else None,
            "mode": self.config.mode,
            "style": self.config.style,
            "start_fen": self.line.start_fen,
            "main_line": self.line.san_line(),
            "plies": len(self.line.moves),
            "variations": sum(len(record.variations) for record in self.line.moves),
            "stop_reason": self.stop_reason,
            "engine": self.engine_info,
            "validation": self.report.to_dict(),
            "request": self.request.to_dict(),
            "warnings": list(self.warnings),
        }
        if self.traps:
            payload["traps"] = self.traps
        if include_trace:
            payload["trace"] = self.trace
        return payload


class OpeningGenerator:
    """Generates validated, annotated opening PGN from a parsed request."""

    def __init__(
        self,
        config: Config,
        *,
        book: Optional[OpeningBook] = None,
        engine: Optional[EngineManager] = None,
    ) -> None:
        self.config = config
        self.book = book or get_book()
        self._owns_engine = engine is None
        self.engine = engine or EngineManager(config.engine, deterministic=config.deterministic)
        self.selector = MoveSelector(config, self.book, self.engine, rng=random.Random(config.seed))
        self.annotator = Annotator(config, self.book, engine=self.engine)
        self.warnings: List[str] = []
        self._critical_fens: List[str] = []
        self._analysed_fens: List[str] = []
        self._trap_finder_cache: Optional[TrapFinder] = None

    # -- lifecycle --------------------------------------------------------- #

    def __enter__(self) -> "OpeningGenerator":
        if self._owns_engine:
            self.engine.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._owns_engine:
            self.engine.__exit__(*exc)

    # -- entry point ------------------------------------------------------- #

    def generate(self, request: Request) -> GenerationResult:
        """Produce a validated PGN for ``request``."""
        start_board, start_fen, prefix_records = self._starting_position(request)

        if self.config.mode == "repertoire":
            line, stop_reason, trace, traps = self._generate_repertoire(
                request, start_board, start_fen, prefix_records
            )
        else:
            line, stop_reason, trace, traps = self._generate_line(
                request, start_board, start_fen, prefix_records
            )

        pgn_text, _game = line_to_pgn(line, self.config)
        report = validate(
            line=line,
            pgn_text=pgn_text,
            config=self.config,
            book=self.book,
            engine_available=self.engine.available,
            analysed_positions=self.engine.positions_analysed,
            critical_positions=self._critical_fens,
            analysed_fens=self._analysed_fens,
            requested_fen=request.start_fen,
            requested_moves=request.start_moves,
            requested_eco=request.opening_entry.eco if request.opening_entry else None,
            requested_name=request.opening_entry.name if request.opening_entry else None,
            requested_uci=self._requested_uci(request),
        )
        warnings = list(self.warnings) + list(request.warnings) + list(self.engine.warnings)
        warnings.extend(finding.message for finding in report.warnings)

        if not report.ok:
            raise ValidationFailedError(
                "generated PGN failed validation",
                failures=[finding.to_dict() for finding in report.errors],
                pgn=pgn_text,
            )

        final_entry = self._classify_line(line)
        return GenerationResult(
            pgn=pgn_text,
            line=line,
            config=self.config,
            request=request,
            report=report,
            engine_info=self.engine.info(),
            warnings=_dedupe(warnings),
            trace=trace,
            stop_reason=stop_reason,
            opening=final_entry,
            traps=traps,
        )

    # -- starting position ------------------------------------------------- #

    def _starting_position(
        self, request: Request
    ) -> Tuple[chess.Board, str, List[MoveRecord]]:
        """Build the board and the fixed prefix the output must contain.

        Priority: an explicit FEN defines the root; supplied moves are replayed
        from it; otherwise, if an opening was requested and no moves were given,
        the canonical book line for that opening becomes the prefix so the output
        demonstrably *is* that opening.
        """
        root = chess.Board(request.start_fen) if request.start_fen else chess.Board()
        start_fen = root.fen()
        board = root.copy(stack=True)
        records: List[MoveRecord] = []

        tokens = list(request.start_moves)
        if not tokens and request.opening_entry is not None and not request.start_fen:
            entry = request.opening_entry
            replay = chess.Board()
            for uci in entry.uci:
                move = chess.Move.from_uci(uci)
                records.append(MoveRecord(move=move, san=replay.san(move)))
                replay.push(move)
            board = replay
            return board, start_fen, records

        for san in tokens:
            move = board.parse_san(san)
            records.append(MoveRecord(move=move, san=board.san(move)))
            board.push(move)
        return board, start_fen, records

    def _requested_uci(self, request: Request) -> List[str]:
        if request.start_moves_uci:
            return list(request.start_moves_uci)
        if request.opening_entry is not None and not request.start_fen:
            return list(request.opening_entry.uci)
        return []

    # -- main generation --------------------------------------------------- #

    def _generate_line(
        self,
        request: Request,
        board: chess.Board,
        start_fen: str,
        prefix: List[MoveRecord],
    ) -> Tuple[LineData, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        config = self.config
        records: List[MoveRecord] = list(prefix)
        trace: List[Dict[str, Any]] = []
        traps: List[Dict[str, Any]] = []
        selections: List[Optional[SelectionResult]] = [None] * len(records)
        focus = _focus_side(request)
        hero = self._hero_colour(request, board)

        target_plies = config.target_plies
        budget = max(0, target_plies - len(records))
        previous: Optional[MoveFeatures] = None
        stop_reason = "reached the requested length"

        trap: Optional[TrapCandidate] = None
        trap_attempts = 0
        hunting_traps = config.mode == "trap"

        for _step in range(budget):
            if board.is_game_over(claim_draw=False):
                stop_reason = "the game ended"
                break

            # Trap mode: look for a trap from each position where the hero is to
            # move, until one is found or the attempt budget runs out. Searching
            # along the line (rather than only at the root) is what lets traps be
            # found where they actually live - six or eight moves deep.
            if hunting_traps and trap is None and board.turn == hero and trap_attempts < _TRAP_ATTEMPTS:
                trap_attempts += 1
                trap = self._trap_finder().find(board, hero)

            forced_move: Optional[chess.Move] = None
            if trap is not None:
                forced_move = self._trap_move_for(trap, board)
            constraint = self._constraint_move(request, board)
            if constraint is not None:
                forced_move = constraint

            selection = self._select_move(
                board,
                previous=previous,
                hero=hero,
                focus=focus,
                forced_move=forced_move,
            )
            if selection is None:
                stop_reason = "no acceptable continuation was found"
                break

            candidate = selection.chosen
            if selection.critical:
                self._critical_fens.append(board.fen())
            if selection.analysis is not None:
                self._analysed_fens.append(board.fen())

            board_after = board.copy(stack=False)
            board_after.push(candidate.move)
            annotation = self.annotator.annotate_move(
                board_before=board,
                board_after=board_after,
                candidate=candidate,
                selection=selection,
                ply=len(records),
                engine_available=self.engine.available,
                is_critical=selection.critical,
                extra_comment=self._trap_comment(trap, candidate),
            )
            records.append(MoveRecord(move=candidate.move, san=candidate.san, annotation=annotation))
            selections.append(selection)
            trace.append(self._trace_entry(board, candidate, selection))
            previous = candidate.features
            board.push(candidate.move)

            if self._should_stop(board, records):
                stop_reason = "the opening has been demonstrated"
                break

        if hunting_traps and trap is None:
            self.warnings.append(
                "no engine-validated trap was found along this line; produced a sharp tactical "
                "line instead"
            )

        if not records:
            raise GenerationError("could not generate any move for this request")

        line = LineData(
            start_fen=start_fen,
            from_initial=start_fen == chess.Board().fen(),
            moves=records,
            headers=self._headers(records, start_fen, request),
        )
        self._attach_variations(line, selections, request, trap=trap)
        self._finalise_comments(line, request, board)
        if trap is not None:
            traps.append(trap.to_dict())
        return line, stop_reason, trace, traps

    def _select_move(
        self,
        board: chess.Board,
        *,
        previous: Optional[MoveFeatures],
        hero: chess.Color,
        focus: Optional[str],
        forced_move: Optional[chess.Move] = None,
    ) -> Optional[SelectionResult]:
        """Choose the next move, honouring a forced move when one is required.

        A forced move (a trap's bait, or a caller constraint like "against 1...e5")
        is played even if the style scoring would prefer something else - but it is
        still scored by the engine, so its annotation and evaluation stay honest.
        """
        selection = self.selector.select(
            board,
            previous=previous,
            style=self._style_for(board, hero),
            force_engine_best=self.config.mode == "engine",
            prefer_moves=[forced_move] if forced_move else (),
            focus_side=focus if board.turn == hero else None,
        )
        if selection is None or forced_move is None or selection.chosen.move == forced_move:
            return selection
        candidate = self.selector.score_specific(board, forced_move, critical=True)
        return SelectionResult(
            chosen=candidate,
            alternatives=[selection.chosen] + selection.alternatives,
            analysis=selection.analysis,
            critical=True,
            engine_available=selection.engine_available,
            book_entry=selection.book_entry,
        )

    def _trap_finder(self) -> TrapFinder:
        if self._trap_finder_cache is None:
            self._trap_finder_cache = TrapFinder(self.config, self.book, self.engine)
        return self._trap_finder_cache

    # -- repertoire -------------------------------------------------------- #

    def _generate_repertoire(
        self,
        request: Request,
        board: chess.Board,
        start_fen: str,
        prefix: List[MoveRecord],
    ) -> Tuple[LineData, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """One consistent line for the hero, with a branch per opponent reply.

        The hero's own moves are chosen once per position and reused, which is what
        makes the output a repertoire rather than a set of unrelated games.
        """
        config = self.config
        hero = self._hero_colour(request, board)
        records: List[MoveRecord] = list(prefix)
        selections: List[Optional[SelectionResult]] = [None] * len(records)
        trace: List[Dict[str, Any]] = []
        focus = _focus_side(request)

        target_plies = config.target_plies
        cursor = board.copy(stack=True)
        previous: Optional[MoveFeatures] = None
        branch_points: List[Tuple[int, chess.Board, List[ScoredMove]]] = []

        while len(records) < target_plies and not cursor.is_game_over(claim_draw=False):
            constrained = self._constraint_move(request, cursor)
            selection = self._select_move(
                cursor,
                previous=previous,
                hero=hero,
                focus=focus,
                forced_move=constrained,
            )
            if selection is None:
                break
            candidate = selection.chosen
            # Branch on the opponent's real choices only. A ply the caller pinned
            # ("against 1...e5") is not a choice, so branching there would answer a
            # question that was not asked.
            if cursor.turn != hero and constrained is None:
                alternatives = [
                    alt for alt in selection.top_alternatives(config.repertoire_branches)
                    if alt.move != candidate.move
                ]
                if alternatives:
                    branch_points.append((len(records), cursor.copy(stack=True), alternatives))

            if selection.critical:
                self._critical_fens.append(cursor.fen())
            if selection.analysis is not None:
                self._analysed_fens.append(cursor.fen())

            after = cursor.copy(stack=False)
            after.push(candidate.move)
            annotation = self.annotator.annotate_move(
                board_before=cursor,
                board_after=after,
                candidate=candidate,
                selection=selection,
                ply=len(records),
                engine_available=self.engine.available,
                is_critical=selection.critical,
            )
            records.append(MoveRecord(move=candidate.move, san=candidate.san, annotation=annotation))
            selections.append(selection)
            trace.append(self._trace_entry(cursor, candidate, selection))
            previous = candidate.features
            cursor.push(candidate.move)

        line = LineData(
            start_fen=start_fen,
            from_initial=start_fen == chess.Board().fen(),
            moves=records,
            headers=self._headers(records, start_fen, request),
        )

        # Build one branch per notable opponent reply, each answered by the hero.
        budget = config.repertoire_branches
        for ply, branch_board, alternatives in branch_points:
            if budget <= 0:
                break
            for alternative in alternatives[:1]:
                if budget <= 0:
                    break
                variation = self._build_repertoire_branch(
                    branch_board, alternative, hero, ply, focus
                )
                if variation is not None:
                    line.moves[ply].variations.append(variation)
                    budget -= 1

        self._finalise_comments(line, request, cursor)
        return line, "repertoire branches generated", trace, []

    def _build_repertoire_branch(
        self,
        board: chess.Board,
        first: ScoredMove,
        hero: chess.Color,
        ply: int,
        focus: Optional[str],
    ) -> Optional[VariationRecord]:
        """Answer one alternative opponent move with the hero's recipe."""
        start_fen = board.fen()
        cursor = board.copy(stack=True)
        records: List[MoveRecord] = []

        annotation = MoveAnnotation(san=first.san)
        if self.config.output.comments:
            annotation.comment_parts = [
                f"If {first.san}" + (f" ({first.theory.entry.name})" if first.theory and first.theory.entry else "")
            ]
        records.append(MoveRecord(move=first.move, san=first.san, annotation=annotation))
        cursor.push(first.move)

        depth = max(2, self.config.variation_plies)
        previous = first.features
        for _ in range(depth - 1):
            if cursor.is_game_over(claim_draw=False):
                break
            selection = self.selector.select(
                cursor,
                previous=previous,
                style=self._style_for(cursor, hero),
                focus_side=focus if cursor.turn == hero else None,
            )
            if selection is None:
                break
            candidate = selection.chosen
            after = cursor.copy(stack=False)
            after.push(candidate.move)
            note = self.annotator.annotate_move(
                board_before=cursor,
                board_after=after,
                candidate=candidate,
                selection=selection,
                ply=ply + len(records),
                engine_available=self.engine.available,
                is_critical=selection.critical,
            )
            records.append(MoveRecord(move=candidate.move, san=candidate.san, annotation=note))
            previous = candidate.features
            cursor.push(candidate.move)
            if selection.analysis is not None:
                self._analysed_fens.append(after.fen())

        if len(records) < 2:
            return None
        return VariationRecord(
            start_ply=ply,
            start_fen=start_fen,
            moves=records,
            purpose="Repertoire branch",
            kind="alternative",
        )

    # -- variations -------------------------------------------------------- #

    def _attach_variations(
        self,
        line: LineData,
        selections: Sequence[Optional[SelectionResult]],
        request: Request,
        *,
        trap: Optional[TrapCandidate],
    ) -> None:
        """Add sidelines with a stated purpose at the most interesting plies."""
        wanted = self.config.variations
        if wanted <= 0:
            return
        limit_ply = self.config.variation_max_ply or len(line.moves)

        scored: List[Tuple[float, int, SelectionResult]] = []
        for ply, selection in enumerate(selections):
            if selection is None or ply >= limit_ply:
                continue
            if not selection.top_alternatives(1):
                continue
            scored.append((self._branch_interest(selection), ply, selection))
        scored.sort(key=lambda item: (-item[0], item[1]))

        added = 0
        for _interest, ply, selection in scored:
            if added >= wanted:
                break
            alternative = self._pick_alternative(selection)
            if alternative is None:
                continue
            variation = self._build_variation(line, ply, alternative, selection)
            if variation is None:
                continue
            line.moves[ply].variations.append(variation)
            added += 1

        if trap is not None:
            self._attach_trap_variation(line, trap)

    def _branch_interest(self, selection: SelectionResult) -> float:
        """How much a branch at this ply would teach the reader."""
        score = 0.0
        chosen = selection.chosen
        alternatives = selection.top_alternatives(2)
        if not alternatives:
            return 0.0
        best_alt = alternatives[0]
        if best_alt.theory is not None:
            score += 1.0 + min(1.0, best_alt.theory.breadth / 60.0)
        if best_alt.cp_loss is not None and best_alt.cp_loss <= 30:
            score += 0.8   # a genuine alternative, not a bad move
        if chosen.features.is_sacrifice or chosen.features.forks:
            score += 0.6
        if selection.critical:
            score += 0.4
        return score

    def _pick_alternative(self, selection: SelectionResult) -> Optional[ScoredMove]:
        for alternative in selection.top_alternatives(3):
            if alternative.move == selection.chosen.move:
                continue
            if alternative.cp_loss is not None and alternative.cp_loss > 120:
                continue
            return alternative
        return None

    def _build_variation(
        self,
        line: LineData,
        ply: int,
        first: ScoredMove,
        parent_selection: SelectionResult,
    ) -> Optional[VariationRecord]:
        """Extend one alternative into a short, purposeful sideline."""
        board = chess.Board(line.start_fen)
        for record in line.moves[:ply]:
            board.push(record.move)
        if first.move not in board.legal_moves:
            return None
        start_fen = board.fen()

        kind = "theory" if first.theory is not None else "engine"
        purpose = self._variation_purpose(first, parent_selection, kind)

        records: List[MoveRecord] = []
        cursor = board.copy(stack=True)
        annotation = self.annotator.annotate_move(
            board_before=cursor,
            board_after=_pushed(cursor, first.move),
            candidate=first,
            selection=parent_selection,
            ply=ply,
            engine_available=self.engine.available,
            is_critical=parent_selection.critical,
            intentional=False,
        )
        records.append(MoveRecord(move=first.move, san=first.san, annotation=annotation))
        cursor.push(first.move)

        previous = first.features
        for _ in range(max(1, self.config.variation_plies) - 1):
            if cursor.is_game_over(claim_draw=False):
                break
            selection = self.selector.select(cursor, previous=previous)
            if selection is None:
                break
            candidate = selection.chosen
            after = _pushed(cursor, candidate.move)
            note = self.annotator.annotate_move(
                board_before=cursor,
                board_after=after,
                candidate=candidate,
                selection=selection,
                ply=ply + len(records),
                engine_available=self.engine.available,
                is_critical=selection.critical,
            )
            records.append(MoveRecord(move=candidate.move, san=candidate.san, annotation=note))
            previous = candidate.features
            cursor.push(candidate.move)
            if selection.analysis is not None:
                self._analysed_fens.append(after.fen())

        return VariationRecord(
            start_ply=ply,
            start_fen=start_fen,
            moves=records,
            purpose=purpose if self.config.output.comments else "",
            kind=kind,
        )

    def _variation_purpose(
        self, first: ScoredMove, parent: SelectionResult, kind: str
    ) -> str:
        if first.theory is not None and first.theory.entry is not None:
            return f"Theory alternative: {first.theory.entry.name} ({first.theory.entry.eco})"
        if kind == "theory":
            return "Book alternative"
        if first.cp_loss is not None and first.cp_loss <= 15:
            return "Engine's alternative of equal value"
        if first.cp_loss is not None:
            return f"Engine alternative ({first.cp_loss} cp worse)"
        return "Alternative continuation"

    def _attach_trap_variation(self, line: LineData, trap: TrapCandidate) -> None:
        """Show the refuted mistake as a sideline when the main line avoided it.

        The bait is located by position, not by move, so a repeated move elsewhere
        in the line cannot attach the branch at the wrong ply.
        """
        board = chess.Board(line.start_fen)
        target_ply: Optional[int] = None
        for ply, record in enumerate(line.moves):
            if board.fen() == trap.origin_fen and record.move == trap.bait:
                target_ply = ply + 1
                break
            board.push(record.move)
        if target_ply is None or target_ply >= len(line.moves):
            return
        if line.moves[target_ply].move == trap.victim:
            return  # the main line already plays into the trap

        branch_board = chess.Board(line.start_fen)
        for record in line.moves[:target_ply]:
            branch_board.push(record.move)
        if trap.victim not in branch_board.legal_moves:
            return

        records: List[MoveRecord] = []
        cursor = branch_board.copy(stack=True)
        victim_candidate = self.selector.score_specific(cursor, trap.victim, critical=True)
        victim_annotation = self.annotator.annotate_move(
            board_before=cursor,
            board_after=_pushed(cursor, trap.victim),
            candidate=victim_candidate,
            selection=None,
            ply=target_ply,
            engine_available=self.engine.available,
            is_critical=True,
            intentional=False,
            extra_comment="The natural move, and the mistake the trap is built around.",
        )
        records.append(MoveRecord(move=trap.victim, san=trap.victim_san, annotation=victim_annotation))
        cursor.push(trap.victim)

        refutation_candidate = self.selector.score_specific(cursor, trap.refutation, critical=True)
        refutation_annotation = self.annotator.annotate_move(
            board_before=cursor,
            board_after=_pushed(cursor, trap.refutation),
            candidate=refutation_candidate,
            selection=None,
            ply=target_ply + 1,
            engine_available=self.engine.available,
            is_critical=True,
            extra_comment=(
                f"The refutation: {trap.cp_swing} centipawns swing according to Stockfish."
            ),
        )
        records.append(
            MoveRecord(move=trap.refutation, san=trap.refutation_san, annotation=refutation_annotation)
        )
        cursor.push(trap.refutation)

        for move in trap.followup[:3]:
            if move not in cursor.legal_moves:
                break
            candidate = self.selector.score_specific(cursor, move, critical=False)
            note = self.annotator.annotate_move(
                board_before=cursor,
                board_after=_pushed(cursor, move),
                candidate=candidate,
                selection=None,
                ply=target_ply + len(records),
                engine_available=self.engine.available,
                is_critical=False,
            )
            records.append(MoveRecord(move=move, san=candidate.san, annotation=note))
            cursor.push(move)

        line.moves[target_ply].variations.append(
            VariationRecord(
                start_ply=target_ply,
                start_fen=branch_board.fen(),
                moves=records,
                purpose="The trap: this natural reply loses material" if self.config.output.comments else "",
                kind="trap",
            )
        )

    # -- trap helpers ------------------------------------------------------ #

    def _trap_move_for(self, trap: TrapCandidate, board: chess.Board) -> Optional[chess.Move]:
        """The bait, but only from the exact position the trap was found in."""
        if board.fen() != trap.origin_fen:
            return None
        return trap.bait if trap.bait in board.legal_moves else None

    def _trap_comment(self, trap: Optional[TrapCandidate], candidate: ScoredMove) -> Optional[str]:
        if trap is None or candidate.move != trap.bait:
            return None
        if not self.config.output.comments:
            return None
        temptation = {
            "book": "a known book reply",
            "greedy": "a capture that looks like free material",
            "shallow": "the natural-looking move",
        }.get(trap.temptation, "the natural move")
        return (
            f"Sets the trap: {trap.victim_san} is {temptation} here, but "
            f"{trap.refutation_san} refutes it."
        )

    # -- shared helpers ---------------------------------------------------- #

    def _hero_colour(self, request: Request, board: chess.Board) -> chess.Color:
        if request.side == "white" or self.config.side == "white":
            return chess.WHITE
        if request.side == "black" or self.config.side == "black":
            return chess.BLACK
        return board.turn

    def _style_for(self, board: chess.Board, hero: chess.Color) -> str:
        """The hero plays in the requested style; the opponent plays soundly.

        Without this, an "aggressive" request would make *both* sides throw
        material at each other, which is not what the user asked for.
        """
        if self.config.mode == "engine":
            return "engine_best"
        if board.turn == hero:
            return self.config.style
        if self.config.style in ("gambit", "aggressive", "sharp_tactical"):
            return "classical_gm"
        return self.config.style

    def _constraint_move(self, request: Request, board: chess.Board) -> Optional[chess.Move]:
        """Apply "against 1...e5"-style constraints at the right ply."""
        ply = len(board.move_stack)
        for target_ply, san in request.constraints:
            if target_ply != ply:
                continue
            try:
                return board.parse_san(san)
            except ValueError:
                self.warnings.append(
                    f"requested reply {san!r} is not legal at ply {ply}; ignored"
                )
        return None

    def _should_stop(self, board: chess.Board, records: Sequence[MoveRecord]) -> bool:
        """Stop once the opening has been demonstrated.

        Only applies when the caller asked for it (``stop_when_out_of_theory``) and
        the position has left the book, and only after enough moves that the
        opening is actually recognisable.
        """
        if not self.config.stop_when_out_of_theory:
            return False
        if len(records) < _MIN_PLIES_BEFORE_EARLY_STOP:
            return False
        return not self.book.contains(board)

    def _headers(
        self, records: Sequence[MoveRecord], start_fen: str, request: Request
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        entry = self._classify_records(start_fen, records)
        if entry is not None and self.config.output.include_eco_headers:
            headers["ECO"] = entry.eco
            headers["Opening"] = entry.family
            if entry.variation:
                headers["Variation"] = entry.variation
            if entry.subvariation:
                headers["SubVariation"] = entry.subvariation
        include_fen = self.config.output.include_fen_headers
        # A custom start position must carry its FEN or the movetext cannot be
        # replayed, so ``never`` is honoured only for games from the initial position.
        if include_fen == "always" or start_fen != chess.Board().fen():
            headers["FEN"] = start_fen
            headers["SetUp"] = "1"
        headers["Annotator"] = self._annotator_credit()
        return headers

    def _annotator_credit(self) -> str:
        if self.engine.available and self.engine.name:
            return f"pgn-generator ({self.config.mode} mode, {self.engine.name})"
        return f"pgn-generator ({self.config.mode} mode, no engine)"

    def _classify_records(
        self, start_fen: str, records: Sequence[MoveRecord]
    ) -> Optional[BookEntry]:
        """Deepest book name reached anywhere along the line."""
        board = chess.Board(start_fen)
        best: Optional[BookEntry] = None
        for entry in self.book.entries_at(board):
            best = entry
        for record in records:
            board.push(record.move)
            for entry in self.book.entries_at(board):
                if best is None or entry.ply_count >= best.ply_count:
                    best = entry
        if best is None:
            match = self.book.classify(board)
            best = match.entry if match else None
        return best

    def _classify_line(self, line: LineData) -> Optional[BookEntry]:
        return self._classify_records(line.start_fen, line.moves)

    def _finalise_comments(self, line: LineData, request: Request, final_board: chess.Board) -> None:
        if not self.config.output.comments:
            return
        entry = self._classify_line(line)
        intro = self.annotator.opening_intro(
            chess.Board(line.start_fen), entry, request.summary() or None
        )
        if intro:
            line.intro_comment = intro

        final_score: Optional[Score] = None
        if self.engine.available:
            board = line.final_board()
            final_score = self.engine.evaluate(board)
            if final_score is not None:
                self._analysed_fens.append(board.fen())
        closing = self.annotator.closing_note(
            line.final_board(),
            final_score=final_score,
            entry=entry,
            engine_available=self.engine.available,
            reason="",
        )
        if closing:
            line.closing_comment = closing

    def _trace_entry(
        self, board: chess.Board, candidate: ScoredMove, selection: SelectionResult
    ) -> Dict[str, Any]:
        return {
            "ply": len(board.move_stack),
            "fen": board.fen(),
            "chosen": candidate.to_dict(),
            "critical": selection.critical,
            "alternatives": [alt.to_dict() for alt in selection.alternatives[:4]],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pushed(board: chess.Board, move: chess.Move) -> chess.Board:
    probe = board.copy(stack=False)
    probe.push(move)
    return probe


def _focus_side(request: Request) -> Optional[str]:
    if "kingside" in request.focus:
        return "kingside"
    if "queenside" in request.focus:
        return "queenside"
    return None


def _dedupe(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Top-level convenience API
# --------------------------------------------------------------------------- #


def generate_pgn(
    text: str = "",
    *,
    overrides: Optional[Dict[str, Any]] = None,
    explicit: Optional[Dict[str, Any]] = None,
    book: Optional[OpeningBook] = None,
    strict_moves: bool = True,
) -> GenerationResult:
    """Parse ``text``, build the config, and generate a validated PGN.

    ``overrides`` are config values (``{"engine": {"depth": 18}}``); ``explicit``
    are request fields the caller already knows (``{"fen": ..., "moves": ...}``).
    """
    book = book or get_book()
    parser = RequestParser(book)
    request = parser.parse(text, explicit=explicit, strict_moves=strict_moves)

    merged: Dict[str, Any] = {}
    merged.update(request.overrides)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    mode = (overrides or {}).get("mode") or request.mode or "gm"
    if request.style and "style" not in (overrides or {}):
        merged.setdefault("style", request.style)
    if request.side and "side" not in (overrides or {}):
        merged.setdefault("side", request.side)

    config = build_config(merged, mode=mode)
    with OpeningGenerator(config, book=book) as generator:
        return generator.generate(request)
