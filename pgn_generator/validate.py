"""Validation pipeline.

Every PGN leaves the generator through :func:`validate`, which runs the eight
checks the specification requires and refuses anything that fails a hard check:

1. **Move legality** - replay every move of every line from its own start position.
2. **SAN** - each stored SAN must be exactly what python-chess produces.
3. **PGN** - the serialised text must re-parse with no parser errors and yield the
   same move sequence.
4. **Engine** - critical positions must have been analysed; if the engine was
   unavailable this is reported as a warning and the result is marked
   ``engine_validated: false`` (never silently passed off as validated).
5. **Annotations** - every judgement NAG must carry a recorded, engine-backed
   justification.
6. **Variations** - every branch must start from the position its parent implies.
7. **FEN** - a requested start FEN must be the exact starting position.
8. **Opening** - the line must classify as (or continue) the requested opening.

Severity: ``error`` blocks output, ``warning`` is attached to the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import chess
import chess.pgn

from .annotate import JUDGEMENT_NAGS
from .book import OpeningBook
from .config import Config
from .pgn import LineData, MoveRecord, parse_pgn


@dataclass
class Finding:
    """One validation result."""

    check: str
    severity: str          # error | warning | info
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"check": self.check, "severity": self.severity, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class ValidationReport:
    """Aggregated validation outcome."""

    findings: List[Finding] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)
    engine_validated: bool = False
    positions_analysed: int = 0

    def add(self, check: str, severity: str, message: str, **details: Any) -> None:
        self.findings.append(Finding(check=check, severity=severity, message=message, details=details))

    def mark(self, check: str) -> None:
        if check not in self.checks_run:
            self.checks_run.append(check)

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "engine_validated": self.engine_validated,
            "checks_run": list(self.checks_run),
            "positions_analysed": self.positions_analysed,
            "errors": [f.to_dict() for f in self.errors],
            "warnings": [f.to_dict() for f in self.warnings],
        }


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _walk_line(
    board: chess.Board,
    records: Sequence[MoveRecord],
    report: ValidationReport,
    *,
    path: str,
) -> chess.Board:
    """Replay ``records``, checking legality and SAN, and recursing into branches."""
    cursor = board.copy(stack=True)
    for index, record in enumerate(records):
        # Variations branch from the position before this move.
        for variation in record.variations:
            expected_fen = cursor.fen()
            if variation.start_fen and variation.start_fen != expected_fen:
                report.add(
                    "variations",
                    "error",
                    f"{path}: variation at ply {index} declares start FEN "
                    f"{variation.start_fen!r} but its parent position is {expected_fen!r}",
                    path=path,
                    ply=index,
                )
            _walk_line(
                cursor,
                variation.moves,
                report,
                path=f"{path}>var@{index}",
            )

        if record.move not in cursor.legal_moves:
            report.add(
                "legality",
                "error",
                f"{path}: move {record.move.uci()} ({record.san}) is illegal in {cursor.fen()}",
                path=path,
                ply=index,
                fen=cursor.fen(),
                move=record.move.uci(),
            )
            return cursor
        actual_san = cursor.san(record.move)
        if record.san != actual_san:
            report.add(
                "san",
                "error",
                f"{path}: stored SAN {record.san!r} does not match {actual_san!r} in {cursor.fen()}",
                path=path,
                ply=index,
                expected=actual_san,
                stored=record.san,
            )
        cursor.push(record.move)
    return cursor


def check_legality_and_san(line: LineData, report: ValidationReport) -> None:
    report.mark("legality")
    report.mark("san")
    report.mark("variations")
    try:
        board = chess.Board(line.start_fen)
    except ValueError as exc:
        report.add("fen", "error", f"start FEN is invalid: {exc}", fen=line.start_fen)
        return
    _walk_line(board, line.moves, report, path="main")


def check_pgn_roundtrip(pgn_text: str, line: LineData, config: Config, report: ValidationReport) -> None:
    """Re-parse the serialised PGN and compare it against the source line."""
    report.mark("pgn")
    game, errors = parse_pgn(pgn_text)
    if game is None:
        report.add("pgn", "error", "generated PGN does not parse", parser_errors=errors)
        return
    if errors:
        report.add("pgn", "error", "PGN parser reported errors", parser_errors=errors)

    if not line.from_initial:
        setup = game.headers.get("FEN")
        if not setup:
            report.add("pgn", "error", "line starts from a custom position but no FEN header was written")
        elif chess.Board(setup).fen() != chess.Board(line.start_fen).fen():
            report.add(
                "pgn",
                "error",
                "FEN header does not match the requested starting position",
                header=setup,
                expected=line.start_fen,
            )

    expected = [record.move.uci() for record in line.moves]
    actual = [move.uci() for move in game.mainline_moves()]
    if expected != actual:
        report.add(
            "pgn",
            "error",
            "main line changed during PGN round-trip",
            expected=expected,
            actual=actual,
        )

    # Variation structure: every branch must be legal from its parent position.
    for node in game.mainline():
        parent = node.parent
        if parent is None:
            continue
        if len(parent.variations) <= 1:
            continue
        parent_board = parent.board()
        for sibling in parent.variations[1:]:
            if sibling.move not in parent_board.legal_moves:
                report.add(
                    "variations",
                    "error",
                    f"variation move {sibling.move.uci()} is illegal in the parent position "
                    f"{parent_board.fen()}",
                    fen=parent_board.fen(),
                )


def check_annotations(line: LineData, report: ValidationReport, *, engine_available: bool) -> None:
    """Every judgement NAG must have a recorded, engine-backed reason."""
    report.mark("annotations")

    def visit(records: Sequence[MoveRecord], path: str) -> None:
        for index, record in enumerate(records):
            annotation = record.annotation
            if annotation is not None and annotation.nag in JUDGEMENT_NAGS:
                if not engine_available:
                    report.add(
                        "annotations",
                        "error",
                        f"{path}: {record.san}{annotation.nag} was annotated without engine analysis",
                        path=path,
                        ply=index,
                    )
                elif not annotation.nag_reason:
                    report.add(
                        "annotations",
                        "error",
                        f"{path}: {record.san}{annotation.nag} has no recorded justification",
                        path=path,
                        ply=index,
                    )
            for variation in record.variations:
                visit(variation.moves, f"{path}>var@{index}")

    visit(line.moves, "main")


def check_engine_coverage(
    line: LineData,
    report: ValidationReport,
    *,
    engine_available: bool,
    analysed_positions: int,
    critical_positions: Sequence[str],
    analysed_fens: Sequence[str],
) -> None:
    """Confirm the engine actually looked at the critical positions."""
    report.mark("engine")
    report.positions_analysed = analysed_positions
    if not engine_available:
        report.engine_validated = False
        report.add(
            "engine",
            "warning",
            "ENGINE VALIDATION UNAVAILABLE: no Stockfish analysis was performed; moves are "
            "legality-checked and theory-based only, and no evaluations are included",
        )
        return

    missing = [fen for fen in critical_positions if fen not in set(analysed_fens)]
    if missing:
        report.add(
            "engine",
            "warning",
            f"{len(missing)} critical position(s) were not analysed at the deeper setting",
            examples=missing[:3],
        )
    report.engine_validated = True


def check_start_position(
    line: LineData,
    report: ValidationReport,
    *,
    requested_fen: Optional[str],
    requested_moves: Sequence[str],
) -> None:
    """A requested FEN or move prefix must be honoured exactly."""
    report.mark("fen")
    if requested_fen:
        try:
            wanted = chess.Board(requested_fen).fen()
        except ValueError as exc:
            report.add("fen", "error", f"requested FEN is invalid: {exc}", fen=requested_fen)
            return
        if chess.Board(line.start_fen).fen() != wanted:
            report.add(
                "fen",
                "error",
                "generated line does not start from the requested FEN",
                requested=wanted,
                actual=line.start_fen,
            )
    if requested_moves:
        actual = [record.san for record in line.moves[: len(requested_moves)]]
        if actual != list(requested_moves):
            report.add(
                "fen",
                "error",
                "generated line does not begin with the requested moves",
                requested=list(requested_moves),
                actual=actual,
            )


def check_opening(
    line: LineData,
    book: OpeningBook,
    report: ValidationReport,
    *,
    requested_eco: Optional[str],
    requested_name: Optional[str],
    requested_uci: Sequence[str],
) -> None:
    """The line must actually be the opening that was asked for.

    Accepted evidence, in order: the requested book line is a prefix of the
    generated line (or vice versa for very short requests); or some position in
    the line classifies under the requested ECO code; or under the requested
    opening family. Anything else is a warning - the PGN is still legal and sound,
    but the caller is told it drifted.
    """
    report.mark("opening")
    if not (requested_eco or requested_name or requested_uci):
        return

    line_uci = [record.move.uci() for record in line.moves]
    if requested_uci:
        shared = min(len(requested_uci), len(line_uci))
        if list(requested_uci[:shared]) == line_uci[:shared] and shared > 0:
            if len(line_uci) >= len(requested_uci):
                return  # exact continuation of the requested line
            report.add(
                "opening",
                "warning",
                "generated line is shorter than the canonical line for the requested opening",
                requested_plies=len(requested_uci),
                actual_plies=len(line_uci),
            )
            return

    # Walk the line and collect every classification along the way.
    board = chess.Board(line.start_fen)
    seen_names: List[str] = []
    seen_ecos: List[str] = []
    for entry in book.entries_at(board):
        seen_names.append(entry.name)
        seen_ecos.append(entry.eco)
    for record in line.moves:
        board.push(record.move)
        for entry in book.entries_at(board):
            seen_names.append(entry.name)
            seen_ecos.append(entry.eco)

    if requested_eco and requested_eco in seen_ecos:
        return
    if requested_name:
        family = requested_name.split(":", 1)[0].strip().lower()
        if any(name.lower().startswith(family) for name in seen_names):
            return
    if seen_names:
        report.add(
            "opening",
            "warning",
            f"generated line classifies as {seen_names[-1]!r}, not the requested "
            f"{requested_name or requested_eco!r}",
            classified_as=seen_names[-1],
            requested=requested_name or requested_eco,
        )
    else:
        report.add(
            "opening",
            "warning",
            "generated line does not appear in the ECO book, so the requested opening could "
            "not be confirmed",
            requested=requested_name or requested_eco,
        )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def validate(
    *,
    line: LineData,
    pgn_text: str,
    config: Config,
    book: OpeningBook,
    engine_available: bool,
    analysed_positions: int = 0,
    critical_positions: Sequence[str] = (),
    analysed_fens: Sequence[str] = (),
    requested_fen: Optional[str] = None,
    requested_moves: Sequence[str] = (),
    requested_eco: Optional[str] = None,
    requested_name: Optional[str] = None,
    requested_uci: Sequence[str] = (),
) -> ValidationReport:
    """Run the full validation pipeline over a generated line."""
    report = ValidationReport()
    check_legality_and_san(line, report)
    check_pgn_roundtrip(pgn_text, line, config, report)
    check_engine_coverage(
        line,
        report,
        engine_available=engine_available,
        analysed_positions=analysed_positions,
        critical_positions=critical_positions,
        analysed_fens=analysed_fens,
    )
    check_annotations(line, report, engine_available=engine_available)
    check_start_position(line, report, requested_fen=requested_fen, requested_moves=requested_moves)
    check_opening(
        line,
        book,
        report,
        requested_eco=requested_eco,
        requested_name=requested_name,
        requested_uci=requested_uci,
    )
    return report
