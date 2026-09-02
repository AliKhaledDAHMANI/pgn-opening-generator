"""Chess feature extraction: the factual basis for annotations and comments.

Nothing in this module scores a position - that is Stockfish's job. What it does
is answer objective, verifiable questions about a move ("is this a sacrifice?",
"does it fork?", "does it leave a piece hanging?") and about a position ("who is
castled?", "is there an isolated queen's pawn?").

The annotator combines these facts with engine numbers, so that every ``!`` and
every comment can be traced back to something concrete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess

#: Material values in centipawns (classical, used for SEE and material counts).
PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

PIECE_NAMES: Dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
EXTENDED_CENTER = CENTER_SQUARES + (chess.C3, chess.C4, chess.C5, chess.C6,
                                    chess.D3, chess.D6, chess.E3, chess.E6,
                                    chess.F3, chess.F4, chess.F5, chess.F6)

#: Minimum material loss (centipawns) that counts as a sacrifice. SEE already
#: nets out recaptures, so an even trade scores 0 and only a real investment
#: crosses this line.
_SACRIFICE_THRESHOLD = 90


def piece_value(piece_type: Optional[int]) -> int:
    return PIECE_VALUES.get(piece_type, 0) if piece_type else 0


def square_name(square: int) -> str:
    return chess.square_name(square)


# --------------------------------------------------------------------------- #
# Static exchange evaluation
# --------------------------------------------------------------------------- #


def _least_valuable_attacker(
    board: chess.Board, color: chess.Color, square: int, occupied: int
) -> Tuple[Optional[int], Optional[int]]:
    """Cheapest piece of ``color`` attacking ``square`` given ``occupied``."""
    attackers = board.attackers_mask(color, square, occupied) & occupied
    if not attackers:
        return None, None
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        subset = attackers & board.pieces_mask(piece_type, color)
        if subset:
            return chess.lsb(subset), piece_type
    return None, None


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Centipawn outcome of the capture sequence on ``move.to_square``.

    Standard swap-off SEE, for **capture moves**. Two documented approximations:
    absolute pins are ignored, and promotions inside the recapture sequence are
    not modelled. Used only to classify material investment (sacrifice vs. sound
    capture); every evaluation claim comes from the engine.

    For quiet moves use :func:`material_risk_of_quiet_move` instead - SEE of a
    non-capture answers a different question and reads every developing move as a
    sacrifice.
    """
    to_square = move.to_square
    occupied = board.occupied

    if board.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
        captured_pawn = to_square + (-8 if board.turn == chess.WHITE else 8)
        occupied &= ~chess.BB_SQUARES[captured_pawn]
    else:
        captured_value = piece_value(board.piece_type_at(to_square))

    attacker_type = board.piece_type_at(move.from_square)
    if attacker_type is None:  # pragma: no cover - illegal input
        return 0
    gains: List[int] = [captured_value]
    if move.promotion:
        gains[0] += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
        attacker_type = move.promotion

    occupied &= ~chess.BB_SQUARES[move.from_square]
    side = not board.turn
    depth = 0

    while True:
        from_square, next_attacker = _least_valuable_attacker(board, side, to_square, occupied)
        if from_square is None:
            break
        depth += 1
        gains.append(piece_value(attacker_type) - gains[depth - 1])
        if max(-gains[depth - 1], gains[depth]) < 0:
            break
        occupied &= ~chess.BB_SQUARES[from_square]
        attacker_type = next_attacker
        side = not side

    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return gains[0]


def best_capture_see(board: chess.Board, square: int) -> int:
    """Best static-exchange outcome available to the side to move on ``square``.

    ``0`` when no capture there wins material. This is the correct way to ask
    "does leaving a piece here lose material?".
    """
    best = 0
    for move in board.generate_legal_captures(to_mask=chess.BB_SQUARES[square]):
        best = max(best, static_exchange_eval(board, move))
    return best


def material_risk_of_quiet_move(board_after: chess.Board, square: int) -> int:
    """Material the mover stands to lose on ``square`` after a non-capture.

    ``board_after`` must be the position *after* the move, so the side to move is
    the opponent. Returns a positive number of centipawns when the opponent has a
    profitable capture there, otherwise ``0``.
    """
    return max(0, best_capture_see(board_after, square))


def material_balance(board: chess.Board) -> int:
    """Material in centipawns from White's point of view (kings excluded)."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type == chess.KING:
            continue
        total += value * (
            len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))
        )
    return total


def hanging_pieces(board: chess.Board, color: chess.Color, *, min_value: int = 300) -> List[int]:
    """Squares where ``color`` loses material to a capture.

    Requires ``color`` to be the side *not* to move (i.e. the opponent is about to
    capture). Uses SEE so that adequately defended pieces are not reported.
    """
    if board.turn == color:
        return []
    out: List[int] = []
    for move in board.generate_legal_captures():
        victim = board.piece_at(move.to_square)
        if victim is None or victim.piece_type == chess.KING:
            continue
        if piece_value(victim.piece_type) < min_value:
            continue
        if static_exchange_eval(board, move) > 0:
            out.append(move.to_square)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Motifs
# --------------------------------------------------------------------------- #


@dataclass
class Motif:
    """One verifiable tactical/strategic observation about a move."""

    kind: str
    text: str
    #: ``(from, to)`` square pairs worth drawing as arrows.
    arrows: List[Tuple[int, int]] = field(default_factory=list)
    #: Squares worth highlighting.
    squares: List[int] = field(default_factory=list)
    #: Rough importance, used to cap how much gets rendered.
    weight: float = 1.0


@dataclass
class MoveFeatures:
    """Objective properties of one move, computed before/after pushing it."""

    san: str
    uci: str
    color: chess.Color
    piece_type: int
    from_square: int
    to_square: int

    is_capture: bool = False
    captured_type: Optional[int] = None
    is_en_passant: bool = False
    is_castling: bool = False
    castles_short: bool = False
    is_promotion: bool = False
    promotion_type: Optional[int] = None

    gives_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False

    see_cp: int = 0
    #: Material the mover stands to lose on the arrival square by static exchange
    #: (0 when the square is safe). Static only - see :attr:`is_sacrifice`.
    risk_cp: int = 0
    #: Material apparently given up. Derived from :func:`static_exchange_eval`, so
    #: it is a *candidate* sacrifice: SEE only looks at capture sequences on one
    #: square and can flag main-line moves whose point lies elsewhere. Callers that
    #: describe a move as a sacrifice must confirm it with the engine first
    #: (:meth:`pgn_generator.annotate.Annotator.confirm_investment`).
    invested_cp: int = 0
    is_sacrifice: bool = False
    is_exchange_sacrifice: bool = False

    forced: bool = False          # only legal move
    legal_move_count: int = 0

    attacks_higher_value: List[int] = field(default_factory=list)   # target squares
    forks: List[int] = field(default_factory=list)                  # forked squares
    creates_pin: List[Tuple[int, int]] = field(default_factory=list)  # (pinned, behind)
    discovered_attack: List[Tuple[int, int]] = field(default_factory=list)  # (from, target)
    leaves_hanging: List[int] = field(default_factory=list)
    captures_defended_piece: bool = False
    attacks_king_zone: bool = False
    king_zone_pressure: int = 0

    develops_piece: bool = False
    central_pawn_move: bool = False
    is_pawn_move: bool = False
    is_quiet: bool = False
    opens_file_for_rook: bool = False
    loses_castling_rights: bool = False
    creates_threat_of_mate: bool = False

    motifs: List[Motif] = field(default_factory=list)

    @property
    def piece_name(self) -> str:
        return PIECE_NAMES.get(self.piece_type, "piece")

    @property
    def captured_name(self) -> Optional[str]:
        return PIECE_NAMES.get(self.captured_type) if self.captured_type else None

    @property
    def is_forcing(self) -> bool:
        return self.gives_check or self.is_capture or bool(self.attacks_higher_value) or self.is_checkmate

    def motif_kinds(self) -> List[str]:
        return [m.kind for m in self.motifs]

    def to_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "see_cp": self.see_cp,
            "risk_cp": self.risk_cp,
            "invested_cp": self.invested_cp,
            "is_sacrifice": self.is_sacrifice,
            "gives_check": self.gives_check,
            "is_capture": self.is_capture,
            "forced": self.forced,
            "motifs": self.motif_kinds(),
        }


def _king_zone(board: chess.Board, color: chess.Color) -> int:
    """King square plus its ring, for the king of ``color``."""
    king = board.king(color)
    if king is None:
        return 0
    return chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]


def _is_developing(board_before: chess.Board, move: chess.Move, piece_type: int, color: chess.Color) -> bool:
    """A minor piece or rook leaving its home rank for the first time.

    The queen is excluded: early queen sorties are not "development" in the sense
    the comments use the word.
    """
    if piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        return False
    home_rank = 0 if color == chess.WHITE else 7
    return chess.square_rank(move.from_square) == home_rank and chess.square_rank(move.to_square) != home_rank


def analyse_move(
    board: chess.Board,
    move: chess.Move,
    *,
    board_after: Optional[chess.Board] = None,
) -> MoveFeatures:
    """Extract objective features for ``move`` played from ``board``."""
    color = board.turn
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {square_name(move.from_square)} for move {move.uci()}")

    san = board.san(move)
    legal_count = board.legal_moves.count()
    is_capture = board.is_capture(move)
    is_ep = board.is_en_passant(move)
    captured_type = chess.PAWN if is_ep else board.piece_type_at(move.to_square)

    after = board_after
    if after is None:
        after = board.copy(stack=False)
        after.push(move)

    # Material accounting. Captures use SEE on the move itself; quiet moves ask
    # what the opponent can now win on the arrival square. Mixing the two would
    # label every developed knight a sacrifice.
    if is_capture:
        see_value = static_exchange_eval(board, move)
        risk = max(0, -see_value)
    else:
        see_value = 0
        risk = material_risk_of_quiet_move(after, move.to_square)

    features = MoveFeatures(
        san=san,
        uci=move.uci(),
        color=color,
        piece_type=piece.piece_type,
        from_square=move.from_square,
        to_square=move.to_square,
        is_capture=is_capture,
        captured_type=captured_type,
        is_en_passant=is_ep,
        is_castling=board.is_castling(move),
        castles_short=board.is_kingside_castling(move),
        is_promotion=move.promotion is not None,
        promotion_type=move.promotion,
        gives_check=after.is_check(),
        is_checkmate=after.is_checkmate(),
        is_stalemate=after.is_stalemate(),
        see_cp=see_value,
        forced=legal_count == 1,
        legal_move_count=legal_count,
        is_pawn_move=piece.piece_type == chess.PAWN,
        develops_piece=_is_developing(board, move, piece.piece_type, color),
        central_pawn_move=piece.piece_type == chess.PAWN and move.to_square in CENTER_SQUARES,
        loses_castling_rights=(
            piece.piece_type == chess.KING
            and not board.is_castling(move)
            and bool(board.clean_castling_rights() & (chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8))
        ),
    )
    features.is_quiet = not is_capture and not features.gives_check and not features.is_promotion
    features.risk_cp = risk

    # -- material investment ------------------------------------------------ #
    # A sacrifice is an investment of at least a pawn's worth of material that the
    # opponent can actually collect. Even trades (dxe5 dxe5) score 0 by SEE and are
    # therefore not sacrifices, while a genuine gambit pawn is.
    if risk >= _SACRIFICE_THRESHOLD:
        features.invested_cp = risk
        features.is_sacrifice = True
        if piece.piece_type == chess.ROOK and captured_type in (chess.KNIGHT, chess.BISHOP):
            features.is_exchange_sacrifice = True

    if is_capture and not is_ep:
        features.captures_defended_piece = board.is_attacked_by(not color, move.to_square)

    # -- threats created by the moved piece --------------------------------- #
    mover_value = piece_value(features.promotion_type or piece.piece_type)
    targets: List[int] = []
    for target in chess.SquareSet(int(after.attacks(move.to_square)) & after.occupied_co[not color]):
        target_piece = after.piece_at(target)
        if target_piece is None:
            continue
        if target_piece.piece_type == chess.KING:
            continue
        defended = after.is_attacked_by(not color, target)
        if piece_value(target_piece.piece_type) > mover_value or not defended:
            targets.append(target)
    features.attacks_higher_value = sorted(targets)

    # A fork only deserves the name when the forking piece cannot simply be taken.
    # ``risk_cp > 0`` means the opponent wins material by capturing on the arrival
    # square, so the "fork" is really a sacrifice and is reported as such instead.
    mover_safe = features.risk_cp == 0
    fork_targets = [sq for sq in features.attacks_higher_value if piece_value(after.piece_type_at(sq)) >= 300]
    if not mover_safe:
        features.forks = []
    elif features.gives_check and not features.is_checkmate:
        # With check the opponent must react, so an undefended second target is lost.
        loose = [sq for sq in fork_targets if not after.is_attacked_by(not color, sq)]
        enemy_king = after.king(not color)
        if loose and enemy_king is not None:
            features.forks = sorted(set(loose + [enemy_king]))
    elif len(fork_targets) >= 2:
        features.forks = sorted(fork_targets)

    # -- pins created ------------------------------------------------------- #
    # Only report pins that actually restrict the opponent: the pinned unit must be
    # a piece (not a pawn), and the unit behind it must be the king or clearly more
    # valuable. This avoids labelling ordinary recaptures ("Bxc6, b7 is pinned") as
    # pins.
    if features.promotion_type or piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in chess.SquareSet(after.occupied_co[not color]):
            target_piece = after.piece_at(square)
            if target_piece is None or target_piece.piece_type == chess.KING:
                continue
            if piece_value(target_piece.piece_type) < 300:
                continue
            if square not in after.attacks(move.to_square):
                continue
            behind = _first_piece_behind(after, move.to_square, square)
            if behind is None:
                continue
            occupant = after.piece_at(behind)
            if occupant is None or occupant.color != (not color):
                continue
            if occupant.piece_type == chess.KING or piece_value(occupant.piece_type) > piece_value(
                target_piece.piece_type
            ):
                features.creates_pin.append((square, behind))

    # -- discovered attacks ------------------------------------------------- #
    for square in chess.SquareSet(after.occupied_co[color]):
        moved_piece = after.piece_at(square)
        if moved_piece is None or square == move.to_square:
            continue
        if moved_piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        before_attacks = int(board.attacks(square)) if board.piece_at(square) else 0
        gained = int(after.attacks(square)) & ~before_attacks & after.occupied_co[not color]
        for target in chess.SquareSet(gained):
            target_piece = after.piece_at(target)
            if target_piece is None:
                continue
            if piece_value(target_piece.piece_type) >= 300 or target_piece.piece_type == chess.KING:
                features.discovered_attack.append((square, target))

    # -- king pressure ------------------------------------------------------ #
    zone = _king_zone(after, not color)
    if zone:
        pressure = chess.popcount(int(after.attacks(move.to_square)) & zone)
        features.king_zone_pressure = pressure
        features.attacks_king_zone = pressure > 0 or bool(chess.BB_SQUARES[move.to_square] & zone)

    # -- own weaknesses ----------------------------------------------------- #
    # ``after`` has the opponent to move, so this reports material the mover is
    # about to lose. Sacrifices are excluded: the investment is already reported.
    if not features.is_sacrifice:
        features.leaves_hanging = [
            sq for sq in hanging_pieces(after, color, min_value=300) if sq != move.to_square
        ]

    # -- mate threat -------------------------------------------------------- #
    if not features.is_checkmate and features.gives_check is False:
        features.creates_threat_of_mate = _threatens_mate(after)

    # -- rook file ---------------------------------------------------------- #
    if piece.piece_type == chess.PAWN and chess.square_file(move.from_square) != chess.square_file(
        move.to_square
    ):
        file_mask = chess.BB_FILES[chess.square_file(move.from_square)]
        if not (after.pieces_mask(chess.PAWN, color) & file_mask):
            features.opens_file_for_rook = True

    features.motifs = _build_motifs(board, after, move, features)
    return features


def _first_piece_behind(board: chess.Board, attacker: int, target: int) -> Optional[int]:
    """First occupied square beyond ``target`` on the attacker->target ray.

    Scans the whole ray rather than only the adjacent square, so a pin through a
    gap (Bg5 pinning Nf6 to Qd8 with e7 empty) is found.
    """
    if not chess.ray(attacker, target):
        return None
    file_step = _sign(chess.square_file(target) - chess.square_file(attacker))
    rank_step = _sign(chess.square_rank(target) - chess.square_rank(attacker))
    if file_step == 0 and rank_step == 0:  # pragma: no cover - defensive
        return None
    file_index = chess.square_file(target) + file_step
    rank_index = chess.square_rank(target) + rank_step
    while 0 <= file_index <= 7 and 0 <= rank_index <= 7:
        square = chess.square(file_index, rank_index)
        if board.piece_at(square) is not None:
            return square
        file_index += file_step
        rank_index += rank_step
    return None


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _threatens_mate(board: chess.Board) -> bool:
    """True when the side that just moved threatens mate on the next move.

    Implemented with a null move so the mover gets a "free" turn; only checking
    moves are tested, since mate must be delivered with check.
    """
    if board.is_game_over(claim_draw=False) or board.is_check():
        return False
    probe = board.copy(stack=False)
    probe.push(chess.Move.null())
    if probe.is_check():  # pragma: no cover - would mean the prior move was illegal
        return False
    for move in probe.legal_moves:
        if not probe.gives_check(move):
            continue
        probe.push(move)
        mate = probe.is_checkmate()
        probe.pop()
        if mate:
            return True
    return False


def _build_motifs(
    board: chess.Board, after: chess.Board, move: chess.Move, features: MoveFeatures
) -> List[Motif]:
    motifs: List[Motif] = []
    color = features.color
    enemy_king = after.king(not color)

    if features.is_checkmate:
        motifs.append(
            Motif("mate", "checkmate", arrows=[(move.from_square, move.to_square)],
                  squares=[enemy_king] if enemy_king else [], weight=3.0)
        )
    if features.forks:
        names = ", ".join(
            f"{PIECE_NAMES.get(after.piece_type_at(sq), 'piece')} on {square_name(sq)}"
            for sq in features.forks
            if after.piece_at(sq)
        )
        motifs.append(
            Motif(
                "fork",
                f"the {features.piece_name} hits {names} at once" if names else "double attack",
                arrows=[(move.to_square, sq) for sq in features.forks],
                squares=list(features.forks),
                weight=2.4,
            )
        )
    if features.creates_pin:
        pinned, behind = features.creates_pin[0]
        behind_piece = after.piece_at(behind)
        behind_name = PIECE_NAMES.get(behind_piece.piece_type, "piece") if behind_piece else "piece"
        motifs.append(
            Motif(
                "pin",
                f"the {PIECE_NAMES.get(after.piece_type_at(pinned), 'piece')} on {square_name(pinned)} "
                f"is pinned against the {behind_name} on {square_name(behind)}",
                arrows=[(move.to_square, behind)],
                squares=[pinned],
                weight=2.0,
            )
        )
    if features.discovered_attack:
        origin, target = features.discovered_attack[0]
        motifs.append(
            Motif(
                "discovered_attack",
                f"the {PIECE_NAMES.get(after.piece_type_at(origin), 'piece')} on {square_name(origin)} "
                f"is uncovered against {square_name(target)}",
                arrows=[(origin, target)],
                squares=[target],
                weight=2.0,
            )
        )
    if features.is_sacrifice:
        # Neutral wording on purpose: whether the investment is a sound sacrifice or
        # a plain blunder is decided by the engine, in annotate.py.
        if features.is_exchange_sacrifice:
            detail = f"gives up rook for minor piece on {square_name(move.to_square)}"
        elif features.is_capture:
            detail = (
                f"invests the {features.piece_name} on {square_name(move.to_square)} "
                f"({features.invested_cp} cp by static exchange)"
            )
        else:
            detail = (
                f"offers the {features.piece_name} on {square_name(move.to_square)} "
                f"({features.invested_cp} cp by static exchange)"
            )
        motifs.append(
            Motif(
                "material_investment",
                detail,
                arrows=[(move.from_square, move.to_square)],
                squares=[move.to_square],
                weight=2.6,
            )
        )
    if features.gives_check and not features.is_checkmate:
        motifs.append(
            Motif(
                "check",
                "with check",
                arrows=[(move.to_square, enemy_king)] if enemy_king else [],
                squares=[],
                weight=1.2,
            )
        )
    if features.attacks_king_zone and features.king_zone_pressure >= 2 and not features.gives_check:
        motifs.append(
            Motif(
                "king_attack",
                f"pressure builds on the squares around the {'black' if color else 'white'} king",
                arrows=[],
                squares=[enemy_king] if enemy_king else [],
                weight=1.4,
            )
        )
    if features.attacks_higher_value and not features.forks and not features.gives_check:
        target = features.attacks_higher_value[0]
        target_piece = after.piece_at(target)
        if target_piece is not None:
            motifs.append(
                Motif(
                    "threat",
                    f"threatening the {PIECE_NAMES.get(target_piece.piece_type, 'piece')} on "
                    f"{square_name(target)}",
                    arrows=[(move.to_square, target)],
                    squares=[target],
                    weight=1.3,
                )
            )
    if features.leaves_hanging:
        square = features.leaves_hanging[0]
        piece_at = after.piece_at(square)
        if piece_at is not None:
            motifs.append(
                Motif(
                    "hanging",
                    f"the {PIECE_NAMES.get(piece_at.piece_type, 'piece')} on {square_name(square)} "
                    "is left undefended",
                    arrows=[],
                    squares=[square],
                    weight=1.5,
                )
            )
    if features.is_castling:
        motifs.append(
            Motif(
                "castling",
                "castling into safety" if features.castles_short else "castling long",
                arrows=[(move.from_square, move.to_square)],
                squares=[],
                weight=0.9,
            )
        )
    if features.develops_piece:
        motifs.append(
            Motif(
                "development",
                f"developing the {features.piece_name} to {square_name(move.to_square)}",
                arrows=[(move.from_square, move.to_square)],
                squares=[],
                weight=0.7,
            )
        )
    if features.central_pawn_move:
        motifs.append(
            Motif(
                "center",
                f"staking a claim in the centre with {features.san}",
                arrows=[(move.from_square, move.to_square)],
                squares=[move.to_square],
                weight=0.8,
            )
        )
    if features.opens_file_for_rook:
        motifs.append(
            Motif(
                "open_file",
                f"the {chess.FILE_NAMES[chess.square_file(move.from_square)]}-file opens for the rooks",
                arrows=[],
                squares=[],
                weight=0.8,
            )
        )
    if features.creates_threat_of_mate:
        motifs.append(Motif("mate_threat", "with a mating threat", weight=2.2))

    motifs.sort(key=lambda m: -m.weight)
    return motifs


# --------------------------------------------------------------------------- #
# Position features
# --------------------------------------------------------------------------- #


@dataclass
class PawnStructure:
    doubled: List[int] = field(default_factory=list)
    isolated: List[int] = field(default_factory=list)
    passed: List[int] = field(default_factory=list)
    backward_count: int = 0
    islands: int = 0
    center_pawns: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "doubled": [square_name(s) for s in self.doubled],
            "isolated": [square_name(s) for s in self.isolated],
            "passed": [square_name(s) for s in self.passed],
            "islands": self.islands,
            "center_pawns": self.center_pawns,
        }


def pawn_structure(board: chess.Board, color: chess.Color) -> PawnStructure:
    pawns = board.pieces(chess.PAWN, color)
    files: Dict[int, List[int]] = {}
    for square in pawns:
        files.setdefault(chess.square_file(square), []).append(square)

    structure = PawnStructure()
    for file_index, squares in files.items():
        if len(squares) > 1:
            structure.doubled.extend(squares)
        neighbours = [f for f in (file_index - 1, file_index + 1) if 0 <= f <= 7]
        if not any(n in files for n in neighbours):
            structure.isolated.extend(squares)

    enemy_pawns = board.pieces(chess.PAWN, not color)
    for square in pawns:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        blocked = False
        for enemy in enemy_pawns:
            e_file = chess.square_file(enemy)
            if abs(e_file - file_index) > 1:
                continue
            e_rank = chess.square_rank(enemy)
            if (color == chess.WHITE and e_rank > rank_index) or (
                color == chess.BLACK and e_rank < rank_index
            ):
                blocked = True
                break
        if not blocked:
            structure.passed.append(square)

    occupied_files = sorted(files.keys())
    islands = 0
    previous: Optional[int] = None
    for file_index in occupied_files:
        if previous is None or file_index - previous > 1:
            islands += 1
        previous = file_index
    structure.islands = islands
    structure.center_pawns = sum(1 for square in pawns if square in EXTENDED_CENTER)
    structure.doubled = sorted(set(structure.doubled))
    structure.isolated = sorted(set(structure.isolated))
    structure.passed = sorted(set(structure.passed))
    return structure


@dataclass
class PositionFeatures:
    fen: str
    material: int
    castled: Dict[str, bool]
    king_in_center: Dict[str, bool]
    development: Dict[str, int]
    structure: Dict[str, PawnStructure]
    open_files: List[int]
    semi_open_files: Dict[str, List[int]] = field(default_factory=dict)
    iqp: Dict[str, bool] = field(default_factory=dict)
    tension: int = 0
    space: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "material": self.material,
            "castled": self.castled,
            "development": self.development,
            "open_files": [chess.FILE_NAMES[f] for f in self.open_files],
            "iqp": self.iqp,
            "white_structure": self.structure["white"].to_dict(),
            "black_structure": self.structure["black"].to_dict(),
        }


def _has_castled(board: chess.Board, color: chess.Color) -> bool:
    king = board.king(color)
    if king is None:
        return False
    if color == chess.WHITE:
        return king in (chess.G1, chess.H1, chess.C1, chess.B1) and not (
            board.castling_rights & chess.BB_RANK_1
        )
    return king in (chess.G8, chess.H8, chess.C8, chess.B8) and not (
        board.castling_rights & chess.BB_RANK_8
    )


def _development_count(board: chess.Board, color: chess.Color) -> int:
    home_rank = 0 if color == chess.WHITE else 7
    count = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP):
        for square in board.pieces(piece_type, color):
            if chess.square_rank(square) != home_rank:
                count += 1
    return count


def analyse_position(board: chess.Board) -> PositionFeatures:
    """Structural facts about a position, used to ground strategic comments."""
    white_pawn_files = {chess.square_file(s) for s in board.pieces(chess.PAWN, chess.WHITE)}
    black_pawn_files = {chess.square_file(s) for s in board.pieces(chess.PAWN, chess.BLACK)}
    open_files = [f for f in range(8) if f not in white_pawn_files and f not in black_pawn_files]

    white_structure = pawn_structure(board, chess.WHITE)
    black_structure = pawn_structure(board, chess.BLACK)

    tension = sum(1 for _ in board.generate_legal_captures())

    def _space(color: chess.Color) -> int:
        pawns = board.pieces(chess.PAWN, color)
        if color == chess.WHITE:
            return sum(1 for s in pawns if chess.square_rank(s) >= 3)
        return sum(1 for s in pawns if chess.square_rank(s) <= 4)

    def _iqp(color: chess.Color, structure: PawnStructure) -> bool:
        """Isolated queen's pawn: an isolated pawn on the d-file."""
        return any(chess.square_file(sq) == 3 for sq in structure.isolated)

    return PositionFeatures(
        fen=board.fen(),
        material=material_balance(board),
        castled={"white": _has_castled(board, chess.WHITE), "black": _has_castled(board, chess.BLACK)},
        king_in_center={
            "white": chess.square_file(board.king(chess.WHITE) or chess.E1) in (3, 4)
            and not _has_castled(board, chess.WHITE),
            "black": chess.square_file(board.king(chess.BLACK) or chess.E8) in (3, 4)
            and not _has_castled(board, chess.BLACK),
        },
        development={
            "white": _development_count(board, chess.WHITE),
            "black": _development_count(board, chess.BLACK),
        },
        structure={"white": white_structure, "black": black_structure},
        open_files=open_files,
        semi_open_files={
            "white": [f for f in range(8) if f not in white_pawn_files and f in black_pawn_files],
            "black": [f for f in range(8) if f not in black_pawn_files and f in white_pawn_files],
        },
        iqp={"white": _iqp(chess.WHITE, white_structure), "black": _iqp(chess.BLACK, black_structure)},
        tension=tension,
        space={"white": _space(chess.WHITE), "black": _space(chess.BLACK)},
    )
