"""Natural-language request parsing.

Turns a free-text description ("aggressive Sicilian line where Black attacks the
kingside", "best line after 1.e4 c5 2.Nf3 d6 3.d4") into a structured
:class:`Request` with explicit fields plus the config overrides implied by the
wording. Every inference is recorded in :attr:`Request.inferences` so the caller
can see how a vague prompt was interpreted.

Parsing is deliberately conservative: anything it cannot understand is left as a
default rather than guessed, and an unparsable *move* is an error rather than a
silent omission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

from .book import BookEntry, OpeningBook, expand_aliases, normalize_name
from .errors import IllegalMoveError, InvalidFENError, RequestError

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

_MODE_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"\btrap(s)?\b|\bpunish\w*\b|\brefut\w*\b|\bpitfall\b|\bsucker\b", "trap"),
    (r"\brepertoire\b|\bagainst everything\b|\bfull system\b|\bcomplete system\b", "repertoire"),
    (r"\btraining\b|\bteach\b|\blearn\b|\bexplain\b|\binstruct\w*\b|\bstudy\b|\bbeginner\b|\btutorial\b", "training"),
    (r"\bengine[- ]best\b|\bstockfish[- ]best\b|\bobjectiv\w+\b|\bbest move[s]?\b|\bcomputer\b", "engine"),
    (r"\bgm[- ]style\b|\bgrandmaster\b|\bgm\b", "gm"),
)

_STYLE_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"\bgambit\b|\bsacrific\w*\b|\bsac\b|\bpawn for\b|\boffer\w* a pawn\b", "gambit"),
    (r"\bsharp\b|\btactical\b|\bdouble[- ]edged\b|\bmessy\b|\bwild\b|\bcomplicat\w+\b", "sharp_tactical"),
    (r"\baggressive\b|\battack\w*\b|\bkingside attack\b|\bcrush\w*\b|\bviolent\b|\bkill\w*\b", "aggressive"),
    (r"\bsolid\b|\bsafe\b|\bquiet\b|\bstable\b|\bno risk\b|\block[- ]down\b", "solid"),
    (r"\bpositional\b|\bstrategic\b|\bmanoeuvr\w*\b|\bmaneuver\w*\b|\bsqueeze\b|\bslow\b", "positional"),
    (r"\bpractical\b|\bclub\b|\bblitz\b|\brapid\b|\beasy to play\b|\bsimple\b", "practical"),
    (r"\bengine[- ]best\b|\bobjectiv\w+\b|\bcomputer\b", "engine_best"),
    (r"\bmain[- ]?line\b|\btheoretical\b|\btheory\b|\bcritical line\b|\bmain variation\b", "theoretical"),
    (r"\bclassical\b|\bgm\b|\bgrandmaster\b", "classical_gm"),
)

_SIDE_WHITE = re.compile(r"\bfor white\b|\bas white\b|\bwhite'?s\b|\bwith white\b|\bwhite to\b|\bwhite plays\b")
_SIDE_BLACK = re.compile(r"\bfor black\b|\bas black\b|\bblack'?s\b|\bwith black\b|\bblack to\b|\bblack plays\b")

_RARE = re.compile(r"\brare\b|\bobscure\b|\boffbeat\b|\bunusual\b|\bsurprise\b|\bnovelty\b|\bunorthodox\b|\bsideline\b")
_LONG = re.compile(r"\b(\d{1,2})[- ]move[s]?\b")
_DEEP = re.compile(r"\bdeep\w*\b|\blong\b|\bextended\b")
_SHORT = re.compile(r"\bshort\b|\bbrief\b|\bquick\b")
_NO_COMMENTS = re.compile(r"\bno comment\w*\b|\bwithout comment\w*\b|\bmoves only\b|\bbare\b")
_NO_ARROWS = re.compile(r"\bno arrow\w*\b|\bwithout arrow\w*\b")
_WITH_ARROWS = re.compile(r"\barrow\w*\b|\bhighlight\w*\b|\bvisual\w*\b")
_NO_EVALS = re.compile(r"\bno eval\w*\b|\bwithout eval\w*\b|\bno score\w*\b")
_WITH_EVALS = re.compile(r"\beval\w*\b|\bscore\w*\b|\bcentipawn\w*\b|\bstockfish number\w*\b")
_VARIATION_COUNT = re.compile(r"\b(\d{1,2})\s+(?:variation|sideline|branch|alternative)s?\b")
_KINGSIDE = re.compile(r"\bkingside\b|\bking side\b|\bg-?file\b|\bh-?file\b")
_QUEENSIDE = re.compile(r"\bqueenside\b|\bqueen side\b|\bb-?file\b|\ba-?file\b|\bminority attack\b")

_STOPWORDS_EXTRA = re.compile(
    r"\b(?:show|give|create|generate|build|make|please|me|my|us|a|an|the|of|for|with|in|to|and|or|"
    r"can|you|i|want|need|would|like|some|best|line|lines|variation|variations|opening|openings|"
    r"pgn|analysis|annotated|annotations|tactical|position|response|responses|move|moves|"
    r"repertoire|trap|traps|style|gm|grandmaster|main|mainline|theoretical|theory|sharp|"
    r"aggressive|solid|positional|practical|rare|deep|short|long|against|vs|versus|anti|"
    r"where|which|that|this|after|from|continue|continues|play|plays|playing|sacrifices|"
    r"sacrifice|pawn|initiative|kingside|queenside|attacks|attack|white|black|side|sides|"
    r"engine|stockfish|objective|training|teach|learn|explain|study|idea|ideas|plan|plans|"
    r"good|nice|interesting|typical|modern|classical|instructive|is|are|it|its|on|at|by|as)\b"
)

#: Openings named for the *reply* rather than the opening move, so a request
#: phrased as "against 1...e5" maps to a first move for the other colour.
_AGAINST = re.compile(r"\bagainst\b|\bvs\.?\b|\bversus\b|\banti[- ]", re.IGNORECASE)

#: Black-move notation: "1...e5", "3... Nf6".
_BLACK_MOVE_PREFIX = re.compile(r"\b(\d+)\s*\.\s*\.\s*\.")


# --------------------------------------------------------------------------- #
# Move-text parsing
# --------------------------------------------------------------------------- #

_MOVE_NUMBER = re.compile(r"^\d+\.(\.\.)?$")
_SAN_TOKEN = re.compile(
    r"^(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x?[a-h][1-8](?:=[QRBN])?|[a-h][1-8](?:=[QRBN])?|"
    r"O-O(?:-O)?|0-0(?:-0)?)[+#]?[!?]{0,2}$"
)
_UCI_TOKEN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
_FEN_RE = re.compile(
    r"([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s+[wb]\s+(?:[KQkq]{1,4}|-)\s+(?:[a-h][36]|-)"
    r"(?:\s+\d+\s+\d+)?"
)

#: A move list must be introduced by a move number or an explicit cue, otherwise
#: stray words like "be4" or "e4" inside prose would be read as moves.
_MOVES_CUE = re.compile(
    r"(?:after|from|continue[s]?(?: from)?|position|line|following|starting with|begins? with|opening)\s*:?\s*",
    re.IGNORECASE,
)


def _strip_suffix(token: str) -> str:
    return token.rstrip("!?")


def _normalise_castling(token: str) -> str:
    return token.replace("0", "O")


def extract_fen(text: str) -> Optional[str]:
    """Find a FEN in free text, if present."""
    match = _FEN_RE.search(text)
    if not match:
        return None
    fen = match.group(0).strip()
    parts = fen.split()
    if len(parts) == 4:
        fen = fen + " 0 1"
    return fen


def parse_move_tokens(text: str) -> List[str]:
    """Extract SAN/UCI move tokens from a move-list-like string.

    The move list must start immediately: the first token that is neither a move
    number nor a move ends the scan. Without that rule, stray words that happen to
    look like moves would be picked up out of surrounding prose - a FEN's
    en-passant field ("... w KQkq c6 0 2") being the classic trap.
    """
    tokens: List[str] = []
    for raw in re.split(r"[\s,;]+", text.strip()):
        if not raw:
            continue
        token = raw.strip()
        if _MOVE_NUMBER.match(token):
            continue
        # Split "1.e4" and "1...c5" into their move part.
        merged = re.match(r"^\d+\.(?:\.\.)?(.+)$", token)
        if merged:
            token = merged.group(1)
        token = _normalise_castling(token)
        if _SAN_TOKEN.match(token) or _UCI_TOKEN.match(token.lower()):
            tokens.append(token)
        else:
            # Result markers and anything unrecognised end the move list.
            break
    return tokens


def apply_moves(
    board: chess.Board, tokens: Sequence[str], *, strict: bool = True
) -> Tuple[List[chess.Move], List[Dict[str, Any]]]:
    """Play ``tokens`` on ``board``.

    Returns the moves accepted and a list of problems. With ``strict`` the first
    bad token raises :class:`IllegalMoveError`; otherwise parsing stops there and
    the caller continues from the last legal position.
    """
    moves: List[chess.Move] = []
    problems: List[Dict[str, Any]] = []
    for index, token in enumerate(tokens):
        clean = _strip_suffix(token)
        move: Optional[chess.Move] = None
        try:
            move = board.parse_san(clean)
        except ValueError:
            if _UCI_TOKEN.match(clean.lower()):
                try:
                    candidate = chess.Move.from_uci(clean.lower())
                    if candidate in board.legal_moves:
                        move = candidate
                except ValueError:
                    move = None
        if move is None:
            legal = sorted(board.san(m) for m in board.legal_moves)[:12]
            detail = {
                "token": token,
                "index": index,
                "fen": board.fen(),
                "ply": len(board.move_stack),
                "legal_sample": legal,
                "accepted_prefix": [m.uci() for m in moves],
            }
            if strict:
                raise IllegalMoveError(
                    f"{token!r} is not a legal move in the position after "
                    f"{len(moves)} move(s) ({board.fen()})",
                    move=token,
                    fen=board.fen(),
                    ply=len(board.move_stack),
                    legal_sample=legal,
                    accepted_prefix=[m.uci() for m in moves],
                )
            problems.append(detail)
            break
        moves.append(move)
        board.push(move)
    return moves, problems


# --------------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    """A parsed generation request."""

    text: str = ""
    mode: Optional[str] = None
    style: Optional[str] = None
    #: Side whose perspective drives the analysis.
    side: Optional[str] = None
    opening_query: Optional[str] = None
    #: Resolved book entry for the requested opening, when one was found.
    opening_entry: Optional[BookEntry] = None
    alternates: List[BookEntry] = field(default_factory=list)
    start_fen: Optional[str] = None
    #: Moves the caller supplied, which the output must begin with.
    start_moves: List[str] = field(default_factory=list)
    start_moves_uci: List[str] = field(default_factory=list)
    #: Constraints of the form "against 1...e5": ``(ply_index, san)``, resolved
    #: against the book into :attr:`start_moves` when possible.
    constraints: List[Tuple[int, str]] = field(default_factory=list)
    #: Config overrides implied by the wording.
    overrides: Dict[str, Any] = field(default_factory=dict)
    #: Human-readable log of every inference made while parsing.
    inferences: List[str] = field(default_factory=list)
    #: Non-fatal problems (e.g. an opening name that could not be resolved).
    warnings: List[str] = field(default_factory=list)
    #: Free-text intent notes, used for the PGN intro comment.
    focus: List[str] = field(default_factory=list)

    def summary(self) -> str:
        pieces: List[str] = []
        if self.opening_entry is not None:
            pieces.append(self.opening_entry.name)
        elif self.opening_query:
            pieces.append(self.opening_query)
        if self.mode:
            pieces.append(f"{self.mode} mode")
        if self.style:
            pieces.append(self.style.replace("_", " "))
        return ", ".join(pieces)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode,
            "style": self.style,
            "side": self.side,
            "opening_query": self.opening_query,
            "opening": self.opening_entry.to_dict() if self.opening_entry else None,
            "start_fen": self.start_fen,
            "start_moves": list(self.start_moves),
            "constraints": [{"ply": ply, "san": san} for ply, san in self.constraints],
            "overrides": dict(self.overrides),
            "inferences": list(self.inferences),
            "warnings": list(self.warnings),
            "focus": list(self.focus),
        }


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #


class RequestParser:
    """Interprets natural-language opening requests."""

    def __init__(self, book: OpeningBook) -> None:
        self.book = book

    def parse(
        self,
        text: str,
        *,
        explicit: Optional[Dict[str, Any]] = None,
        strict_moves: bool = True,
    ) -> Request:
        """Parse ``text``, with ``explicit`` values (CLI flags) taking precedence."""
        explicit = dict(explicit or {})
        request = Request(text=text or "")
        lowered = " " + normalize_name(text or "") + " "

        self._parse_mode(lowered, request)
        self._parse_style(lowered, request)
        self._parse_side(lowered, request)
        self._parse_shape(lowered, request)
        self._parse_focus(lowered, request)

        # Position: explicit FEN/moves win over anything in the prose.
        self._parse_position(text or "", request, explicit=explicit, strict_moves=strict_moves)
        self._parse_opening(text or "", request, explicit=explicit)

        # Explicit flags override inferences.
        for key in ("mode", "style", "side"):
            if explicit.get(key):
                setattr(request, key, explicit[key])
                request.inferences.append(f"{key} set explicitly to {explicit[key]!r}")
        return request

    # -- individual aspects ------------------------------------------------ #

    def _parse_mode(self, lowered: str, request: Request) -> None:
        for pattern, mode in _MODE_PATTERNS:
            if re.search(pattern, lowered):
                request.mode = mode
                request.inferences.append(f"mode={mode} (matched wording in the request)")
                return

    def _parse_style(self, lowered: str, request: Request) -> None:
        for pattern, style in _STYLE_PATTERNS:
            if re.search(pattern, lowered):
                request.style = style
                request.inferences.append(f"style={style} (matched wording in the request)")
                return

    def _parse_side(self, lowered: str, request: Request) -> None:
        white = bool(_SIDE_WHITE.search(lowered))
        black = bool(_SIDE_BLACK.search(lowered))
        if white and not black:
            request.side = "white"
        elif black and not white:
            request.side = "black"
        elif white and black:
            # "for White against Black's KID" - the first mention is the subject.
            first_white = _SIDE_WHITE.search(lowered)
            first_black = _SIDE_BLACK.search(lowered)
            request.side = "white" if first_white.start() < first_black.start() else "black"
        if request.side:
            request.inferences.append(f"side={request.side}")

    def _parse_shape(self, lowered: str, request: Request) -> None:
        overrides = request.overrides
        match = _LONG.search(lowered)
        if match:
            count = int(match.group(1))
            if 3 <= count <= 40:
                overrides["main_line_moves"] = count
                request.inferences.append(f"main_line_moves={count} (explicit move count)")
        elif _DEEP.search(lowered):
            overrides["main_line_moves"] = 14
            request.inferences.append("main_line_moves=14 (asked for a deep line)")
        elif _SHORT.search(lowered):
            overrides["main_line_moves"] = 7
            request.inferences.append("main_line_moves=7 (asked for a short line)")

        match = _VARIATION_COUNT.search(lowered)
        if match:
            count = int(match.group(1))
            if 0 <= count <= 12:
                overrides["variations"] = count
                request.inferences.append(f"variations={count} (explicit count)")

        output: Dict[str, Any] = overrides.setdefault("output", {})
        if _NO_COMMENTS.search(lowered):
            # "moves only" means bare moves: prose, evaluations and arrows all go.
            output["comments"] = False
            output["annotations"] = "minimal"
            output["evals"] = "none"
            output["arrows"] = False
            request.inferences.append("bare output requested: comments, evals and arrows disabled")
        if _NO_ARROWS.search(lowered):
            output["arrows"] = False
            request.inferences.append("arrows disabled")
        elif _WITH_ARROWS.search(lowered):
            output["arrows"] = True
            request.inferences.append("arrows enabled")
        if _NO_EVALS.search(lowered):
            output["evals"] = "none"
            request.inferences.append("engine evaluations disabled")
        elif _WITH_EVALS.search(lowered):
            output["evals"] = "all"
            request.inferences.append("engine evaluations on every move")
        if _RARE.search(lowered):
            # Rare lines live outside the main book: relax theory preference and
            # widen the engine window so offbeat-but-sound moves can win.
            overrides["prefer_theory"] = False
            overrides["style_cp_tolerance"] = max(
                60, int(overrides.get("style_cp_tolerance", 45))
            )
            overrides.setdefault("engine", {})["multipv"] = 6
            request.inferences.append(
                "rare/offbeat requested: theory preference relaxed, engine window widened"
            )
        if not output:
            overrides.pop("output", None)

    def _parse_focus(self, lowered: str, request: Request) -> None:
        if _KINGSIDE.search(lowered):
            request.focus.append("kingside")
        if _QUEENSIDE.search(lowered):
            request.focus.append("queenside")
        if re.search(r"\binitiative\b|\bcompensation\b|\bpressure\b", lowered):
            request.focus.append("initiative")
        if re.search(r"\bcentre\b|\bcenter\b", lowered):
            request.focus.append("centre")
        if re.search(r"\bnovelty\b|\bnew move\b|\btheoretical novelty\b", lowered):
            request.focus.append("novelty")
        if re.search(r"\btransposition\b|\bmove order\b", lowered):
            request.focus.append("transposition")
        if request.focus:
            request.inferences.append("focus: " + ", ".join(request.focus))

    # -- position ---------------------------------------------------------- #

    def _parse_position(
        self,
        text: str,
        request: Request,
        *,
        explicit: Dict[str, Any],
        strict_moves: bool,
    ) -> None:
        fen = explicit.get("fen") or extract_fen(text)
        if fen:
            board = self._board_from_fen(fen)
            request.start_fen = board.fen()
            request.inferences.append(f"starting position taken from FEN: {request.start_fen}")
            # Strip the FEN out of the text so its trailing fields ("... c6 0 2")
            # cannot be read back as moves.
            text = text.replace(fen, " ")
            trimmed = " ".join(fen.split()[:4])
            text = text.replace(trimmed, " ")

        tokens: List[str] = []
        if explicit.get("moves"):
            raw = explicit["moves"]
            tokens = parse_move_tokens(raw) if isinstance(raw, str) else [str(t) for t in raw]
            if not tokens:
                raise RequestError(f"could not read any move from {raw!r}")
        else:
            tokens, constraints = self._extract_moves_from_prose(text)
            if constraints and not tokens:
                self._resolve_constraints(constraints, request)
                return

        if tokens:
            board = chess.Board(request.start_fen) if request.start_fen else chess.Board()
            moves, problems = apply_moves(board, tokens, strict=strict_moves)
            if problems:
                problem = problems[0]
                request.warnings.append(
                    f"ignored {problem['token']!r} and everything after it: not legal in "
                    f"{problem['fen']}"
                )
            if moves:
                replay = chess.Board(request.start_fen) if request.start_fen else chess.Board()
                sans: List[str] = []
                for move in moves:
                    sans.append(replay.san(move))
                    replay.push(move)
                request.start_moves = sans
                request.start_moves_uci = [m.uci() for m in moves]
                request.inferences.append("continuing from the supplied moves: " + " ".join(sans))

    def _board_from_fen(self, fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise InvalidFENError(f"invalid FEN {fen!r}: {exc}", fen=fen) from exc
        status = board.status()
        if status & chess.STATUS_NO_WHITE_KING or status & chess.STATUS_NO_BLACK_KING:
            raise InvalidFENError(f"FEN {fen!r} is missing a king", fen=fen)
        if status & chess.STATUS_TOO_MANY_KINGS:
            raise InvalidFENError(f"FEN {fen!r} has too many kings", fen=fen)
        if status & chess.STATUS_OPPOSITE_CHECK:
            raise InvalidFENError(
                f"FEN {fen!r} leaves the side that just moved in check", fen=fen
            )
        if board.is_game_over(claim_draw=False):
            raise InvalidFENError(f"FEN {fen!r} is already a finished game", fen=fen)
        return board

    def _extract_moves_from_prose(self, text: str) -> Tuple[List[str], List[Tuple[int, str]]]:
        """Find a move list inside a sentence.

        Returns ``(tokens, constraints)``. ``tokens`` is a playable sequence
        starting from move 1 (or from a supplied FEN). ``constraints`` holds
        partial specifications such as "against 1...e5", where only one side's
        move is given: ``[(1, "e5")]`` means Black's first move must be ...e5.
        Requires either a numbered move or an explicit cue word, so ordinary
        prose is never mistaken for moves.
        """
        # "against 1...e5" / "vs 1...c5": a reply constraint, not a full line.
        constraints: List[Tuple[int, str]] = []
        for match in re.finditer(r"\b(\d+)\s*\.\s*\.\s*\.\s*([A-Za-z][A-Za-z0-9=+#-]*)", text):
            number = int(match.group(1))
            token = _normalise_castling(_strip_suffix(match.group(2)))
            if _SAN_TOKEN.match(token):
                constraints.append((number * 2 - 1, token))

        numbered = re.search(r"\b1\s*\.\s*(?!\s*\.)\s*[A-Za-z]", text)
        if numbered:
            return parse_move_tokens(text[numbered.start():]), constraints
        if constraints:
            return [], constraints
        cue = _MOVES_CUE.search(text)
        if cue:
            tokens = parse_move_tokens(text[cue.end():])
            if tokens:
                return tokens, constraints
        return [], constraints

    def _resolve_constraints(
        self, constraints: Sequence[Tuple[int, str]], request: Request
    ) -> None:
        """Record reply constraints such as "against 1...e5".

        These do not fix a starting line by themselves (White's move is the thing
        being asked for), so they are stored for the generator, which will play
        the constrained reply when it reaches that ply.
        """
        request.constraints = list(constraints)
        readable = ", ".join(f"move {(ply + 1) // 2}...{san}" for ply, san in constraints)
        request.inferences.append(f"opponent reply constrained to {readable}")

    # -- opening ----------------------------------------------------------- #

    def _parse_opening(self, text: str, request: Request, *, explicit: Dict[str, Any]) -> None:
        query = explicit.get("opening")
        if query:
            request.opening_query = query
        else:
            query = self._guess_opening_query(text, request)
            request.opening_query = query
        if not query:
            if request.start_moves or request.start_fen:
                board = self._request_board(request)
                match = self.book.classify(board)
                if match is not None:
                    request.opening_entry = match.entry
                    request.inferences.append(
                        f"position classified as {match.entry.name} ({match.entry.eco})"
                    )
            return

        try:
            entry, alternates = self.book.resolve(query)
        except Exception as exc:  # OpeningNotFoundError
            request.warnings.append(str(exc))
            return
        request.opening_entry = entry
        request.alternates = alternates
        request.inferences.append(f"opening resolved to {entry.name} ({entry.eco}) from {query!r}")

    def _guess_opening_query(self, text: str, request: Request) -> Optional[str]:
        """Pull the opening name out of a sentence.

        Strategy: drop the move list, the FEN and the request vocabulary
        ("show me", "sharp", "trap", ...), then hand whatever remains to the
        book's fuzzy search. When nothing substantive is left, no opening was
        named and ``None`` is returned rather than a guess.
        """
        stripped = text
        numbered = re.search(r"\b1\s*\.\s*(?:\.\.)?\s*[A-Za-z]", stripped)
        if numbered:
            stripped = stripped[: numbered.start()]
        fen = _FEN_RE.search(stripped)
        if fen:
            stripped = stripped[: fen.start()] + " " + stripped[fen.end():]

        cleaned = normalize_name(stripped)
        if not cleaned:
            return None
        residue = _STOPWORDS_EXTRA.sub(" ", cleaned)
        residue = re.sub(r"[^a-z0-9' \-]+", " ", residue)
        residue = re.sub(r"\s+", " ", residue).strip()
        # Need at least one word of four letters or more to be an opening name.
        if not re.search(r"[a-z']{4,}", residue):
            return None
        expanded = expand_aliases(residue)
        return expanded or None

    def _request_board(self, request: Request) -> chess.Board:
        board = chess.Board(request.start_fen) if request.start_fen else chess.Board()
        for uci in request.start_moves_uci:
            board.push(chess.Move.from_uci(uci))
        return board


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def parse_request(
    text: str,
    book: OpeningBook,
    *,
    explicit: Optional[Dict[str, Any]] = None,
    strict_moves: bool = True,
) -> Request:
    return RequestParser(book).parse(text, explicit=explicit, strict_moves=strict_moves)
