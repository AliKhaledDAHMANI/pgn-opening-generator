"""ECO opening book: classification, theory continuations and name resolution.

Data source: the Lichess ``chess-openings`` data set (CC0, public domain), shipped
in ``pgn_generator/data/eco/*.tsv``. Each row is ``eco``, ``name``, ``pgn`` where
``pgn`` is the canonical shortest move sequence reaching that named position.

Two things this module deliberately does *not* do:

* invent opening names or ECO codes - every label returned here comes from the
  data set;
* claim master-game statistics. The popularity signal exposed as
  :attr:`TheoryMove.breadth` is the number of *named book lines* in the subtree,
  i.e. how much catalogued theory exists behind a move. It is a structural
  measure, not a game count.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chess

from .errors import OpeningNotFoundError

#: Data ships inside the package, so an installed copy works without the repo.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ECO_DIR = os.path.join(DATA_DIR, "eco")
ECO_VOLUMES = ("a", "b", "c", "d", "e")
#: Compiled index (gzipped JSON). Holds the whole transposition graph so that
#: start-up needs no SAN parsing and no move pushing at all.
INDEX_PATH = os.path.join(ECO_DIR, "index.json.gz")
INDEX_VERSION = 3


def _epd(board: chess.Board) -> str:
    """Position key: EPD with a legal-only en-passant field (matches Lichess)."""
    return board.epd(en_passant="legal")


# --------------------------------------------------------------------------- #
# Name normalisation and aliases
# --------------------------------------------------------------------------- #

_ALIASES: Dict[str, str] = {
    "spanish game": "ruy lopez",
    "spanish opening": "ruy lopez",
    "spanish": "ruy lopez",
    "kid": "king's indian defense",
    "qgd": "queen's gambit declined",
    "qga": "queen's gambit accepted",
    "kia": "king's indian attack",
    "nimzo": "nimzo-indian defense",
    "grunfeld": "grünfeld defense",
    "gruenfeld": "grünfeld defense",
    "caro kann": "caro-kann defense",
    "petroff": "petrov's defense",
    "petrov": "petrov's defense",
    "russian game": "petrov's defense",
    "berlin": "ruy lopez berlin defense",
    "italian": "italian game",
    "giuoco pianissimo": "italian game giuoco pianissimo",
    "two knights": "italian game two knights defense",
    "sicilian": "sicilian defense",
    "najdorf": "najdorf variation",
    "dragon": "sicilian defense dragon variation",
    "sveshnikov": "sicilian defense sveshnikov",
    "scheveningen": "sicilian defense scheveningen",
    "accelerated dragon": "sicilian defense accelerated dragon",
    "french": "french defense",
    "winawer": "french defense winawer variation",
    "slav": "slav defense",
    "semi slav": "semi-slav defense",
    "catalan": "catalan opening",
    "london": "london system",
    "queens gambit": "queen's gambit",
    "kings gambit": "king's gambit",
    "vienna": "vienna game",
    "scotch": "scotch game",
    "english": "english opening",
    "reti": "réti opening",
    "birds opening": "bird's opening",
    "philidor": "philidor defense",
    "pirc": "pirc defense",
    "modern defence": "modern defense",
    "alekhine": "alekhine defense",
    "scandinavian": "scandinavian defense",
    "center counter": "scandinavian defense",
    "budapest": "budapest defense",
    "benko": "benko gambit",
    "volga": "benko gambit",
    "evans gambit": "italian game evans gambit",
    "fried liver": "italian game two knights defense fried liver attack",
    "traxler": "italian game two knights defense traxler counterattack",
    "wilkes barre": "italian game two knights defense traxler counterattack",
    "max lange": "italian game max lange attack",
    "marshall": "ruy lopez marshall attack",
    "open spanish": "ruy lopez open variation",
    "exchange spanish": "ruy lopez exchange variation",
    "smith morra": "sicilian defense smith-morra gambit",
    "morra": "sicilian defense smith-morra gambit",
    "grand prix": "sicilian defense grand prix attack",
    "closed sicilian": "sicilian defense closed",
    "rossolimo": "sicilian defense nyezhmetdinov-rossolimo attack",
    "moscow variation": "sicilian defense canal attack",
}

#: Bare family names that need a qualifier added, but only when the user did not
#: already supply one. "King's Indian" means the Defense; "King's Indian Attack"
#: must be left alone.
_QUALIFIED_ALIASES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "king's indian": ("king's indian defense", ("attack", "defense", "variation", "system")),
    "queen's indian": ("queen's indian defense", ("attack", "defense", "variation")),
    "kings indian": ("king's indian defense", ("attack", "defense", "variation", "system")),
    "queens indian": ("queen's indian defense", ("attack", "defense", "variation")),
    "nimzo indian": ("nimzo-indian defense", ("defense", "variation")),
    "old indian": ("old indian defense", ("defense", "variation")),
}

_DEFENCE_RE = re.compile(r"\bdefence\b")
_PUNCT_RE = re.compile(r"[^a-z0-9' \-]+")
_WS_RE = re.compile(r"\s+")

_STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "me", "show", "give", "create", "generate",
    "build", "make", "please", "line", "lines", "main", "mainline", "variation",
    "variations", "opening", "openings", "against", "with", "play", "best", "some",
    "sharp", "aggressive", "solid", "positional", "please", "using", "and", "or",
    "moves", "move", "pgn", "analysis", "style", "gm", "grandmaster", "theory",
    "theoretical", "typical", "modern", "classic",
}

#: Words that appear in hundreds of opening names and therefore carry no
#: identifying information on their own.
_GENERIC_TERMS = {
    "defense", "defence", "attack", "gambit", "game", "system", "opening",
    "variation", "accepted", "declined", "deferred", "counterattack", "counter",
    "classical", "modern", "old", "new", "main", "normal", "closed", "open",
    "advance", "exchange", "quiet", "wing", "center", "centre", "double", "three",
    "two", "four", "knights", "knight", "bishop", "rook", "queen", "king", "pawn",
    "line", "lines", "defensive", "aggressive",
}

#: Minimum :meth:`OpeningBook.search` score for :meth:`OpeningBook.resolve` to
#: accept a match. Calibrated so real names pass and names that only share generic
#: vocabulary are rejected.
_RESOLVE_THRESHOLD = 3.2


def normalize_name(text: str) -> str:
    """Lowercase, strip accents/punctuation and unify UK/US spellings."""
    lowered = unicodedata.normalize("NFKD", text.lower())
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    lowered = _DEFENCE_RE.sub("defense", lowered)
    lowered = lowered.replace("&", " and ")
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def expand_aliases(query: str) -> str:
    """Rewrite common shorthand ("Spanish", "KID", "Najdorf") to book wording.

    Each alias is applied at most once and never inside a phrase it has already
    produced, so "Sicilian Najdorf" does not expand into a repeated family name.
    """
    norm = normalize_name(query)
    if norm in _ALIASES:
        return normalize_name(_ALIASES[norm])

    out = norm
    for bare, (replacement, blockers) in _QUALIFIED_ALIASES.items():
        pattern = r"\b" + re.escape(bare) + r"\b"
        if re.search(pattern, out) and not any(word in out for word in blockers):
            out = re.sub(pattern, replacement, out, count=1)

    for alias, canonical in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if not alias:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        if not re.search(pattern, out):
            continue
        replacement = normalize_name(canonical)
        # Skip when the canonical wording is already present: "sicilian defense
        # najdorf" must not become "sicilian defense sicilian defense najdorf".
        if replacement in out:
            continue
        out = re.sub(pattern, replacement, out, count=1)
    return _WS_RE.sub(" ", out).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in normalize_name(text).replace("-", " ").split() if t]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BookEntry:
    """One named line from the ECO data set."""

    eco: str
    name: str
    uci: Tuple[str, ...]
    epd: str

    @property
    def family(self) -> str:
        return self.name.split(":", 1)[0].strip()

    @property
    def tail(self) -> str:
        return self.name.split(":", 1)[1].strip() if ":" in self.name else ""

    @property
    def variation(self) -> str:
        tail = self.tail
        return tail.split(",")[0].strip() if tail else ""

    @property
    def subvariation(self) -> str:
        tail = self.tail
        parts = [p.strip() for p in tail.split(",")] if tail else []
        return ", ".join(parts[1:]) if len(parts) > 1 else ""

    @property
    def ply_count(self) -> int:
        return len(self.uci)

    def board(self) -> chess.Board:
        board = chess.Board()
        for uci in self.uci:
            board.push(chess.Move.from_uci(uci))
        return board

    def san_line(self) -> str:
        board = chess.Board()
        moves = [chess.Move.from_uci(u) for u in self.uci]
        return board.variation_san(moves) if moves else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eco": self.eco,
            "name": self.name,
            "family": self.family,
            "variation": self.variation,
            "subvariation": self.subvariation,
            "uci": list(self.uci),
            "plies": self.ply_count,
        }


@dataclass
class BookMatch:
    """Result of classifying a position against the book."""

    entry: BookEntry
    #: Plies of the classified position (may exceed ``entry.ply_count`` when the
    #: name was found by walking backwards through a transposition).
    matched_ply: int
    exact: bool

    @property
    def eco(self) -> str:
        return self.entry.eco

    @property
    def name(self) -> str:
        return self.entry.name


@dataclass
class TheoryMove:
    """A book continuation from some position."""

    move: chess.Move
    san: str
    #: Number of named book lines reachable through this move (theory breadth).
    breadth: int
    #: Direct name attached to the resulting position, when one exists.
    entry: Optional[BookEntry] = None
    #: Names of notable lines behind this move (sampled, deepest-first).
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "san": self.san,
            "uci": self.move.uci(),
            "breadth": self.breadth,
            "name": self.entry.name if self.entry else None,
            "eco": self.entry.eco if self.entry else None,
        }


@dataclass
class _Node:
    epd: str
    children: Dict[str, str] = field(default_factory=dict)   # uci -> child epd
    entries: List[int] = field(default_factory=list)          # indices into OpeningBook.entries
    breadth: int = 0                                          # named lines in subtree
    max_depth: int = 0                                        # deepest ply reachable


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #


class OpeningBook:
    """In-memory ECO book with an epd-keyed transposition graph."""

    def __init__(
        self,
        entries: Sequence[BookEntry],
        *,
        nodes: Optional[Dict[str, _Node]] = None,
    ) -> None:
        self.entries: List[BookEntry] = list(entries)
        self.nodes: Dict[str, _Node] = nodes if nodes is not None else {}
        self._by_epd: Dict[str, List[int]] = {}
        self.root_epd = _epd(chess.Board())
        if nodes is None:
            self._build_graph()
        else:
            for index, entry in enumerate(self.entries):
                self._by_epd.setdefault(entry.epd, []).append(index)
                node = self.nodes.get(entry.epd)
                if node is not None and index not in node.entries:
                    node.entries.append(index)

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, path: Optional[str] = None) -> "OpeningBook":
        """Load the book, preferring the compiled index for speed."""
        directory = path or ECO_DIR
        if os.path.isdir(directory):
            for name in ("index.json.gz", "index.json"):
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate):
                    try:
                        return cls._from_index(candidate)
                    except (ValueError, KeyError, OSError, json.JSONDecodeError):
                        continue  # fall through to TSV parsing
            return cls._from_tsv(directory)
        if os.path.isfile(directory):
            return cls._from_index(directory)
        raise OpeningNotFoundError(f"ECO data path not found: {directory!r}")

    @classmethod
    def _from_index(cls, path: str) -> "OpeningBook":
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            payload = json.load(handle)
        if int(payload.get("version", 0)) != INDEX_VERSION:
            raise ValueError(f"index version {payload.get('version')!r} != {INDEX_VERSION}")

        epds: List[str] = payload["epds"]
        raw_children: List[str] = payload["children"]
        breadth: List[int] = payload["breadth"]
        depth: List[int] = payload["depth"]
        if not (len(epds) == len(raw_children) == len(breadth) == len(depth)):
            raise ValueError("corrupt index: column length mismatch")

        nodes: Dict[str, _Node] = {}
        for i, epd in enumerate(epds):
            nodes[epd] = _Node(epd=epd, breadth=breadth[i], max_depth=depth[i])
        for i, packed in enumerate(raw_children):
            if not packed:
                continue
            node = nodes[epds[i]]
            for part in packed.split(" "):
                uci, child_id = part.rsplit(":", 1)
                node.children[uci] = epds[int(child_id)]

        entries = [
            BookEntry(
                eco=row["eco"],
                name=row["name"],
                uci=tuple(row["uci"].split()),
                epd=epds[int(row["node"])],
            )
            for row in payload["entries"]
        ]
        return cls(entries, nodes=nodes)

    @classmethod
    def _from_tsv(cls, directory: str) -> "OpeningBook":
        entries: List[BookEntry] = []
        found = False
        for volume in ECO_VOLUMES:
            tsv = os.path.join(directory, f"{volume}.tsv")
            if not os.path.isfile(tsv):
                continue
            found = True
            with open(tsv, "r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    board = chess.Board()
                    ucis: List[str] = []
                    for token in row["pgn"].split():
                        if token.endswith(".") or token in ("1-0", "0-1", "1/2-1/2", "*"):
                            continue
                        try:
                            move = board.parse_san(token)
                        except ValueError:  # pragma: no cover - upstream data is clean
                            ucis = []
                            break
                        ucis.append(move.uci())
                        board.push(move)
                    if not ucis:
                        continue
                    entries.append(
                        BookEntry(eco=row["eco"], name=row["name"], uci=tuple(ucis), epd=_epd(board))
                    )
        if not found:
            raise OpeningNotFoundError(
                f"no ECO data found in {directory!r}; expected a.tsv..e.tsv or index.json.gz"
            )
        return cls(entries)

    def to_index_payload(self) -> Dict[str, Any]:
        """Serialise the whole graph (positions, edges, breadth) for fast reload."""
        epds = list(self.nodes.keys())
        ids = {epd: i for i, epd in enumerate(epds)}
        children = [
            " ".join(f"{uci}:{ids[child]}" for uci, child in self.nodes[epd].children.items())
            for epd in epds
        ]
        return {
            "version": INDEX_VERSION,
            "epds": epds,
            "children": children,
            "breadth": [self.nodes[epd].breadth for epd in epds],
            "depth": [self.nodes[epd].max_depth for epd in epds],
            "entries": [
                {"eco": e.eco, "name": e.name, "uci": " ".join(e.uci), "node": ids[e.epd]}
                for e in self.entries
            ],
        }

    # -- graph ------------------------------------------------------------- #

    def _node(self, epd: str) -> _Node:
        node = self.nodes.get(epd)
        if node is None:
            node = _Node(epd=epd)
            self.nodes[epd] = node
        return node

    def _build_graph(self) -> None:
        self.root_epd = _epd(chess.Board())
        self._node(self.root_epd)
        for index, entry in enumerate(self.entries):
            board = chess.Board()
            node = self._node(self.root_epd)
            for uci in entry.uci:
                move = chess.Move.from_uci(uci)
                board.push(move)
                child_epd = _epd(board)
                node.children.setdefault(uci, child_epd)
                node = self._node(child_epd)
            node.entries.append(index)
            self._by_epd.setdefault(entry.epd, []).append(index)

        # Breadth / depth via memoised DFS (the graph is a DAG over positions).
        memo: Dict[str, Tuple[int, int]] = {}
        stack: List[Tuple[str, bool]] = [(self.root_epd, False)]
        on_path: set = set()
        while stack:
            epd, expanded = stack.pop()
            if expanded:
                node = self.nodes[epd]
                breadth = len(node.entries)
                depth = 0
                for child_epd in node.children.values():
                    c_breadth, c_depth = memo.get(child_epd, (0, 0))
                    breadth += c_breadth
                    depth = max(depth, c_depth + 1)
                node.breadth = breadth
                node.max_depth = depth
                memo[epd] = (breadth, depth)
                on_path.discard(epd)
                continue
            if epd in memo:
                continue
            if epd in on_path:  # pragma: no cover - defensive against data cycles
                continue
            on_path.add(epd)
            stack.append((epd, True))
            for child_epd in self.nodes[epd].children.values():
                if child_epd not in memo:
                    stack.append((child_epd, False))

    # -- lookups ----------------------------------------------------------- #

    def entries_at(self, board: chess.Board) -> List[BookEntry]:
        """Names attached to exactly this position."""
        return [self.entries[i] for i in self._by_epd.get(_epd(board), [])]

    def contains(self, board: chess.Board) -> bool:
        return _epd(board) in self.nodes

    def classify(self, board: chess.Board) -> Optional[BookMatch]:
        """Classify a position, walking moves backwards on transpositions.

        Mirrors the recommended Lichess procedure: use the deepest named position
        found by undoing moves from the current one.
        """
        exact = self.entries_at(board)
        if exact:
            best = max(exact, key=lambda e: (e.ply_count, -len(e.name)))
            return BookMatch(entry=best, matched_ply=len(board.move_stack), exact=True)

        probe = board.copy(stack=True)
        undone = 0
        while probe.move_stack:
            probe.pop()
            undone += 1
            found = self.entries_at(probe)
            if found:
                best = max(found, key=lambda e: (e.ply_count, -len(e.name)))
                return BookMatch(
                    entry=best, matched_ply=len(board.move_stack) - undone, exact=False
                )
        return None

    def theory(self, board: chess.Board, *, limit: Optional[int] = None) -> List[TheoryMove]:
        """Book continuations from ``board``, widest theory first."""
        node = self.nodes.get(_epd(board))
        if node is None:
            return []
        out: List[TheoryMove] = []
        for uci, child_epd in node.children.items():
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:  # transposition artefact
                continue
            child = self.nodes.get(child_epd)
            if child is None:
                continue
            direct = [self.entries[i] for i in child.entries]
            entry = max(direct, key=lambda e: (e.ply_count, -len(e.name))) if direct else None
            out.append(
                TheoryMove(
                    move=move,
                    san=board.san(move),
                    breadth=max(1, child.breadth),
                    entry=entry,
                    labels=[e.name for e in direct[:3]],
                )
            )
        out.sort(key=lambda t: (-t.breadth, t.san))
        return out[:limit] if limit else out

    def theory_moves_set(self, board: chess.Board) -> set:
        return {t.move for t in self.theory(board)}

    def deepest_line_through(self, board: chess.Board, move: chess.Move) -> Optional[BookEntry]:
        """The deepest named entry reachable after playing ``move``."""
        probe = board.copy(stack=False)
        probe.push(move)
        node = self.nodes.get(_epd(probe))
        if node is None:
            return None
        best: Optional[BookEntry] = None
        seen: set = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current.epd in seen:
                continue
            seen.add(current.epd)
            for i in current.entries:
                entry = self.entries[i]
                if best is None or entry.ply_count > best.ply_count:
                    best = entry
            for child_epd in current.children.values():
                child = self.nodes.get(child_epd)
                if child is not None:
                    stack.append(child)
        return best

    # -- name search ------------------------------------------------------- #

    def search(self, query: str, *, limit: int = 8) -> List[Tuple[float, BookEntry]]:
        """Rank book entries against a natural-language opening name.

        Scoring is coverage-first: what fraction of the query's words appear in the
        entry name. Generic chess vocabulary ("gambit", "attack", "defense",
        "variation") is discounted, because matching only those words tells us
        nothing - "Frobnicator Attack" must not resolve to "Bongcloud Attack".
        """
        expanded = expand_aliases(query)
        q_tokens = _content_tokens(expanded)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        q_joined = " ".join(q_tokens)
        distinctive = q_set - _GENERIC_TERMS
        # A query made only of generic words has nothing to identify an opening by.
        if not distinctive:
            return []

        scored: List[Tuple[float, BookEntry]] = []
        for entry in self.entries:
            name_norm = normalize_name(entry.name)
            name_tokens = set(_tokens(entry.name))
            overlap = q_set & name_tokens
            distinctive_hits = distinctive & name_tokens
            if not distinctive_hits and q_joined not in name_norm:
                continue

            coverage = len(overlap) / len(q_set)
            distinctive_coverage = len(distinctive_hits) / len(distinctive)
            precision = len(overlap) / max(1, len(name_tokens))
            ratio = SequenceMatcher(None, q_joined, name_norm).ratio()

            score = distinctive_coverage * 3.0 + coverage * 1.0 + precision * 0.8 + ratio * 0.6
            if q_joined and q_joined in name_norm:
                score += 1.5
            if normalize_name(entry.family) == q_joined:
                score += 1.0
            # Prefer canonical (shorter) lines at equal relevance.
            score -= 0.012 * entry.ply_count
            scored.append((score, entry))

        scored.sort(key=lambda pair: (-pair[0], pair[1].ply_count, pair[1].name))
        return scored[:limit]

    def resolve(self, query: str) -> Tuple[BookEntry, List[BookEntry]]:
        """Resolve an opening name to its best entry plus alternates.

        Raises :class:`OpeningNotFoundError` when nothing matches confidently. The
        threshold is set so that a name sharing only generic words with the book
        ("Frobnicator Attack") is rejected rather than silently mapped onto an
        unrelated opening.
        """
        ranked = self.search(query, limit=12)
        if not ranked:
            raise OpeningNotFoundError(
                f"could not resolve opening {query!r} against the ECO book", query=query
            )
        best_score, best_entry = ranked[0]
        if best_score < _RESOLVE_THRESHOLD:
            raise OpeningNotFoundError(
                f"no confident ECO match for {query!r}",
                query=query,
                suggestions=[entry.name for _, entry in ranked[:5]],
            )
        return best_entry, [entry for _, entry in ranked[1:6]]

    def family_entries(self, family: str) -> List[BookEntry]:
        target = normalize_name(family)
        return [e for e in self.entries if normalize_name(e.family) == target]

    # -- stats ------------------------------------------------------------- #

    def stats(self) -> Dict[str, int]:
        return {
            "entries": len(self.entries),
            "positions": len(self.nodes),
            "families": len({e.family for e in self.entries}),
        }


_CACHED_BOOK: Optional[OpeningBook] = None


def get_book(path: Optional[str] = None) -> OpeningBook:
    """Process-wide cached book instance."""
    global _CACHED_BOOK
    if path is not None:
        return OpeningBook.load(path)
    if _CACHED_BOOK is None:
        _CACHED_BOOK = OpeningBook.load()
    return _CACHED_BOOK


def iter_book_lines(entries: Iterable[BookEntry]) -> Iterable[List[chess.Move]]:
    for entry in entries:
        yield [chess.Move.from_uci(u) for u in entry.uci]
