"""Stockfish (UCI) integration.

The rule this module enforces: *the engine validates the chess*. Nothing here
ever invents a score. If Stockfish cannot be started, :class:`EngineManager`
degrades to a null engine that returns ``None`` for every analysis and records a
warning, so downstream code can only omit evaluations - never fake them.

Determinism: with ``config.deterministic`` (the default) the search is bounded by
depth and/or nodes, ``Threads`` is pinned to 1 and time-based limits are dropped,
so repeated runs on the same machine return byte-identical PGN.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine

from .config import EngineConfig
from .errors import EngineFailureError, EngineUnavailableError

log = logging.getLogger("pgn_generator.engine")

#: Centipawn magnitude used to represent a mate score when a number is required.
MATE_SCORE_CP = 10_000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Score:
    """An engine score, stored from White's point of view.

    ``mate`` is signed: ``+3`` means White mates in 3, ``-2`` means Black mates
    in 2. Exactly one of :attr:`cp` / :attr:`mate` is meaningful, but ``cp`` is
    always populated (mate scores are mapped onto a large centipawn value) so
    comparisons never need special-casing.
    """

    cp: int
    mate: Optional[int] = None

    @classmethod
    def from_pov(cls, pov: chess.engine.PovScore) -> "Score":
        white = pov.white()
        mate = white.mate()
        if mate is not None:
            cp = MATE_SCORE_CP - min(abs(mate), 100) if mate > 0 else -(MATE_SCORE_CP - min(abs(mate), 100))
            return cls(cp=cp, mate=mate)
        raw = white.score()
        return cls(cp=int(raw if raw is not None else 0), mate=None)

    @property
    def is_mate(self) -> bool:
        return self.mate is not None

    def for_side(self, color: chess.Color) -> "Score":
        """Return this score from ``color``'s point of view."""
        if color == chess.WHITE:
            return self
        return Score(cp=-self.cp, mate=None if self.mate is None else -self.mate)

    def cp_for(self, color: chess.Color) -> int:
        return self.cp if color == chess.WHITE else -self.cp

    def to_dict(self) -> Dict[str, Any]:
        return {"cp": self.cp, "mate": self.mate}


@dataclass
class Candidate:
    """One MultiPV line returned by the engine."""

    move: chess.Move
    score: Score
    pv: List[chess.Move] = field(default_factory=list)
    depth: Optional[int] = None
    nodes: Optional[int] = None
    rank: int = 1

    def to_dict(self, board: Optional[chess.Board] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "uci": self.move.uci(),
            "score": self.score.to_dict(),
            "depth": self.depth,
            "rank": self.rank,
        }
        if board is not None:
            try:
                data["san"] = board.san(self.move)
                data["pv_san"] = board.variation_san(self.pv) if self.pv else ""
            except (ValueError, AssertionError):  # pragma: no cover - defensive
                pass
        return data


@dataclass
class Analysis:
    """Result of analysing a single position."""

    fen: str
    turn: chess.Color
    candidates: List[Candidate]
    depth: Optional[int] = None
    critical: bool = False

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def best_score(self) -> Optional[Score]:
        return self.candidates[0].score if self.candidates else None

    def by_move(self, move: chess.Move) -> Optional[Candidate]:
        for cand in self.candidates:
            if cand.move == move:
                return cand
        return None

    def cp_loss(self, move: chess.Move) -> Optional[int]:
        """Centipawns lost by ``move`` relative to the best candidate.

        ``None`` when the move was outside the MultiPV window (the caller then
        has to analyse it explicitly if the number matters).
        """
        cand = self.by_move(move)
        if cand is None or not self.candidates:
            return None
        best = self.candidates[0].score.cp_for(self.turn)
        got = cand.score.cp_for(self.turn)
        return max(0, best - got)


# --------------------------------------------------------------------------- #
# Engine wrapper
# --------------------------------------------------------------------------- #


class EngineManager:
    """Lazily-started Stockfish wrapper with caching and graceful degradation.

    Use as a context manager so the subprocess is always reaped::

        with EngineManager(cfg.engine) as engine:
            info = engine.analyse(board)
    """

    def __init__(self, config: EngineConfig, *, deterministic: bool = True) -> None:
        self.config = config
        self.deterministic = deterministic
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._started = False
        self._failed = False
        self._cache: Dict[Tuple[str, int, int, Optional[int], Optional[int]], Analysis] = {}
        self.warnings: List[str] = []
        self.engine_path: Optional[str] = None
        self.engine_id: Dict[str, str] = {}
        self.positions_analysed = 0
        self.cache_hits = 0

    # -- lifecycle --------------------------------------------------------- #

    def __enter__(self) -> "EngineManager":
        if self.config.enabled:
            self.start()
        else:
            self._warn("Engine disabled by configuration: no engine validation was performed.")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> bool:
        """Attempt to start the engine. Returns ``True`` when available."""
        if self._started:
            return self._engine is not None
        self._started = True
        if not self.config.enabled:
            return False

        path = self.config.resolved_path()
        if not path:
            msg = (
                "Stockfish binary not found. Set engine.path, PGNGEN_ENGINE_PATH, or install "
                "Stockfish 16+ on PATH."
            )
            if self.config.required or not self.config.allow_fallback:
                raise EngineUnavailableError(msg)
            self._failed = True
            self._warn("ENGINE VALIDATION UNAVAILABLE: " + msg)
            return False

        try:
            engine = chess.engine.SimpleEngine.popen_uci(
                path, timeout=self.config.startup_timeout_s, debug=False
            )
        except Exception as exc:  # noqa: BLE001 - any spawn failure degrades the same way
            msg = f"failed to start engine at {path}: {exc}"
            if self.config.required or not self.config.allow_fallback:
                raise EngineUnavailableError(msg, path=path) from exc
            self._failed = True
            self._warn("ENGINE VALIDATION UNAVAILABLE: " + msg)
            return False

        self._engine = engine
        self.engine_path = path
        self.engine_id = dict(getattr(engine, "id", {}) or {})
        self._configure()
        return True

    def _configure(self) -> None:
        engine = self._engine
        if engine is None:
            return
        available = set(engine.options.keys())
        options: Dict[str, Any] = {}
        threads = 1 if self.deterministic else self.config.threads
        if "Threads" in available:
            options["Threads"] = threads
        if "Hash" in available:
            options["Hash"] = self.config.hash_mb
        if self.config.syzygy_path and "SyzygyPath" in available:
            options["SyzygyPath"] = os.path.expanduser(self.config.syzygy_path)
        for key, value in self.config.uci_options.items():
            if key in available:
                options[key] = value
            else:
                self._warn(f"engine option {key!r} is not supported by this binary; ignored")
        if options:
            try:
                engine.configure(options)
            except Exception as exc:  # noqa: BLE001
                self._warn(f"engine configure failed ({exc}); continuing with defaults")

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                try:
                    self._engine.close()
                except Exception:  # noqa: BLE001
                    pass
            self._engine = None

    # -- state ------------------------------------------------------------- #

    @property
    def available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self._started:
            return self.start()
        return self._engine is not None

    @property
    def name(self) -> Optional[str]:
        return self.engine_id.get("name")

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        log.warning(message)

    def info(self) -> Dict[str, Any]:
        """Provenance block embedded in the result envelope."""
        return {
            "available": self.available,
            "name": self.name,
            "path": self.engine_path,
            "depth": self.config.depth,
            "critical_depth": self.config.critical_depth,
            "nodes": self.config.nodes,
            "multipv": self.config.multipv,
            "threads": 1 if self.deterministic else self.config.threads,
            "hash_mb": self.config.hash_mb,
            "deterministic": self.deterministic,
            "positions_analysed": self.positions_analysed,
            "cache_hits": self.cache_hits,
        }

    # -- analysis ---------------------------------------------------------- #

    def _limit(self, critical: bool, depth_override: Optional[int] = None) -> chess.engine.Limit:
        cfg = self.config
        if depth_override is not None:
            return chess.engine.Limit(depth=max(1, min(60, depth_override)))
        depth = cfg.critical_depth if critical else cfg.depth
        nodes = cfg.critical_nodes if critical else cfg.nodes
        time_ms = None if self.deterministic else (cfg.critical_time_ms if critical else cfg.time_ms)
        kwargs: Dict[str, Any] = {}
        if depth:
            kwargs["depth"] = depth
        if nodes:
            kwargs["nodes"] = nodes
        if time_ms:
            kwargs["time"] = time_ms / 1000.0
        if not kwargs:  # never allow an unbounded search
            kwargs["depth"] = 12
        return chess.engine.Limit(**kwargs)

    def analyse(
        self,
        board: chess.Board,
        *,
        critical: bool = False,
        multipv: Optional[int] = None,
        root_moves: Optional[Sequence[chess.Move]] = None,
        depth: Optional[int] = None,
    ) -> Optional[Analysis]:
        """Analyse ``board``.

        ``depth`` overrides the configured limits entirely, which the trap finder
        uses for deliberately shallow "what a human sees at a glance" searches.
        Returns ``None`` when the engine is unavailable (caller must then avoid
        claiming engine validation) or when the position is already terminal.
        """
        if board.is_game_over(claim_draw=False):
            return Analysis(fen=board.fen(), turn=board.turn, candidates=[], critical=critical)
        if not self.available:
            return None

        want = multipv if multipv is not None else (
            self.config.critical_multipv if critical else self.config.multipv
        )
        legal_count = board.legal_moves.count()
        if legal_count == 0:
            return Analysis(fen=board.fen(), turn=board.turn, candidates=[], critical=critical)
        want = max(1, min(want, legal_count if root_moves is None else len(list(root_moves))))

        limit = self._limit(critical, depth)
        cache_key = (
            board.epd(en_passant="legal") + f" {board.turn}",
            want,
            limit.depth or 0,
            limit.nodes,
            None if root_moves is None else hash(tuple(sorted(m.uci() for m in root_moves))),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        engine = self._engine
        assert engine is not None
        try:
            raw = engine.analyse(
                board,
                limit,
                multipv=want,
                root_moves=list(root_moves) if root_moves else None,
                info=chess.engine.INFO_ALL,
            )
        except chess.engine.EngineTerminatedError as exc:
            self._engine = None
            self._failed = True
            msg = f"engine terminated during analysis: {exc}"
            if self.config.required or not self.config.allow_fallback:
                raise EngineFailureError(msg, fen=board.fen()) from exc
            self._warn("ENGINE VALIDATION INCOMPLETE: " + msg)
            return None
        except Exception as exc:  # noqa: BLE001
            msg = f"engine analysis failed: {exc}"
            if self.config.required or not self.config.allow_fallback:
                raise EngineFailureError(msg, fen=board.fen()) from exc
            self._warn("ENGINE VALIDATION INCOMPLETE: " + msg)
            return None

        infos = raw if isinstance(raw, list) else [raw]
        candidates: List[Candidate] = []
        for info in infos:
            pv = list(info.get("pv") or [])
            if not pv:
                continue
            pov = info.get("score")
            if pov is None:
                continue
            candidates.append(
                Candidate(
                    move=pv[0],
                    score=Score.from_pov(pov),
                    pv=pv,
                    depth=info.get("depth"),
                    nodes=info.get("nodes"),
                    rank=int(info.get("multipv") or (len(candidates) + 1)),
                )
            )
        candidates.sort(key=lambda c: c.rank)
        # Guard against duplicate root moves across MultiPV entries.
        seen: set = set()
        unique: List[Candidate] = []
        for cand in candidates:
            if cand.move in seen:
                continue
            seen.add(cand.move)
            unique.append(cand)

        analysis = Analysis(
            fen=board.fen(),
            turn=board.turn,
            candidates=unique,
            depth=limit.depth,
            critical=critical,
        )
        self.positions_analysed += 1
        self._cache[cache_key] = analysis
        return analysis

    def score_move(
        self, board: chess.Board, move: chess.Move, *, critical: bool = False
    ) -> Optional[Score]:
        """Score a specific move by searching it as the only root move."""
        if move not in board.legal_moves:
            return None
        analysis = self.analyse(board, critical=critical, multipv=1, root_moves=[move])
        if analysis is None or not analysis.candidates:
            return None
        return analysis.candidates[0].score

    def evaluate(self, board: chess.Board, *, critical: bool = False) -> Optional[Score]:
        """Static-ish evaluation of ``board`` (search score of the best move)."""
        if board.is_checkmate():
            # Side to move is mated: score is decisive for the other side.
            return Score(cp=-MATE_SCORE_CP if board.turn == chess.WHITE else MATE_SCORE_CP, mate=0)
        if board.is_game_over(claim_draw=False):
            return Score(cp=0, mate=None)
        analysis = self.analyse(board, critical=critical, multipv=1)
        if analysis is None or not analysis.candidates:
            return None
        return analysis.candidates[0].score
