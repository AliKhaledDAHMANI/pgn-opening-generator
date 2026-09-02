"""Configuration objects for the PGN opening generator.

Three layers, merged in this order (later wins):

1. Library defaults defined here.
2. ``PGNGEN_*`` environment variables (engine discovery mostly).
3. Explicit values from the caller (CLI flags / JSON config / kwargs).

Everything an agent can tune lives in :class:`Config`, so a request can be
reproduced exactly from ``result["config"]``.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .errors import ConfigError

# --------------------------------------------------------------------------- #
# Enumerations (plain strings so JSON round-trips cleanly)
# --------------------------------------------------------------------------- #

MODES = ("gm", "engine", "training", "trap", "repertoire")

STYLES = (
    "classical_gm",
    "sharp_tactical",
    "aggressive",
    "solid",
    "positional",
    "gambit",
    "practical",
    "engine_best",
    "theoretical",
)

ANNOTATION_LEVELS = ("none", "minimal", "standard", "rich")
EVAL_MODES = ("none", "critical", "all")
EVAL_FORMATS = ("pawns", "centipawns", "verbose")
SIDES = ("white", "black", "both")

#: Candidate binary names searched on ``PATH`` when no engine path is given.
ENGINE_BINARY_CANDIDATES = (
    "stockfish",
    "stockfish17",
    "stockfish16",
    "stockfish-ubuntu-x86-64-avx2",
    "stockfish-ubuntu-x86-64-bmi2",
    "stockfish-ubuntu-x86-64-sse41-popcnt",
    "stockfish-ubuntu-x86-64",
    "stockfish.exe",
)

#: Fixed locations probed after ``PATH`` (in order).
ENGINE_PATH_HINTS = (
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/games/stockfish",
    "/opt/stockfish/stockfish",
    "~/.local/bin/stockfish",
    "~/.local/share/pgn-generator/engine/stockfish",
)


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    raise ConfigError(f"{name} must be a boolean, got {value!r}")


def _as_int(value: Any, name: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be an integer, got {value!r}") from None
    if minimum is not None and out < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {out}")
    return out


def _as_float(value: Any, name: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if minimum is not None and out < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {out}")
    return out


def _as_choice(value: Any, name: str, choices: tuple) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {value!r}")
    low = value.strip().lower().replace("-", "_").replace(" ", "_")
    if low not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return low


# --------------------------------------------------------------------------- #
# Engine configuration
# --------------------------------------------------------------------------- #


@dataclass
class EngineConfig:
    """Stockfish (UCI) parameters.

    ``depth``/``nodes`` are the deterministic knobs and are preferred. ``time_ms``
    is available for callers who want wall-clock budgeting, but it makes results
    machine-dependent and is therefore ignored while :attr:`deterministic` is set
    on the parent :class:`Config`.
    """

    path: Optional[str] = None
    enabled: bool = True
    #: Fail loudly instead of degrading to legality-only output.
    required: bool = False
    #: Permit legality-only output (with an explicit warning) when the engine is missing.
    allow_fallback: bool = True

    depth: int = 16
    nodes: Optional[int] = None
    time_ms: Optional[int] = None
    multipv: int = 4

    #: Settings used for positions flagged as critical (branch points, tactics).
    critical_depth: int = 20
    critical_nodes: Optional[int] = None
    critical_time_ms: Optional[int] = None
    critical_multipv: int = 6

    threads: int = 1
    hash_mb: int = 128
    syzygy_path: Optional[str] = None
    #: Extra raw UCI options, e.g. ``{"UCI_Chess960": "false"}``.
    uci_options: Dict[str, Any] = field(default_factory=dict)

    #: Seconds allowed for engine start-up and for each analysis call.
    startup_timeout_s: float = 20.0
    analysis_timeout_s: float = 120.0

    def resolved_path(self) -> Optional[str]:
        """Locate a usable engine binary, or return ``None``."""
        if self.path:
            candidate = os.path.expanduser(self.path)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            found = shutil.which(candidate)
            return found
        env_path = os.environ.get("PGNGEN_ENGINE_PATH") or os.environ.get("STOCKFISH_PATH")
        if env_path:
            candidate = os.path.expanduser(env_path)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            found = shutil.which(candidate)
            if found:
                return found
        for name in ENGINE_BINARY_CANDIDATES:
            found = shutil.which(name)
            if found:
                return found
        for hint in ENGINE_PATH_HINTS:
            candidate = os.path.expanduser(hint)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def validate(self) -> None:
        self.depth = _as_int(self.depth, "engine.depth", minimum=1, maximum=60)
        self.critical_depth = _as_int(self.critical_depth, "engine.critical_depth", minimum=1, maximum=60)
        self.multipv = _as_int(self.multipv, "engine.multipv", minimum=1, maximum=32)
        self.critical_multipv = _as_int(self.critical_multipv, "engine.critical_multipv", minimum=1, maximum=64)
        self.threads = _as_int(self.threads, "engine.threads", minimum=1, maximum=256)
        self.hash_mb = _as_int(self.hash_mb, "engine.hash_mb", minimum=1, maximum=131072)
        if self.nodes is not None:
            self.nodes = _as_int(self.nodes, "engine.nodes", minimum=1000)
        if self.critical_nodes is not None:
            self.critical_nodes = _as_int(self.critical_nodes, "engine.critical_nodes", minimum=1000)
        if self.time_ms is not None:
            self.time_ms = _as_int(self.time_ms, "engine.time_ms", minimum=10)
        if self.critical_time_ms is not None:
            self.critical_time_ms = _as_int(self.critical_time_ms, "engine.critical_time_ms", minimum=10)
        self.startup_timeout_s = _as_float(self.startup_timeout_s, "engine.startup_timeout_s", minimum=1.0)
        self.analysis_timeout_s = _as_float(self.analysis_timeout_s, "engine.analysis_timeout_s", minimum=1.0)
        self.enabled = _as_bool(self.enabled, "engine.enabled")
        self.required = _as_bool(self.required, "engine.required")
        self.allow_fallback = _as_bool(self.allow_fallback, "engine.allow_fallback")
        if not isinstance(self.uci_options, dict):
            raise ConfigError("engine.uci_options must be an object")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Presentation configuration
# --------------------------------------------------------------------------- #


@dataclass
class OutputConfig:
    """Controls what ends up inside the PGN text."""

    comments: bool = True
    annotations: str = "standard"       # none | minimal | standard | rich
    #: Emit ``$1``-style NAGs instead of inline ``!``/``?`` suffixes.
    use_nag_codes: bool = False
    evals: str = "critical"             # none | critical | all
    eval_format: str = "pawns"          # pawns | centipawns | verbose
    #: Evaluations are always reported from White's point of view when true.
    eval_white_pov: bool = True
    arrows: bool = True
    arrow_format: str = "cal_csl"       # cal_csl (Lichess/ChessBase-compatible)
    include_eco_headers: bool = True
    #: ``auto`` writes FEN/SetUp only for custom start positions, ``always`` writes
    #: them for every game. Note that a line starting from a custom position always
    #: carries its FEN, whatever this is set to - without it the movetext could not
    #: be replayed, so correctness overrides presentation here.
    include_fen_headers: str = "auto"   # auto | always | never
    columns: Optional[int] = None       # None -> one long move-text line
    event: str = "Opening Analysis"
    site: str = "?"
    date: str = "????.??.??"
    round_: str = "-"
    white: str = "White"
    black: str = "Black"
    result: str = "*"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.comments = _as_bool(self.comments, "output.comments")
        self.annotations = _as_choice(self.annotations, "output.annotations", ANNOTATION_LEVELS)
        self.use_nag_codes = _as_bool(self.use_nag_codes, "output.use_nag_codes")
        self.evals = _as_choice(self.evals, "output.evals", EVAL_MODES)
        self.eval_format = _as_choice(self.eval_format, "output.eval_format", EVAL_FORMATS)
        self.eval_white_pov = _as_bool(self.eval_white_pov, "output.eval_white_pov")
        self.arrows = _as_bool(self.arrows, "output.arrows")
        self.arrow_format = _as_choice(self.arrow_format, "output.arrow_format", ("cal_csl",))
        self.include_eco_headers = _as_bool(self.include_eco_headers, "output.include_eco_headers")
        self.include_fen_headers = _as_choice(
            self.include_fen_headers, "output.include_fen_headers", ("auto", "always", "never")
        )
        if self.columns is not None:
            self.columns = _as_int(self.columns, "output.columns", minimum=20)
        if not isinstance(self.extra_headers, dict):
            raise ConfigError("output.extra_headers must be an object")

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["round"] = data.pop("round_")
        return data


# --------------------------------------------------------------------------- #
# Top-level configuration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """Full generation configuration."""

    mode: str = "gm"
    style: str = "classical_gm"
    #: 0.0 = maximally safe, 1.0 = maximally forcing. Nudges candidate scoring.
    aggressiveness: float = 0.5
    #: Side whose repertoire/plans drive the selection (``repertoire`` mode needs it).
    side: str = "both"

    #: Length of the main line in *full moves* (a move pair). Plies = 2x, minus 1 if odd.
    main_line_moves: int = 10
    #: Hard ceiling on plies, applied after ``main_line_moves``.
    max_plies: Optional[int] = None
    #: Number of sideline branches to attach to the main line.
    variations: int = 2
    #: Depth of each sideline, in plies.
    variation_plies: int = 6
    #: Sidelines are only attached at or before this ply (keeps output opening-shaped).
    variation_max_ply: Optional[int] = None

    #: Stay in the ECO book while a theoretical continuation exists.
    prefer_theory: bool = True
    #: Maximum centipawn loss (vs. engine best) tolerated for a theory move.
    theory_cp_tolerance: int = 60
    #: Maximum centipawn loss tolerated for a style-driven (non-book) move.
    style_cp_tolerance: int = 45
    #: Leave the book only when the engine says it is safe to do so.
    allow_out_of_book: bool = True
    #: Stop as soon as the requested opening has been demonstrated and theory dries up.
    stop_when_out_of_theory: bool = False

    #: Reproducible output: fixed nodes/depth, single thread, no time-based search.
    deterministic: bool = True
    seed: int = 0

    #: Number of repertoire branches for ``repertoire`` mode.
    repertoire_branches: int = 3

    engine: EngineConfig = field(default_factory=EngineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        data = dict(data or {})
        engine_data = data.pop("engine", None) or {}
        output_data = data.pop("output", None) or {}
        if not isinstance(engine_data, dict):
            raise ConfigError("engine must be an object")
        if not isinstance(output_data, dict):
            raise ConfigError("output must be an object")

        if "round" in output_data:
            output_data["round_"] = output_data.pop("round")

        known_top = {f.name for f in dataclasses.fields(cls)} - {"engine", "output"}
        unknown = set(data) - known_top
        if unknown:
            raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")

        known_engine = {f.name for f in dataclasses.fields(EngineConfig)}
        unknown_engine = set(engine_data) - known_engine
        if unknown_engine:
            raise ConfigError(f"unknown engine config keys: {', '.join(sorted(unknown_engine))}")

        known_output = {f.name for f in dataclasses.fields(OutputConfig)}
        unknown_output = set(output_data) - known_output
        if unknown_output:
            raise ConfigError(f"unknown output config keys: {', '.join(sorted(unknown_output))}")

        cfg = cls(engine=EngineConfig(**engine_data), output=OutputConfig(**output_data), **data)
        cfg.apply_environment()
        cfg.validate()
        return cfg

    def apply_environment(self) -> None:
        """Apply ``PGNGEN_*`` environment overrides that were not set explicitly."""
        env = os.environ
        if self.engine.path is None and (env.get("PGNGEN_ENGINE_PATH") or env.get("STOCKFISH_PATH")):
            self.engine.path = env.get("PGNGEN_ENGINE_PATH") or env.get("STOCKFISH_PATH")
        if "PGNGEN_ENGINE_DEPTH" in env:
            self.engine.depth = _as_int(env["PGNGEN_ENGINE_DEPTH"], "PGNGEN_ENGINE_DEPTH", minimum=1, maximum=60)
            self.engine.critical_depth = max(self.engine.critical_depth, self.engine.depth)
        if "PGNGEN_ENGINE_THREADS" in env:
            self.engine.threads = _as_int(env["PGNGEN_ENGINE_THREADS"], "PGNGEN_ENGINE_THREADS", minimum=1)
        if "PGNGEN_ENGINE_HASH" in env:
            self.engine.hash_mb = _as_int(env["PGNGEN_ENGINE_HASH"], "PGNGEN_ENGINE_HASH", minimum=1)
        if "PGNGEN_DISABLE_ENGINE" in env and _as_bool(env["PGNGEN_DISABLE_ENGINE"], "PGNGEN_DISABLE_ENGINE"):
            self.engine.enabled = False

    # -- validation -------------------------------------------------------- #

    def validate(self) -> None:
        self.mode = _as_choice(self.mode, "mode", MODES)
        self.style = _as_choice(self.style, "style", STYLES)
        self.side = _as_choice(self.side, "side", SIDES)
        self.aggressiveness = _as_float(self.aggressiveness, "aggressiveness", minimum=0.0, maximum=1.0)
        self.main_line_moves = _as_int(self.main_line_moves, "main_line_moves", minimum=1, maximum=40)
        if self.max_plies is not None:
            self.max_plies = _as_int(self.max_plies, "max_plies", minimum=1, maximum=80)
        self.variations = _as_int(self.variations, "variations", minimum=0, maximum=12)
        self.variation_plies = _as_int(self.variation_plies, "variation_plies", minimum=1, maximum=30)
        if self.variation_max_ply is not None:
            self.variation_max_ply = _as_int(self.variation_max_ply, "variation_max_ply", minimum=1, maximum=80)
        self.prefer_theory = _as_bool(self.prefer_theory, "prefer_theory")
        self.theory_cp_tolerance = _as_int(self.theory_cp_tolerance, "theory_cp_tolerance", minimum=0, maximum=400)
        self.style_cp_tolerance = _as_int(self.style_cp_tolerance, "style_cp_tolerance", minimum=0, maximum=400)
        self.allow_out_of_book = _as_bool(self.allow_out_of_book, "allow_out_of_book")
        self.stop_when_out_of_theory = _as_bool(self.stop_when_out_of_theory, "stop_when_out_of_theory")
        self.deterministic = _as_bool(self.deterministic, "deterministic")
        self.seed = _as_int(self.seed, "seed", minimum=0)
        self.repertoire_branches = _as_int(self.repertoire_branches, "repertoire_branches", minimum=1, maximum=12)
        self.engine.validate()
        self.output.validate()

        if self.style == "engine_best":
            # Objective mode: theory must never override the engine's choice.
            self.prefer_theory = False
        if self.mode == "engine":
            self.prefer_theory = False
        if self.deterministic:
            # Time-based search is not reproducible.
            self.engine.time_ms = None
            self.engine.critical_time_ms = None
            self.engine.threads = 1
        if self.engine.required and not self.engine.enabled:
            raise ConfigError("engine.required is true but engine.enabled is false")

    # -- derived values ---------------------------------------------------- #

    @property
    def target_plies(self) -> int:
        plies = self.main_line_moves * 2
        if self.max_plies is not None:
            plies = min(plies, self.max_plies)
        return plies

    def to_dict(self) -> Dict[str, Any]:
        data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("engine", "output")
        }
        data["engine"] = self.engine.to_dict()
        data["output"] = self.output.to_dict()
        return data


#: Per-mode configuration defaults, layered under caller-supplied values.
MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "gm": {
        "style": "classical_gm",
        "prefer_theory": True,
        "main_line_moves": 12,
        "variations": 2,
        "output": {"annotations": "standard", "evals": "critical"},
    },
    "engine": {
        "style": "engine_best",
        "prefer_theory": False,
        "main_line_moves": 10,
        "variations": 2,
        "output": {"annotations": "minimal", "evals": "all", "arrows": False},
        "engine": {"depth": 18, "critical_depth": 22, "multipv": 4},
    },
    "training": {
        "style": "theoretical",
        "prefer_theory": True,
        "main_line_moves": 10,
        "variations": 3,
        "output": {"annotations": "rich", "evals": "critical", "arrows": True},
    },
    "trap": {
        "style": "sharp_tactical",
        "aggressiveness": 0.85,
        "main_line_moves": 9,
        "variations": 2,
        "output": {"annotations": "rich", "evals": "critical", "arrows": True},
        "engine": {"critical_depth": 22, "critical_multipv": 6},
    },
    "repertoire": {
        "style": "practical",
        "prefer_theory": True,
        "main_line_moves": 8,
        "variations": 0,
        "repertoire_branches": 3,
        "output": {"annotations": "standard", "evals": "critical"},
    },
}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_config(
    overrides: Optional[Dict[str, Any]] = None,
    *,
    mode: Optional[str] = None,
    apply_mode_preset: bool = True,
) -> Config:
    """Build a :class:`Config` from mode presets plus caller overrides.

    ``overrides`` always wins over the preset, so ``{"mode": "trap",
    "main_line_moves": 6}`` keeps the trap preset's annotations but shortens the
    line.
    """
    overrides = dict(overrides or {})
    resolved_mode = _as_choice(mode or overrides.get("mode") or "gm", "mode", MODES)
    overrides["mode"] = resolved_mode
    data: Dict[str, Any] = {}
    if apply_mode_preset:
        data = _deep_merge(data, MODE_PRESETS.get(resolved_mode, {}))
    data = _deep_merge(data, overrides)
    return Config.from_dict(data)


#: Keys accepted by :func:`build_config` at the top level (handy for CLI help).
TOP_LEVEL_KEYS = tuple(sorted(f.name for f in dataclasses.fields(Config)))
