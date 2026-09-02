"""pgn-generator: natural language -> legal, engine-validated, annotated opening PGN.

Public API::

    from pgn_generator import generate_pgn

    result = generate_pgn("main line of the Italian Game")
    print(result.pgn)

The guiding rule throughout: the language model (or the caller) supplies intent,
the chess engine validates the chess. Nothing is emitted that has not been
replayed for legality, re-parsed as PGN, and - when Stockfish is available -
checked by the engine. When the engine is missing, the output says so instead of
pretending otherwise.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .book import BookEntry, OpeningBook, get_book
from .config import Config, EngineConfig, OutputConfig, build_config
from .engine import EngineManager, Score
from .errors import (
    ConfigError,
    EngineFailureError,
    EngineUnavailableError,
    GenerationError,
    IllegalMoveError,
    InvalidFENError,
    OpeningNotFoundError,
    PGNGeneratorError,
    RequestError,
    ValidationFailedError,
)
from .generator import GenerationResult, OpeningGenerator, generate_pgn
from .request import Request, RequestParser, parse_request
from .validate import ValidationReport, validate

__all__ = [
    "__version__",
    # high-level API
    "generate_pgn",
    "GenerationResult",
    "OpeningGenerator",
    # configuration
    "Config",
    "EngineConfig",
    "OutputConfig",
    "build_config",
    # request parsing
    "Request",
    "RequestParser",
    "parse_request",
    # book
    "BookEntry",
    "OpeningBook",
    "get_book",
    # engine
    "EngineManager",
    "Score",
    # validation
    "ValidationReport",
    "validate",
    # errors
    "PGNGeneratorError",
    "ConfigError",
    "RequestError",
    "IllegalMoveError",
    "InvalidFENError",
    "EngineUnavailableError",
    "EngineFailureError",
    "OpeningNotFoundError",
    "ValidationFailedError",
    "GenerationError",
]
