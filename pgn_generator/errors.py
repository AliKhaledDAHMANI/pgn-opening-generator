"""Error types for the PGN opening generator.

Every failure mode that an agent may need to distinguish gets its own class so
that the CLI can map it onto a stable ``error.code`` in the JSON envelope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PGNGeneratorError(Exception):
    """Base class for all generator errors."""

    code = "generator_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = {k: v for k, v in details.items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(PGNGeneratorError):
    """Invalid configuration supplied by the caller."""

    code = "invalid_config"


class RequestError(PGNGeneratorError):
    """The natural-language request or structured request is unusable."""

    code = "invalid_request"


class IllegalMoveError(PGNGeneratorError):
    """A caller-supplied move is not legal in the position it was played from."""

    code = "illegal_move"

    def __init__(
        self,
        message: str,
        *,
        move: Optional[str] = None,
        fen: Optional[str] = None,
        ply: Optional[int] = None,
        legal_sample: Optional[List[str]] = None,
        accepted_prefix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            move=move,
            fen=fen,
            ply=ply,
            legal_sample=legal_sample,
            accepted_prefix=accepted_prefix,
        )


class InvalidFENError(PGNGeneratorError):
    """A caller-supplied FEN is malformed or describes an unusable position."""

    code = "invalid_fen"


class EngineUnavailableError(PGNGeneratorError):
    """Stockfish could not be started, and no fallback was permitted."""

    code = "engine_unavailable"


class EngineFailureError(PGNGeneratorError):
    """Stockfish was started but died or misbehaved during analysis."""

    code = "engine_failure"


class OpeningNotFoundError(PGNGeneratorError):
    """The requested opening could not be resolved against the ECO book."""

    code = "opening_not_found"

    def __init__(self, message: str, *, query: Optional[str] = None, suggestions: Optional[List[str]] = None) -> None:
        super().__init__(message, query=query, suggestions=suggestions)


class ValidationFailedError(PGNGeneratorError):
    """The generated PGN failed the mandatory validation pipeline."""

    code = "validation_failed"

    def __init__(self, message: str, *, failures: Optional[List[Dict[str, Any]]] = None, pgn: Optional[str] = None) -> None:
        super().__init__(message, failures=failures)
        self.pgn = pgn


class GenerationError(PGNGeneratorError):
    """No line could be produced that satisfies the request."""

    code = "generation_failed"
