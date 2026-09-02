"""Command-line interface.

Two output shapes:

* ``--format pgn`` (default) writes PGN to stdout - convenient for humans and for
  piping into a viewer.
* ``--format json`` writes a single JSON envelope with the PGN plus provenance,
  validation results and warnings - the shape an agent should consume.

The exit code is ``0`` on success, ``1`` on a handled error (invalid request,
illegal move, validation failure) and ``2`` on a usage error. Handled errors are
still reported as JSON when ``--format json`` is set, so callers never have to
parse a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence

from .book import get_book
from .config import (
    ANNOTATION_LEVELS,
    EVAL_FORMATS,
    EVAL_MODES,
    MODES,
    SIDES,
    STYLES,
    build_config,
)
from .errors import PGNGeneratorError, ValidationFailedError
from .generator import OpeningGenerator
from .request import RequestParser

PROGRAM = "pgn-generator"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Generate legal, engine-validated, annotated PGN opening variations from a "
            "natural-language description."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  {PROGRAM} \"main line of the Italian Game\"\n"
            f"  {PROGRAM} \"sharp Najdorf\" --mode gm --moves-count 12 --variations 2\n"
            f"  {PROGRAM} \"best line here\" --moves \"1.e4 c5 2.Nf3 d6 3.d4\"\n"
            f"  {PROGRAM} \"trap for White\" --mode trap --format json\n"
        ),
    )
    parser.add_argument(
        "request",
        nargs="?",
        default="",
        help="natural-language description of the opening line to generate",
    )

    position = parser.add_argument_group("starting position")
    position.add_argument("--fen", help="start from this FEN instead of the initial position")
    position.add_argument(
        "--moves",
        help='moves the line must begin with, e.g. "1.e4 c5 2.Nf3" or "e4 c5 Nf3"',
    )
    position.add_argument("--opening", help="opening name to resolve against the ECO book")
    position.add_argument(
        "--lenient-moves",
        action="store_true",
        help="ignore an illegal supplied move and continue from the last legal position",
    )

    shape = parser.add_argument_group("line shape")
    shape.add_argument("--mode", choices=MODES, help="quality mode (default: inferred, else gm)")
    shape.add_argument("--style", choices=STYLES, help="move-selection style")
    shape.add_argument("--side", choices=SIDES, help="side whose plans drive the line")
    shape.add_argument(
        "--aggressiveness", type=float, metavar="0.0-1.0", help="bias towards forcing play"
    )
    shape.add_argument(
        "--moves-count", type=int, metavar="N", dest="main_line_moves",
        help="length of the main line in full moves",
    )
    shape.add_argument("--max-plies", type=int, metavar="N", help="hard ceiling on half-moves")
    shape.add_argument("--variations", type=int, metavar="N", help="number of sidelines to attach")
    shape.add_argument(
        "--variation-plies", type=int, metavar="N", help="length of each sideline in half-moves"
    )
    shape.add_argument(
        "--repertoire-branches", type=int, metavar="N",
        help="number of opponent replies to answer in repertoire mode",
    )
    shape.add_argument(
        "--no-theory", dest="prefer_theory", action="store_false", default=None,
        help="do not prefer book moves over engine choices",
    )
    shape.add_argument(
        "--stop-out-of-theory", dest="stop_when_out_of_theory", action="store_true", default=None,
        help="end the line as soon as it leaves the ECO book",
    )

    presentation = parser.add_argument_group("presentation")
    presentation.add_argument(
        "--annotations", choices=ANNOTATION_LEVELS, help="annotation richness (default per mode)"
    )
    presentation.add_argument(
        "--no-comments", dest="comments", action="store_false", default=None,
        help="omit PGN comments entirely",
    )
    presentation.add_argument("--evals", choices=EVAL_MODES, help="when to include engine evaluations")
    presentation.add_argument("--eval-format", choices=EVAL_FORMATS, help="how to print evaluations")
    presentation.add_argument(
        "--no-arrows", dest="arrows", action="store_false", default=None,
        help="omit [%%cal]/[%%csl] board markup",
    )
    presentation.add_argument(
        "--nag-codes", dest="use_nag_codes", action="store_true", default=None,
        help="write $1-style NAGs instead of inline !/? suffixes",
    )
    presentation.add_argument("--event", help='Event header (default "Opening Analysis")')
    presentation.add_argument("--white", help="White header")
    presentation.add_argument("--black", help="Black header")
    presentation.add_argument("--columns", type=int, help="wrap move text at this column")

    engine = parser.add_argument_group("engine")
    engine.add_argument("--engine", dest="engine_path", help="path to the Stockfish binary")
    engine.add_argument("--depth", type=int, help="search depth for ordinary positions")
    engine.add_argument("--critical-depth", type=int, help="search depth for critical positions")
    engine.add_argument("--nodes", type=int, help="node limit instead of depth")
    engine.add_argument("--multipv", type=int, help="candidate moves per position")
    engine.add_argument("--critical-multipv", type=int, help="candidates in critical positions")
    engine.add_argument("--threads", type=int, help="engine threads (forces --non-deterministic)")
    engine.add_argument("--hash", type=int, dest="hash_mb", help="engine hash table size in MiB")
    engine.add_argument(
        "--time", type=int, dest="time_ms", metavar="MS",
        help="per-position time budget (forces --non-deterministic)",
    )
    engine.add_argument("--syzygy", dest="syzygy_path", help="Syzygy tablebase directory")
    engine.add_argument(
        "--no-engine", dest="engine_enabled", action="store_false", default=None,
        help="disable engine analysis (output is legality-checked only, and says so)",
    )
    engine.add_argument(
        "--require-engine", action="store_true",
        help="fail instead of degrading when Stockfish is unavailable",
    )
    engine.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false", default=None,
        help="allow time-based search and multiple threads",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--format", choices=("pgn", "json"), default="pgn", help="output format (default: pgn)"
    )
    output.add_argument("--out", metavar="FILE", help="write the PGN to a file as well as stdout")
    output.add_argument("--trace", action="store_true", help="include the per-move decision trace (JSON)")
    output.add_argument("--eco-data", metavar="DIR", help="alternative ECO data directory")
    output.add_argument("--config", metavar="FILE", help="JSON file of configuration overrides")
    output.add_argument("--quiet", action="store_true", help="suppress warnings on stderr")
    output.add_argument("--verbose", "-v", action="count", default=0, help="increase log verbosity")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI flags into a config-override dictionary."""
    overrides: Dict[str, Any] = {}
    engine: Dict[str, Any] = {}
    output: Dict[str, Any] = {}

    if args.config:
        with open(os.path.expanduser(args.config), "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise SystemExit(f"{PROGRAM}: --config must contain a JSON object")
        overrides.update(loaded)
        engine.update(overrides.pop("engine", {}) or {})
        output.update(overrides.pop("output", {}) or {})

    for name in (
        "mode", "style", "side", "aggressiveness", "main_line_moves", "max_plies",
        "variations", "variation_plies", "repertoire_branches", "prefer_theory",
        "stop_when_out_of_theory", "deterministic",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    if args.engine_path:
        engine["path"] = args.engine_path
    for flag, key in (
        ("depth", "depth"), ("critical_depth", "critical_depth"), ("nodes", "nodes"),
        ("multipv", "multipv"), ("critical_multipv", "critical_multipv"),
        ("threads", "threads"), ("hash_mb", "hash_mb"), ("time_ms", "time_ms"),
        ("syzygy_path", "syzygy_path"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            engine[key] = value
    if args.engine_enabled is False:
        engine["enabled"] = False
    if args.require_engine:
        engine["required"] = True
        engine["allow_fallback"] = False
    # Threads and time budgets are incompatible with reproducible output.
    if (args.threads is not None and args.threads > 1) or args.time_ms is not None:
        overrides.setdefault("deterministic", False)

    for flag, key in (
        ("annotations", "annotations"), ("comments", "comments"), ("evals", "evals"),
        ("eval_format", "eval_format"), ("arrows", "arrows"), ("use_nag_codes", "use_nag_codes"),
        ("event", "event"), ("white", "white"), ("black", "black"), ("columns", "columns"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            output[key] = value

    if engine:
        overrides["engine"] = engine
    if output:
        overrides["output"] = output
    return overrides


def _explicit_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    explicit: Dict[str, Any] = {}
    if args.fen:
        explicit["fen"] = args.fen
    if args.moves:
        explicit["moves"] = args.moves
    if args.opening:
        explicit["opening"] = args.opening
    if args.mode:
        explicit["mode"] = args.mode
    if args.style:
        explicit["style"] = args.style
    if args.side:
        explicit["side"] = args.side
    return explicit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.quiet:
        logging.disable(logging.WARNING)

    if not args.request and not (args.fen or args.moves or args.opening):
        parser.error("give a natural-language request, or --opening/--moves/--fen")

    try:
        overrides = _overrides_from_args(args)
        explicit = _explicit_from_args(args)
        book = get_book(args.eco_data)
        request = RequestParser(book).parse(
            args.request, explicit=explicit, strict_moves=not args.lenient_moves
        )

        merged: Dict[str, Any] = dict(request.overrides)
        merged = _deep_merge(merged, overrides)
        mode = overrides.get("mode") or request.mode or "gm"
        if request.style and "style" not in overrides:
            merged.setdefault("style", request.style)
        if request.side and "side" not in overrides:
            merged.setdefault("side", request.side)

        config = build_config(merged, mode=mode)
        with OpeningGenerator(config, book=book) as generator:
            result = generator.generate(request)
    except ValidationFailedError as exc:
        return _fail(args, exc, extra={"pgn": exc.pgn} if exc.pgn else None)
    except PGNGeneratorError as exc:
        return _fail(args, exc)
    except FileNotFoundError as exc:
        return _fail(args, PGNGeneratorError(f"file not found: {exc}"))
    except json.JSONDecodeError as exc:
        return _fail(args, PGNGeneratorError(f"invalid JSON in --config: {exc}"))
    except KeyboardInterrupt:  # pragma: no cover
        sys.stderr.write("interrupted\n")
        return 130

    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as handle:
            handle.write(result.pgn)

    if args.format == "json":
        payload = {"ok": True, **result.to_dict(include_trace=args.trace)}
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result.pgn)
        if result.warnings and not args.quiet:
            for warning in result.warnings:
                sys.stderr.write(f"warning: {warning}\n")
    return 0


def _fail(args: argparse.Namespace, exc: PGNGeneratorError, *, extra: Optional[Dict[str, Any]] = None) -> int:
    if args.format == "json":
        payload: Dict[str, Any] = {"ok": False, "error": exc.to_dict()}
        if extra:
            payload.update(extra)
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stderr.write(f"{PROGRAM}: {exc.code}: {exc.message}\n")
        for key, value in (exc.details or {}).items():
            sys.stderr.write(f"  {key}: {value}\n")
    return 1


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
