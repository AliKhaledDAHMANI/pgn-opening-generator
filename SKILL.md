---
name: pgn-opening-generator
description: Use when a user asks for a chess opening line, variation, repertoire, trap, or gambit as PGN - e.g. "show me the main line of the Italian Game", "sharp Najdorf variation", "trap against the King's Indian", "best line after 1.e4 c5 2.Nf3 d6 3.d4", "build a White repertoire against 1...e5". Converts natural-language opening requests into legal, Stockfish-validated, annotated PGN with ECO headers, engine evaluations, nested variations and board arrows. Do not use for full-game analysis, endgames, or annotating an existing complete game.
license: MIT
---

# PGN Opening Generator

Turns a natural-language description of an opening idea into a legal,
engine-validated, annotated PGN variation.

**Division of labour: you supply the intent, the engine validates the chess.**
Never hand-write PGN moves for the user and never state an evaluation from
memory. Call the tool; it replays every move for legality, re-parses its own
output, and asks Stockfish for every number it prints.

## When to use

Use for: opening main lines and sidelines, gambits, traps, repertoires,
transpositions, theoretical novelties, "best line from this position", opening
plans and structures.

Do not use for: middlegame or endgame analysis, annotating a complete game,
tactics puzzles, or engine-vs-engine games. The generator stops when the opening
has been demonstrated.

## Invocation

```bash
python3 -m pgn_generator "REQUEST" [flags]
```

Run from the skill directory, or add it to `PYTHONPATH`. Requires `python-chess`;
Stockfish 16+ is strongly recommended (see *Engine* below).

Pass the user's words through largely unchanged - the parser reads style, mode,
side, length, opening name and starting moves out of the sentence. Add flags only
for things the user stated that the sentence cannot carry (an explicit FEN, an
exact depth) or that you need for your own workflow (`--format json`).

```bash
# Typical calls
python3 -m pgn_generator "Show me the main line of the Italian Game."
python3 -m pgn_generator "Create a sharp Sicilian Najdorf variation." --moves-count 12
python3 -m pgn_generator "Give me a GM-style trap against the King's Indian." --format json
python3 -m pgn_generator "best line here" --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
python3 -m pgn_generator "Build an opening repertoire for White against 1...e5."
```

Python API, if you are already in a Python context:

```python
from pgn_generator import generate_pgn

result = generate_pgn("main line of the Italian Game",
                      overrides={"main_line_moves": 10, "variations": 2})
print(result.pgn)                        # validated PGN text
result.report.engine_validated           # True only if Stockfish really ran
```

## Output

`--format pgn` (default) prints PGN to stdout. Use it when the user just wants
the line.

`--format json` prints one envelope. Use it when you need to check provenance
before quoting anything:

```json
{
  "ok": true,
  "pgn": "[Event \"Opening Analysis\"]\n...",
  "opening": {"eco": "C54", "name": "Italian Game: Classical Variation", ...},
  "mode": "gm", "style": "classical_gm",
  "main_line": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 ...",
  "plies": 18, "variations": 2,
  "stop_reason": "reached the requested length",
  "engine": {"available": true, "name": "Stockfish 18", "depth": 16, ...},
  "validation": {"ok": true, "engine_validated": true, "checks_run": [...],
                 "errors": [], "warnings": []},
  "request": {"inferences": ["style=theoretical (matched wording...)", ...]},
  "warnings": []
}
```

**Read `validation.engine_validated` before you describe the output.** If it is
`false`, say the line is legality- and theory-checked but not engine-verified,
and do not quote evaluations (there will be none). Errors arrive as
`{"ok": false, "error": {"code": ..., "message": ..., "details": {...}}}` with
exit code 1.

## Modes

| Mode | `--mode` | Use when the user wants |
| --- | --- | --- |
| GM (default) | `gm` | What a strong player would actually play: theory first, sound and practical |
| Engine | `engine` | Objective best play; theory preference off, Stockfish's top move every time |
| Training | `training` | To learn: rich comments, arrows, alternatives, plans explained |
| Trap | `trap` | A trap or a punishment for a natural-looking mistake |
| Repertoire | `repertoire` | One consistent system for one colour, branching on opponent replies |

The parser infers the mode from wording ("trap", "teach me", "objectively best",
"repertoire"). Override with `--mode` only when the user was explicit.

## Styles

`--style`: `classical_gm`, `sharp_tactical`, `aggressive`, `solid`,
`positional`, `gambit`, `practical`, `engine_best`, `theoretical`.

Inferred from words like "sharp", "solid", "gambit", "main line". Only the side
the user cares about plays in the requested style; the opponent plays soundly, so
an "aggressive" request does not turn into both sides blundering. `--aggressiveness
0.0-1.0` fine-tunes the bias towards forcing moves.

## Inputs the parser understands

* **Opening name** - resolved against the ECO book, including shorthand
  ("Spanish", "KID", "Najdorf", "Traxler"). Names are never invented: an
  unrecognisable name produces a warning, not a guess.
* **Starting moves** - `"best line after 1.e4 c5 2.Nf3 d6 3.d4"`, or `--moves
  "1.e4 c5 2.Nf3"`. The output begins with exactly those moves.
* **FEN** - in the sentence or via `--fen`. The line starts from exactly that
  position, and the PGN carries `FEN`/`SetUp` headers.
* **Reply constraints** - `"against 1...e5"` forces Black's first move.
* **Length** - "15-move variation", "short", "deep", or `--moves-count N`.
* **Side** - "for White", "as Black", or `--side`.
* **Presentation** - "no comments", "with arrows", "show evaluations".
* **Focus** - "attacks the kingside", "queenside play".
* **"rare" / "offbeat"** - relaxes theory preference and widens the engine's
  candidate window, so sound-but-unusual moves can win.

Vague requests get sensible defaults rather than a question. Every inference is
listed in `request.inferences`, so you can tell the user what you assumed.

## Engine

Stockfish is found automatically on `PATH`, or set `PGNGEN_ENGINE_PATH`
(alternatively `--engine /path/to/stockfish`). Tunable: `--depth`,
`--critical-depth`, `--nodes`, `--multipv`, `--critical-multipv`, `--threads`,
`--hash`, `--time`, `--syzygy`.

Positions the generator judges *critical* (checks, heavy capture tension, a piece
just invested, mate threats) are searched at the deeper `--critical-depth`
setting. Raise both depths for analysis the user will rely on; `--depth 10` keeps
things quick for exploration.

Output is deterministic by default (fixed depth/nodes, one thread, no
time-based search), so the same request gives byte-identical PGN. `--threads` or
`--time` implies `--non-deterministic`.

**If Stockfish is missing**, the generator emits a legality- and theory-checked
line, sets `engine_validated: false`, adds an `ENGINE VALIDATION UNAVAILABLE`
warning, and omits all evaluations and all judgement symbols. Never present such
output as engine-verified. Use `--require-engine` to fail instead of degrading,
or `--no-engine` to skip the engine deliberately.

## Annotation syntax

Judgement symbols are attached only when the engine's numbers justify them:

| Symbol | Emitted when |
| --- | --- |
| `??` | loses >= 250 cp against the best move |
| `?` | loses >= 120 cp |
| `?!` | loses >= 55 cp |
| `!` | (near-)best *and* hard to find: a confirmed sacrifice, the only move that holds, a non-obvious double attack, a best underpromotion, or mate |
| `!!` | a confirmed sacrifice of >= 250 cp that is also the only move keeping the advantage |

"Best move" alone never earns a `!`. A claimed sacrifice is confirmed by
following the engine's principal variation and checking the material is still
missing; unconfirmed ones are dropped silently. Every symbol carries a recorded
reason, and validation rejects any symbol that does not.

`--annotations none|minimal|standard|rich` controls comment density.
`--nag-codes` writes strict `$1`-style NAGs instead of inline `!`/`?`. Both
forms re-parse identically.

## Evaluations

`--evals critical` (default) annotates critical and judged moves; `all`
annotates every move; `none` omits them. `--eval-format pawns|centipawns|verbose`
renders `= +0.35`, `+35 cp`, or `Stockfish: +0.35`. Always from White's point of
view. Mate reads `#3` / `#-2`. Numbers only ever come from the engine.

## Board arrows

Machine-readable markup inside comments, understood by Lichess, ChessBase and
SCID:

```
{[%cal Gd1h5,Gc4f7][%csl Re5]}
```

`[%cal <colour><from><to>,...]` for arrows, `[%csl <colour><square>,...]` for
highlights. Green marks the idea being executed, red marks targets and
weaknesses, yellow marks squares of structural interest. Capped at three of each
per move. Disable with `--no-arrows`.

## Variations

Sidelines are attached where they teach the most, each with a stated purpose
(theory alternative, engine alternative, refutation, trap). Every branch starts
from the position its parent implies - validation checks this explicitly.
`--variations N` sets how many, `--variation-plies N` how long.

```
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 ({Two Knights Defense} 3... Nf6 4. O-O) 4. c3
```

## Validation

Eight checks run before anything is returned; a failure means no output, not a
warning:

1. **Legality** - every move replayed from its own start position.
2. **SAN** - every stored SAN matches what the position produces.
3. **PGN** - the text re-parses with no parser errors and the same moves.
4. **Engine** - critical positions were analysed (a warning, and
   `engine_validated: false`, if the engine was unavailable).
5. **Annotations** - every judgement symbol has an engine-backed reason.
6. **Variations** - every branch starts from the correct position.
7. **FEN** - a requested FEN or move prefix is honoured exactly.
8. **Opening** - the line corresponds to the requested opening (a warning if it
   drifted).

Failures 1, 2, 3, 5, 6 and 7 are hard errors and produce
`{"ok": false, "error": {"code": "validation_failed", ...}}`.

## Error handling

| Code | Meaning | What to do |
| --- | --- | --- |
| `illegal_move` | A supplied move is illegal | `details` gives the FEN, ply, legal moves and the accepted prefix. Tell the user which move failed and why; retry with `--lenient-moves` to continue from the last legal position. |
| `invalid_fen` | Malformed or finished position | Report the problem; ask for a corrected FEN. |
| `opening_not_found` | Name unresolvable | `details.suggestions` lists near matches; offer them. |
| `engine_unavailable` | No Stockfish and `--require-engine` | Install Stockfish or drop `--require-engine` and say the line is unverified. |
| `invalid_config` | Bad flag or config value | Fix the flag; the message names the field. |
| `validation_failed` | Internal check failed | Do not show the PGN. Report it as a bug with `error.details`. |
| `generation_failed` | No line satisfies the request | Loosen the constraints (shorter line, different style). |

Non-fatal problems arrive in `warnings` and the line is still valid. Always
surface them: "no trap was found, so this is a sharp tactical line instead" is
information the user needs.

## Example output

```
[Event "Opening Analysis"]
[Site "?"]
[Date "????.??.??"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "*"]
[ECO "C57"]
[Opening "Italian Game"]
[Variation "Two Knights Defense"]
[Annotator "pgn-generator (trap mode, Stockfish 18)"]

{Italian Game: Two Knights Defense, trap mode.} 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6
4. d4 {Sets the trap: Nxe4 is a capture that looks like free material here, but
dxe5 refutes it.} 4... exd4 ({The trap: this natural reply loses material}
4... Nxe4? {[%csl Re1] +/= +1.22 (loses 134 cp compared with the best move)}
5. dxe5 {+/= +1.26 The refutation: 138 centipawns swing according to Stockfish.})
5. e5 *
```

## Reference

`--help` lists every flag. `reference/PARAMETERS.md` documents the full
configuration surface, including values reachable only through `--config
file.json`.
