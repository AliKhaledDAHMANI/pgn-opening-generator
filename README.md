# PGN Opening Generator

Turns a natural-language description of a chess opening idea into a legal,
Stockfish-validated, annotated PGN variation.

```console
$ python3 -m pgn_generator "Show me a trap in the Two Knights Defense." --mode trap
```

```
[Event "Opening Analysis"]
[ECO "C59"]
[Opening "Italian Game"]
[Variation "Two Knights Defense"]
[Annotator "pgn-generator (trap mode, Stockfish 18)"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 {Sets the trap: Nxe4 is a known book reply
here, but Bxf7+ refutes it.} 4... d5 ({The trap: this natural reply loses material}
4... Nxe4?! {+/- +1.64 ... (sharp, but objectively 119 cp worse than the best move)}
5. Bxf7+ {+/- +1.99 The refutation: 154 centipawns swing according to Stockfish.})
5. exd5 ... *
```

Built as an [agent skill](SKILL.md): an AI agent reads `SKILL.md`, calls the CLI,
and gets back PGN it can trust plus a machine-readable report saying exactly what
was verified.

## Why

Language models are fluent about chess and unreliable at it. They hallucinate
illegal moves, invent ECO codes, and quote evaluations they never computed. This
tool splits the work: **the model supplies the intent, the engine validates the
chess.**

Concretely, that means:

- Every move is replayed for legality from its own start position.
- Every generated PGN is re-parsed and compared against the source line.
- Every evaluation comes from Stockfish. There is no code path that prints a
  number the engine did not produce.
- Every `!` and `?` carries a recorded, engine-backed justification, and
  validation rejects any symbol that does not.
- Opening names and ECO codes come from a shipped data set, never from a guess.
- If Stockfish is unavailable, the output says so, drops all evaluations and all
  judgement symbols, and sets `engine_validated: false`.

## Install

Requires Python 3.9+ and [python-chess](https://python-chess.readthedocs.io/).
Stockfish 16 or newer is strongly recommended.

```console
$ git clone https://github.com/AliKhaledDAHMANI/pgn-opening-generator.git
$ cd pgn-opening-generator
$ pip install -r requirements.txt
$ ./scripts/install.sh          # checks the setup, offers to fetch Stockfish
```

`scripts/install.sh` verifies python-chess, locates Stockfish (or downloads it
into `~/.local/share/pgn-generator/engine`), rebuilds the opening index and runs
the fast test suite.

Point the tool at a specific engine binary with `--engine /path/to/stockfish` or
`export PGNGEN_ENGINE_PATH=/path/to/stockfish`.

### As an agent skill

Copy or symlink the directory into your agent's skills path:

```console
$ ln -s "$PWD" ~/.config/opencode/skills/pgn-opening-generator
```

## Use

```console
# Opening theory
$ python3 -m pgn_generator "Show me the main line of the Italian Game."
$ python3 -m pgn_generator "Create a sharp Sicilian Najdorf variation." --moves-count 12

# From a position
$ python3 -m pgn_generator "best line after 1.e4 c5 2.Nf3 d6 3.d4"
$ python3 -m pgn_generator "best response here" --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"

# Traps and repertoires
$ python3 -m pgn_generator "trap against the King's Indian" --mode trap
$ python3 -m pgn_generator "opening repertoire for White against 1...e5"

# Machine-readable output for an agent
$ python3 -m pgn_generator "Italian Game" --format json --trace
```

From Python:

```python
from pgn_generator import generate_pgn

result = generate_pgn("main line of the Italian Game",
                      overrides={"main_line_moves": 10, "variations": 2})

print(result.pgn)
assert result.report.ok                    # validation passed
assert result.report.engine_validated      # Stockfish really ran
print(result.opening.eco, result.opening.name)
```

`python3 -m pgn_generator --help` lists every flag;
[`reference/PARAMETERS.md`](reference/PARAMETERS.md) documents the full
configuration surface.

## Modes

| Mode | For |
| --- | --- |
| `gm` (default) | What a strong player would actually play: theory first, sound and practical |
| `engine` | Objective best play; Stockfish's top move every time |
| `training` | Learning: rich comments, arrows, alternatives, plans explained |
| `trap` | Traps, with an engine-verified refutation of a natural-looking mistake |
| `repertoire` | One consistent system for one colour, branching on opponent replies |

Styles (`classical_gm`, `sharp_tactical`, `aggressive`, `solid`, `positional`,
`gambit`, `practical`, `engine_best`, `theoretical`) shape move choice within a
mode. Only the side the user asked about plays in the requested style; the
opponent plays soundly, so "aggressive" does not become both sides blundering.

Mode and style are inferred from the request text ("sharp", "teach me", "trap",
"objectively best") and can be overridden with flags.

## How it works

```
request text
    |
    v
RequestParser ......... opening name, starting moves/FEN, style, mode, length
    |
    v
MoveSelector .......... ECO book theory  x  Stockfish MultiPV  x  style weights
    |                   (rejects anything losing more than the cp tolerance)
    v
Annotator ............. NAGs, comments, evaluations, arrows - all justified
    |
    v
PGN writer ............ nested variations, headers, inline suffixes
    |
    v
Validator ............. 8 checks; nothing is returned unless they pass
```

Move choice blends six scored components - engine quality, theory breadth, king
attack, solidity, tactics and practicality - weighted per style. A move that
loses more than the configured centipawn tolerance is rejected outright, so no
style can talk the generator into an unsound line.

Traps are found mechanically, not from a hard-coded list: a sound bait, a reply
that is near-best in a deliberately *shallow* search (what a human plays at a
glance) or wins material by static exchange, and a punishment the engine confirms
swings at least 120 centipawns at full depth.

Output is deterministic by default: fixed depth and nodes, one thread, no
time-based search, so the same request on the same machine gives byte-identical
PGN.

## Validation

Eight checks run before anything is returned. A failure means no output.

1. **Legality** - every move of every line replayed from its own start position.
2. **SAN** - each stored SAN matches what the position produces.
3. **PGN** - the text re-parses with no parser errors and the same moves.
4. **Engine** - critical positions were analysed at the deeper setting.
5. **Annotations** - every judgement symbol has an engine-backed reason.
6. **Variations** - every branch starts from the position its parent implies.
7. **FEN** - a requested FEN or move prefix is honoured exactly.
8. **Opening** - the line corresponds to the requested opening.

## Annotation syntax

| Symbol | Emitted when |
| --- | --- |
| `??` | loses >= 250 centipawns against the best move |
| `?` | loses >= 120 |
| `?!` | loses >= 55 |
| `!` | (near-)best *and* hard to find: a confirmed sacrifice, the only move that holds, a non-obvious double attack, a best underpromotion, or mate |
| `!!` | a confirmed sacrifice of >= 250 cp that is also the only move keeping the advantage |

"Best move" alone never earns a `!`. A claimed sacrifice is confirmed by
following the engine's principal variation and checking the material is still
gone; unconfirmed ones are dropped.

Board markup uses the Lichess/ChessBase convention, understood by every major
viewer:

```
{[%cal Gd1h5,Gc4f7][%csl Re5]}
```

Green shows the idea being executed, red marks targets and weaknesses, yellow
marks squares of structural interest.

## Examples

[`examples/`](examples/) holds generated output for each mode, produced at depth
12/15 by the commands in the file headers.

## Data

`pgn_generator/data/eco/` contains the [Lichess chess-openings](https://github.com/lichess-org/chess-openings)
data set (CC0 public domain): 3810 named lines across 149 families.
`index.json.gz` is the compiled position graph the loader reads; rebuild it with
`python3 scripts/build_eco_index.py`.

The `breadth` figure the selector uses counts *catalogued book lines* behind a
move, not games played - the data set has no game counts, so none are claimed.

## Development

```console
$ pip install -r requirements-dev.txt
$ pytest -q                       # engine tests skip automatically without Stockfish
$ PGNGEN_ENGINE_PATH=/path/to/stockfish pytest -q
```

261 tests cover the book, features, selection, annotation discipline, PGN
round-tripping, validation, the CLI, end-to-end generation for every mode, and
regressions for every bug found during development. Tests that need an engine are
skipped rather than failing when none is present - which is also the degradation
path the library itself supports.

## Licence

MIT (see [LICENSE](LICENSE)). The bundled ECO data is CC0.
