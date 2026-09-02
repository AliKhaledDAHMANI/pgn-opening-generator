# Configuration reference

Every knob the generator exposes. CLI flags cover the common ones; anything else
is reachable through `--config file.json` or the `overrides` argument of
`generate_pgn()`.

Precedence, lowest to highest: library defaults -> mode preset -> `PGNGEN_*`
environment variables -> `--config` file -> inferences from the request text ->
explicit CLI flags.

## Top level

| Key | CLI | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `--mode` | `gm` | `gm`, `engine`, `training`, `trap`, `repertoire` |
| `style` | `--style` | per mode | `classical_gm`, `sharp_tactical`, `aggressive`, `solid`, `positional`, `gambit`, `practical`, `engine_best`, `theoretical` |
| `aggressiveness` | `--aggressiveness` | `0.5` | 0.0 safest, 1.0 most forcing. Scales the attack and solidity components, and widens the centipawn tolerance above 0.5. |
| `side` | `--side` | `both` | Whose plans drive the line: `white`, `black`, `both` (the side to move at the start) |
| `main_line_moves` | `--moves-count` | per mode (8-12) | Main line length in **full moves** |
| `max_plies` | `--max-plies` | none | Hard ceiling in half-moves, applied after `main_line_moves` |
| `variations` | `--variations` | per mode (0-3) | Sidelines attached to the main line |
| `variation_plies` | `--variation-plies` | `6` | Length of each sideline, in half-moves |
| `variation_max_ply` | - | none | Sidelines only before this ply, keeping output opening-shaped |
| `prefer_theory` | `--no-theory` | `true` (except engine modes) | Stay in the ECO book while a sound book move exists |
| `theory_cp_tolerance` | - | `60` | Max centipawn loss tolerated for a book move |
| `style_cp_tolerance` | - | `45` | Max centipawn loss tolerated for a style-driven move |
| `allow_out_of_book` | - | `true` | Permit leaving the book when the engine approves |
| `stop_when_out_of_theory` | `--stop-out-of-theory` | `false` | End the line once it leaves the book (never before ply 8) |
| `deterministic` | `--non-deterministic` | `true` | Fixed depth/nodes, one thread, no time-based search |
| `seed` | - | `0` | Seed for the (currently tie-breaking only) RNG |
| `repertoire_branches` | `--repertoire-branches` | `3` | Opponent replies answered in repertoire mode |

Note on `deterministic`: it forces `engine.threads = 1` and clears
`engine.time_ms`, because neither is reproducible. Setting `--threads` or
`--time` on the command line implies `--non-deterministic`.

## `engine`

| Key | CLI | Default | Meaning |
| --- | --- | --- | --- |
| `path` | `--engine` | auto-discovered | Stockfish binary |
| `enabled` | `--no-engine` | `true` | Run the engine at all |
| `required` | `--require-engine` | `false` | Fail rather than degrade when the engine is missing |
| `allow_fallback` | - | `true` | Permit legality-only output (with a warning) |
| `depth` | `--depth` | `16` | Search depth for ordinary positions |
| `nodes` | `--nodes` | none | Node limit; combined with depth if both are set |
| `time_ms` | `--time` | none | Per-position budget in ms (ignored while deterministic) |
| `multipv` | `--multipv` | `4` | Candidate moves per position |
| `critical_depth` | `--critical-depth` | `20` | Depth for critical positions |
| `critical_nodes` | - | none | Node limit for critical positions |
| `critical_time_ms` | - | none | Time budget for critical positions |
| `critical_multipv` | `--critical-multipv` | `6` | Candidates in critical positions |
| `threads` | `--threads` | `1` | Engine threads |
| `hash_mb` | `--hash` | `128` | Hash table size in MiB |
| `syzygy_path` | `--syzygy` | none | Syzygy tablebase directory |
| `uci_options` | - | `{}` | Raw UCI options; unsupported names are warned about and ignored |
| `startup_timeout_s` | - | `20.0` | Seconds allowed for the engine handshake |
| `analysis_timeout_s` | - | `120.0` | Seconds allowed per analysis call |

**Which positions count as critical:** the side to move is in check, three or
more captures are available, or the previous move was a sacrifice, a promotion, a
fork, a check, or built a mate threat or heavy king pressure.

**Engine discovery order:** `engine.path` -> `PGNGEN_ENGINE_PATH` ->
`STOCKFISH_PATH` -> `stockfish`, `stockfish17`, `stockfish16`, the
`stockfish-ubuntu-x86-64*` variants and `stockfish.exe` on `PATH` ->
`/usr/local/bin`, `/usr/bin`, `/usr/games`, `/opt/stockfish`, `~/.local/bin`,
`~/.local/share/pgn-generator/engine`.

## `output`

| Key | CLI | Default | Meaning |
| --- | --- | --- | --- |
| `comments` | `--no-comments` | `true` | Prose inside comment braces |
| `annotations` | `--annotations` | per mode | `none`, `minimal`, `standard`, `rich` |
| `use_nag_codes` | `--nag-codes` | `false` | Write `$1` instead of `!` |
| `evals` | `--evals` | `critical` | `none`, `critical`, `all` |
| `eval_format` | `--eval-format` | `pawns` | `pawns` (`+0.35`), `centipawns` (`+35 cp`), `verbose` (`Stockfish: +0.35`) |
| `eval_white_pov` | - | `true` | Scores from White's point of view |
| `arrows` | `--no-arrows` | `true` | `[%cal]`/`[%csl]` markup |
| `arrow_format` | - | `cal_csl` | Only format currently supported |
| `include_eco_headers` | - | `true` | Write `ECO`, `Opening`, `Variation`, `SubVariation` |
| `include_fen_headers` | - | `auto` | `auto`, `always`, `never`. A custom start position always gets `FEN`/`SetUp` regardless, since the movetext cannot be replayed without it. |
| `columns` | `--columns` | none | Wrap movetext at this column (`None` = one line) |
| `event` | `--event` | `Opening Analysis` | `Event` header |
| `site` | - | `?` | `Site` header |
| `date` | - | `????.??.??` | `Date` header |
| `round` | - | `-` | `Round` header |
| `white` | `--white` | `White` | `White` header |
| `black` | `--black` | `Black` | `Black` header |
| `result` | - | `*` | `Result` header |
| `extra_headers` | - | `{}` | Additional headers, written verbatim |

Comment density: `minimal` comments only important moves, `standard` leaves at
least 3 plies between ordinary comments, `rich` leaves 2 and adds strategic
notes. Judgement symbols, new opening names and tactical points always earn a
comment regardless of spacing.

## Mode presets

Presets sit under caller overrides, so `{"mode": "trap", "main_line_moves": 6}`
keeps the trap preset's rich annotations but shortens the line.

| Mode | style | main_line_moves | variations | annotations | evals | other |
| --- | --- | --- | --- | --- | --- | --- |
| `gm` | `classical_gm` | 12 | 2 | standard | critical | theory preferred |
| `engine` | `engine_best` | 10 | 2 | minimal | all | no theory, no arrows, depth 18/22 |
| `training` | `theoretical` | 10 | 3 | rich | critical | arrows on |
| `trap` | `sharp_tactical` | 9 | 2 | rich | critical | aggressiveness 0.85, critical depth 22 |
| `repertoire` | `practical` | 8 | 0 | standard | critical | 3 branches |

## Environment variables

| Variable | Effect |
| --- | --- |
| `PGNGEN_ENGINE_PATH` | Stockfish binary (also accepts `STOCKFISH_PATH`) |
| `PGNGEN_ENGINE_DEPTH` | Overrides `engine.depth`, raising `critical_depth` to match |
| `PGNGEN_ENGINE_THREADS` | Overrides `engine.threads` |
| `PGNGEN_ENGINE_HASH` | Overrides `engine.hash_mb` |
| `PGNGEN_DISABLE_ENGINE` | Truthy value disables the engine |

## Style weights

Each candidate move is scored on six components, normalised to roughly 0..1, then
combined with these weights. `aggressiveness` scales `attack` up and `solid` down.

| Style | engine | theory | attack | solid | tactics | practical | cp bonus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `classical_gm` | 1.00 | 0.85 | 0.15 | 0.35 | 0.15 | 0.45 | 0 |
| `sharp_tactical` | 0.80 | 0.45 | 0.75 | 0.00 | 0.95 | 0.20 | +25 |
| `aggressive` | 0.75 | 0.40 | 1.00 | 0.00 | 0.60 | 0.20 | +25 |
| `solid` | 0.95 | 0.80 | 0.00 | 0.95 | 0.00 | 0.55 | -10 |
| `positional` | 0.95 | 0.70 | 0.10 | 0.70 | 0.05 | 0.45 | 0 |
| `gambit` | 0.55 | 0.55 | 0.80 | 0.00 | 0.65 | 0.15 | +45 |
| `practical` | 0.85 | 0.75 | 0.20 | 0.40 | 0.15 | 1.00 | 0 |
| `engine_best` | 1.00 | 0 | 0 | 0 | 0 | 0 | forces top move |
| `theoretical` | 0.80 | 1.00 | 0.10 | 0.25 | 0.10 | 0.40 | +10 |

"cp bonus" adjusts `style_cp_tolerance`: a gambit line may give up more material
than a solid one, because that is the point of a gambit.

Only the hero plays in the requested style. The opponent uses `classical_gm`
whenever the hero's style is `gambit`, `aggressive` or `sharp_tactical`, so
requests for sharp play do not turn into both sides blundering.

## Annotation thresholds

| Symbol | Condition |
| --- | --- |
| `??` | centipawn loss >= 250 |
| `?` | >= 120 |
| `?!` | >= 55 |
| `!` | loss <= 10 **and** not forced **and** one of: confirmed sacrifice >= 90 cp; only move holding the evaluation (>= 150 cp better than the runner-up) and not found by a depth-6 search; non-obvious fork >= 60 cp better; best underpromotion; checkmate |
| `!!` | confirmed sacrifice >= 250 cp that is also the only move keeping the advantage |

A "confirmed" sacrifice is one where the material is still missing after
following the engine's principal variation for up to 8 plies. Unconfirmed
candidates lose the label and the associated comment.

## Data

`pgn_generator/data/eco/{a,b,c,d,e}.tsv` is the Lichess `chess-openings` data set (CC0): 3810
named lines, 149 families. `pgn_generator/data/eco/index.json.gz` is the compiled graph (7855
positions with theory-breadth counts) that the loader actually reads; rebuild it
with `python3 scripts/build_eco_index.py` after editing the TSV files.

`TheoryMove.breadth` counts the named book lines reachable through a move. It
measures how much catalogued theory sits behind a move, not how often the move is
played - the data set carries no game counts, so none are claimed.
