#!/usr/bin/env bash
# Set up the PGN opening generator: check dependencies, find or fetch Stockfish,
# rebuild the opening index, and run the fast tests.
#
# Safe to re-run. Nothing outside the repository is modified except the optional
# Stockfish download into ~/.local/share/pgn-generator/engine.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
ENGINE_DIR="${HOME}/.local/share/pgn-generator/engine"
ENGINE_BIN="${ENGINE_DIR}/stockfish"
PYTHON="${PYTHON:-python3}"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found; install Python 3.9 or newer"

info "Python: $("$PYTHON" --version 2>&1)"
"$PYTHON" - <<'PY' || die "Python 3.9 or newer is required"
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY

if "$PYTHON" -c "import chess" >/dev/null 2>&1; then
    info "python-chess: $("$PYTHON" -c 'import chess; print(chess.__version__)')"
else
    warn "python-chess is not installed"
    printf 'Install it now with pip? [y/N] '
    read -r reply
    case "$reply" in
        [yY]*)
            "$PYTHON" -m pip install --user "chess>=1.10" \
                || die "pip install failed; install python-chess manually (pip install chess)"
            ;;
        *)
            die "python-chess is required: pip install chess"
            ;;
    esac
fi

# --------------------------------------------------------------------------- #
# Stockfish
# --------------------------------------------------------------------------- #

find_engine() {
    "$PYTHON" - <<'PY'
import sys, os
sys.path.insert(0, os.environ["ROOT"])
from pgn_generator.config import EngineConfig
print(EngineConfig().resolved_path() or "")
PY
}

export ROOT
FOUND="$(find_engine)"

if [ -n "$FOUND" ]; then
    info "Stockfish: $FOUND"
    "$FOUND" <<< "uci
quit" 2>/dev/null | grep -m1 '^id name' | sed 's/^id name /    /' || true
else
    warn "no Stockfish binary found on PATH"
    cat <<'EOF'

Without an engine the generator still produces legal, theory-based lines, but it
reports engine_validated: false and omits all evaluations and judgement symbols.

EOF
    printf 'Download Stockfish 18 into %s? [y/N] ' "$ENGINE_DIR"
    read -r reply
    case "$reply" in
        [yY]*)
            command -v curl >/dev/null 2>&1 || die "curl is required for the download"
            command -v tar  >/dev/null 2>&1 || die "tar is required for the download"

            # Pick the binary matching this CPU's instruction set.
            variant="$("$PYTHON" - <<'PY'
import re
try:
    flags = set(re.search(r"flags\s*:(.*)", open("/proc/cpuinfo").read()).group(1).split())
except Exception:
    flags = set()
if "avx512f" in flags and "avx512vl" in flags:
    print("avx512")
elif "bmi2" in flags:
    print("bmi2")
elif "avx2" in flags:
    print("avx2")
elif "sse4_1" in flags and "popcnt" in flags:
    print("sse41-popcnt")
else:
    print("")
PY
)"
            if [ -n "$variant" ]; then
                asset="stockfish-ubuntu-x86-64-${variant}"
            else
                asset="stockfish-ubuntu-x86-64"
            fi
            url="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/${asset}.tar"

            info "Downloading ${asset} (about 110 MiB)"
            tmp="$(mktemp -d)"
            trap 'rm -rf "$tmp"' EXIT
            curl -fL --progress-bar -o "$tmp/sf.tar" "$url" \
                || die "download failed; fetch Stockfish manually from https://stockfishchess.org/download/"
            tar -xf "$tmp/sf.tar" -C "$tmp"
            mkdir -p "$ENGINE_DIR"
            cp "$tmp/stockfish/${asset}" "$ENGINE_BIN"
            chmod +x "$ENGINE_BIN"
            info "Installed $ENGINE_BIN"
            FOUND="$ENGINE_BIN"
            cat <<EOF

Add this to your shell profile so the generator finds it automatically:

    export PGNGEN_ENGINE_PATH="$ENGINE_BIN"

EOF
            ;;
        *)
            info "Skipping the engine download"
            ;;
    esac
fi

# --------------------------------------------------------------------------- #
# Opening index
# --------------------------------------------------------------------------- #

info "Rebuilding the ECO index"
( cd "$ROOT" && "$PYTHON" scripts/build_eco_index.py )

# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

info "Generating a test line"
if [ -n "$FOUND" ]; then
    ( cd "$ROOT" && PGNGEN_ENGINE_PATH="$FOUND" "$PYTHON" -m pgn_generator \
        "main line of the Italian Game" --depth 10 --moves-count 5 --variations 0 --quiet )
else
    ( cd "$ROOT" && "$PYTHON" -m pgn_generator \
        "main line of the Italian Game" --no-engine --moves-count 5 --variations 0 --quiet )
fi

if "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
    info "Running the test suite"
    if [ -n "$FOUND" ]; then
        ( cd "$ROOT" && PGNGEN_ENGINE_PATH="$FOUND" "$PYTHON" -m pytest -q )
    else
        ( cd "$ROOT" && "$PYTHON" -m pytest -q )
    fi
else
    warn "pytest is not installed; skipping the test suite (pip install -r requirements-dev.txt)"
fi

info "Ready. Try: python3 -m pgn_generator \"sharp Najdorf variation\""
