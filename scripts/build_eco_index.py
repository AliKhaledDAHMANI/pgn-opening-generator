#!/usr/bin/env python3
"""Compile ``pgn_generator/data/eco/*.tsv`` into ``index.json.gz``.

The index stores the full opening graph (positions, edges, theory breadth), which
removes all SAN parsing and move generation from start-up: ~3.5s -> ~0.15s on a
slow CPU. Run this after editing the TSV files.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pgn_generator.book import ECO_DIR, INDEX_PATH, OpeningBook  # noqa: E402


def main() -> int:
    book = OpeningBook._from_tsv(ECO_DIR)  # noqa: SLF001 - intentional: bypass the index
    payload = book.to_index_payload()
    with gzip.open(INDEX_PATH, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
    stats = book.stats()
    size_kb = os.path.getsize(INDEX_PATH) / 1024
    print(
        f"wrote {INDEX_PATH} ({size_kb:.0f} KiB): "
        f"{stats['entries']} entries, {stats['positions']} positions, {stats['families']} families"
    )

    # Verify the index reloads and agrees with the TSV source.
    reloaded = OpeningBook.load(ECO_DIR)
    assert reloaded.stats() == stats, "index reload changed book stats"
    for epd, node in book.nodes.items():
        other = reloaded.nodes[epd]
        assert other.children == node.children, f"children mismatch at {epd}"
        assert other.breadth == node.breadth, f"breadth mismatch at {epd}"
    print("index verified against TSV source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
