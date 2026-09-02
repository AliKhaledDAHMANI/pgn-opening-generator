"""Tests for the ECO opening book."""

from __future__ import annotations

import chess
import pytest

from pgn_generator.book import OpeningBook, expand_aliases, get_book, normalize_name
from pgn_generator.errors import OpeningNotFoundError


def test_book_loads_with_expected_shape(book: OpeningBook) -> None:
    stats = book.stats()
    assert stats["entries"] > 3000
    assert stats["positions"] > 7000
    assert stats["families"] > 100


def test_every_entry_is_legal_and_matches_its_epd(book: OpeningBook) -> None:
    """The compiled index must describe real, legal positions."""
    for entry in book.entries:
        board = chess.Board()
        for uci in entry.uci:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f"{entry.name}: {uci} illegal"
            board.push(move)
        assert board.epd(en_passant="legal") == entry.epd


def test_index_and_tsv_agree() -> None:
    """The gzipped index must reproduce the graph built from the raw TSV files."""
    from pgn_generator.book import ECO_DIR

    from_index = OpeningBook.load(ECO_DIR)
    from_tsv = OpeningBook._from_tsv(ECO_DIR)  # noqa: SLF001
    assert from_index.stats() == from_tsv.stats()
    assert len(from_index.nodes) == len(from_tsv.nodes)
    for epd, node in from_tsv.nodes.items():
        assert from_index.nodes[epd].children == node.children
        assert from_index.nodes[epd].breadth == node.breadth


@pytest.mark.parametrize(
    "query,expected_eco,expected_line",
    [
        ("Italian Game", "C50", "1. e4 e5 2. Nf3 Nc6 3. Bc4"),
        ("Ruy Lopez", "C60", "1. e4 e5 2. Nf3 Nc6 3. Bb5"),
        ("Sicilian Najdorf", "B90", "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6"),
        ("Spanish", "C60", "1. e4 e5 2. Nf3 Nc6 3. Bb5"),
        ("French Defense", "C00", "1. e4 e6"),
        ("Grunfeld", "D80", "1. d4 Nf6 2. c4 g6 3. Nc3 d5"),
        ("KID", "E61", "1. d4 Nf6 2. c4 g6 3. Nc3"),
    ],
)
def test_resolve_known_openings(book: OpeningBook, query, expected_eco, expected_line) -> None:
    entry, _alternates = book.resolve(query)
    assert entry.eco == expected_eco
    assert entry.san_line() == expected_line


def test_kings_indian_defaults_to_the_defense_not_the_attack(book: OpeningBook) -> None:
    """"King's Indian" without a qualifier means the Defense."""
    entry, _ = book.resolve("King's Indian")
    assert "Defense" in entry.name
    attack, _ = book.resolve("King's Indian Attack")
    assert "Attack" in attack.name


def test_unknown_opening_raises_with_suggestions(book: OpeningBook) -> None:
    with pytest.raises(OpeningNotFoundError):
        book.resolve("Zzyzx Gambit Deferred")


def test_classify_exact_and_transposed(book: OpeningBook) -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4"):
        board.push_san(san)
    match = book.classify(board)
    assert match is not None and match.exact
    assert match.eco == "C50"

    # A position past the end of the book still classifies, via backtracking.
    for san in ("Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"):
        board.push_san(san)
    deep = book.classify(board)
    assert deep is not None
    assert deep.name.startswith("Italian Game")


def test_theory_is_sorted_by_breadth(book: OpeningBook) -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4"):
        board.push_san(san)
    theory = book.theory(board)
    assert theory, "the Italian should have book continuations"
    breadths = [move.breadth for move in theory]
    assert breadths == sorted(breadths, reverse=True)
    assert theory[0].san in ("Bc5", "Nf6")
    for move in theory:
        assert move.move in board.legal_moves


def test_theory_is_empty_outside_the_book(book: OpeningBook) -> None:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "d5"):
        board.push_san(san)
    assert book.theory(board) == []
    assert not book.contains(board)


def test_alias_expansion_does_not_duplicate_family_names() -> None:
    assert expand_aliases("Sicilian Najdorf") == "sicilian defense najdorf variation"
    assert expand_aliases("Najdorf") == "najdorf variation"
    assert expand_aliases("Spanish") == "ruy lopez"


def test_normalize_name_handles_accents_and_uk_spelling() -> None:
    assert normalize_name("Grünfeld Defence") == "grunfeld defense"
    assert normalize_name("Réti Opening") == "reti opening"


def test_entry_name_parts(book: OpeningBook) -> None:
    entry, _ = book.resolve("Najdorf English Attack")
    assert entry.family == "Sicilian Defense"
    assert entry.variation == "Najdorf Variation"
    assert "English Attack" in entry.subvariation or "English Attack" in entry.name
