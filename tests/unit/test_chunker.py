from __future__ import annotations

import itertools

import pytest

from enterprise_rag.ingestion.chunker import (
    RecursiveChunker,
    TiktokenTokenizer,
    WhitespaceTokenizer,
)

WS = WhitespaceTokenizer()


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_and_short_text() -> None:
    c = RecursiveChunker(WS, chunk_size=10, overlap=2)
    assert c.chunk("   ") == []
    (only,) = c.chunk("one two three")
    assert only.text == "one two three" and only.token_count == 3 and only.index == 0


def test_exactly_chunk_size_is_one_chunk() -> None:
    c = RecursiveChunker(WS, chunk_size=10, overlap=2)
    assert len(c.chunk(words(10))) == 1
    assert len(c.chunk(words(11))) == 2


def test_every_chunk_within_budget_and_indices_sequential() -> None:
    c = RecursiveChunker(WS, chunk_size=20, overlap=5)
    chunks = c.chunk(words(237))
    assert all(ch.token_count <= 20 for ch in chunks)
    assert [ch.index for ch in chunks] == list(range(len(chunks)))


def test_overlap_between_consecutive_chunks() -> None:
    c = RecursiveChunker(WS, chunk_size=20, overlap=5)
    chunks = c.chunk(words(100))
    for a, b in itertools.pairwise(chunks):
        tail = a.text.split()[-5:]
        assert b.text.split()[:5] == tail


def test_no_content_lost() -> None:
    c = RecursiveChunker(WS, chunk_size=20, overlap=5)
    text = words(150)
    seen = {w for ch in c.chunk(text) for w in ch.text.split()}
    assert seen == set(text.split())


def test_prefers_paragraph_boundaries() -> None:
    c = RecursiveChunker(WS, chunk_size=12, overlap=0)
    text = words(8, "a") + "\n\n" + words(8, "b")
    chunks = c.chunk(text)
    assert [ch.text for ch in chunks] == [words(8, "a"), words(8, "b")]


def test_sentence_boundary_fallback() -> None:
    c = RecursiveChunker(WS, chunk_size=8, overlap=0)
    text = "one two three four five. six seven eight nine ten. eleven twelve"
    chunks = [ch.text for ch in c.chunk(text)]
    assert chunks[0].endswith("five.")
    assert chunks[1].startswith("six")


def test_header_is_prepended_and_budgeted() -> None:
    c = RecursiveChunker(WS, chunk_size=20, overlap=4)
    chunks = c.chunk(words(80), header="Document: Title X\n\n")
    assert all(ch.text.startswith("Document: Title X\n\n") for ch in chunks)
    assert all(ch.token_count <= 20 for ch in chunks)


def test_header_too_large_rejected() -> None:
    c = RecursiveChunker(WS, chunk_size=10, overlap=2)
    with pytest.raises(ValueError):
        c.chunk("a b c", header=words(9))


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(WS, chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveChunker(WS, chunk_size=10, overlap=10)


def test_hard_split_for_unbreakable_text() -> None:
    tok = TiktokenTokenizer()
    c = RecursiveChunker(tok, chunk_size=16, overlap=4, separators=("\n\n",))
    text = "x" * 400  # no separator at all
    chunks = c.chunk(text)
    assert len(chunks) > 1
    assert all(ch.token_count <= 16 for ch in chunks)


def test_real_tokenizer_512_64() -> None:
    tok = TiktokenTokenizer()
    c = RecursiveChunker(tok, chunk_size=512, overlap=64)
    para = "The firm requires dual authorisation for large wires. " * 30
    text = "\n\n".join(para for _ in range(6))
    chunks = c.chunk(text)
    assert len(chunks) >= 3
    assert all(ch.token_count <= 512 for ch in chunks)
    assert all(ch.token_count == len(tok.encode(ch.text)) for ch in chunks)
