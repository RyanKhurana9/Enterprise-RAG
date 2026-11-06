"""Recursive, token-aware text chunker.

Text is split on the coarsest separator that exists (paragraph -> line -> sentence -> word)
until every piece fits the token budget, then pieces are greedily merged back into chunks of
at most ``chunk_size`` tokens. Consecutive chunks share a tail of up to ``overlap`` tokens so
that a fact straddling a boundary is still retrievable from one chunk.

Splitting keeps separators attached to the preceding piece, so ``"".join(pieces) == text``
(the split is lossless) and chunk text reads naturally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    """cl100k_base BPE. Close enough to the embedder's tokenizer for budgeting."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return list(self._enc.encode(text, disallowed_special=()))

    def decode(self, tokens: list[int]) -> str:
        return str(self._enc.decode(tokens))


class WhitespaceTokenizer:
    """Dependency-free tokenizer (one token per whitespace-delimited word). For tests."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]

    def decode(self, tokens: list[int]) -> str:  # pragma: no cover - lossy by design
        return " ".join("x" * t for t in tokens)


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    token_count: int


DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")


class RecursiveChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = tuple(separators)

    # ----------------------------------------------------------------------- public ----
    def chunk(self, text: str, *, header: str = "") -> list[TextChunk]:
        """Split ``text``; if given, ``header`` is prepended to every chunk.

        A contextual header (e.g. the document title) lets a chunk from the middle of a
        document still match queries that name the document's subject. Its tokens are
        reserved out of the budget, so ``header + body`` never exceeds ``chunk_size``.
        """
        text = text.strip()
        if not text:
            return []
        reserve = self._len(header) if header else 0
        if reserve >= self.chunk_size - self.overlap:
            raise ValueError("header leaves no room for content")
        budget = self.chunk_size - reserve
        pieces = self._split(text, self.separators, budget)
        chunks: list[TextChunk] = []
        for body in self._merge(pieces, budget):
            bodies = self._hard_split(body, budget) if self._len(body) > budget else [body]
            for b in bodies:  # hard split only if BPE merged across piece boundaries (rare)
                full = header + b
                chunks.append(TextChunk(len(chunks), full, self._len(full)))
        return chunks

    # ---------------------------------------------------------------------- helpers ----
    def _len(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split(self, text: str, separators: Sequence[str], budget: int) -> list[str]:
        if self._len(text) <= budget:
            return [text]
        for i, sep in enumerate(separators):
            if sep in text:
                parts = text.split(sep)
                pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
                out: list[str] = []
                for piece in pieces:
                    if not piece:
                        continue
                    if self._len(piece) <= budget:
                        out.append(piece)
                    else:
                        out.extend(self._split(piece, separators[i + 1 :], budget))
                return out
        return self._hard_split(text, budget)

    def _hard_split(self, text: str, budget: int) -> list[str]:
        """Last resort for a single run of text with no separators: split on token ids."""
        toks = self.tokenizer.encode(text)
        step = max(1, budget - self.overlap)
        out: list[str] = []
        i = 0
        while True:
            out.append(self.tokenizer.decode(toks[i : i + budget]))
            if i + budget >= len(toks):
                return out
            i += step

    def _merge(self, pieces: list[str], budget: int) -> list[str]:
        sizes = [self._len(p) for p in pieces]
        chunks: list[str] = []
        window: list[int] = []  # indices into pieces
        total = 0
        for i, size in enumerate(sizes):
            if window and total + size > budget:
                chunks.append("".join(pieces[j] for j in window).strip())
                # Keep the longest tail of the window that fits in the overlap budget and
                # still leaves room for the incoming piece.
                while window and (total > self.overlap or total + size > budget):
                    total -= sizes[window.pop(0)]
            window.append(i)
            total += size
        if window:  # always holds at least one piece not yet emitted
            chunks.append("".join(pieces[j] for j in window).strip())
        return [c for c in chunks if c]
