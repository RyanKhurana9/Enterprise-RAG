"""Document loaders: turn a file on disk into normalised text + metadata."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DOC_TYPE_BY_DIR = {"policies": "policy", "10k": "10k", "faq": "faq"}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    doc_id: str
    source_path: str
    title: str
    doc_type: str
    text: str


def _doc_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def _doc_type(path: Path) -> str:
    return DOC_TYPE_BY_DIR.get(path.parent.name, "other")


class Loader(ABC):
    """Load one file type. Subclasses declare the extensions they handle."""

    extensions: tuple[str, ...] = ()

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument: ...


class MarkdownLoader(Loader):
    extensions = (".md", ".markdown", ".txt")

    def load(self, path: Path) -> LoadedDocument:
        text = path.read_text(encoding="utf-8")
        title = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")),
            path.stem,
        )
        return LoadedDocument(_doc_id(path), str(path), title, _doc_type(path), text.strip())


class PdfLoader(Loader):
    extensions = (".pdf",)

    def load(self, path: Path) -> LoadedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        return LoadedDocument(_doc_id(path), str(path), title, _doc_type(path), text)


class CompositeLoader(Loader):
    """Dispatches to the first child loader that accepts the file's extension."""

    def __init__(self, loaders: Sequence[Loader]) -> None:
        self._loaders = list(loaders)
        self.extensions = tuple(ext for ld in loaders for ext in ld.extensions)

    def load(self, path: Path) -> LoadedDocument:
        for loader in self._loaders:
            if loader.can_load(path):
                return loader.load(path)
        raise ValueError(f"no loader for {path}")

    def discover(self, root: Path) -> Iterable[Path]:
        for p in sorted(root.rglob("*")):
            if p.is_file() and self.can_load(p):
                yield p


def default_loader() -> CompositeLoader:
    return CompositeLoader([MarkdownLoader(), PdfLoader()])
