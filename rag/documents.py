"""
Document ingestion and normalization.

Supports .txt, .md and .pdf (when PyMuPDF available).
Normalizes to Document(id, content, source, metadata).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None


@dataclass
class Document:
    id: str
    content: str
    source: str
    metadata: Dict[str, str]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed; cannot read PDFs")
    doc = fitz.open(path)
    pages = []
    for p in doc:
        pages.append(p.get_text())
    return "\n\n".join(pages)


def ingest_from_paths(paths: Iterable[str]) -> Iterator[Document]:
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = _read_text_file(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            # Unsupported - skip
            continue
        yield Document(id=str(path.resolve()), content=text, source=str(path), metadata={"filename": path.name, "suffix": suffix})


def ingest_from_string(source_id: str, content: str, filename: str | None = None) -> Document:
    return Document(id=source_id, content=content, source=filename or source_id, metadata={"filename": filename or source_id})
