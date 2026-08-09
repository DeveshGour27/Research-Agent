"""
Chunking module.

Responsibilities:
- Split documents into deterministic, overlapping chunks
- Preserve metadata and deterministic chunk IDs

Design:
- Simple character-based chunking using settings.chunk_size and settings.chunk_overlap
- Tokenization is intentionally lightweight to avoid heavy dependencies in this project
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Iterable, List

from app.config import settings
from app.constants import CHUNK_SEPARATOR, MIN_CHUNK_CHARACTERS


@dataclass
class Document:
    id: str
    content: str
    source: str
    metadata: dict


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    start: int
    end: int
    metadata: dict


def _deterministic_chunk_id(document_id: str, start: int, end: int) -> str:
    key = f"{document_id}:{start}:{end}"
    return sha256(key.encode("utf-8")).hexdigest()


def chunk_document(doc: Document, chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[Chunk]:
    """Split a document into overlapping chunks.

    Uses character counts as a lightweight approximation of token counts.

    Returns list of Chunk preserving metadata.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    text = doc.content or ""
    if not text:
        return []

    n = len(text)
    if n <= chunk_size:
        # single chunk
        cid = _deterministic_chunk_id(doc.id, 0, n)
        return [Chunk(id=cid, document_id=doc.id, text=text, start=0, end=n, metadata=dict(doc.metadata))]

    chunks: List[Chunk] = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("Invalid chunking parameters: step must be positive")

    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        piece = text[pos:end]
        if len(piece.strip()) < MIN_CHUNK_CHARACTERS and end < n:
            # If chunk too small (e.g. whitespace), extend to reach minimum meaningful length
            extend = min(n, end + (MIN_CHUNK_CHARACTERS - len(piece)))
            piece = text[pos:extend]
            end = extend
        cid = _deterministic_chunk_id(doc.id, pos, end)
        chunks.append(Chunk(id=cid, document_id=doc.id, text=piece, start=pos, end=end, metadata=dict(doc.metadata)))
        if end == n:
            break
        pos += step

    return chunks


def chunk_documents(docs: Iterable[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return out
