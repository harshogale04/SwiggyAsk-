"""
rag_engine.py
Core RAG pipeline: PDF ingestion, chunking, embedding, FAISS indexing, retrieval.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults (overridden via environment or direct call args)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove null bytes, normalize whitespace, strip lone page numbers."""
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely a number (page numbers)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Tries to break at paragraph or sentence boundaries.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Prefer paragraph break, then sentence break
            para_break = text.rfind("\n\n", start, end)
            sent_break = max(
                text.rfind(". ", start, end),
                text.rfind(".\n", start, end),
            )
            boundary = para_break if para_break > start + chunk_size // 2 else sent_break
            if boundary > start + chunk_size // 3:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from every page.
    Returns list of {page: int, text: str}.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = clean_text(raw)
        if len(cleaned) > 40:  # skip near-empty pages
            pages.append({"page": page_num, "text": cleaned})
    doc.close()
    return pages


def build_document_chunks(
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Convert pages into overlapping chunks with metadata.
    Returns list of {chunk_id, page, text}.
    """
    chunks: list[dict] = []
    chunk_id = 0
    for page_data in pages:
        for text in split_into_chunks(page_data["text"], chunk_size, overlap):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_data["page"],
                    "text": text,
                }
            )
            chunk_id += 1
    return chunks


# ---------------------------------------------------------------------------
# Vector store (FAISS)
# ---------------------------------------------------------------------------

class VectorStore:
    """Sentence-transformer embeddings backed by a FAISS flat L2 index."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Embed all chunks and build the FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        logger.info("Embedding %d chunks ...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner-product == cosine on normalized vecs
        self.index.add(embeddings.astype(np.float32))
        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save FAISS index + chunk metadata to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Index saved to %s", directory)

    def load(self, directory: str) -> bool:
        """Load from disk. Returns True on success."""
        path = Path(directory)
        idx_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"
        if not idx_file.exists() or not chunks_file.exists():
            return False
        self.index = faiss.read_index(str(idx_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(
            "Index loaded from %s: %d chunks", directory, len(self.chunks)
        )
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Return the top_k most relevant chunks for the query.
        Each result: {chunk_id, page, text, score}.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")

        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results


# ---------------------------------------------------------------------------
# Ingest pipeline (one-shot helper)
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: str,
    index_dir: str,
    model_name: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> VectorStore:
    """Full ingest: PDF -> chunks -> embeddings -> FAISS index saved to disk."""
    logger.info("Ingesting PDF: %s", pdf_path)
    pages = extract_pages(pdf_path)
    logger.info("Extracted %d pages with content", len(pages))

    chunks = build_document_chunks(pages, chunk_size, overlap)
    logger.info("Created %d chunks", len(chunks))

    vs = VectorStore(model_name)
    vs.build(chunks)
    vs.save(index_dir)
    return vs