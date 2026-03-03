#!/usr/bin/env python3
"""
cli.py
Command-line interface for the Swiggy RAG QA system.
Run with: python cli.py
"""

import os
import sys
import textwrap
import logging
from pathlib import Path

from dotenv import load_dotenv

# Add backend to path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from rag_engine import VectorStore, ingest_pdf
from llm import GeminiLLM

load_dotenv()

logging.basicConfig(level=logging.WARNING)  # suppress info logs in CLI mode

PDF_PATH    = os.getenv("PDF_PATH",   "./data/swiggy_annual_report.pdf")
INDEX_DIR   = os.getenv("INDEX_PATH", "./data/faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K       = int(os.getenv("TOP_K", "5"))

DIVIDER   = "-" * 72
SEPARATOR = "=" * 72


def print_header():
    print(SEPARATOR)
    print("  Swiggy Annual Report - RAG Question Answering System")
    print("  FY 2023-24  |  Powered by Gemini + FAISS")
    print(SEPARATOR)
    print()


def wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip():
            wrapped.append(
                textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            )
        else:
            wrapped.append("")
    return "\n".join(wrapped)


def display_result(question: str, answer: str, context_chunks: list[dict]) -> None:
    print()
    print(DIVIDER)
    print(f"  QUESTION: {question}")
    print(DIVIDER)
    print()
    print("  ANSWER:")
    print(wrap(answer))
    print()
    print(DIVIDER)
    print("  SUPPORTING CONTEXT")
    print(DIVIDER)
    for i, chunk in enumerate(context_chunks, start=1):
        score_pct = round(chunk["score"] * 100, 1)
        print(f"\n  [{i}] Page {chunk['page']}  |  Relevance: {score_pct}%")
        print()
        print(wrap(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else "")))
    print()
    print(SEPARATOR)
    print()


def load_system() -> tuple[VectorStore, GeminiLLM]:
    vs = VectorStore(EMBED_MODEL)
    loaded = vs.load(INDEX_DIR)

    if not loaded:
        print("  No saved index found. Ingesting PDF - this may take a few minutes ...")
        if not Path(PDF_PATH).exists():
            print(f"\n  ERROR: PDF not found at '{PDF_PATH}'")
            print(
                "  Place the Swiggy Annual Report PDF at that path and try again.\n"
            )
            sys.exit(1)
        vs = ingest_pdf(PDF_PATH, INDEX_DIR, EMBED_MODEL)
        print(f"  Ingestion complete. {len(vs.chunks)} chunks indexed.\n")
    else:
        print(f"  Loaded index: {len(vs.chunks)} chunks ready.\n")

    llm = GeminiLLM()
    return vs, llm


def main():
    print_header()
    print("  Initializing system ...")
    vs, llm = load_system()
    print("  System ready. Type your question or 'exit' to quit.")
    print()

    while True:
        try:
            question = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye.\n")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("\n  Goodbye.\n")
            break

        print("\n  Searching ...\n")
        chunks = vs.search(question, top_k=TOP_K)
        result = llm.answer(question, chunks)
        display_result(question, result["answer"], result["context_used"])


if __name__ == "__main__":
    main()