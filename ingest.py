#!/usr/bin/env python3
"""
ingest.py
Run once to pre-process the PDF and build the FAISS index.
This avoids re-ingesting on every server restart.

Usage:
    python ingest.py
    python ingest.py --pdf path/to/report.pdf --index ./data/faiss_index
"""

import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from rag_engine import ingest_pdf

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest PDF and build FAISS index")
    parser.add_argument(
        "--pdf",
        default="./data/swiggy_annual_report.pdf",
        help="Path to the Swiggy Annual Report PDF",
    )
    parser.add_argument(
        "--index",
        default="./data/faiss_index",
        help="Directory to save the FAISS index",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Character chunk size",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Chunk overlap in characters",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at '{pdf_path}'")
        print("Download the Swiggy FY 2023-24 Annual Report and place it at that path.")
        sys.exit(1)

    print(f"PDF      : {pdf_path}")
    print(f"Index dir: {args.index}")
    print(f"Model    : {args.model}")
    print(f"Chunk    : {args.chunk_size} chars, overlap {args.overlap}")
    print()

    vs = ingest_pdf(
        str(pdf_path),
        args.index,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print()
    print(f"Done. {len(vs.chunks)} chunks indexed and saved to '{args.index}'")


if __name__ == "__main__":
    main()