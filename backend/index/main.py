"""
main.py
FastAPI application - serves the UI and the /ask endpoint.
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import VectorStore, ingest_pdf
from .llm import GeminiLLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

PDF_PATH   = os.getenv("PDF_PATH",   "./data/swiggy_annual_report.pdf")
INDEX_DIR  = os.getenv("INDEX_PATH", "./data/faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K      = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
vector_store: VectorStore | None = None
llm: GeminiLLM | None = None


# ---------------------------------------------------------------------------
# Lifespan (replaces on_event startup/shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, llm

    # -- Vector store --
    vector_store = VectorStore(EMBED_MODEL)
    loaded = vector_store.load(INDEX_DIR)

    if not loaded:
        logger.info("No saved index found - ingesting PDF from %s", PDF_PATH)
        if not Path(PDF_PATH).exists():
            raise FileNotFoundError(
                f"PDF not found at {PDF_PATH}. "
                "Place the Swiggy Annual Report PDF there and restart."
            )
        vector_store = ingest_pdf(PDF_PATH, INDEX_DIR, EMBED_MODEL)
    else:
        logger.info("Loaded existing FAISS index (%d chunks)", len(vector_store.chunks))

    # -- LLM --
    llm = GeminiLLM()
    logger.info("Application ready")
    yield
    # cleanup (nothing needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Swiggy RAG QA",
    description="RAG-based QA over Swiggy Annual Report FY 2023-24",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


class ContextChunk(BaseModel):
    chunk_id: int
    page: int
    text: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_used: list[ContextChunk]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page frontend."""
    html_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Main QA endpoint."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store is None or llm is None:
        raise HTTPException(status_code=503, detail="System is still initializing.")

    # Retrieve
    retrieved = vector_store.search(question, top_k=TOP_K)

    # Generate
    result = llm.answer(question, retrieved)

    return AnswerResponse(
        question=question,
        answer=result["answer"],
        context_used=[
            ContextChunk(
                chunk_id=c["chunk_id"],
                page=c["page"],
                text=c["text"],
                score=c["score"],
            )
            for c in result["context_used"]
        ],
    )


@app.get("/health")
async def health():
    """Quick health / readiness check."""
    ready = vector_store is not None and llm is not None
    chunks = len(vector_store.chunks) if vector_store else 0
    return {"status": "ok" if ready else "initializing", "indexed_chunks": chunks}