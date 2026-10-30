"""
FastAPI application — single process, no framework beyond standard library + LLM client.
Entry point: uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid
import logging

from backend.pipeline import handle_chat
from backend.session import session_store
from backend.guardrails import check_input
from backend.document_upload import extract_text, DocumentError
from backend.models import ChatResponse, ConfidenceLevel, CitationCheckOutcome

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Startup ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model before the first request: lazily it takes ~20s,
    # which would land on whoever asks the first question (e.g. a judge).
    from backend.ingestion.embed import get_model
    from backend.keyword_search import get_index
    await asyncio.to_thread(get_model)
    await asyncio.to_thread(get_index)
    logger.info("[startup] embedding model + keyword index ready")
    yield


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qanoony — Egyptian Law AI Assistant",
    version="1.0.0",
    description="AI-powered Egyptian legal assistant with grounded citations and explainability.",
    lifespan=lifespan,
)

# ── CORS (allow frontend dev server + deployed frontend) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev
        "http://127.0.0.1:3000",
    ],
    # Wildcards aren't matched in allow_origins — deployed Vercel frontends need a regex.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas (Pydantic for FastAPI validation) ─────────

class ChatRequestBody(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "ar"   # "ar" | "en"


class ChatResponseBody(BaseModel):
    answer: str
    citations: list[dict] = []
    graph_path: list[dict] = []
    confidence: str = "medium"
    citation_check_outcome: str = "passed"
    session_id: str = ""
    language: str = "ar"
    disclaimer: str = (
        "هذا المساعد يقدم معلومات قانونية عامة ولا يُغني عن استشارة محامٍ مختص. "
        "This assistant provides general legal information and is not a substitute "
        "for professional legal advice."
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used for deployment smoke tests."""
    return {"status": "ok", "service": "qanony"}


@app.post("/chat", response_model=ChatResponseBody)
async def chat(body: ChatRequestBody):
    """
    Main chat endpoint.
    Flow: guardrails → pipeline (planner → translation → retriever →
          context fusion → LLM → citation check) → response.
    """
    # Assign session ID if not provided
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(f"[chat] session={session_id} lang={body.language} msg={body.message[:80]}...")

    # ── Input guardrails ─────────────────────────────────────────────
    guardrail_result = check_input(body.message)
    if not guardrail_result["safe"]:
        logger.warning(f"[guardrails] blocked: {guardrail_result['reason']}")
        return ChatResponseBody(
            answer=guardrail_result.get(
                "user_message",
                "عذراً، لا أستطيع معالجة هذا الطلب. / Sorry, I cannot process this request."
            ),
            session_id=session_id,
            language=body.language,
            confidence="low",
        )

    # ── Pipeline ─────────────────────────────────────────────────────
    try:
        result: ChatResponse = await handle_chat(
            message=body.message,
            session_id=session_id,
            language=body.language,
        )
    except Exception as e:
        logger.error(f"[pipeline] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error.")

    # ── Serialize response ───────────────────────────────────────────
    return ChatResponseBody(
        answer=result.answer,
        citations=[
            {"law_name": c.law_name, "article_id": c.article_id, "text_snippet": c.text_snippet}
            for c in result.citations
        ],
        graph_path=[
            {"node_id": n.node_id, "label": n.label, "name": n.name, "relationship": n.relationship}
            for n in result.graph_path
        ],
        confidence=result.confidence.value,
        citation_check_outcome=result.citation_check_outcome.value,
        session_id=result.session_id,
        language=result.language,
        disclaimer=result.disclaimer,
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve conversation history for a session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "turns": [{"role": t.role, "content": t.content} for t in session.turns],
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session."""
    session_store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    """
    Extract text from an uploaded PDF/TXT/MD and pin it to the session.
    Later /chat turns in that session explain the document against the law.
    """
    data = await file.read()
    try:
        text = await asyncio.to_thread(extract_text, file.filename or "", data)
    except DocumentError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session_store.pin_document(session_id, text)
    logger.info(f"[upload] session={session_id} file={file.filename} chars={len(text)}")
    return {
        "session_id": session_id,
        "filename": file.filename,
        "chars": len(text),
        "preview": text[:300],
    }


@app.delete("/sessions/{session_id}/document")
async def remove_document(session_id: str):
    """Unpin the uploaded document from a session."""
    session_store.unpin_document(session_id)
    return {"status": "removed", "session_id": session_id}
